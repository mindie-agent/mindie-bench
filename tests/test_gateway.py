import base64
import json
import socket
import struct
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway


def post(url, path, payload):
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=5)


def test_no_pr_issue_or_arbitrary_http_egress(case):
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
    ) as gateway:
        for path in ["/repos/mindie-agent/knowledge/pulls", "/repos/x/y/issues", "/v1/uploads"]:
            with pytest.raises(urllib.error.HTTPError) as failure:
                post(gateway.url, path, {"body": "should never leave this machine"})
            assert failure.value.code == 403
        assert gateway.model_calls == 0


def test_claude_native_beta_query_is_forwarded_without_opening_arbitrary_queries(case, monkeypatch):
    accepted = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            accepted.append(self.path)
            raw = json.dumps(
                {
                    "id": "message-fixture",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "fixture"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BENCH_TEST_KEY", "fixture-only")
    harness = HarnessConfig(
        name="claude-code",
        executable="/usr/bin/true",
        model="pinned",
        expected_version="fixture",
        api_key_env="BENCH_TEST_KEY",
        provider_url=f"http://127.0.0.1:{server.server_port}/v1",
    )
    try:
        with Gateway(
            harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
        ) as gateway:
            with post(gateway.url, "/v1/messages?beta=true", {"model": "pinned"}) as response:
                assert response.status == 200
                response.read()
            for path in [
                "/v1/messages?beta=false",
                "/v1/messages?beta=true&redirect=external",
                "/v1/responses?beta=true",
            ]:
                with pytest.raises(urllib.error.HTTPError) as failure:
                    post(gateway.url, path, {"model": "pinned"})
                assert failure.value.code == 403
            assert gateway.model_calls == 1
            assert not gateway.unknown
        assert accepted == ["/v1/messages?beta=true"]
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_replay_rule_paths_never_leave_parent_and_have_hard_budget(case):
    case["spec"].budget.max_npu_runs = 1
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
    ) as gateway:
        with post(gateway.url, "/bench/replay", {"argv": ["pytest", "-q"]}) as response:
            payload = json.load(response)
            assert payload["exit_code"] == 1
            assert "checks" not in payload
        with post(gateway.url, "/bench/replay", {"argv": ["pytest"]}) as response:
            assert json.load(response)["exit_code"] == 124
        assert gateway.stop_reason == "max_npu_runs"
        assert gateway.replay_calls == 1


def test_replay_symlink_escape_rejected(case):
    (case["repo"] / "module.py").unlink()
    (case["repo"] / "module.py").symlink_to(case["task"] / "oracle/replay_rules.json")
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], case["root"] / "journal"
    ) as gateway:
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/bench/replay", {"argv": ["pytest"]})
        assert failure.value.code == 500


def test_unknown_paid_request_not_retried_and_key_stays_parent(case, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_KEY", "synthetic-parent-only-key")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        model="test",
        expected_version="test",
        api_key_env="BENCH_TEST_KEY",
        provider_url="https://example.invalid/v1",
    )
    attempts = []

    class BrokenProvider:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def request(self, method, path, body, headers):
            attempts.append(headers["Authorization"])
            raise TimeoutError("synthetic lost provider receipt")

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", BrokenProvider)
    with Gateway(harness, case["spec"], case["task"], case["repo"], case["root"] / "journal") as gateway:
        try:
            post(gateway.url, "/v1/responses", {"model": "test", "input": "sample"})
        except (OSError, Exception):
            pass
        assert gateway.unknown
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/v1/responses", {"model": "test", "input": "sample"})
        assert failure.value.code == 409
        assert len(attempts) == 1
        assert gateway.model_calls == 1
    assert "synthetic-parent-only-key" not in (case["root"] / "journal").read_text()


def test_agent_cannot_change_the_controlled_model(case, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_KEY", "synthetic-key")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        model="pinned-model",
        expected_version="test",
        api_key_env="BENCH_TEST_KEY",
        provider_url="https://example.invalid/v1",
    )
    with Gateway(harness, case["spec"], case["task"], case["repo"], case["root"] / "journal") as gateway:
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/v1/responses", {"model": "different-model", "input": "sample"})
        assert failure.value.code == 403
        assert gateway.model_calls == 0
        assert gateway.infrastructure_error


@pytest.mark.parametrize(
    "url,accepted",
    [
        ("http://127.0.0.1:3425/v1", True),
        ("http://[::1]:3425/v1", True),
        ("http://localhost:3425/v1", False),
        ("http://192.168.13.153/v1", False),
        ("http://127.0.0.1@example.com/v1", False),
        ("https://example.com/v1", True),
    ],
)
def test_cleartext_relay_is_restricted_to_numeric_loopback(case, monkeypatch, url, accepted):
    monkeypatch.setenv("BENCH_TEST_KEY", "local-relay-placeholder")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        model="test",
        expected_version="test",
        api_key_env="BENCH_TEST_KEY",
        provider_url=url,
    )
    args = (harness, case["spec"], case["task"], case["repo"], case["root"] / "journal")
    if accepted:
        Gateway(*args)
    else:
        with pytest.raises(ValueError, match="HTTPS"):
            Gateway(*args)


def test_unknown_remote_write_stops_later_distinct_commands(case):
    calls = []

    class LostReceipt:
        def execute(self, command, stdin):
            calls.append(command)
            raise TimeoutError("remote acknowledgement was lost")

    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=LostReceipt(),
    ) as gateway:
        with pytest.raises(urllib.error.HTTPError) as first:
            post(gateway.url, "/bench/ssh", {"command": "write first", "stdin_b64": ""})
        assert first.value.code == 502
        with pytest.raises(urllib.error.HTTPError) as second:
            post(gateway.url, "/bench/ssh", {"command": "write second", "stdin_b64": ""})
        assert second.value.code == 409
        assert calls == ["write first"]


def test_stream_rpc_admission_precedes_forwarding_and_ignores_worker_source(case):
    case["spec"].budget.max_npu_runs = 1
    exchanges = []

    class Remote:
        def open_stream(self, command):
            return {"stream_id": "owned"}

        def exchange_stream(self, identity, stdin, *, eof):
            exchanges.append(base64.b64decode(stdin))
            return {"stdout_b64": "", "stderr_b64": "", "exit_code": None}

    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=Remote(),
    ) as gateway:
        with post(gateway.url, "/bench/ssh-stream", {"action": "open", "command": "rpc worker"}):
            pass
        source = (
            json.dumps(
                {
                    "id": 1,
                    "kind": "python",
                    "code": "pytest worker source",
                    "payload": {"path": "test.py"},
                }
            ).encode()
            + b"\n"
        )
        request = (
            json.dumps(
                {
                    "id": 2,
                    "kind": "python",
                    "payload": {"spec": {"command": "python3 -I test_rmsnorm.py"}},
                }
            ).encode()
            + b"\n"
        )

        def exchange(data):
            return post(
                gateway.url,
                "/bench/ssh-stream",
                {
                    "action": "exchange",
                    "stream_id": "owned",
                    "stdin_b64": base64.b64encode(data).decode(),
                    "eof": False,
                },
            )

        with exchange(source):
            pass
        with exchange(request[:10]):
            pass
        assert gateway.remote_test_calls == 0
        assert exchanges == [source, b""]
        with exchange(request[10:]):
            pass
        assert exchanges[-1] == request
        assert gateway.remote_test_calls == 1
        with pytest.raises(urllib.error.HTTPError) as failure:
            exchange(request.replace(b'"id": 2', b'"id": 3'))
        assert failure.value.code == 429
        assert len(exchanges) == 3


def test_unknown_stream_blocks_even_a_distinct_later_ssh_request(case):
    class Remote:
        def open_stream(self, command):
            raise TimeoutError("lost open acknowledgement")

        def execute(self, *args):
            raise AssertionError("must not send after unknown")

    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=Remote(),
    ) as gateway:
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(gateway.url, "/bench/ssh-stream", {"action": "open", "command": "worker"})
        assert failure.value.code == 502
        with pytest.raises(urllib.error.HTTPError) as later:
            post(gateway.url, "/bench/ssh", {"command": "new write", "stdin_b64": ""})
        assert later.value.code == 409
        assert gateway.unknown


@pytest.mark.parametrize("kind", ["one-shot", "stream-open", "empty-poll"])
def test_delivery_disconnect_preserves_known_results_and_unknown_stream_boundary(case, kind):
    entered, release, observed = threading.Event(), threading.Event(), threading.Event()

    class Remote:
        def execute(self, command, stdin):
            entered.set()
            assert release.wait(5)
            return {"stdout_b64": "", "stderr_b64": "", "exit_code": 0}

        def open_stream(self, command):
            if kind == "stream-open":
                entered.set()
                assert release.wait(5)
            return {"stream_id": "owned"}

        def exchange_stream(self, identity, stdin, *, eof):
            entered.set()
            assert release.wait(5)
            return {"stdout_b64": "", "stderr_b64": "", "exit_code": None}

    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=Remote(),
    ) as gateway:
        original = gateway.record

        def record(kind, **values):
            original(kind, **values)
            if kind == "remote_delivery_failed":
                observed.set()

        gateway.record = record
        if kind == "empty-poll":
            with post(gateway.url, "/bench/ssh-stream", {"action": "open", "command": "worker"}):
                pass
        route = "/bench/ssh" if kind == "one-shot" else "/bench/ssh-stream"
        request = (
            {"command": "read source", "stdin_b64": ""}
            if kind == "one-shot"
            else {"action": "open", "command": "worker"}
            if kind == "stream-open"
            else {"action": "exchange", "stream_id": "owned", "stdin_b64": "", "eof": False}
        )
        body = json.dumps(request).encode()
        connection = socket.create_connection(("127.0.0.1", gateway.port), timeout=5)
        try:
            connection.sendall(
                f"POST {route} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            assert entered.wait(5)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        finally:
            connection.close()
            release.set()
        assert observed.wait(5)
        assert gateway.unknown is (kind == "stream-open")
        failure = gateway.remote_delivery_failures[0]
        if kind == "empty-poll":
            assert failure["receipt_id"] is None
        else:
            durable = json.loads(
                (case["root"] / "remote-results" / f"{failure['receipt_id']}.json").read_text()
            )
            assert durable["result"] == (
                {"stdout_b64": "", "stderr_b64": "", "exit_code": 0}
                if kind == "one-shot"
                else {"stream_id": "owned"}
            )
