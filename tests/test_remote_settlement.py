import base64
import http.client
import json
import socket
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mindie_bench.models import CorpusConfig, Group, GroupConfig, MCPServer, RemoteConfig, VerifierResult
from mindie_bench.runner import orchestrator
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.process import ProcessResult


def post(gateway, path, payload):
    request = urllib.request.Request(
        gateway.url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


class DelayedRemote:
    def __init__(self, *, fails):
        self.fails = fails
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def exchange_stream(self, identity, stdin, *, eof):
        return self._result(None)

    def execute(self, command, stdin):
        return self._result(0)

    def _result(self, exit_code):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(3)
        if self.fails:
            raise BrokenPipeError("late remote acknowledgement lost")
        return {"stdout_b64": "", "stderr_b64": "", "exit_code": exit_code}


def exchange_in_thread(gateway, remote, transport="stream"):
    gateway.remote = remote
    gateway.stream_input["owned-stream"] = b""
    results = []

    def exchange():
        try:
            path = "/bench/ssh-stream" if transport == "stream" else "/bench/ssh"
            payload = (
                {
                    "action": "exchange",
                    "stream_id": "owned-stream",
                    "stdin_b64": base64.b64encode(b'{"payload":{}}\n').decode(),
                }
                if transport == "stream"
                else {"command": "owned zero-paid write", "stdin_b64": ""}
            )
            results.append(post(gateway, path, payload))
        except urllib.error.HTTPError as exc:
            results.append(exc.code)

    thread = threading.Thread(target=exchange)
    thread.start()
    assert remote.entered.wait(2)
    return thread, results


def gateway_for(case):
    return Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "remote-settlement.jsonl",
    )


def test_late_remote_error_is_settled_before_verifier_after_native_timeout(case, monkeypatch):
    remote = DelayedRemote(fails=True)
    workers = []
    verified = []
    gateways = []

    def process(*args, gateway, **kwargs):
        gateways.append(gateway)
        thread, _ = exchange_in_thread(gateway, remote)
        workers.append(thread)
        threading.Timer(0.05, remote.release.set).start()
        return ProcessResult(started=True, elapsed=0.1, stop_reason="timeout")

    def verifier(*args, **kwargs):
        verified.append(True)
        workers[0].join(timeout=2)
        return VerifierResult(reward=1.0), {"returncode": 0}

    monkeypatch.setattr(orchestrator, "stream_process", process)
    monkeypatch.setattr(orchestrator, "local_verifier", verifier)
    monkeypatch.setattr(orchestrator, "native_command", lambda command, **kwargs: command)
    output = case["root"] / "run"
    output.mkdir()
    try:
        receipt = orchestrator.run_trial(
            case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=output
        )
    finally:
        remote.release.set()
        for worker in workers:
            worker.join(timeout=2)
    assert receipt.terminal_status.value == "unknown", receipt.model_dump_json(indent=2)
    assert not verified
    assert remote.calls == 1 and gateways[0].model_calls == 0
    assert receipt.primary_metrics["reward"] is None
    settlement = receipt.auxiliary_metrics["remote_settlement"]
    assert settlement["sealed"] and settlement["active"] == 0 and settlement["unknown"]
    assert settlement["failures"][0]["message"] == "late remote acknowledgement lost"


@pytest.mark.parametrize("transport", ["stream", "ssh"])
def test_remote_settlement_deadline_is_unknown_and_seals_new_admission(case, transport):
    remote = DelayedRemote(fails=False)
    with gateway_for(case) as gateway:
        thread, results = exchange_in_thread(gateway, remote, transport)
        try:
            gateway.finish_remote(timeout=0.01)
            assert gateway.unknown
            assert gateway.remote_state()["settlement_expired"]
            with pytest.raises(urllib.error.HTTPError) as rejected:
                post(
                    gateway,
                    "/bench/ssh-stream",
                    {"action": "exchange", "stream_id": "owned-stream"},
                )
            assert rejected.value.code == 409
            assert remote.calls == 1
        finally:
            remote.release.set()
            thread.join(timeout=2)
        assert results and gateway.unknown
        assert gateway.remote_state()["active"] == 0
        assert "remote_calls_unsettled" in gateway.journal.read_text()


def test_rpc_client_disconnect_is_recorded_before_settlement_completes(case):
    remote = DelayedRemote(fails=False)
    with gateway_for(case) as gateway:
        gateway.remote = remote
        gateway.stream_input["owned-stream"] = b""
        body = json.dumps(
            {
                "action": "exchange",
                "stream_id": "owned-stream",
                "stdin_b64": base64.b64encode(b'{"payload":{}}\n').decode(),
            }
        ).encode()
        client = socket.create_connection(("127.0.0.1", gateway.port), timeout=2)
        client.sendall(
            b"POST /bench/ssh-stream HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        assert remote.entered.wait(2)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        remote.release.set()
        gateway.finish_remote(timeout=1)
        assert gateway.unknown and gateway.remote_state()["active"] == 0
        failure = gateway.remote_state()["failures"][0]
        assert failure["kind"] == "remote_delivery_failed"
        assert failure["external_effects"] == "unknown"
        assert remote.calls == 1 and gateway.model_calls == 0


def test_partial_http_body_cannot_admit_a_remote_write_after_seal(case):
    remote = DelayedRemote(fails=False)
    with gateway_for(case) as gateway:
        gateway.remote = remote
        body = json.dumps({"command": "owned write", "stdin_b64": ""}).encode()
        client = socket.create_connection(("127.0.0.1", gateway.port), timeout=2)
        try:
            client.sendall(
                b"POST /bench/ssh HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body[:5]
            )
            gateway.finish_remote(timeout=0.1)
            client.sendall(body[5:])
            response = http.client.HTTPResponse(client)
            response.begin()
            assert response.status == 409
            response.read()
            assert remote.calls == 0
            assert not gateway.unknown
        finally:
            client.close()


def configure_g3(case, corpus):
    source, commit = corpus
    case["config"].corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    case["config"].groups[Group.G3] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(command="/usr/bin/true", component="knowledge"),
        },
        capture_command=["/usr/bin/false"],
    )


def test_unsettled_remote_timeout_skips_both_judge_and_capture(case, corpus, monkeypatch):
    configure_g3(case, corpus)
    remote = DelayedRemote(fails=False)
    workers = []
    stages = []
    original = Gateway.finish_remote

    def brief_settlement(self, timeout=1):
        original(self, timeout=0.01)

    def process(*args, gateway, output, **kwargs):
        stages.append(output.name)
        assert output.name == "native_events.jsonl"
        worker, _ = exchange_in_thread(gateway, remote)
        workers.append(worker)
        return ProcessResult(started=True, elapsed=0.1, stop_reason="timeout")

    def never_judge(*args, **kwargs):
        pytest.fail("Unsettled remote operation must not reach the judge")

    monkeypatch.setattr(Gateway, "finish_remote", brief_settlement)
    monkeypatch.setattr(orchestrator, "stream_process", process)
    monkeypatch.setattr(orchestrator, "local_verifier", never_judge)
    monkeypatch.setattr(orchestrator, "native_command", lambda command, **kwargs: command)
    output = case["root"] / "run"
    output.mkdir()
    try:
        receipt = orchestrator.run_trial(
            case["task"], case["config"], tier="l2", group=Group.G3, repetition=1, run_root=output
        )
    finally:
        remote.release.set()
        for worker in workers:
            worker.join(timeout=2)
    assert receipt.terminal_status.value == "unknown"
    assert stages == ["native_events.jsonl"]
    assert receipt.auxiliary_metrics["remote_settlement"]["settlement_expired"]
    assert "capture_process" not in receipt.auxiliary_metrics


def test_successful_remote_handler_finishes_before_worker_quiesce(case, monkeypatch):
    task = case["task"] / "task.toml"
    task.write_text(
        task.read_text()
        + f'\n[environment.l3]\nrepo_path = "{case["repo"]}"\n'
        + f'base_commit = "{case["commit"]}"\nnpu_model = "synthetic"\nnpu_count = 1\n'
    )
    case["config"].remote = RemoteConfig(
        host="synthetic",
        container="synthetic",
        repo_path="/synthetic",
        devices=[0],
        probe_command=["synthetic"],
    )
    remote = DelayedRemote(fails=False)
    gateways = []
    workers = []
    quiesced = []

    class Worker:
        preflight = {"_environment": {"image_digest": "synthetic"}}

        def __init__(self, *args, **kwargs):
            self.path = "/synthetic-owned-slot"

        def prepare(self):
            pass

        def upload_public(self, source):
            pass

        def quiesce(self):
            assert gateways[0].remote_state()["active"] == 0
            quiesced.append(True)

        def verify(self, script):
            return {
                "exit_code": 0,
                "stdout_b64": base64.b64encode(b'{"reward":1.0}\n').decode(),
                "stderr_b64": "",
            }

        def close(self):
            pass

    def process(*args, gateway, consume, **kwargs):
        gateways.append(gateway)
        worker, _ = exchange_in_thread(gateway, remote)
        workers.append(worker)
        threading.Timer(0.05, remote.release.set).start()
        consume({"type": "fixture.trajectory", "trajectory": {"native_completed": True}}, 0, 0)
        return ProcessResult(started=True, returncode=0, elapsed=0.1)

    monkeypatch.setattr(orchestrator, "RemoteSlot", Worker)
    monkeypatch.setattr(orchestrator, "stream_process", process)
    monkeypatch.setattr(orchestrator, "native_command", lambda command, **kwargs: command)
    output = case["root"] / "run"
    output.mkdir()
    try:
        receipt = orchestrator.run_trial(
            case["task"],
            case["config"],
            tier="l3",
            group=Group.G0,
            repetition=1,
            run_root=output,
            devices=[0],
        )
    finally:
        remote.release.set()
        for worker in workers:
            worker.join(timeout=2)
    assert receipt.terminal_status.value == "completed_pass", receipt.model_dump_json(indent=2)
    assert quiesced == [True]
    assert not receipt.auxiliary_metrics["remote_settlement"]["unknown"]


def test_late_unknown_and_original_capture_failure_are_both_saved(case, corpus, monkeypatch):
    configure_g3(case, corpus)
    original_exit = Gateway.__exit__
    foreground = []

    def late_unknown(self, *args):
        original_exit(self, *args)
        if not foreground:
            foreground.append(self)
        elif self is not foreground[0]:
            foreground[0].unknown = True
            foreground[0].record(
                "remote_unknown", action="exchange", error="BrokenPipeError", message="late ack"
            )

    monkeypatch.setattr(Gateway, "__exit__", late_unknown)
    output = case["root"] / "run"
    output.mkdir()
    receipt = orchestrator.run_trial(
        case["task"], case["config"], tier="l2", group=Group.G3, repetition=1, run_root=output
    )
    assert receipt.terminal_status.value == "unknown", receipt.model_dump_json(indent=2)
    assert receipt.auxiliary_metrics["business_terminal_status"] == "completed_pass"
    assert receipt.auxiliary_metrics["capture_process"]["returncode"] == 1
    assert {error["stage"] for error in receipt.errors} >= {"capture", "remote_settlement"}
    assert receipt.auxiliary_metrics["status_before_remote_unknown"]["terminal_status"] == "infra_failed"


def test_remote_settlement_can_complete_without_unknown_or_extra_admission(case):
    remote = DelayedRemote(fails=False)
    with gateway_for(case) as gateway:
        thread, results = exchange_in_thread(gateway, remote)
        threading.Timer(0.02, remote.release.set).start()
        gateway.finish_remote(timeout=1)
        thread.join(timeout=2)
        assert not gateway.unknown
        assert gateway.remote_state() == {
            "sealed": True,
            "active": 0,
            "settlement_expired": False,
            "unknown": False,
            "failures": [],
        }
        with pytest.raises(urllib.error.HTTPError) as rejected:
            post(gateway, "/bench/ssh-stream", {"action": "exchange", "stream_id": "owned-stream"})
        assert rejected.value.code == 409
        assert len(results) == remote.calls == 1


def test_final_late_unknown_preserves_original_verifier_failure(case, monkeypatch):
    original_exit = Gateway.__exit__

    def late_unknown(self, *args):
        original_exit(self, *args)
        self.unknown = True
        self.record("remote_unknown", action="exchange", error="BrokenPipeError", message="late ack")

    def unavailable(*args, **kwargs):
        raise ValueError("required external judge unavailable")

    monkeypatch.setattr(Gateway, "__exit__", late_unknown)
    monkeypatch.setattr(orchestrator, "local_verifier", unavailable)
    output = case["root"] / "run"
    output.mkdir()
    receipt = orchestrator.run_trial(
        case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=output
    )
    assert receipt.terminal_status.value == "unknown", receipt.model_dump_json(indent=2)
    assert any(error["stage"] == "verifier" for error in receipt.errors)
    assert any(error["stage"] == "remote_settlement" for error in receipt.errors)
    assert receipt.auxiliary_metrics["status_before_remote_unknown"] == {
        "terminal_status": "infra_failed",
        "failure_reason": "verifier: ValueError: required external judge unavailable",
    }
    assert receipt.primary_metrics["reward"] is None
    saved = json.loads((Path(receipt.artifacts["private_trial_dir"]) / "trial_receipt.json").read_text())
    assert saved["terminal_status"] == "unknown"
