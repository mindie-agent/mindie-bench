"""Zero-paid loopback provider checks for cancellation and usage settlement."""

import http.client
import http.server
import json
import socket
import struct
import threading
import time
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


PARTIAL = event({"type": "response.output_text.delta", "delta": "started"})
FINAL = event(
    {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 19,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 7,
            }
        },
    }
)


@contextmanager
def provider(serve):
    attempts = []
    failures = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            attempts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            try:
                serve(self)
            except TimeoutError:
                raise
            except OSError:
                # The deadline test intentionally aborts the owned provider socket.
                pass
            except Exception as exc:
                failures.append(exc)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", attempts
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
        assert not failures


def start_response(handler, *, length=None):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    if length is not None:
        handler.send_header("Content-Length", str(length))
    handler.end_headers()
    handler.wfile.write(PARTIAL)
    handler.wfile.flush()


def client(gateway, *, identity="first"):
    body = json.dumps({"model": "mock", "input": identity}).encode()
    connection = socket.create_connection(("127.0.0.1", gateway.port), timeout=5)
    connection.sendall(
        f"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    observed = b""
    while PARTIAL not in observed:
        chunk = connection.recv(65536)
        assert chunk
        observed += chunk
    return connection


def disconnect(connection):
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    connection.close()


def gateway_for(case, monkeypatch, url):
    monkeypatch.setenv("BENCH_MOCK_KEY", "zero-paid-loopback")
    harness = HarnessConfig(
        name="codex",
        executable="/usr/bin/true",
        expected_version="mock",
        model="mock",
        provider_url=url,
        api_key_env="BENCH_MOCK_KEY",
    )
    return Gateway(
        harness, case["spec"], case["task"], case["repo"], case["root"] / "model-journal.jsonl"
    )


def observe(gateway, kind):
    observed = threading.Event()
    original = gateway.record

    def record(record_kind, **values):
        original(record_kind, **values)
        if record_kind == kind:
            observed.set()

    gateway.record = record
    return observed


def test_downstream_disconnect_still_drains_final_usage(case, monkeypatch):
    send_more, send_final = threading.Event(), threading.Event()

    def serve(handler):
        start_response(handler)
        assert send_more.wait(5)
        handler.wfile.write(PARTIAL)
        handler.wfile.flush()
        assert send_final.wait(5)
        handler.wfile.write(FINAL)
        handler.wfile.flush()

    with provider(serve) as (url, attempts):
        with gateway_for(case, monkeypatch, url) as gateway:
            delivery_failed = observe(gateway, "model_delivery_failed")
            connection = client(gateway)
            disconnect(connection)
            send_more.set()
            try:
                assert delivery_failed.wait(2), gateway.journal.read_text()
            finally:
                send_final.set()
            gateway.finish_inference(timeout=2)
            assert not gateway.unknown
            assert gateway.active == 0
            assert gateway.metered_tokens().model_dump() == dict(
                input_uncached=15,
                input_cached=4,
                cache_creation=0,
                output=7,
                indexer_small_model_total=0,
                complete=True,
            )
            records = [json.loads(line) for line in gateway.journal.read_text().splitlines()]
            assert [r["kind"] for r in records].count("model_delivery_failed") == 1
            assert [r["kind"] for r in records].count("model_end") == 1
            assert records[-1]["kind"] == "model_usage"
        assert len(attempts) == 1


@pytest.mark.parametrize("disconnect_client", [False, True])
@pytest.mark.parametrize("final_usage", [False, True])
def test_truncated_provider_stream_stays_unknown(case, monkeypatch, disconnect_client, final_usage):
    send_final = threading.Event()

    def serve(handler):
        start_response(handler, length=len(PARTIAL) * 2 + len(FINAL) + 1)
        assert send_final.wait(5)
        handler.wfile.write(PARTIAL)
        if final_usage:
            handler.wfile.write(FINAL)
        handler.wfile.flush()
        # EOF before both the declared response length and final usage.

    with provider(serve) as (url, attempts):
        with gateway_for(case, monkeypatch, url) as gateway:
            connection = client(gateway)
            if disconnect_client:
                disconnect(connection)
            send_final.set()
            try:
                gateway.finish_inference(timeout=2)
            finally:
                connection.close()
            assert gateway.unknown
            assert gateway.metered_tokens().complete is final_usage
            records = [json.loads(line) for line in gateway.journal.read_text().splitlines()]
            assert any(
                record["kind"] == "model_error" and record.get("error") == "IncompleteRead"
                for record in records
            )
        assert len(attempts) == 1


def test_provider_eof_without_final_usage_stays_unknown(case, monkeypatch):
    def serve(handler):
        start_response(handler)

    with provider(serve) as (url, attempts):
        with gateway_for(case, monkeypatch, url) as gateway:
            connection = client(gateway)
            try:
                gateway.finish_inference(timeout=2)
            finally:
                connection.close()
            assert gateway.unknown
            assert not gateway.metered_tokens().complete
            assert "IncompleteProviderUsage" in gateway.journal.read_text()
        assert len(attempts) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_known_provider_rejection_without_usage_remains_infra(case, monkeypatch, status):
    def serve(handler):
        body = b'{"error":"synthetic rejection before inference"}'
        handler.send_response(status)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    with provider(serve) as (url, attempts):
        with gateway_for(case, monkeypatch, url) as gateway:
            request = Request(
                gateway.url + "/v1/responses",
                data=json.dumps({"model": "mock", "input": "rejected"}).encode(),
            )
            with pytest.raises(HTTPError) as rejected:
                urlopen(request, timeout=2)
            assert rejected.value.code == status
            gateway.finish_inference(timeout=2)
            assert gateway.infrastructure_error == f"model provider rejected request: HTTP {status}"
            assert not gateway.unknown
        assert len(attempts) == 1


def test_delivery_drain_has_its_own_deadline(case, monkeypatch):
    send_more, release = threading.Event(), threading.Event()
    monkeypatch.setattr("mindie_bench.runner.gateway.MODEL_SETTLEMENT_GRACE_SEC", 0.05)

    def serve(handler):
        start_response(handler)
        assert send_more.wait(5)
        handler.wfile.write(PARTIAL)
        handler.wfile.flush()
        assert release.wait(5)
        handler.wfile.write(FINAL)

    with provider(serve) as (url, attempts):
        try:
            with gateway_for(case, monkeypatch, url) as gateway:
                settled = observe(gateway, "model_usage")
                connection = client(gateway)
                disconnect(connection)
                send_more.set()
                assert settled.wait(2), gateway.journal.read_text()
                gateway.finish_inference(timeout=2)
                assert gateway.unknown
                assert not gateway.metered_tokens().complete
                assert "TimeoutError" in gateway.journal.read_text()
        finally:
            release.set()
        assert len(attempts) == 1


def test_settlement_deadline_aborts_owned_read_without_reclassifying_unknown(case, monkeypatch):
    send_more, release = threading.Event(), threading.Event()

    def serve(handler):
        start_response(handler)
        assert send_more.wait(5)
        handler.wfile.write(PARTIAL)
        handler.wfile.flush()
        assert release.wait(5)
        handler.wfile.write(FINAL)
        handler.wfile.flush()

    with provider(serve) as (url, attempts):
        try:
            with gateway_for(case, monkeypatch, url) as gateway:
                delivery_failed = observe(gateway, "model_delivery_failed")
                connection = client(gateway)
                disconnect(connection)
                send_more.set()
                assert delivery_failed.wait(2), gateway.journal.read_text()
                with gateway.lock:
                    [(upstream, transport)] = gateway.connections.items()
                    assert upstream.sock is None
                    assert transport is not None
                started = time.monotonic()
                gateway.finish_inference(timeout=0.05)
                assert time.monotonic() - started < 1
                assert gateway.unknown
                assert gateway.active == 0
                assert not gateway.metered_tokens().complete
                assert "model_calls_unsettled" in gateway.journal.read_text()
                release.set()
                gateway.finish_inference(timeout=1)
                assert gateway.unknown
        finally:
            release.set()
        assert len(attempts) == 1


def test_chunked_settlement_keeps_live_response_close_with_its_reader(case, monkeypatch):
    release = threading.Event()
    reading, resume_reader = threading.Event(), threading.Event()
    live_close_threads = []
    caller = threading.get_ident()
    original = http.client.HTTPResponse

    class ObservedResponse(original):
        def read1(self, *args):
            if self.chunked and PARTIAL in getattr(self, "observed", b""):
                reading.set()
                assert resume_reader.wait(5)
            chunk = super().read1(*args)
            self.observed = getattr(self, "observed", b"") + chunk
            return chunk

        def close(self):
            if self.fp is not None:
                live_close_threads.append(threading.get_ident())
            super().close()

    monkeypatch.setattr(http.client.HTTPConnection, "response_class", ObservedResponse)

    def serve(handler):
        handler.protocol_version = "HTTP/1.1"
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        handler.wfile.write(f"{len(PARTIAL):x}\r\n".encode() + PARTIAL + b"\r\n")
        handler.wfile.flush()
        assert release.wait(5)

    with provider(serve) as (url, attempts):
        try:
            with gateway_for(case, monkeypatch, url) as gateway:
                settled = observe(gateway, "model_usage")
                connection = client(gateway)
                try:
                    assert reading.wait(2)
                    gateway.finish_inference(timeout=0.05)
                    resume_reader.set()
                    assert settled.wait(2), gateway.journal.read_text()
                    with gateway.model_idle:
                        assert gateway.model_idle.wait_for(lambda: gateway.active == 0, timeout=1)
                    assert gateway.unknown
                    assert not gateway.metered_tokens().complete
                    assert caller not in live_close_threads
                    records = [json.loads(line) for line in gateway.journal.read_text().splitlines()]
                    errors = [row for row in records if row["kind"] == "model_error"]
                    assert errors and errors[-1]["transport_stage"] == "response_body"
                finally:
                    resume_reader.set()
                    connection.close()
        finally:
            release.set()
        assert len(attempts) == 1


def test_sealing_inference_admits_no_new_paid_request_while_draining(case, monkeypatch):
    send_final = threading.Event()

    def serve(handler):
        start_response(handler)
        assert send_final.wait(5)
        handler.wfile.write(FINAL)
        handler.wfile.flush()

    with provider(serve) as (url, attempts):
        with gateway_for(case, monkeypatch, url) as gateway:
            connection = client(gateway)
            worker = threading.Thread(target=lambda: gateway.finish_inference(timeout=2))
            worker.start()
            try:
                with gateway.model_idle:
                    assert gateway.model_idle.wait_for(lambda: gateway.models_sealed, timeout=1)
                request = Request(
                    gateway.url + "/v1/responses",
                    data=json.dumps({"model": "mock", "input": "second-distinct"}).encode(),
                )
                with pytest.raises(HTTPError) as rejected:
                    urlopen(request, timeout=2)
                assert rejected.value.code == 409
                assert len(attempts) == gateway.model_calls == 1
            finally:
                send_final.set()
                worker.join(3)
                connection.close()
            assert not worker.is_alive()
            assert not gateway.unknown
            assert gateway.metered_tokens().complete


def test_settlement_close_cannot_implicitly_reconnect_before_submission(case, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    connects = []
    base = http.client.HTTPConnection

    class BarrierConnection(base):
        def connect(self):
            connects.append(True)
            super().connect()

        def request(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().request(*args, **kwargs)

    monkeypatch.setattr("http.client.HTTPConnection", BarrierConnection)

    def serve(handler):
        start_response(handler)
        handler.wfile.write(FINAL)
        handler.wfile.flush()

    with provider(serve) as (url, attempts):
        with gateway_for(case, monkeypatch, url) as gateway:
            settled = observe(gateway, "model_usage")
            body = json.dumps({"model": "mock", "input": "cancel-before-send"}).encode()
            connection = socket.create_connection(("127.0.0.1", gateway.port), timeout=5)
            try:
                connection.sendall(
                    f"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                assert entered.wait(2)
                gateway.finish_inference(timeout=0.05)
                assert gateway.unknown
                release.set()
                assert settled.wait(2), gateway.journal.read_text()
                assert connects == [True]
                assert attempts == []
                assert gateway.model_calls == 1
                assert not gateway.metered_tokens().complete
            finally:
                release.set()
                connection.close()
