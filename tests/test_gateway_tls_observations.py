"""Real local TLS transport observations remain additive, bounded, and secret-free."""

import http.client
import json
import select
import socket
import ssl
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mindie_bench.evaluation.broker_knowledge import summarize_broker_knowledge
from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway

FIXTURES = Path(__file__).parent / "fixtures/transport"
CERT, KEY = FIXTURES / "test-cert.pem", FIXTURES / "test-key.pem"
CAPTURE = Path(__file__).parent / "fixtures/deepseek-flash-usage.json"
ALERT_TRIGGER = b"\x17\x03\x03\x00\x05hello"  # valid TLS record framing, invalid MAC


def _read_http_request(tls):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = tls.recv(4096)
        if not chunk:
            raise ConnectionError("client closed before sending a request")
        data += chunk
        if len(data) > 1024 * 1024:
            raise ValueError("request exceeds test bound")
    head, _, body = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    while len(body) < length:
        chunk = tls.recv(4096)
        if not chunk:
            raise ConnectionError("client closed before sending its body")
        body += chunk
    return head.split(b"\r\n")[0], body[:length], len(head) + 4 + len(body)


class LocalTlsEndpoint:
    """Loopback TLS endpoint plus a pass-through proxy that corrupts one record."""

    def __init__(self, mode="alert", response=b""):
        self.requests, self.server_error, self.injected, self.tls_version = [], None, False, None
        self.mode, self.response = mode, response
        self._ready, self._stop = threading.Event(), threading.Event()
        self._server_conn = self._client = self._upstream = None
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._proxy_listener = socket.socket()
        self._proxy_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._proxy_listener.bind(("127.0.0.1", 0))
        self._proxy_listener.listen(1)
        # A closed listener does not reliably wake accept() on another thread on
        # Linux. Keep both accepts on a short bounded timeout and stop them with
        # the event instead of relying on cross-thread close().
        self._listener.settimeout(0.2)
        self._proxy_listener.settimeout(0.2)
        self.tls_port = self._listener.getsockname()[1]
        self.proxy_port = self._proxy_listener.getsockname()[1]
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(CERT, KEY)
        self._server_thread = threading.Thread(target=self._serve_tls, daemon=True)
        self._proxy_thread = threading.Thread(target=self._proxy_once, daemon=True)
        self._server_thread.start()
        self._proxy_thread.start()

    @property
    def direct_url(self):
        return f"https://127.0.0.1:{self.tls_port}/v1"

    @property
    def proxy_url(self):
        return f"https://127.0.0.1:{self.proxy_port}/v1"

    def _accept_until_stopped(self, listener):
        while not self._stop.is_set():
            try:
                return listener.accept()
            except socket.timeout:
                continue
        return None

    def _serve_tls(self):
        try:
            connection = self._accept_until_stopped(self._listener)
            if connection is None:
                return
            accepted, _ = connection
            self._server_conn = accepted
            with self._context.wrap_socket(accepted, server_side=True) as tls:
                self.tls_version = tls.version()
                request_line, body, wire_bytes = _read_http_request(tls)
                self.requests.append((request_line, len(body), wire_bytes))
                self._ready.set()
                tls.settimeout(5)
                if self.mode == "alert":
                    tls.recv(4096)  # The bad record makes OpenSSL send a fatal alert.
                else:
                    tls.sendall(self.response)
        except ssl.SSLError as exc:
            self.server_error = exc.reason or exc.strerror
        except TimeoutError:
            raise
        except (ConnectionError, OSError, ValueError) as exc:
            self.server_error = type(exc).__name__

    def _proxy_once(self):
        try:
            connection = self._accept_until_stopped(self._proxy_listener)
            if connection is None:
                return
            client, _ = connection
            self._client = client
            upstream = socket.create_connection(self._listener.getsockname(), timeout=5)
            self._upstream = upstream
            while not self._stop.is_set():
                readable, _, _ = select.select([client, upstream], [], [], 0.05)
                for source in readable:
                    target = upstream if source is client else client
                    try:
                        data = source.recv(65536)
                    except TimeoutError:
                        raise
                    except OSError:
                        data = b""
                    if not data:
                        try:
                            target.shutdown(socket.SHUT_WR)
                        except TimeoutError:
                            raise
                        except OSError:
                            pass
                        return
                    target.sendall(data)
                if self._ready.is_set() and not self.injected:
                    upstream.sendall(ALERT_TRIGGER)
                    self.injected = True
        except TimeoutError:
            raise
        except OSError:
            return

    def close(self):
        self._stop.set()
        # Wake active recv/select loops, then let the bounded accept loops observe
        # _stop. Closing a listening socket from this thread is not a portable
        # way to interrupt another thread's blocking accept().
        for sock in (self._server_conn, self._client, self._upstream):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except TimeoutError:
                    raise
                except OSError:
                    pass
        for thread in (self._server_thread, self._proxy_thread):
            thread.join(timeout=5)
        for sock in (
            self._server_conn,
            self._client,
            self._upstream,
            self._listener,
            self._proxy_listener,
        ):
            if sock is not None:
                try:
                    sock.close()
                except TimeoutError:
                    raise
                except OSError:
                    pass
        assert not self._server_thread.is_alive() and not self._proxy_thread.is_alive()


def _harness(endpoint, monkeypatch, key="BENCH_TLS_TEST_KEY"):
    secret = "fixture-provider-parent-secret"
    monkeypatch.setenv(key, secret)
    return HarnessConfig(
        name="claude-code",
        executable="/usr/bin/true",
        model="pinned",
        expected_version="fixture",
        api_key_env=key,
        provider_url=endpoint,
    ), secret


def _rows(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()]


def test_fatal_alert_after_request_write_is_unknown_usage_incomplete_and_not_replayed(case, monkeypatch):
    endpoint = LocalTlsEndpoint()
    monkeypatch.setenv("SSL_CERT_FILE", str(CERT))
    harness, secret = _harness(endpoint.proxy_url, monkeypatch)
    journal = case["root"] / "tls-alert-journal.jsonl"
    payload = json.dumps({"model": "pinned", "messages": [{"role": "user", "content": "fixture"}]})
    try:
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/chat/completions",
                data=payload.encode(),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(http.client.RemoteDisconnected):
                urllib.request.urlopen(request, timeout=5)
            assert gateway.unknown and gateway.model_calls == 1
            assert gateway.usage_meters[1].tokens.complete is False
            with pytest.raises(urllib.error.HTTPError) as replay:
                urllib.request.urlopen(request, timeout=5)
            assert replay.value.code == 409
            assert gateway.model_calls == 1
            gateway.finish_inference(timeout=5)
    finally:
        endpoint.close()
    assert len(endpoint.requests) == 1
    request_line, body_bytes, wire_bytes = endpoint.requests[0]
    assert request_line == b"POST /v1/chat/completions HTTP/1.1"
    assert body_bytes == len(payload)
    assert endpoint.injected and endpoint.server_error
    rows = _rows(journal)
    assert [row["kind"] for row in rows].count("model_start") == 1
    assert not any(row["kind"] == "model_end" for row in rows)
    failure = next(row for row in rows if row["kind"] == "model_error")
    assert failure["error"] == "SSLError"
    assert failure["transport_stage"] == "response_headers"
    assert "ALERT" in failure["ssl_reason"].upper()
    assert failure["alert_reason"] == failure["ssl_reason"]
    assert failure["alert_direction"] == "inferred_received_from_peer"
    assert failure["alert_direction_basis"] == "openssl_reason_contains_ALERT"
    assert failure["alert_direction_measured"] is False
    assert failure["tls_version"].startswith("TLS")
    assert failure["tls_cipher"]
    assert failure["tls_alpn"] is None
    # getpeername() is the loopback proxy, never evidence of the provider edge.
    assert failure["peer_scope"] == "immediate_socket_peer"
    assert failure["peer_ip"] == "127.0.0.1"
    assert failure["peer_port"] == endpoint.proxy_port
    assert failure["request_bytes_written"] == wire_bytes
    assert failure["request_write_complete"] is True
    assert failure["request_write_uncertain_bytes"] == 0
    assert failure["transport_stage_sec"] > 0
    assert set(failure["transport_stage_durations"]) == {
        "connect",
        "submit",
        "response_headers",
        "response_body",
    }
    assert failure["transport_stage_durations"]["connect"] > 0
    assert failure["transport_stage_durations"]["submit"] > 0
    assert failure["transport_stage_durations"]["response_headers"] > 0
    assert failure["transport_stage_durations"]["response_body"] is None
    usage = next(row for row in rows if row["kind"] == "model_usage")
    assert usage["tokens"]["complete"] is False
    assert secret not in journal.read_text()
    assert "fixture" not in journal.read_text()


def test_failed_request_submission_separates_confirmed_and_uncertain_bytes(case, monkeypatch):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    provider_port = listener.getsockname()[1]
    observed = {}

    def provider():
        try:
            client, _ = listener.accept()
        except TimeoutError:
            raise
        except OSError:
            return
        with client:
            client.settimeout(5)
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = client.recv(4096)
                if not chunk:
                    break
                head += chunk
            observed["header_bytes"] = len(head)
            try:
                observed["after_close"] = client.recv(1)
            except TimeoutError:
                raise
            except OSError:
                observed["after_close"] = b""

    worker = threading.Thread(target=provider, daemon=True)
    worker.start()
    secret = "fixture-provider-parent-secret"
    monkeypatch.setenv("BENCH_PARTIAL_KEY", secret)
    harness = HarnessConfig(
        name="claude-code",
        executable="/usr/bin/true",
        model="pinned",
        expected_version="fixture",
        api_key_env="BENCH_PARTIAL_KEY",
        provider_url=f"http://127.0.0.1:{provider_port}/v1",
    )
    journal = case["root"] / "submit-failure-journal.jsonl"
    payload = json.dumps(
        {"model": "pinned", "messages": [{"role": "user", "content": "x" * 256}]}
    ).encode()
    original_send = http.client.HTTPConnection.send
    sends = {"headers": 0}

    def inject_body_failure(connection, data):
        try:
            peer = connection.sock.getpeername()
        except TimeoutError:
            raise
        except (AttributeError, OSError):
            peer = None
        if peer is not None and peer[1] == provider_port:
            if sends["headers"] == 0:
                sends["headers"] = 1
                return original_send(connection, data)  # Confirm the header chunk.
            raise BrokenPipeError(32, "injected failure at request body write")
        return original_send(connection, data)

    monkeypatch.setattr(http.client.HTTPConnection, "send", inject_body_failure)
    try:
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises((http.client.RemoteDisconnected, ConnectionResetError)):
                urllib.request.urlopen(request, timeout=5)
            assert gateway.unknown and gateway.model_calls == 1
            assert gateway.usage_meters[1].tokens.complete is False
            with pytest.raises(urllib.error.HTTPError) as replay:
                urllib.request.urlopen(request, timeout=5)
            assert replay.value.code == 409
            assert gateway.model_calls == 1
            gateway.finish_inference(timeout=5)
    finally:
        worker.join(timeout=5)
        listener.close()
    assert not worker.is_alive()
    assert sends["headers"] == 1
    assert observed["header_bytes"] > 0
    assert observed["after_close"] == b""
    rows = _rows(journal)
    assert [row["kind"] for row in rows].count("model_start") == 1
    assert not any(row["kind"] == "model_end" for row in rows)
    failure = next(row for row in rows if row["kind"] == "model_error")
    assert failure["error"] == "BrokenPipeError"
    assert failure["errno"] == 32
    assert failure["transport_stage"] == "submit"
    assert failure["transport_stage_durations"]["submit"] > 0
    assert failure["transport_stage_durations"]["response_headers"] is None
    assert failure["request_write_complete"] is False
    assert failure["request_bytes_written"] == observed["header_bytes"]
    assert failure["request_write_uncertain_bytes"] == len(payload)
    assert failure["alert_reason"] is None
    assert failure["alert_direction"] is None
    assert failure["peer_scope"] == "immediate_socket_peer"
    assert failure["peer_ip"] == "127.0.0.1"
    usage = next(row for row in rows if row["kind"] == "model_usage")
    assert usage["tokens"]["complete"] is False
    assert secret not in journal.read_text()


def test_certificate_validation_stays_on_and_unavailable_fields_are_null(case, monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    endpoint = LocalTlsEndpoint(mode="success", response=b"")
    harness, _ = _harness(endpoint.direct_url, monkeypatch)
    journal = case["root"] / "tls-untrusted-journal.jsonl"
    body = json.dumps({"model": "pinned", "messages": []}).encode()
    try:
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(http.client.RemoteDisconnected):
                urllib.request.urlopen(request, timeout=5)
            gateway.finish_inference(timeout=5)
    finally:
        endpoint.close()
    failure = next(row for row in _rows(journal) if row["kind"] == "model_error")
    assert failure["transport_stage"] == "connect"
    assert failure["error"] in {"SSLError", "SSLCertVerificationError"}
    assert "CERTIFICATE_VERIFY_FAILED" in (failure["ssl_reason"] or "").upper()
    assert failure["tls_version"] is None
    assert failure["tls_cipher"] is None
    assert failure["tls_alpn"] is None
    assert failure["peer_ip"] is None
    assert failure["peer_port"] is None
    assert failure["peer_scope"] is None
    assert failure["alert_reason"] is None
    assert failure["alert_direction"] is None
    assert failure["alert_direction_basis"] is None
    assert failure["alert_direction_measured"] is None
    assert failure["request_bytes_written"] == 0
    assert failure["request_write_complete"] is False
    assert failure["transport_stage_durations"]["submit"] is None
    assert endpoint.requests == []


def test_success_stream_bytes_usage_and_model_end_schema_are_unchanged(case, monkeypatch):
    payload = json.loads(CAPTURE.read_text())
    expected = b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"
    header = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        b"Content-Length: " + str(len(expected)).encode() + b"\r\nConnection: close\r\n\r\n"
    )
    endpoint = LocalTlsEndpoint(mode="success", response=header + expected)
    monkeypatch.setenv("SSL_CERT_FILE", str(CERT))
    harness, _ = _harness(endpoint.direct_url, monkeypatch)
    journal = case["root"] / "tls-success-journal.jsonl"
    body = json.dumps({"model": "pinned", "messages": [], "stream": True}).encode()
    try:
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.read() == expected
            gateway.finish_inference()
            assert gateway.usage_meters[1].tokens.complete
            assert gateway.usage_meters[1].tokens.output == 11
            assert not gateway.unknown
    finally:
        # The direct success endpoint has no proxy connection; closing is still bounded.
        endpoint.close()
    rows = _rows(journal)
    assert not any(row["kind"] == "model_error" for row in rows)
    end = next(row for row in rows if row["kind"] == "model_end")
    assert set(end) == {"kind", "monotonic_sec", "ordinal", "http_status"}
    assert end["http_status"] == 200


def test_existing_journal_consumer_ignores_additive_transport_fields(tmp_path):
    legacy = {
        "kind": "model_error",
        "monotonic_sec": 10.0,
        "ordinal": 4,
        "error": "SSLError",
        "transport_stage": "response_headers",
        "errno": 1,
        "ssl_reason": "SSLV3_ALERT_BAD_RECORD_MAC",
    }
    extended = dict(
        legacy,
        ordinal=5,
        transport_stage_sec=0.43,
        transport_stage_durations={
            "connect": 0.1,
            "submit": 0.2,
            "response_headers": 0.43,
            "response_body": None,
        },
        request_bytes_written=2048,
        request_write_complete=True,
        request_write_uncertain_bytes=0,
        alert_reason=legacy["ssl_reason"],
        alert_direction="inferred_received_from_peer",
        alert_direction_basis="openssl_reason_contains_ALERT",
        alert_direction_measured=False,
        tls_version="TLSv1.3",
        tls_cipher="TLS_AES_256_GCM_SHA384",
        tls_alpn=None,
        peer_ip="127.0.0.1",
        peer_port=443,
        peer_scope="immediate_socket_peer",
    )
    events = [
        legacy,
        extended,
        {
            "kind": "knowledge_tool_start",
            "monotonic_sec": 20.0,
            "server": "knowledge",
            "session_id": "s",
            "request_id": 1,
            "tool": "knowledge_query",
        },
        {
            "kind": "knowledge_tool_end",
            "monotonic_sec": 23.0,
            "server": "knowledge",
            "session_id": "s",
            "request_id": 1,
            "tool": "knowledge_query",
            "elapsed_sec": 3.0,
            "is_error": False,
        },
    ]
    journal = tmp_path / "broker.jsonl"
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    observed = summarize_broker_knowledge(journal)
    assert observed["query_count"] == 1
    assert observed["timing"]["reported_elapsed_cumulative_sec"] == 3.0
    assert len(observed["journal_sha256"]) == 64


def test_plain_endpoint_tls_failure_records_stage_without_request_credentials(case, monkeypatch):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    accepted = []

    def plain_endpoint():
        with listener:
            client, _ = listener.accept()
            with client:
                client.settimeout(5)
                accepted.append(client.recv(4096))
                client.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")

    worker = threading.Thread(target=plain_endpoint, daemon=True)
    worker.start()
    secret = "fixture-provider-parent-secret"
    monkeypatch.setenv("BENCH_TLS_KEY", secret)
    harness = HarnessConfig(
        name="claude-code",
        executable="/usr/bin/true",
        model="pinned",
        expected_version="fixture",
        api_key_env="BENCH_TLS_KEY",
        provider_url=f"https://127.0.0.1:{listener.getsockname()[1]}/v1",
    )
    journal = case["root"] / "tls-journal.jsonl"
    try:
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/messages",
                data=json.dumps({"model": "pinned", "messages": []}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(http.client.RemoteDisconnected):
                urllib.request.urlopen(request, timeout=5)
            assert gateway.unknown
            assert gateway.model_calls == 1
    finally:
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(accepted) == 1
    assert secret.encode() not in accepted[0]
    rows = _rows(journal)
    failure = next(row for row in rows if row["kind"] == "model_error")
    assert failure["error"] == "SSLError"
    assert failure["transport_stage"] == "connect"
    # OpenSSL 3.5 reports a record-layer failure; older builds name the version.
    assert failure["ssl_reason"]
    assert failure["errno"] is not None
    assert secret not in journal.read_text()
    assert not any(row["kind"] == "model_end" for row in rows)
