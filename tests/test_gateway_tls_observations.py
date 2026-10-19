"""Real TLS handshake failure remains unknown and records no provider secrets."""

import http.client
import json
import socket
import threading
import urllib.request

import pytest

from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway


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
        name="claude-code", executable="/usr/bin/true", model="pinned",
        expected_version="fixture", api_key_env="BENCH_TLS_KEY",
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
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    failure = next(row for row in rows if row["kind"] == "model_error")
    assert failure["error"] == "SSLError"
    assert failure["transport_stage"] == "connect"
    # OpenSSL 3.5 reports a record-layer failure; older builds name the version.
    assert failure["ssl_reason"]
    assert failure["errno"] is not None
    assert secret not in journal.read_text()
    assert not any(row["kind"] == "model_end" for row in rows)
