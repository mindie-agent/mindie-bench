import base64
import json
import select
import socket
import ssl
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.ssh_shim import _post
from mindie_bench.shim import npu_replay_shim

FIXTURES = Path(__file__).parent / "fixtures/transport"


@contextmanager
def serving(handler, tls=False):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(FIXTURES / "test-cert.pem", FIXTURES / "test-key.pem")
            server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=2)


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass


def set_proxy(monkeypatch, value, bypass=""):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(key, value)
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, bypass)


@pytest.mark.parametrize("bypass,tls", [("", True), ("127.0.0.1", True), ("", False)])
def test_model_connect_uses_proxy_auth_and_verified_tls_without_forwarding_credentials(
    case, monkeypatch, bypass, tls
):
    tunnels, received = [], []

    class Provider(QuietHandler):
        def do_POST(self):
            received.append((self.path, dict(self.headers)))
            self.rfile.read(int(self.headers["Content-Length"]))
            capture = json.loads((FIXTURES.parent / "deepseek-flash-usage.json").read_text())
            body = b"data: " + json.dumps(capture).encode() + b"\n\ndata: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class Proxy(QuietHandler):
        def do_CONNECT(self):
            tunnels.append((self.path, self.headers["Proxy-Authorization"]))
            with socket.create_connection(("127.0.0.1", provider.server_port), timeout=3) as upstream:
                self.send_response(200)
                self.end_headers()
                while True:
                    ready, _, _ = select.select([self.connection, upstream], [], [], 3)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        target = upstream if source is self.connection else self.connection
                        target.sendall(data)

    with serving(Provider, tls=tls) as provider, serving(Proxy) as proxy:
        set_proxy(monkeypatch, f"http://test-user:test%3Apass@127.0.0.1:{proxy.server_port}", bypass)
        monkeypatch.setenv("SSL_CERT_FILE", str(FIXTURES / "test-cert.pem"))
        monkeypatch.setenv("BENCH_TEST_KEY", "provider-only-secret")
        harness = HarnessConfig(
            name="codex",
            executable="/usr/bin/true",
            model="pinned",
            expected_version="fixture",
            api_key_env="BENCH_TEST_KEY",
            provider_url=f"{'https' if tls else 'http'}://127.0.0.1:{provider.server_port}/v1",
        )
        journal = case["root"] / "proxy-journal"
        with Gateway(harness, case["spec"], case["task"], case["repo"], journal) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/chat/completions",
                data=json.dumps({"model": "pinned", "stream": True}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                request, timeout=5
            ) as response:
                assert b"[DONE]" in response.read()
            gateway.finish_inference(timeout=2)
            assert gateway.model_calls == 1
            assert not gateway.unknown
            assert gateway.metered_tokens().complete
        if bypass or not tls:
            assert tunnels == []
        else:
            auth = base64.b64encode(b"test-user:test:pass").decode()
            assert tunnels == [(f"127.0.0.1:{provider.server_port}", "Basic " + auth)]
        assert len(received) == 1
        assert received[0][0] == "/v1/chat/completions"
        assert received[0][1]["Authorization"] == "Bearer provider-only-secret"
        assert "Proxy-Authorization" not in received[0][1]
        text = journal.read_text()
        assert "test:pass" not in text and "provider-only-secret" not in text


@pytest.mark.parametrize("client", ["ssh", "replay"])
def test_local_rpc_bypasses_proxy_without_no_proxy(monkeypatch, client):
    calls = []

    class Broker(QuietHandler):
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"stdout":"","stderr":"","exit_code":0}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with serving(Broker) as broker:
        set_proxy(monkeypatch, "http://127.0.0.1:1")
        url = f"http://127.0.0.1:{broker.server_port}"
        if client == "ssh":
            assert _post(url, "/bench/ssh", {"command": "true"})["exit_code"] == 0
        else:
            monkeypatch.setenv("MINDIE_BENCH_GATEWAY", url)
            assert npu_replay_shim.main() == 0
        assert calls == ["/bench/" + client]
