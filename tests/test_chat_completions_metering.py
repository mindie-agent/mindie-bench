import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mindie_bench.models import HarnessConfig
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.usage import UsageMeter

CAPTURE = Path(__file__).parent / "fixtures/deepseek-flash-usage.json"


def test_actual_deepseek_stream_usage_survives_fragmented_delivery():
    # Captured from deepseek-flash on 2026-10-19; only the request ID was replaced.
    raw = b"data: " + CAPTURE.read_bytes().replace(b"\n", b"") + b"\n\ndata: [DONE]\n\n"
    meter = UsageMeter()
    for index in range(0, len(raw), 7):
        meter.feed(raw[index : index + 7])
    meter.finish()
    assert meter.tokens.complete
    assert meter.tokens.input_uncached == 35
    assert meter.tokens.input_cached == 0
    assert meter.tokens.output == 11


def test_dsh_chat_completions_forwarding_and_usage(case, monkeypatch):
    received = []
    payload = json.loads(CAPTURE.read_text())

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body["model"], self.headers.get("Authorization")))
            raw = b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BENCH_DSH_TEST_KEY", "local-test-only")
    harness = HarnessConfig(
        name="dsh",
        executable="/usr/bin/true",
        model="deepseek-flash",
        expected_version="0.2.0-rc.2",
        api_protocol="openai-completions",
        provider_url=f"http://127.0.0.1:{provider.server_port}/v1",
        api_key_env="BENCH_DSH_TEST_KEY",
    )
    try:
        with Gateway(
            harness,
            case["spec"],
            case["task"],
            case["repo"],
            case["root"] / "journal",
        ) as gateway:
            request = urllib.request.Request(
                gateway.url + "/v1/chat/completions",
                data=json.dumps({"model": "deepseek-flash", "messages": [], "stream": True}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
            gateway.finish_inference()
            assert gateway.usage_meters[1].tokens.complete
            assert gateway.usage_meters[1].tokens.output == 11
            assert not gateway.unknown
        assert received == [("/v1/chat/completions", "deepseek-flash", "Bearer local-test-only")]
    finally:
        provider.shutdown()
        thread.join(timeout=2)
        provider.server_close()
