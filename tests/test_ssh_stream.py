import base64
import contextlib
import http.server
import json
import os
import select
import subprocess
import sys
import threading


def encoded(value):
    return base64.b64encode(value).decode()


@contextlib.contextmanager
def broker(respond):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            status, result = respond(self.path, payload)
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def launch(url, *, stream=True):
    environment = dict(os.environ, MINDIE_BENCH_GATEWAY=url)
    if stream:
        environment["MINDIE_BENCH_SSH_STREAM"] = "1"
    else:
        environment.pop("MINDIE_BENCH_SSH_STREAM", None)
    return subprocess.Popen(
        [sys.executable, "-I", "-m", "mindie_bench.runner.ssh_shim", "--", "root@bench-slot", "rpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def read_line(stream):
    assert select.select([stream], [], [], 5)[0], "SSH facade did not forward the remote frame"
    return stream.readline()


def test_stream_forwards_ready_before_caller_sends_request_or_eof():
    ready = False
    input_closed = False
    received = []

    def respond(path, request):
        nonlocal ready, input_closed
        assert path == "/bench/ssh-stream"
        if request["action"] == "open":
            assert request["command"] == "rpc"
            return 200, {"stream_id": "owned-stream"}
        assert request["stream_id"] == "owned-stream"
        data = base64.b64decode(request["stdin_b64"])
        output = error = b""
        if not ready:
            assert not data and not request["eof"]
            ready = True
            output = b'{"ready":true}\n'
        elif data:
            received.append(json.loads(data))
            output = b'{"result":"acknowledged"}\n'
            error = b"remote diagnostic\n"
        elif request["eof"]:
            input_closed = True
            output = b"drained after EOF\n"
        return 200, {
            "stdout_b64": encoded(output),
            "stderr_b64": encoded(error),
            "exit_code": 0 if input_closed and not request["eof"] else None,
        }

    with broker(respond) as (url, requests):
        process = launch(url)
        try:
            assert read_line(process.stdout) == b'{"ready":true}\n'
            process.stdin.write(b'{"request":"once"}\n')
            process.stdin.flush()
            assert read_line(process.stdout) == b'{"result":"acknowledged"}\n'
            assert read_line(process.stderr) == b"remote diagnostic\n"
            process.stdin.close()
            assert read_line(process.stdout) == b"drained after EOF\n"
            assert process.wait(timeout=5) == 0
            assert received == [{"request": "once"}]
            assert sum(request.get("eof", False) for _, request in requests) == 1
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()


def test_stream_does_not_retry_an_unknown_exchange():
    def respond(path, request):
        assert path == "/bench/ssh-stream"
        if request["action"] == "open":
            return 200, {"stream_id": "owned-stream"}
        return 502, {"error": "remote operation has unknown effects; no retry"}

    with broker(respond) as (url, requests):
        process = launch(url)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert stdout == b""
        assert b"remote operation has unknown effects; no retry" in stderr
        assert [request["action"] for _, request in requests] == ["open", "exchange"]


def test_one_shot_keeps_original_bounded_stdin_protocol():
    def respond(path, request):
        assert path == "/bench/ssh"
        assert request == {"command": "rpc", "stdin_b64": encoded(b"one shot\n")}
        return 200, {
            "stdout_b64": encoded(b"done\n"),
            "stderr_b64": encoded(b"diagnostic\n"),
            "exit_code": 7,
        }

    with broker(respond) as (url, requests):
        process = launch(url, stream=False)
        stdout, stderr = process.communicate(input=b"one shot\n", timeout=5)
        assert process.returncode == 7
        assert stdout == b"done\n"
        assert stderr == b"diagnostic\n"
        assert len(requests) == 1
