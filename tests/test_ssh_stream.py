import base64
import contextlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import weakref

from mindie_bench.runner.host_platform import PipeReader, subprocess_window_kwargs

_STREAM_READERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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
        [sys.executable, "-I", "-X", "utf8", "-m", "mindie_bench.runner.ssh_shim", "--", "root@bench-slot", "rpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        **subprocess_window_kwargs(),
    )


def read_line(stream):
    pair = _STREAM_READERS.get(stream)
    if pair is None:
        pair = (PipeReader(stream), bytearray())
        _STREAM_READERS[stream] = pair
    reader, buf = pair
    deadline = time.monotonic() + 5
    while b"\n" not in buf:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "SSH facade did not forward the remote frame"
        timed_out, chunk = reader.read_chunk(remaining)
        assert not timed_out and chunk, "SSH facade did not forward the remote frame"
        buf.extend(chunk)
    idx = buf.index(b"\n") + 1
    line = bytes(buf[:idx])
    del buf[:idx]
    return line


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


def test_one_shot_timeout_zero_exit_is_not_success():
    def respond(path, request):
        assert path == "/bench/ssh"
        return 200, {
            "stdout_b64": encoded(b""),
            "stderr_b64": encoded(b"killed\n"),
            "exit_code": 0,
            "timed_out": True,
        }

    with broker(respond) as (url, requests):
        process = launch(url, stream=False)
        stdout, stderr = process.communicate(input=b"", timeout=5)
        assert process.returncode == 124
        assert stdout == b""
        assert stderr == b"killed\n"
        assert len(requests) == 1
