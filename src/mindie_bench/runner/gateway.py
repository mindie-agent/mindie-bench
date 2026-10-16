"""Parent-side model/replay broker with deny-by-default egress and no retries."""

from __future__ import annotations

import hashlib
import http.client
import http.server
import json
import os
import re
import socketserver
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from urllib.parse import urlsplit

from mindie_bench.io import read_json, safe_relative
from mindie_bench.models import HarnessConfig, TaskSpec
from mindie_bench.runner.usage import UsageMeter, sum_usage


class Gateway(AbstractContextManager):
    def __init__(
        self,
        harness: HarnessConfig,
        task: TaskSpec,
        task_root: Path,
        workspace: Path,
        journal: Path,
        *,
        unix_socket: Path | None = None,
        remote=None,
    ):
        self.harness, self.task, self.task_root = harness, task, task_root
        self.workspace, self.journal = workspace, journal
        self.lock = threading.RLock()
        self.model_calls = self.replay_calls = 0
        self.stop_reason: str | None = None
        self.unknown = False
        self.infrastructure_error: str | None = None
        self.seen: set[str] = set()
        self.active = 0
        self.closed = False
        self.connections = set()
        self.usage_meters = {}
        self.unix_socket = unix_socket
        self.remote = remote
        self.remote_test_calls = 0
        self.api_key = os.environ.get(harness.api_key_env or "", "")
        if harness.name != "fixture" and not self.api_key:
            raise ValueError(f"missing explicitly selected credential: {harness.api_key_env}")
        if harness.provider_url:
            parsed = urlsplit(harness.provider_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("model provider_url must be a credential-free HTTPS URL")
        self.server = None
        self.thread = None

    def record(self, kind: str, **values):
        with self.lock, self.journal.open("a", encoding="utf-8") as out:
            out.write(json.dumps(dict(kind=kind, monotonic_sec=time.monotonic(), **values)) + "\n")
            out.flush()

    def replay(self, request: dict) -> dict:
        argv = request.get("argv")
        declared = self.task.environment.get("l2")
        if (
            not isinstance(argv, list)
            or not argv
            or not declared
            or argv[0] not in declared.replay_commands
        ):
            raise ValueError("unregistered replay command")
        rules = read_json(self.task_root / "oracle/replay_rules.json")
        if rules.get("schema_version") != "1.0" or not isinstance(rules.get("checks"), list):
            raise ValueError("invalid replay rules")
        with self.lock:
            if self.replay_calls >= self.task.budget.max_npu_runs:
                self.stop_reason = "max_npu_runs"
                return dict(stdout="", stderr="benchmark test-run budget exhausted\n", exit_code=124)
            self.replay_calls += 1
        passed = True
        for check in rules["checks"]:
            path = (self.workspace / safe_relative(check["path"])).resolve()
            if not path.is_relative_to(self.workspace.resolve()):
                raise ValueError("replay check escapes workspace")
            content = path.read_text() if path.is_file() else ""
            if len(content) > 4 * 1024 * 1024:
                raise ValueError("replay input exceeds bound")
            if "contains" in check:
                passed = passed and check["contains"] in content
            if "regex" in check:
                passed = passed and re.search(check["regex"], content, re.MULTILINE) is not None
            if "not_contains" in check:
                passed = passed and check["not_contains"] not in content
        outcome = rules["pass" if passed else "fail"]
        self.record("replay", ordinal=self.replay_calls, argv=argv, passed=passed)
        return dict(
            stdout=str(outcome.get("stdout", "")),
            stderr=str(outcome.get("stderr", "")),
            exit_code=int(outcome["exit_code"]),
        )

    def __enter__(self):
        gateway = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reject(self, code, message):
                payload = json.dumps(
                    {"error": {"type": "benchmark_gateway", "message": message}}
                ).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 32 * 1024 * 1024:
                        return self.reject(413, "invalid request size")
                    body = self.rfile.read(length)
                    if len(body) != length:
                        return self.reject(400, "incomplete request")
                    if self.path == "/bench/replay":
                        payload = json.dumps(gateway.replay(json.loads(body))).encode()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    if self.path == "/bench/ssh":
                        if gateway.remote is None:
                            return self.reject(403, "no remote slot is assigned")
                        import base64

                        from mindie_bench.scorer.metrics import NPU_TEST

                        request = json.loads(body)
                        command, stdin_b64 = request["command"], request["stdin_b64"]
                        stdin = base64.b64decode(stdin_b64, validate=True)
                        if not isinstance(command, str) or len(stdin) > 8 * 1024 * 1024:
                            return self.reject(400, "invalid remote command")
                        test_run = bool(NPU_TEST.search(command + "\n" + stdin.decode(errors="replace")))
                        with gateway.lock:
                            if (
                                test_run
                                and gateway.remote_test_calls >= gateway.task.budget.max_npu_runs
                            ):
                                gateway.stop_reason = "max_npu_runs"
                                return self.reject(429, "benchmark NPU test budget exhausted")
                            gateway.remote_test_calls += int(test_run)
                        started = time.monotonic()
                        try:
                            result = gateway.remote.execute(command, stdin_b64)
                        except Exception as exc:
                            gateway.unknown = True
                            gateway.record("remote_unknown", error=type(exc).__name__, message=str(exc))
                            return self.reject(502, "remote operation has unknown effects; no retry")
                        gateway.record(
                            "remote_end",
                            duration_sec=time.monotonic() - started,
                            exit_code=result["exit_code"],
                            test_run=test_run,
                        )
                        payload = json.dumps(result).encode()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    if self.path not in {"/v1/responses", "/responses", "/v1/messages", "/messages"}:
                        return self.reject(403, "egress endpoint is not permitted")
                    if gateway.harness.name == "fixture":
                        return self.reject(403, "fixture trials cannot call a paid model")
                    if json.loads(body).get("model") != gateway.harness.model:
                        gateway.infrastructure_error = "request attempted to change the pinned model"
                        return self.reject(403, gateway.infrastructure_error)
                    request_hash = hashlib.sha256(body).hexdigest()
                    with gateway.lock:
                        if gateway.closed or gateway.unknown:
                            return self.reject(409, "prior request is unknown; no retry")
                        if request_hash in gateway.seen:
                            return self.reject(409, "duplicate paid request; no replay")
                        if gateway.model_calls >= gateway.task.budget.max_turns:
                            gateway.stop_reason = "max_turns"
                            return self.reject(429, "benchmark turn budget exhausted")
                        gateway.seen.add(request_hash)
                        gateway.model_calls += 1
                        gateway.active += 1
                        ordinal = gateway.model_calls
                        gateway.usage_meters[ordinal] = UsageMeter()
                    gateway.record("model_start", ordinal=ordinal, request_sha256=request_hash)
                    endpoint = urlsplit(gateway.harness.provider_url)
                    connection = http.client.HTTPSConnection(
                        endpoint.hostname, endpoint.port, timeout=120
                    )
                    with gateway.lock:
                        gateway.connections.add(connection)
                    target = endpoint.path.rstrip("/") + "/" + self.path.removeprefix("/v1/").lstrip("/")
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": self.headers.get("Accept", "*/*"),
                    }
                    if gateway.harness.name == "codex":
                        headers["Authorization"] = "Bearer " + gateway.api_key
                    else:
                        headers["x-api-key"] = gateway.api_key
                        headers["anthropic-version"] = self.headers.get(
                            "anthropic-version", "2023-06-01"
                        )
                        if self.headers.get("anthropic-beta"):
                            headers["anthropic-beta"] = self.headers["anthropic-beta"]
                    try:
                        connection.request("POST", target, body=body, headers=headers)
                        response = connection.getresponse()
                        if response.status >= 400:
                            if response.status in {400, 401, 403, 404, 422}:
                                gateway.infrastructure_error = (
                                    f"model provider rejected request: HTTP {response.status}"
                                )
                            else:
                                gateway.unknown = True
                            gateway.record("model_error", ordinal=ordinal, http_status=response.status)
                        self.send_response(response.status)
                        self.send_header(
                            "Content-Type", response.getheader("Content-Type", "application/json")
                        )
                        self.end_headers()
                        while chunk := response.read1(65536):
                            gateway.usage_meters[ordinal].feed(chunk)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        gateway.record("model_end", ordinal=ordinal, http_status=response.status)
                        gateway.usage_meters[ordinal].finish()
                    except (OSError, http.client.HTTPException) as exc:
                        gateway.unknown = True
                        gateway.record("model_error", ordinal=ordinal, error=type(exc).__name__)
                        self.close_connection = True
                    finally:
                        gateway.record(
                            "model_usage",
                            ordinal=ordinal,
                            tokens=gateway.usage_meters[ordinal].tokens.model_dump(),
                        )
                        connection.close()
                        with gateway.lock:
                            gateway.connections.discard(connection)
                            gateway.active -= 1
                except (ValueError, KeyError, OSError) as exc:
                    gateway.record("broker_error", error=type(exc).__name__, message=str(exc))
                    gateway.infrastructure_error = f"{type(exc).__name__}: {exc}"
                    self.reject(500, "benchmark broker failed; inspect its private journal")

            def do_GET(self):
                self.reject(403, "egress endpoint is not permitted")

            do_CONNECT = do_GET
            do_PUT = do_GET
            do_DELETE = do_GET
            do_PATCH = do_GET

        if self.unix_socket:

            class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
                daemon_threads = True

            self.server = UnixServer(str(self.unix_socket), Handler)
            self.port = 38467
        else:
            self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.server.daemon_threads = True
            self.port = self.server.server_port
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def metered_tokens(self):
        with self.lock:
            return sum_usage(self.usage_meters, self.model_calls)

    def __exit__(self, *args):
        self.closed = True
        with self.lock:
            for connection in tuple(self.connections):
                connection.close()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)
        if self.unix_socket:
            self.unix_socket.unlink(missing_ok=True)
