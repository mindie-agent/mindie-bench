"""Parent-side model/replay broker with deny-by-default egress and no retries."""

from __future__ import annotations

import base64
import hashlib
import http.client
import http.server
import ipaddress
import json
import os
import re
import socket
import socketserver
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from urllib.parse import urlsplit

from mindie_bench.io import read_json, safe_relative, write_json
from mindie_bench.models import HarnessConfig, TaskSpec, UnknownExternalEffects
from mindie_bench.runner.usage import UsageMeter, sum_usage
from mindie_bench.scorer.command_observations import is_recognized_npu_command

MODEL_SETTLEMENT_GRACE_SEC = 10.0


class RemoteBudgetExhausted(ValueError):
    pass


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
        self.model_idle = threading.Condition(self.lock)
        self.remote_idle = threading.Condition(self.lock)
        self.models_sealed = False
        self.model_settlement_expired = False
        self.remotes_sealed = False
        self.remote_settlement_expired = False
        self.remote_active = 0
        self.remote_unknown = False
        self.remote_failures: list[dict] = []
        self.model_calls = self.replay_calls = 0
        self.stop_reason: str | None = None
        self.unknown = False
        self.infrastructure_error: str | None = None
        self.seen: set[str] = set()
        self.active = 0
        self.closed = False
        self.connections = {}
        self.usage_meters = {}
        self.model_delivery_failures: list[dict] = []
        self.unix_socket = unix_socket
        self.remote = remote
        self.remote_test_calls = 0
        self.stream_input: dict[str, bytes] = {}
        self.remote_result_count = 0
        self.remote_delivery_failures: list[dict] = []
        self.api_key = os.environ.get(harness.api_key_env or "", "")
        if harness.name != "fixture" and not self.api_key:
            raise ValueError(f"missing explicitly selected credential: {harness.api_key_env}")
        if harness.provider_url:
            parsed = urlsplit(harness.provider_url)
            try:
                loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
            except ValueError:
                loopback = False
            if (
                not (parsed.scheme == "https" or parsed.scheme == "http" and loopback)
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "model provider_url requires HTTPS, or numeric loopback HTTP for a local relay"
                )
        self.server = None
        self.thread = None
        self.knowledge = None

    def attach_knowledge(self, profile, read_paths):
        """Keep the configured real knowledge runtime in the parent capability domain."""
        from mindie_bench.runner.knowledge_broker import ParentKnowledgeBroker

        if self.knowledge is not None:
            raise ValueError("knowledge broker is already attached")
        if profile.knowledge_servers:
            self.knowledge = ParentKnowledgeBroker(profile, read_paths, self.record)

    def record(self, kind: str, **values):
        with self.lock, self.journal.open("a", encoding="utf-8") as out:
            if kind in {"remote_unknown", "remote_calls_unsettled"} or (
                kind == "remote_delivery_failed" and values.get("external_effects") == "unknown"
            ):
                self.remote_unknown = True
                self.remote_failures.append(dict(kind=kind, **values))
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

    def ssh_stream(self, request: dict) -> dict:
        """Bridge remote-dev's existing NDJSON RPC without waiting for stdin EOF."""
        action = request["action"]
        if action == "open":
            command = request["command"]
            if not isinstance(command, str) or not command or len(command.encode()) > 8 * 1024 * 1024:
                raise ValueError("invalid remote stream command")
            test_run = is_recognized_npu_command(command)
            with self.lock:
                if test_run and self.remote_test_calls >= self.task.budget.max_npu_runs:
                    self.stop_reason = "max_npu_runs"
                    raise RemoteBudgetExhausted("benchmark NPU test budget exhausted")
                self.remote_test_calls += int(test_run)
            try:
                result = self.remote.open_stream(command)
            except Exception as exc:
                self.unknown = True
                self.record("remote_unknown", action=action, error=type(exc).__name__, message=str(exc))
                raise UnknownExternalEffects("remote stream creation has unknown effects") from exc
            identity = result["stream_id"]
            if not isinstance(identity, str) or identity in self.stream_input:
                self.unknown = True
                raise UnknownExternalEffects("invalid remote stream creation receipt")
            self.stream_input[identity] = b""
            self.record("remote_stream_open", stream_id=identity, test_run=test_run)
            return result
        if action != "exchange":
            raise ValueError("unsupported remote stream action")
        identity, eof = request["stream_id"], request.get("eof", False)
        data = base64.b64decode(request.get("stdin_b64", ""), validate=True)
        if type(eof) is not bool or len(data) > 8 * 1024 * 1024:
            raise ValueError("invalid remote stream input")
        with self.lock:
            pending = self.stream_input[identity] + data
            if len(pending) > 8 * 1024 * 1024:
                raise ValueError("remote RPC frame exceeds evidence bound")
            lines = pending.split(b"\n")
            remainder = lines.pop()
            if eof and remainder:
                lines.append(remainder)
                remainder = b""
            # Count commands in RPC payloads, never the worker's uploaded source.
            # Buffer partial frames so admission happens before any RPC submission.
            tests = 0
            for line in lines:
                message = json.loads(line)
                payload = message.get("payload", {})
                if not isinstance(payload, dict):
                    raise ValueError("invalid remote RPC payload")
                spec = payload.get("spec", {})
                if not isinstance(spec, dict):
                    raise ValueError("invalid remote job specification")
                command = payload.get("command", payload.get("cmd", spec.get("command", "")))
                tests += int(isinstance(command, str) and is_recognized_npu_command(command))
            if self.remote_test_calls + tests > self.task.budget.max_npu_runs:
                self.stop_reason = "max_npu_runs"
                raise RemoteBudgetExhausted("benchmark NPU test budget exhausted")
            self.remote_test_calls += tests
            self.stream_input[identity] = remainder
        submitted = b"".join(line + b"\n" for line in lines)
        try:
            result = self.remote.exchange_stream(identity, base64.b64encode(submitted).decode(), eof=eof)
        except Exception as exc:
            self.unknown = True
            self.record("remote_unknown", action=action, error=type(exc).__name__, message=str(exc))
            raise UnknownExternalEffects("remote stream exchange has unknown effects") from exc
        if submitted or eof or result.get("exit_code") is not None:
            self.record(
                "remote_stream_exchange",
                stream_id=identity,
                submitted_bytes=len(submitted),
                submitted_sha256=hashlib.sha256(submitted).hexdigest(),
                test_runs=tests,
                eof=eof,
                exit_code=result.get("exit_code"),
            )
        return result

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

            def remote_response(
                self, result, *, request_body, uncertain_on_delivery, persist_result=True
            ):
                identity = None
                if persist_result:
                    with gateway.lock:
                        gateway.remote_result_count += 1
                        identity = f"{gateway.remote_result_count:08d}"
                        directory = gateway.journal.parent / "remote-results"
                        directory.mkdir(exist_ok=True)
                        write_json(
                            directory / f"{identity}.json",
                            dict(
                                request_sha256=hashlib.sha256(request_body).hexdigest(),
                                result=result,
                            ),
                            exclusive=True,
                        )
                payload = json.dumps(result).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    self.wfile.flush()
                except OSError as exc:
                    # A completed one-shot result is already authoritative and
                    # durable. A lost RPC stream frame can still hide a submission.
                    gateway.unknown = gateway.unknown or uncertain_on_delivery
                    failure = dict(
                        receipt_id=identity,
                        error=type(exc).__name__,
                        external_effects="unknown" if uncertain_on_delivery else "completed",
                    )
                    gateway.remote_delivery_failures.append(failure)
                    gateway.record("remote_delivery_failed", **failure)
                    self.close_connection = True

            def do_POST(self):
                remote_admitted = False
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 32 * 1024 * 1024:
                        return self.reject(413, "invalid request size")
                    body = self.rfile.read(length)
                    if len(body) != length:
                        return self.reject(400, "incomplete request")
                    if self.path == "/bench/knowledge-stdio":
                        from mindie_bench.runner.knowledge_broker import KnowledgeInputError

                        if gateway.knowledge is None:
                            return self.reject(403, "no parent knowledge component is assigned")
                        if gateway.closed:
                            return self.reject(409, "knowledge admission is closed")
                        try:
                            try:
                                request = json.loads(body)
                            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                                raise KnowledgeInputError(
                                    "invalid knowledge relay JSON request"
                                ) from exc
                            result = gateway.knowledge.request(request)
                        except KnowledgeInputError as exc:
                            gateway.record(
                                "knowledge_client_rejected",
                                error=type(exc).__name__,
                                message=str(exc),
                                backend_admitted=False,
                            )
                            return self.reject(400, str(exc))
                        except Exception as exc:
                            gateway.infrastructure_error = "required parent knowledge component failed"
                            gateway.record(
                                "knowledge_error",
                                error=type(exc).__name__,
                                message=str(exc),
                            )
                            return self.reject(
                                500, "required knowledge component failed; inspect parent journal"
                            )
                        payload = json.dumps(result).encode()
                        try:
                            self.send_response(200)
                            self.send_header("Content-Length", str(len(payload)))
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(payload)
                            self.wfile.flush()
                        except OSError as exc:
                            pure_poll = (
                                request.get("action") == "exchange"
                                and not request.get("stdin_b64")
                                and not request.get("eof")
                                and not result.get("stdout_b64")
                                and result.get("exit_code") is None
                            )
                            if not pure_poll:
                                gateway.infrastructure_error = "knowledge MCP result delivery failed"
                            gateway.record(
                                "knowledge_delivery_failed",
                                error=type(exc).__name__,
                                external_effects="none" if pure_poll else "completed",
                            )
                            self.close_connection = True
                        return
                    if self.path in {"/bench/ssh", "/bench/ssh-stream"}:
                        with gateway.remote_idle:
                            if gateway.remotes_sealed or gateway.closed:
                                return self.reject(409, "remote admission is sealed; no new operation")
                            gateway.remote_active += 1
                            remote_admitted = True
                    if self.path == "/bench/replay":
                        payload = json.dumps(gateway.replay(json.loads(body))).encode()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    if self.path == "/bench/ssh-stream":
                        if gateway.remote is None:
                            return self.reject(403, "no remote slot is assigned")
                        if gateway.unknown or gateway.closed or gateway.infrastructure_error:
                            return self.reject(409, "prior remote effects require reconciliation")
                        try:
                            request = json.loads(body)
                            result = gateway.ssh_stream(request)
                        except RemoteBudgetExhausted as exc:
                            return self.reject(429, str(exc))
                        except UnknownExternalEffects:
                            return self.reject(502, "remote stream has unknown effects; no retry")
                        pure_poll = (
                            request["action"] == "exchange"
                            and not request.get("stdin_b64")
                            and not request.get("eof")
                            and not result.get("stdout_b64")
                            and not result.get("stderr_b64")
                            and result.get("exit_code") is None
                        )
                        return self.remote_response(
                            result,
                            request_body=body,
                            uncertain_on_delivery=not pure_poll,
                            persist_result=not pure_poll,
                        )
                    if self.path == "/bench/ssh":
                        if gateway.remote is None:
                            return self.reject(403, "no remote slot is assigned")
                        import base64

                        request = json.loads(body)
                        command, stdin_b64 = request["command"], request["stdin_b64"]
                        stdin = base64.b64decode(stdin_b64, validate=True)
                        if not isinstance(command, str) or len(stdin) > 8 * 1024 * 1024:
                            return self.reject(400, "invalid remote command")
                        # Stdin is opaque input, including patch and file bodies.
                        test_run = is_recognized_npu_command(command)
                        with gateway.lock:
                            if gateway.unknown or gateway.closed or gateway.infrastructure_error:
                                return self.reject(
                                    409,
                                    "a prior operation has unknown effects; remote writes are stopped",
                                )
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
                        return self.remote_response(
                            result, request_body=body, uncertain_on_delivery=False
                        )
                    if self.path not in {
                        "/v1/responses",
                        "/responses",
                        "/v1/messages",
                        "/messages",
                        "/v1/messages?beta=true",
                        "/messages?beta=true",
                    }:
                        return self.reject(403, "egress endpoint is not permitted")
                    if gateway.harness.name == "fixture":
                        return self.reject(403, "fixture trials cannot call a paid model")
                    if json.loads(body).get("model") != gateway.harness.model:
                        gateway.infrastructure_error = "request attempted to change the pinned model"
                        return self.reject(403, gateway.infrastructure_error)
                    request_hash = hashlib.sha256(body).hexdigest()
                    with gateway.lock:
                        if gateway.closed or gateway.unknown or gateway.models_sealed:
                            return self.reject(409, "prior request is unknown; no retry")
                        if request_hash in gateway.seen:
                            return self.reject(409, "duplicate paid request; no replay")
                        if gateway.model_calls >= gateway.task.budget.max_turns:
                            gateway.stop_reason = "max_turns"
                            gateway.record(
                                "model_admission_rejected",
                                reason="max_turns",
                                http_status=429,
                                request_sha256=request_hash,
                                admitted_model_calls=gateway.model_calls,
                                limit=gateway.task.budget.max_turns,
                            )
                            return self.reject(429, "benchmark turn budget exhausted")
                        gateway.seen.add(request_hash)
                        gateway.model_calls += 1
                        gateway.active += 1
                        ordinal = gateway.model_calls
                        gateway.usage_meters[ordinal] = UsageMeter()
                    gateway.record("model_start", ordinal=ordinal, request_sha256=request_hash)
                    endpoint = urlsplit(gateway.harness.provider_url)
                    connection_type = (
                        http.client.HTTPSConnection
                        if endpoint.scheme == "https"
                        else http.client.HTTPConnection
                    )
                    connection = connection_type(endpoint.hostname, endpoint.port, timeout=120)
                    # A concurrent settlement close must never make request()
                    # silently reconnect and submit after the deadline.
                    connection.auto_open = 0
                    with gateway.lock:
                        gateway.connections[connection] = None
                    target = endpoint.path.rstrip("/") + "/" + self.path.removeprefix("/v1/").lstrip("/")
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": self.headers.get("Accept", "*/*"),
                    }
                    if gateway.harness.name in {"codex", "dsh"}:
                        headers["Authorization"] = "Bearer " + gateway.api_key
                    else:
                        headers["x-api-key"] = gateway.api_key
                        headers["anthropic-version"] = self.headers.get(
                            "anthropic-version", "2023-06-01"
                        )
                        if self.headers.get("anthropic-beta"):
                            headers["anthropic-beta"] = self.headers["anthropic-beta"]
                    try:
                        connection.connect()
                        transport = connection.sock
                        with gateway.lock:
                            gateway.connections[connection] = transport
                            if gateway.closed or gateway.model_settlement_expired:
                                raise TimeoutError("model settlement deadline expired before submission")
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
                        delivery_deadline = None

                        def delivery_failed(exc):
                            nonlocal delivery_deadline
                            delivery_deadline = time.monotonic() + MODEL_SETTLEMENT_GRACE_SEC
                            failure = dict(ordinal=ordinal, error=type(exc).__name__)
                            gateway.model_delivery_failures.append(failure)
                            gateway.record("model_delivery_failed", **failure)
                            self.close_connection = True

                        try:
                            self.send_response(response.status)
                            self.send_header(
                                "Content-Type", response.getheader("Content-Type", "application/json")
                            )
                            self.end_headers()
                        except OSError as exc:
                            delivery_failed(exc)
                        while True:
                            if delivery_deadline is not None:
                                remaining = delivery_deadline - time.monotonic()
                                if remaining <= 0:
                                    raise TimeoutError("model delivery drain deadline expired")
                                transport.settimeout(remaining)
                            chunk = response.read1(65536)
                            if not chunk:
                                break
                            gateway.usage_meters[ordinal].feed(chunk)
                            if delivery_deadline is None:
                                try:
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                                except OSError as exc:
                                    # Losing the native client does not cancel an
                                    # already submitted provider request or its bill.
                                    delivery_failed(exc)
                        if response.length not in {None, 0}:
                            raise http.client.IncompleteRead(b"", response.length)
                        gateway.usage_meters[ordinal].finish()
                        if response.status < 400 and not gateway.usage_meters[ordinal].tokens.complete:
                            gateway.unknown = True
                            gateway.record(
                                "model_error", ordinal=ordinal, error="IncompleteProviderUsage"
                            )
                        gateway.record("model_end", ordinal=ordinal, http_status=response.status)
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
                            gateway.connections.pop(connection, None)
                            gateway.active -= 1
                            gateway.model_idle.notify_all()
                except (ValueError, KeyError, OSError) as exc:
                    gateway.record("broker_error", error=type(exc).__name__, message=str(exc))
                    gateway.infrastructure_error = f"{type(exc).__name__}: {exc}"
                    self.reject(500, "benchmark broker failed; inspect its private journal")
                finally:
                    if remote_admitted:
                        with gateway.remote_idle:
                            gateway.remote_active -= 1
                            gateway.remote_idle.notify_all()

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

    def _close_model_connections(self):
        with self.lock:
            connections = tuple(self.connections.items())
        for connection, transport in connections:
            # HTTPResponse can own the socket after getresponse() clears
            # HTTPConnection.sock for a Connection: close response.
            transport = transport or getattr(connection, "sock", None)
            if transport is not None:
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()
        return bool(connections)

    def finish_inference(self, timeout=MODEL_SETTLEMENT_GRACE_SEC):
        """Seal model admission before deciding the trial's terminal status."""
        with self.model_idle:
            if self.models_sealed:
                return
            self.models_sealed = True
            self.model_idle.notify_all()
            if not self.model_idle.wait_for(lambda: self.active == 0, timeout=timeout):
                self.unknown = True
                self.model_settlement_expired = True
                self.record("model_calls_unsettled", active=self.active)
            else:
                return
        if self._close_model_connections():
            with self.model_idle:
                self.model_idle.wait_for(lambda: self.active == 0, timeout=min(timeout, 0.25))

    def finish_remote(self, timeout=MODEL_SETTLEMENT_GRACE_SEC):
        """Settle admitted RPC handlers before the judge can quiesce their worker."""
        with self.remote_idle:
            if self.remotes_sealed:
                return
            self.remotes_sealed = True
            if not self.remote_idle.wait_for(lambda: self.remote_active == 0, timeout=timeout):
                self.unknown = True
                self.remote_settlement_expired = True
                self.record("remote_calls_unsettled", active=self.remote_active)

    def remote_state(self):
        with self.lock:
            return dict(
                sealed=self.remotes_sealed,
                active=self.remote_active,
                settlement_expired=self.remote_settlement_expired,
                unknown=self.remote_unknown,
                failures=list(self.remote_failures),
            )

    def __exit__(self, *args):
        failures = []

        def cleanup(stage, operation):
            try:
                operation()
            except BaseException as exc:
                failures.append(exc)
                self.infrastructure_error = "required broker settlement or cleanup failed"
                try:
                    self.record(
                        "broker_cleanup_error",
                        stage=stage,
                        error=type(exc).__name__,
                        message=str(exc),
                    )
                except BaseException as receipt_error:
                    exc.add_note(f"broker cleanup receipt also failed: {receipt_error}")

        cleanup("model_settlement", self.finish_inference)
        cleanup("remote_settlement", self.finish_remote)
        if self.knowledge is not None:
            cleanup("knowledge_cleanup", self.knowledge.close)
        self.closed = True
        cleanup("model_connections", self._close_model_connections)
        if self.server:
            cleanup("server_shutdown", self.server.shutdown)
            cleanup("server_close", self.server.server_close)
        if self.thread:
            cleanup("server_join", lambda: self.thread.join(timeout=2))
        if self.unix_socket:
            cleanup("socket_unlink", lambda: self.unix_socket.unlink(missing_ok=True))
        if failures:
            primary = failures[0]
            for failure in failures[1:]:
                primary.add_note(f"another broker cleanup failed: {type(failure).__name__}: {failure}")
            if args and args[1] is not None:
                args[1].add_note(f"broker cleanup also failed: {type(primary).__name__}: {primary}")
            else:
                raise primary
