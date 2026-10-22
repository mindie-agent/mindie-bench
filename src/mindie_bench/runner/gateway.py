# ruff: noqa: I001
"""Parent-side model, remote-SSH, and knowledge RPC broker with deny-by-default egress.

[Role]
Mediates all agent egress (LLM inference, L2 deterministic replay, L3 remote SSH/stream
commands, and parent-hosted `knowledge` stdio RPC) over a local HTTP/Unix-socket gateway,
enforcing turn/NPU-test budgets and recording an append-only `broker.jsonl` ledger.

[Invariants]
- Zero automatic retries on model or remote mutations: duplicate paid model requests (`seen` SHA-256)
  or requests after `unknown=True` are rejected with HTTP 409.
- NPU test-run budget (`max_npu_runs`) is checked and incremented under `self.lock` before any
  `/bench/ssh` or `/bench/ssh-stream` command is admitted to the remote worker; patch stdin and
  uploaded Python worker source are never miscounted as test runs.
- One-shot `/bench/ssh` results are persisted to `remote-results/<id>.json` before socket write so a
  broken client pipe records `external_effects="completed"` rather than polluting `unknown`.
- `model_error` transport metadata is measured from the live connection (stage durations, TLS
  identity, immediate socket peer, bytes handed to the socket, observed alert reason). Alert
  direction is only inferred from the OpenSSL reason, and the socket peer may be a local
  proxy/TUN fake-IP, not the provider edge. It never records bodies or credentials; `null` means
  unavailable/not observed, never inferred.

[Timeouts & Locks]
- `self.lock` (`RLock`) serializes budget accounting, journal writes, and connection tracking.
- `finish_inference()` and `finish_remote()` seal new admissions and wait up to
  `MODEL_SETTLEMENT_GRACE_SEC` (`30.0s`) for in-flight model/remote handlers to drain before
  the verifier runs.

[Side-Effect Policy]
- Writes append-only event records to `journal` (`broker.jsonl`) and one-shot remote receipts to
  `remote-results/`; owns a short private temporary directory for Unix sockets.
- `__exit__` drains model/remote channels, closes the parent `knowledge` broker,
  shuts down the HTTP/Unix server, and removes its temporary socket directory.
"""

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
import tempfile
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from urllib.parse import urlsplit

from mindie_bench.io import read_json, safe_relative
from mindie_bench.models import HarnessConfig, TaskSpec, UnknownExternalEffects
from mindie_bench.runner.gateway_model_proxy import (
    MODEL_SETTLEMENT_GRACE_SEC,
    RemoteBudgetExhausted,
    model_proxy_handler,
)
from mindie_bench.runner.usage import sum_usage
from mindie_bench.scorer.command_observations import is_recognized_npu_command

from mindie_bench.io import write_json  # noqa: F401
from mindie_bench.runner.gateway_model_proxy import _SETTLEMENT_GRACE_SECONDS  # noqa: F401
from mindie_bench.runner.gateway_model_proxy import _TRANSPORT_STAGES  # noqa: F401
from mindie_bench.runner.gateway_model_proxy import _tls_transport_observation  # noqa: F401
from mindie_bench.runner.provider_transport import provider_connection  # noqa: F401
from mindie_bench.runner.usage import UsageMeter  # noqa: F401


class Gateway(AbstractContextManager):
    def __init__(
        self,
        harness: HarnessConfig,
        task: TaskSpec,
        task_root: Path,
        workspace: Path,
        journal: Path,
        *,
        use_unix_socket: bool = False,
        remote=None,
    ):
        self.harness, self.task, self.task_root = harness, task, task_root
        self.workspace, self.journal = workspace, journal
        self.lock = threading.RLock()
        self.model_idle = threading.Condition(self.lock)
        self.remote_idle = threading.Condition(self.lock)
        self.models_sealed = self.model_settlement_expired = False
        self.remotes_sealed = self.remote_settlement_expired = False
        self.remote_active = self.model_calls = self.replay_calls = 0
        self.remote_test_calls = self.remote_result_count = self.active = 0
        self.remote_unknown = self.unknown = self.closed = False
        self.stop_reason: str | None = None
        self.infrastructure_error: str | None = None
        self.seen: set[str] = set()
        self.connections: dict = {}
        self.usage_meters: dict = {}
        self.remote_failures: list[dict] = []
        self.model_delivery_failures: list[dict] = []
        self.remote_delivery_failures: list[dict] = []
        self.stream_input: dict[str, bytes] = {}
        self.use_unix_socket, self.remote = use_unix_socket, remote
        self.unix_socket = self.socket_directory = None
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
        self.server = self.thread = self.knowledge = None

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
        from mindie_bench.runner import gateway_model_proxy

        gateway_model_proxy.MODEL_SETTLEMENT_GRACE_SEC = MODEL_SETTLEMENT_GRACE_SEC
        Handler = model_proxy_handler(self)

        if self.use_unix_socket:

            class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
                daemon_threads = True

            # sockaddr_un has a fixed length bound; report paths are unrestricted.
            self.socket_directory = tempfile.TemporaryDirectory(prefix="mb-gw-", dir="/tmp")
            self.unix_socket = Path(self.socket_directory.name) / "gateway.sock"
            try:
                self.server = UnixServer(str(self.unix_socket), Handler)
            except BaseException:
                self.socket_directory.cleanup()
                raise
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
                except TimeoutError:
                    raise
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
        if self.socket_directory:
            cleanup("socket_directory", self.socket_directory.cleanup)
        if failures:
            primary = failures[0]
            for failure in failures[1:]:
                primary.add_note(f"another broker cleanup failed: {type(failure).__name__}: {failure}")
            if args and args[1] is not None:
                args[1].add_note(f"broker cleanup also failed: {type(primary).__name__}: {primary}")
            else:
                raise primary
