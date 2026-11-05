"""TLS observation and model-proxy request handler for the metering gateway.

[Role]
Reads observed TLS/peer metadata and serves the deny-by-default model proxy
request handler used by `Gateway.__enter__`.
[Invariants]
- No automatic model retries; duplicate paid requests are rejected by the owning
  gateway, and a lost response stream marks delivery unknown without resending.
- TLS/peer fields are observations; unavailable values stay null and the socket
  peer is never described as the provider edge.
[Timeouts & Locks]
- All state locks belong to the owning `Gateway`; this module acquires none of
  its own. Model settlement grace defaults to `30.0s`.
[Side-Effect Policy]
- Writes `broker.jsonl` events and one-shot `remote-results/<id>.json` through
  the owning gateway; never writes provider bodies or credentials.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import http.server
import json
import os
import ssl
import time

from mindie_bench.io import write_json
from mindie_bench.models import UnknownExternalEffects
from mindie_bench.runner.provider_transport import provider_connection
from mindie_bench.runner.usage import UsageMeter
from mindie_bench.scorer.command_observations import is_recognized_npu_command

_SETTLEMENT_GRACE_SECONDS = float(os.environ.get("MINDIE_BENCH_SETTLEMENT_GRACE_SECONDS", "30.0"))
MODEL_SETTLEMENT_GRACE_SEC = _SETTLEMENT_GRACE_SECONDS
_TRANSPORT_STAGES = ("connect", "submit", "response_headers", "response_body")


class RemoteBudgetExhausted(ValueError):
    """A remote action exceeded its declared budget; no retry is admitted."""

def _tls_transport_observation(transport) -> dict:
    """Read TLS identity and the immediate socket peer; null means unavailable.

    `peer_ip`/`peer_port` come from getpeername(), so they identify the next TCP hop
    (possibly a local proxy or TUN fake-IP), not the provider edge.
    """
    observation = dict(
        tls_version=None,
        tls_cipher=None,
        tls_alpn=None,
        peer_ip=None,
        peer_port=None,
        peer_scope=None,
    )
    try:
        observation["tls_version"] = transport.version()
        cipher = transport.cipher()
        observation["tls_cipher"] = cipher[0] if cipher else None
        observation["tls_alpn"] = transport.selected_alpn_protocol()
    except TimeoutError:
        raise
    except (AttributeError, OSError, ValueError):
        pass
    try:
        peer = transport.getpeername()
        if isinstance(peer, tuple) and len(peer) >= 2:
            observation["peer_ip"] = str(peer[0])
            observation["peer_port"] = int(peer[1])
            observation["peer_scope"] = "immediate_socket_peer"
    except TimeoutError:
        raise
    except (AttributeError, OSError, ValueError):
        pass
    return observation




def model_proxy_handler(gateway):
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
            except TimeoutError:
                raise
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
                    except TimeoutError:
                        raise
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
                    "/v1/chat/completions",
                    "/chat/completions",
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
                connection, target = provider_connection(
                    gateway.harness.provider_url, self.path
                )
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
                # A concurrent settlement close must never make request()
                # silently reconnect and submit after the deadline.
                connection.auto_open = 0
                with gateway.lock:
                    gateway.connections[connection] = None
                headers = {
                    "Content-Type": "application/json",
                    "Accept": self.headers.get("Accept", "*/*"),
                }
                if "/messages" not in self.path:
                    headers["Authorization"] = "Bearer " + gateway.api_key
                else:
                    headers["x-api-key"] = gateway.api_key
                    headers["anthropic-version"] = self.headers.get(
                        "anthropic-version", "2023-06-01"
                    )
                    if self.headers.get("anthropic-beta"):
                        headers["anthropic-beta"] = self.headers["anthropic-beta"]
                transport_stage = "connect"
                stage_started, stage_durations = {}, dict.fromkeys(_TRANSPORT_STAGES)
                stage_started["connect"] = time.monotonic()
                # Count only complete socket writes. A failed sendall leaves its
                # chunk's delivery unknown; never report that as "not sent".
                wire = {"written": 0, "pending": 0, "complete": False}
                tls_observation = dict.fromkeys(
                    ("tls_version", "tls_cipher", "tls_alpn", "peer_ip", "peer_port", "peer_scope")
                )
                try:
                    connection.connect()
                    transport = connection.sock
                    tls_observation = _tls_transport_observation(transport)
                    stage_durations["connect"] = time.monotonic() - stage_started["connect"]
                    with gateway.lock:
                        gateway.connections[connection] = transport
                        if gateway.closed or gateway.model_settlement_expired:
                            raise TimeoutError("model settlement deadline expired before submission")
                    transport_stage = "submit"
                    stage_started["submit"] = time.monotonic()
                    unmodified_send = getattr(connection, "send", None)
                    if callable(unmodified_send):

                        def counting_send(data):
                            wire["pending"] = len(data)
                            unmodified_send(data)
                            wire["written"] += len(data)
                            wire["pending"] = 0

                        connection.send = counting_send
                    connection.request("POST", target, body=body, headers=headers)
                    wire["complete"] = True
                    stage_durations["submit"] = time.monotonic() - stage_started["submit"]
                    transport_stage = "response_headers"
                    stage_started["response_headers"] = time.monotonic()
                    response = connection.getresponse()
                    stage_durations["response_headers"] = (
                        time.monotonic() - stage_started["response_headers"]
                    )
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
                    except (TimeoutError, OSError) as exc:
                        delivery_failed(exc)
                    transport_stage = "response_body"
                    stage_started["response_body"] = time.monotonic()
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
                            except (TimeoutError, OSError) as exc:
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
                except (TimeoutError, OSError, http.client.HTTPException) as exc:
                    # Explicit timeout handling: a lost provider receipt means
                    # the paid request's external effect is unknown, never a
                    # silent success or retry.
                    gateway.unknown = True
                    stage_durations[transport_stage] = (
                        time.monotonic() - stage_started[transport_stage]
                        if transport_stage in stage_started
                        else None
                    )
                    ssl_reason = (
                        exc.reason or exc.strerror if isinstance(exc, ssl.SSLError) else None
                    )
                    alert_reason = (
                        ssl_reason if ssl_reason and "ALERT" in ssl_reason.upper() else None
                    )
                    # The reason string says an alert was seen, but not which direction
                    # carried it on the wire. Keep the measured/inferred split explicit.
                    alert_direction = "inferred_received_from_peer" if alert_reason else None
                    gateway.record(
                        "model_error",
                        ordinal=ordinal,
                        error=type(exc).__name__,
                        transport_stage=transport_stage,
                        errno=exc.errno if isinstance(exc, OSError) else None,
                        ssl_reason=ssl_reason,
                        transport_stage_sec=stage_durations[transport_stage],
                        transport_stage_durations=stage_durations,
                        request_bytes_written=wire["written"],
                        request_write_complete=wire["complete"],
                        request_write_uncertain_bytes=wire["pending"],
                        alert_reason=alert_reason,
                        alert_direction=alert_direction,
                        alert_direction_basis=(
                            "openssl_reason_contains_ALERT" if alert_reason else None
                        ),
                        alert_direction_measured=(False if alert_reason else None),
                        **tls_observation,
                    )
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
            except TimeoutError:
                raise
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

    return Handler
