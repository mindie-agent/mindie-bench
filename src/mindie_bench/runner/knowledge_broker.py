"""Parent-owned real MCP stdio processes, outside the foreground file grants.

[Role]
Owns one `knowledge` MCP stdio child per profile (`KnowledgeSession`) and the
parent-side broker (`KnowledgeBroker`) that routes JSON-RPC frames and records
call intervals for scoring.

[Invariants]
- Each request id is answered exactly once; frames are capped at `MAX_FRAME`.
  A child that exits or breaks framing is reported as failed, never as an empty
  successful result, and a failed call is never retried automatically.
- A shutdown error stays attached to the primary error instead of replacing it.

[Timeouts & Locks]
- `self.lock` (`RLock`) guards only the pending-request map and buffers.
  `exchange()` waits for one complete frame; `close()` drains with bounded
  waits before killing the owned process.

[Side-Effect Policy]
- Spawns a private stdio-only child under the profile's private root and writes
  `protocol.jsonl`/`stderr` there. No network access and no workspace mutation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from mindie_bench.runner.sandbox import native_command

MAX_FRAME = 1024 * 1024
MAX_SESSION_BYTES = 32 * MAX_FRAME
TOOLS = {"knowledge_query", "knowledge_explain", "query", "explain"}
METHODS = {
    "initialize",
    "ping",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/templates/list",
    "prompts/list",
    "notifications/initialized",
    "notifications/cancelled",
}


class KnowledgeInputError(ValueError):
    """Rejected caller input, before any bytes from that exchange reach the backend."""


class KnowledgeSession:
    def __init__(self, identity, name, spec, profile, read_paths, record):
        self.identity, self.name, self.record = identity, name, record
        self.root = profile.private_root / "mcp" / identity
        self.root.mkdir(parents=True)
        self.lock = threading.RLock()
        self.pending = b""
        self.input_buffer = b""
        self.output_buffer = b""
        self.local_output = b""
        self.requests = {}
        self.eof = False
        self.closed = False
        self.output_bytes = 0
        self.process = None
        self.owner_fd = None
        self.raw = (self.root / "protocol.jsonl").open("xb")
        self.errors = (self.root / "stderr").open("xb")
        env = dict(profile.env, **profile.private_env, **spec["env"])
        for key in (
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
            "XDG_STATE_HOME",
            "CODEX_HOME",
            "CLAUDE_CONFIG_DIR",
            "KIMI_HOME",
            "TMPDIR",
        ):
            path = self.root / key.lower()
            path.mkdir()
            env[key] = str(path)
        read_fd = None
        try:
            command = native_command(
                [spec["command"], *spec["args"]],
                read=[*read_paths, Path(spec["command"]).absolute()],
                write=[profile.private_root],
                cwd=profile.private_root,
                policy_path=self.root / "backend.sb",
            )
            read_fd, self.owner_fd = os.pipe()
            self.process = subprocess.Popen(
                [sys.executable, "-I", "-m", "mindie_bench.runner.owner", str(read_fd), *command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=profile.private_root,
                env=env,
                start_new_session=True,
                pass_fds=(read_fd,),
                bufsize=0,
            )
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                os.set_blocking(stream.fileno(), False)
        except BaseException:
            if self.owner_fd is not None:
                os.close(self.owner_fd)
                self.owner_fd = None
            self.raw.close()
            self.errors.close()
            raise
        finally:
            if read_fd is not None:
                os.close(read_fd)
        try:
            self.record("knowledge_session_open", server=name, session_id=identity, pid=self.process.pid)
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup:
                exc.add_note(f"knowledge startup cleanup also failed: {cleanup}")
            raise

    def _raw(self, direction, frame):
        self.raw.write(json.dumps({"direction": direction, "frame": frame}).encode() + b"\n")
        self.raw.flush()

    def _submit(self, data, eof):
        if self.eof and (data or eof):
            raise KnowledgeInputError("invalid input: knowledge stdin is already closed")
        content = self.input_buffer + data
        if len(content) > MAX_FRAME:
            raise KnowledgeInputError("invalid input: knowledge MCP frame exceeds bound")
        lines = content.split(b"\n")
        remainder = lines.pop()
        if eof and remainder:
            raise KnowledgeInputError(
                "invalid input: knowledge MCP stdin ended with an incomplete frame"
            )
        validated = []
        identities = set(self.requests)
        for line in lines:
            if not line:
                continue
            try:
                frame = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise KnowledgeInputError("invalid knowledge MCP JSON-lines request") from exc
            if (
                not isinstance(frame, dict)
                or frame.get("jsonrpc") != "2.0"
                or not isinstance(frame.get("method"), str)
            ):
                raise KnowledgeInputError("invalid knowledge MCP request")
            params = frame.get("params", {})
            if not isinstance(params, dict):
                raise KnowledgeInputError("invalid knowledge MCP params: expected an object")
            identity = frame.get("id")
            if identity is not None and (type(identity) not in {int, str} or identity in identities):
                raise KnowledgeInputError("invalid or duplicate knowledge request ID")
            if identity is not None:
                identities.add(identity)
            error = (
                (-32601, "knowledge MCP method is not permitted")
                if frame["method"] not in METHODS
                else (-32602, "knowledge tool is not permitted")
                if frame["method"] == "tools/call" and params.get("name") not in TOOLS
                else None
            )
            validated.append((line, frame, error))
        incoming = b"".join(line + b"\n" for line, _, error in validated if error is None)
        if len(self.pending) + len(incoming) > MAX_FRAME:
            raise KnowledgeInputError("invalid input: knowledge MCP backlog exceeds bound")
        for line, frame, error in validated:
            identity = frame.get("id")
            if error is not None:
                # A scoped protocol denial is a caller-visible response, not a
                # failed backend process. Never admit the rejected operation.
                request_hash = hashlib.sha256(line).hexdigest()
                self._raw("rejected_request", frame)
                self.record(
                    "knowledge_request_rejected",
                    server=self.name,
                    session_id=self.identity,
                    request_id=identity,
                    method=frame["method"],
                    tool=frame.get("params", {}).get("name"),
                    request_sha256=request_hash,
                    error_code=error[0],
                    backend_admitted=False,
                )
                if identity is not None:
                    response = {
                        "jsonrpc": "2.0",
                        "id": identity,
                        "error": {
                            "code": error[0],
                            "message": error[1],
                            "data": {
                                "method": frame["method"][:256],
                                "request_sha256": request_hash,
                            },
                        },
                    }
                    encoded = json.dumps(response).encode() + b"\n"
                    self.output_bytes += len(encoded)
                    if self.output_bytes > MAX_SESSION_BYTES:
                        raise ValueError("knowledge MCP session output exceeds bound")
                    self._raw("local_response", response)
                    self.local_output += encoded
                continue
            if identity is not None:
                params = frame.get("params", {})
                request = {
                    "method": frame["method"],
                    "tool": params.get("name") if frame["method"] == "tools/call" else None,
                    "started": time.monotonic(),
                }
                self.requests[identity] = request
                if request["tool"]:
                    self.record(
                        "knowledge_tool_start",
                        server=self.name,
                        session_id=self.identity,
                        request_id=identity,
                        tool=request["tool"],
                        request_sha256=hashlib.sha256(
                            json.dumps(frame, sort_keys=True).encode()
                        ).hexdigest(),
                    )
            self._raw("request", frame)
        self.pending += incoming
        self.input_buffer, self.eof = remainder, self.eof or eof

    def _observe(self, data):
        self.output_buffer += data
        if len(self.output_buffer) > MAX_FRAME:
            raise ValueError("knowledge MCP output frame exceeds bound")
        output = bytearray()
        while b"\n" in self.output_buffer:
            line, self.output_buffer = self.output_buffer.split(b"\n", 1)
            if not line:
                continue
            frame = json.loads(line)
            self._raw("response", frame)
            if not isinstance(frame, dict) or frame.get("jsonrpc") != "2.0":
                raise ValueError("knowledge server returned an invalid MCP frame")
            identity = frame.get("id")
            request = self.requests.pop(identity, None)
            method = request.get("method") if request else None
            result = frame.get("result")
            projected = frame
            catalog, selected = None, None
            if isinstance(result, dict) and method == "initialize" and "capabilities" in result:
                # This proxy exposes knowledge tools, plus read-only catalogs;
                # it does not grant resource reads or prompt execution.
                catalog = result["capabilities"]
                if not isinstance(catalog, dict):
                    raise ValueError("knowledge server returned invalid capabilities")
                selected = {key: value for key, value in catalog.items() if key == "tools"}
                projected = dict(frame, result=dict(result, capabilities=selected))
            elif isinstance(result, dict) and method == "tools/list":
                catalog = result.get("tools")
                if not isinstance(catalog, list) or any(
                    not isinstance(tool, dict) or not isinstance(tool.get("name"), str)
                    for tool in catalog
                ):
                    raise ValueError("knowledge server returned an invalid tool catalog")
                selected = [tool for tool in catalog if tool["name"] in TOOLS]
                projected = dict(frame, result=dict(result, tools=selected))
            delivered = json.dumps(projected).encode() if projected != frame else line
            if catalog is not None:
                self.record(
                    "knowledge_catalog_projection",
                    server=self.name,
                    session_id=self.identity,
                    request_id=identity,
                    method=method,
                    original_sha256=hashlib.sha256(line).hexdigest(),
                    projected_sha256=hashlib.sha256(delivered).hexdigest(),
                    original_catalog_sha256=hashlib.sha256(
                        json.dumps(catalog, sort_keys=True).encode()
                    ).hexdigest(),
                    projected_catalog_sha256=hashlib.sha256(
                        json.dumps(selected, sort_keys=True).encode()
                    ).hexdigest(),
                    changed=projected != frame,
                )
                if projected != frame:
                    self._raw("projected_response", projected)
            if request and request["tool"]:
                self.record(
                    "knowledge_tool_end",
                    server=self.name,
                    session_id=self.identity,
                    request_id=identity,
                    tool=request["tool"],
                    elapsed_sec=time.monotonic() - request["started"],
                    response_sha256=hashlib.sha256(line).hexdigest(),
                    is_error=(
                        "error" in frame or isinstance(result, dict) and result.get("isError") is True
                    ),
                )
            output.extend(delivered + b"\n")
        return bytes(output)

    def exchange(self, data=b"", eof=False):
        with self.lock:
            if self.closed:
                raise ValueError("knowledge session is closed")
            self._submit(data, eof)
            output = bytearray(self.local_output)
            self.local_output = b""
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                if output and not self.pending:
                    break
                if self.eof and not self.pending and not self.process.stdin.closed:
                    self.process.stdin.close()
                readers = [p for p in (self.process.stdout, self.process.stderr) if not p.closed]
                writers = [self.process.stdin] if self.pending else []
                readable, writable, _ = select.select(
                    readers, writers, [], max(0, deadline - time.monotonic())
                )
                if writable:
                    try:
                        count = os.write(self.process.stdin.fileno(), self.pending[:65536])
                        self.pending = self.pending[count:]
                    except BlockingIOError:
                        pass
                for stream in readable:
                    chunk = os.read(stream.fileno(), 65536)
                    if not chunk:
                        stream.close()
                        continue
                    self.output_bytes += len(chunk)
                    if self.output_bytes > MAX_SESSION_BYTES:
                        raise ValueError("knowledge MCP session output exceeds bound")
                    if stream is self.process.stdout:
                        output.extend(self._observe(chunk))
                    else:
                        self.errors.write(chunk)
                        self.errors.flush()
                if output or self.process.poll() is not None:
                    break
            code = (
                self.process.poll()
                if self.process.stdout.closed and self.process.stderr.closed
                else None
            )
            if code is not None and self.output_buffer:
                raise ValueError("knowledge server exited with an incomplete MCP frame")
            return {
                "stdout_b64": base64.b64encode(output).decode(),
                "exit_code": code,
                "pending_requests": len(self.requests),
            }

    def close(self):
        with self.lock:
            if self.closed:
                return
            primary = None

            def drain(*, eof=False):
                result = self.exchange(eof=eof)
                data = base64.b64decode(result["stdout_b64"], validate=True)
                if data:
                    self.record(
                        "knowledge_delivery_failed",
                        server=self.name,
                        session_id=self.identity,
                        stage="parent_cleanup_drain",
                        external_effects="completed",
                        stdout_bytes=len(data),
                        stdout_sha256=hashlib.sha256(data).hexdigest(),
                    )

            try:
                if not self.eof:
                    drain(eof=True)
                deadline = time.monotonic() + 5
                while (
                    self.process.poll() is None
                    or not self.process.stdout.closed
                    or not self.process.stderr.closed
                ) and time.monotonic() < deadline:
                    drain()
                if (
                    self.process.poll() is None
                    or not self.process.stdout.closed
                    or not self.process.stderr.closed
                ):
                    raise TimeoutError("knowledge MCP did not stop after stdin EOF")
                if self.process.returncode != 0:
                    raise RuntimeError(
                        f"knowledge MCP exited with code {self.process.returncode}; inspect parent stderr"
                    )
                if self.requests:
                    raise RuntimeError("knowledge MCP exited with unanswered requests")
            except BaseException as exc:
                primary = exc
            finally:
                self.closed = True
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    if primary is None:
                        primary = exc
                    else:
                        primary.add_note(f"knowledge group cleanup also failed: {exc}")
                if self.owner_fd is not None:
                    os.close(self.owner_fd)
                    self.owner_fd = None
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    if primary is None:
                        primary = exc
                    else:
                        primary.add_note(f"knowledge process wait also failed: {exc}")
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    stream.close()
                self.raw.close()
                self.errors.close()
            try:
                self.record(
                    "knowledge_session_close",
                    server=self.name,
                    session_id=self.identity,
                    returncode=self.process.returncode,
                    pending_requests=len(self.requests),
                    error=type(primary).__name__ if primary else None,
                    outcome="unknown" if self.requests else "failed" if primary else "completed",
                )
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    primary.add_note(f"knowledge cleanup receipt also failed: {exc}")
            if primary:
                raise primary


class ParentKnowledgeBroker:
    def __init__(self, profile, read_paths, record):
        self.profile, self.read_paths, self.record = profile, read_paths, record
        self.sessions = {}
        self.lock = threading.RLock()
        self.sealed = False

    def request(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
            raise KnowledgeInputError("invalid knowledge relay request: expected an action object")
        with self.lock:
            if self.sealed:
                raise KnowledgeInputError("invalid request: knowledge admission is sealed")
            action = payload["action"]
            if action == "open":
                name = payload.get("server")
                if not isinstance(name, str) or name not in self.profile.knowledge_servers:
                    raise KnowledgeInputError("invalid parent knowledge server")
                if len(self.sessions) >= 8:
                    raise KnowledgeInputError("invalid admission: knowledge session count exceeds bound")
                identity = uuid.uuid4().hex
                session = KnowledgeSession(
                    identity,
                    name,
                    self.profile.knowledge_servers[name],
                    self.profile,
                    self.read_paths,
                    self.record,
                )
                self.sessions[identity] = session
                return {"session_id": identity}
            if action not in {"exchange", "close"}:
                raise KnowledgeInputError("invalid knowledge stdio action")
            identity = payload.get("session_id")
            if not isinstance(identity, str) or identity not in self.sessions:
                raise KnowledgeInputError("invalid knowledge session ID")
            session = self.sessions[identity]
        if action == "exchange":
            encoded = payload.get("stdin_b64", "")
            if not isinstance(encoded, str):
                raise KnowledgeInputError("invalid knowledge stdio encoding")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, UnicodeEncodeError) as exc:
                raise KnowledgeInputError("invalid knowledge stdio encoding") from exc
            eof = payload.get("eof", False)
            if type(eof) is not bool or len(data) > MAX_FRAME:
                raise KnowledgeInputError("invalid knowledge stdio input")
            return session.exchange(data, eof)
        if action == "close":
            session.close()
            return {"closed": True}
        raise AssertionError("validated knowledge relay action was not dispatched")

    def close(self):
        with self.lock:
            self.sealed = True
            sessions = list(self.sessions.values())
        errors = []
        for session in sessions:
            try:
                session.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"another knowledge session cleanup failed: {error}")
            raise primary
