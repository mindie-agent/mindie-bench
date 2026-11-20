"""Byte-zero stdout MCP stdio client (`MCPWireSession`) and Scenarios 1 & 2.

[Role]
Implements `MCPWireSession` (verifying zero non-protocol bytes on `stdout` from byte 0
across `Content-Length` and JSONL framings), Scenario 1 (`cold_native_launch`), and
Scenario 2 (`public_install_py` with fail-closed `git` `exit 99` trap).

[Invariants]
- `PipeReader` inspects `stdout` from byte 0; any stray text or non-JSON-RPC object
  (such as `{"status": "provisioned"}`) immediately raises `MCPProtocolError`.
- Never mutates `os.environ`; passes explicit `env` and `**subprocess_window_kwargs()`.
- Post-verdict session/service teardown failures append to `cleanup_warnings` only.

[Timeouts & Locks]
- Concurrent server startup honors caller monotonic `deadline` via `PipeReader`/`PipeWriter`.

[Side-Effect Policy]
- Spawns MCP child servers strictly inside caller-owned `TemporaryDirectory` sandboxes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mindie_bench.host_acceptance import (
    HostAcceptanceError,
    build_sandbox_env,
    install_git_trap,
    resolve_base_python,
    stop_sandbox_knowledge_services,
)
from mindie_bench.host_acceptance.manifests import (
    EXPECTED_HOST_SERVERS,
    HostManifestSpec,
    MCPServerSpec,
    parse_host_manifest,
    run_wb_installer,
)
from mindie_bench.runner.host_platform import (
    PipeReader,
    PipeWriter,
    is_windows_store_stub,
    subprocess_window_kwargs,
    which_native,
)

BANNED_LEGACY_TOOLS = frozenset({"remote_BashOutput", "remote_KillShell"})
FORBIDDEN_REQUIRED_KEYS = frozenset({"_mindie_bind_id", "request_nonce", "context_file", "_session_id", "mindie_activation"})
_HEADER_LINE_RE = re.compile(rb"^[A-Za-z0-9-]+:\s*[^\r\n]+$")


class MCPProtocolError(HostAcceptanceError):
    """Raised when an MCP server violates stdio framing or leaks non-RPC stdout bytes."""


class MCPWireSession:
    """Strict byte-0 MCP stdio client supporting `Content-Length` and JSONL framings."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        if "/" in spec.command or "\\" in spec.command:
            cmd_path = Path(spec.command)
            if not cmd_path.is_file() or is_windows_store_stub(cmd_path):
                raise FileNotFoundError(f"MCP command unavailable or stub: {spec.command}")
            resolved = str(cmd_path)
        else:
            found = which_native(spec.command, path=spec.env.get("PATH"))
            if not found:
                raise FileNotFoundError(f"MCP command {spec.command!r} not found on isolated PATH")
            resolved = found
        self.proc = subprocess.Popen(
            [resolved, *spec.args],
            cwd=str(spec.cwd),
            env=dict(spec.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **subprocess_window_kwargs(),
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None and self.proc.stderr is not None
        self._writer, self._reader = PipeWriter(self.proc.stdin), PipeReader(self.proc.stdout)
        self._buf, self._offset, self._next_id = bytearray(), 0, 1
        self._stderr_tail, self._stderr_lock = bytearray(), threading.Lock()
        threading.Thread(target=self._drain_stderr, args=(self.proc.stderr,), daemon=True).start()

    def _drain_stderr(self, stream: Any) -> None:
        fd = stream.fileno()
        while True:
            try:
                chunk = os.read(fd, 16384)
            except TimeoutError:
                raise
            except (OSError, ValueError):
                return
            if not chunk:
                return
            with self._stderr_lock:
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > 16384:
                    del self._stderr_tail[:-16384]

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()[-800:]

    def _fill(self, deadline: float) -> bool:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError(f"Timed out waiting for MCP stdout on {self.spec.name}; stderr={self.stderr_tail()}")
        timed_out, chunk = self._reader.read_chunk(rem)
        if timed_out:
            raise TimeoutError(f"Timed out reading MCP stdout on {self.spec.name}; stderr={self.stderr_tail()}")
        if not chunk:
            return False
        self._buf.extend(chunk)
        return True

    def read_message(self, deadline: float) -> tuple[dict[str, Any], str]:
        prefix = b"content-length:"
        while True:
            if not self._buf and not self._fill(deadline):
                raise MCPProtocolError(f"EOF before JSON-RPC reply on {self.spec.name} (rc={self.proc.poll()}): {self.stderr_tail()}")
            lead = bytes(self._buf[: len(prefix)]).lower()
            if len(self._buf) < len(prefix) and prefix.startswith(lead):
                if not self._fill(deadline):
                    raise MCPProtocolError(f"Truncated leading bytes on {self.spec.name}: {bytes(self._buf)!r}")
                continue
            if lead == prefix:
                sep_pos, sep_len = self._buf.find(b"\r\n\r\n"), 4
                if sep_pos < 0:
                    sep_pos, sep_len = self._buf.find(b"\n\n"), 2
                if sep_pos < 0:
                    if len(self._buf) > 8192 or not self._fill(deadline):
                        raise MCPProtocolError(f"Malformed/incomplete MCP header on {self.spec.name}")
                    continue
                clen: int | None = None
                for hline in bytes(self._buf[:sep_pos]).split(b"\n"):
                    clean = hline.rstrip(b"\r")
                    if not _HEADER_LINE_RE.match(clean):
                        raise MCPProtocolError(f"Malformed MCP header on {self.spec.name}: {clean!r}")
                    k, _, v = clean.partition(b":")
                    if k.strip().lower() == b"content-length":
                        clen = int(v.strip().decode("ascii"))
                if clen is None or clen < 0:
                    raise MCPProtocolError(f"Missing Content-Length on {self.spec.name}")
                total = sep_pos + sep_len + clen
                while len(self._buf) < total:
                    if not self._fill(deadline):
                        raise MCPProtocolError(f"EOF inside Content-Length body on {self.spec.name}")
                body = bytes(self._buf[sep_pos + sep_len : total])
                del self._buf[:total]
                self._offset += total
                return self._parse_rpc_object(body, "content-length")

            if self._buf[0:1] == b"{":
                while (nl := self._buf.find(b"\n")) < 0:
                    if not self._fill(deadline):
                        raise MCPProtocolError(f"Unterminated JSON line on {self.spec.name}: {bytes(self._buf[:200])!r}")
                line = bytes(self._buf[:nl]).rstrip(b"\r")
                del self._buf[: nl + 1]
                self._offset += nl + 1
                return self._parse_rpc_object(line, "jsonl")

            raise MCPProtocolError(f"Stray non-protocol stdout bytes at offset {self._offset} on {self.spec.name}: {bytes(self._buf[:200])!r}")

    def _parse_rpc_object(self, raw: bytes, framing: str) -> tuple[dict[str, Any], str]:
        if raw.startswith(b"\xef\xbb\xbf"):
            raise MCPProtocolError(f"UTF-8 BOM leaked onto MCP stdout on {self.spec.name}")
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise MCPProtocolError(f"Invalid JSON on {self.spec.name} stdout: {raw[:200]!r}") from exc
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            raise MCPProtocolError(f"Non-JSON-RPC object leaked onto {self.spec.name} stdout: {raw[:200]!r}")
        return msg, framing

    def _send(self, payload: dict[str, Any], deadline: float, framing: str) -> None:
        enc = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(enc)}\r\n\r\n".encode("ascii") + enc if framing == "content-length" else enc + b"\n"
        try:
            self._writer.write_frame(frame, deadline, timeout_message=f"Timed out writing MCP frame to {self.spec.name}")
        except TimeoutError:
            raise
        except OSError as exc:
            if self._buf or self.proc.poll() is not None:
                return
            raise MCPProtocolError(f"MCP stdin closed unexpectedly on {self.spec.name}: {exc}") from exc

    def request(self, method: str, params: Mapping[str, Any] | None, deadline: float, *, framing: str = "content-length") -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = dict(params)
        self._send(msg, deadline, framing)
        while True:
            reply, reply_framing = self.read_message(deadline)
            if reply_framing != framing:
                raise MCPProtocolError(f"Framing mismatch on {self.spec.name}: expected {framing}, got {reply_framing}")
            if reply.get("id") == req_id:
                if "error" in reply:
                    raise MCPProtocolError(f"RPC error from {self.spec.name}.{method}: {reply['error']}")
                res = reply.get("result")
                if not isinstance(res, dict):
                    raise MCPProtocolError(f"Malformed result from {self.spec.name}.{method}: {reply}")
                return res

    def initialize(self, deadline: float) -> dict[str, Any]:
        res = self.request(
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "mindie-host-accept", "version": "0.4.16"}},
            deadline,
            framing="content-length",
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, deadline, "content-length")
        self.request("ping", {}, deadline, framing="jsonl")
        return res

    def list_tools(self, deadline: float) -> list[dict[str, Any]]:
        tools = self.request("tools/list", {}, deadline).get("tools")
        if not isinstance(tools, list):
            raise MCPProtocolError(f"{self.spec.name} tools/list returned non-list")
        return tools

    def call_tool(self, name: str, arguments: Mapping[str, Any], deadline: float) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": dict(arguments)}, deadline)

    def close(self) -> None:
        self._writer.close()
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except TimeoutError:
                raise
            except OSError:
                pass
        try:
            self.proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2.0)


def boot_all_servers(spec: HostManifestSpec, deadline: float) -> tuple[dict[str, MCPWireSession], dict[str, list[dict[str, Any]]]]:
    sessions: dict[str, MCPWireSession] = {}
    listed: dict[str, list[dict[str, Any]]] = {}
    lock = threading.Lock()

    def _start_one(srv: MCPServerSpec) -> None:
        sess = MCPWireSession(srv)
        with lock:
            sessions[srv.name] = sess
        sess.initialize(deadline)
        tools = sess.list_tools(deadline)
        with lock:
            listed[srv.name] = tools

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(spec.servers))) as pool:
            for fut in [pool.submit(_start_one, srv) for srv in spec.servers.values()]:
                fut.result()
    except BaseException:
        for s in list(sessions.values()):
            try:
                s.close()
            except Exception:
                pass
        raise
    return sessions, listed


def _teardown_wire(sessions: Mapping[str, MCPWireSession], sandbox_root: Path, cleanup_warnings: list[str]) -> None:
    for sname, sess in sessions.items():
        try:
            sess.close()
        except Exception as exc:
            cleanup_warnings.append(f"close({sname}): {type(exc).__name__}: {exc}")
    try:
        stop_sandbox_knowledge_services(sandbox_root)
    except Exception as exc:
        cleanup_warnings.append(f"stop_knowledge: {type(exc).__name__}: {exc}")


def _verify_parity_against_managed_adapter(
    host: str, spec: HostManifestSpec, listed: Mapping[str, list[dict[str, Any]]], env: Mapping[str, str], deadline: float
) -> None:
    cfg_file = Path(env["MINDIE_CONFIG_HOME"]) / f"{host}.json"
    py_exe = json.loads(cfg_file.read_text(encoding="utf-8")).get("python") if cfg_file.is_file() else None
    if not isinstance(py_exe, str) or not Path(py_exe).is_file():
        runtimes = Path(env["MINDIE_DATA_HOME"]) / "runtimes"
        for receipt in sorted(runtimes.rglob("runtime.json")) if runtimes.is_dir() else ():
            cand = json.loads(receipt.read_text(encoding="utf-8")).get("python")
            if isinstance(cand, str) and Path(cand).is_file():
                py_exe = cand
                break
    if not isinstance(py_exe, str) or not Path(py_exe).is_file():
        raise HostAcceptanceError(f"Cold launch did not provision a validated runtime Python for {host}")

    surfaces = {name: srv.surface for name, srv in spec.servers.items()}
    code = (
        "import json, sys\nfrom mindie_harness.paths import ProfileRef\nfrom mindie_harness.mcp.gateway import ManagedMCPAdapter\n"
        "host, plugin_root, surfaces = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])\n"
        "profile = ProfileRef.resolve(host, plugin_root=plugin_root)\n"
        "out = {s: {t['name']: t.get('inputSchema') for t in ManagedMCPAdapter(host, plugin_root=plugin_root, "
        "include_knowledge=(False if host == 'wb' and surf == 'tools' else None)).mcp_tools(surf, profile)} for s, surf in surfaces.items()}\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [py_exe, "-I", "-X", "utf8", "-c", code, host, str(spec.plugin_dir), json.dumps(surfaces)],
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1.0, deadline - time.monotonic()),
        check=False,
        **subprocess_window_kwargs(),
    )
    if proc.returncode != 0:
        raise HostAcceptanceError(f"ManagedMCPAdapter parity probe failed: {proc.stderr[:400]}")
    expected_map = json.loads(proc.stdout.strip().splitlines()[-1])
    for sname, tools in listed.items():
        actual = {str(t["name"]): t.get("inputSchema") for t in tools}
        exp = expected_map.get(sname) or {}
        if set(actual) != set(exp) or any(actual[k] != exp[k] for k in actual):
            raise HostAcceptanceError(f"{host}/{sname} tool/schema parity mismatch vs ManagedMCPAdapter")


def find_server_for_tool(sessions: Mapping[str, MCPWireSession], listed: Mapping[str, list[dict[str, Any]]], tool_name: str) -> MCPWireSession:
    for sname, tools in listed.items():
        if any(t.get("name") == tool_name for t in tools):
            return sessions[sname]
    raise HostAcceptanceError(f"Tool {tool_name!r} is not exposed by any MCP server")


def run_scenario_cold_native_launch(
    host: str, plugin_dir: Path, *, sandbox_root: Path, wheelhouse_stage: Path | None,
    redactor: Path | None, prepacked_dir: Path | None, deadline: float, cleanup_warnings: list[str],
) -> dict[str, Any]:
    """Scenario 1: Cold native launch, byte-0 stdout cleanliness, 1:1 parity & zero-`_meta` calls."""
    env, _ = build_sandbox_env(sandbox_root, wheelhouse_stage=wheelhouse_stage, redactor=redactor)
    if (Path(env["MINDIE_CONFIG_HOME"]) / f"{host}.json").exists():
        raise HostAcceptanceError(f"Sandbox was not cold for {host}")
    spec = parse_host_manifest(host, plugin_dir, sandbox_root=sandbox_root, env=env, prepacked_dir=prepacked_dir, timeout=max(2.0, deadline - time.monotonic()))
    expected_counts = EXPECTED_HOST_SERVERS[host]
    if set(spec.servers) != set(expected_counts):
        raise HostAcceptanceError(f"{host} server set {set(spec.servers)} != {set(expected_counts)}")

    sessions: dict[str, MCPWireSession] = {}
    try:
        sessions, listed = boot_all_servers(spec, deadline)
        actual_counts = {k: len(v) for k, v in listed.items()}
        if actual_counts != expected_counts:
            raise HostAcceptanceError(f"{host} tool counts {actual_counts} != {expected_counts}")
        all_names = [str(t["name"]) for tools in listed.values() for t in tools]
        if len(all_names) != len(set(all_names)) or (BANNED_LEGACY_TOOLS & set(all_names)):
            raise HostAcceptanceError(f"{host} duplicate or banned legacy tools in {all_names}")
        for sname, tools in listed.items():
            for tool in tools:
                req_fields = set((tool.get("inputSchema") or {}).get("required") or ())
                if FORBIDDEN_REQUIRED_KEYS & req_fields:
                    raise HostAcceptanceError(f"{host}/{sname}/{tool['name']} requires forbidden key: {req_fields}")
        if host == "agy" and "remote_list_dir" not in {t["name"] for t in listed["mindie-remote-dev"]}:
            raise HostAcceptanceError("agy mindie-remote-dev missing remote_list_dir")

        _verify_parity_against_managed_adapter(host, spec, listed, env, deadline)
        kq = find_server_for_tool(sessions, listed, "knowledge_query").call_tool("knowledge_query", {"query": "aclgraph", "domain": "vllm-ascend"}, deadline)
        if kq.get("isError") is not False or list(sandbox_root.rglob("*admission*.sqlite*")):
            raise HostAcceptanceError(f"{host} knowledge_query failed or leaked admission SQLite: {kq}")
        for tname, targs in (("mindie_doctor", {"probe_remote": False}), ("diagnostics_status", {}), ("sim", {"argv": ["catalog"]}), ("remote_auth", {"action": "status"})):
            res = find_server_for_tool(sessions, listed, tname).call_tool(tname, targs, deadline)
            if res.get("isError") is not False:
                raise HostAcceptanceError(f"{host} {tname} failed: {res}")

        if host == "codex":
            w_tool = next((t for t in listed["mindie-remote-dev"] if t["name"] == "remote_write_stdin"), None)
            if not w_tool or "session_id" not in ((w_tool.get("inputSchema") or {}).get("properties") or {}):
                raise HostAcceptanceError("codex remote_write_stdin missing session_id property")
            w_res = sessions["mindie-remote-dev"].call_tool("remote_write_stdin", {"session_id": 999999, "chars": ""}, deadline)
            if "forged_identity_argument" in json.dumps(w_res) or "ModuleNotFoundError" in json.dumps(w_res):
                raise HostAcceptanceError(f"codex remote_write_stdin identity/import regression: {w_res}")
        if host == "wb":
            wb_res = run_wb_installer(spec.plugin_dir / "scripts/wb_install.py", spec.plugin_dir, Path(env["WORKBUDDY_CONFIG_DIR"]), env)
            cmd_pinned = json.loads((Path(wb_res["cache_install_path"]) / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["mindie-tools"]["command"]
            if cmd_pinned in {"python", "python3"} or not Path(cmd_pinned).is_file():
                raise HostAcceptanceError(f"wb_install did not pin runtime python: {cmd_pinned}")
        return {"servers": actual_counts, "total_tools": len(all_names)}
    finally:
        _teardown_wire(sessions, sandbox_root, cleanup_warnings)


def run_scenario_public_install_py(
    host: str, plugin_dir: Path, *, sandbox_root: Path, wheelhouse_stage: Path | None,
    install_script: Path | None, redactor: Path | None, prepacked_dir: Path | None,
    deadline: float, cleanup_warnings: list[str],
) -> dict[str, Any]:
    """Scenario 2: Public `install.py` with fail-closed `git` `exit 99` trap -> Turn-1 MCP usability."""
    if wheelhouse_stage is None or not wheelhouse_stage.is_dir():
        raise FileNotFoundError("Scenario `public_install_py` requires `--wheelhouse-stage`")
    if install_script is None or not install_script.is_file():
        raise FileNotFoundError("Scenario `public_install_py` requires `--install-script` (`install.py`)")

    base_env, is_offline = build_sandbox_env(sandbox_root, wheelhouse_stage=wheelhouse_stage, redactor=redactor)
    env = install_git_trap(sandbox_root / "git-trap", base_env)
    cmd = [str(resolve_base_python()), "-I", str(install_script.resolve()), "--host", host, "--channel-base-url", wheelhouse_stage.resolve().as_uri(), *(["--offline"] if is_offline else []), "--json"]
    proc = subprocess.run(
        cmd,
        cwd=str(sandbox_root),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(2.0, deadline - time.monotonic()),
        check=False,
        **subprocess_window_kwargs(),
    )
    if proc.returncode != 0:
        raise HostAcceptanceError(f"install.py failed ({proc.returncode}): {proc.stderr[:300] or proc.stdout[:300]}")
    install_doc = json.loads(proc.stdout.strip().splitlines()[-1])
    if install_doc.get("status") != "installed" or host not in (install_doc.get("configured_hosts") or []):
        raise HostAcceptanceError(f"install.py did not configure {host}: {install_doc}")

    spec = parse_host_manifest(host, plugin_dir, sandbox_root=sandbox_root, env=env, prepacked_dir=prepacked_dir, timeout=max(2.0, deadline - time.monotonic()))
    sessions: dict[str, MCPWireSession] = {}
    try:
        sessions, listed = boot_all_servers(spec, deadline)
        for tname, targs in (("knowledge_query", {"query": "aclgraph", "domain": "vllm-ascend"}), ("mindie_doctor", {"probe_remote": False}), ("sim", {"argv": ["catalog"]})):
            res = find_server_for_tool(sessions, listed, tname).call_tool(tname, targs, deadline)
            if res.get("isError") is not False:
                raise HostAcceptanceError(f"Post-install.py Turn-1 {tname} failed on {host}: {res}")
        return {"install_status": install_doc["status"], "configured_hosts": install_doc["configured_hosts"]}
    finally:
        _teardown_wire(sessions, sandbox_root, cleanup_warnings)
