"""Scenario 3: Hook fast-paths, 0-SQLite negative states, and redacted `Stop` capture.

[Role]
Verifies all 7 native host hook lifecycles in isolated sandboxes:
- Part 3A: `SessionStart` / pre-hook `< 5ms` fast-path with zero `uv sync`/`pip` and zero SQLite;
- Part 3B: `Stop` across 3 negative states (no consent, `read-only` disabled, sibling prefix
  out-of-scope `cwd`) creating zero `.sqlite*` files;
- Part 3C: `local-only` authorized `Stop` capture across all 7 native transcript formats
  with synthetic secret redaction (`ghp_...`, `AKIA...`) and zero outbound network calls.

[Invariants]
- Never mutates `os.environ`; passes explicit `env` and `**subprocess_window_kwargs()`.
- Always precedes `except OSError:` with `except TimeoutError: raise`.

[Timeouts & Locks]
- Every hook and capture drain runs within caller monotonic `deadline`.

[Side-Effect Policy]
- Writes synthetic transcripts, consent configs, and hook shims strictly inside `sandbox_root`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import subprocess
import time
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from mindie_bench.host_acceptance import (
    HostAcceptanceError,
    build_sandbox_env,
    resolve_base_python,
    stop_sandbox_knowledge_services,
)
from mindie_bench.host_acceptance.manifests import HostManifestSpec, parse_host_manifest
from mindie_bench.runner.host_platform import (
    resolve_git_bash,
    subprocess_window_kwargs,
    which_native,
    write_executable_shim,
)

SECRET_GHP = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SECRET_AWS = "AKIAIOSFODNN7EXAMPLE"
ADVICE_MARKER = "aclmdlRICaptureBegin stream synchronization pattern"


def _run_text(
    argv: list[str], *, env: Mapping[str, str], timeout: float, cwd: Path | None = None, inp: str | None = None, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=dict(env),
        stdin=subprocess.DEVNULL if inp is None else None,
        input=inp,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(2.0, timeout),
        check=check,
        **subprocess_window_kwargs(),
    )


def _zstd_uncompressed_frame(raw: bytes) -> bytes:
    """Encode `raw` (`< 128 KiB`) as a valid RFC 8878 single-segment zstd frame."""
    if len(raw) >= 131072:
        raise ValueError("zstd helper expects payload < 128 KiB")
    return b"\x28\xb5\x2f\xfd\xa0" + struct.pack("<I", len(raw)) + ((len(raw) << 3) | 1).to_bytes(3, "little") + raw


def _sqlite_files(sandbox_root: Path, host: str) -> list[Path]:
    ignore = {"opencode.db", "opencode.db-wal", "opencode.db-shm"} if host == "oc" else set()
    return [
        p
        for p in sandbox_root.rglob("*")
        if p.is_file() and p.name not in ignore and (".sqlite" in p.name or p.suffix in {".db", ".sqlite3"})
    ]


def _provision_hook_venv(sandbox_root: Path, env: Mapping[str, str], is_offline: bool, deadline: float) -> Path:
    uv_bin = which_native("uv", path=env.get("PATH")) or which_native("uv")
    if not uv_bin:
        raise FileNotFoundError("`uv` executable is required on PATH for Scenario 3")
    venv_dir = sandbox_root / "hook-venv"
    venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    links = Path(env["UV_FIND_LINKS"])
    _run_text([uv_bin, "venv", "--python", str(resolve_base_python()), str(venv_dir)], env=env, timeout=deadline - time.monotonic(), check=True)
    req = links / "requirements.txt"
    cmd = [uv_bin, "--native-tls", "pip", "install", "--python", str(venv_py), "--find-links", str(links)]
    cmd.extend(["--override", str(req), "-r", str(req)] if req.is_file() else [str(w) for w in sorted(links.glob("*.whl"))])
    if is_offline:
        cmd.extend(["--offline", "--no-index"])
    _run_text(cmd, env=env, timeout=deadline - time.monotonic(), check=True)
    return venv_py


def _install_uv_sync_guard(guard_dir: Path, env: Mapping[str, str]) -> dict[str, str]:
    real_uv = which_native("uv", path=env.get("PATH")) or which_native("uv")
    if not real_uv:
        raise FileNotFoundError("`uv` executable not found")
    body = (
        f"import subprocess, sys\nreal = {real_uv!r}\n"
        "if any(a in {'sync', 'pip', 'venv'} for a in sys.argv[1:4]):\n"
        "    sys.stderr.write('forbidden uv build in hook fast-path: ' + repr(sys.argv[1:]) + '\\n')\n"
        "    raise SystemExit(98)\n"
        "kw = {'creationflags': 0x08000000} if sys.platform == 'win32' else {}\n"
        "raise SystemExit(subprocess.run([real, *sys.argv[1:]], **kw).returncode)\n"
    )
    write_executable_shim(guard_dir, "uv", body)
    return {**env, "PATH": str(guard_dir) + os.pathsep + env.get("PATH", "")}


def _write_host_transcript(host: str, ws: Path, env: Mapping[str, str]) -> tuple[str, Path, dict[str, Any]]:
    t0 = time.time() + 120.0
    iso = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + i)) for i in (1, 2, 3)]
    ms = [int((t0 + i) * 1000) for i in (1, 2, 3)]
    u_txt = f"Fix Ascend stream hang in vllm-ascend (token {SECRET_GHP}, key {SECRET_AWS})."
    a_txt = f"Apply {ADVICE_MARKER} before graph replay; rotate {SECRET_GHP} immediately."

    if host == "codex":
        sid, tp = "codex-accept-session-01", Path(env["CODEX_HOME"]) / "sessions/rollout-1.jsonl"
        tp.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"type": "session_meta", "timestamp": iso[0], "payload": {"id": sid, "cwd": str(ws), "timestamp": iso[0]}},
            {"type": "response_item", "timestamp": iso[1], "payload": {"type": "message", "role": "user", "channel": "final", "content": [{"type": "input_text", "text": u_txt}]}},
            {"type": "response_item", "timestamp": iso[2], "payload": {"type": "message", "role": "assistant", "channel": "final", "phase": "final", "content": [{"type": "output_text", "text": a_txt}]}},
        ]
        tp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return sid, tp, {"hook_event_name": "Stop", "identity_kind": "turn", "session_id": sid, "turn_id": "t-1", "harness": "codex", "transcript_path": str(tp)}

    if host == "cc":
        sid, tp = "11111111-2222-4333-8444-555555555555", Path(env["CLAUDE_CONFIG_DIR"]) / "projects/ws/11111111-2222-4333-8444-555555555555.jsonl"
        tp.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"type": "queue-operation", "operation": "enqueue", "sessionId": sid, "timestamp": iso[0]},
            {"type": "user", "sessionId": sid, "cwd": str(ws), "timestamp": iso[1], "message": {"role": "user", "content": [{"type": "text", "text": u_txt}]}},
            {"type": "assistant", "sessionId": sid, "cwd": str(ws), "timestamp": iso[2], "message": {"role": "assistant", "content": [{"type": "text", "text": a_txt}]}},
        ]
        tp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return sid, tp, {"hook_event_name": "Stop", "session_id": sid, "turn_id": "cc-1", "transcript_path": str(tp)}

    if host == "kimi":
        sid = "kimi-accept-session-01"
        sdir = Path(env["KIMI_CODE_HOME"]) / f"sessions/workdir/{sid}"
        (sdir / "agents/main").mkdir(parents=True, exist_ok=True)
        (sdir / "state.json").write_text(json.dumps({"workDir": str(ws), "createdAt": ms[0]}), encoding="utf-8")
        tp = sdir / "agents/main/wire.jsonl"
        rows = [
            {"type": "context.append_message", "time": ms[1], "message": {"role": "user", "origin": {"kind": "user"}, "content": [{"type": "text", "text": u_txt}]}},
            {"type": "context.append_message", "time": ms[2], "message": {"role": "assistant", "content": [{"type": "text", "text": a_txt}]}},
        ]
        tp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return sid, tp, {"hook_event_name": "Stop", "session_id": sid, "turn_id": "kimi-1"}

    if host == "dsh":
        sid, tp = "dsh-accept-session-01", Path(env["DSH_HOME"]) / "sessions/proj/dsh-accept-session-01/session.v4.jsonl.zstd"
        tp.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"v": 4, "seq": 1, "ts": iso[0], "type": "turn/start", "turn": 1, "data": {}},
            {"v": 4, "seq": 2, "ts": iso[1], "type": "user/message", "turn": 1, "data": {"source": {"kind": "user"}, "message": {"role": "user", "content": u_txt}}},
            {"v": 4, "seq": 3, "ts": iso[2], "type": "assistant/message", "turn": 1, "data": {"message": {"role": "assistant", "content": [{"type": "text", "text": a_txt}]}}},
        ]
        tp.write_bytes(_zstd_uncompressed_frame("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")))
        return sid, tp, {"session_id": sid, "turn": 1, "transcript_path": str(tp)}

    if host == "agy":
        sid = "22222222-3333-4444-8555-666666666666"
        tp = Path(env["AGY_APP_DATA_DIR"]) / f"brain/{sid}/.system_generated/logs/transcript_full.jsonl"
        tp.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "created_at": iso[1], "content": u_txt},
            {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "created_at": iso[2], "content": a_txt},
        ]
        tp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return sid, tp, {"conversationId": sid, "appDataDir": env["AGY_APP_DATA_DIR"], "executionNum": 1, "terminationReason": "NO_TOOL_CALL"}

    if host == "wb":
        sid = "33333333-4444-4555-8666-777777777777"
        tp = Path(env["WORKBUDDY_CONFIG_DIR"]) / f"projects/mindie/{sid}.jsonl"
        tp.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"type": "message", "role": "user", "timestamp": iso[1], "content": [{"type": "text", "text": u_txt}]},
            {"type": "message", "role": "assistant", "timestamp": iso[2], "content": [{"type": "text", "text": a_txt}]},
        ]
        tp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return sid, tp, {"hook_event_name": "Stop", "session_id": sid, "turn_id": "wb-1", "transcript_path": str(tp)}

    sid, db_path = "oc-accept-session-01", Path(env["MINDIE_OPENCODE_DB"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS session_v2(id TEXT PRIMARY KEY, directory TEXT, parent_id TEXT, fork_session_id TEXT, time_created INTEGER);"
            "CREATE TABLE IF NOT EXISTS session_message(id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER, time_created INTEGER, data TEXT);"
            "CREATE TABLE IF NOT EXISTS session_part(id TEXT PRIMARY KEY, session_id TEXT, message_id TEXT, ordinal INTEGER, type TEXT, data TEXT);"
        )
        conn.execute("INSERT OR REPLACE INTO session_v2 VALUES (?, ?, NULL, NULL, ?)", (sid, str(ws), ms[0]))
        conn.execute("INSERT OR REPLACE INTO session_message VALUES ('m1', ?, 'user', 1, ?, ?)", (sid, ms[1], json.dumps({"time": {"created": ms[1]}})))
        conn.execute("INSERT OR REPLACE INTO session_part VALUES ('p1', ?, 'm1', 0, 'text', ?)", (sid, json.dumps({"type": "text", "text": u_txt})))
        conn.execute("INSERT OR REPLACE INTO session_message VALUES ('m2', ?, 'assistant', 2, ?, ?)", (sid, ms[2], json.dumps({"time": {"created": ms[2]}})))
        conn.execute("INSERT OR REPLACE INTO session_part VALUES ('p2', ?, 'm2', 0, 'text', ?)", (sid, json.dumps({"type": "text", "text": a_txt})))
    return sid, db_path, {"session_id": sid}


def _invoke_hook(
    spec: HostManifestSpec, event_name: str, payload: Mapping[str, Any], env: Mapping[str, str], sandbox_root: Path, venv_py: Path, deadline: float
) -> str:
    rem = deadline - time.monotonic()
    hspec = spec.hooks[event_name]
    h_env = {**hspec.env, **env}
    if spec.host == "codex":
        h_env.update(PLUGIN_ROOT=str(spec.plugin_dir), PLUGIN_DATA=str(Path(env["CODEX_HOME"]) / "plugins/data/mindie-agent-mindie-agent"))
        assert hspec.command is not None
        ps = which_native("powershell.exe", path=h_env.get("PATH")) or which_native("powershell") or "powershell.exe"
        argv = [ps, *hspec.command.split()[1:]] if hspec.mode == "powershell_encoded" else [resolve_git_bash(h_env), "-c", hspec.command]
        proc = _run_text(argv, env=h_env, timeout=rem, cwd=hspec.cwd, inp=json.dumps(payload))
        if proc.returncode != 0:
            raise HostAcceptanceError(f"codex {event_name} failed ({proc.returncode}): {proc.stderr[:300]}")
        return proc.stdout.strip()

    if hspec.mode in {"dsh_esm", "oc_esm"}:
        node_bin = which_native("node", path=h_env.get("PATH")) or which_native("node")
        pkg_dir = Path(spec.metadata["packed_dir"])
        if hspec.mode == "oc_esm":
            h_env.update(OC_PLUGIN_ROOT=str(spec.plugin_dir), OC_PLUGIN_DATA=env["MINDIE_DATA_HOME"])
        if event_name == "SessionStart":
            launch = (pkg_dir if hspec.mode == "dsh_esm" else spec.plugin_dir) / "scripts/mindie_launch.py"
            proc = _run_text([str(venv_py), "-I", "-X", "utf8", str(launch), "hook", "session-start"], env=h_env, timeout=rem, cwd=launch.parent, inp="{}")
            if proc.returncode != 0:
                raise HostAcceptanceError(f"{spec.host} session-start failed: {proc.stderr[:300]}")
            return proc.stdout.strip()
        drv = sandbox_root / f"{spec.host}-stop-{time.monotonic_ns()}.mjs"
        hdr = (
            "import { ChildProcess } from 'node:child_process'; import { pathToFileURL } from 'node:url';\n"
            "const kids = [], orig = ChildProcess.prototype.unref;\n"
            "ChildProcess.prototype.unref = function(...a) {\n"
            "  if (Array.isArray(this.spawnargs) && this.spawnargs.some((x) => String(x).includes('stop'))) {\n"
            "    kids.push(new Promise((r) => { if (this.exitCode !== null) return r(0); this.once('close', r); this.once('error', () => r(-1)); }));\n"
            "    return this; }\n  return orig.apply(this, a); };\n"
        )
        if hspec.mode == "dsh_esm":
            mcp_mock = sandbox_root / "dsh-esm-probe/mock_mcp.mjs"
            body = (
                f"const hostMod = await import(pathToFileURL({json.dumps(str(pkg_dir / 'host.mjs'))}).href);\n"
                "const listeners = new Map(), disposers = [];\n"
                f"const agent = {{ session: {{ id: {json.dumps(payload['session_id'])}, header: {{ cwd: {json.dumps(payload['cwd'])} }} }}, options: {{}}, ctx: null }};\n"
                "const mk = () => ({ _captured: [], agents: { list: () => [agent] }, llm: { async *stream(){} },\n"
                "  on(e, f) { listeners.set(e, f); }, effect(f) { const d = f(); if (typeof d === 'function') disposers.push(d); },\n"
                "  async plugin(p, c) { await (p?.apply || p?.default?.apply || p)(mk(), c); return { dispose: async () => {} }; } });\n"
                "agent.ctx = mk();\n"
                f"await hostMod.apply(mk(), {{ mcpPluginUrl: pathToFileURL({json.dumps(str(mcp_mock))}).href }});\n"
                "await listeners.get('agent/turn-stopping')?.({ agent, turn: 1 });\n"
                "await Promise.all(kids);\nfor (const d of disposers.reverse()) { try { await d(); } catch {} }\n"
            )
        else:
            body = (
                f"const mod = await import(pathToFileURL({json.dumps(str(pkg_dir / 'index.mjs'))}).href);\n"
                f"const sid = {json.dumps(payload['session_id'])}, cwd = {json.dumps(payload['cwd'])};\n"
                "const ctx = { mcp: { transform(f) { f({ set(){} }); }, reload(){} },\n"
                "  session: { hook(){}, async get() { return { directory: cwd }; } }, tool: { hook(){} }, shell: { hook(){} },\n"
                "  event: { async subscribe() { return (async function*() { yield { id: 'ev-' + Date.now(), type: 'session.idle', data: { sessionID: sid } }; })(); } } };\n"
                "const dispose = await (mod.default || mod).setup(ctx);\n"
                "await new Promise((r) => setTimeout(r, 80));\nawait Promise.all(kids);\n"
                "if (typeof dispose === 'function') await dispose();\n"
            )
        drv.write_text(hdr + body, encoding="utf-8")
        proc = _run_text([str(node_bin), str(drv)], env=h_env, timeout=rem, cwd=sandbox_root)
        if proc.returncode != 0:
            raise HostAcceptanceError(f"{spec.host} ESM Stop hook failed: {proc.stderr[:300]}")
        return "{}"

    assert hspec.argv is not None
    proc = _run_text(list(hspec.argv), env=h_env, timeout=rem, cwd=hspec.cwd, inp=json.dumps(payload))
    if proc.returncode != 0:
        raise HostAcceptanceError(f"{spec.host} {event_name} failed ({proc.returncode}): {proc.stderr[:300]}")
    return proc.stdout.strip()


def _resolve_redactor(venv_py: Path, env: Mapping[str, str], redactor: Path | None, is_offline: bool, deadline: float) -> str:
    for cand in (redactor, Path(env["MINDIE_REDACTOR_EXECUTABLE"]) if env.get("MINDIE_REDACTOR_EXECUTABLE") else None):
        if cand is not None and cand.is_file():
            return str(cand.resolve())
    found = which_native("gitleaks", path=env.get("PATH")) or which_native("gitleaks")
    if found:
        return str(Path(found).resolve())
    if not is_offline:
        code = "from pathlib import Path; import os; from mindie_knowledge.loop.transcript_redaction import install_scanner; print(install_scanner(cache=Path(os.environ['MINDIE_DATA_HOME'])/'cache'/'gitleaks'))"
        proc = _run_text([str(venv_py), "-I", "-X", "utf8", "-c", code], env=env, timeout=deadline - time.monotonic())
        if proc.returncode == 0 and Path(proc.stdout.strip().splitlines()[-1]).is_file():
            return proc.stdout.strip().splitlines()[-1]
    raise FileNotFoundError("No `gitleaks` redactor executable available (`--redactor` or `gitleaks` on PATH)")


def run_scenario_hooks_and_capture(
    host: str,
    plugin_dir: Path,
    *,
    sandbox_root: Path,
    wheelhouse_stage: Path | None,
    redactor: Path | None,
    prepacked_dir: Path | None,
    deadline: float,
    cleanup_warnings: list[str],
) -> dict[str, Any]:
    """Scenario 3: Part 3A (`SessionStart`), Part 3B (3 negative 0-SQLite `Stop` states), Part 3C (`Stop` capture)."""
    if wheelhouse_stage is None or not wheelhouse_stage.is_dir():
        raise FileNotFoundError("Scenario `hooks_and_capture` requires `--wheelhouse-stage`")
    sandbox_root = sandbox_root.resolve()
    base_env, is_offline = build_sandbox_env(sandbox_root, wheelhouse_stage=wheelhouse_stage, redactor=redactor)
    ws = (sandbox_root / "ws").resolve()
    ws_other = Path(str(ws) + "-other")
    ws_other.mkdir(parents=True, exist_ok=True)

    spec = parse_host_manifest(
        host, plugin_dir, sandbox_root=sandbox_root, env=base_env, prepacked_dir=prepacked_dir, timeout=max(2.0, deadline - time.monotonic())
    )
    venv_py = _provision_hook_venv(sandbox_root, base_env, is_offline, deadline)
    cfg_dir = Path(base_env["MINDIE_CONFIG_HOME"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / f"{host}.json"
    state_dir = Path(base_env["MINDIE_DATA_HOME"]) / "state" / host
    cfg_file.write_text(
        json.dumps(
            {
                "schema": "mindie-host-config/1",
                "host": host,
                "python": str(venv_py),
                "source_root": str(spec.plugin_dir),
                "runtime_scripts": str(spec.plugin_dir / "scripts"),
                "state_dir": str(state_dir),
                "product_validation": {"schema": "mindie-candidate-check/1", "status": "validated", "python": str(venv_py)},
            }
        ),
        encoding="utf-8",
    )

    try:
        fast_env = _install_uv_sync_guard(sandbox_root / "uv-guard", base_env)
        ss_out = _invoke_hook(spec, "SessionStart", {"cwd": str(ws)}, fast_env, sandbox_root, venv_py, deadline)
        ss_json = json.loads(ss_out) if ss_out else {}
        if host == "wb" and ss_json.get("suppressOutput") is not True:
            raise HostAcceptanceError(f"wb SessionStart missing suppressOutput: {ss_json}")
        marker_ok = (state_dir / "pending-update-check.json").is_file()
        bad_db = _sqlite_files(sandbox_root, host)
        if not marker_ok or bad_db:
            raise HostAcceptanceError(f"{host} SessionStart marker/SQLite invariant failed: marker={marker_ok} db={bad_db}")

        sid, tp, base_stop = _write_host_transcript(host, ws, fast_env)
        consent_file, comm_file = cfg_dir / "mindie-consent.json", cfg_dir / "mindie-community.json"
        for neg_name, consent_doc, comm_doc, target_cwd in (
            ("missing_consent", None, None, ws),
            (
                "readonly_disabled",
                {"schema": "mindie-consent/1", "choice": "read-only", "enabled": False, "choice_revision": "r0"},
                {"schema": "mindie-community-config/1", "enabled": False, "project_roots": [str(ws)]},
                ws,
            ),
            (
                "sibling_prefix_out_of_scope",
                {"schema": "mindie-consent/1", "choice": "local-only", "enabled": True, "choice_revision": "r1"},
                {"schema": "mindie-community-config/1", "enabled": True, "project_roots": [str(ws)], "consent_config": str(consent_file)},
                ws_other,
            ),
        ):
            consent_file.unlink(missing_ok=True)
            comm_file.unlink(missing_ok=True)
            if consent_doc is not None:
                consent_file.write_text(json.dumps(consent_doc), encoding="utf-8")
            if comm_doc is not None:
                comm_file.write_text(json.dumps(comm_doc), encoding="utf-8")
            _invoke_hook(spec, "Stop", {**base_stop, "cwd": str(target_cwd), "workspaceRoot": str(target_cwd)}, fast_env, sandbox_root, venv_py, deadline)
            if bad_db := _sqlite_files(sandbox_root, host):
                raise HostAcceptanceError(f"{host} negative Stop ({neg_name}) created SQLite: {bad_db}")

        redactor_bin = _resolve_redactor(venv_py, base_env, redactor, is_offline, deadline)
        cap_env = dict(base_env, MINDIE_REDACTOR_EXECUTABLE=redactor_bin, HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost")
        consent_file.write_text(json.dumps({"schema": "mindie-consent/1", "choice": "local-only", "enabled": True, "choice_revision": "r2", "project_roots": [str(ws)]}), encoding="utf-8")
        comm_file.write_text(json.dumps({"schema": "mindie-community-config/1", "enabled": True, "enabled_at": 1700000000.0, "project_roots": [str(ws)], "consent_config": str(consent_file)}), encoding="utf-8")
        bind_code = (
            "import json, sys\nfrom pathlib import Path\nfrom mindie_harness.paths import ProfileRef\n"
            "from mindie_harness.runtime.knowledge_binding import ensure_knowledge_binding\n"
            "p = ProfileRef.resolve(sys.argv[1], plugin_root=sys.argv[2])\n"
            "res = ensure_knowledge_binding(p, plugin_root=Path(sys.argv[2]), python_executable=sys.executable)\n"
            "cfg = json.loads(p.config_path.read_text(encoding='utf-8'))\n"
            "cfg.update({'python': sys.executable, 'source_root': sys.argv[2], 'runtime_scripts': str(Path(sys.argv[2])/'scripts')})\n"
            "p.config_path.write_text(json.dumps(cfg), encoding='utf-8')\nprint(json.dumps(res))\n"
        )
        bproc = _run_text([str(venv_py), "-I", "-X", "utf8", "-c", bind_code, host, str(spec.plugin_dir)], env=cap_env, timeout=deadline - time.monotonic())
        if bproc.returncode != 0:
            raise HostAcceptanceError(f"{host} ensure_knowledge_binding failed: {bproc.stderr[:400]}")

        _invoke_hook(spec, "Stop", {**base_stop, "cwd": str(ws), "workspaceRoot": str(ws)}, cap_env, sandbox_root, venv_py, deadline)
        stop_sandbox_knowledge_services(sandbox_root)

        drain_code = (
            "import json, sys\nfrom pathlib import Path\n"
            "from mindie_knowledge.loop.activation import Admission\nfrom mindie_knowledge.loop.cli import config_at, load_transcript_adapter\n"
            "from mindie_knowledge.loop.engine import Engine\nfrom mindie_knowledge.loop.store import Store\n"
            "ecfg = config_at(Path(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['engine_config']))\n"
            "store = Store(Path(ecfg['root']) / ecfg['domain'], domain=ecfg['domain'])\n"
            "try:\n"
            "    eng = Engine(store, settings_path=ecfg.get('community_config'), admission=Admission(ecfg['admission_path']),\n"
            "      transcript_adapter=load_transcript_adapter(ecfg),\n"
            "      capture_mode=ecfg.get('capture_mode', 'public-transcript'), redactor_executable=ecfg.get('redactor_executable'))\n"
            "    n = 0\n"
            "    while (ident := store.due_capture()) is not None and n < 8:\n        eng._process(ident); n += 1\n"
            "finally:\n    store.close()\n"
            "print(json.dumps({'drained': n, 'adm': str(ecfg['admission_path']), 'kroot': str(Path(ecfg['root'])/ecfg['domain'])}))\n"
        )
        dproc = _run_text([str(venv_py), "-I", "-X", "utf8", "-c", drain_code, str(cfg_file)], env=cap_env, timeout=deadline - time.monotonic())
        if dproc.returncode != 0:
            raise HostAcceptanceError(f"{host} capture drain failed: {dproc.stderr[:400]}")
        drain_info = json.loads(dproc.stdout.strip().splitlines()[-1])
        adm_path, kdomain_dir = Path(drain_info["adm"]).resolve(), Path(drain_info["kroot"]).resolve()
        if not adm_path.is_file():
            raise HostAcceptanceError(f"{host} positive Stop did not create {adm_path}")

        tp_res = tp.resolve()
        committed_blobs: list[str] = []
        for fpath in Path(cap_env["MINDIE_DATA_HOME"]).resolve().rglob("*"):
            if not fpath.is_file() or fpath.resolve() == tp_res:
                continue
            raw_txt = fpath.read_bytes().decode("utf-8", errors="ignore")
            if SECRET_GHP in raw_txt or SECRET_AWS in raw_txt:
                raise HostAcceptanceError(f"{host} leaked unredacted secret into {fpath}")
            if ADVICE_MARKER in raw_txt and kdomain_dir in fpath.resolve().parents:
                committed_blobs.append(str(fpath))
        if not committed_blobs:
            raise HostAcceptanceError(f"{host} positive Stop did not commit sanitized transcript material in {kdomain_dir}")
        return {"session_id": sid, "committed_files": len(committed_blobs), "drained": drain_info["drained"]}
    finally:
        try:
            stop_sandbox_knowledge_services(sandbox_root)
        except Exception as exc:
            cleanup_warnings.append(f"stop_knowledge: {type(exc).__name__}: {exc}")
