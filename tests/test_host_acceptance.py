"""Offline end-to-end tests for Gate Layer 2 (`mindie_bench.host_acceptance`).

[Role]
Builds a self-contained 10-wheel pure-stdlib wheelhouse stage, `install.py`, a
`gitleaks`-compatible secret redactor shim, and realistic 7-host native plugin trees
(`codex`, `cc`, `kimi`, `dsh`, `agy`, `wb`, `oc`) to verify all 4 scenarios, byte-0
stdout framing, `git` `exit 99` traps, 0-SQLite negative `Stop` states, secret
redaction, retired/deleted engine upgrade cutover, and `BENCH-VERDICT` honesty.

[Invariants]
- 100% offline (`< 15s`), zero `pytest.skip`, zero host home/config pollution.
- `mindie_harness` wheel stays 100% pure stdlib at top-level import (`HAR-ABI`),
  matching `install.py`'s `--no-deps` bootstrap venv contract.

[Timeouts & Locks]
- Session-scoped fixture builds immutable wheelhouse & plugin templates once per run.

[Side-Effect Policy]
- Writes exclusively under `tmp_path_factory` / `tmp_path` sandboxes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mindie_bench.cli import main as bench_main
from mindie_bench.host_acceptance import (
    ALL_HOSTS,
    ALL_SCENARIOS,
    EXPECTED_HOST_SERVERS,
    MCPProtocolError,
    MCPServerSpec,
    MCPWireSession,
    build_sandbox_env,
    exit_code_for_report,
    parse_host_manifest,
    resolve_base_python,
    run_host_acceptance,
)
from mindie_bench.runner.host_platform import write_executable_shim

WHEEL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mindie-harness", "mindie_harness", "0.1.5"),
    ("mindie-knowledge", "mindie_knowledge", "0.13.25"),
    ("remote-dev", "remote_dev", "0.1.9"),
    ("mindie-diagnostics", "mindie_diagnostics", "0.1.8"),
    ("mindie-debugger", "mindie_debugger", "0.1.7"),
    ("mindie-profiler", "mindie_profiler", "0.1.7"),
    ("mindie-sim", "mindie_sim", "0.1.8"),
    ("npu-top", "npu_top", "0.1.6"),
    ("mindie-coordinator", "mindie_coordinator", "0.1.7"),
    ("reme-ai", "reme_ai", "0.3.1"),
)


def _build_wheel(dest_dir: Path, dist_name: str, pkg_name: str, version: str, files: dict[str, str]) -> tuple[Path, str]:
    dist_norm = dist_name.replace("-", "_")
    whl_name = f"{dist_norm}-{version}-py3-none-any.whl"
    whl_path = dest_dir / whl_name
    dist_info = f"{dist_norm}-{version}.dist-info"
    records: list[str] = []
    with zipfile.ZipFile(whl_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        all_files = {
            **{f"{pkg_name}/{rel}": body for rel, body in files.items()},
            f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n\n",
            f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nGenerator: mindie-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        }
        for arcname, text in sorted(all_files.items()):
            raw = text.encode("utf-8")
            zf.writestr(arcname, raw)
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode("ascii")
            records.append(f"{arcname},sha256={digest},{len(raw)}")
        records.append(f"{dist_info}/RECORD,,")
        zf.writestr(f"{dist_info}/RECORD", "\n".join(records) + "\n")
    return whl_path, hashlib.sha256(whl_path.read_bytes()).hexdigest()


def _mini_harness_files() -> dict[str, str]:
    paths_py = r"""from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

HOSTS = ("codex", "cc", "kimi", "dsh", "agy", "wb", "oc")

@dataclass(frozen=True)
class ProfileRef:
    host: str
    plugin_root: Path
    config_path: Path
    consent_path: Path
    data_dir: Path
    state_dir: Path
    generations_dir: Path
    admission_db: Path

    @classmethod
    def resolve(cls, host: str, *, plugin_root=None, config_path=None, data_dir=None, env=None):
        e = os.environ if env is None else env
        h = str(host).strip().lower()
        croot = Path(e.get("MINDIE_CONFIG_HOME") or (Path(e.get("HOME", ".")) / ".config/mindie-agent")).resolve()
        droot = Path(e.get("MINDIE_DATA_HOME") or (Path(e.get("HOME", ".")) / ".local/share/mindie-agent")).resolve()
        cfile = Path(config_path).resolve() if config_path else (croot / f"{h}.json")
        proot = Path(plugin_root).resolve() if plugin_root else Path.cwd().resolve()
        ddir = Path(data_dir).resolve() if data_dir else droot
        sdir = ddir / "state" / h
        return cls(h, proot, cfile, cfile.parent / "mindie-consent.json", ddir, sdir, ddir / "runtimes" / h, sdir / "admission.sqlite3")
"""
    gateway_py = r"""from __future__ import annotations
import json, os, sys
from pathlib import Path

REMOTE_COUNTS = {"codex": 10, "cc": 13, "kimi": 14, "dsh": 14, "agy": 12, "wb": 15, "oc": 14}
KNOWLEDGE_TOOLS = ("knowledge_query", "knowledge_explain", "knowledge_feedback", "knowledge_distill", "knowledge_sync")
COMPONENT_TOOLS = (
    "mindie_status", "mindie_doctor", "profiler", "sim", "coordinator",
    "npu_top", "debugger", "rsi_loop", "experiment_ledger", "diagnostics_status", "diagnostics_bundle",
)
BASE_REMOTE = [
    "remote_read_file", "remote_write_file", "remote_edit_file", "remote_run_command",
    "remote_manage_task", "remote_find_by_name", "remote_grep_search", "remote_auth",
    "remote_probe", "remote_artifact_pull", "remote_artifact_push", "remote_list_dir",
    "remote_stat", "remote_mkdir", "remote_rm",
]

def _schema(props=None, req=()):
    return {"type": "object", "properties": props or {}, "required": list(req)}

class ManagedMCPAdapter:
    def __init__(self, host: str, *, plugin_root=None, include_knowledge=None):
        self.host = host.strip().lower()
        self.plugin_root = Path(plugin_root).resolve() if plugin_root else Path.cwd()
        self.include_knowledge = include_knowledge

    def mcp_tools(self, surface: str, profile=None) -> list[dict]:
        s = surface.strip().lower()
        if s == "knowledge":
            return [{"name": n, "description": n, "inputSchema": _schema({"query": {"type": "string"}, "domain": {"type": "string"}})} for n in KNOWLEDGE_TOOLS]
        if s == "components":
            return [{"name": n, "description": n, "inputSchema": _schema({"argv": {"type": "array"}})} for n in COMPONENT_TOOLS]
        if s == "tools":
            names = list(COMPONENT_TOOLS) + ([] if self.include_knowledge is False else list(KNOWLEDGE_TOOLS))
            return [{"name": n, "description": n, "inputSchema": _schema({"query": {"type": "string"}, "domain": {"type": "string"}, "argv": {"type": "array"}, "probe_remote": {"type": "boolean"}})} for n in names]
        cnt = REMOTE_COUNTS[self.host]
        if self.host == "codex":
            rnames = [
                "remote_read_file", "remote_write_file", "remote_edit_file", "remote_run_command",
                "remote_write_stdin", "remote_manage_task", "remote_auth", "remote_probe",
                "remote_artifact_pull", "remote_artifact_push",
            ]
        elif self.host == "agy":
            rnames = BASE_REMOTE[:12]
        else:
            rnames = [n for n in BASE_REMOTE if n != "remote_list_dir"][:cnt]
            while len(rnames) < cnt:
                rnames.append(f"remote_extra_{len(rnames)}")
        out = []
        for n in rnames:
            props = {"session_id": {"type": "integer"}, "chars": {"type": "string"}} if n == "remote_write_stdin" else {"action": {"type": "string"}}
            out.append({"name": n, "description": n, "inputSchema": _schema(props)})
        return out

def _handle_call(profile, name: str, args: dict) -> dict:
    if name == "knowledge_query":
        from mindie_knowledge.loop.store import Store
        cfg = json.loads(profile.config_path.read_text(encoding="utf-8")) if profile.config_path.is_file() else {}
        dom = str(args.get("domain") or "vllm-ascend")
        ep = (cfg.get("domain_configs") or {}).get(dom) or cfg.get("engine_config")
        kroot = Path(json.loads(Path(ep).read_text(encoding="utf-8"))["root"]) if ep and Path(ep).is_file() else (profile.data_dir / "knowledge")
        st = Store(kroot, domain=dom)
        try:
            units = st.graded.query(str(args.get("query") or ""))
        finally:
            st.close()
        return {"content": [{"type": "text", "text": json.dumps({"domain": dom, "units": units})}], "isError": False}
    if name == "remote_write_stdin":
        return {"content": [{"type": "text", "text": json.dumps({"status": "session_not_found", "session_id": args.get("session_id")})}], "isError": False}
    return {"content": [{"type": "text", "text": json.dumps({"tool": name, "status": "ok"})}], "isError": False}

def serve_stdio(host: str, surface: str, plugin_root: str | Path, *, include_knowledge=None) -> int:
    from mindie_harness.paths import ProfileRef
    from mindie_harness.runtime.knowledge_binding import ensure_knowledge_binding
    if os.environ.get("MINDIE_TEST_LEAK_STDOUT") == "1":
        sys.stdout.buffer.write(b'{"status": "provisioned"}\n')
        sys.stdout.buffer.flush()
    profile = ProfileRef.resolve(host, plugin_root=plugin_root)
    if not profile.config_path.is_file():
        ensure_knowledge_binding(profile, plugin_root=Path(plugin_root), python_executable=sys.executable)
    adapter = ManagedMCPAdapter(host, plugin_root=plugin_root, include_knowledge=include_knowledge)
    tools = adapter.mcp_tools(surface, profile)
    inp = sys.stdin.buffer
    while True:
        first = inp.readline()
        if not first:
            return 0
        if first.lower().startswith(b"content-length:"):
            clen = int(first.split(b":", 1)[1].strip())
            while inp.readline() not in (b"\r\n", b"\n", b""):
                pass
            raw = inp.read(clen)
            framing = "cl"
        else:
            raw = first.strip()
            if not raw:
                continue
            framing = "jsonl"
        msg = json.loads(raw.decode("utf-8"))
        if "id" not in msg:
            continue
        m = msg.get("method")
        if m == "initialize":
            res = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": f"mindie-{surface}", "version": "0.1.5"}}
        elif m == "ping":
            res = {}
        elif m == "tools/list":
            res = {"tools": tools}
        elif m == "tools/call":
            p = msg.get("params") or {}
            res = _handle_call(profile, str(p.get("name") or ""), dict(p.get("arguments") or {}))
        else:
            res = {}
        body = json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}, separators=(",", ":")).encode("utf-8")
        if framing == "cl":
            sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        else:
            sys.stdout.buffer.write(body + b"\n")
        sys.stdout.buffer.flush()
"""
    kb_py = r"""from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

DOMAINS = ("vllm-ascend", "operators", "xllm", "mindie-sd", "infer-core", "infer-distributed", "infer-observability", "infer-ops-compat", "agent-engineering")
ADAPTERS = {
    "codex": "scripts/codex_transcript.py", "cc": "scripts/transcript.py", "kimi": "scripts/kimi_transcript.py",
    "dsh": "scripts/dsh_transcript.py", "agy": "scripts/agy_transcript.py", "wb": "scripts/wb_transcript.py",
    "oc": "scripts/opencode_transcript.py",
}

def _inside(cand: str, root: str) -> bool:
    try:
        c, r = os.path.normcase(os.path.realpath(cand)), os.path.normcase(os.path.realpath(root))
        return os.path.splitdrive(c)[0] == os.path.splitdrive(r)[0] and os.path.commonpath([c, r]) == r
    except Exception:
        return False

def ensure_knowledge_binding(profile, *, plugin_root=None, python_executable=None, install_redactor=False, retire_previous=False):
    proot = Path(plugin_root or profile.plugin_root).resolve()
    py = str(Path(python_executable or sys.executable))
    cfg = json.loads(profile.config_path.read_text(encoding="utf-8")) if profile.config_path.is_file() else {}
    prev_ep = cfg.get("engine_config")
    prev_doc = json.loads(Path(prev_ep).read_text(encoding="utf-8")) if isinstance(prev_ep, str) and Path(prev_ep).is_file() else {}
    if retire_previous and isinstance(prev_ep, str) and Path(prev_ep).is_file():
        Path(prev_ep + ".retired").write_text("retired\n", encoding="utf-8")
    prev_root = prev_doc.get("root")
    if isinstance(prev_root, str) and prev_root and not _inside(prev_root, str(profile.generations_dir)):
        kroot = Path(prev_root)
    else:
        kroot = profile.data_dir / "knowledge"
    adapter = (proot / ADAPTERS[profile.host]).resolve()
    red = os.environ.get("MINDIE_REDACTOR_EXECUTABLE") or str(profile.data_dir / "cache/gitleaks/gitleaks")
    key = hashlib.sha256(f"{profile.host}:{py}:{adapter}:{red}:{kroot}".encode()).hexdigest()[:20]
    edir = profile.config_path.parent / "engines" / profile.host / key
    edir.mkdir(parents=True, exist_ok=True)
    dmap = {}
    for dom in DOMAINS:
        ep = edir / f"{dom}.engine.json"
        ep.write_text(json.dumps({
            "root": str(kroot), "domain": dom, "capture_mode": "public-transcript" if Path(red).is_file() else "disabled",
            "transcript_adapter": str(adapter), "redactor_executable": red,
            "admission_path": str(profile.admission_db), "community_config": str(profile.config_path.parent / "mindie-community.json"),
        }), encoding="utf-8")
        dmap[dom] = str(ep)
    cfg.update({
        "schema": "mindie-host-config/1", "host": profile.host, "python": py, "source_root": str(proot),
        "runtime_scripts": str(proot / "scripts"), "state_dir": str(profile.state_dir),
        "engine_config": dmap["vllm-ascend"], "domain_configs": dmap,
        "product_validation": dict(cfg.get("product_validation") or {}, schema="mindie-candidate-check/1", status="validated", python=py),
    })
    profile.config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg = profile.config_path.with_name(f".{profile.config_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_cfg, profile.config_path)
    return {"status": "configured", "engine_config": dmap["vllm-ascend"], "domain_configs": dmap}

def run_stop_capture(profile, event, *, plugin_root=None, timeout_sec=6.0):
    from mindie_knowledge.loop.activation import Admission
    from mindie_knowledge.loop.handoff import accept_stop
    comm_p = profile.config_path.parent / "mindie-community.json"
    cons_p = profile.consent_path
    if not comm_p.is_file() or not cons_p.is_file():
        return {"stage": "inert", "reason": "sharing-disabled"}
    comm = json.loads(comm_p.read_text(encoding="utf-8"))
    cons = json.loads(cons_p.read_text(encoding="utf-8"))
    if comm.get("enabled") is not True or cons.get("choice") not in {"contribute", "local-only"} or not cons.get("choice_revision"):
        return {"stage": "inert", "reason": "sharing-disabled"}
    roots = [r for r in (comm.get("project_roots") or []) if isinstance(r, str) and os.path.isabs(r)]
    cwd = next((event.get(k) for k in ("cwd", "workspaceRoot", "directory", "workspace") if isinstance(event.get(k), str) and os.path.isabs(event[k])), None)
    if not cwd or not any(_inside(cwd, r) for r in roots):
        return {"stage": "inert", "reason": "out-of-scope"}
    cfg = json.loads(profile.config_path.read_text(encoding="utf-8")) if profile.config_path.is_file() else {}
    ep = cfg.get("engine_config")
    if not ep or not Path(ep).is_file():
        ep = ensure_knowledge_binding(profile, plugin_root=plugin_root)["engine_config"]
    sid = str(event.get("session_id") or event.get("conversationId") or event.get("sessionID") or "")
    adm = Admission(profile.admission_db)
    lease = adm.activate(sid, project_root=roots[0], root_session=sid, not_before=0.0)
    ev = dict(event, hook_event_name="Stop", session_id=sid, mindie_activation=lease["token"])
    return accept_stop(Path(ep), ev)
"""
    activation_py = r"""from __future__ import annotations
import json, shutil, time
from pathlib import Path

def write_pending_update_check(profile, *, reason="session_start"):
    profile.state_dir.mkdir(parents=True, exist_ok=True)
    target = profile.state_dir / "pending-update-check.json"
    target.write_text(json.dumps({"schema": "mindie-pending-update/1", "host": profile.host, "reason": reason, "requested_at": time.time()}) + "\n", encoding="utf-8")
    return target

def maintain_generations(profile):
    gdir = profile.generations_dir
    if not gdir.is_dir():
        return {"retained": []}
    cfg = json.loads(profile.config_path.read_text(encoding="utf-8")) if profile.config_path.is_file() else {}
    active_py = Path(cfg["python"]).resolve() if cfg.get("python") else None
    gens = []
    for d in gdir.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        rc = json.loads((d / "runtime.json").read_text(encoding="utf-8")) if (d / "runtime.json").is_file() else {}
        ts = float((rc.get("validation") or {}).get("validated_at") or 0.0)
        is_active = active_py is not None and (active_py == Path(rc.get("python", "")).resolve() or d.resolve() in active_py.parents)
        gens.append((1 if is_active else 0, ts, d))
    gens.sort(key=lambda item: (item[0], item[1]), reverse=True)
    keep = {item[2] for item in gens[:2]}
    for _, _, d in gens[2:]:
        trash = gdir / f".trash.{d.name}"
        if trash.exists():
            shutil.rmtree(trash, ignore_errors=True)
        d.rename(trash)
        shutil.rmtree(trash, ignore_errors=True)
    return {"retained": [d.name for d in keep]}
"""
    wheelhouse_py = r"""from __future__ import annotations
import json, os, shutil, subprocess, sys, tarfile, time, urllib.parse, urllib.request
from pathlib import Path
from mindie_harness.runtime.activation import maintain_generations

def _stage_to_path(uri_or_path: str) -> Path:
    if uri_or_path.startswith("file://"):
        return Path(urllib.request.url2pathname(urllib.parse.urlparse(uri_or_path).path)).resolve()
    return Path(uri_or_path).resolve()

def update_runtime(profile, *, force=False, offline=False, channel_base_url=None, env=None, **_kw):
    e = dict(os.environ if env is None else env)
    stage = _stage_to_path(str(channel_base_url or e.get("MINDIE_CHANNEL_BASE_URL")))
    chan = json.loads((stage / "release-channel.json").read_text(encoding="utf-8"))
    gen_dir = profile.generations_dir / f"gen-{time.monotonic_ns()}"
    wheels_dir = gen_dir / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(stage / "mindie-wheels-bundle.tar.gz", "r:gz") as tar:
        tar.extractall(wheels_dir, filter="data")
    uv_bin = shutil.which("uv", path=e.get("PATH")) or "uv"
    base_py = Path(sys.base_prefix) / ("python.exe" if os.name == "nt" else "bin/python")
    if not base_py.is_file():
        base_py = Path(sys.executable)
    venv_dir = gen_dir / ".venv"
    venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    subprocess.run([uv_bin, "venv", "--python", str(base_py), str(venv_dir)], env=e, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    req = wheels_dir / "requirements.txt"
    cmd = [uv_bin, "--native-tls", "pip", "install", "--python", str(venv_py), "--find-links", str(wheels_dir), "--override", str(req), "-r", str(req)]
    if offline or e.get("UV_OFFLINE") == "1":
        cmd.extend(["--offline", "--no-index"])
    subprocess.run(cmd, env=e, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    shutil.rmtree(wheels_dir, ignore_errors=True)
    (gen_dir / "runtime.json").write_text(json.dumps({
        "schema": "mindie-candidate-runtime/1", "python": str(venv_py),
        "validation": {"status": "validated", "validated_at": time.time() + 500.0},
    }), encoding="utf-8")
    cfg = json.loads(profile.config_path.read_text(encoding="utf-8")) if profile.config_path.is_file() else {}
    cc = dict(cfg.get("component_commands") or {})
    for k, argv in list(cc.items()):
        if isinstance(argv, list) and len(argv) >= 3 and "-m" in argv:
            m_idx = argv.index("-m")
            if m_idx + 1 < len(argv) and str(argv[m_idx + 1]).startswith(("mindie_", "remote_dev", "npu_top")):
                cc[k] = [str(venv_py), *argv[1:]]
    cfg.update({
        "schema": "mindie-host-config/1", "host": profile.host, "python": str(venv_py),
        "component_release": str(gen_dir), "component_commands": cc,
        "product_validation": {"schema": "mindie-candidate-check/1", "status": "validated", "python": str(venv_py), "components": chan.get("components", {})},
    })
    profile.config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg = profile.config_path.with_name(f".{profile.config_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_cfg, profile.config_path)
    maintain_generations(profile)
    return {"status": "active", "generation": str(gen_dir), "python": str(venv_py)}
"""
    cli_py = r"""from __future__ import annotations
import argparse, json, os, subprocess
from mindie_harness.paths import ProfileRef
from mindie_harness.runtime.knowledge_binding import ensure_knowledge_binding
from mindie_harness.runtime.wheelhouse import update_runtime

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    inst = sub.add_parser("install")
    inst.add_argument("--host", required=True)
    inst.add_argument("--channel-base-url", required=True)
    inst.add_argument("--offline", action="store_true")
    args = p.parse_args(argv)
    if os.environ.get("MINDIE_TEST_CALL_GIT") == "1":
        rc = subprocess.run(["git", "status"], check=False).returncode
        if rc != 0:
            print(json.dumps({"status": "error", "reason": f"git_exit_{rc}"}))
            return 1
    prof = ProfileRef.resolve(args.host)
    up = update_runtime(prof, force=True, offline=args.offline, channel_base_url=args.channel_base_url)
    ensure_knowledge_binding(prof, python_executable=up["python"])
    print(json.dumps({"status": "installed", "configured_hosts": [args.host], "python": up["python"]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
    return {
        "__init__.py": "",
        "paths.py": paths_py,
        "cli.py": cli_py,
        "__main__.py": "from mindie_harness.cli import main\nraise SystemExit(main())\n",
        "mcp/__init__.py": "",
        "mcp/gateway.py": gateway_py,
        "runtime/__init__.py": "",
        "runtime/knowledge_binding.py": kb_py,
        "runtime/activation.py": activation_py,
        "runtime/wheelhouse.py": wheelhouse_py,
    }


def _mini_knowledge_files() -> dict[str, str]:
    units_py = r"""from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GradedUnit:
    metadata: dict
    body: str
    path: str = ""

    @property
    def id(self) -> str:
        return self.metadata["id"]

    @classmethod
    def from_parts(cls, frontmatter: dict, body: str, path: str = "") -> "GradedUnit":
        meta = dict(frontmatter)
        meta.setdefault("revision", 1)
        return cls(meta, body.strip() + "\n", path)
"""
    store_py = r"""from __future__ import annotations
import json, sqlite3, uuid
from pathlib import Path

class _GradedRepo:
    def __init__(self, store: "Store") -> None:
        self.store = store

    def install(self, units, *, origin_tier=None, **_kw):
        rows = [{"id": u.id, "title": u.metadata["title"], "body": u.body, "origin_tier": origin_tier} for u in units]
        with self.store.conn:
            self.store.conn.execute("INSERT OR REPLACE INTO kv VALUES ('graded', ?)", (json.dumps(rows),))
        return {"status": "installed", "units": len(rows)}

    def query(self, q: str) -> list[dict]:
        row = self.store.conn.execute("SELECT val FROM kv WHERE key='graded'").fetchone()
        if not row:
            return [{"id": "unit-default-000", "title": f"{self.store.domain}/default-aclgraph", "body": "Default offline unit"}]
        return json.loads(row[0])

class Store:
    def __init__(self, root: Path, domain: str = "vllm-ascend") -> None:
        self.root = Path(root) / domain
        self.domain = domain
        self.root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.root / "state-v4.sqlite3")
        with self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, val TEXT)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY, done INTEGER, payload TEXT)")
        self.graded = _GradedRepo(self)

    def enqueue_capture(self, event: dict) -> str:
        cid = f"cap-{uuid.uuid4().hex[:10]}"
        with self.conn:
            self.conn.execute("INSERT INTO captures VALUES (?, 0, ?)", (cid, json.dumps(event)))
        return cid

    def due_capture(self) -> dict | None:
        row = self.conn.execute("SELECT id, payload FROM captures WHERE done=0 LIMIT 1").fetchone()
        return {"id": row[0], "event": json.loads(row[1])} if row else None

    def mark_done(self, cid: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE captures SET done=1 WHERE id=?", (cid,))

    def close(self) -> None:
        self.conn.close()
"""
    act_py = r"""from __future__ import annotations
from contextlib import closing
import sqlite3, uuid
from pathlib import Path

class Admission:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS leases(session_id TEXT PRIMARY KEY, project_root TEXT, token TEXT)")

    def activate(self, session_id: str, *, project_root: str, root_session: str = "", not_before: float = 0.0) -> dict:
        tok = f"lease-{uuid.uuid4().hex[:12]}"
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO leases VALUES (?, ?, ?)", (session_id, project_root, tok))
        return {"session_id": session_id, "token": tok, "entry_enabled": True}
"""
    cli_py = r"""from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

def config_at(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_transcript_adapter(config: dict):
    ap = Path(config["transcript_adapter"])
    spec = importlib.util.spec_from_file_location("mindie_transcript_adapter", ap)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
"""
    handoff_py = r"""from __future__ import annotations
from pathlib import Path
from mindie_knowledge.loop.cli import config_at
from mindie_knowledge.loop.store import Store

def accept_stop(engine_path: str | Path, event: dict, **_kw) -> dict:
    ecfg = config_at(engine_path)
    st = Store(Path(ecfg["root"]), domain=ecfg["domain"])
    try:
        cid = st.enqueue_capture(event)
    finally:
        st.close()
    return {"stage": "accepted", "capture_id": cid}
"""
    engine_py = r"""from __future__ import annotations
import json, os, subprocess
from pathlib import Path

class Engine:
    def __init__(self, store, *, settings_path=None, admission=None, transcript_adapter=None, transcript_adapter_path=None, capture_mode="public-transcript", redactor_executable=None, **_kw):
        self.store = store
        self.transcript = transcript_adapter
        self.redactor_executable = redactor_executable

    def _redact_with_scanner(self, text: str) -> str:
        if not self.redactor_executable or not Path(self.redactor_executable).is_file():
            raise RuntimeError(f"Missing redactor_executable: {self.redactor_executable}")
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
        proc = subprocess.run(
            [self.redactor_executable, "stdin", "--report-format", "json", "--report-path", "-", "--exit-code", "0"],
            input=text, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, **kw
        )
        findings = json.loads(proc.stdout or "[]")
        out = text
        for item in findings:
            sec = item.get("Secret")
            if isinstance(sec, str) and sec:
                out = out.replace(sec, f"<redacted:{item.get('RuleID', 'secret')}>")
        return out

    def _process(self, ident: dict) -> None:
        ev = ident["event"]
        tp = self.transcript.resolve_transcript(ev)
        mat = self.transcript.read_material(tp, session_id=ev.get("session_id"))
        clean = self._redact_with_scanner(mat["text"])
        mdir = self.store.root / "materials"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / f"{ident['id']}.txt").write_text(clean, encoding="utf-8")
        self.store.mark_done(ident["id"])
"""
    return {
        "__init__.py": "",
        "graded/__init__.py": "",
        "graded/units.py": units_py,
        "loop/__init__.py": "",
        "loop/store.py": store_py,
        "loop/activation.py": act_py,
        "loop/cli.py": cli_py,
        "loop/handoff.py": handoff_py,
        "loop/engine.py": engine_py,
        "loop/transcript_redaction.py": "def install_scanner(cache): raise RuntimeError('offline test uses explicit redactor')\n",
    }


def _build_stage(stage_dir: Path, *, harness_version: str = "0.1.5") -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    wheels_tmp = stage_dir / "_build_wheels"
    wheels_tmp.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    req_lines: list[str] = []
    inner_sums: list[str] = []
    components: dict[str, dict[str, str]] = {}

    for dist_name, pkg_name, ver in WHEEL_SPECS:
        eff_ver = harness_version if dist_name == "mindie-harness" else ver
        if dist_name == "mindie-harness":
            files = _mini_harness_files()
        elif dist_name == "mindie-knowledge":
            files = _mini_knowledge_files()
        else:
            files = {
                "__init__.py": f"__version__ = {eff_ver!r}\n",
                "__main__.py": "import json, sys; print(json.dumps({'status': 'ok', 'argv': sys.argv[1:]}))\n",
            }
        wpath, wsha = _build_wheel(wheels_tmp, dist_name, pkg_name, eff_ver, files)
        manifest_rows.append({"package": dist_name, "version": eff_ver, "filename": wpath.name, "sha256": wsha})
        req_lines.append(f"{dist_name}=={eff_ver}")
        inner_sums.append(f"{wsha}  {wpath.name}\n")
        if dist_name != "reme-ai":
            components[dist_name] = {"package": dist_name, "version": eff_ver, "revision": "a" * 40, "role": "shared_component"}

    (wheels_tmp / "requirements.txt").write_text("\n".join(req_lines) + "\n", encoding="utf-8")
    (wheels_tmp / "bundle-manifest.json").write_text(
        json.dumps({"schema": "mindie-wheels-bundle/1", "wheels": manifest_rows}, indent=2) + "\n", encoding="utf-8"
    )
    (wheels_tmp / "SHA256SUMS").write_text("".join(inner_sums), encoding="utf-8")

    bundle_tar = stage_dir / "mindie-wheels-bundle.tar.gz"
    with tarfile.open(bundle_tar, "w:gz") as tar:
        for fpath in sorted(wheels_tmp.iterdir()):
            tar.add(fpath, arcname=fpath.name)
    shutil.rmtree(wheels_tmp)

    chan_path = stage_dir / "release-channel.json"
    chan_path.write_text(
        json.dumps(
            {
                "schema": "mindie-release-channel/1",
                "current_patch_version": f"0.1.{18 if harness_version == '0.1.5' else 17}",
                "components": components,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    b_sha = hashlib.sha256(bundle_tar.read_bytes()).hexdigest()
    c_sha = hashlib.sha256(chan_path.read_bytes()).hexdigest()
    (stage_dir / "SHA256SUMS").write_text(
        f"{b_sha}  mindie-wheels-bundle.tar.gz\n{c_sha}  release-channel.json\n", encoding="utf-8"
    )
    return stage_dir


def _make_seed_bundle(assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    seed_main = r"""import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--host", required=True)
p.add_argument("--plugin-root", required=True)
p.add_argument("--config", default=None)
a, rest = p.parse_known_args()

droot = Path(os.environ["MINDIE_DATA_HOME"])
droot.mkdir(parents=True, exist_ok=True)
lock_file = droot / f".prepare-{a.host}.lock"
deadline = time.monotonic() + 30.0
fd = None
while fd is None:
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"seed lock timeout: {lock_file}")
        time.sleep(0.02)

kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
try:
    gdir = droot / "runtimes" / a.host / "gen-seed"
    vdir = gdir / ".venv"
    vpy = vdir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    rc_file = gdir / "runtime.json"
    if not rc_file.is_file():
        uv = shutil.which("uv") or "uv"
        subprocess.run([uv, "venv", "--python", sys.executable, str(vdir)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
        links = os.environ["UV_FIND_LINKS"]
        req = str(Path(links) / "requirements.txt")
        subprocess.run([uv, "pip", "install", "--python", str(vpy), "--find-links", links, "--offline", "--no-index", "-r", req], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
        rc_file.write_text(json.dumps({"schema": "mindie-candidate-runtime/1", "python": str(vpy), "validation": {"status": "validated", "validated_at": time.time()}}), encoding="utf-8")
    code = (
        "import sys; from pathlib import Path; from mindie_harness.paths import ProfileRef; "
        "from mindie_harness.runtime.knowledge_binding import ensure_knowledge_binding; "
        "pr = ProfileRef.resolve(sys.argv[1], plugin_root=sys.argv[2], config_path=sys.argv[3] or None); "
        "ensure_knowledge_binding(pr, plugin_root=Path(sys.argv[2]), python_executable=sys.executable)"
    )
    subprocess.run([str(vpy), "-I", "-X", "utf8", "-c", code, a.host, a.plugin_root, a.config or ""], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
finally:
    os.close(fd)
    try:
        lock_file.unlink(missing_ok=True)
    except OSError:
        pass

if rest:
    role = rest[-1]
    surf = {"remote-mcp": "remote", "tools-mcp": "tools", "knowledge-mcp": "knowledge"}.get(role, role)
    sc = "import sys; from mindie_harness.mcp.gateway import serve_stdio; raise SystemExit(serve_stdio(sys.argv[1], sys.argv[2], sys.argv[3]))"
    raise SystemExit(subprocess.run([str(vpy), "-I", "-X", "utf8", "-c", sc, a.host, surf, a.plugin_root], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, **kw).returncode)
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__main__.py", seed_main)
    pyz_bytes = b"#!/usr/bin/env python3\n" + buf.getvalue()
    pyz = assets_dir / "mindie-harness-seed.pyz"
    pyz.write_bytes(pyz_bytes)
    (assets_dir / "seed-lock.json").write_text(
        json.dumps({"schema": "mindie-harness-seed/1", "sha256": hashlib.sha256(pyz_bytes).hexdigest(), "size_bytes": len(pyz_bytes)}),
        encoding="utf-8",
    )


def _transcript_adapter_source(host: str) -> str:
    body = r"""from __future__ import annotations
from contextlib import closing
from dataclasses import dataclass
import json, os, sqlite3, struct
from pathlib import Path

HOST = __HOST__

@dataclass(frozen=True)
class FileIdentity:
    path: str
    size: int

def identify(path):
    p = Path(path)
    return FileIdentity(str(p), p.stat().st_size if p.is_file() else 0)

def _unzstd(raw: bytes) -> str:
    if raw[:5] == b"\x28\xb5\x2f\xfd\xa0":
        sz = struct.unpack_from("<I", raw, 5)[0]
        return raw[12 : 12 + sz].decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def resolve_transcript(event: dict) -> Path:
    if event.get("transcript_path") and Path(event["transcript_path"]).is_file():
        return Path(event["transcript_path"])
    sid = str(event.get("session_id") or event.get("conversationId") or event.get("sessionID") or "")
    if HOST == "kimi":
        return Path(os.environ["KIMI_CODE_HOME"]) / f"sessions/workdir/{sid}/agents/main/wire.jsonl"
    if HOST == "dsh":
        return Path(os.environ["DSH_HOME"]) / f"sessions/proj/{sid}/session.v4.jsonl.zstd"
    if HOST == "agy":
        return Path(os.environ["AGY_APP_DATA_DIR"]) / f"brain/{sid}/.system_generated/logs/transcript_full.jsonl"
    if HOST == "oc":
        return Path(event.get("opencodeDb") or os.environ["MINDIE_OPENCODE_DB"])
    raise FileNotFoundError(f"Transcript missing for {HOST}: {event}")

def read_material(path, start: int = 0, *, session_id=None, **_kw) -> dict:
    p = Path(path)
    if HOST == "oc" or p.suffix == ".db":
        with closing(sqlite3.connect(p)) as conn:
            rows = [r[0] for r in conn.execute("SELECT data FROM session_part WHERE session_id=? ORDER BY ordinal ASC", (session_id,))]
        txt = "\n".join(json.loads(r).get("text", "") for r in rows)
    elif p.name.endswith(".zstd"):
        txt = _unzstd(p.read_bytes())
    else:
        txt = p.read_text(encoding="utf-8", errors="replace")
    return {"status": "ok", "start": 0, "end": len(txt), "text": txt, "records": 2}
"""
    return body.replace("__HOST__", repr(host))


def _shared_launcher_source(host: str) -> str:
    body = r"""from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

HOST = __HOST__
SUPPRESS = __SUPPRESS__
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
KW = {"creationflags": 0x08000000} if os.name == "nt" else {}

def _cfg_path() -> Path:
    return Path(os.environ["MINDIE_CONFIG_HOME"]) / f"{HOST}.json"

def _state_dir() -> Path:
    return Path(os.environ["MINDIE_DATA_HOME"]) / "state" / HOST

def _emit(doc: dict) -> int:
    out = {"suppressOutput": True, **doc} if SUPPRESS and not doc else doc
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0

def _in_scope(ev: dict, roots: list[str]) -> bool:
    cwd = next((ev.get(k) for k in ("cwd", "workspaceRoot", "directory", "workspace") if isinstance(ev.get(k), str) and os.path.isabs(ev[k])), None)
    if not cwd:
        return False
    c = os.path.normcase(os.path.realpath(cwd))
    for r in roots:
        rn = os.path.normcase(os.path.realpath(r))
        if os.path.splitdrive(c)[0] == os.path.splitdrive(rn)[0] and os.path.commonpath([c, rn]) == rn:
            return True
    return False

def _ensure_py() -> str:
    cp = _cfg_path()
    if cp.is_file():
        py = json.loads(cp.read_text(encoding="utf-8")).get("python")
        if isinstance(py, str) and Path(py).is_file():
            return py
    seed = PLUGIN_ROOT / "assets/mindie-harness-seed.pyz"
    subprocess.run([sys.executable, "-I", "-X", "utf8", str(seed), "--host", HOST, "--plugin-root", str(PLUGIN_ROOT), "--config", str(cp)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **KW)
    return json.loads(cp.read_text(encoding="utf-8"))["python"]

def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    while len(args) >= 2 and args[0] in {"--config", "--release-version", "--release", "--source"}:
        args = args[2:]
    if not args:
        return _emit({})
    role = args[0]
    if role in {"session-start", "user-prompt-submit"} or (role == "hook" and args[1:2] in (["session-start"], ["user-prompt-submit"], ["pre-invocation"], ["pre-tool-use"])):
        act = role if role != "hook" else args[1]
        if act == "session-start":
            sd = _state_dir(); sd.mkdir(parents=True, exist_ok=True)
            (sd / "pending-update-check.json").write_text(json.dumps({"schema": "mindie-pending-update/1", "host": HOST, "requested_at": time.time()}) + "\n", encoding="utf-8")
        return _emit({})
    if role == "stop" or (role == "hook" and args[1:2] == ["stop"]):
        if "--event-json" in args:
            ev = json.loads(args[args.index("--event-json") + 1])
        else:
            raw = sys.stdin.buffer.read(1048576)
            ev = json.loads(raw.decode("utf-8-sig")) if raw.strip() else {}
        cdir = _cfg_path().parent
        comm_f, cons_f = cdir / "mindie-community.json", cdir / "mindie-consent.json"
        if not comm_f.is_file() or not cons_f.is_file():
            return _emit({})
        comm, cons = json.loads(comm_f.read_text(encoding="utf-8")), json.loads(cons_f.read_text(encoding="utf-8"))
        if comm.get("enabled") is not True or cons.get("choice") not in {"contribute", "local-only"} or not _in_scope(ev, comm.get("project_roots") or []):
            return _emit({})
        py = _ensure_py()
        code = (
            "import json, sys; from pathlib import Path; from mindie_harness.paths import ProfileRef; "
            "from mindie_harness.runtime.knowledge_binding import run_stop_capture; "
            "pr = ProfileRef.resolve(sys.argv[1], plugin_root=sys.argv[2]); "
            "run_stop_capture(pr, json.loads(sys.argv[3]), plugin_root=Path(sys.argv[2]))"
        )
        subprocess.run([py, "-I", "-X", "utf8", "-c", code, HOST, str(PLUGIN_ROOT), json.dumps(ev)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **KW)
        return _emit({})
    if role == "mcp" or role in {"remote-mcp", "tools-mcp", "knowledge-mcp"}:
        surf = args[1] if role == "mcp" else role.replace("-mcp", "")
        inc_k = "0" if ("--no-knowledge" in args or (HOST == "wb" and surf == "tools")) else "1"
        py = _ensure_py()
        code = "import sys; from mindie_harness.mcp.gateway import serve_stdio; raise SystemExit(serve_stdio(sys.argv[1], sys.argv[2], sys.argv[3], include_knowledge=(False if sys.argv[4]=='0' else None)))"
        return subprocess.run([py, "-I", "-X", "utf8", "-c", code, HOST, surf, str(PLUGIN_ROOT), inc_k], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, **KW).returncode
    return _emit({})

if __name__ == "__main__":
    raise SystemExit(main())
"""
    return body.replace("__HOST__", repr(host)).replace("__SUPPRESS__", "True" if host == "wb" else "False")


def _encode_ps(script: str) -> str:
    return "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + base64.b64encode(script.encode("utf-16le")).decode("ascii")


def _build_seven_host_plugins(plugins_root: Path) -> Path:
    plugins_root.mkdir(parents=True, exist_ok=True)
    adapter_names = {
        "codex": "codex_transcript.py",
        "cc": "transcript.py",
        "kimi": "kimi_transcript.py",
        "dsh": "dsh_transcript.py",
        "agy": "agy_transcript.py",
        "wb": "wb_transcript.py",
        "oc": "opencode_transcript.py",
    }

    # 1. codex
    codex_sub = plugins_root / "mindie-agent-codex/plugins/mindie-agent"
    (codex_sub / "scripts").mkdir(parents=True, exist_ok=True)
    (codex_sub / "hooks").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(codex_sub / "assets")
    (codex_sub / "scripts/hook_launcher.py").write_text(_shared_launcher_source("codex"), encoding="utf-8")
    (codex_sub / "scripts/entry.py").write_text(_shared_launcher_source("codex"), encoding="utf-8")
    (codex_sub / f"scripts/{adapter_names['codex']}").write_text(_transcript_adapter_source("codex"), encoding="utf-8")
    (codex_sub / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mindie-remote-dev": {"command": "python", "args": ["-I", "-X", "utf8", "scripts/hook_launcher.py", "remote-mcp"]},
                    "mindie-tools": {"command": "python", "args": ["-I", "-X", "utf8", "scripts/hook_launcher.py", "tools-mcp"]},
                }
            }
        ),
        encoding="utf-8",
    )
    ps_ss = '& python -I -X utf8 "$env:PLUGIN_ROOT/scripts/hook_launcher.py" session-start; exit $LASTEXITCODE'
    ps_up = '& python -I -X utf8 "$env:PLUGIN_ROOT/scripts/hook_launcher.py" user-prompt-submit; exit $LASTEXITCODE'
    ps_st = '$in = [Console]::In.ReadToEnd(); $in | & python -I -X utf8 "$env:PLUGIN_ROOT/scripts/hook_launcher.py" stop; exit $LASTEXITCODE'
    (codex_sub / "hooks/hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [{"hooks": [{"type": "command", "command": 'python3 "$PLUGIN_ROOT/scripts/hook_launcher.py" session-start', "commandWindows": _encode_ps(ps_ss)}]}],
                    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": 'python3 "$PLUGIN_ROOT/scripts/hook_launcher.py" user-prompt-submit', "commandWindows": _encode_ps(ps_up)}]}],
                    "Stop": [{"hooks": [{"type": "command", "command": 'python3 "$PLUGIN_ROOT/scripts/hook_launcher.py" stop', "commandWindows": _encode_ps(ps_st)}]}],
                    "Interrupt": [
                        {
                            "hooks": [
                                {"type": "mcp_tool", "server": "mindie-remote-dev", "tool": "mindie_interrupt"},
                                {"type": "mcp_tool", "server": "mindie-tools", "tool": "mindie_interrupt"},
                            ]
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    # 2. cc
    cc_root = plugins_root / "mindie-agent-cc"
    (cc_root / "scripts").mkdir(parents=True, exist_ok=True)
    (cc_root / "hooks").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(cc_root / "assets")
    (cc_root / "scripts/plugin_entry.py").write_text(_shared_launcher_source("cc"), encoding="utf-8")
    (cc_root / f"scripts/{adapter_names['cc']}").write_text(_transcript_adapter_source("cc"), encoding="utf-8")
    (cc_root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    s: {"command": "python3", "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/plugin_entry.py", "--release-version", "0.1.32", "mcp", s]}
                    for s in ("knowledge", "remote", "components")
                }
            }
        ),
        encoding="utf-8",
    )
    (cc_root / "hooks/hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    ev: [{"hooks": [{"type": "command", "command": f'python3 "${{CLAUDE_PLUGIN_ROOT}}/scripts/plugin_entry.py" hook {act}'}]}]
                    for ev, act in (("PreToolUse", "pre-tool-use"), ("SessionStart", "session-start"), ("Stop", "stop"))
                }
            }
        ),
        encoding="utf-8",
    )

    # 3. kimi
    kimi_root = plugins_root / "mindie-agent-kimi"
    (kimi_root / "scripts").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(kimi_root / "assets")
    (kimi_root / "scripts/mindie_launch.py").write_text(_shared_launcher_source("kimi"), encoding="utf-8")
    (kimi_root / f"scripts/{adapter_names['kimi']}").write_text(_transcript_adapter_source("kimi"), encoding="utf-8")
    (kimi_root / "kimi.plugin.json").write_text(
        json.dumps(
            {
                "name": "mindie-agent",
                "mcpServers": {
                    s: {"command": "python3", "args": ["./scripts/mindie_launch.py", "mcp", s]}
                    for s in ("knowledge", "remote", "components")
                },
                "hooks": [
                    {"event": "PreToolUse", "command": "python3 ./scripts/mindie_launch.py hook pre-tool-use"},
                    {"event": "Stop", "command": "python3 ./scripts/mindie_launch.py hook stop"},
                ],
            }
        ),
        encoding="utf-8",
    )

    # 4. dsh
    dsh_root = plugins_root / "mindie-agent-dsh"
    (dsh_root / "scripts").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(dsh_root / "assets")
    (dsh_root / "scripts/mindie_launch.py").write_text(_shared_launcher_source("dsh"), encoding="utf-8")
    (dsh_root / f"scripts/{adapter_names['dsh']}").write_text(_transcript_adapter_source("dsh"), encoding="utf-8")
    (dsh_root / "package.json").write_text(
        json.dumps({"name": "@mindie-agent/dsh", "version": "0.1.28", "type": "module", "files": ["host.mjs", "host3.mjs", "scripts", "assets"]}),
        encoding="utf-8",
    )
    (dsh_root / "host3.mjs").write_text(
        r"""import { spawn } from 'node:child_process'; import fs from 'node:fs'; import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
export async function apply(ctx) {
  ctx.on('agent/created', async () => {});
  ctx.on('agent/pre-step', (_ev, next) => next());
  ctx.on('agent/turn-stopping', ({ agent, turn }) => {
    const sid = agent?.session?.id, cwd = agent?.session?.header?.cwd;
    if (!sid || !cwd) return;
    const tp = path.join(process.env.DSH_HOME || '', 'sessions', 'proj', sid, 'session.v4.jsonl.zstd');
    if (!fs.existsSync(tp)) return;
    const child = spawn('python3', [path.join(HERE, 'scripts', 'mindie_launch.py'), 'hook', 'stop'], { cwd: HERE, env: process.env, windowsHide: true, stdio: ['pipe', 'ignore', 'ignore'] });
    child.stdin.end(JSON.stringify({ hook_event_name: 'Stop', identity_kind: 'turn', session_id: sid, turn_id: String(turn), transcript_path: tp, cwd }) + '\n');
    child.unref();
  });
  ctx.effect(() => () => {});
}
export default { name: 'mindie-dsh-host3', apply };
""",
        encoding="utf-8",
    )
    (dsh_root / "host.mjs").write_text(
        r"""import path from 'node:path'; import { fileURLToPath } from 'node:url'; import * as host3 from './host3.mjs';
const HERE = path.dirname(fileURLToPath(import.meta.url));
export async function apply(ctx, cfg = {}) {
  const mcpMod = await import(cfg.mcpPluginUrl);
  const launch = path.join(HERE, 'scripts', 'mindie_launch.py');
  await ctx.plugin(mcpMod, { serverName: 'remote-dev', command: 'python3', args: [launch, 'mcp', 'remote'], cwd: HERE, env: { ...process.env } });
  await ctx.plugin(mcpMod, { serverName: 'mindie-tools', command: 'python3', args: [launch, 'mcp', 'tools'], cwd: HERE, env: { ...process.env } });
  await ctx.plugin(host3, {});
}
export default { name: 'mindie-dsh', apply };
""",
        encoding="utf-8",
    )

    # 5. agy
    agy_sub = plugins_root / "mindie-agent-agy/plugins/mindie-agent"
    (agy_sub / "scripts").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(agy_sub / "assets")
    (agy_sub / "scripts/mindie_launch.py").write_text(_shared_launcher_source("agy"), encoding="utf-8")
    (agy_sub / f"scripts/{adapter_names['agy']}").write_text(_transcript_adapter_source("agy"), encoding="utf-8")
    (agy_sub / "mcp_config.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mindie-remote-dev": {"command": "python3", "args": ["scripts/mindie_launch.py", "mcp", "remote"], "cwd": ".", "env": {"AGY_PLUGIN_ROOT": "."}},
                    "mindie-tools": {"command": "python3", "args": ["scripts/mindie_launch.py", "mcp", "tools"], "cwd": ".", "env": {"AGY_PLUGIN_ROOT": "."}},
                }
            }
        ),
        encoding="utf-8",
    )
    (agy_sub / "hooks.json").write_text(
        json.dumps(
            {
                "mindie-agent": {
                    ev: [{"command": f"python3 scripts/mindie_launch.py hook {act}"}]
                    for ev, act in (("SessionStart", "session-start"), ("PreInvocation", "pre-invocation"), ("PreToolUse", "pre-tool-use"), ("Stop", "stop"))
                }
            }
        ),
        encoding="utf-8",
    )

    # 6. wb
    wb_sub = plugins_root / "mindie-agent-wb/plugins/mindie-agent"
    (wb_sub / "scripts").mkdir(parents=True, exist_ok=True)
    (wb_sub / "hooks").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(wb_sub / "assets")
    (wb_sub / "scripts/mindie_launch.py").write_text(_shared_launcher_source("wb"), encoding="utf-8")
    (wb_sub / f"scripts/{adapter_names['wb']}").write_text(_transcript_adapter_source("wb"), encoding="utf-8")
    for fname, root_var in ((".mcp.json", "CODEBUDDY_PLUGIN_ROOT"), ("mcp.json", "PLUGIN_ROOT")):
        (wb_sub / fname).write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "mindie-remote-dev": {"command": "python3", "args": [f"${{{root_var}}}/scripts/mindie_launch.py", "mcp", "remote"]},
                        "mindie-tools": {"command": "python3", "args": [f"${{{root_var}}}/scripts/mindie_launch.py", "mcp", "tools", "--no-knowledge"]},
                        "mindie-knowledge": {"command": "python3", "args": [f"${{{root_var}}}/scripts/mindie_launch.py", "mcp", "knowledge"]},
                    }
                }
            ),
            encoding="utf-8",
        )
    (wb_sub / "hooks/hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    ev: [{"hooks": [{"type": "command", "command": f'python3 "${{CODEBUDDY_PLUGIN_ROOT}}/scripts/mindie_launch.py" hook {act}'}]}]
                    for ev, act in (("SessionStart", "session-start"), ("UserPromptSubmit", "user-prompt-submit"), ("PreToolUse", "pre-tool-use"), ("Stop", "stop"))
                }
            }
        ),
        encoding="utf-8",
    )
    (wb_sub / "scripts/wb_install.py").write_text(
        r"""from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path
def install_workbuddy_plugin(*, wb_dir, source_root):
    src, dst = Path(source_root).resolve(), Path(wb_dir).resolve() / "plugins/cache/mindie-agent"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    cfg_f = Path(os.environ.get("MINDIE_CONFIG_HOME", ".")) / "wb.json"
    py = json.loads(cfg_f.read_text(encoding="utf-8")).get("python") if cfg_f.is_file() else sys.executable
    py = py if isinstance(py, str) and Path(py).is_file() else sys.executable
    mcp = json.loads((dst / "mcp.json").read_text(encoding="utf-8"))
    for srv in mcp["mcpServers"].values():
        srv["command"] = py
    (dst / "mcp.json").write_text(json.dumps(mcp, indent=2), encoding="utf-8")
    return {"status": "installed", "cache_install_path": str(dst), "python": py}
""",
        encoding="utf-8",
    )

    # 7. oc
    oc_root = plugins_root / "mindie-agent-oc"
    oc_sub = oc_root / "plugins/mindie-agent"
    (oc_sub / "scripts").mkdir(parents=True, exist_ok=True)
    _make_seed_bundle(oc_sub / "assets")
    (oc_sub / "plugin.json").write_text(json.dumps({"name": "mindie-agent", "version": "0.1.12"}), encoding="utf-8")
    (oc_sub / "scripts/mindie_launch.py").write_text(_shared_launcher_source("oc"), encoding="utf-8")
    (oc_sub / f"scripts/{adapter_names['oc']}").write_text(_transcript_adapter_source("oc"), encoding="utf-8")
    (oc_root / "package.json").write_text(
        json.dumps({"name": "@mindie-agent/oc", "version": "0.1.12", "type": "module", "files": ["index.mjs", "plugins"]}),
        encoding="utf-8",
    )
    (oc_sub / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mindie-remote-dev": {"command": "python3", "args": ["${OC_PLUGIN_ROOT}/scripts/mindie_launch.py", "mcp", "remote"]},
                    "mindie-tools": {"command": "python3", "args": ["${OC_PLUGIN_ROOT}/scripts/mindie_launch.py", "mcp", "tools"]},
                }
            }
        ),
        encoding="utf-8",
    )
    (oc_root / "index.mjs").write_text(
        r"""import { spawn } from 'node:child_process'; import path from 'node:path'; import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SUB = path.join(HERE, 'plugins', 'mindie-agent');
const LAUNCH = path.join(SUB, 'scripts', 'mindie_launch.py');
export default { id: 'mindie-agent', async setup(ctx) {
  let disposed = false;
  await ctx.mcp?.transform?.((ed) => {
    ed.set('mindie-remote-dev', { type: 'local', command: ['python3', LAUNCH, 'mcp', 'remote'], environment: { OC_PLUGIN_ROOT: SUB } });
    ed.set('mindie-tools', { type: 'local', command: ['python3', LAUNCH, 'mcp', 'tools'], environment: { OC_PLUGIN_ROOT: SUB } });
  });
  ctx.session?.hook?.('context', async () => {});
  ctx.tool?.hook?.('execute.before', () => {});
  ctx.shell?.hook?.('create.before', () => {});
  const stream = await ctx.event?.subscribe?.();
  if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
    (async () => {
      for await (const ev of stream) {
        if (disposed || ev?.type !== 'session.idle') continue;
        const sid = ev?.data?.sessionID;
        const info = await ctx.session.get({ sessionID: sid });
        const payload = { sessionID: sid, directory: info?.directory, opencodeDb: process.env.MINDIE_OPENCODE_DB };
        const ch = spawn('python3', [LAUNCH, 'hook', 'stop', '--event-json', JSON.stringify(payload)], { cwd: SUB, env: { ...process.env, OC_PLUGIN_ROOT: SUB }, windowsHide: true, stdio: 'ignore' });
        ch.unref();
      }
    })();
  }
  return async () => { disposed = true; };
} };
""",
        encoding="utf-8",
    )
    return plugins_root


def _build_redactor_shim(bin_dir: Path) -> Path:
    body = r"""import json, re, sys
text = sys.stdin.read()
patterns = [("github-pat", r"ghp_[A-Za-z0-9]{36}"), ("aws-access-token", r"AKIA[0-9A-Z]{16}")]
findings = []
for idx, line in enumerate(text.splitlines(), start=1):
    for rule_id, pat in patterns:
        for m in re.finditer(pat, line):
            findings.append({"RuleID": rule_id, "StartLine": idx, "EndLine": idx, "Secret": m.group(0), "Match": m.group(0)})
sys.stdout.write(json.dumps(findings))
"""
    return write_executable_shim(bin_dir, "gitleaks-shim", body)


def _resolve_or_write_install_py(dest: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    for cand in (repo_root.parent / "mindie-agent/install.py", repo_root.parent.parent / "mindie-agent/install.py"):
        if cand.is_file():
            target = dest / "install.py"
            shutil.copy2(cand, target)
            return target
    fallback = dest / "install.py"
    fallback.write_text(
        r"""import argparse, json, os, shutil, subprocess, sys, tarfile, tempfile, urllib.parse, urllib.request
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--host', required=True); p.add_argument('--channel-base-url', required=True)
p.add_argument('--offline', action='store_true'); p.add_argument('--json', action='store_true')
a = p.parse_args()
src = Path(urllib.request.url2pathname(urllib.parse.urlparse(a.channel_base_url).path)) if a.channel_base_url.startswith('file://') else Path(a.channel_base_url)
with tempfile.TemporaryDirectory() as td:
    t = Path(td); w = t / 'wheels'; w.mkdir()
    with tarfile.open(src / 'mindie-wheels-bundle.tar.gz', 'r:gz') as tar: tar.extractall(w, filter='data')
    uv = shutil.which('uv') or 'uv'; bv = t / 'boot'
    bpy = bv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    kw = {'creationflags': 0x08000000} if os.name == 'nt' else {}
    subprocess.run([uv, 'venv', '--python', sys.executable, str(bv)], check=True, **kw)
    hw = next(w.glob('mindie_harness-*.whl'))
    subprocess.run([uv, 'pip', 'install', '--python', str(bpy), '--offline', '--no-index', '--no-deps', str(hw)], check=True, **kw)
    out = subprocess.check_output([str(bpy), '-I', '-X', 'utf8', '-m', 'mindie_harness.cli', 'install', '--host', a.host, '--channel-base-url', str(src), *(['--offline'] if a.offline else [])], text=True, **kw)
    doc = json.loads(out.strip().splitlines()[-1])
    print(json.dumps({'schema': 'mindie-install/1', 'status': doc.get('status', 'error'), 'configured_hosts': doc.get('configured_hosts', [])}))
    raise SystemExit(0 if doc.get('status') == 'installed' else 1)
""",
        encoding="utf-8",
    )
    return fallback


@pytest.fixture(scope="session")
def host_accept_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("host-accept-suite").resolve()
    cand_stage = _build_stage(root / "candidate-stage", harness_version="0.1.5")
    base_stage = _build_stage(root / "baseline-stage", harness_version="0.1.4")
    plugins_root = _build_seven_host_plugins(root / "plugins")
    redactor = _build_redactor_shim(root / "redactor-bin")
    install_py = _resolve_or_write_install_py(root)
    return {
        "cand_stage": cand_stage,
        "base_stage": base_stage,
        "plugins_root": plugins_root,
        "redactor": redactor,
        "install_py": install_py,
    }


def test_parse_all_seven_host_manifests_and_real_sibling_checkouts(
    host_accept_fixture: dict[str, Path], tmp_path: Path
) -> None:
    plugins_root = host_accept_fixture["plugins_root"]
    repo_root = Path(__file__).resolve().parents[1]
    real_candidates = {
        "codex": ("codex-0.1.40", "mindie-agent-codex"),
        "cc": ("cc-0.1.32", "mindie-agent-cc"),
        "kimi": ("kimi-0.1.33", "mindie-agent-kimi"),
        "dsh": ("dsh-0.1.28", "mindie-agent-dsh"),
        "agy": ("agy-0.1.22", "mindie-agent-agy"),
        "wb": ("wb-0.1.20", "mindie-agent-wb"),
        "oc": ("oc-0.1.12", "mindie-agent-oc"),
    }
    tasks: list[tuple[str, Path, Path]] = [
        (host, plugins_root / f"mindie-agent-{host}", (tmp_path / f"fx-{host}").resolve()) for host in ALL_HOSTS
    ]
    for host, dir_names in real_candidates.items():
        found_dir = next(
            (base / dname for base in (repo_root.parent, repo_root.parent.parent) for dname in dir_names if (base / dname).is_dir()),
            None,
        )
        if found_dir is not None:
            tasks.append((host, found_dir, (tmp_path / f"real-{host}").resolve()))

    def _check(item: tuple[str, Path, Path]) -> None:
        host, pdir, sb = item
        env, _ = build_sandbox_env(sb)
        spec = parse_host_manifest(host, pdir, sandbox_root=sb, env=env)
        assert set(spec.servers) == set(EXPECTED_HOST_SERVERS[host])
        assert "SessionStart" in spec.hooks and "Stop" in spec.hooks

    with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as pool:
        for fut in [pool.submit(_check, t) for t in tasks]:
            fut.result()


def test_all_seven_hosts_four_scenarios_cli_end_to_end(
    host_accept_fixture: dict[str, Path], tmp_path: Path
) -> None:
    report_path = tmp_path / "host-accept-all.json"
    rc = bench_main(
        [
            "host-accept",
            "--all-hosts",
            "--plugins-root",
            str(host_accept_fixture["plugins_root"]),
            "--wheelhouse-stage",
            str(host_accept_fixture["cand_stage"]),
            "--redactor",
            str(host_accept_fixture["redactor"]),
            "--scenarios",
            "cold_native_launch,hooks_and_capture",
            "--timeout",
            "45",
            "--output",
            str(report_path),
        ]
    )
    assert rc == 0, report_path.read_text(encoding="utf-8") if report_path.is_file() else "no report"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "mindie-host-acceptance/1"
    assert report["status"] == "passed"
    assert report["terminal_status"] == "completed_pass"
    assert report["summary"]["total_hosts"] == 7
    assert report["summary"]["passed_hosts"] == 7
    assert report["bundle_sha256"] and len(report["bundle_sha256"]) == 64
    for host in ALL_HOSTS:
        hrow = report["hosts"][host]
        assert hrow["terminal_status"] == "completed_pass"
        for scen in ("cold_native_launch", "hooks_and_capture"):
            assert hrow["scenarios"][scen]["terminal_status"] == "completed_pass"

    # Exercise public_install_py and upgrade_cutover across Python/PS, Node ESM (dsh, oc), and dual-manifest (wb).
    upg_report = tmp_path / "host-accept-install-upgrade.json"
    rc_upg = bench_main(
        [
            "host-accept",
            "--host",
            "oc",
            "--plugins-root",
            str(host_accept_fixture["plugins_root"]),
            "--wheelhouse-stage",
            str(host_accept_fixture["cand_stage"]),
            "--baseline-wheelhouse-stage",
            str(host_accept_fixture["base_stage"]),
            "--install-script",
            str(host_accept_fixture["install_py"]),
            "--redactor",
            str(host_accept_fixture["redactor"]),
            "--scenarios",
            ",".join(ALL_SCENARIOS),
            "--timeout",
            "45",
            "--output",
            str(upg_report),
        ]
    )
    assert rc_upg == 0, upg_report.read_text(encoding="utf-8")
    upg_doc = json.loads(upg_report.read_text(encoding="utf-8"))
    assert upg_doc["terminal_status"] == "completed_pass"
    assert set(upg_doc["hosts"]["oc"]["scenarios"]) == set(ALL_SCENARIOS)
    assert len(upg_doc["hosts"]["oc"]["scenarios"]["upgrade_cutover"]["details"]["retained_generations"]) <= 2


def test_byte_zero_stdout_leak_and_git_trap_fail_closed(
    host_accept_fixture: dict[str, Path], tmp_path: Path
) -> None:
    base_py = resolve_base_python()
    env, _ = build_sandbox_env((tmp_path / "leak-sb").resolve())
    for label, leak_code in (
        ("seed_json_leak", "import sys; sys.stdout.write('{\"status\": \"provisioned\"}\\n'); sys.stdout.flush()"),
        ("banner_text_leak", "import sys; sys.stdout.write('MindIE starting...\\nContent-Length: 2\\r\\n\\r\\n{}'); sys.stdout.flush()"),
    ):
        srv = MCPServerSpec(
            name=f"leak-{label}",
            command=str(base_py),
            args=("-I", "-c", leak_code),
            cwd=tmp_path,
            env=env,
            surface="tools",
        )
        sess = MCPWireSession(srv)
        try:
            with pytest.raises(MCPProtocolError):
                sess.initialize(time.monotonic() + 5.0)
        finally:
            sess.close()

    bad_install = tmp_path / "bad_install.py"
    bad_install.write_text(
        "import subprocess, sys\nrc = subprocess.run(['git', 'rev-parse', 'HEAD'], check=False).returncode\nraise SystemExit(rc)\n",
        encoding="utf-8",
    )
    out_json = tmp_path / "git-trap-report.json"
    rc = bench_main(
        [
            "host-accept",
            "--host",
            "kimi",
            "--plugin-dir",
            str(host_accept_fixture["plugins_root"] / "mindie-agent-kimi"),
            "--wheelhouse-stage",
            str(host_accept_fixture["cand_stage"]),
            "--install-script",
            str(bad_install),
            "--scenarios",
            "public_install_py",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 1
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["terminal_status"] == "completed_fail"
    assert "99" in rep["hosts"]["kimi"]["scenarios"]["public_install_py"]["error"]["message"]


def test_bench_verdict_cleanup_failure_never_overwrites_business_verdict(
    host_accept_fixture: dict[str, Path], tmp_path: Path
) -> None:
    def _boom(host: str, scenario: str, _sandbox: Path) -> None:
        raise PermissionError(f"simulated post-verdict lock on {host}/{scenario}")

    pass_rep = run_host_acceptance(
        hosts=("kimi",),
        plugin_dir=host_accept_fixture["plugins_root"] / "mindie-agent-kimi",
        wheelhouse_stage=host_accept_fixture["cand_stage"],
        scenarios=("cold_native_launch",),
        cleanup_hook=_boom,
    )
    assert pass_rep["terminal_status"] == "completed_pass"
    assert exit_code_for_report(pass_rep) == 0
    assert any("PermissionError" in w for w in pass_rep["cleanup_warning"])

    broken_kimi = tmp_path / "broken-kimi"
    shutil.copytree(host_accept_fixture["plugins_root"] / "mindie-agent-kimi", broken_kimi)
    kdoc = json.loads((broken_kimi / "kimi.plugin.json").read_text(encoding="utf-8"))
    kdoc["mcpServers"].pop("knowledge")
    (broken_kimi / "kimi.plugin.json").write_text(json.dumps(kdoc), encoding="utf-8")

    fail_rep = run_host_acceptance(
        hosts=("kimi",),
        plugin_dir=broken_kimi,
        wheelhouse_stage=host_accept_fixture["cand_stage"],
        scenarios=("cold_native_launch",),
        cleanup_hook=_boom,
    )
    assert fail_rep["terminal_status"] == "completed_fail"
    assert exit_code_for_report(fail_rep) == 1
    assert any("PermissionError" in w for w in fail_rep["cleanup_warning"])
