"""Dynamic verbatim parsing of all 7 MindIE host plugin manifests and shims.

[Role]
Parses the live MCP server definitions and hook entrypoints from candidate plugin
trees (`codex`, `cc`, `kimi`, `dsh`, `agy`, `wb`, `oc`), including `npm pack` +
Node ESM execution for `dsh` (`host.mjs` / `host3.mjs`) and `oc` (`index.mjs`),
and dual-manifest parity + `wb_install.py` verification for `wb`.

[Invariants]
- Never mutates `os.environ`; requires explicit `env: Mapping[str, str]`.
- Rejects Windows `WindowsApps` 0-byte Python stubs and `/mnt/<drive>` WSL shims.
- Every `subprocess.run` passes `**subprocess_window_kwargs()`.

[Timeouts & Locks]
- Bounded subprocess timeouts (`<= 25s`) for `npm pack`, Node ESM, and `wb_install`.

[Side-Effect Policy]
- Writes unpacked tarballs and ESM probe scripts strictly inside `sandbox_root`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindie_bench.host_acceptance import HostAcceptanceError, resolve_base_python
from mindie_bench.runner.host_platform import (
    is_windows_store_stub,
    resolve_git_bash,
    subprocess_window_kwargs,
    which_native,
)

ALL_HOSTS: tuple[str, ...] = ("codex", "cc", "kimi", "dsh", "agy", "wb", "oc")
EXPECTED_HOST_SERVERS: dict[str, dict[str, int]] = {
    "codex": {"mindie-remote-dev": 10, "mindie-tools": 16},
    "cc": {"knowledge": 5, "remote": 13, "components": 11},
    "kimi": {"knowledge": 5, "remote": 14, "components": 11},
    "dsh": {"remote-dev": 14, "mindie-tools": 16},
    "agy": {"mindie-remote-dev": 12, "mindie-tools": 16},
    "wb": {"mindie-remote-dev": 15, "mindie-tools": 11, "mindie-knowledge": 5},
    "oc": {"mindie-remote-dev": 14, "mindie-tools": 16},
}


class ManifestValidationError(HostAcceptanceError):
    """Raised when a native host plugin manifest or installer contract is violated."""


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    surface: str


@dataclass(frozen=True)
class HookCommandSpec:
    event: str
    command: str | None
    argv: tuple[str, ...] | None
    cwd: Path
    env: dict[str, str]
    mode: str = "argv"


@dataclass(frozen=True)
class HostManifestSpec:
    host: str
    repo_root: Path
    plugin_dir: Path
    servers: dict[str, MCPServerSpec]
    hooks: dict[str, HookCommandSpec]
    metadata: dict[str, Any] = field(default_factory=dict)


def _run_cmd(argv: list[str], *, env: Mapping[str, str], timeout: float, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        **subprocess_window_kwargs(),
    )


def stage_npm_package(plugin_dir: Path, stage_dir: Path, env: Mapping[str, str], *, timeout: float = 25.0) -> Path:
    """Pack a Node host plugin via `npm pack` and extract `package/` safely."""
    npm_bin = which_native("npm", path=env.get("PATH")) or which_native("npm")
    if not npm_bin:
        raise FileNotFoundError("`npm` executable is required on PATH to pack Node host plugins")
    pack_out = stage_dir / "npm-pack"
    pack_out.mkdir(parents=True, exist_ok=True)
    proc = _run_cmd([npm_bin, "pack", "--pack-destination", str(pack_out)], env=env, timeout=timeout, cwd=plugin_dir)
    if proc.returncode != 0:
        raise ManifestValidationError(f"npm pack failed ({proc.returncode}): {proc.stderr[:400]}")
    tarballs = sorted(pack_out.glob("*.tgz"))
    if not tarballs:
        raise ManifestValidationError(f"npm pack produced no .tgz in {pack_out}")
    extract_dir = stage_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    root_res = extract_dir.resolve()
    with tarfile.open(tarballs[-1], "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or member.isfifo() or member.isdev():
                raise ManifestValidationError(f"Unsafe npm tarball member: {member.name}")
            dest = (extract_dir / member.name).resolve()
            if dest != root_res and root_res not in dest.parents:
                raise ManifestValidationError(f"Tarball path traversal: {member.name}")
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if (src := tar.extractfile(member)) is not None:
                    with src:
                        dest.write_bytes(src.read())
    pkg_dir = extract_dir / "package"
    if not pkg_dir.is_dir():
        raise ManifestValidationError(f"Extracted npm archive missing package/: {tarballs[-1]}")
    return pkg_dir


def _expand(value: str, mapping: Mapping[str, str]) -> str:
    for key, repl in mapping.items():
        value = value.replace(f"${{{key}}}", repl)
    return value


def _infer_surface(name: str, args: tuple[str, ...]) -> str:
    joined = " ".join(args).lower()
    if "remote-mcp" in joined or name in {"remote", "remote-dev", "mindie-remote-dev"} or args[-1:] == ("remote",):
        return "remote"
    if "knowledge-mcp" in joined or name in {"knowledge", "mindie-knowledge"} or args[-1:] == ("knowledge",):
        return "knowledge"
    return "components" if (name == "components" or args[-1:] == ("components",)) else "tools"


def _parse_mcp_json_servers(
    doc: Mapping[str, Any], *, default_cwd: Path, base_env: Mapping[str, str], var_map: Mapping[str, str]
) -> dict[str, MCPServerSpec]:
    raw_servers = doc.get("mcpServers")
    if not isinstance(raw_servers, Mapping) or not raw_servers:
        raise ManifestValidationError("Manifest missing non-empty `mcpServers` object")
    servers: dict[str, MCPServerSpec] = {}
    for name, raw in raw_servers.items():
        if not isinstance(raw, Mapping):
            raise ManifestValidationError(f"Server {name!r} entry is not an object")
        cmd, raw_args = _expand(str(raw.get("command") or ""), var_map), raw.get("args")
        if not cmd or not isinstance(raw_args, list):
            raise ManifestValidationError(f"Server {name!r} missing command/args")
        args = tuple(_expand(str(a), var_map) for a in raw_args)
        raw_cwd = raw.get("cwd")
        if isinstance(raw_cwd, str) and raw_cwd.strip():
            exp = _expand(raw_cwd.strip(), var_map)
            cwd = Path(exp) if Path(exp).is_absolute() else (default_cwd / exp).resolve()
        else:
            cwd = default_cwd
        merged = {**base_env, **{str(ek): _expand(str(ev), var_map) for ek, ev in (raw.get("env") or {}).items()}}
        servers[str(name)] = MCPServerSpec(str(name), cmd, args, cwd, merged, _infer_surface(str(name), args))
    return servers


def _run_node_esm_probe(script: Path, env: Mapping[str, str], *, timeout: float) -> dict[str, Any]:
    node_bin = which_native("node", path=env.get("PATH")) or which_native("node")
    if not node_bin:
        raise FileNotFoundError("`node` executable is required on PATH for ESM manifest probes")
    proc = _run_cmd([node_bin, str(script)], env=env, timeout=timeout, cwd=script.parent)
    if proc.returncode != 0:
        raise ManifestValidationError(f"Node ESM probe failed ({proc.returncode}): {(proc.stderr or proc.stdout)[:500]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("MINDIE_PROBE_JSON:")]
    if not lines:
        raise ManifestValidationError(f"Node ESM probe emitted no result: {proc.stdout[:300]}")
    return json.loads(lines[-1][len("MINDIE_PROBE_JSON:") :])


def _parse_dsh_manifest(repo_root: Path, pkg_dir: Path, sandbox_root: Path, env: Mapping[str, str], *, timeout: float) -> HostManifestSpec:
    probe_dir = sandbox_root / "dsh-esm-probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    mcp_mock = probe_dir / "mock_mcp.mjs"
    mcp_mock.write_text(
        "export function apply(ctx, cfg) { ctx._captured.push(cfg); ctx.effect(() => () => {}); }\n"
        "export default { name: 'mock-dsh-mcp', apply };\n",
        encoding="utf-8",
    )
    driver = probe_dir / "probe_dsh.mjs"
    driver.write_text(
        "import { pathToFileURL } from 'node:url';\n"
        f"const pkgDir = {json.dumps(str(pkg_dir))}, wsDir = {json.dumps(str(sandbox_root / 'ws'))};\n"
        f"const mcpUrl = pathToFileURL({json.dumps(str(mcp_mock))}).href;\n"
        "const hostMod = await import(pathToFileURL(pkgDir + '/host.mjs').href);\n"
        "const captured = [], listeners = new Map(), disposers = [];\n"
        "const makeCtx = () => ({ _captured: captured, agents: { list: () => [agent] }, llm: { async *stream() { yield { type: 'text-delta', text: '{}' }; } },\n"
        "  on(ev, fn) { listeners.set(ev, fn); }, effect(fn) { const d = fn(); if (typeof d === 'function') disposers.push(d); },\n"
        "  async plugin(p, cfg) { await (p?.apply || p?.default?.apply || p)(makeCtx(), cfg); return { dispose: async () => {} }; } });\n"
        "const agent = { session: { id: 'dsh-probe-1', header: { cwd: wsDir } }, options: { provider: 'deepseek', model: 'deepseek-chat' }, ctx: null };\n"
        "agent.ctx = makeCtx();\nawait hostMod.apply(makeCtx(), { mcpPluginUrl: mcpUrl });\n"
        "let preStepped = false;\nlisteners.get('agent/pre-step')?.({}, () => { preStepped = true; });\n"
        "await listeners.get('agent/created')?.({ agent });\nconst hooks = [...listeners.keys()];\n"
        "for (const d of disposers.reverse()) { try { await d(); } catch {} }\n"
        "process.stdout.write('MINDIE_PROBE_JSON:' + JSON.stringify({ captured, hooks, preStepped }) + '\\n');\n",
        encoding="utf-8",
    )
    probe = _run_node_esm_probe(driver, env, timeout=timeout)
    if any(h not in probe.get("hooks", []) for h in ("agent/created", "agent/pre-step", "agent/turn-stopping")) or not probe.get("preStepped"):
        raise ManifestValidationError(f"DSH host3.mjs missing required hook contract: {probe}")
    servers = {
        str(r["serverName"]): MCPServerSpec(
            str(r["serverName"]),
            str(r["command"]),
            (a := tuple(str(x) for x in r["args"])),
            Path(str(r["cwd"])),
            {str(k): str(v) for k, v in (r.get("env") or {}).items()},
            _infer_surface(str(r["serverName"]), a),
        )
        for r in (probe.get("captured") or [])
    }
    hooks = {
        "SessionStart": HookCommandSpec("SessionStart", None, (str(pkg_dir), "agent/created"), pkg_dir, dict(env), "dsh_esm"),
        "Stop": HookCommandSpec("Stop", None, (str(pkg_dir), "agent/turn-stopping"), pkg_dir, dict(env), "dsh_esm"),
    }
    return HostManifestSpec("dsh", repo_root, pkg_dir, servers, hooks, {"packed_dir": str(pkg_dir)})


def _parse_oc_manifest(repo_root: Path, pkg_dir: Path, sandbox_root: Path, env: Mapping[str, str], *, timeout: float) -> HostManifestSpec:
    sub, probe_dir = pkg_dir / "plugins/mindie-agent", sandbox_root / "oc-esm-probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    driver = probe_dir / "probe_oc.mjs"
    driver.write_text(
        "import { pathToFileURL } from 'node:url';\n"
        f"const pkgDir = {json.dumps(str(pkg_dir))}, wsDir = {json.dumps(str(sandbox_root / 'ws'))};\n"
        "const mod = await import(pathToFileURL(pkgDir + '/index.mjs').href);\n"
        "const servers = {}, sessionHooks = [], toolHooks = [], shellHooks = [];\nlet subscribed = false;\n"
        "const ctx = { mcp: { transform(fn) { fn({ set(k, v) { servers[k] = v; } }); }, reload() {} },\n"
        "  session: { hook(n) { sessionHooks.push(n); }, async get() { return { directory: wsDir }; } },\n"
        "  tool: { hook(n) { toolHooks.push(n); } }, shell: { hook(n) { shellHooks.push(n); } },\n"
        "  event: { async subscribe() { subscribed = true; return (async function*(){})(); } } };\n"
        "const dispose = await (mod.default || mod).setup(ctx);\nif (typeof dispose === 'function') await dispose();\n"
        "process.stdout.write('MINDIE_PROBE_JSON:' + JSON.stringify({ servers, sessionHooks, toolHooks, shellHooks, subscribed }) + '\\n');\n",
        encoding="utf-8",
    )
    oc_env = dict(env, OC_PLUGIN_ROOT=str(sub), OC_PLUGIN_DATA=env["MINDIE_DATA_HOME"])
    probe = _run_node_esm_probe(driver, oc_env, timeout=timeout)
    if "context" not in probe.get("sessionHooks", []) or "execute.before" not in probe.get("toolHooks", []) or "create.before" not in probe.get("shellHooks", []) or not probe.get("subscribed"):
        raise ManifestValidationError(f"OpenCode setup() missing required hooks: {probe}")
    esm_servers = {
        str(sn): MCPServerSpec(
            str(sn),
            str(r["command"][0]),
            (a := tuple(str(x) for x in r["command"][1:])),
            sub,
            {**oc_env, **{str(k): str(v) for k, v in (r.get("environment") or {}).items()}},
            _infer_surface(str(sn), a),
        )
        for sn, r in (probe.get("servers") or {}).items()
    }
    static_servers = _parse_mcp_json_servers(
        json.loads((sub / ".mcp.json").read_text(encoding="utf-8")),
        default_cwd=sub,
        base_env=oc_env,
        var_map={"OC_PLUGIN_ROOT": str(sub), "OC_PLUGIN_DATA": env["MINDIE_DATA_HOME"]},
    )
    if {k: v.surface for k, v in esm_servers.items()} != {k: v.surface for k, v in static_servers.items()}:
        raise ManifestValidationError("OpenCode index.mjs and .mcp.json server parity mismatch")
    hooks = {
        "SessionStart": HookCommandSpec("SessionStart", None, (str(pkg_dir), "context"), sub, oc_env, "oc_esm"),
        "Stop": HookCommandSpec("Stop", None, (str(pkg_dir), "session.idle"), sub, oc_env, "oc_esm"),
    }
    return HostManifestSpec("oc", repo_root, sub, esm_servers, hooks, {"packed_dir": str(pkg_dir)})


def run_wb_installer(wb_install_py: Path, source_root: Path, wb_dir: Path, env: Mapping[str, str], *, timeout: float = 15.0) -> dict[str, Any]:
    """Execute `wb_install.install_workbuddy_plugin` in an isolated child process."""
    code = (
        "import importlib.util as u, json, sys\nsp = u.spec_from_file_location('wb_install', sys.argv[1])\n"
        "m = u.module_from_spec(sp); sp.loader.exec_module(m)\n"
        "print(json.dumps(m.install_workbuddy_plugin(wb_dir=sys.argv[2], source_root=sys.argv[3])))\n"
    )
    proc = _run_cmd([str(resolve_base_python()), "-I", "-X", "utf8", "-c", code, str(wb_install_py), str(wb_dir), str(source_root)], env=env, timeout=timeout)
    if proc.returncode != 0:
        raise ManifestValidationError(f"wb_install.py failed ({proc.returncode}): {proc.stderr[:400]}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    if res.get("status") != "installed" or not Path(str(res.get("cache_install_path") or "")).is_dir():
        raise ManifestValidationError(f"wb_install.py returned unexpected payload: {res}")
    return res


def parse_host_manifest(
    host: str,
    plugin_dir: Path,
    *,
    sandbox_root: Path,
    env: Mapping[str, str],
    prepacked_dir: Path | None = None,
    pack_npm: bool = True,
    timeout: float = 60.0,
) -> HostManifestSpec:
    """Dynamically parse and structurally validate the native manifest(s) for `host`."""
    clean = host.strip().lower()
    if clean not in ALL_HOSTS:
        raise ManifestValidationError(f"Unsupported host {host!r}")
    repo_root = plugin_dir.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Host plugin directory not found: {repo_root}")

    if clean == "codex":
        sub = repo_root / "plugins/mindie-agent" if (repo_root / "plugins/mindie-agent/.mcp.json").is_file() else repo_root
        c_env = dict(env, PLUGIN_ROOT=str(sub), PLUGIN_DATA=str(Path(env["CODEX_HOME"]) / "plugins/data/mindie-agent-mindie-agent"))
        servers = _parse_mcp_json_servers(json.loads((sub / ".mcp.json").read_text(encoding="utf-8")), default_cwd=sub, base_env=c_env, var_map={})
        hdoc = json.loads((sub / "hooks/hooks.json").read_text(encoding="utf-8")).get("hooks") or {}
        intr = [i for grp in (hdoc.get("Interrupt") or []) for i in (grp.get("hooks") or [])]
        if {i.get("server") for i in intr if i.get("type") == "mcp_tool" and i.get("tool") == "mindie_interrupt"} != {"mindie-remote-dev", "mindie-tools"}:
            raise ManifestValidationError("Codex hooks.json missing Interrupt mindie_interrupt")
        mode, key = ("powershell_encoded", "commandWindows") if os.name == "nt" else ("posix_sh", "command")
        hooks = {ev: HookCommandSpec(ev, hdoc[ev][0]["hooks"][0][key], None, sub, c_env, mode=mode) for ev in ("SessionStart", "UserPromptSubmit", "Stop")}
        return HostManifestSpec("codex", repo_root, sub, servers, hooks)

    if clean == "cc":
        sub = repo_root / "plugins/mindie-agent" if (repo_root / "plugins/mindie-agent/.mcp.json").is_file() else repo_root
        vmap = {"CLAUDE_PLUGIN_ROOT": str(sub)}
        cc_env = dict(env, CLAUDE_PLUGIN_ROOT=str(sub), CLAUDE_PLUGIN_DATA=str(Path(env["CLAUDE_CONFIG_DIR"]) / "plugin-data"))
        servers = _parse_mcp_json_servers(json.loads((sub / ".mcp.json").read_text(encoding="utf-8")), default_cwd=sub, base_env=cc_env, var_map=vmap)
        bash_exe, hdoc = resolve_git_bash(cc_env), json.loads((sub / "hooks/hooks.json").read_text(encoding="utf-8")).get("hooks") or {}
        hooks = {ev: HookCommandSpec(ev, (cmd := _expand(hdoc[ev][0]["hooks"][0]["command"], vmap)), (bash_exe, "-c", cmd), sub, cc_env) for ev in ("PreToolUse", "SessionStart", "Stop")}
        return HostManifestSpec("cc", repo_root, sub, servers, hooks)

    if clean == "kimi":
        kdoc = json.loads((repo_root / "kimi.plugin.json").read_text(encoding="utf-8"))
        servers = _parse_mcp_json_servers(kdoc, default_cwd=repo_root, base_env=env, var_map={})
        for srv in servers.values():
            resolved_py3 = which_native(srv.command, path=env.get("PATH"))
            if not resolved_py3 or is_windows_store_stub(resolved_py3):
                raise ManifestValidationError(f"Kimi command {srv.command!r} resolved to stub/missing")
        py3 = which_native("python3", path=env.get("PATH")) or str(resolve_base_python())
        hooks: dict[str, HookCommandSpec] = {}
        for it in kdoc.get("hooks") or []:
            raw_cmd = str(it["command"])
            toks = raw_cmd.split()
            h_argv = (
                (py3, "-B", "-X", "utf8", *toks[toks.index("./scripts/mindie_launch.py") :])
                if "./scripts/mindie_launch.py" in toks
                else (py3, *toks[1:])
            )
            hooks[str(it["event"])] = HookCommandSpec(str(it["event"]), raw_cmd, h_argv, repo_root, dict(env))
        hooks.setdefault("SessionStart", HookCommandSpec("SessionStart", "python3 ./scripts/mindie_launch.py hook session-start", (py3, "./scripts/mindie_launch.py", "hook", "session-start"), repo_root, dict(env)))
        return HostManifestSpec("kimi", repo_root, repo_root, servers, hooks)

    if clean == "dsh":
        pkg = prepacked_dir or (stage_npm_package(repo_root, sandbox_root / "dsh-pack", env, timeout=timeout) if pack_npm else repo_root)
        return _parse_dsh_manifest(repo_root, pkg, sandbox_root, env, timeout=timeout)

    if clean == "agy":
        sub = repo_root / "plugins/mindie-agent" if (repo_root / "plugins/mindie-agent/mcp_config.json").is_file() else repo_root
        servers = _parse_mcp_json_servers(json.loads((sub / "mcp_config.json").read_text(encoding="utf-8")), default_cwd=sub, base_env=env, var_map={})
        hdoc, bash_exe = json.loads((sub / "hooks.json").read_text(encoding="utf-8")).get("mindie-agent") or {}, resolve_git_bash(env)
        hooks = {
            ev: HookCommandSpec(ev, (cs := hdoc[ev][0]["hooks"][0]["command"] if "hooks" in hdoc[ev][0] else hdoc[ev][0]["command"]), (bash_exe, "-c", cs), sub, dict(env))
            for ev in ("SessionStart", "PreInvocation", "PreToolUse", "Stop")
        }
        return HostManifestSpec("agy", repo_root, sub, servers, hooks)

    if clean == "wb":
        sub = repo_root / "plugins/mindie-agent" if (repo_root / "plugins/mindie-agent/.mcp.json").is_file() else repo_root
        wb_data = str(Path(env["MINDIE_DATA_HOME"]) / "wb-plugin-data")
        dot = _parse_mcp_json_servers(json.loads((sub / ".mcp.json").read_text(encoding="utf-8")), default_cwd=sub, base_env=env, var_map={"CODEBUDDY_PLUGIN_ROOT": str(sub), "CODEBUDDY_PLUGIN_DATA": wb_data})
        plain = _parse_mcp_json_servers(json.loads((sub / "mcp.json").read_text(encoding="utf-8")), default_cwd=sub, base_env=env, var_map={"PLUGIN_ROOT": str(sub), "PLUGIN_DATA": wb_data})
        if [(k, v.command, v.args, v.cwd, v.surface) for k, v in sorted(dot.items())] != [(k, v.command, v.args, v.cwd, v.surface) for k, v in sorted(plain.items())]:
            raise ManifestValidationError("WorkBuddy .mcp.json and mcp.json server parity mismatch")
        wb_res = run_wb_installer(sub / "scripts/wb_install.py", sub, Path(env["WORKBUDDY_CONFIG_DIR"]), env, timeout=timeout)
        hdoc = json.loads((sub / "hooks/hooks.json").read_text(encoding="utf-8")).get("hooks") or {}
        py_bin, wb_env = which_native("python", path=env.get("PATH")) or str(resolve_base_python()), dict(env, WB_PLUGIN_ROOT=str(sub), WB_PLUGIN_DATA=wb_data)
        hooks = {
            ev: HookCommandSpec(ev, _expand(hdoc[ev][0]["hooks"][0]["command"], {"CODEBUDDY_PLUGIN_ROOT": str(sub)}), (py_bin, "-I", "-X", "utf8", str(sub / "scripts/mindie_launch.py"), "hook", re.sub(r"(?<!^)(?=[A-Z])", "-", ev).lower()), sub, wb_env)
            for ev in ("SessionStart", "UserPromptSubmit", "PreToolUse", "Stop")
        }
        return HostManifestSpec("wb", repo_root, sub, dot, hooks, {"wb_install": wb_res})

    pkg = prepacked_dir or (stage_npm_package(repo_root, sandbox_root / "oc-pack", env, timeout=timeout) if pack_npm else repo_root)
    return _parse_oc_manifest(repo_root, pkg, sandbox_root, env, timeout=timeout)
