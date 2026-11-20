"""Universal 7-host native manifest, MCP wire, hook capture, and upgrade verifier.

[Role]
Exposes Gate Layer 2 (`mindie-bench host-accept`) and provides shared sandbox
isolation (`build_sandbox_env`, `unpack_wheelhouse_links`, `install_git_trap`,
`stop_sandbox_knowledge_services`) across all 4 host acceptance scenarios.

[Invariants]
- Zero production reverse-contamination (`BENCH-ISOLATION`): host-specific manifest
  parsers and synthetic transcript fixtures live strictly inside `mindie_bench`.
- Verdict honesty (`BENCH-VERDICT`): post-verdict teardown errors append to
  `cleanup_warning` and never overwrite settled `completed_pass`/`completed_fail` states.

[Timeouts & Locks]
- Default per-host outer budget is `240.0s` (`setup + inner_budget + cleanup_grace`).
- Zero process-global locks or `os.environ` mutations; child envs are explicit dicts.

[Side-Effect Policy]
- Executes strictly inside disposable `TemporaryDirectory` sandboxes and writes an
  optional JSON report only when `--output` is requested.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from email.parser import Parser
from pathlib import Path

from mindie_bench.runner.host_platform import (
    is_windows_store_stub,
    which_native,
    write_command_alias,
    write_executable_shim,
)

_STRIP_ENV_PREFIXES = (
    "MINDIE_",
    "REMOTE_DEV_",
    "CODEX_",
    "CLAUDE",
    "KIMI_",
    "DSH_",
    "AGY_",
    "WB_",
    "WORKBUDDY_",
    "CODEBUDDY_",
    "OC_",
    "OPENCODE_",
    "GH_",
    "GITHUB_",
)
_NORM_PKG_RE = re.compile(r"[-_.]+")


class HostAcceptanceError(AssertionError):
    """Base business-contract assertion error for Gate Layer 2 host acceptance."""


def resolve_base_python() -> Path:
    exe_rel = "python.exe" if os.name == "nt" else "bin/python"
    for cand in (
        Path(sys.base_prefix) / exe_rel,
        Path(sys.base_prefix) / "python",
        Path(sys.executable),
    ):
        if cand.is_file() and not is_windows_store_stub(cand):
            return cand.resolve()
    raise FileNotFoundError("Unable to locate a non-stub base Python interpreter")


def norm_pkg_name(name: str) -> str:
    return _NORM_PKG_RE.sub("-", name.strip()).lower()


def unpack_wheelhouse_links(wheelhouse_stage: Path, links_dir: Path) -> tuple[Path, bool]:
    """Unpack `mindie-wheels-bundle.tar.gz` into `links_dir` and return `(dir, self_contained)`."""
    links_dir.mkdir(parents=True, exist_ok=True)
    bundle = wheelhouse_stage / "mindie-wheels-bundle.tar.gz"
    if bundle.is_file():
        root = links_dir.resolve()
        with tarfile.open(bundle, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name != member.name or ".." in Path(member.name).parts:
                    raise HostAcceptanceError(f"Unsafe wheel bundle member: {member.name}")
                target = (links_dir / name).resolve()
                if root not in target.parents:
                    raise HostAcceptanceError(f"Escaped wheel bundle target: {target}")
                src = tar.extractfile(member)
                if src is not None:
                    with src:
                        target.write_bytes(src.read())
    else:
        for item in wheelhouse_stage.iterdir():
            if item.is_file() and (
                item.suffix == ".whl" or item.name in {"requirements.txt", "requirements.lock"}
            ):
                (links_dir / item.name).write_bytes(item.read_bytes())

    wheels = sorted(links_dir.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No wheels found in wheelhouse stage {wheelhouse_stage}")
    bundled = {norm_pkg_name(w.name.split("-")[0]) for w in wheels}
    self_contained = True
    for whl in wheels:
        with zipfile.ZipFile(whl) as zf:
            meta_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
            if not meta_names:
                continue
            meta = Parser().parsestr(zf.read(meta_names[0]).decode("utf-8", errors="replace"))
            for req in meta.get_all("Requires-Dist") or []:
                if re.search(r";\s*extra\s*==", req):
                    continue
                dep_match = re.match(r"^\s*([A-Za-z0-9._-]+)", req)
                if dep_match and norm_pkg_name(dep_match.group(1)) not in bundled:
                    self_contained = False
                    break
    return links_dir, self_contained


def install_git_trap(trap_dir: Path, env: Mapping[str, str]) -> dict[str, str]:
    """Prepend a fail-closed `git` shim (`exit 99`) onto `env['PATH']`."""
    write_executable_shim(
        trap_dir,
        "git",
        "import sys\nsys.stderr.write('git trap: ' + repr(sys.argv[1:]) + '\\n')\nraise SystemExit(99)\n",
    )
    updated = dict(env)
    updated["PATH"] = str(trap_dir) + os.pathsep + updated.get("PATH", "")
    return updated


def build_sandbox_env(
    sandbox_root: Path,
    *,
    wheelhouse_stage: Path | None = None,
    redactor: Path | None = None,
    offline_override: bool | None = None,
) -> tuple[dict[str, str], bool]:
    """Build an isolated environment dictionary without mutating `os.environ`."""
    base_py = resolve_base_python()
    dirs = {
        k: sandbox_root / rel
        for k, rel in (
            ("home", "home"),
            ("appdata", "appdata"),
            ("local", "localappdata"),
            ("xcfg", "xdg-config"),
            ("xdata", "xdg-data"),
            ("xstate", "xdg-state"),
            ("xcache", "xdg-cache"),
            ("mcfg", "config/mindie-agent"),
            ("mdata", "data/mindie-agent"),
            ("diag", "diagnostics"),
            ("rstate", "remote-state"),
            ("rauth", "ssh-auth"),
            ("codex", "codex-home"),
            ("claude", "claude-config"),
            ("kimi", "kimi-home"),
            ("dsh", "dsh-home"),
            ("agy", "agy-appdata"),
            ("wb", "wb-config"),
            ("oc", "oc-home"),
            ("gh", "gh-config"),
            ("tmp", "tmp"),
            ("bin", "bin"),
            ("ws", "ws"),
        )
    }
    for key in ("home", "appdata", "local", "xcfg", "xdata", "xstate", "xcache", "tmp", "bin", "ws", "gh"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    write_command_alias(dirs["bin"], "python", [str(base_py)])
    write_command_alias(dirs["bin"], "python3", [str(base_py)])

    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}
        and not any(k.upper().startswith(p) for p in _STRIP_ENV_PREFIXES)
    }
    path_parts = [
        p
        for p in env.get("PATH", "").split(os.pathsep)
        if p and ".venv" not in p.replace("\\", "/").lower() and "windowsapps" not in p.lower()
    ]
    extra_bins = [str(dirs["bin"]), str(base_py.parent)]
    for tool in ("uv", "node", "npm", "git"):
        found = which_native(tool)
        if found:
            tdir = str(Path(found).resolve().parent)
            if tool == "uv":
                extra_bins.append(tdir)
            elif tdir not in extra_bins and tdir not in path_parts:
                path_parts.append(tdir)
    env["PATH"] = os.pathsep.join([*extra_bins, *path_parts])

    if base_py.parent.name.startswith("cpython-") and base_py.parent.parent.is_dir():
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(base_py.parent.parent))
    elif base_py.parent.parent.name.startswith("cpython-") and base_py.parent.parents[1].is_dir():
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(base_py.parent.parents[1]))

    env.update(
        {
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(dirs["xcache"] / "uv"),
            "UV_LINK_MODE": os.environ.get("UV_LINK_MODE", "copy" if os.name == "nt" else "hardlink"),
            "UV_NO_CONFIG": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "HOME": str(dirs["home"]),
            "USERPROFILE": str(dirs["home"]),
            "APPDATA": str(dirs["appdata"]),
            "LOCALAPPDATA": str(dirs["local"]),
            "XDG_CONFIG_HOME": str(dirs["xcfg"]),
            "XDG_DATA_HOME": str(dirs["xdata"]),
            "XDG_STATE_HOME": str(dirs["xstate"]),
            "XDG_CACHE_HOME": str(dirs["xcache"]),
            "MINDIE_CONFIG_HOME": str(dirs["mcfg"]),
            "MINDIE_DATA_HOME": str(dirs["mdata"]),
            "MINDIE_DIAGNOSTICS_ROOT": str(dirs["diag"]),
            "MINDIE_DIAGNOSTICS_CONFIG": str(dirs["mcfg"] / "diagnostics.json"),
            "MINDIE_REMOTE_STATE_DIR": str(dirs["rstate"]),
            "REMOTE_DEV_STATE_DIR": str(dirs["rstate"]),
            "REMOTE_DEV_AUTH_DIR": str(dirs["rauth"]),
            "REMOTE_DEV_CREDENTIAL_FILE": str(dirs["mcfg"] / "ssh-credentials.json"),
            "CODEX_HOME": str(dirs["codex"]),
            "CLAUDE_CONFIG_DIR": str(dirs["claude"]),
            "KIMI_CODE_HOME": str(dirs["kimi"]),
            "DSH_HOME": str(dirs["dsh"]),
            "MINDIE_DSH_UPDATE_DELAY_MS": "600000",
            "AGY_APP_DATA_DIR": str(dirs["agy"]),
            "WORKBUDDY_CONFIG_DIR": str(dirs["wb"]),
            "OPENCODE_TEST_HOME": str(dirs["oc"]),
            "MINDIE_OPENCODE_DB": str(dirs["oc"] / "opencode.db"),
            "GH_CONFIG_DIR": str(dirs["gh"]),
            "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
            "TMPDIR": str(dirs["tmp"]),
            "TEMP": str(dirs["tmp"]),
            "TMP": str(dirs["tmp"]),
        }
    )
    if redactor is not None:
        env["MINDIE_REDACTOR_EXECUTABLE"] = str(redactor.resolve())

    is_offline = False
    if wheelhouse_stage is not None:
        links_dir, self_contained = unpack_wheelhouse_links(
            wheelhouse_stage, sandbox_root / "wheelhouse-links"
        )
        is_offline = self_contained if offline_override is None else bool(offline_override)
        env["MINDIE_CHANNEL_BASE_URL"] = wheelhouse_stage.resolve().as_uri()
        env["UV_FIND_LINKS"] = str(links_dir)
        req_txt = links_dir / "requirements.txt"
        if req_txt.is_file():
            env["MINDIE_WHEELHOUSE_REQUIREMENTS"] = str(req_txt)
        if is_offline:
            env["_MINDIE_UV_FLAGS"] = "--offline"
            env["UV_OFFLINE"] = "1"
            env["UV_NO_INDEX"] = "1"
            env["HTTP_PROXY"] = "http://127.0.0.1:9"
            env["HTTPS_PROXY"] = "http://127.0.0.1:9"
            env["NO_PROXY"] = "127.0.0.1,localhost"
    return env, is_offline


def stop_sandbox_knowledge_services(sandbox_root: Path) -> None:
    """Send loopback `stop` RPC to any knowledge service started inside `sandbox_root`."""
    if not sandbox_root.is_dir():
        return
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    conns = list(sandbox_root.rglob("connection.json"))
    for conn_file in conns:
        try:
            doc = json.loads(conn_file.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, ValueError):
            continue
        url = doc.get("url") if isinstance(doc, dict) else None
        token = doc.get("token") if isinstance(doc, dict) else None
        if not isinstance(url, str) or not url.startswith("http://127.0.0.1:") or not token:
            continue
        req = urllib.request.Request(
            f"{url.rstrip('/')}/rpc",
            data=b'{"method":"stop","arguments":{}}',
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        try:
            with opener.open(req, timeout=1.5) as resp:
                resp.read(4096)
        except TimeoutError:
            pass
        except (urllib.error.URLError, OSError):
            continue
    deadline = time.monotonic() + 1.5
    for conn_file in conns:
        while conn_file.exists() and time.monotonic() < deadline:
            time.sleep(0.03)


from mindie_bench.host_acceptance.cli import (  # noqa: E402
    ALL_SCENARIOS,
    add_host_accept_subparser,
    dispatch_host_accept,
    exit_code_for_report,
    main,
    run_host_acceptance,
)
from mindie_bench.host_acceptance.manifests import (  # noqa: E402
    ALL_HOSTS,
    EXPECTED_HOST_SERVERS,
    HookCommandSpec,
    HostManifestSpec,
    ManifestValidationError,
    MCPServerSpec,
    parse_host_manifest,
)
from mindie_bench.host_acceptance.mcp_wire import MCPProtocolError, MCPWireSession  # noqa: E402

__all__ = [
    "ALL_HOSTS",
    "ALL_SCENARIOS",
    "EXPECTED_HOST_SERVERS",
    "HookCommandSpec",
    "HostAcceptanceError",
    "HostManifestSpec",
    "MCPProtocolError",
    "MCPServerSpec",
    "MCPWireSession",
    "ManifestValidationError",
    "add_host_accept_subparser",
    "build_sandbox_env",
    "dispatch_host_accept",
    "exit_code_for_report",
    "install_git_trap",
    "main",
    "parse_host_manifest",
    "resolve_base_python",
    "run_host_acceptance",
    "stop_sandbox_knowledge_services",
    "unpack_wheelhouse_links",
]
