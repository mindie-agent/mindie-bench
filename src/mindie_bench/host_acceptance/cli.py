"""CLI runner and `BENCH-VERDICT` report builder for `mindie-bench host-accept`.

[Role]
Implements `mindie-bench host-accept` (`schema: "mindie-host-acceptance/1"`),
running Scenarios 1–4 in isolated per-scenario sandboxes across any subset or all 7
native MindIE host plugins (`codex`, `cc`, `kimi`, `dsh`, `agy`, `wb`, `oc`).

[Invariants]
- Verdict honesty (`BENCH-VERDICT`): once a scenario's business verdict settles as
  `completed_pass` or `completed_fail`, post-verdict cleanup errors (`close`, service
  stop, `rmtree_force`, `cleanup_hook`) NEVER turn it into `infra_failed` — they are
  recorded in `cleanup_warning`.
- Enforces `outer_timeout >= setup(2s) + inner_budget + cleanup_grace(5s)`.

[Timeouts & Locks]
- Default per-host outer budget is `600.0s`; scenarios run sequentially per host (`1`
  worker) with bounded host concurrency (`<= 2`) across all platforms so 3–4 vCPU CI
  runners never starve 6s `Stop` hook budgets while sharing one run-scoped `uv` cache.

[Side-Effect Policy]
- Uses isolated `TemporaryDirectory` sandboxes and writes `--output` JSON atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mindie_bench.host_acceptance import (
    HostAcceptanceError,
    build_sandbox_env,
    stop_sandbox_knowledge_services,
)
from mindie_bench.host_acceptance.hooks_capture import (
    provision_shared_hook_assets,
    run_scenario_hooks_and_capture,
)
from mindie_bench.host_acceptance.manifests import ALL_HOSTS, stage_npm_package
from mindie_bench.host_acceptance.mcp_wire import (
    run_scenario_cold_native_launch,
    run_scenario_public_install_py,
)
from mindie_bench.host_acceptance.upgrade_cutover import run_scenario_upgrade_cutover
from mindie_bench.io import write_json
from mindie_bench.runner.host_platform import rmtree_force

ALL_SCENARIOS: tuple[str, ...] = (
    "cold_native_launch",
    "public_install_py",
    "hooks_and_capture",
    "upgrade_cutover",
)
SETUP_BUDGET_SEC = 2.0
CLEANUP_GRACE_SEC = 5.0


def _default_plugins_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    for cand in (repo_root.parent, repo_root.parent.parent, Path.cwd().parent):
        if any((cand / f"mindie-agent-{h}").is_dir() for h in ALL_HOSTS) or (cand / "mindie-agent").is_dir():
            return cand
    return repo_root.parent


def _sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _aggregate_terminal(statuses: Sequence[str]) -> str:
    if not statuses:
        return "infra_failed"
    if all(s == "completed_pass" for s in statuses):
        return "completed_pass"
    if any(s == "unknown" for s in statuses):
        return "unknown"
    if any(s == "infra_failed" for s in statuses):
        return "infra_failed"
    return "completed_fail"


def _public_status(terminal_status: str) -> str:
    if terminal_status == "completed_pass":
        return "passed"
    return "failed" if terminal_status == "completed_fail" else "infra_failed"


def exit_code_for_report(report: dict[str, Any]) -> int:
    term = str(report.get("terminal_status") or "")
    if term == "completed_pass":
        return 0
    return 1 if term == "completed_fail" else 2


def _run_one_scenario(
    host: str,
    scenario: str,
    plugin_dir: Path,
    *,
    wheelhouse_stage: Path | None,
    baseline_wheelhouse_stage: Path | None,
    baseline_plugin_dir: Path | None,
    install_script: Path | None,
    redactor: Path | None,
    prepacked_dir: Path | None,
    inner_deadline: float,
    uv_cache_dir: Path | None = None,
    shared_hook_venv_py: Path | None = None,
    cleanup_hook: Callable[[str, str, Path], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    cleanup_warnings: list[str] = []
    sandbox_dir = Path(tempfile.mkdtemp(prefix=f"mindie-ha-{host}-{scenario}-")).resolve()
    terminal_status = "unknown"
    error_info: dict[str, str] | None = None
    details: dict[str, Any] = {}

    common = dict(
        sandbox_root=sandbox_dir,
        wheelhouse_stage=wheelhouse_stage,
        redactor=redactor,
        prepacked_dir=prepacked_dir,
        deadline=inner_deadline,
        cleanup_warnings=cleanup_warnings,
        uv_cache_dir=uv_cache_dir,
    )
    try:
        try:
            if scenario == "cold_native_launch":
                details = run_scenario_cold_native_launch(host, plugin_dir, **common)
            elif scenario == "public_install_py":
                details = run_scenario_public_install_py(host, plugin_dir, install_script=install_script, **common)
            elif scenario == "hooks_and_capture":
                details = run_scenario_hooks_and_capture(
                    host, plugin_dir, shared_hook_venv_py=shared_hook_venv_py, **common
                )
            elif scenario == "upgrade_cutover":
                details = run_scenario_upgrade_cutover(
                    host,
                    plugin_dir,
                    baseline_wheelhouse_stage=baseline_wheelhouse_stage,
                    baseline_plugin_dir=baseline_plugin_dir,
                    **common,
                )
            else:
                raise ValueError(f"Unknown host acceptance scenario: {scenario!r}")
            terminal_status = "completed_pass"
        except HostAcceptanceError as exc:
            terminal_status = "completed_fail"
            error_info = {"type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            terminal_status = "infra_failed"
            error_info = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        # Post-verdict cleanup: NEVER mutate `terminal_status` (`BENCH-VERDICT`).
        if cleanup_hook is not None:
            try:
                cleanup_hook(host, scenario, sandbox_dir)
            except Exception as exc:
                cleanup_warnings.append(f"cleanup_hook: {type(exc).__name__}: {exc}")
        try:
            stop_sandbox_knowledge_services(sandbox_dir)
        except Exception as exc:
            cleanup_warnings.append(f"stop_services: {type(exc).__name__}: {exc}")
        try:
            rmtree_force(sandbox_dir)
            if sandbox_dir.exists():
                shutil.rmtree(sandbox_dir, ignore_errors=True)
        except Exception as exc:
            cleanup_warnings.append(f"rmtree_sandbox: {type(exc).__name__}: {exc}")

    row: dict[str, Any] = {
        "scenario": scenario,
        "terminal_status": terminal_status,
        "status": _public_status(terminal_status),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "details": details,
        "cleanup_warning": cleanup_warnings,
    }
    if error_info is not None:
        row["error"] = error_info
    return row


def _run_one_host(
    host: str,
    plugin_dir: Path,
    *,
    scenarios: Sequence[str],
    wheelhouse_stage: Path | None,
    baseline_wheelhouse_stage: Path | None,
    baseline_plugin_dir: Path | None,
    install_script: Path | None,
    redactor: Path | None,
    timeout_sec: float,
    uv_cache_dir: Path | None = None,
    shared_hook_venv_py: Path | None = None,
    cleanup_hook: Callable[[str, str, Path], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    outer_budget = max(10.0, float(timeout_sec))
    inner_budget = max(4.0, outer_budget - SETUP_BUDGET_SEC - CLEANUP_GRACE_SEC)
    inner_deadline = started + SETUP_BUDGET_SEC + inner_budget
    host_warnings: list[str] = []
    pack_sandbox: Path | None = None
    prepacked_dir: Path | None = None

    if not plugin_dir.is_dir():
        return {
            "host": host,
            "plugin_dir": str(plugin_dir),
            "terminal_status": "infra_failed",
            "status": "infra_failed",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "error": {"type": "FileNotFoundError", "message": f"Host plugin directory missing: {plugin_dir}"},
            "cleanup_warning": [],
            "scenarios": {},
        }

    try:
        if host in {"dsh", "oc"}:
            pack_sandbox = Path(tempfile.mkdtemp(prefix=f"mindie-ha-pack-{host}-")).resolve()
            pack_env, _ = build_sandbox_env(pack_sandbox, uv_cache_dir=uv_cache_dir)
            prepacked_dir = stage_npm_package(
                plugin_dir.resolve(),
                pack_sandbox / "shared-pack",
                pack_env,
                timeout=max(2.0, inner_deadline - time.monotonic()),
            )
    except HostAcceptanceError as exc:
        if pack_sandbox is not None:
            try:
                rmtree_force(pack_sandbox)
            except Exception as c_exc:
                host_warnings.append(f"rmtree_pack: {type(c_exc).__name__}: {c_exc}")
        return {
            "host": host,
            "plugin_dir": str(plugin_dir),
            "terminal_status": "completed_fail",
            "status": "failed",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "cleanup_warning": host_warnings,
            "scenarios": {},
        }
    except Exception as exc:
        if pack_sandbox is not None:
            try:
                rmtree_force(pack_sandbox)
            except Exception as c_exc:
                host_warnings.append(f"rmtree_pack: {type(c_exc).__name__}: {c_exc}")
        return {
            "host": host,
            "plugin_dir": str(plugin_dir),
            "terminal_status": "infra_failed",
            "status": "infra_failed",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "cleanup_warning": host_warnings,
            "scenarios": {},
        }

    scenario_rows: dict[str, dict[str, Any]] = {}
    for scen in scenarios:
        scenario_rows[scen] = _run_one_scenario(
            host,
            scen,
            plugin_dir,
            wheelhouse_stage=wheelhouse_stage,
            baseline_wheelhouse_stage=baseline_wheelhouse_stage,
            baseline_plugin_dir=baseline_plugin_dir,
            install_script=install_script,
            redactor=redactor,
            prepacked_dir=prepacked_dir,
            inner_deadline=inner_deadline,
            uv_cache_dir=uv_cache_dir,
            shared_hook_venv_py=shared_hook_venv_py,
            cleanup_hook=cleanup_hook,
        )

    if pack_sandbox is not None:
        try:
            rmtree_force(pack_sandbox)
        except Exception as exc:
            host_warnings.append(f"rmtree_pack: {type(exc).__name__}: {exc}")

    for srow in scenario_rows.values():
        host_warnings.extend(f"{srow['scenario']}.{w}" for w in srow.get("cleanup_warning", ()))
    term = _aggregate_terminal([r["terminal_status"] for r in scenario_rows.values()])
    return {
        "host": host,
        "plugin_dir": str(plugin_dir),
        "terminal_status": term,
        "status": _public_status(term),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "cleanup_warning": host_warnings,
        "scenarios": scenario_rows,
    }


def run_host_acceptance(
    *,
    hosts: Sequence[str],
    plugin_dir: Path | None = None,
    plugins_root: Path | None = None,
    wheelhouse_stage: Path | None = None,
    baseline_wheelhouse_stage: Path | None = None,
    baseline_plugin_dir: Path | None = None,
    install_script: Path | None = None,
    redactor: Path | None = None,
    scenarios: Sequence[str] = ALL_SCENARIOS,
    timeout_sec: float = 600.0,
    cleanup_hook: Callable[[str, str, Path], None] | None = None,
) -> dict[str, Any]:
    """Run Gate Layer 2 host acceptance across `hosts` and return the structured report."""
    started = time.monotonic()
    clean_hosts = tuple(dict.fromkeys(h.strip().lower() for h in hosts if h.strip()))
    if not clean_hosts or any(h not in ALL_HOSTS for h in clean_hosts):
        raise ValueError(f"Invalid hosts {clean_hosts}; expected subset of {ALL_HOSTS}")
    clean_scenarios = tuple(dict.fromkeys(s.strip() for s in scenarios if s.strip()))
    if not clean_scenarios or any(s not in ALL_SCENARIOS for s in clean_scenarios):
        raise ValueError(f"Invalid scenarios {clean_scenarios}; expected subset of {ALL_SCENARIOS}")
    if plugin_dir is not None and len(clean_hosts) > 1:
        raise ValueError("`--plugin-dir` cannot be combined with multiple hosts; use `--plugins-root`")

    eff_root = plugins_root.resolve() if plugins_root is not None else _default_plugins_root()
    eff_install = install_script.resolve() if install_script is not None else (eff_root / "mindie-agent/install.py")
    if install_script is None and not eff_install.is_file():
        eff_install = None

    shared_uv_cache = Path(tempfile.mkdtemp(prefix="mindie-ha-uvcache-")).resolve()
    shared_hook_stage: Path | None = None
    shared_hook_py: Path | None = None
    eff_redactor = redactor.resolve() if redactor else None
    run_warnings: list[str] = []
    host_reports: dict[str, dict[str, Any]] = {}
    host_workers = min(2, len(clean_hosts))
    try:
        if len(clean_hosts) > 1 and "hooks_and_capture" in clean_scenarios and wheelhouse_stage is not None and wheelhouse_stage.is_dir():
            shared_hook_stage = Path(tempfile.mkdtemp(prefix="mindie-ha-hookstage-")).resolve()
            try:
                shared_hook_py, cached_red = provision_shared_hook_assets(
                    shared_hook_stage,
                    wheelhouse_stage=wheelhouse_stage.resolve(),
                    redactor=eff_redactor,
                    uv_cache_dir=shared_uv_cache,
                    timeout_sec=min(300.0, float(timeout_sec)),
                )
                if eff_redactor is None and cached_red is not None and cached_red.is_file():
                    eff_redactor = cached_red
            except Exception as exc:
                run_warnings.append(f"shared_hook_stage: {type(exc).__name__}: {exc}")
                shared_hook_py = None

        with ThreadPoolExecutor(max_workers=host_workers) as pool:
            futs = {
                h: pool.submit(
                    _run_one_host,
                    h,
                    plugin_dir.resolve() if plugin_dir is not None else (eff_root / f"mindie-agent-{h}"),
                    scenarios=clean_scenarios,
                    wheelhouse_stage=wheelhouse_stage.resolve() if wheelhouse_stage else None,
                    baseline_wheelhouse_stage=baseline_wheelhouse_stage.resolve() if baseline_wheelhouse_stage else None,
                    baseline_plugin_dir=baseline_plugin_dir.resolve() if baseline_plugin_dir else None,
                    install_script=eff_install,
                    redactor=eff_redactor,
                    timeout_sec=timeout_sec,
                    uv_cache_dir=shared_uv_cache,
                    shared_hook_venv_py=shared_hook_py,
                    cleanup_hook=cleanup_hook,
                )
                for h in clean_hosts
            }
            for h in clean_hosts:
                host_reports[h] = futs[h].result()
    finally:
        if shared_hook_stage is not None:
            try:
                rmtree_force(shared_hook_stage)
                if shared_hook_stage.exists():
                    shutil.rmtree(shared_hook_stage, ignore_errors=True)
            except Exception as exc:
                run_warnings.append(f"rmtree_hookstage: {type(exc).__name__}: {exc}")
        try:
            rmtree_force(shared_uv_cache)
            if shared_uv_cache.exists():
                shutil.rmtree(shared_uv_cache, ignore_errors=True)
        except Exception as exc:
            run_warnings.append(f"rmtree_uvcache: {type(exc).__name__}: {exc}")

    overall_term = _aggregate_terminal([r["terminal_status"] for r in host_reports.values()])
    bundle_sha = _sha256_if_exists((wheelhouse_stage / "mindie-wheels-bundle.tar.gz") if wheelhouse_stage else None)
    channel_sha = _sha256_if_exists((wheelhouse_stage / "release-channel.json") if wheelhouse_stage else None)
    return {
        "schema": "mindie-host-acceptance/1",
        "status": _public_status(overall_term),
        "terminal_status": overall_term,
        "platform": sys.platform,
        "timeout_budget_sec": float(timeout_sec),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "bundle_sha256": bundle_sha,
        "channel_sha256": channel_sha,
        "scenarios_requested": list(clean_scenarios),
        "hosts": host_reports,
        "summary": {
            "total_hosts": len(host_reports),
            "passed_hosts": sum(1 for r in host_reports.values() if r["terminal_status"] == "completed_pass"),
            "failed_hosts": sum(1 for r in host_reports.values() if r["terminal_status"] == "completed_fail"),
            "infra_failed_hosts": sum(
                1 for r in host_reports.values() if r["terminal_status"] in {"infra_failed", "unknown"}
            ),
        },
        "cleanup_warning": [
            *[w for r in host_reports.values() for w in r.get("cleanup_warning", ())],
            *run_warnings,
        ],
    }


def add_host_accept_subparser(commands: Any) -> argparse.ArgumentParser:
    sub = commands.add_parser(
        "host-accept",
        help="run Gate Layer 2 native manifest, MCP wire, hook capture, and upgrade acceptance across 7 hosts",
    )
    target = sub.add_mutually_exclusive_group(required=True)
    target.add_argument("--host", choices=list(ALL_HOSTS), help="single host plugin to verify")
    target.add_argument("--all-hosts", action="store_true", help="verify all 7 host plugins")
    sub.add_argument("--plugin-dir", type=Path, default=None)
    sub.add_argument("--plugins-root", type=Path, default=None)
    sub.add_argument("--wheelhouse-stage", type=Path, default=None)
    sub.add_argument("--baseline-wheelhouse-stage", type=Path, default=None)
    sub.add_argument("--baseline-plugin-dir", type=Path, default=None)
    sub.add_argument("--install-script", type=Path, default=None)
    sub.add_argument("--redactor", type=Path, default=None)
    sub.add_argument("--scenarios", type=str, default=",".join(ALL_SCENARIOS))
    sub.add_argument("--timeout", type=float, default=600.0)
    sub.add_argument("--output", type=Path, default=None)
    return sub


def dispatch_host_accept(args: argparse.Namespace) -> int:
    hosts = ALL_HOSTS if args.all_hosts else (args.host,)
    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    report = run_host_acceptance(
        hosts=hosts,
        plugin_dir=args.plugin_dir,
        plugins_root=args.plugins_root,
        wheelhouse_stage=args.wheelhouse_stage,
        baseline_wheelhouse_stage=args.baseline_wheelhouse_stage,
        baseline_plugin_dir=args.baseline_plugin_dir,
        install_script=args.install_script,
        redactor=args.redactor,
        scenarios=scenarios,
        timeout_sec=args.timeout,
    )
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return exit_code_for_report(report)


def main(argv: Sequence[str] | None = None) -> int:
    root = argparse.ArgumentParser(prog="mindie-bench")
    commands = root.add_subparsers(dest="command", required=True)
    add_host_accept_subparser(commands)
    return dispatch_host_accept(root.parse_args(argv))
