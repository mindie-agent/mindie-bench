"""Scenario 4: `n-1` baseline -> candidate wheelhouse & plugin upgrade cutover.

[Role]
Verifies live generation upgrade cutover (`update_runtime` + `ensure_knowledge_binding` +
`maintain_generations`) across all 7 native host plugins:
- Preserves `mindie-consent.json` byte-for-byte;
- Preserves custom `component_commands` wrappers while rebinding standard `-m mindie_sim`;
- Retires `gen-n1` knowledge engine & heals deleted `domain_configs` entries so Turn-1
  `knowledge_query` serves `vllm-ascend/current-unit-002` (never `retired-unit-001`);
- Enforces `Active + Rollback <= 2` generation retention and Windows `pythonw.exe` GUI subsystem.

[Invariants]
- Runs with fail-closed `git` (`exit 99`) trap on `PATH` and zero `os.environ` mutation.
- Always precedes `except OSError:` with `except TimeoutError: raise`.

[Timeouts & Locks]
- Honors caller monotonic `deadline`; serializes `<host>.json` via `mindie_harness` locks.

[Side-Effect Policy]
- Writes baseline/candidate generations and seeded stores strictly inside `sandbox_root`.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mindie_bench.host_acceptance import (
    HostAcceptanceError,
    build_sandbox_env,
    install_git_trap,
    resolve_base_python,
    stop_sandbox_knowledge_services,
    unpack_wheelhouse_links,
)
from mindie_bench.host_acceptance.manifests import parse_host_manifest
from mindie_bench.host_acceptance.mcp_wire import (
    MCPWireSession,
    boot_all_servers,
    find_server_for_tool,
)
from mindie_bench.runner.host_platform import (
    subprocess_window_kwargs,
    which_native,
    write_executable_shim,
)


def _pe_subsystem(exe_path: Path) -> int:
    header = exe_path.read_bytes()[:4096]
    if len(header) < 256 or header[:2] != b"MZ":
        raise HostAcceptanceError(f"Invalid MZ header on {exe_path}")
    pe_off = struct.unpack_from("<I", header, 0x3C)[0]
    if header[pe_off : pe_off + 4] != b"PE\x00\x00":
        raise HostAcceptanceError(f"Invalid PE signature on {exe_path}")
    return int(struct.unpack_from("<H", header, pe_off + 24 + 68)[0])


def _install_generation(
    gen_dir: Path, links_dir: Path, env: Mapping[str, str], is_offline: bool, validated_at: float, deadline: float
) -> Path:
    uv_bin = which_native("uv", path=env.get("PATH")) or which_native("uv")
    if not uv_bin:
        raise FileNotFoundError("`uv` executable is required on PATH for Scenario 4")
    venv_dir = gen_dir / ".venv"
    venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    gen_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [uv_bin, "venv", "--python", str(resolve_base_python()), str(venv_dir)],
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(2.0, deadline - time.monotonic()),
        check=True,
        **subprocess_window_kwargs(),
    )
    req = links_dir / "requirements.txt"
    cmd = [uv_bin, "--native-tls", "pip", "install", "--python", str(venv_py), "--find-links", str(links_dir)]
    cmd.extend(["--override", str(req), "-r", str(req)] if req.is_file() else [str(w) for w in sorted(links_dir.glob("*.whl"))])
    if is_offline:
        cmd.extend(["--offline", "--no-index"])
    subprocess.run(
        cmd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(2.0, deadline - time.monotonic()),
        check=True,
        **subprocess_window_kwargs(),
    )
    (gen_dir / "runtime.json").write_text(
        json.dumps(
            {
                "schema": "mindie-candidate-runtime/1",
                "python": str(venv_py),
                "validation": {"status": "validated", "validated_at": validated_at},
            }
        ),
        encoding="utf-8",
    )
    return venv_py


def _seed_store_unit(
    venv_py: Path, store_root: Path, unit_id: str, title: str, env: Mapping[str, str], deadline: float
) -> None:
    code = (
        "import sys\nfrom pathlib import Path\n"
        "from mindie_knowledge.graded.units import GradedUnit\nfrom mindie_knowledge.loop.store import Store\n"
        "root, uid, title = Path(sys.argv[1]), sys.argv[2], sys.argv[3]\n"
        "unit = GradedUnit.from_parts({\n"
        "  'schema': 'mindie-graded-unit/1', 'id': uid, 'revision': 1, 'tier': 'L2', 'category': 'sop',\n"
        "  'evidence_strength': 'software_contract', 'title': title, 'domain_path': 'serving',\n"
        "  'inherits_from': [], 'conditions': {'workload_scope': 'general'},\n"
        "  'triggers': {'error_signatures': [], 'keywords': ['unit', 'aclgraph']},\n"
        "  'provenance': {'source_type': 'doc', 'source_locator': 'mindie-agent/knowledge-vllm-ascend@1111111:serving/u.md', 'evidence_refs': []}\n"
        "}, f'Verified rule body for {title} in unit upgrade acceptance.')\n"
        "st = Store(root, domain='vllm-ascend')\n"
        "try:\n    st.graded.install([unit], origin_tier='local-only')\nfinally:\n    st.close()\n"
    )
    proc = subprocess.run(
        [str(venv_py), "-I", "-X", "utf8", "-c", code, str(store_root), unit_id, title],
        env=dict(env),
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
        raise HostAcceptanceError(f"Failed seeding graded unit {title}: {proc.stderr[:400]}")


def run_scenario_upgrade_cutover(
    host: str,
    plugin_dir: Path,
    *,
    sandbox_root: Path,
    wheelhouse_stage: Path | None,
    baseline_wheelhouse_stage: Path | None,
    baseline_plugin_dir: Path | None,
    redactor: Path | None,
    prepacked_dir: Path | None,
    deadline: float,
    cleanup_warnings: list[str],
) -> dict[str, Any]:
    """Scenario 4: Baseline `n-1` (+ `n-2`) -> candidate wheelhouse cutover & Turn-1 query."""
    if wheelhouse_stage is None or not wheelhouse_stage.is_dir():
        raise FileNotFoundError("Scenario `upgrade_cutover` requires `--wheelhouse-stage`")

    raw_env, is_offline = build_sandbox_env(sandbox_root, wheelhouse_stage=wheelhouse_stage, redactor=redactor)
    env = install_git_trap(sandbox_root / "git-trap", raw_env)
    cand_links = Path(env["UV_FIND_LINKS"])
    if baseline_wheelhouse_stage is not None and baseline_wheelhouse_stage.is_dir():
        base_links, base_self = unpack_wheelhouse_links(baseline_wheelhouse_stage, sandbox_root / "baseline-links")
        base_offline = is_offline and base_self
    else:
        base_links, base_offline = cand_links, is_offline

    runtimes_host = Path(env["MINDIE_DATA_HOME"]) / "runtimes" / host
    n2_dir = runtimes_host / "gen-n2"
    n2_py = n2_dir / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    n2_py.parent.mkdir(parents=True, exist_ok=True)
    n2_py.write_bytes(b"idle-n2")
    (n2_dir / "runtime.json").write_text(
        json.dumps({"schema": "mindie-candidate-runtime/1", "python": str(n2_py), "validation": {"status": "validated", "validated_at": 100.0}}),
        encoding="utf-8",
    )
    n1_dir = runtimes_host / "gen-n1"
    n1_py = _install_generation(n1_dir, base_links, env, base_offline, 200.0, deadline)

    spec = parse_host_manifest(
        host, plugin_dir, sandbox_root=sandbox_root, env=env, prepacked_dir=prepacked_dir, timeout=max(2.0, deadline - time.monotonic())
    )
    base_src = baseline_plugin_dir.resolve() if baseline_plugin_dir and baseline_plugin_dir.is_dir() else spec.plugin_dir

    retired_root = n1_dir / "retired-knowledge"
    active_root = Path(env["MINDIE_DATA_HOME"]) / "knowledge"
    _seed_store_unit(n1_py, retired_root, "unit-retired-001", "vllm-ascend/retired-unit-001", env, deadline)
    _seed_store_unit(n1_py, active_root, "unit-current-002", "vllm-ascend/current-unit-002", env, deadline)

    cfg_dir = Path(env["MINDIE_CONFIG_HOME"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    consent_file = cfg_dir / "mindie-consent.json"
    consent_payload = (
        json.dumps(
            {
                "schema": "mindie-consent/1",
                "choice": "local-only",
                "enabled": True,
                "choice_revision": "upgrade-consent-001",
                "project_roots": [str(sandbox_root / "ws")],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    consent_file.write_bytes(consent_payload)

    retired_eng = n1_dir / "retired.engine.json"
    retired_eng.write_text(
        json.dumps(
            {
                "root": str(retired_root),
                "domain": "vllm-ascend",
                "capture_mode": "disabled",
                "admission_path": str(Path(env["MINDIE_DATA_HOME"]) / "state" / host / "admission.sqlite3"),
                "community_config": str(cfg_dir / "mindie-community.json"),
            }
        ),
        encoding="utf-8",
    )
    deleted_eng = n1_dir / "deleted-operators.engine.json"
    deleted_eng.write_text("{}", encoding="utf-8")
    deleted_eng.unlink()

    wrapper_dir = sandbox_root / "custom-wrappers"
    custom_prof = write_executable_shim(wrapper_dir, "custom-profiler", "import sys; print('custom-profiler')\n")
    cfg_file = cfg_dir / f"{host}.json"
    cfg_file.write_text(
        json.dumps(
            {
                "schema": "mindie-host-config/1",
                "host": host,
                "python": str(n1_py),
                "component_release": str(n1_dir),
                "source_root": str(base_src),
                "runtime_scripts": str(base_src / "scripts"),
                "state_dir": str(Path(env["MINDIE_DATA_HOME"]) / "state" / host),
                "engine_config": str(retired_eng),
                "domain_configs": {"vllm-ascend": str(retired_eng), "operators": str(deleted_eng)},
                "component_commands": {
                    "profiler": [str(custom_prof), "--wrapped"],
                    "sim": [str(n1_py), "-I", "-X", "utf8", "-m", "mindie_sim"],
                },
                "product_validation": {
                    "schema": "mindie-candidate-check/1",
                    "status": "validated",
                    "native_version": "0.0.1",
                    "components": {"mindie-harness": {"version": "0.0.1", "revision": "0" * 40}},
                },
            }
        ),
        encoding="utf-8",
    )

    cutover_n1_code = (
        "import json, os, sys\nfrom pathlib import Path\n"
        "from mindie_harness.paths import ProfileRef\n"
        "from mindie_harness.runtime.wheelhouse import update_runtime\n"
        "host, plugin_root, stage_uri, offline = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4] == '1'\n"
        "profile = ProfileRef.resolve(host, plugin_root=plugin_root)\n"
        "up = update_runtime(profile, force=True, offline=offline, channel_base_url=stage_uri, env=os.environ)\n"
        "print(json.dumps({'update': up}))\n"
    )
    cproc = subprocess.run(
        [
            str(n1_py),
            "-I",
            "-X",
            "utf8",
            "-c",
            cutover_n1_code,
            host,
            str(spec.plugin_dir),
            wheelhouse_stage.resolve().as_uri(),
            "1" if is_offline else "0",
        ],
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
    if cproc.returncode != 0:
        raise HostAcceptanceError(f"{host} upgrade cutover failed ({cproc.returncode}): {cproc.stderr[:400]}")
    cutover_res = json.loads(cproc.stdout.strip().splitlines()[-1])
    if cutover_res["update"].get("status") != "active":
        raise HostAcceptanceError(f"{host} update_runtime did not activate new generation: {cutover_res['update']}")

    cand_finalize_py = Path(cutover_res["update"]["python"])
    finalize_cand_code = (
        "import json, sys\nfrom pathlib import Path\n"
        "from mindie_harness.paths import ProfileRef\n"
        "from mindie_harness.runtime.activation import maintain_generations\n"
        "try:\n"
        "    from mindie_harness.runtime.knowledge_binding import ensure_knowledge_binding\n"
        "except ImportError:\n"
        "    ensure_knowledge_binding = None\n"
        "host, plugin_root = sys.argv[1], Path(sys.argv[2])\n"
        "profile = ProfileRef.resolve(host, plugin_root=plugin_root)\n"
        "kb = ensure_knowledge_binding(profile, plugin_root=plugin_root, python_executable=sys.executable, retire_previous=True) if ensure_knowledge_binding else {'status': 'legacy'}\n"
        "cfg = json.loads(profile.config_path.read_text(encoding='utf-8'))\n"
        "cfg['source_root'] = str(plugin_root); cfg['runtime_scripts'] = str(plugin_root / 'scripts')\n"
        "profile.config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "gc = maintain_generations(profile)\n"
        "print(json.dumps({'kb': kb, 'gc': gc}))\n"
    )
    fproc = subprocess.run(
        [str(cand_finalize_py), "-I", "-X", "utf8", "-c", finalize_cand_code, host, str(spec.plugin_dir)],
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
    if fproc.returncode != 0:
        raise HostAcceptanceError(f"{host} post-cutover candidate finalize failed ({fproc.returncode}): {fproc.stderr[:400]}")

    if consent_file.read_bytes() != consent_payload:
        raise HostAcceptanceError(f"{host} mindie-consent.json was mutated during upgrade cutover")

    def _norm_venv_py(p: Path) -> Path:
        return p.parent.resolve() / p.name

    after_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    cand_py = Path(after_cfg["python"])
    if _norm_venv_py(cand_py) == _norm_venv_py(n1_py):
        raise HostAcceptanceError(f"{host} python did not switch off baseline generation {n1_py}")
    if after_cfg["component_commands"]["profiler"] != [str(custom_prof), "--wrapped"]:
        raise HostAcceptanceError(f"{host} custom profiler wrapper was clobbered: {after_cfg['component_commands']}")
    if _norm_venv_py(Path(after_cfg["component_commands"]["sim"][0])) != _norm_venv_py(cand_py):
        raise HostAcceptanceError(f"{host} standard sim command did not rebind to {cand_py}")

    retained_gens = [
        d.name
        for d in runtimes_host.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("trash")
    ]
    if len(retained_gens) > 2:
        raise HostAcceptanceError(f"{host} retained >2 runtime generations after GC: {retained_gens}")
    if os.name == "nt":
        pyw = cand_py.with_name("pythonw.exe")
        if not pyw.is_file() or _pe_subsystem(pyw) != 2:
            raise HostAcceptanceError(f"{host} Windows pythonw.exe missing or non-GUI subsystem: {pyw}")

    post_spec = parse_host_manifest(
        host, plugin_dir, sandbox_root=sandbox_root, env=env, prepacked_dir=prepacked_dir, timeout=max(2.0, deadline - time.monotonic())
    )
    sessions: dict[str, MCPWireSession] = {}
    try:
        sessions, listed = boot_all_servers(post_spec, deadline)
        kq = find_server_for_tool(sessions, listed, "knowledge_query").call_tool(
            "knowledge_query", {"query": "unit", "domain": "vllm-ascend"}, deadline
        )
        kq_blob = json.dumps(kq, ensure_ascii=False)
        if (
            kq.get("isError") is not False
            or "vllm-ascend/current-unit-002" not in kq_blob
            or "retired-unit-001" in kq_blob
            or "FileNotFoundError" in kq_blob
        ):
            raise HostAcceptanceError(f"{host} post-cutover knowledge_query returned wrong unit: {kq}")
        sim_res = find_server_for_tool(sessions, listed, "sim").call_tool(
            "sim", {"argv": ["catalog"]}, deadline
        )
        if sim_res.get("isError") is not False:
            raise HostAcceptanceError(f"{host} post-cutover sim failed: {sim_res}")
        return {"retained_generations": sorted(retained_gens), "candidate_python": str(cand_py)}
    finally:
        for sname, sess in sessions.items():
            try:
                sess.close()
            except Exception as exc:
                cleanup_warnings.append(f"close({sname}): {type(exc).__name__}: {exc}")
        try:
            stop_sandbox_knowledge_services(sandbox_root)
        except Exception as exc:
            cleanup_warnings.append(f"stop_knowledge: {type(exc).__name__}: {exc}")
