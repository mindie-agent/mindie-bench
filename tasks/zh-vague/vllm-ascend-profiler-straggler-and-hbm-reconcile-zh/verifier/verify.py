#!/usr/bin/env python3
"""Behavioral verifier for vllm-ascend-profiler-straggler-and-hbm-reconcile-zh."""
from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ALLOWED_FILES = {
    "vllm_ascend/profiling_config.py",
    "vllm_ascend/worker/worker.py",
}


def _check_git_diff(workspace: Path) -> None:
    assert not (workspace / "vllm_ascend_diag").exists(), "Forbidden stub directory vllm_ascend_diag exists"
    res = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    changed = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    for path in changed:
        assert path in ALLOWED_FILES or path.startswith("tests/"), (
            f"Unexpected modified file outside allowed target_files: {path}"
        )


def _load_profiling_symbols(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    wanted = {
        "merge_intervals_ns",
        "interval_union_length_ns",
        "analyze_msprof_rank_db",
        "identify_multi_rank_straggler",
        "reconcile_hbm_memory_layers",
    }
    selected = [
        node
        for node in tree.body
        if getattr(node, "lineno", 0) >= 590 and isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
    ]
    assert sum(1 for node in selected if isinstance(node, ast.FunctionDef) and node.name in wanted) == 5, (
        "Missing straggler/HBM functions in vllm_ascend/profiling_config.py"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod = ast.Module(body=[future, *selected], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, Any] = {
        "Path": Path,
        "Any": Any,
        "json": json,
        "math": math,
        "os": os,
        "sqlite3": __import__("sqlite3"),
    }
    exec(compile(mod, str(path), "exec"), ns)
    return ns


def _load_worker_method(path: Path, prof_ns: dict):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    method_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NPUWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "reconcile_profiled_memory_and_straggler":
                    method_node = item
                    break
    assert method_node is not None, "Missing NPUWorker.reconcile_profiled_memory_and_straggler in worker.py"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod = ast.Module(body=[future, method_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = dict(prof_ns)
    exec(compile(mod, str(path), "exec"), ns)
    return ns["reconcile_profiled_memory_and_straggler"]


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
    _check_git_diff(workspace)

    prof_path = workspace / "vllm_ascend/profiling_config.py"
    worker_path = workspace / "vllm_ascend/worker/worker.py"
    assert prof_path.is_file(), f"Missing {prof_path}"
    assert worker_path.is_file(), f"Missing {worker_path}"

    prof_ns = _load_profiling_symbols(prof_path)
    reconcile_worker = _load_worker_method(worker_path, prof_ns)

    trace_root = workspace / "traces/tp2_decode_msprof"
    snap_path = trace_root / "hbm_snapshot.json"

    res = reconcile_worker(SimpleNamespace(), str(trace_root), str(snap_path))
    straggler = res["straggler"]
    hbm = res["hbm_reconciliation"]

    # Scenario 1: Multi-rank msprof straggler attribution
    assert straggler["straggler_rank_id"] == "rank_1", (
        f"Expected true straggler 'rank_1', got {straggler['straggler_rank_id']}"
    )
    r0 = straggler["per_rank"]["rank_0"]
    r1 = straggler["per_rank"]["rank_1"]

    assert r0["compute_union_ns"] == 1_200_000
    assert r0["comm_union_ns"] == 3_400_000
    assert r0["busy_union_ns"] == 3_500_000, f"Expected rank_0 busy_union_ns=3500000, got {r0['busy_union_ns']}"
    assert r0["comm_overlapped_ns"] == 1_100_000, (
        f"Expected rank_0 comm_overlapped_ns=1100000, got {r0['comm_overlapped_ns']}"
    )
    assert r0["exposed_comm_ns"] == 2_300_000
    assert r0["comm_wait_ns"] == 2_000_000
    assert r0["active_work_ns"] == 1_500_000, (
        f"Expected rank_0 active_work_ns=1500000 after deducting barrier waitNs, got {r0['active_work_ns']}"
    )

    assert r1["compute_union_ns"] == 3_200_000
    assert r1["comm_union_ns"] == 300_000
    assert r1["busy_union_ns"] == 3_500_000
    assert r1["exposed_comm_ns"] == 300_000
    assert r1["comm_wait_ns"] == 0
    assert r1["active_work_ns"] == 3_500_000

    # Scenario 2: Non-additive HBM memory reconciliation
    assert math.isclose(hbm["accounted_physical_mb"], 50400.0, rel_tol=1e-6), (
        f"Expected accounted_physical_mb=50400.0 (44000.0 + 6400.0), got {hbm['accounted_physical_mb']}"
    )
    assert math.isclose(hbm["unaccounted_residual_mb"], 1600.0, rel_tol=1e-6), (
        f"Expected positive unaccounted_residual_mb=1600.0, got {hbm['unaccounted_residual_mb']}"
    )
    assert hbm["unaccounted_residual_mb"] >= 0.0

    # Scenario 3: Separate HCCL pool not included in CANN reserved (hccl_in_cann_reserved=False)
    reconcile_hbm = prof_ns["reconcile_hbm_memory_layers"]
    separate_hccl = reconcile_hbm(
        {
            "device_used_mb": 52000.0,
            "torch_reserved_mb": 44000.0,
            "torch_allocated_mb": 41500.0,
            "cann_runtime_reserved_mb": 4000.0,
            "hccl_comm_buffers_mb": 2400.0,
            "hccl_in_cann_reserved": False,
        }
    )
    assert math.isclose(separate_hccl["accounted_physical_mb"], 50400.0, rel_tol=1e-6)
    assert math.isclose(separate_hccl["unaccounted_residual_mb"], 1600.0, rel_tol=1e-6)

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "vllm_ascend_profiler_straggler_and_hbm_reconcile",
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"reward": 0.0, "failure_reason": str(exc), "failure_type": type(exc).__name__}))
        raise SystemExit(1)
