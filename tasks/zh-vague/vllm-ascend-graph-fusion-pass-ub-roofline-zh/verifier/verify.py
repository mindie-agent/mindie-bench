#!/usr/bin/env python3
"""Behavioral verifier for vllm-ascend-graph-fusion-pass-ub-roofline-zh."""
from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ALLOWED_FILES = {
    "vllm_ascend/compilation/graph_fusion_pass_manager.py",
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
        assert path in ALLOWED_FILES, f"Unexpected modified file outside allowed target_files: {path}"


def _load_fusion_symbols(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    method_node = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "FusionPlanCandidate":
            selected.append(node)
        elif isinstance(node, ast.FunctionDef):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "GraphFusionPassManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "select_optimal_fusion_plan":
                    method_node = item
                    break
    assert method_node is not None, (
        "Missing fusion pass symbols in vllm_ascend/compilation/graph_fusion_pass_manager.py"
    )
    mod = ast.Module(body=[*selected, method_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, Any] = {"dataclass": dataclass, "Sequence": Sequence, "Any": Any, "math": math}
    exec(compile(mod, str(path), "exec"), ns)
    return ns


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
    _check_git_diff(workspace)

    mgr_path = workspace / "vllm_ascend/compilation/graph_fusion_pass_manager.py"
    assert mgr_path.is_file(), f"Missing {mgr_path}"

    ns = _load_fusion_symbols(mgr_path)
    UB_CAPACITY_BYTES = ns["UB_CAPACITY_BYTES"]
    FusionPlanCandidate = ns["FusionPlanCandidate"]
    compute_ub_working_set_bytes = ns["compute_ub_working_set_bytes"]
    evaluate_fusion_plan_roofline = ns["evaluate_fusion_plan_roofline"]
    select_optimal_fusion_plan = ns["select_optimal_fusion_plan"]

    assert UB_CAPACITY_BYTES == 196608

    # Scenario 1: Exact 3 candidates from logs/fusion_pass_ub_overflow.log
    c_2op = FusionPlanCandidate(
        name="norm_quant_2op",
        fused_ops=2,
        tile_tokens=8,
        hidden_size=4096,
        activation_dtype_bytes=2,
        scratch_buffers=1,
        double_buffer=True,
        t_cube_us=12.0,
        t_vector_us=18.0,
        t_mte_hbm_us=22.0,
        t_unoverlapped_tail_us=3.0,
        unfused_latency_us=34.0,
    )
    c_3op = FusionPlanCandidate(
        name="qknorm_rope_3op",
        fused_ops=3,
        tile_tokens=6,
        hidden_size=4096,
        activation_dtype_bytes=2,
        scratch_buffers=1,
        double_buffer=True,
        t_cube_us=15.0,
        t_vector_us=16.0,
        t_mte_hbm_us=19.0,
        t_unoverlapped_tail_us=2.0,
        unfused_latency_us=34.0,
    )
    c_4op_overflow = FusionPlanCandidate(
        name="aggressive_norm_quant_matmul_swiglu_4op",
        fused_ops=4,
        tile_tokens=16,
        hidden_size=4096,
        activation_dtype_bytes=2,
        scratch_buffers=1,
        double_buffer=True,
        t_cube_us=14.0,
        t_vector_us=15.0,
        t_mte_hbm_us=28.0,
        t_unoverlapped_tail_us=4.0,
        unfused_latency_us=34.0,
    )

    assert compute_ub_working_set_bytes(c_2op) == 131072
    assert compute_ub_working_set_bytes(c_3op) == 98304
    assert compute_ub_working_set_bytes(c_4op_overflow) == 262144, (
        f"Expected double-buffered UB 262144 bytes, got {compute_ub_working_set_bytes(c_4op_overflow)}"
    )

    ev_2op = evaluate_fusion_plan_roofline(c_2op)
    ev_3op = evaluate_fusion_plan_roofline(c_3op)
    ev_4op = evaluate_fusion_plan_roofline(c_4op_overflow)

    assert ev_2op["ub_feasible"] is True
    assert math.isclose(ev_2op["critical_path_us"], 25.0, rel_tol=1e-6)
    assert ev_2op["dominant_pipe"] == "mte_hbm"
    assert math.isclose(ev_2op["saving_us"], 9.0, rel_tol=1e-6)

    assert ev_3op["ub_feasible"] is True
    assert math.isclose(ev_3op["critical_path_us"], 21.0, rel_tol=1e-6)
    assert math.isclose(ev_3op["saving_us"], 13.0, rel_tol=1e-6)

    assert ev_4op["ub_feasible"] is False, f"262144 B > 196608 B must be ub_feasible=False: {ev_4op}"
    assert math.isclose(ev_4op["critical_path_us"], 32.0, rel_tol=1e-6)
    assert ev_4op["is_beneficial"] is False

    mgr_stub = SimpleNamespace()
    chosen = select_optimal_fusion_plan(mgr_stub, [c_2op, c_3op, c_4op_overflow])
    assert chosen["selected_plan"] == "qknorm_rope_3op", (
        f"Expected 'qknorm_rope_3op', got {chosen['selected_plan']}"
    )

    # Scenario 2: All candidates overflow UB or degrade latency -> returns None
    c_slow = FusionPlanCandidate(
        name="slow_fusion",
        fused_ops=2,
        tile_tokens=4,
        hidden_size=4096,
        activation_dtype_bytes=2,
        scratch_buffers=1,
        double_buffer=False,
        t_cube_us=10.0,
        t_vector_us=30.0,
        t_mte_hbm_us=12.0,
        t_unoverlapped_tail_us=6.0,
        unfused_latency_us=34.0,
    )
    none_chosen = select_optimal_fusion_plan(mgr_stub, [c_4op_overflow, c_slow])
    assert none_chosen["selected_plan"] is None, f"Expected None when no viable plan exists, got {none_chosen}"

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "vllm_ascend_graph_fusion_pass_ub_roofline",
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
