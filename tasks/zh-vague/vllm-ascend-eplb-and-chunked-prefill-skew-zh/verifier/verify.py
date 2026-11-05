#!/usr/bin/env python3
"""Behavioral verifier for vllm-ascend-eplb-and-chunked-prefill-skew-zh."""
from __future__ import annotations

import ast
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


ALLOWED_FILES = {
    "vllm_ascend/eplb/eplb_updator.py",
    "vllm_ascend/ascend_config.py",
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


def _load_eplb_symbols(updator_path: Path) -> dict:
    source = updator_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(updator_path))
    selected = []
    method_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "evaluate_dp_barrier_skew",
            "should_trigger_eplb_rebalance",
        }:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "EplbUpdator":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "evaluate_rebalance_and_skew":
                    method_node = item
                    break
    assert len(selected) == 2 and method_node is not None, (
        "Missing skew/rebalance symbols in vllm_ascend/eplb/eplb_updator.py"
    )
    mod = ast.Module(body=[*selected, method_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, Any] = {"Sequence": Sequence, "Any": Any}
    exec(compile(mod, str(updator_path), "exec"), ns)
    return ns


def _load_config_symbols(config_path: Path) -> dict:
    source = config_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(config_path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"ProfilingChunkConfig", "EplbConfig"}:
            selected.append(node)
    assert len(selected) == 2, "Missing ProfilingChunkConfig or EplbConfig in vllm_ascend/ascend_config.py"
    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    logger_stub = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    ns: dict[str, Any] = {"os": os, "logger": logger_stub, "Any": Any}
    exec(compile(mod, str(config_path), "exec"), ns)
    return ns


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
    _check_git_diff(workspace)

    updator_path = workspace / "vllm_ascend/eplb/eplb_updator.py"
    config_path = workspace / "vllm_ascend/ascend_config.py"
    assert updator_path.is_file(), f"Missing {updator_path}"
    assert config_path.is_file(), f"Missing {config_path}"

    eplb_ns = _load_eplb_symbols(updator_path)
    cfg_ns = _load_config_symbols(config_path)

    evaluate_dp_barrier_skew = eplb_ns["evaluate_dp_barrier_skew"]
    should_trigger_eplb_rebalance = eplb_ns["should_trigger_eplb_rebalance"]
    evaluate_rebalance_and_skew = eplb_ns["evaluate_rebalance_and_skew"]
    ProfilingChunkConfig = cfg_ns["ProfilingChunkConfig"]
    EplbConfig = cfg_ns["EplbConfig"]

    # Scenario 1: Multi-DP AllToAllV straggler barrier wait from logs/dsv3_dp_skew_sla.log
    skew = evaluate_dp_barrier_skew(
        dp_compute_us=[18000.0, 19500.0, 21000.0, 31500.0],
        dp_tokens=[16, 16, 20, 48],
        alltoallv_us_per_token=120.0,
    )
    assert math.isclose(skew["t_compute_max_us"], 31500.0, rel_tol=1e-6), f"Bad t_compute_max_us: {skew}"
    assert math.isclose(skew["t_alltoallv_sync_us"], 5760.0, rel_tol=1e-6), f"Bad t_alltoallv_sync_us: {skew}"
    assert math.isclose(skew["t_sync_us"], 37260.0, rel_tol=1e-6), f"Bad t_sync_us: {skew}"
    assert math.isclose(skew["mean_rank_us"], 25500.0, rel_tol=1e-6), f"Bad mean_rank_us: {skew}"
    assert math.isclose(skew["dp_barrier_wait_us"], 11760.0, rel_tol=1e-6), (
        f"Expected dp_barrier_wait_us=11760.0, got {skew['dp_barrier_wait_us']}"
    )

    # Scenario 2: EPLB rebalance gating on imbalance ratio and net D2D benefit
    eplb_cfg = EplbConfig({"imbalance_trigger_ratio": 1.15, "expert_heat_collection_interval": 50})
    assert math.isclose(float(eplb_cfg.imbalance_trigger_ratio), 1.15, rel_tol=1e-6)

    # 2a: Minor routing jitter (max/mean = 104/101 = 1.0297 < 1.15) -> must NOT rebalance
    jitter = should_trigger_eplb_rebalance(
        expert_loads=[100.0, 104.0, 98.0, 102.0],
        imbalance_trigger_ratio=1.15,
        rebalance_window_steps=50,
        per_step_skew_saving_us=600.0,
        migrated_experts=2,
        d2d_transfer_us_per_expert=2500.0,
    )
    assert jitter["should_rebalance"] is False, f"Jitter must not trigger rebalance: {jitter}"

    # 2b: Hot expert skew (max/mean = 180/120 = 1.50 >= 1.15) with positive net benefit -> MUST rebalance
    hot_worth = should_trigger_eplb_rebalance(
        expert_loads=[100.0, 100.0, 100.0, 180.0],
        imbalance_trigger_ratio=1.15,
        rebalance_window_steps=50,
        per_step_skew_saving_us=600.0,
        migrated_experts=4,
        d2d_transfer_us_per_expert=2500.0,
    )
    assert math.isclose(hot_worth["imbalance_ratio"], 1.5, rel_tol=1e-6)
    assert math.isclose(hot_worth["net_benefit_us"], 20000.0, rel_tol=1e-6)
    assert hot_worth["should_rebalance"] is True, f"Profitable hot skew must trigger rebalance: {hot_worth}"

    # 2c: Hot expert skew where D2D migration cost exceeds short window savings -> must NOT rebalance
    hot_unworth = should_trigger_eplb_rebalance(
        expert_loads=[100.0, 100.0, 100.0, 180.0],
        imbalance_trigger_ratio=1.15,
        rebalance_window_steps=10,
        per_step_skew_saving_us=600.0,
        migrated_experts=4,
        d2d_transfer_us_per_expert=2500.0,
    )
    assert math.isclose(hot_unworth["net_benefit_us"], -4000.0, rel_tol=1e-6)
    assert hot_unworth["should_rebalance"] is False, (
        f"Negative net benefit migration must not trigger rebalance: {hot_unworth}"
    )

    # 2d: EplbUpdator.evaluate_rebalance_and_skew method integration
    updator_stub = SimpleNamespace(eplb_config=eplb_cfg)
    combined = evaluate_rebalance_and_skew(
        updator_stub,
        dp_compute_us=[18000.0, 19500.0, 21000.0, 31500.0],
        dp_tokens=[16, 16, 20, 48],
        alltoallv_us_per_token=120.0,
        expert_loads=[100.0, 100.0, 100.0, 180.0],
        migrated_experts=4,
        d2d_transfer_us_per_expert=2500.0,
    )
    assert math.isclose(combined["dp_barrier_wait_us"], 11760.0, rel_tol=1e-6)
    assert combined["should_rebalance"] is True

    # Scenario 3: ProfilingChunkConfig.solve_sla_chunk_budget
    chunk_cfg = ProfilingChunkConfig({"enabled": True, "min_chunk": 4096})
    budget = chunk_cfg.solve_sla_chunk_budget(
        tpot_sla_ms=50.0,
        decode_step_us=25500.0,
        dp_barrier_wait_us=11760.0,
        prefill_us_per_token=12.0,
        max_num_batched_tokens=4096,
        decode_tokens=100,
        align_block_size=128,
    )
    assert budget == 1024, f"Expected block-aligned chunk budget 1024, got {budget}"

    zero_budget = chunk_cfg.solve_sla_chunk_budget(
        tpot_sla_ms=35.0,
        decode_step_us=25500.0,
        dp_barrier_wait_us=11760.0,
        prefill_us_per_token=12.0,
        max_num_batched_tokens=4096,
        decode_tokens=100,
        align_block_size=128,
    )
    assert zero_budget == 0, f"Expected 0 chunk budget when decode+skew exceeds SLA, got {zero_budget}"

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "vllm_ascend_eplb_and_chunked_prefill_skew",
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
