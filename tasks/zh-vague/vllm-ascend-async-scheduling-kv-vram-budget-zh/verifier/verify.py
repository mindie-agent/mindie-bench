#!/usr/bin/env python3
"""Behavioral verifier for vllm-ascend-async-scheduling-kv-vram-budget-zh."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ALLOWED_FILES = {
    "vllm_ascend/core/kv_cache_interface.py",
    "vllm_ascend/worker/model_runner_v1.py",
}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


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


def _load_kv_cache_functions(kv_path: Path) -> dict:
    source = kv_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(kv_path))
    wanted_classes = {"AscendAsyncKVCacheBudget"}
    wanted_funcs = {"compute_kv_vram_budget", "solve_sla_safe_concurrency"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted_classes:
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            selected.append(node)
    assert len(selected) == 3, "Missing required budget symbols in vllm_ascend/core/kv_cache_interface.py"
    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"dataclass": dataclass, "cdiv": cdiv}
    exec(compile(mod, str(kv_path), "exec"), ns)
    return ns


def _load_runner_method(runner_path: Path, kv_ns: dict):
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner_path))
    method_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "resolve_async_kv_cache_budget":
                    method_node = item
                    break
    assert method_node is not None, "Missing NPUModelRunner.resolve_async_kv_cache_budget in model_runner_v1.py"
    mod = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = dict(kv_ns)
    exec(compile(mod, str(runner_path), "exec"), ns)
    return ns["resolve_async_kv_cache_budget"]


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
    _check_git_diff(workspace)

    kv_path = workspace / "vllm_ascend/core/kv_cache_interface.py"
    runner_path = workspace / "vllm_ascend/worker/model_runner_v1.py"
    assert kv_path.is_file(), f"Missing {kv_path}"
    assert runner_path.is_file(), f"Missing {runner_path}"

    kv_ns = _load_kv_cache_functions(kv_path)
    compute_kv_vram_budget = kv_ns["compute_kv_vram_budget"]
    solve_sla_safe_concurrency = kv_ns["solve_sla_safe_concurrency"]
    resolve_async_kv_cache_budget = _load_runner_method(runner_path, kv_ns)

    # Scenario 1: Ascend 950PR (A5) 64 GiB workload from logs/a5_async_sched_oom.log
    common_kwargs = dict(
        hbm_capacity_bytes=68719476736,
        gpu_memory_utilization=0.90,
        weight_bytes=14495514624,
        workspace_bytes=1610612736,
        kv_bytes_per_token=163840,
        max_seq_len=8192,
        block_size=128,
        delayed_release_steps=1,
        num_speculative_tokens=2,
        runtime_overhead_bytes=1610612736,
    )

    sync_vram = compute_kv_vram_budget(**common_kwargs, async_scheduling=False)
    async_vram = compute_kv_vram_budget(**common_kwargs, async_scheduling=True)

    assert sync_vram.effective_workspace_bytes == 1610612736, (
        f"Expected single-buffered sync workspace 1610612736, got {sync_vram.effective_workspace_bytes}"
    )
    assert async_vram.effective_workspace_bytes == 2 * 1610612736, (
        f"Expected double-buffered async workspace {2 * 1610612736}, got {async_vram.effective_workspace_bytes}"
    )
    assert sync_vram.tokens_per_seq_padded == 8192
    assert async_vram.tokens_per_seq_padded == 8320, (
        f"Expected async padded tokens 8320 (8192 + 3 lookahead aligned to 128), got {async_vram.tokens_per_seq_padded}"
    )
    assert sync_vram.b_vram_max == 32, f"Expected sync b_vram_max=32, got {sync_vram.b_vram_max}"
    assert async_vram.b_vram_max == 30, f"Expected async b_vram_max=30, got {async_vram.b_vram_max}"
    assert async_vram.b_vram_max < sync_vram.b_vram_max

    sla_kwargs = dict(
        tpot_sla_ms=35.0,
        base_step_us=14200.0,
        per_seq_step_us=580.0,
        host_overhead_us=1800.0,
        host_overhead_saved_us=1200.0,
        max_num_seqs=64,
    )
    sync_solved = solve_sla_safe_concurrency(sync_vram, **sla_kwargs)
    async_solved = solve_sla_safe_concurrency(async_vram, **sla_kwargs)

    assert sync_solved.b_latency_max == 32, f"Expected sync b_latency_max=32, got {sync_solved.b_latency_max}"
    assert async_solved.b_latency_max == 34, f"Expected async b_latency_max=34, got {async_solved.b_latency_max}"
    assert async_solved.b_safe == 30, (
        f"Expected async b_safe=min(b_vram_max=30, b_latency_max=34)=30, got {async_solved.b_safe}"
    )
    assert async_solved.hbm_bound is True

    # Scenario 2: Latency-bound regime (tight TPOT SLA 22.0 ms)
    tight_sync = solve_sla_safe_concurrency(sync_vram, **{**sla_kwargs, "tpot_sla_ms": 22.0})
    tight_async = solve_sla_safe_concurrency(async_vram, **{**sla_kwargs, "tpot_sla_ms": 22.0})
    assert tight_sync.b_latency_max == 10
    assert tight_async.b_latency_max == 12
    assert tight_async.b_safe == 12
    assert tight_async.hbm_bound is False

    # Scenario 3: NPUModelRunner.resolve_async_kv_cache_budget end-to-end check
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.90),
        max_model_len=8192,
        block_size=128,
        max_num_reqs=64,
    )
    runner_budget = resolve_async_kv_cache_budget(
        runner,
        hbm_capacity_bytes=68719476736,
        weight_bytes=14495514624,
        workspace_bytes=1610612736,
        kv_bytes_per_token=163840,
        tpot_sla_ms=35.0,
        base_step_us=14200.0,
        per_seq_step_us=580.0,
        host_overhead_us=1800.0,
        host_overhead_saved_us=1200.0,
        runtime_overhead_bytes=1610612736,
        delayed_release_steps=1,
    )
    assert runner_budget.async_scheduling is True
    assert runner_budget.b_vram_max == 30
    assert runner_budget.b_latency_max == 34
    assert runner_budget.b_safe == 30

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "vllm_ascend_async_kv_vram_budget",
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
