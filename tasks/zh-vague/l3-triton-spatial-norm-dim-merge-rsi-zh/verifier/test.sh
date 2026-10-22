#!/bin/bash
set -eu
exec python3 -I -B -c 'import ast, importlib.util, json, os, sys, torch
from pathlib import Path

def _arm(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    if isinstance(d.get("checks"), dict):
        out.update(d["checks"])
    if isinstance(d.get("metrics"), dict):
        out.update(d["metrics"])
    out.update(d)
    return out

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    target = root / "vllm_ascend/ops/spatial_group_norm_reduce.py"
    src = target.read_text(encoding="utf-8")
    tree = ast.parse(src)
    unp = ast.unparse(tree)
    assert "select_optimal_block_size" in unp, "Missing L4 Factor 2 select_optimal_block_size helper"

    spec = importlib.util.spec_from_file_location("spatial_group_norm_reduce", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "select_optimal_block_size"), "Missing select_optimal_block_size() function"
    assert mod.select_optimal_block_size(16384, torch.bfloat16, 3) >= 2048, "select_optimal_block_size did not scale BLOCK_SIZE >= 2048 for bf16"
    assert hasattr(mod, "run_ablation_suite"), "Missing run_ablation_suite() in spatial_group_norm_reduce.py"

    suite = mod.run_ablation_suite()
    assert isinstance(suite, dict), f"run_ablation_suite() must return a dict keyed by arm name, got {type(suite).__name__}"
    for req_arm in ("full_candidate", "minus_dim_merge", "minus_block_scaling", "minus_multibuffer_cast_guard"):
        assert req_arm in suite, f"Missing required leave-one-out arm: {req_arm}"

    full = _arm(suite["full_candidate"])
    md = _arm(suite["minus_dim_merge"])
    mb = _arm(suite["minus_block_scaling"])
    mg = _arm(suite["minus_multibuffer_cast_guard"])

    assert full["nested_loop_depth"] == 1 and full["block_size"] >= 2048 and full["enable_multibuffer"] is False and full["intermediate_gm_copies"] == 0 and full.get("correctness_ok") is True, f"Invalid full_candidate metrics: {full}"
    assert md["nested_loop_depth"] == 3 and md["latency_ms"] > full["latency_ms"], f"Invalid minus_dim_merge metrics: {md}"
    assert mb["block_size"] == 512 and mb["latency_ms"] > full["latency_ms"], f"Invalid minus_block_scaling metrics: {mb}"
    assert (mg["enable_multibuffer"] is True or mg["intermediate_gm_copies"] > 0) and mg["latency_ms"] > full["latency_ms"], f"Invalid minus_multibuffer_cast_guard metrics: {mg}"

    ledger_path = root / "experiment-ledger.jsonl"
    progress_path = root / "PROGRESS.md"
    assert ledger_path.is_file(), "Missing experiment-ledger.jsonl"
    assert progress_path.is_file(), "Missing PROGRESS.md"
    ledger_text = ledger_path.read_text(encoding="utf-8")
    for req_arm in ("full_candidate", "minus_dim_merge", "minus_block_scaling", "minus_multibuffer_cast_guard"):
        assert req_arm in ledger_text, f"experiment-ledger.jsonl missing arm {req_arm}"
    assert len(progress_path.read_text(encoding="utf-8").strip()) >= 40, "PROGRESS.md is empty or too short"

    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
