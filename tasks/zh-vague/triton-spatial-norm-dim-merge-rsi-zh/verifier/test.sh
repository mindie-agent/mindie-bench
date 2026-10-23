#!/bin/bash
set -eu
if [ -n "${MINDIE_BENCH_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_PYTHON}" ]; then
    PYTHON_BIN="${MINDIE_BENCH_PYTHON}"
elif [ -n "${MINDIE_BENCH_CPU_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_CPU_PYTHON}" ]; then
    PYTHON_BIN="${MINDIE_BENCH_CPU_PYTHON}"
elif [ -x "/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python" ]; then
    PYTHON_BIN="/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python"
else
    PYTHON_BIN="python3"
fi
exec "${PYTHON_BIN}" -I -B -c 'import ast, importlib.util, inspect, json, os, sys, types, torch
import torch.nn.functional as F
from pathlib import Path

if importlib.util.find_spec("triton") is None:
    triton_mod = types.ModuleType("triton")
    triton_mod.jit = lambda fn: fn
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.next_power_of_2 = lambda n: 1 if n <= 1 else (1 << (int(n) - 1).bit_length())
    tl_mod = types.ModuleType("triton.language")
    tl_mod.constexpr = int
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = tl_mod

def _arm(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for sub in ("checks", "correctness", "metrics"):
        if isinstance(d.get(sub), dict):
            out.update(d[sub])
    out.update(d)
    return out

def _call_op_arm(mod, x, w, b, num_groups, eps, arm):
    flags = {
        "full_candidate": (True, True, True),
        "minus_dim_merge": (False, True, True),
        "minus_block_scaling": (True, False, True),
        "minus_multibuffer_cast_guard": (True, True, False),
    }[arm]
    fn = mod.spatial_group_norm_reduce
    params = set(inspect.signature(fn).parameters)
    if {"enable_dim_merge", "enable_block_scaling", "enable_multibuffer_cast_guard"}.issubset(params):
        return fn(x, w, b, num_groups=num_groups, eps=eps, enable_dim_merge=flags[0], enable_block_scaling=flags[1], enable_multibuffer_cast_guard=flags[2])
    if {"merge_dims", "scale_block", "guard_multibuffer_cast"}.issubset(params):
        return fn(x, w, b, num_groups=num_groups, eps=eps, merge_dims=flags[0], scale_block=flags[1], guard_multibuffer_cast=flags[2])
    if {"merge_dims", "scale_block", "guard_cast"}.issubset(params):
        return fn(x, w, b, num_groups=num_groups, eps=eps, merge_dims=flags[0], scale_block=flags[1], guard_cast=flags[2])
    if "config" in params and hasattr(mod, "ARM_CONFIGS"):
        return fn(x, w, b, num_groups=num_groups, eps=eps, config=mod.ARM_CONFIGS[arm])
    if hasattr(mod, "_run_configuration"):
        return mod._run_configuration(x, w, b, num_groups, eps, dim_merge=flags[0], block_scaling=flags[1], cast_guard=flags[2])
    raise AssertionError("spatial_group_norm_reduce does not expose separable L4 factor switches")

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    ledger_path = root / "experiment-ledger.jsonl"
    progress_path = root / "PROGRESS.md"
    assert ledger_path.is_file(), "Missing experiment-ledger.jsonl"
    assert progress_path.is_file(), "Missing PROGRESS.md"
    ledger_text = ledger_path.read_text(encoding="utf-8")
    for req_arm in ("full_candidate", "minus_dim_merge", "minus_block_scaling", "minus_multibuffer_cast_guard"):
        assert req_arm in ledger_text, f"experiment-ledger.jsonl missing arm {req_arm}"
    assert len(progress_path.read_text(encoding="utf-8").strip()) >= 40, "PROGRESS.md is empty or too short"

    target = root / "vllm_ascend/ops/spatial_group_norm_reduce.py"
    src = target.read_text(encoding="utf-8")
    tree = ast.parse(src)
    unp = ast.unparse(tree)
    assert "select_optimal_block_size" in unp, "Missing L4 Factor 2 select_optimal_block_size helper"

    spec = importlib.util.spec_from_file_location("spatial_group_norm_reduce", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for req_fn in ("select_optimal_block_size", "compute_persistent_grid", "spatial_group_norm_reduce", "run_ablation_suite"):
        assert hasattr(mod, req_fn), f"Missing {req_fn}() in spatial_group_norm_reduce.py"

    assert mod.select_optimal_block_size(16384, torch.bfloat16, 3) >= 8192, "select_optimal_block_size did not scale BLOCK_SIZE >= 8192 for bf16"
    assert mod.select_optimal_block_size(16384, torch.float32, 3) == 4096, "select_optimal_block_size must cap fp32 BLOCK_SIZE at 4096"
    assert mod.select_optimal_block_size(256, torch.bfloat16, 3) == 512, "select_optimal_block_size must clamp small numel to >= 512"
    assert tuple(mod.compute_persistent_grid(256, 48)) == (48,) and tuple(mod.compute_persistent_grid(12, 48)) == (12,), "compute_persistent_grid failed persistent VectorCore clamp"

    orig_sel = mod.select_optimal_block_size
    orig_grid = mod.compute_persistent_grid
    orig_rsqrt = torch.rsqrt
    sel_calls = []
    grid_calls = []
    rsqrt_calls = []

    def spy_sel(*args, **kwargs):
        res = orig_sel(*args, **kwargs)
        sel_calls.append((args, res))
        return res

    def spy_grid(*args, **kwargs):
        res = orig_grid(*args, **kwargs)
        grid_calls.append((args, res))
        return res

    def spy_rsqrt(t, *args, **kwargs):
        assert isinstance(t, torch.Tensor), "torch.rsqrt expects Tensor"
        rsqrt_calls.append(tuple(t.shape))
        return orig_rsqrt(t, *args, **kwargs)

    mod.select_optimal_block_size = spy_sel
    mod.compute_persistent_grid = spy_grid
    torch.rsqrt = spy_rsqrt
    try:
        suite = mod.run_ablation_suite()
    finally:
        mod.select_optimal_block_size = orig_sel
        mod.compute_persistent_grid = orig_grid
        torch.rsqrt = orig_rsqrt

    assert isinstance(suite, dict), f"run_ablation_suite() must return a dict keyed by arm name, got {type(suite).__name__}"
    for req_arm in ("full_candidate", "minus_dim_merge", "minus_block_scaling", "minus_multibuffer_cast_guard"):
        assert req_arm in suite, f"Missing required leave-one-out arm: {req_arm}"

    assert len(sel_calls) >= 3 and len(grid_calls) >= 4 and len(rsqrt_calls) >= 4, (
        f"run_ablation_suite() did not execute real tensor/tiling passes: sel={len(sel_calls)}, grid={len(grid_calls)}, rsqrt={len(rsqrt_calls)}"
    )

    full = _arm(suite["full_candidate"])
    md = _arm(suite["minus_dim_merge"])
    mb = _arm(suite["minus_block_scaling"])
    mg = _arm(suite["minus_multibuffer_cast_guard"])

    assert (
        full["nested_loop_depth"] == 1
        and full["block_size"] >= 2048
        and full["enable_multibuffer"] is False
        and full["intermediate_gm_copies"] == 0
        and full.get("correctness_ok") is True
        and float(full.get("max_abs_err", 1.0)) <= 5e-3
    ), f"Invalid full_candidate metrics: {full}"
    assert md["nested_loop_depth"] == 3 and md["latency_ms"] > full["latency_ms"], f"Invalid minus_dim_merge metrics: {md}"
    assert mb["block_size"] == 512 and mb["latency_ms"] > full["latency_ms"], f"Invalid minus_block_scaling metrics: {mb}"
    mg_degraded = (
        mg["latency_ms"] > full["latency_ms"]
        or mg.get("correctness_ok") is False
        or float(mg.get("max_abs_err", 0.0)) > float(full.get("max_abs_err", 0.0))
    )
    assert (mg["enable_multibuffer"] is True or mg["intermediate_gm_copies"] > 0) and mg_degraded, f"Invalid minus_multibuffer_cast_guard metrics: {mg}"

    # Independent verifier-constructed tensor probes across two distinct shapes + direct 4-arm ablation check
    gen = torch.Generator().manual_seed(91099)
    for shape, num_groups, min_blk in (((2, 16, 16, 16), 4, 1024), ((3, 32, 24, 24), 8, 2048)):
        c = shape[1]
        x = (torch.randn(shape, generator=gen, dtype=torch.float32) * 1.8 + 0.7).to(torch.bfloat16)
        w = (torch.randn((c,), generator=gen, dtype=torch.float32) * 0.5 + 1.0).to(torch.bfloat16)
        b = (torch.randn((c,), generator=gen, dtype=torch.float32) * 0.2).to(torch.bfloat16)
        ref = F.group_norm(x.float(), num_groups, w.float(), b.float(), eps=1e-5).to(torch.bfloat16)
        out, m = mod.spatial_group_norm_reduce(x, w, b, num_groups=num_groups, eps=1e-5)
        assert isinstance(out, torch.Tensor) and out.shape == shape and out.dtype == torch.bfloat16, f"Invalid output tensor for shape {shape}"
        err = float((out.float() - ref.float()).abs().max().item())
        assert err <= 5e-3 and torch.allclose(out.float(), ref.float(), atol=5e-3, rtol=5e-3), f"Numerical check failed on shape {shape}: max_abs_err={err}"
        assert m["nested_loop_depth"] == 1 and m["enable_multibuffer"] is False and m["intermediate_gm_copies"] == 0 and m["block_size"] >= min_blk, f"Unexpected operator metrics for {shape}: {m}"

    x_ab = (torch.randn((2, 16, 32, 32), generator=gen, dtype=torch.float32) * 2.5 + 1.5).to(torch.bfloat16)
    w_ab = torch.ones((16,), dtype=torch.bfloat16)
    b_ab = torch.zeros((16,), dtype=torch.bfloat16)
    ref_ab = F.group_norm(x_ab.float(), 4, w_ab.float(), b_ab.float(), eps=1e-5).to(torch.bfloat16)
    out_f, m_f = _call_op_arm(mod, x_ab, w_ab, b_ab, 4, 1e-5, "full_candidate")
    out_d, m_d = _call_op_arm(mod, x_ab, w_ab, b_ab, 4, 1e-5, "minus_dim_merge")
    out_b, m_b = _call_op_arm(mod, x_ab, w_ab, b_ab, 4, 1e-5, "minus_block_scaling")
    out_g, m_g = _call_op_arm(mod, x_ab, w_ab, b_ab, 4, 1e-5, "minus_multibuffer_cast_guard")
    err_f = float((out_f.float() - ref_ab.float()).abs().max().item())
    err_d = float((out_d.float() - ref_ab.float()).abs().max().item())
    err_b = float((out_b.float() - ref_ab.float()).abs().max().item())
    err_g = float((out_g.float() - ref_ab.float()).abs().max().item())
    assert err_f <= 5e-3 and err_d <= 5e-3 and err_b <= 5e-3, f"FP32 accumulator arms exceeded 5e-3 error: {err_f}, {err_d}, {err_b}"
    assert err_g > err_f and err_g > 5e-3, f"Disabling cast guard did not trigger bf16 partial truncation error: err_g={err_g}, err_f={err_f}"
    assert m_d["nested_loop_depth"] == 3 and m_d["scalar_iterations"] > m_f["scalar_iterations"], f"minus_dim_merge did not restore 3-level loop: {m_d}"
    assert m_b["block_size"] == 512 and m_b["scalar_iterations"] > m_f["scalar_iterations"], f"minus_block_scaling did not restore block_size=512: {m_b}"

    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
