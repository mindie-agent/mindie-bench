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
exec "${PYTHON_BIN}" -I -B -c 'import ast, importlib.util, inspect, json, os, sys, torch
from pathlib import Path

def _arm(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for sub in ("checks", "correctness", "metrics"):
        if isinstance(d.get(sub), dict):
            out.update(d[sub])
    out.update(d)
    return out

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    ledger_path = root / "experiment-ledger.jsonl"
    progress_path = root / "PROGRESS.md"
    assert ledger_path.is_file(), "Missing experiment-ledger.jsonl"
    assert progress_path.is_file(), "Missing PROGRESS.md"
    ledger_text = ledger_path.read_text(encoding="utf-8")
    for req_arm in ("full_candidate", "minus_q_fusion", "minus_view_alias", "minus_device_guard"):
        assert req_arm in ledger_text, f"experiment-ledger.jsonl missing arm {req_arm}"
    assert len(progress_path.read_text(encoding="utf-8").strip()) >= 40, "PROGRESS.md is empty or too short"

    rt_path = root / "vllm_ascend/models/deepseek_v4/_ascend_qli_runtime.py"
    target = root / "vllm_ascend/models/deepseek_v4/sparse_mla_decode.py"
    src = target.read_text(encoding="utf-8")
    tree = ast.parse(src)
    unp = ast.unparse(tree)
    assert "fused_q_dequant_rmsnorm_rope_split" in unp, "Missing L4 Factor 1 fused_q_dequant_rmsnorm_rope_split op call"

    rt_spec = importlib.util.spec_from_file_location("_ascend_qli_runtime_ref", rt_path)
    rt_mod = importlib.util.module_from_spec(rt_spec)
    rt_spec.loader.exec_module(rt_mod)
    rt_mod.install_ascend_ops()

    spec = importlib.util.spec_from_file_location("sparse_mla_decode", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "sparse_mla_decode_step"), "Missing sparse_mla_decode_step() in sparse_mla_decode.py"
    assert hasattr(mod, "run_ablation_suite"), "Missing run_ablation_suite() in sparse_mla_decode.py"

    ascend_ns = torch.ops._C_ascend
    orig_q_fused = ascend_ns.fused_q_dequant_rmsnorm_rope_split
    orig_qli = ascend_ns.npu_quant_lightning_indexer
    orig_sfa = ascend_ns.npu_sparse_flash_attention

    q_fused_calls = []
    qli_calls = []
    sfa_calls = []

    def spy_q_fused(*args, **kwargs):
        assert len(args) >= 6 and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.int8, "Invalid qr_i8 tensor passed to fused_q_dequant_rmsnorm_rope_split"
        out = orig_q_fused(*args, **kwargs)
        q_fused_calls.append({"qr_shape": tuple(args[0].shape), "out_ptr": out.untyped_storage().data_ptr()})
        return out

    def spy_qli(q_idx, k_idx, *args, **kwargs):
        assert isinstance(q_idx, torch.Tensor) and isinstance(k_idx, torch.Tensor), "QLI expects tensor inputs"
        q_is_view = (q_idx._base is not None) and (not q_idx.is_contiguous()) and (q_idx.stride(-1) == 1)
        q_is_cloned = (q_idx._base is None) and q_idx.is_contiguous()
        out = orig_qli(q_idx, k_idx, *args, **kwargs)
        qli_calls.append({
            "q_is_view": q_is_view,
            "q_is_cloned": q_is_cloned,
            "q_ptr": q_idx.untyped_storage().data_ptr(),
            "k_ptr": k_idx.untyped_storage().data_ptr(),
            "has_neg_topk": bool((out < 0).any().item()),
        })
        return out

    def spy_sfa(q, k_cache, v_cache, *args, **kwargs):
        bt = kwargs.get("block_table")
        assert isinstance(bt, torch.Tensor), "block_table must be a Tensor"
        has_neg_bt = bool((bt < 0).any().item())
        try:
            out = orig_sfa(q, k_cache, v_cache, *args, **kwargs)
            sfa_calls.append({"has_neg_bt": has_neg_bt, "raised_invalid_gm": False, "out_shape": tuple(out.shape)})
            return out
        except RuntimeError as exc:
            sfa_calls.append({"has_neg_bt": has_neg_bt, "raised_invalid_gm": "invalid GM address" in str(exc)})
            raise

    ascend_ns.fused_q_dequant_rmsnorm_rope_split = spy_q_fused
    ascend_ns.npu_quant_lightning_indexer = spy_qli
    ascend_ns.npu_sparse_flash_attention = spy_sfa
    try:
        suite = mod.run_ablation_suite()
    finally:
        ascend_ns.fused_q_dequant_rmsnorm_rope_split = orig_q_fused
        ascend_ns.npu_quant_lightning_indexer = orig_qli
        ascend_ns.npu_sparse_flash_attention = orig_sfa

    assert isinstance(suite, dict), f"run_ablation_suite() must return a dict keyed by arm name, got {type(suite).__name__}"
    for req_arm in ("full_candidate", "minus_q_fusion", "minus_view_alias", "minus_device_guard"):
        assert req_arm in suite, f"Missing required leave-one-out arm: {req_arm}"

    assert len(q_fused_calls) >= 3, f"Expected >= 3 fused Q op calls across ablation suite, got {len(q_fused_calls)}"
    assert len(qli_calls) >= 4 and len(qli_calls) > len(q_fused_calls), "minus_q_fusion must execute QLI without calling fused Q op"
    assert sum(1 for c in qli_calls if c["q_is_view"]) >= 3, f"Expected >= 3 zero-copy view calls to QLI, got {qli_calls}"
    assert sum(1 for c in qli_calls if c["q_is_cloned"]) >= 1, f"Expected >= 1 cloned QLI call in minus_view_alias, got {qli_calls}"
    assert sum(1 for c in sfa_calls if not c["has_neg_bt"] and not c["raised_invalid_gm"]) >= 3, f"Expected >= 3 guarded SFA calls, got {sfa_calls}"
    assert sum(1 for c in sfa_calls if c["has_neg_bt"] and c["raised_invalid_gm"]) >= 1, f"Expected minus_device_guard to trigger invalid GM address in SFA, got {sfa_calls}"

    full = _arm(suite["full_candidate"])
    mq = _arm(suite["minus_q_fusion"])
    mv = _arm(suite["minus_view_alias"])
    mg = _arm(suite["minus_device_guard"])

    full_json = json.dumps(suite["full_candidate"])
    mg_json = json.dumps(suite["minus_device_guard"])
    full_ok = (
        full.get("correctness_ok") is True
        or full.get("no_invalid_gm_address") is True
        or full.get("completed") is True
        or full.get("verdict") in ("supported", "keep", "attained")
    ) and "AICore invalid GM address" not in full_json
    assert full["tensor_move_count"] == 0 and full["host_d2h_sync_count"] == 0 and full_ok, f"Invalid full_candidate metrics: {full}"
    assert mq["tensor_move_count"] >= 3 and mq["tpot_ms"] > full["tpot_ms"], f"Invalid minus_q_fusion metrics: {mq}"
    assert mv["tensor_move_count"] >= 2 and mv["tpot_ms"] > full["tpot_ms"], f"Invalid minus_view_alias metrics: {mv}"
    mg_fail = (
        mg.get("invalid_gm_triggered") is True
        or mg.get("invalid_gm_address") is True
        or mg.get("correctness_ok") is False
        or mg.get("no_invalid_gm_address") is False
        or mg.get("completed") is False
        or "invalid GM address" in mg_json
        or "negative block id" in mg_json
    )
    assert mg["host_d2h_sync_count"] >= 1 and mg_fail, f"Invalid minus_device_guard metrics: {mg}"

    # Independent verifier-constructed tensor probe & numerical reference check
    gen = torch.Generator().manual_seed(91042)
    t_len, n_heads, q_lora_rank = 4, 16, 64
    nope_dim, rope_dim, index_dim = 512, 64, 128
    topk, compress_ratio, padded_capacity = 8, 4, 128
    qr_i8 = torch.randint(-32, 32, (t_len, q_lora_rank), generator=gen, dtype=torch.int8)
    qr_scale = torch.linspace(0.02, 0.08, t_len, dtype=torch.float32)
    rms_weight = torch.randn((q_lora_rank,), generator=gen, dtype=torch.float32)
    w_q_absorbed = torch.randn((n_heads * (nope_dim + rope_dim), q_lora_rank), generator=gen, dtype=torch.float32) * 0.02
    angles = torch.randn((t_len, rope_dim), generator=gen, dtype=torch.float32)
    cos, sin = angles.cos(), angles.sin()
    k_idx_cache = torch.randn((padded_capacity, 16, n_heads, index_dim), generator=gen, dtype=torch.float32)
    k_cache = torch.randn((padded_capacity, 16, n_heads, nope_dim + rope_dim), generator=gen, dtype=torch.float32)
    v_cache = torch.randn((padded_capacity, 16, n_heads, nope_dim), generator=gen, dtype=torch.float32)
    weights = torch.ones((t_len, n_heads), dtype=torch.float32)
    q_scale = torch.ones((t_len, n_heads), dtype=torch.float32)
    k_scale = torch.ones((padded_capacity,), dtype=torch.float32)
    block_table = torch.zeros((t_len, padded_capacity), dtype=torch.int32)
    topk_slots = torch.arange(topk, dtype=torch.int64).unsqueeze(0).expand(t_len, topk).contiguous()
    query_start_loc = torch.arange(0, t_len + 1, dtype=torch.int32)
    seqused_k = torch.tensor([64, 12, 32, 8], dtype=torch.int32)

    ref_q = rt_mod._op_q_fused(qr_i8, qr_scale, rms_weight, w_q_absorbed, cos, sin, eps=1e-6, n_heads=n_heads, nope_dim=nope_dim, rope_dim=rope_dim)
    ref_topk = rt_mod._op_qli(ref_q[..., :index_dim], k_idx_cache, weights, q_scale, k_scale, block_table=block_table, query_start_loc=query_start_loc, seqused_k=seqused_k, topk=topk, compress_ratio=compress_ratio, layout_query="TND", layout_key="PA_BSND")
    ref_bt = block_table.clone()
    ref_bt.scatter_(1, topk_slots, torch.where(ref_topk < 0, torch.zeros_like(ref_topk), ref_topk))
    ref_valid = torch.clamp(seqused_k, min=0, max=topk).to(torch.int32)
    ref_attn = rt_mod._op_sfa(ref_q, k_cache, v_cache, block_table=ref_bt, seqused_k=ref_valid, max_seqlen_k=padded_capacity, sparse_mode=3)

    q_fused_calls.clear(); qli_calls.clear(); sfa_calls.clear()
    ascend_ns.fused_q_dequant_rmsnorm_rope_split = spy_q_fused
    ascend_ns.npu_quant_lightning_indexer = spy_qli
    ascend_ns.npu_sparse_flash_attention = spy_sfa
    try:
        cand_out, cand_m = mod.sparse_mla_decode_step(
            qr_i8, qr_scale, rms_weight, w_q_absorbed, cos, sin,
            k_idx_cache, k_cache, v_cache, weights, q_scale, k_scale,
            block_table.clone(), topk_slots, query_start_loc, seqused_k,
            n_heads=n_heads, nope_dim=nope_dim, rope_dim=rope_dim,
            index_dim=index_dim, topk=topk, compress_ratio=compress_ratio,
            padded_capacity=padded_capacity, eps=1e-6,
        )
        assert isinstance(cand_out, torch.Tensor) and cand_out.shape == ref_attn.shape, f"Output shape mismatch: {cand_out.shape if isinstance(cand_out, torch.Tensor) else type(cand_out).__name__}"
        assert torch.allclose(cand_out, ref_attn, atol=1e-4, rtol=1e-4), "Candidate sparse_mla_decode_step output does not match reference tensor"
        assert len(q_fused_calls) == 1 and len(qli_calls) == 1 and qli_calls[0]["q_is_view"] and len(sfa_calls) == 1 and not sfa_calls[0]["has_neg_bt"], "Direct full_candidate probe did not execute expected ops/views"

        sig_params = set(inspect.signature(mod.sparse_mla_decode_step).parameters)
        if {"enable_q_fusion", "enable_view_alias", "enable_device_guard"}.issubset(sig_params):
            kw_q, kw_v, kw_g = "enable_q_fusion", "enable_view_alias", "enable_device_guard"
        else:
            kw_q, kw_v, kw_g = "fused_q_qr_dequant_and_rope_split", "contiguous_view_alias_for_lightning_indexer", "device_resident_topk_block_mask_guard"

        q_fused_calls.clear(); qli_calls.clear(); sfa_calls.clear()
        out_mq, _ = mod.sparse_mla_decode_step(
            qr_i8, qr_scale, rms_weight, w_q_absorbed, cos, sin,
            k_idx_cache, k_cache, v_cache, weights, q_scale, k_scale,
            block_table.clone(), topk_slots, query_start_loc, seqused_k,
            n_heads=n_heads, nope_dim=nope_dim, rope_dim=rope_dim,
            index_dim=index_dim, topk=topk, compress_ratio=compress_ratio,
            padded_capacity=padded_capacity, eps=1e-6, **{kw_q: False},
        )
        assert len(q_fused_calls) == 0 and torch.allclose(out_mq, ref_attn, atol=1e-4, rtol=1e-4), "Direct minus_q_fusion probe failed"

        q_fused_calls.clear(); qli_calls.clear(); sfa_calls.clear()
        out_mv, _ = mod.sparse_mla_decode_step(
            qr_i8, qr_scale, rms_weight, w_q_absorbed, cos, sin,
            k_idx_cache, k_cache, v_cache, weights, q_scale, k_scale,
            block_table.clone(), topk_slots, query_start_loc, seqused_k,
            n_heads=n_heads, nope_dim=nope_dim, rope_dim=rope_dim,
            index_dim=index_dim, topk=topk, compress_ratio=compress_ratio,
            padded_capacity=padded_capacity, eps=1e-6, **{kw_v: False},
        )
        assert len(qli_calls) == 1 and qli_calls[0]["q_is_cloned"] and torch.allclose(out_mv, ref_attn, atol=1e-4, rtol=1e-4), "Direct minus_view_alias probe failed"

        q_fused_calls.clear(); qli_calls.clear(); sfa_calls.clear()
        raised = False
        try:
            mod.sparse_mla_decode_step(
                qr_i8, qr_scale, rms_weight, w_q_absorbed, cos, sin,
                k_idx_cache, k_cache, v_cache, weights, q_scale, k_scale,
                block_table.clone(), topk_slots, query_start_loc, seqused_k,
                n_heads=n_heads, nope_dim=nope_dim, rope_dim=rope_dim,
                index_dim=index_dim, topk=topk, compress_ratio=compress_ratio,
                padded_capacity=padded_capacity, eps=1e-6, **{kw_g: False},
            )
        except RuntimeError as exc:
            raised = "invalid GM address" in str(exc)
        assert raised and len(sfa_calls) == 1 and sfa_calls[0]["raised_invalid_gm"], "Direct minus_device_guard probe did not trigger invalid GM address"
    finally:
        ascend_ns.fused_q_dequant_rmsnorm_rope_split = orig_q_fused
        ascend_ns.npu_quant_lightning_indexer = orig_qli
        ascend_ns.npu_sparse_flash_attention = orig_sfa

    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
