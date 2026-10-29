# [Role] Independently verify CPU tensor and leave-one-out fixture contracts.
# [Invariants] Candidate metrics and source spelling never establish correctness or speed.
# [Timeouts & Locks] The caller bounds this process; no locks or device contexts.
# [Side-Effect Policy] Loads candidate Python in this process; emits a JSON verdict only.

import importlib.util
import json
import os
import sys
from pathlib import Path

evidence = {
    "validation_scope": "controlled_cpu_tensor_and_ablation_contract",
    "correctness_verified": None,
    "ablation_contract_verified": None,
    "native_kernel_bound": None,
    "native_kernel_status": "unverified",
    "measured_npu_speedup": None,
    "speedup_ratio": None,
    "tpot_ms": None,
    "latency_ms": None,
    "npu_card_hours": None,
    "performance_status": "unmeasured",
    "measurement_source": None,
    "operator_binding": "verifier_owned_python_cpu_contract",
}
root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    rt_path = Path(__file__).with_name("reference_cpu.py")
    target = root / "vllm_ascend/models/deepseek_v4/sparse_mla_decode.py"
    import torch

    rt_spec = importlib.util.spec_from_file_location("_ascend_qli_runtime_ref", rt_path)
    rt_mod = importlib.util.module_from_spec(rt_spec)
    rt_spec.loader.exec_module(rt_mod)
    # The trusted CPU fixture is installed before loading candidate imports.
    # It models interfaces only; QLI/SFA here are not native attention kernels.
    rt_mod.install_cpu_contract()

    spec = importlib.util.spec_from_file_location("sparse_mla_decode", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "sparse_mla_decode_step"), (
        "Missing sparse_mla_decode_step() in sparse_mla_decode.py"
    )
    assert hasattr(mod, "run_ablation_suite"), "Missing run_ablation_suite() in sparse_mla_decode.py"

    ascend_ns = torch.ops._C_ascend
    orig_q_fused = ascend_ns.fused_q_dequant_rmsnorm_rope_split
    orig_qli = ascend_ns.npu_quant_lightning_indexer
    orig_sfa = ascend_ns.npu_sparse_flash_attention

    q_fused_calls = []
    qli_calls = []
    sfa_calls = []
    direct_arm = None

    def spy_q_fused(*args, **kwargs):
        assert len(args) >= 6 and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.int8, (
            "Invalid qr_i8 tensor passed to fused_q_dequant_rmsnorm_rope_split"
        )
        out = orig_q_fused(*args, **kwargs)
        q_fused_calls.append(
            {"qr_shape": tuple(args[0].shape), "out_ptr": out.untyped_storage().data_ptr()}
        )
        return out

    def spy_qli(q_idx, k_idx, *args, **kwargs):
        assert isinstance(q_idx, torch.Tensor) and isinstance(k_idx, torch.Tensor), (
            "QLI expects tensor inputs"
        )
        q_is_view = (q_idx._base is not None) and (not q_idx.is_contiguous()) and (q_idx.stride(-1) == 1)
        q_is_cloned = (q_idx._base is None) and q_idx.is_contiguous()
        if direct_arm is not None:
            assert torch.allclose(q_idx, ref_q[..., :index_dim], atol=1e-4, rtol=1e-4), (
                "QLI query values differ from the independent Q reference"
            )
            k_aliases = k_idx.untyped_storage().data_ptr() == active_k_idx.untyped_storage().data_ptr()
            if direct_arm == "full_candidate":
                assert k_aliases, "Full-candidate QLI K cache must preserve storage alias"
            if direct_arm in ("full_candidate", "minus_device_guard"):
                assert q_idx.untyped_storage().data_ptr() == q_fused_calls[-1]["out_ptr"], (
                    "QLI Q view must alias the fused output storage"
                )
        out = orig_qli(q_idx, k_idx, *args, **kwargs)
        qli_calls.append(
            {
                "q_is_view": q_is_view,
                "q_is_cloned": q_is_cloned,
                "q_ptr": q_idx.untyped_storage().data_ptr(),
                "k_ptr": k_idx.untyped_storage().data_ptr(),
                "has_neg_topk": bool((out < 0).any().item()),
            }
        )
        return out

    def spy_sfa(q, k_cache, v_cache, *args, **kwargs):
        bt = kwargs.get("block_table")
        assert isinstance(bt, torch.Tensor), "block_table must be a Tensor"
        if direct_arm is not None:
            unsafe = direct_arm == "auto" and bool((bt < 0).any())
            expected_bt = raw_bt if unsafe else ref_bt
            expected_seq = original_seq if unsafe else ref_valid
            expected_capacity = int(original_seq.max()) if unsafe else padded_capacity
            assert torch.equal(bt, expected_bt), (
                "SFA block_table does not preserve the exact masked/unmasked QLI indices"
            )
            assert torch.equal(kwargs["seqused_k"], expected_seq), (
                "SFA sequence lengths differ from the arm contract"
            )
            assert kwargs["max_seqlen_k"] == expected_capacity, (
                "SFA capacity differs from the arm contract"
            )
            assert torch.equal(active_seq, original_seq), "Candidate mutated caller sequence lengths"
        has_neg_bt = bool((bt < 0).any().item())
        try:
            out = orig_sfa(q, k_cache, v_cache, *args, **kwargs)
            sfa_calls.append(
                {"has_neg_bt": has_neg_bt, "raised_invalid_gm": False, "out_shape": tuple(out.shape)}
            )
            return out
        except RuntimeError as exc:
            sfa_calls.append(
                {"has_neg_bt": has_neg_bt, "raised_invalid_gm": "invalid GM address" in str(exc)}
            )
            raise

    # Independent verifier-constructed tensor probe & numerical reference check
    gen = torch.Generator().manual_seed(91042)
    t_len, n_heads, q_lora_rank = 4, 4, 24
    nope_dim, rope_dim, index_dim = 512, 64, 128
    topk, compress_ratio, padded_capacity = 8, 4, 32
    qr_i8 = torch.randint(-32, 32, (t_len, q_lora_rank), generator=gen, dtype=torch.int8)
    qr_scale = torch.linspace(0.02, 0.08, t_len, dtype=torch.float32)
    rms_weight = torch.randn((q_lora_rank,), generator=gen, dtype=torch.float32)
    w_q_absorbed = (
        torch.randn((n_heads * (nope_dim + rope_dim), q_lora_rank), generator=gen, dtype=torch.float32)
        * 0.02
    )
    angles = torch.randn((t_len, rope_dim), generator=gen, dtype=torch.float32)
    cos, sin = angles.cos(), angles.sin()
    k_idx_cache = torch.randn(
        (padded_capacity, 16, n_heads, index_dim), generator=gen, dtype=torch.float32
    )
    k_cache = torch.randn(
        (padded_capacity, 16, n_heads, nope_dim + rope_dim), generator=gen, dtype=torch.float32
    )
    v_cache = torch.randn((padded_capacity, 16, n_heads, nope_dim), generator=gen, dtype=torch.float32)
    weights = torch.ones((t_len, n_heads), dtype=torch.float32)
    q_scale = torch.ones((t_len, n_heads), dtype=torch.float32)
    k_scale = torch.ones((padded_capacity,), dtype=torch.float32)
    block_table = torch.zeros((t_len, padded_capacity), dtype=torch.int32)
    topk_slots = torch.arange(topk, dtype=torch.int64).unsqueeze(0).expand(t_len, topk).contiguous()
    query_start_loc = torch.arange(0, t_len + 1, dtype=torch.int32)
    seqused_k = torch.tensor([64, 12, 32, 8], dtype=torch.int32)

    ref_q = rt_mod._op_q_fused(
        qr_i8,
        qr_scale,
        rms_weight,
        w_q_absorbed,
        cos,
        sin,
        eps=1e-6,
        n_heads=n_heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    ref_topk = rt_mod._op_qli(
        ref_q[..., :index_dim],
        k_idx_cache,
        weights,
        q_scale,
        k_scale,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seqused_k=seqused_k,
        topk=topk,
        compress_ratio=compress_ratio,
        layout_query="TND",
        layout_key="PA_BSND",
    )
    original_seq = seqused_k.clone()
    raw_bt = block_table.clone()
    raw_bt.scatter_(1, topk_slots, ref_topk)
    ref_bt = block_table.clone()
    ref_bt.scatter_(1, topk_slots, torch.where(ref_topk < 0, torch.zeros_like(ref_topk), ref_topk))
    # QLI returns compressed block IDs; SFA may consume only its valid prefix.
    ref_valid = (ref_topk >= 0).sum(dim=-1).to(torch.int32)
    ref_attn = rt_mod._op_sfa(
        ref_q,
        k_cache,
        v_cache,
        block_table=ref_bt,
        seqused_k=ref_valid,
        max_seqlen_k=padded_capacity,
        sparse_mode=3,
    )

    sys.path.insert(0, str(Path(__file__).parent))
    from runtime_controls import CallCapture, UnsupportedCPUExecution, invoke_suite, output_tensor

    inputs = (
        qr_i8,
        qr_scale,
        rms_weight,
        w_q_absorbed,
        cos,
        sin,
        k_idx_cache,
        k_cache,
        v_cache,
        weights,
        q_scale,
        k_scale,
        block_table,
        topk_slots,
        query_start_loc,
        seqused_k,
    )
    input_names = (
        "qr_i8",
        "qr_scale",
        "rms_weight",
        "w_q_absorbed",
        "cos",
        "sin",
        "k_idx_cache",
        "k_cache",
        "v_cache",
        "weights",
        "q_scale",
        "k_scale",
        "block_table",
        "topk_slots",
        "query_start_loc",
        "seqused_k",
    )
    core = dict(
        n_heads=n_heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        index_dim=index_dim,
        topk=topk,
        compress_ratio=compress_ratio,
        padded_capacity=padded_capacity,
        eps=1e-6,
    )

    def execute(call, mode="auto"):
        global direct_arm, active_k_idx, active_seq
        active_inputs = tuple(t.clone() for t in inputs)
        active_k_idx, active_seq = active_inputs[6], active_inputs[15]
        direct_arm = mode
        q_fused_calls.clear()
        qli_calls.clear()
        sfa_calls.clear()
        out = None
        error = None
        try:
            out = output_tensor(call(active_inputs))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = {
            "q_fused_calls": len(q_fused_calls),
            "qli_calls": len(qli_calls),
            "sfa_calls": len(sfa_calls),
            "correct": False,
            "error": error,
        }
        if out is not None and out.shape == ref_attn.shape:
            result["correct"] = bool(torch.allclose(out, ref_attn, atol=1e-4, rtol=1e-4))
        if len(qli_calls) == 1:
            observed = qli_calls[0]
            result.update(
                q_is_view=observed["q_is_view"],
                q_is_cloned=observed["q_is_cloned"],
                k_alias=observed["k_ptr"] == active_k_idx.untyped_storage().data_ptr(),
                q_alias=bool(q_fused_calls) and observed["q_ptr"] == q_fused_calls[-1]["out_ptr"],
            )
        result["unsafe_guard_exercised"] = (
            len(sfa_calls) == 1 and sfa_calls[0]["has_neg_bt"] and sfa_calls[0]["raised_invalid_gm"]
        )
        result["safe_handoff"] = (
            len(sfa_calls) == 1
            and not sfa_calls[0]["has_neg_bt"]
            and not sfa_calls[0]["raised_invalid_gm"]
        )
        return result

    def is_full(result):
        return (
            result["correct"]
            and result["q_fused_calls"] == result["qli_calls"] == 1
            and result.get("q_is_view")
            and result.get("q_alias")
            and result.get("k_alias")
            and result["safe_handoff"]
        )

    ascend_ns.fused_q_dequant_rmsnorm_rope_split = spy_q_fused
    ascend_ns.npu_quant_lightning_indexer = spy_qli
    ascend_ns.npu_sparse_flash_attention = spy_sfa
    suite_error = None
    try:
        with CallCapture(mod, "sparse", core) as capture:
            full = execute(lambda values: mod.sparse_mla_decode_step(*values, **core), "full_candidate")
            assert is_full(full), f"Independent full-candidate tensor/handoff check failed: {full}"
            evidence["correctness_verified"] = True
            capture.phase = "suite"
            direct_arm = None
            try:
                suite = invoke_suite(mod, "sparse", inputs, core, input_names)
                evidence["candidate_suite_type"] = type(suite).__name__
            except (ImportError, NameError):
                raise
            except Exception as exc:
                suite_error = exc
        selected = None
        failures = []
        for entrypoint, controls in capture.entries.items():
            values = list(controls.values())
            if len(values) < 4 or not any(c.is_default for c in values):
                continue
            observed = []
            for control in values:
                result = execute(lambda tensors: control.replay(tensors, core))
                result.update(control=control.description, default_path=control.is_default)
                observed.append(result)
            defaults = [r for r in observed if r["default_path"] and is_full(r)]
            q_arms = [
                r
                for r in observed
                if r["correct"]
                and r["q_fused_calls"] == 0
                and r["qli_calls"] == 1
                and r.get("q_is_view")
                and r.get("k_alias")
                and r["safe_handoff"]
            ]
            view_arms = [
                r
                for r in observed
                if r["correct"]
                and r["q_fused_calls"] == r["qli_calls"] == 1
                and r.get("q_is_cloned")
                and not r.get("k_alias")
                and r["safe_handoff"]
            ]
            guard_arms = [
                r
                for r in observed
                if r["unsafe_guard_exercised"]
                and r["q_fused_calls"] == r["qli_calls"] == 1
                and r.get("q_is_view")
                and r.get("k_alias")
            ]
            if defaults and q_arms and view_arms and guard_arms:
                selected = {
                    "entrypoint": entrypoint,
                    "full_candidate": defaults[0],
                    "minus_q_fusion": q_arms[0],
                    "minus_view_alias": view_arms[0],
                    "minus_device_guard": guard_arms[0],
                }
                break
            failures.append({"entrypoint": entrypoint, "observed": observed})
        evidence["observed_cpu_ablation"] = selected or {"entrypoints": failures}
        if selected is None and suite_error is not None:
            if isinstance(suite_error, (ImportError, NotImplementedError)):
                raise UnsupportedCPUExecution(f"CPU ablation execution unavailable: {suite_error}")
            raise AssertionError(
                f"Executed CPU ablation failed: {type(suite_error).__name__}: {suite_error}"
            ) from suite_error
        assert selected is not None, (
            "Independent replay did not establish three distinct operator/view/guard ablations"
        )
    finally:
        ascend_ns.fused_q_dequant_rmsnorm_rope_split = orig_q_fused
        ascend_ns.npu_quant_lightning_indexer = orig_qli
        ascend_ns.npu_sparse_flash_attention = orig_sfa
    evidence.update(
        correctness_verified=True, ablation_contract_verified=True, verification_status="completed"
    )
    print(json.dumps({"reward": 1.0, **evidence}))
except Exception as exc:
    evidence["verification_status"] = (
        "unsupported"
        if isinstance(exc, (ModuleNotFoundError, ImportError, NotImplementedError))
        else "failed"
    )
    print(
        json.dumps(
            {"reward": 0.0, "failure_reason": str(exc), "error_type": type(exc).__name__, **evidence}
        )
    )
    raise SystemExit(1)
