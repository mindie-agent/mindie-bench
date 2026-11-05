#!/usr/bin/env python3
"""Verifier for vllm-ascend-mla-prolog-rope-kvcache-fusion-rsi-zh."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("USER", "mindie")
os.environ.setdefault("LOGNAME", "mindie")

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
ALLOWED_FILES = {"vllm_ascend/ops/mla.py"}


def check_diff_scope() -> None:
    if not (ROOT / ".git").exists():
        return
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    unexpected = changed - ALLOWED_FILES
    if unexpected:
        raise AssertionError(f"Modified files outside allowed scope: {sorted(unexpected)}")


def load_mla_module():
    for mod_name in [
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.forward_context",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
        "vllm.model_executor.layers.mla",
        "vllm.model_executor.layers.quantization",
        "vllm.utils",
        "vllm.utils.torch_utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm_ascend",
        "vllm_ascend.ascend_forward_context",
        "vllm_ascend.utils",
    ]:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    sys.modules["vllm.config"].CacheConfig = object
    sys.modules["vllm.config"].get_current_vllm_config = lambda: None
    sys.modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
    sys.modules["vllm.forward_context"].ForwardContext = object
    sys.modules["vllm.forward_context"].get_forward_context = lambda: None
    sys.modules["vllm.model_executor.layers.attention"].MLAAttention = object
    sys.modules["vllm.model_executor.layers.mla"].MLAModules = object
    sys.modules["vllm.model_executor.layers.mla"].MultiHeadLatentAttentionWrapper = torch.nn.Module
    sys.modules["vllm.model_executor.layers.quantization"].QuantizationConfig = object
    sys.modules["vllm.utils.torch_utils"].direct_register_custom_op = lambda **kw: None
    sys.modules["vllm.v1.attention.backend"].AttentionMetadata = object
    sys.modules["vllm_ascend.ascend_forward_context"]._EXTRA_CTX = types.SimpleNamespace(flash_comm_v1_enabled=False)
    sys.modules["vllm_ascend.utils"].is_vl_model = lambda cfg: False
    sys.modules["vllm_ascend.utils"].parse_layer_idx = lambda p: 0

    spec = importlib.util.spec_from_file_location("vllm_ascend.ops.mla", ROOT / "vllm_ascend/ops/mla.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vllm_ascend.ops.mla"] = mod
    spec.loader.exec_module(mod)
    return mod


class TensorTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_ops = 0
        self.clone_repeat_ops = 0
        self.select_slice_ops = 0
        self.linear_ops = 0

    def __torch_dispatch__(self, fn, types_arg, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.tensor_ops += 1
        if fn in (torch.ops.aten.clone.default, torch.ops.aten.repeat.default):
            self.clone_repeat_ops += 1
        if fn in (torch.ops.aten.select.int, torch.ops.aten.slice.Tensor):
            self.select_slice_ops += 1
        if fn in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default, torch.ops.aten.linear.default):
            self.linear_ops += 1
        return fn(*args, **kwargs)

    def summary(self) -> dict[str, int]:
        return {
            "tensor_ops": self.tensor_ops,
            "clone_repeat_ops": self.clone_repeat_ops,
            "select_slice_ops": self.select_slice_ops,
            "linear_ops": self.linear_ops,
        }


def reference_mla_prolog(
    hidden_states: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_rms_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_rms_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_dtype = hidden_states.dtype
    T = hidden_states.shape[0]
    head_dim = qk_nope_head_dim + qk_rope_head_dim

    q_latent = F.linear(hidden_states, w_q_a).float()
    q_rstd = torch.rsqrt(q_latent.pow(2).mean(dim=-1, keepdim=True) + eps)
    q_normed = (q_latent * q_rstd * q_a_rms_weight.float()).to(orig_dtype)

    kv_all = F.linear(hidden_states, w_kv_a).float()
    kv_latent = kv_all[:, :kv_lora_rank]
    k_pe_raw = kv_all[:, kv_lora_rank:].to(orig_dtype)
    kv_rstd = torch.rsqrt(kv_latent.pow(2).mean(dim=-1, keepdim=True) + eps)
    kv_nope_normed = (kv_latent * kv_rstd * kv_a_rms_weight.float()).to(orig_dtype)

    q_all = F.linear(q_normed, w_q_b).view(T, num_heads, head_dim)
    q_nope = q_all[:, :, :qk_nope_head_dim]
    q_pe = q_all[:, :, qk_nope_head_dim:].float()

    half = qk_rope_head_dim // 2
    q_rot_half = torch.cat((-q_pe[..., half:], q_pe[..., :half]), dim=-1)
    q_pe_rot = (q_pe * cos.float().unsqueeze(1) + q_rot_half * sin.float().unsqueeze(1)).to(orig_dtype)

    k_pe_f = k_pe_raw.float()
    k_rot_half = torch.cat((-k_pe_f[..., half:], k_pe_f[..., :half]), dim=-1)
    k_pe_rot = (k_pe_f * cos.float() + k_rot_half * sin.float()).to(orig_dtype)

    q_out = torch.cat((q_nope, q_pe_rot), dim=-1)

    cache_nope_ref = kv_cache_nope.clone()
    cache_rope_ref = kv_cache_rope.clone()
    valid = slot_mapping >= 0
    cache_nope_ref.view(-1, kv_lora_rank).index_copy_(0, slot_mapping[valid].to(torch.int64), kv_nope_normed[valid])
    cache_rope_ref.view(-1, qk_rope_head_dim).index_copy_(0, slot_mapping[valid].to(torch.int64), k_pe_rot[valid])
    return q_out, kv_nope_normed, k_pe_rot, cache_nope_ref, cache_rope_ref


def main() -> None:
    evidence = {
        "validation_scope": "controlled_cpu_tensor_and_ablation_contract",
        "correctness_verified": False,
        "ablation_contract_verified": False,
        "native_kernel_bound": None,
        "native_kernel_status": "unverified",
        "measured_npu_speedup": None,
        "speedup_ratio": None,
        "tpot_ms": None,
        "latency_ms": None,
        "npu_card_hours": None,
        "performance_status": "unmeasured",
        "measurement_source": None,
        "operator_binding": "cpu_tensor_execution",
    }
    try:
        check_diff_scope()
        mod = load_mla_module()
        assert hasattr(mod, "mla_prolog_rope_kvcache_step"), "Missing mla_prolog_rope_kvcache_step in mla.py"

        gen = torch.Generator().manual_seed(20261105)
        T, hidden_size = 8, 64
        q_lora_rank, kv_lora_rank = 32, 24
        num_heads, qk_nope_head_dim, qk_rope_head_dim = 4, 16, 8
        num_blocks, block_size = 4, 4

        hidden_states = (torch.randn((T, hidden_size), generator=gen) * 0.5).to(torch.bfloat16)
        w_q_a = (torch.randn((q_lora_rank, hidden_size), generator=gen) * 0.2).to(torch.bfloat16)
        q_a_rms_weight = (torch.rand((q_lora_rank,), generator=gen) * 0.5 + 0.75).to(torch.bfloat16)
        w_q_b = (
            torch.randn((num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank), generator=gen) * 0.2
        ).to(torch.bfloat16)
        w_kv_a = (torch.randn((kv_lora_rank + qk_rope_head_dim, hidden_size), generator=gen) * 0.2).to(torch.bfloat16)
        kv_a_rms_weight = (torch.rand((kv_lora_rank,), generator=gen) * 0.5 + 0.75).to(torch.bfloat16)
        angles = torch.randn((T, qk_rope_head_dim), generator=gen)
        cos = angles.cos().to(torch.bfloat16)
        sin = angles.sin().to(torch.bfloat16)
        kv_cache_nope = torch.zeros((num_blocks, block_size, kv_lora_rank), dtype=torch.bfloat16)
        kv_cache_rope = torch.zeros((num_blocks, block_size, qk_rope_head_dim), dtype=torch.bfloat16)
        slot_mapping = torch.tensor([2, 5, -1, 9, 1, 14, 7, 11], dtype=torch.int64)

        core = {
            "num_heads": num_heads,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_lora_rank": kv_lora_rank,
            "eps": 1e-6,
        }
        inputs = (
            hidden_states,
            w_q_a,
            q_a_rms_weight,
            w_q_b,
            w_kv_a,
            kv_a_rms_weight,
            cos,
            sin,
            kv_cache_nope,
            kv_cache_rope,
            slot_mapping,
        )

        q_ref, kv_nope_ref, k_pe_ref, cache_nope_ref, cache_rope_ref = reference_mla_prolog(
            *tuple(t.clone() for t in inputs), **core
        )

        # 1. Verify default optimal path
        test_inputs = tuple(t.clone() for t in inputs)
        with TensorTrace() as default_trace:
            q_out, kv_nope_out, k_pe_out = mod.mla_prolog_rope_kvcache_step(*test_inputs, **core)

        max_q_err = float((q_out.float() - q_ref.float()).abs().max().item())
        max_kv_err = float((test_inputs[8].float() - cache_nope_ref.float()).abs().max().item())
        max_rope_err = float((test_inputs[9].float() - cache_rope_ref.float()).abs().max().item())
        assert max_q_err <= 2e-2 and max_kv_err <= 2e-2 and max_rope_err <= 2e-2, (
            f"Default path numerical error too large: q={max_q_err}, kv={max_kv_err}, rope={max_rope_err}"
        )
        assert default_trace.clone_repeat_ops == 0, (
            f"Default path still performs redundant clone/repeat allocations: {default_trace.summary()}"
        )
        assert default_trace.linear_ops == 2, (
            f"Default path still performs unfused linear projections (expected 2): {default_trace.summary()}"
        )
        assert default_trace.select_slice_ops <= 12, (
            f"Default path still performs uncoalesced slice/select operations: {default_trace.summary()}"
        )
        assert k_pe_out.ndim == 2, f"Expected 2D k_pe_out, got shape {tuple(k_pe_out.shape)}"
        evidence["correctness_verified"] = True
        evidence["max_abs_error"] = max(max_q_err, max_kv_err, max_rope_err)

        # 2. Verify leave-one-out ablation report and suite execution
        assert hasattr(mod, "run_ablation_report"), "Missing run_ablation_report in mla.py"
        report = mod.run_ablation_report()
        retained = set(report.get("retained_factors", []))
        rejected = set(report.get("rejected_factors", []))
        assert retained == {
            "fuse_qkv_norm_proj",
            "vectorized_rope_no_broadcast",
            "coalesce_kv_cache_scatter",
        }, f"Unexpected retained_factors: {retained}"
        assert "enable_lowbit_qk_truncation" in rejected, f"Decoy factor not rejected: {rejected}"

        # Measure per-arm dispatch traces independently when optional ablation kwargs are supported
        arm_traces: dict[str, dict[str, object]] = {
            "full_candidate": {**default_trace.summary(), "max_abs_err": evidence["max_abs_error"]},
        }
        supports_arm_kwargs = True
        for arm_name, flags in [
            ("full_candidate", dict(fuse_qkv_norm_proj=True, vectorized_rope_no_broadcast=True, coalesce_kv_cache_scatter=True, enable_lowbit_qk_truncation=False)),
            ("minus_qkv_norm_proj_fusion", dict(fuse_qkv_norm_proj=False, vectorized_rope_no_broadcast=True, coalesce_kv_cache_scatter=True, enable_lowbit_qk_truncation=False)),
            ("minus_vectorized_rope", dict(fuse_qkv_norm_proj=True, vectorized_rope_no_broadcast=False, coalesce_kv_cache_scatter=True, enable_lowbit_qk_truncation=False)),
            ("minus_coalesced_kv_scatter", dict(fuse_qkv_norm_proj=True, vectorized_rope_no_broadcast=True, coalesce_kv_cache_scatter=False, enable_lowbit_qk_truncation=False)),
            ("plus_decoy_lowbit_truncation", dict(fuse_qkv_norm_proj=True, vectorized_rope_no_broadcast=True, coalesce_kv_cache_scatter=True, enable_lowbit_qk_truncation=True)),
        ]:
            arm_in = tuple(t.clone() for t in inputs)
            try:
                with TensorTrace() as tr:
                    q_a, _, _ = mod.mla_prolog_rope_kvcache_step(*arm_in, **core, **flags)
            except TypeError:
                supports_arm_kwargs = False
                break
            err_a = float((q_a.float() - q_ref.float()).abs().max().item())
            arm_traces[arm_name] = {**tr.summary(), "max_abs_err": err_a}

        if supports_arm_kwargs:
            if "fuse_qkv_norm_proj" in retained:
                assert arm_traces["minus_qkv_norm_proj_fusion"]["tensor_ops"] > arm_traces["full_candidate"]["tensor_ops"]
            assert (
                arm_traces["minus_vectorized_rope"]["clone_repeat_ops"] > arm_traces["full_candidate"]["clone_repeat_ops"]
                or arm_traces["minus_vectorized_rope"]["tensor_ops"] > arm_traces["full_candidate"]["tensor_ops"]
            )
            assert arm_traces["minus_coalesced_kv_scatter"]["select_slice_ops"] > arm_traces["full_candidate"]["select_slice_ops"]
            assert arm_traces["plus_decoy_lowbit_truncation"]["max_abs_err"] > 2e-2

        evidence["observed_cpu_ablation"] = arm_traces
        evidence.update(
            correctness_verified=True,
            ablation_contract_verified=True,
            verification_status="completed",
        )
        print(json.dumps({"reward": 1.0, **evidence}))
    except Exception as exc:
        evidence["verification_status"] = "failed"
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "failure_reason": str(exc),
                    "error_type": type(exc).__name__,
                    **evidence,
                }
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
