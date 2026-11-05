#!/usr/bin/env python3
"""Verifier for vllm-ascend-kda-varlen-chunk-seed-zh."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import torch

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
ALLOWED_FILES = {"vllm_ascend/ops/triton/kda/kda.py"}


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


def load_kda_module():
    class DummyKernel:
        def __init__(self, fn=None):
            self.fn = fn

        def __getitem__(self, grid):
            return self

        def __call__(self, *args, **kwargs):
            raise RuntimeError("Triton kernel launch not supported in CPU verifier")

    def dummy_decorator(*dargs, **dkwargs):
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return DummyKernel(dargs[0])
        return lambda fn: DummyKernel(fn)

    triton_mod = types.ModuleType("triton")
    triton_mod.jit = dummy_decorator
    triton_mod.heuristics = lambda *a, **kw: (lambda fn: fn)
    triton_mod.autotune = lambda *a, **kw: (lambda fn: fn)
    triton_mod.Config = lambda *a, **kw: {}
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.next_power_of_2 = lambda n: 1 if n <= 1 else (1 << (int(n) - 1).bit_length())

    class DynamicTL(types.ModuleType):
        def __getattr__(self, name: str):
            return int

    tl_mod = DynamicTL("triton.language")
    tl_mod.constexpr = int
    tl_mod.float32 = torch.float32
    tl_mod.int32 = torch.int32
    tl_mod.int64 = torch.int64

    vllm_mod = types.ModuleType("vllm")
    triton_utils_mod = types.ModuleType("vllm.triton_utils")
    triton_utils_mod.HAS_TRITON = True
    triton_utils_mod.tl = tl_mod
    triton_utils_mod.triton = triton_mod

    utils_mod = types.ModuleType("vllm.utils")
    math_utils_mod = types.ModuleType("vllm.utils.math_utils")
    math_utils_mod.cdiv = triton_mod.cdiv
    math_utils_mod.next_power_of_2 = triton_mod.next_power_of_2

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = tl_mod
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.triton_utils"] = triton_utils_mod
    sys.modules["vllm.utils"] = utils_mod
    sys.modules["vllm.utils.math_utils"] = math_utils_mod

    for pkg_name, rel_dir in [
        ("vllm_ascend", "vllm_ascend"),
        ("vllm_ascend.ops", "vllm_ascend/ops"),
        ("vllm_ascend.ops.triton", "vllm_ascend/ops/triton"),
        ("vllm_ascend.ops.triton.kda", "vllm_ascend/ops/triton/kda"),
    ]:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(ROOT / rel_dir)]
        sys.modules[pkg_name] = pkg

    kda_dir = ROOT / "vllm_ascend/ops/triton/kda"
    for stub_name, stub_attr in [
        ("chunk_delta_h", "chunk_gated_delta_rule_fwd_h_kda"),
        ("cumsum", "chunk_local_cumsum"),
        ("fused_recurrent_kda", "fused_recurrent_gated_delta_rule_fwd_kernel"),
        ("l2norm", "l2norm_fwd"),
        ("solve_tril", "solve_tril_kda"),
    ]:
        full_stub = f"vllm_ascend.ops.triton.kda.{stub_name}"
        smod = types.ModuleType(full_stub)
        setattr(smod, stub_attr, DummyKernel())
        sys.modules[full_stub] = smod

    for submod_name in ["utils", "kda"]:
        full_name = f"vllm_ascend.ops.triton.kda.{submod_name}"
        spec = importlib.util.spec_from_file_location(full_name, kda_dir / f"{submod_name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)

    return sys.modules["vllm_ascend.ops.triton.kda.kda"]


def reference_stepwise_recurrence(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T_total, H, K = k.shape
    V = v.shape[-1]
    num_seqs = int(cu_seqlens.numel()) - 1
    v_new_ref = torch.zeros_like(v, dtype=torch.float32)
    final_state_ref = torch.zeros((num_seqs, H, V, K), dtype=torch.float32)

    for seq_idx in range(num_seqs):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        h = initial_state[seq_idx].clone().to(torch.float32)
        for t in range(bos, eos):
            k_t = k[0, t].to(torch.float32)
            v_t = v[0, t].to(torch.float32)
            g_t = g[0, t].to(torch.float32)
            b_t = beta[0, t].to(torch.float32).unsqueeze(-1)
            h = h * torch.exp(g_t).unsqueeze(1)
            v_pred = torch.einsum("hvk,hk->hv", h, k_t)
            v_delta = (v_t - v_pred) * b_t
            v_new_ref[0, t] = v_delta
            h = h + torch.einsum("hv,hk->hvk", v_delta, k_t)
        final_state_ref[seq_idx] = h

    return v_new_ref, final_state_ref


def verify() -> dict:
    check_diff_scope()
    kda_mod = load_kda_module()
    assert hasattr(kda_mod, "chunk_kda_varlen_prefill_recurrence"), (
        "Missing chunk_kda_varlen_prefill_recurrence in vllm_ascend/ops/triton/kda/kda.py"
    )

    gen = torch.Generator().manual_seed(20261103)
    # Packed sequences with lengths [19, 37, 11] and chunk_size=16 (non-multiples of BT)
    cu_seqlens = torch.tensor([0, 19, 56, 67], dtype=torch.int32)
    T_total = int(cu_seqlens[-1].item())
    H, K, V = 4, 16, 16
    chunk_size = 16

    k = torch.randn((1, T_total, H, K), generator=gen, dtype=torch.float32) * 0.3
    v = torch.randn((1, T_total, H, V), generator=gen, dtype=torch.float32) * 0.5
    g = -torch.rand((1, T_total, H, K), generator=gen, dtype=torch.float32) * 0.4
    beta = torch.sigmoid(torch.randn((1, T_total, H), generator=gen, dtype=torch.float32))
    init_state = torch.randn((3, H, V, K), generator=gen, dtype=torch.float32) * 0.2

    v_new_ref, final_state_ref = reference_stepwise_recurrence(k, v, g, beta, cu_seqlens, init_state)
    v_new_out, chunk_states_out, final_state_out = kda_mod.chunk_kda_varlen_prefill_recurrence(
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        cu_seqlens=cu_seqlens,
        initial_state=init_state.clone(),
        chunk_size=chunk_size,
    )

    # Expected total chunks: ceil(19/16) + ceil(37/16) + ceil(11/16) = 2 + 3 + 1 = 6
    assert chunk_states_out.shape == (6, H, V, K), f"Unexpected chunk_states shape: {chunk_states_out.shape}"
    assert torch.allclose(v_new_out.to(torch.float32), v_new_ref, atol=1e-5, rtol=1e-5), (
        f"v_new mismatch against stepwise varlen reference: max_err={(v_new_out.to(torch.float32) - v_new_ref).abs().max().item()}"
    )
    assert torch.allclose(final_state_out.to(torch.float32), final_state_ref, atol=1e-5, rtol=1e-5), (
        f"final_state mismatch against stepwise varlen reference: max_err={(final_state_out.to(torch.float32) - final_state_ref).abs().max().item()}"
    )

    # Cross-sequence isolation test: perturbing sequence 1 and 2 must have zero effect on sequence 0's final_state or v_new
    k_perturbed = k.clone()
    v_perturbed = v.clone()
    g_perturbed = g.clone()
    k_perturbed[:, 19:] += 50.0
    v_perturbed[:, 19:] -= 50.0
    g_perturbed[:, 19:] = -5.0

    v_new_p, _, final_state_p = kda_mod.chunk_kda_varlen_prefill_recurrence(
        k=k_perturbed,
        v=v_perturbed,
        g=g_perturbed,
        beta=beta.clone(),
        cu_seqlens=cu_seqlens,
        initial_state=init_state.clone(),
        chunk_size=chunk_size,
    )
    assert torch.allclose(final_state_out[0], final_state_p[0], atol=1e-6, rtol=1e-6), (
        "Tail chunk of sequence 0 leaked across eos into sequence 1!"
    )
    assert torch.allclose(v_new_out[:, :19], v_new_p[:, :19], atol=1e-6, rtol=1e-6), (
        "v_new of sequence 0 was polluted by sequence 1 tokens!"
    )

    return {
        "reward": 1.0,
        "verification_status": "completed",
    }


def main() -> None:
    try:
        res = verify()
        print(json.dumps(res))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "verification_status": "failed",
                    "failure_reason": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
