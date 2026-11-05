#!/usr/bin/env python3
"""Verifier for vllm-ascend-kda-varlen-chunk-eval-zh."""

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


def reference_mixed_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    continuation_slot_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T_total, H, K = q.shape
    V = v.shape[-1]
    num_seqs = int(cu_seqlens.numel()) - 1
    o_ref = torch.zeros((1, T_total, H, V), dtype=torch.float32)
    final_state_ref = torch.zeros((num_seqs, H, V, K), dtype=torch.float32)

    for seq_idx in range(num_seqs):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        if continuation_slot_map is not None and int(continuation_slot_map[seq_idx].item()) >= 0:
            src_slot = int(continuation_slot_map[seq_idx].item())
            if src_slot < seq_idx:
                h = final_state_ref[src_slot].clone()
            else:
                h = initial_state[src_slot].clone().to(torch.float32)
        else:
            h = initial_state[seq_idx].clone().to(torch.float32)

        for t in range(bos, eos):
            q_t = q[0, t].to(torch.float32)
            k_t = k[0, t].to(torch.float32)
            v_t = v[0, t].to(torch.float32)
            g_t = g[0, t].to(torch.float32)
            b_t = beta[0, t].to(torch.float32).unsqueeze(-1)
            h = h * torch.exp(g_t).unsqueeze(1)
            v_pred = torch.einsum("hvk,hk->hv", h, k_t)
            v_delta = (v_t - v_pred) * b_t
            h = h + torch.einsum("hv,hk->hvk", v_delta, k_t)
            o_ref[0, t] = scale * torch.einsum("hvk,hk->hv", h, q_t)
        final_state_ref[seq_idx] = h

    return o_ref, final_state_ref


def verify() -> dict:
    check_diff_scope()
    kda_mod = load_kda_module()
    assert hasattr(kda_mod, "chunk_kda_varlen_mixed_continuation"), (
        "Missing chunk_kda_varlen_mixed_continuation in vllm_ascend/ops/triton/kda/kda.py"
    )

    gen = torch.Generator().manual_seed(20261104)
    # Mixed batch: Seq 0 (prefill len=21), Seq 1 (decode len=1), Seq 2 (prefill len=18), Seq 3 (decode len=1 continuing Seq 0)
    cu_seqlens = torch.tensor([0, 21, 22, 40, 41], dtype=torch.int32)
    T_total = int(cu_seqlens[-1].item())
    H, K, V = 4, 16, 16
    chunk_size = 16
    scale = K ** -0.5

    q = torch.randn((1, T_total, H, K), generator=gen, dtype=torch.float32) * 0.3
    k = torch.randn((1, T_total, H, K), generator=gen, dtype=torch.float32) * 0.3
    v = torch.randn((1, T_total, H, V), generator=gen, dtype=torch.float32) * 0.5
    g = -torch.rand((1, T_total, H, K), generator=gen, dtype=torch.float32) * 0.4
    beta = torch.sigmoid(torch.randn((1, T_total, H), generator=gen, dtype=torch.float32))
    init_state = torch.randn((4, H, V, K), generator=gen, dtype=torch.float32) * 0.2
    slot_map = None

    o_ref, final_state_ref = reference_mixed_forward(
        q, k, v, g, beta, cu_seqlens, init_state, scale, continuation_slot_map=slot_map
    )
    o_out, final_state_out = kda_mod.chunk_kda_varlen_mixed_continuation(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        cu_seqlens=cu_seqlens,
        initial_state=init_state.clone(),
        scale=scale,
        chunk_size=chunk_size,
        continuation_slot_map=slot_map,
    )

    assert torch.allclose(o_out.to(torch.float32), o_ref, atol=1e-5, rtol=1e-5), (
        f"Mixed continuation output o mismatch: max_err={(o_out.to(torch.float32) - o_ref).abs().max().item()}"
    )
    assert torch.allclose(final_state_out.to(torch.float32), final_state_ref, atol=1e-5, rtol=1e-5), (
        f"Mixed continuation final_state mismatch: max_err={(final_state_out.to(torch.float32) - final_state_ref).abs().max().item()}"
    )

    # Also ensure seed prefill recurrence remains intact
    v_new_out, _, prefill_state_out = kda_mod.chunk_kda_varlen_prefill_recurrence(
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        cu_seqlens=cu_seqlens,
        initial_state=init_state.clone(),
        chunk_size=chunk_size,
    )
    assert torch.allclose(prefill_state_out[:3], final_state_ref[:3], atol=1e-5, rtol=1e-5), (
        "Regression in chunk_kda_varlen_prefill_recurrence states!"
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
