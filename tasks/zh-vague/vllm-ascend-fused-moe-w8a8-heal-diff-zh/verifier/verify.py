#!/usr/bin/env python3
"""Verifier for vllm-ascend-fused-moe-w8a8-heal-diff-zh."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

EXPECTED_REPRO_SHA256 = "7b3e793b689a1885cda9b14962e09d45625c1e80948741a4de5ac59e5172615d"


def _fail(reason: str, error_type: str = "VerificationAssertionError") -> int:
    payload = {
        "reward": 0.0,
        "correctness_verified": False,
        "verification_status": "failed",
        "validation_scope": "cpu_behavior",
        "performance_status": "unmeasured",
        "error_type": error_type,
        "failure_reason": reason,
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 1


def _pass() -> int:
    payload = {
        "reward": 1.0,
        "correctness_verified": True,
        "verification_status": "completed",
        "validation_scope": "cpu_behavior",
        "performance_status": "unmeasured",
        "native_kernel_bound": None,
        "measured_npu_speedup": None,
        "speedup_ratio": None,
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
    repro_path = workspace / "tests" / "repro_moe_quant_accuracy.py"
    if not repro_path.is_file():
        return _fail("Missing tests/repro_moe_quant_accuracy.py in workspace", "MissingTargetFileError")

    # 1. Anti-tampering check on tests/repro_moe_quant_accuracy.py
    if _sha256(repro_path) != EXPECTED_REPRO_SHA256:
        return _fail(
            "Forbidden tampering detected: tests/repro_moe_quant_accuracy.py was modified",
            "ForbiddenTamperingError",
        )

    if (workspace / ".git").exists():
        diff_proc = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--name-only", "HEAD", "--", "tests/repro_moe_quant_accuracy.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.stdout.strip():
            return _fail(
                "Forbidden tampering in git diff: tests/repro_moe_quant_accuracy.py",
                "ForbiddenTamperingError",
            )

    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))

    try:
        import tests.repro_moe_quant_accuracy as repro
        from vllm_ascend.ops.fused_moe.moe_mlp import quant_apply_mlp, unquant_apply_mlp
    except Exception as exc:
        return _fail(f"Failed to import moe_mlp or repro: {exc}", type(exc).__name__)

    # 2. Run multi-layer MoE accuracy evaluation
    try:
        report = repro.run_accuracy_evaluation()
    except Exception as exc:
        return _fail(f"run_accuracy_evaluation raised {type(exc).__name__}: {exc}", type(exc).__name__)

    cos_sim = float(report["final_cosine_similarity"])
    cos_sim_fp32_router = float(report["fp32_router_cosine_similarity"])
    if cos_sim < 0.99:
        return _fail(f"End-to-end W8A8 vs FP32 cosine similarity too low: {cos_sim:.6f} (expected >= 0.99)")
    if cos_sim_fp32_router < 0.999:
        return _fail(
            f"W8A8 vs FP32 cosine similarity with FP32 router too low: {cos_sim_fp32_router:.6f} (expected >= 0.999)"
        )

    # 3. Direct unit verification of quant_apply_mlp vs unquant_apply_mlp with fusion=True and dynamic_scale!=None
    gen = torch.Generator().manual_seed(9876)
    num_tokens, hidden_size, inter_size, num_experts = 12, 32, 64, 4
    x_fp32 = torch.randn((num_tokens, hidden_size), generator=gen, dtype=torch.float32) * torch.linspace(
        0.3, 2.8, steps=num_tokens
    ).unsqueeze(-1)
    w1_fp32 = torch.randn((num_experts, hidden_size, inter_size * 2), generator=gen, dtype=torch.float32) * 0.15
    w2_fp32 = torch.randn((num_experts, inter_size, hidden_size), generator=gen, dtype=torch.float32) * 0.15

    w1_scale = (w1_fp32.abs().amax(dim=1).clamp(min=1e-5) / 127.0).to(torch.bfloat16)
    w1_q = torch.round(w1_fp32 / w1_scale.to(torch.float32).unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    w2_scale = (w2_fp32.abs().amax(dim=1).clamp(min=1e-5) / 127.0).to(torch.bfloat16)
    w2_q = torch.round(w2_fp32 / w2_scale.to(torch.float32).unsqueeze(1)).clamp(-128, 127).to(torch.int8)

    group_list = torch.tensor([3, 4, 2, 3], dtype=torch.int64)
    q_x, dyn_scale = repro._quantize_per_token_int8(x_fp32)

    ref_out, _ = unquant_apply_mlp(
        hidden_states=x_fp32,
        w1=w1_fp32,
        w2=w2_fp32,
        group_list=group_list,
        group_list_type=1,
        need_trans=False,
    )
    quant_out, _ = quant_apply_mlp(
        hidden_states=q_x,
        w1=w1_q,
        w1_scale=w1_scale,
        w2=w2_q,
        w2_scale=w2_scale,
        group_list=group_list,
        group_list_type=1,
        dynamic_scale=dyn_scale,
        fusion=True,
        dynamic_eplb=False,
        act_quant_type=torch.int8,
    )
    direct_cos = float(
        F.cosine_similarity(ref_out.to(torch.float32).reshape(1, -1), quant_out.to(torch.float32).reshape(1, -1)).item()
    )
    if direct_cos < 0.998:
        return _fail(
            f"Direct quant_apply_mlp(fusion=True, dynamic_scale=...) cosine similarity too low: {direct_cos:.6f}"
        )

    # 4. Also verify LockstepDiffer(heal=True) on Layer 4 when mindie_debugger is available
    debugger_repo = Path("/Users/maoxx241/code/debugger")
    if debugger_repo.is_dir() and str(debugger_repo) not in sys.path:
        sys.path.insert(0, str(debugger_repo))
    try:
        from mindie_debugger.precision import LockstepDiffer

        model, x = repro.build_model_and_inputs()
        with torch.no_grad(), LockstepDiffer(model, heal=True) as differ:
            model.use_quant_mode = False
            model.force_fp32_router = True
            differ.begin_baseline("fp32_ref")
            _ = model(x.clone())
            differ.end_baseline()

            model.use_quant_mode = True
            model.force_fp32_router = False
            differ.begin_target("w8a8_quant")
            _ = model(x.clone())
            differ.end_target()
            rep = differ.report

        for m in rep["metrics"]:
            if m.get("module_path") == "layers.4.mlp" and m.get("tree_path") == "outputs":
                if float(m.get("rel_l2_err", 1.0)) > 0.05:
                    return _fail(f"Healed layers.4.mlp rel_l2_err still too high: {m.get('rel_l2_err')}")
    except ImportError:
        pass

    return _pass()


if __name__ == "__main__":
    raise SystemExit(main())
