#!/usr/bin/env python3
"""Verifier for vllm-ascend-aclgraph-capture-and-replay-zh."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

EXPECTED_REPRO_SHA256 = "2699cbbad3f0d694c8704c82974fe047d7dd475f8969dfb0fdb9e080812db9ea"


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
    repro_path = workspace / "tests" / "repro_aclgraph_runner.py"
    if not repro_path.is_file():
        return _fail("Missing tests/repro_aclgraph_runner.py in workspace", "MissingTargetFileError")

    # 1. Anti-tampering check on tests/repro_aclgraph_runner.py
    if _sha256(repro_path) != EXPECTED_REPRO_SHA256:
        return _fail(
            "Forbidden tampering detected: tests/repro_aclgraph_runner.py was modified",
            "ForbiddenTamperingError",
        )

    if (workspace / ".git").exists():
        diff_proc = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--name-only", "HEAD", "--", "tests/repro_aclgraph_runner.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.stdout.strip():
            return _fail(
                "Forbidden tampering in git diff: tests/repro_aclgraph_runner.py",
                "ForbiddenTamperingError",
            )

    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))

    try:
        import tests.repro_aclgraph_runner as repro
    except Exception as exc:
        return _fail(f"Failed to import repro_aclgraph_runner: {exc}", type(exc).__name__)

    # 2. Run capture + multi-step variable-batch replay comparison against Eager
    try:
        report = repro.run_aclgraph_vs_eager()
    except Exception as exc:
        return _fail(f"run_aclgraph_vs_eager raised {type(exc).__name__}: {exc}", type(exc).__name__)

    if not report.get("matches_eager"):
        return _fail(f"ACLGraph replay diverged from Eager: {report}")
    if float(report["step1_max_abs_err"]) >= 1e-5:
        return _fail(f"Step 1 max_abs_err too high: {report['step1_max_abs_err']}")
    if float(report["step2_max_abs_err"]) >= 1e-5:
        return _fail(f"Step 2 max_abs_err too high: {report['step2_max_abs_err']}")
    if float(report["kv_cache_max_abs_err"]) >= 1e-5:
        return _fail(f"KV cache max_abs_err too high: {report['kv_cache_max_abs_err']}")

    # 3. Direct verification of static buffer pointer stability and padded slot_mapping -1 masking
    model = repro.ToyDecodeModel(hidden_size=16, vocab_size=64, num_slots=32)
    model.eval()
    runner = repro._make_lightweight_runner(model, max_tokens=4, max_reqs=4)
    runner.input_batch.block_table[0].slot_mapping.gpu[:4].copy_(torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    int(runner.input_batch.block_table[0].slot_mapping.gpu.data_ptr())
    static_pos_ptr = int(runner.positions.data_ptr())

    attn_capture_dict, _ = runner._build_attention_metadata(
        num_tokens=4,
        num_reqs=4,
        max_query_len=1,
        num_tokens_padded=4,
        num_reqs_padded=4,
        for_cudagraph_capture=True,
    )
    cm_capture = attn_capture_dict["layers.0.self_attn"]
    if int(cm_capture.slot_mapping.untyped_storage().data_ptr()) != int(
        runner.input_batch.block_table[0].slot_mapping.gpu.untyped_storage().data_ptr()
    ):
        return _fail("slot_mapping was rebound to a new storage during capture instead of using static slot_mapping.gpu")

    attn_dict, _ = runner._build_attention_metadata(
        num_tokens=2,
        num_reqs=2,
        max_query_len=1,
        num_tokens_padded=4,
        num_reqs_padded=4,
        for_cudagraph_capture=False,
    )
    attn_dict["layers.0.self_attn"]
    if runner.input_batch.block_table[0].slot_mapping.gpu[2:4].tolist() != [-1, -1]:
        return _fail(
            f"Padded slot_mapping.gpu[2:4] was not masked to -1: got {runner.input_batch.block_table[0].slot_mapping.gpu[:4].tolist()}"
        )

    captured_pos_ptrs: list[int] = []

    def _spy_model(**kwargs):
        captured_pos_ptrs.append(int(kwargs["positions"].data_ptr()))
        return torch.zeros((4, 16), dtype=torch.float32)

    runner.model = _spy_model
    fwd_ctx = repro.get_forward_context()
    fwd_ctx.capturing = False
    fwd_ctx.cudagraph_runtime_mode = repro.CUDAGraphMode.FULL
    _ = runner._model_forward(4, input_ids=runner.input_ids[:4], positions=runner.positions[:4])
    if not captured_pos_ptrs or captured_pos_ptrs[0] != static_pos_ptr:
        return _fail("positions tensor was rebound to a new data_ptr() inside _model_forward during replay")

    # 4. Also verify under ACLGraphVllmInspector when mindie_debugger is available
    debugger_repo = Path("/Users/maoxx241/code/debugger")
    if debugger_repo.is_dir() and str(debugger_repo) not in sys.path:
        sys.path.insert(0, str(debugger_repo))
    try:
        from mindie_debugger.aclgraph import ACLGraphVllmInspector

        with ACLGraphVllmInspector(compare_eager=True) as inspector:
            _ = repro.run_aclgraph_vs_eager()
        insp_rep = inspector.report()
        if insp_rep.get("capture_guard_violations") or insp_rep.get("first_divergent_layer") is not None:
            return _fail(f"ACLGraphVllmInspector reported violations: {insp_rep}")
    except ImportError:
        pass

    return _pass()


if __name__ == "__main__":
    raise SystemExit(main())
