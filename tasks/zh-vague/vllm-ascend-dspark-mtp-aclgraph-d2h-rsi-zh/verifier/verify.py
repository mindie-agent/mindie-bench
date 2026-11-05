#!/usr/bin/env python3
"""Verifier for vllm-ascend-dspark-mtp-aclgraph-d2h-rsi-zh."""

from __future__ import annotations

import contextlib
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
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
ALLOWED_PREFIXES = (
    "vllm_ascend/worker/v2/spec_decode/dspark/",
    "vllm_ascend/worker/v2/spec_decode/dflash/",
    "tests/",
)


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
    changed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    unexpected = [p for p in changed if not p.startswith(ALLOWED_PREFIXES)]
    if unexpected:
        raise AssertionError(f"Modified files outside allowed scope: {sorted(unexpected)}")


def load_speculator_module():
    for mod_name in [
        "vllm",
        "vllm.config",
        "vllm.config.compilation",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention_layer_base",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.input_batch",
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.dspark",
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator",
        "vllm_ascend",
        "vllm_ascend.worker",
        "vllm_ascend.worker.v2",
        "vllm_ascend.worker.v2.attn_utils",
    ]:
        sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["vllm.config"].VllmConfig = object
    sys.modules["vllm.config"].get_layers_from_vllm_config = lambda *a, **k: {}
    sys.modules["vllm.config.compilation"].CUDAGraphMode = object
    sys.modules["vllm.model_executor.layers.attention_layer_base"].AttentionLayerBase = object
    sys.modules["vllm.v1.attention.backend"].AttentionBackend = object
    sys.modules["vllm.v1.worker.gpu.input_batch"].InputBatch = object
    sys.modules["vllm.v1.worker.gpu.spec_decode.dspark.speculator"].DSparkSpeculator = object
    sys.modules["vllm_ascend.worker.v2.attn_utils"].build_attn_metadata_wrapper = contextlib.nullcontext
    sys.modules["vllm_ascend.worker.v2.attn_utils"].build_draft_attn_metadata_factory = (
        lambda *a, **k: contextlib.nullcontext()
    )

    target = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
    spec = importlib.util.spec_from_file_location(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator",
        target,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class DSparkTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_sync_ops = 0
        self.clone_ops = 0

    def __torch_dispatch__(self, fn, types_arg, args=(), kwargs=None):
        kwargs = kwargs or {}
        if fn == torch.ops.aten._local_scalar_dense.default:
            self.scalar_sync_ops += 1
        if fn == torch.ops.aten.clone.default:
            self.clone_ops += 1
        return fn(*args, **kwargs)


def _invoke_step(mod, req_seq_lens, accepted_mask, draft_query_hidden, slot_mapping, **flags):
    step_fn = getattr(mod, "dspark_aclgraph_speculative_step", None) or getattr(mod, "dspark_aclgraph_step", None)
    assert step_fn is not None, "Missing dspark_aclgraph_speculative_step in speculator.py"
    res = step_fn(
        req_seq_lens,
        accepted_mask,
        draft_query_hidden,
        slot_mapping,
        num_speculative_tokens=3,
        num_query_per_req=3,
        bucket_sizes=(4, 8, 12, 16, 24, 32),
        **flags,
    )
    if isinstance(res, tuple):
        updated, metrics = res
    else:
        metrics = res
        updated = metrics["updated_seq_lens"]
    return updated, metrics


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
        mod = load_speculator_module()

        # Probe 1: num_reqs = 3 (draft_raw = 9 -> 12, target_raw = 12 -> 16)
        seq1 = torch.tensor([16, 24, 12], dtype=torch.int32)
        acc1 = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.int32)
        q1 = torch.randn((9, 64), dtype=torch.bfloat16)
        slots1 = torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=torch.int32)
        seq1_in, acc1_in, q1_in, slots1_in = seq1.clone(), acc1.clone(), q1.clone(), slots1.clone()
        with DSparkTrace() as tr1:
            upd1, m1 = _invoke_step(mod, seq1_in, acc1_in, q1_in, slots1_in)
        assert tr1.scalar_sync_ops == 0, f"Captured step executed {tr1.scalar_sync_ops} host .item()/.tolist() D2H syncs on batch=3"
        assert tr1.clone_ops == 0, f"Captured step executed {tr1.clone_ops} tensor .clone() allocations on batch=3"
        assert upd1.tolist() == [19, 26, 16], f"Unexpected updated_seq_lens on batch=3: {upd1.tolist()}"
        assert m1.get("captured_step_d2h_syncs") == 0, f"m1 syncs={m1.get('captured_step_d2h_syncs')}"
        assert m1.get("metadata_allocs_in_capture") == 0, f"m1 metadata_allocs={m1.get('metadata_allocs_in_capture')}"
        assert m1.get("draft_tokens_padded") == 12, f"m1 draft={m1.get('draft_tokens_padded')}"
        assert m1.get("target_tokens_padded") == 16, f"m1 target={m1.get('target_tokens_padded')}"
        aq1 = m1.get("actual_seq_lengths_q")
        aq1_list = aq1.tolist() if isinstance(aq1, torch.Tensor) else list(aq1 or ())
        assert aq1_list == [3, 6, 9, 12], f"Invalid actual_seq_lengths_q on batch=3: {aq1}"

        # Probe 2: unseen batch size num_reqs = 5 (draft_raw = 15 -> 16, target_raw = 20 -> 24)
        seq2 = torch.tensor([8, 19, 31, 42, 7], dtype=torch.int32)
        acc2 = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=torch.int32)
        q2 = torch.randn((15, 64), dtype=torch.bfloat16)
        slots2 = torch.arange(200, 215, dtype=torch.int32)
        seq2_in, acc2_in, q2_in, slots2_in = seq2.clone(), acc2.clone(), q2.clone(), slots2.clone()
        with DSparkTrace() as tr2:
            upd2, m2 = _invoke_step(mod, seq2_in, acc2_in, q2_in, slots2_in)
        assert tr2.scalar_sync_ops == 0, f"Captured step executed {tr2.scalar_sync_ops} host .item()/.tolist() D2H syncs on batch=5"
        assert tr2.clone_ops == 0, f"Captured step executed {tr2.clone_ops} tensor .clone() allocations on batch=5"
        assert upd2.tolist() == [9, 23, 33, 45, 9], f"Unexpected updated_seq_lens on batch=5: {upd2.tolist()}"
        assert m2.get("captured_step_d2h_syncs") == 0, f"m2 syncs={m2.get('captured_step_d2h_syncs')}"
        assert m2.get("metadata_allocs_in_capture") == 0, f"m2 metadata_allocs={m2.get('metadata_allocs_in_capture')}"
        assert m2.get("draft_tokens_padded") == 16, f"Unexpected draft_tokens_padded on batch=5: {m2.get('draft_tokens_padded')}"
        assert m2.get("target_tokens_padded") == 24, f"Unexpected target_tokens_padded on batch=5: {m2.get('target_tokens_padded')}"
        aq2 = m2.get("actual_seq_lengths_q")
        aq2_list = aq2.tolist() if isinstance(aq2, torch.Tensor) else list(aq2 or ())
        assert aq2_list == [3, 6, 9, 12, 15, 16], f"Invalid actual_seq_lengths_q on batch=5: {aq2}"

        evidence["correctness_verified"] = True
        assert hasattr(mod, "run_ablation_suite"), "Missing run_ablation_suite in speculator.py"
        try:
            suite = mod.run_ablation_suite()
        except TypeError:
            suite = mod.run_ablation_suite(seq1.clone(), acc1.clone(), q1.clone(), slots1.clone())
        assert isinstance(suite, dict), f"run_ablation_suite must return a dict, got {type(suite).__name__}"
        evidence["candidate_suite_type"] = type(suite).__name__
        required_arms = {
            "full_candidate",
            "minus_bucket_alignment",
            "minus_metadata_hoist",
            "minus_device_resident_accepted_tokens",
        }
        assert required_arms <= set(suite.keys()), (
            f"run_ablation_suite missing required arms: {sorted(required_arms - set(suite.keys()))}"
        )

        arm_traces: dict[str, dict[str, object]] = {
            "full_candidate": {
                "scalar_sync_ops": tr1.scalar_sync_ops,
                "clone_ops": tr1.clone_ops,
                "draft_tokens_padded": m1.get("draft_tokens_padded"),
                "target_tokens_padded": m1.get("target_tokens_padded"),
                "metadata_allocs_in_capture": m1.get("metadata_allocs_in_capture", 0),
                "captured_step_d2h_syncs": m1.get("captured_step_d2h_syncs", 0),
                "updated_ok": True,
                "status": "ok",
            }
        }
        for arm_name, flags in [
            ("minus_bucket_alignment", dict(align_buckets=False, hoist_metadata=True, device_resident_accepted=True)),
            ("minus_metadata_hoist", dict(align_buckets=True, hoist_metadata=False, device_resident_accepted=True)),
            (
                "minus_device_resident_accepted_tokens",
                dict(align_buckets=True, hoist_metadata=True, device_resident_accepted=False),
            ),
        ]:
            try:
                seq_arm = seq1.clone()
                acc_arm = acc1.clone()
                q_arm = q1.clone()
                slots_arm = slots1.clone()
                with DSparkTrace() as tr_arm:
                    upd_arm, m_arm = _invoke_step(
                        mod,
                        seq_arm,
                        acc_arm,
                        q_arm,
                        slots_arm,
                        **flags,
                    )
                aq_arm = m_arm.get("actual_seq_lengths_q")
                aq_arm_list = aq_arm.tolist() if isinstance(aq_arm, torch.Tensor) else list(aq_arm or ())
                arm_traces[arm_name] = {
                    "scalar_sync_ops": tr_arm.scalar_sync_ops,
                    "clone_ops": tr_arm.clone_ops,
                    "draft_tokens_padded": m_arm.get("draft_tokens_padded"),
                    "target_tokens_padded": m_arm.get("target_tokens_padded"),
                    "actual_seq_lengths_q": aq_arm_list,
                    "metadata_allocs_in_capture": m_arm.get("metadata_allocs_in_capture", 0),
                    "captured_step_d2h_syncs": m_arm.get("captured_step_d2h_syncs", 0),
                    "updated_ok": upd_arm.tolist() == [19, 26, 16],
                    "status": "ok",
                }
            except RuntimeError as exc:
                arm_traces[arm_name] = {
                    "status": "error",
                    "error": str(exc),
                }

        b_arm = arm_traces["minus_bucket_alignment"]
        assert (
            b_arm.get("status") == "error"
            or b_arm.get("target_tokens_padded") != 16
            or (b_arm.get("actual_seq_lengths_q") or [None])[-1] != b_arm.get("draft_tokens_padded")
        ), f"minus_bucket_alignment did not exhibit bucket/TND mismatch: {b_arm}"

        h_arm = arm_traces["minus_metadata_hoist"]
        assert (
            h_arm.get("status") == "ok"
            and int(h_arm.get("metadata_allocs_in_capture", 0)) > int(m1.get("metadata_allocs_in_capture", 0))
            and int(h_arm.get("clone_ops", 0)) > tr1.clone_ops
        ), f"minus_metadata_hoist did not exhibit metadata allocations/clones: {h_arm}"

        d_arm = arm_traces["minus_device_resident_accepted_tokens"]
        assert (
            d_arm.get("status") == "ok"
            and int(d_arm.get("captured_step_d2h_syncs", 0)) > int(m1.get("captured_step_d2h_syncs", 0))
            and int(d_arm.get("scalar_sync_ops", 0)) > tr1.scalar_sync_ops
        ), f"minus_device_resident_accepted_tokens did not exhibit host D2H syncs: {d_arm}"

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
