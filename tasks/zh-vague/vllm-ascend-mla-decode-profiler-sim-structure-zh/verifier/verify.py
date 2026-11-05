#!/usr/bin/env python3
"""Behavioral verifier for vllm-ascend-mla-decode-profiler-sim-structure-zh."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ALLOWED_FILES = {
    "vllm_ascend/ops/mla.py",
    "vllm_ascend/worker/model_runner_v1.py",
    "vllm_ascend/attention/mla_v1.py",
}


class TrackedTensor:
    """Wrapper around torch.Tensor that counts dtype casts, clones, and D2H syncs."""

    def __init__(self, tensor: torch.Tensor, stats: dict[str, int]):
        self._t = tensor
        self._stats = stats

    @property
    def shape(self):
        return self._t.shape

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def device(self):
        return self._t.device

    def data_ptr(self) -> int:
        return self._t.data_ptr()

    def is_contiguous(self) -> bool:
        return self._t.is_contiguous()

    def contiguous(self) -> "TrackedTensor":
        if self._t.is_contiguous():
            return self
        return TrackedTensor(self._t.contiguous(), self._stats)

    def to(self, *args, **kwargs) -> "TrackedTensor":
        target = args[0] if args else kwargs.get("dtype")
        if isinstance(target, torch.dtype) and target != self._t.dtype:
            self._stats["cast_count"] += 1
        return TrackedTensor(self._t.to(*args, **kwargs), self._stats)

    def clone(self) -> "TrackedTensor":
        self._stats["clone_count"] += 1
        return TrackedTensor(self._t.clone(), self._stats)

    def copy_(self, other: Any) -> "TrackedTensor":
        self._stats["copy_count"] += 1
        src = other._t if isinstance(other, TrackedTensor) else other
        self._t.copy_(src)
        return self

    def cpu(self) -> "TrackedTensor":
        self._stats["cpu_sync_count"] += 1
        return TrackedTensor(self._t.cpu(), self._stats)

    def tolist(self):
        return self._t.tolist()

    def view(self, *shape) -> "TrackedTensor":
        return TrackedTensor(self._t.view(*shape), self._stats)


def _check_git_diff(workspace: Path) -> None:
    assert not (workspace / "vllm_ascend_diag").exists(), "Forbidden stub directory vllm_ascend_diag exists"
    res = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    changed = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    for path in changed:
        assert path in ALLOWED_FILES or path.startswith("tests/"), (
            f"Unexpected modified file outside allowed target_files: {path}"
        )


def _load_mla_functions(mla_path: Path, forward_ctx: Any, custom_op_hook: Any) -> dict:
    source = mla_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(mla_path))
    cls_forward = None
    mla_forward_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AscendMultiHeadLatentAttention":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    cls_forward = item
                    break
        elif isinstance(node, ast.FunctionDef) and node.name == "mla_forward":
            mla_forward_node = node
    assert cls_forward is not None and mla_forward_node is not None, (
        "Missing AscendMultiHeadLatentAttention.forward or mla_forward in vllm_ascend/ops/mla.py"
    )
    mod = ast.Module(body=[cls_forward, mla_forward_node], type_ignores=[])
    ast.fix_missing_locations(mod)

    torch_stub = SimpleNamespace(
        Tensor=torch.Tensor,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
        int32=torch.int32,
        empty=lambda shape, dtype=None, device=None: TrackedTensor(
            torch.empty(shape, dtype=dtype, device=device), forward_ctx.stats
        ),
        ops=SimpleNamespace(vllm=SimpleNamespace(mla_forward=custom_op_hook)),
    )
    ns: dict[str, Any] = {
        "torch": torch_stub,
        "AttentionMetadata": Any,
        "ForwardContext": Any,
        "get_forward_context": lambda: forward_ctx,
        "_EXTRA_CTX": SimpleNamespace(flash_comm_v1_enabled=False),
    }
    exec(compile(mod, str(mla_path), "exec"), ns)
    return ns


def _load_runner_metadata_method(runner_path: Path):
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner_path))
    method_node = None
    build_attn_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_prepare_mla_decode_step_metadata":
                    method_node = item
                elif isinstance(item, ast.FunctionDef) and item.name == "_build_attention_metadata":
                    build_attn_node = item
    assert method_node is not None and build_attn_node is not None, (
        "Missing _prepare_mla_decode_step_metadata in model_runner_v1.py"
    )
    mod = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, Any] = {
        "CommonAttentionMetadata": Any,
        "copy": copy,
        "deepcopy": deepcopy,
        "torch": torch,
    }
    exec(compile(mod, str(runner_path), "exec"), ns)
    return ns["_prepare_mla_decode_step_metadata"]


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
    _check_git_diff(workspace)

    mla_path = workspace / "vllm_ascend/ops/mla.py"
    runner_path = workspace / "vllm_ascend/worker/model_runner_v1.py"
    assert mla_path.is_file(), f"Missing {mla_path}"
    assert runner_path.is_file(), f"Missing {runner_path}"

    stats = {"cast_count": 0, "clone_count": 0, "copy_count": 0, "cpu_sync_count": 0}
    recorded_impl_calls = []

    def impl_forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output):
        recorded_impl_calls.append((hidden_states, output))
        src = hidden_states._t if isinstance(hidden_states, TrackedTensor) else hidden_states
        dst = output._t if isinstance(output, TrackedTensor) else output
        dst.copy_(src * 2.0)

    layer_stub = SimpleNamespace(
        hidden_size=64,
        tp_size=2,
        is_vl_first_layer=False,
        prefix="model.layers.0.self_attn",
        mla_attn=SimpleNamespace(
            layer_name="model.layers.0.self_attn.attn",
            kv_cache=torch.zeros((4, 64), dtype=torch.bfloat16),
            impl=SimpleNamespace(forward=impl_forward),
        ),
    )
    forward_ctx = SimpleNamespace(
        no_compile_layers={"model.layers.0.self_attn": layer_stub},
        attn_metadata={"model.layers.0.self_attn.attn": SimpleNamespace()},
        stats=stats,
    )

    mla_ns: dict[str, Any] = {}

    def custom_op_hook(hidden_states, need_gather_q_kv, output, layer_name):
        mla_ns["mla_forward"](hidden_states, need_gather_q_kv, output, layer_name)

    mla_ns.update(_load_mla_functions(mla_path, forward_ctx, custom_op_hook))

    raw_hidden = torch.ones((8, 64), dtype=torch.bfloat16)
    tracked_hidden = TrackedTensor(raw_hidden, stats)
    positions = torch.arange(8, dtype=torch.int64)

    out = mla_ns["forward"](layer_stub, positions, tracked_hidden)
    out_tensor = out._t if isinstance(out, TrackedTensor) else out

    assert torch.allclose(out_tensor, raw_hidden * 2.0), "Incorrect MLA forward output values"
    assert stats["cast_count"] == 0, f"Expected 0 dtype casts in MLA decode forward, got {stats['cast_count']}"
    assert stats["clone_count"] == 0, f"Expected 0 tensor clones in MLA decode forward, got {stats['clone_count']}"
    assert stats["copy_count"] == 0, (
        f"Expected 0 redundant staged output copies in MLA decode forward, got {stats['copy_count']}"
    )
    assert len(recorded_impl_calls) == 1
    passed_hidden, _ = recorded_impl_calls[0]
    assert passed_hidden.data_ptr() == raw_hidden.data_ptr(), (
        "Contiguous hidden_states must be passed zero-copy into mla_attn.impl.forward"
    )

    # Scenario 2: NPUModelRunner._prepare_mla_decode_step_metadata or AscendMLAMetadataBuilder zero-copy check
    prepare_meta = _load_runner_metadata_method(runner_path)
    meta_stats = {"cast_count": 0, "clone_count": 0, "copy_count": 0, "cpu_sync_count": 0}
    seq_lens_t = TrackedTensor(torch.tensor([512, 1024, 2048, 4096], dtype=torch.int32), meta_stats)
    blk_table_t = TrackedTensor(torch.zeros((4, 32), dtype=torch.int32), meta_stats)
    slot_map_t = TrackedTensor(torch.arange(4, dtype=torch.int64), meta_stats)
    cm_base = SimpleNamespace(
        seq_lens=seq_lens_t,
        block_table_tensor=blk_table_t,
        slot_mapping=slot_map_t,
        num_reqs=4,
    )
    cm_out = prepare_meta(SimpleNamespace(), cm_base)
    mla_v1_src = (workspace / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8")
    mla_v1_fixed = mla_v1_src.count("common_attn_metadata.positions[: self.num_actual_tokens].long()") <= 1
    runner_fixed = (
        cm_out is not cm_base
        and meta_stats["cast_count"] == 0
        and meta_stats["clone_count"] == 0
        and meta_stats["cpu_sync_count"] == 0
        and cm_out.seq_lens.data_ptr() == seq_lens_t.data_ptr()
        and cm_out.block_table_tensor.data_ptr() == blk_table_t.data_ptr()
        and cm_out.slot_mapping.data_ptr() == slot_map_t.data_ptr()
    )
    assert runner_fixed or mla_v1_fixed, (
        f"Expected redundant casts/clones eliminated in decode metadata preparation: meta_stats={meta_stats}"
    )

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "vllm_ascend_mla_decode_profiler_sim_structure",
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"reward": 0.0, "failure_reason": str(exc), "failure_type": type(exc).__name__}))
        raise SystemExit(1)
