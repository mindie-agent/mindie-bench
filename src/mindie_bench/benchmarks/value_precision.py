"""Localize five real precision defects with the installed debugger primitives.

[Role] Replay production-shaped GDN, SFA, KDA, GroupNorm/SparseMLA and MoE/sampler
localization cases against the unchanged ``mindie_debugger.precision`` tools.
[Invariants] Every pre-fix path must be detected, every post-fix path must stay clean,
and two complete runs must serialize to byte-identical JSON.
[Timeouts & Locks] All cases are bounded CPU tensor work; no sleeps, locks or devices.
[Side-Effect Policy] Writes only the caller's precision-localization.json and uses
private temporary directories for the synthetic checkpoint fixture.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path

from .value_common import write_json

_ATOL = 5e-3
_DIVERGENCE_EPS = 5e-3
_SOURCE_EXPERTS = 256
_SLIM_EXPERTS = 16
_TOP_K = 4
_ACTIVE_EXPERTS = [0, 2, 5, 7]
_EXPERT_LAYERS = 3
_MOE_BUG_MODULE = "layers.1.experts.2"
_SLIM_LAYER0_NOISE = 2e-3
_SLIM_BUG_BIAS = 0.25


def _torch():
    """Import torch only when a case actually executes tensors."""
    import torch

    return torch


def _rounded(value) -> float:
    return round(float(value), 6)


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a minimal little-endian safetensors shard without a runtime dependency."""
    torch = _torch()
    header: dict = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().to("cpu", copy=True).contiguous()
        raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
        header[name] = {
            "dtype": "F32",
            "shape": [int(item) for item in tensor.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(chunks))


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length).decode("utf-8").rstrip(" "))


def _build_moe_checkpoint(root: Path) -> tuple[dict, dict]:
    """Create a 256-expert checkpoint, slim it to 16 and prove the saved layout."""
    torch = _torch()
    from mindie_debugger.precision import slim_moe_checkpoint

    source = root / "moe-source"
    source.mkdir()
    generator = torch.Generator().manual_seed(11)
    tensors = {}
    for layer in range(_EXPERT_LAYERS):
        prefix = f"layers.{layer}"
        tensors[f"{prefix}.experts.weight"] = (
            torch.randn((_SOURCE_EXPERTS, 8, 8), generator=generator) * 0.05)
        tensors[f"{prefix}.shared.weight"] = torch.randn((8, 8), generator=generator) * 0.05
        tensors[f"{prefix}.gate.weight"] = (
            torch.randn((_SOURCE_EXPERTS, 8), generator=generator) * 0.05)
    _write_safetensors(source / "model.safetensors", tensors)
    config = {"num_routed_experts": _SOURCE_EXPERTS, "num_experts_per_tok": _TOP_K,
              "num_hidden_layers": _EXPERT_LAYERS, "n_group": 8, "topk_group": 4,
              "hidden_size": 8}
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8", newline="\n")
    slimmed = root / "moe-slim"
    report = slim_moe_checkpoint(source, slimmed, _SLIM_EXPERTS)
    header = _read_safetensors_header(slimmed / "model.safetensors")
    updated = json.loads((slimmed / "config.json").read_text(encoding="utf-8"))
    assert (report["current_routed_experts"], report["target_routed_experts"]) == (
        _SOURCE_EXPERTS, _SLIM_EXPERTS)
    assert (report["num_hidden_layers_before"], report["num_hidden_layers_after"]) == (
        _EXPERT_LAYERS, _EXPERT_LAYERS)
    assert header["layers.1.experts.weight"]["shape"] == [_SLIM_EXPERTS, 8, 8]
    assert header["layers.1.gate.weight"]["shape"] == [_SOURCE_EXPERTS, 8]
    assert updated["num_routed_experts"] == _SLIM_EXPERTS
    return report, {
        "sliced_tensor_names": list(report["sliced_tensor_names"]),
        "expert_shape": list(header["layers.1.experts.weight"]["shape"]),
        "gate_shape": list(header["layers.1.gate.weight"]["shape"]),
        "config_num_routed_experts": int(updated["num_routed_experts"]),
        "filtered_expert_keys": int(report["filtered_expert_keys"]),
    }


def _moe_diverged(report: dict) -> tuple[list[str], str | None]:
    """List diverged modules in execution order and the first unhealed one."""
    invocations = {
        (item["module_path"], item["invocation_ordinal"], item["rank"]): item
        for item in report["invocations"]
    }
    diverged: list[str] = []
    first_unhealed = None
    for key in report["module_order"]:
        item = invocations.get(tuple(key))
        if item is None:
            continue
        if float(item.get("max_abs_err", 0.0)) <= _DIVERGENCE_EPS:
            continue
        if item["module_path"] not in diverged:
            diverged.append(item["module_path"])
        if first_unhealed is None and not item["healed"]:
            first_unhealed = item["module_path"]
    return diverged, first_unhealed


def _slim_moe_lockstep(torch, root: Path, stimulus: dict) -> dict:
    """Run healed and naive lockstep traces over a slimmed three-layer MoE."""
    from mindie_debugger.precision import LockstepDiffer, build_deterministic_stimulus

    report, checkpoint = _build_moe_checkpoint(root)
    assert stimulus == build_deterministic_stimulus(
        vocab_size=64, batch_size=2, seq_len=8, seed=7
    )
    tokens = torch.tensor(stimulus["prompt_token_ids"], dtype=torch.int64)
    embedding = torch.randn((64, 8), generator=torch.Generator().manual_seed(21))
    inputs = embedding[tokens].reshape(-1, 8)
    class _Expert(torch.nn.Module):
        def __init__(self, hidden: int, seed: int):
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.proj = torch.nn.Linear(hidden, hidden, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(
                    torch.randn((hidden, hidden), generator=generator) * 0.25
                )
            self.bias_offset = 0.0

        def forward(self, value):
            return torch.nn.functional.gelu(self.proj(value)) + self.bias_offset
    class _MoELayer(torch.nn.Module):
        def __init__(self, index: int, hidden: int, experts: int):
            super().__init__()
            self.active_expert_ids = list(_ACTIVE_EXPERTS)
            self.experts = torch.nn.ModuleList(
                [_Expert(hidden, 100 + index * 100 + expert)
                 for expert in range(experts)]
            )
            self.pre = torch.nn.Linear(hidden, hidden, bias=False)
            generator = torch.Generator().manual_seed(5000 + index)
            with torch.no_grad():
                self.pre.weight.copy_(
                    torch.randn((hidden, hidden), generator=generator)
                    * (8.0 if index == 1 else 0.5)
                )
            self.noise = 0.0

        def forward(self, value):
            hidden = self.pre(value)
            if self.noise:
                hidden = hidden + self.noise
            routed = sum(self.experts[expert](hidden) for expert in self.active_expert_ids)
            return value + routed / len(self.active_expert_ids)
    class _MoEStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [_MoELayer(layer, 8, _SLIM_EXPERTS) for layer in range(_EXPERT_LAYERS)]
            )

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)
            return value

    frontier = [f"layers.{layer}" for layer in range(_EXPERT_LAYERS)]

    def trace(heal: bool) -> dict:
        model = _MoEStack().eval()
        with torch.no_grad(), LockstepDiffer(model, heal=heal, frontier=frontier) as differ:
            differ.run_baseline(lambda: model(inputs.clone()))
            model.layers[0].noise = _SLIM_LAYER0_NOISE
            model.layers[1].experts[2].bias_offset = _SLIM_BUG_BIAS
            return differ.run_target(lambda: model(inputs.clone()))

    naive, healed = trace(heal=False), trace(heal=True)
    assert naive["status"] == "ok" and healed["status"] == "ok"
    naive_diverged, naive_first = _moe_diverged(naive)
    healed_diverged, healed_first = _moe_diverged(healed)
    assert healed_first == _MOE_BUG_MODULE and naive_first != healed_first
    def is_downstream(name: str) -> bool:
        return name in {"", "<root>", "layers.2"} or name.startswith("layers.2.")

    naive_cascade = [name for name in naive_diverged if is_downstream(name)]
    healed_cascade = [name for name in healed_diverged if is_downstream(name)]
    assert naive_cascade and healed_cascade == []
    assert len(naive_diverged) > len(healed_diverged)
    return {
        "source_routed_experts": int(report["current_routed_experts"]),
        "target_routed_experts": int(report["target_routed_experts"]),
        "top_k": _TOP_K, "active_expert_ids": list(_ACTIVE_EXPERTS),
        "num_hidden_layers": _EXPERT_LAYERS,
        "filtered_expert_keys": checkpoint["filtered_expert_keys"],
        "sliced_expert_tensors": len(checkpoint["sliced_tensor_names"]),
        "slimmed_expert_shape": checkpoint["expert_shape"],
        "unchanged_gate_shape": checkpoint["gate_shape"],
        "config_num_routed_experts": checkpoint["config_num_routed_experts"],
        "layer0_noise": _SLIM_LAYER0_NOISE, "bug_bias": _SLIM_BUG_BIAS,
        "first_diverged_module": healed_first,
        "naive_first_diverged_module": naive_first,
        "healed_diverged_modules": healed_diverged,
        "naive_diverged_modules": naive_diverged,
        "healed_downstream_diverged": len(healed_cascade),
        "naive_downstream_diverged": len(naive_cascade),
    }


def _w4a8_and_sampler(torch) -> dict:
    """Separate 4-bit quantization loss from a scale-stride kernel bug and sampler drift."""
    from mindie_debugger import precision

    contract = precision.FakeQuantContract(bits=4, symmetric=True, scale_axis=0)
    generator = torch.Generator().manual_seed(2026)
    original = (torch.randn((8, 16), generator=generator) * 0.5).bfloat16()
    healthy_weight = precision.fake_quantize(original, contract)
    class _QuantBlock(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone().float())

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight)

    quant_input = torch.randn((4, 16), generator=torch.Generator().manual_seed(31))
    healthy = _QuantBlock(healthy_weight)
    healthy_replay = precision.replay_call(
        precision.capture_call(healthy, quant_input), healthy, quant_bits=4,
        original_float_weight=original,
    )
    healthy_check = healthy_replay["fakequant_check"]
    assert healthy_replay["verdict"] == "inherent_quantization_error"
    assert healthy_check["fakequant_vs_kernel"]["max_abs_err"] <= 1e-6
    assert healthy_check["float_vs_fakequant"]["max_abs_err"] > 1e-3

    qmax = 7.0
    scale = original.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    quantized = (original.float() / scale).round().clamp(-qmax, qmax)
    # The kernel walks scales with an even-row stride instead of one scale per row.
    broken_scale = scale.repeat_interleave(2, dim=0)[: scale.shape[0]]
    broken = _QuantBlock(quantized * broken_scale)
    broken_replay = precision.replay_call(
        precision.capture_call(broken, quant_input), broken, quant_bits=4,
        original_float_weight=original,
    )
    broken_check = broken_replay["fakequant_check"]
    assert broken_replay["verdict"] == "real_numerical_bug"
    assert broken_check["fakequant_vs_kernel"]["max_abs_err"] > _ATOL

    def topk_prefix(logits, top_k):
        return torch.topk(logits, k=top_k, dim=-1)

    def topk_fixed(logits, top_k):
        return torch.topk(logits, k=min(int(top_k), int(logits.shape[-1])), dim=-1)

    sampler_logits = torch.tensor([[0.1, 2.0, -0.5, 1.0, 0.7, -1.0, 1.8, 0.3]])
    oversized_rejected = False
    try:
        topk_prefix(sampler_logits, 4096)
    except RuntimeError:
        oversized_rejected = True
    assert oversized_rejected
    values, indices = topk_fixed(sampler_logits, 4096)
    assert tuple(values.shape) == tuple(sampler_logits.shape) and int(indices[0, 0]) == 1

    def softmax_prefix(logits):
        return torch.softmax(logits, dim=-1).float()

    def softmax_fixed(logits):
        return logits.softmax(dim=-1, dtype=torch.float32)

    candidate_logits = torch.tensor(
        [[-5.0625, 1.2109375, 0.68359375, -1.703125]], dtype=torch.bfloat16
    )
    prefix_replay = precision.replay_call(
        precision.capture_call(softmax_prefix, candidate_logits), softmax_prefix,
        mode="reference_check",
    )
    fixed_replay = precision.replay_call(
        precision.capture_call(softmax_fixed, candidate_logits), softmax_fixed,
        mode="reference_check",
    )
    assert prefix_replay["verdict"] == "real_numerical_bug"
    assert prefix_replay["reference_check"]["max_abs_err"] > 1e-3
    assert fixed_replay["verdict"] == "pass"
    assert fixed_replay["reference_check"]["max_abs_err"] <= 1e-3
    return {
        "healthy_verdict": healthy_replay["verdict"],
        "healthy_inherent_error": _rounded(healthy_check["float_vs_fakequant"]["max_abs_err"]),
        "healthy_kernel_error": _rounded(healthy_check["fakequant_vs_kernel"]["max_abs_err"]),
        "scale_bug_verdict": broken_replay["verdict"],
        "scale_bug_error": _rounded(broken_check["fakequant_vs_kernel"]["max_abs_err"]),
        "oversized_topk_rejected": oversized_rejected,
        "oversized_topk_shape": list(values.shape),
        "softmax_prefix_verdict": prefix_replay["verdict"],
        "softmax_fixed_verdict": fixed_replay["verdict"],
        "softmax_max_abs_err": _rounded(prefix_replay["reference_check"]["max_abs_err"]),
    }


def _case_k3_gdn_idle_dp_state() -> dict:
    """Prove the idle-DP dummy GDN call mutates recurrent state only before the fix."""
    torch = _torch()
    from mindie_debugger.precision import MemoryWatchpoint
    class _GDNRecurrent(torch.nn.Module):
        def __init__(self, conv_state, ssm_state):
            super().__init__()
            self.conv_state, self.ssm_state = conv_state, ssm_state
            self.register_buffer("block_ids", torch.tensor([1, 2], dtype=torch.int64))

        def forward(self, mixed_qkv, num_actual_tokens):
            if num_actual_tokens > 0:
                rows = mixed_qkv[:num_actual_tokens].sum(0).unsqueeze(0)
                rows = rows.expand(self.block_ids.numel(), -1)
                self.conv_state.index_add_(0, self.block_ids, rows)
                self.ssm_state.index_add_(0, self.block_ids, rows * 2.0)
            else:
                mixed_qkv[:0].sum()
            return mixed_qkv
    class _GDNModel(torch.nn.Module):
        def __init__(self, conv_state, ssm_state):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].self_attn = _GDNRecurrent(conv_state, ssm_state)

        def forward(self, mixed_qkv, num_actual_tokens):
            for layer in self.layers:
                mixed_qkv = layer.self_attn(mixed_qkv, num_actual_tokens)
            return mixed_qkv

    class _GDNRunner:
        def __init__(self, model):
            self.model = model
            self.last_meta = {}

        def _dummy_run(self, num_tokens, skip_gdn_state_update=False):
            actual = 0 if skip_gdn_state_update else num_tokens
            self.last_meta = {
                "num_actual_tokens": actual,
                "max_query_len": actual,
                "query_start_loc": [0, actual],
            }
            mixed_qkv = torch.arange(
                1, num_tokens * 4 + 1, dtype=torch.float32
            ).reshape(num_tokens, 4)
            return self.model(mixed_qkv, actual)

    def states():
        conv = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        return conv, (torch.arange(32, dtype=torch.float32) + 100.0).reshape(8, 4)

    conv_state, ssm_state = states()
    model = _GDNModel(conv_state, ssm_state)
    with MemoryWatchpoint(
        must_preserve={"conv_state": conv_state, "ssm_state": ssm_state},
        module=model,
    ) as watchpoint:
        _GDNRunner(model)._dummy_run(1, skip_gdn_state_update=False)
    pre_violations = watchpoint.violation_count
    pre_module = watchpoint.offending_module
    assert pre_violations >= 2
    assert pre_module is not None and pre_module.endswith("layers.0.self_attn")

    fixed_conv, fixed_ssm = states()
    fixed_model = _GDNModel(fixed_conv, fixed_ssm)
    conv_before, ssm_before = fixed_conv.clone(), fixed_ssm.clone()
    with MemoryWatchpoint(
        must_preserve={"conv_state": fixed_conv, "ssm_state": fixed_ssm},
        module=fixed_model,
    ) as fixed_watchpoint:
        _GDNRunner(fixed_model)._dummy_run(1, skip_gdn_state_update=True)
    assert fixed_watchpoint.violation_count == 0
    assert fixed_watchpoint.offending_module is None
    assert torch.equal(fixed_conv, conv_before) and torch.equal(fixed_ssm, ssm_before)
    return {"pre_fix_violations": pre_violations,
            "pre_fix_offending_module": pre_module,
            "pre_fix_num_actual_tokens": 1,
            "post_fix_violations": fixed_watchpoint.violation_count,
            "post_fix_num_actual_tokens": 0, "post_fix_state_preserved": True}


def _case_sfa_mtp_draft_seq_lens() -> dict:
    """Prove pre-fix draft metadata aliases one buffer and stomps earlier draft views."""
    torch = _torch()
    from mindie_debugger.precision import MemoryWatchpoint, StorageGraph

    class _DraftMeta:
        def __init__(self, cum_query_lens, seq_lens):
            self.cum_query_lens, self.seq_lens = cum_query_lens, seq_lens

    class _CommonMeta:
        def __init__(self, query_start_loc, seq_lens):
            self.query_start_loc, self.seq_lens = query_start_loc, seq_lens
            self.num_reqs = int(seq_lens.numel())

    class AscendSFAMetadataBuilder:
        def __init__(self, fixed, max_reqs=8, spec_tokens=3):
            self.fixed = bool(fixed)

            def zero():
                return torch.zeros(max_reqs + 1, dtype=torch.int32)

            self.actual_seq_lengths_query, self.actual_seq_lengths_key = zero(), zero()
            self.spec_actual_seq_lengths_query = (
                [zero() for _ in range(spec_tokens)] if fixed else None
            )
            self.spec_actual_seq_lengths_key = (
                [zero() for _ in range(spec_tokens)] if fixed else None
            )

        def build_for_drafting(self, common_attn_metadata, draft_index):
            index = draft_index - 1
            query = (
                self.spec_actual_seq_lengths_query[index]
                if self.fixed else self.actual_seq_lengths_query
            )
            key = (
                self.spec_actual_seq_lengths_key[index]
                if self.fixed else self.actual_seq_lengths_key
            )
            num_reqs = common_attn_metadata.num_reqs
            query.zero_()
            query[:num_reqs].copy_(common_attn_metadata.query_start_loc[1 : num_reqs + 1])
            key.zero_()
            key[:num_reqs].copy_(common_attn_metadata.seq_lens[:num_reqs])
            return _DraftMeta(query[:num_reqs], key[:num_reqs])

    def meta(query, seq):
        return _CommonMeta(torch.tensor(query, dtype=torch.int32),
                           torch.tensor(seq, dtype=torch.int32))

    meta0, meta1, meta2 = (
        meta([0, 1, 2], [11, 22]),
        meta([0, 2, 4], [33, 44]),
        meta([0, 3, 6], [55, 66]),
    )
    prefix_builder = AscendSFAMetadataBuilder(fixed=False)
    draft_0 = prefix_builder.build_for_drafting(meta0, draft_index=1)
    draft_1 = prefix_builder.build_for_drafting(meta1, draft_index=2)
    graph = StorageGraph.from_state(meta0=draft_0, meta1=draft_1)
    pre_shares = graph.shares_storage(draft_0.cum_query_lens, draft_1.cum_query_lens)
    assert pre_shares is True
    watchpoint = MemoryWatchpoint(must_preserve={
        "cum_query_lens": draft_0.cum_query_lens, "seq_lens": draft_0.seq_lens,
    })
    prefix_builder.build_for_drafting(meta2, draft_index=2)
    check = watchpoint.check(step=1)
    pre_violations = watchpoint.violation_count
    assert check["changed"] is True and pre_violations >= 2
    assert check["offending_module"] is None

    fixed_builder = AscendSFAMetadataBuilder(fixed=True)
    fixed_0 = fixed_builder.build_for_drafting(meta0, draft_index=1)
    fixed_1 = fixed_builder.build_for_drafting(meta1, draft_index=2)
    fixed_graph = StorageGraph.from_state(meta0=fixed_0, meta1=fixed_1)
    post_shares = fixed_graph.shares_storage(
        fixed_0.cum_query_lens, fixed_1.cum_query_lens
    )
    assert post_shares is False
    clean_watchpoint = MemoryWatchpoint(must_preserve={
        "cum_query_lens": fixed_0.cum_query_lens, "seq_lens": fixed_0.seq_lens,
    })
    fixed_builder.build_for_drafting(meta2, draft_index=2)
    clean_check = clean_watchpoint.check(step=1)
    assert clean_check["changed"] is False
    assert clean_watchpoint.violation_count == 0
    return {"pre_fix_shares_storage": pre_shares, "post_fix_shares_storage": post_shares,
            "pre_fix_violations": pre_violations,
            "post_fix_violations": clean_watchpoint.violation_count,
            "pre_fix_stomp_detected": bool(check["changed"]),
            "post_fix_stomp_detected": bool(clean_check["changed"])}


def _case_kda_varlen_padded_workspace() -> dict:
    """Prove the KDA varlen prefix workspace stomps u_out at chunk 74 and latches NaN."""
    torch = _torch()
    from mindie_debugger.precision import MemoryWatchpoint, WatchRingBuffer

    def chunk_rows(lengths):
        return [min(64, length - start) for length in lengths
                for start in range(0, length, 64)]

    def run(lengths, padded_workspace):
        total_tokens, rows = sum(lengths), chunk_rows(lengths)
        num_chunks, padded_tokens = len(rows), len(rows) * 64
        offset = padded_tokens if padded_workspace else total_tokens
        workspace = torch.zeros(offset + total_tokens, dtype=torch.float32)
        u_out_view = workspace[offset : offset + total_tokens]
        u_out_view.copy_(torch.arange(1, total_tokens + 1, dtype=torch.float32))
        watchpoint = MemoryWatchpoint(must_preserve={"u_out_scratch": u_out_view})
        ring = WatchRingBuffer(capacity=num_chunks + 1, max_snapshot_numel=total_tokens,
                               device="cpu")
        first_stomp, output = None, None
        for chunk_idx, valid_rows in enumerate(rows):
            start = chunk_idx * 64
            workspace[start : start + valid_rows] = -1.0
            output = torch.sqrt(u_out_view)
            ring.advance_step()
            ring.watch(output)
            changed = watchpoint.check(step=chunk_idx)["changed"]
            if first_stomp is None and changed:
                first_stomp = chunk_idx
        return {
            "total_tokens": total_tokens, "num_chunks": num_chunks,
            "padded_tokens": padded_tokens, "first_stomp": first_stomp,
            "violations": watchpoint.violation_count, "latch": ring.read_latch(),
            "records": ring.read_records()["records"],
            "output_finite": bool(torch.isfinite(output).all()),
        }

    lengths = ([16, 32, 64] * 43)[:127]
    prefix = run(lengths, padded_workspace=False)
    records = {item["epoch"]: item for item in prefix["records"]}
    assert len(lengths) == 127 and sum(lengths) == 4720
    assert (prefix["total_tokens"], prefix["num_chunks"], prefix["padded_tokens"]) == (
        4720, 127, 8128)
    assert prefix["first_stomp"] == 74 and prefix["violations"] >= 1
    assert prefix["latch"]["latch_triggered"] is True
    assert prefix["latch"]["epoch"] == 74 and records[74]["nan_count"] > 0
    assert prefix["output_finite"] is False

    uniform = run([64] * 74, padded_workspace=False)
    assert uniform["first_stomp"] is None and uniform["violations"] == 0
    assert uniform["latch"]["latch_triggered"] is False and uniform["output_finite"]

    postfix = run(lengths, padded_workspace=True)
    assert postfix["first_stomp"] is None and postfix["violations"] == 0
    assert postfix["latch"]["latch_triggered"] is False and postfix["output_finite"]
    return {
        "total_tokens": prefix["total_tokens"], "num_chunks": prefix["num_chunks"],
        "padded_tokens": prefix["padded_tokens"],
        "pre_fix_first_stomp_chunk": prefix["first_stomp"],
        "pre_fix_violations": prefix["violations"],
        "pre_fix_latch_triggered": prefix["latch"]["latch_triggered"],
        "pre_fix_latch_epoch": prefix["latch"]["epoch"],
        "pre_fix_nan_count_at_latch": records[74]["nan_count"],
        "uniform_control_first_stomp": uniform["first_stomp"],
        "uniform_control_violations": uniform["violations"],
        "uniform_control_latch_triggered": uniform["latch"]["latch_triggered"],
        "post_fix_violations": postfix["violations"],
        "post_fix_latch_triggered": postfix["latch"]["latch_triggered"],
        "post_fix_output_finite": postfix["output_finite"],
    }


def _case_spatial_group_norm_bf16_truncation() -> dict:
    """Prove BF16 partial accumulation is a real bug and SparseMLA -1 scatter latches."""
    torch = _torch()
    from mindie_debugger.precision import WatchRingBuffer, capture_call, replay_call

    def prefix(x, weight, bias, num_groups=8, eps=1e-5):
        n_batch, channels, height, width = x.shape
        per_group = channels // num_groups
        reshaped = x.view(n_batch, num_groups, per_group, height, width)
        partial_sum = torch.zeros((n_batch, num_groups), dtype=torch.bfloat16)
        partial_sq = torch.zeros((n_batch, num_groups), dtype=torch.bfloat16)
        for channel in range(per_group):
            for row in range(height):
                for start in range(0, width, 512):
                    tile = reshaped[:, :, channel, row, start:start + 512].bfloat16()
                    partial_sum = (partial_sum + tile.sum(-1)).bfloat16()
                    partial_sq = (partial_sq + (tile * tile).sum(-1)).bfloat16()
        elements = float(per_group * height * width)
        mean = partial_sum.float() / elements
        var = torch.clamp(partial_sq.float() / elements - mean * mean, min=0.0)
        grouped = x.view(n_batch, num_groups, -1).float()
        normed = (grouped - mean.unsqueeze(-1)) * torch.rsqrt(var.unsqueeze(-1) + eps)
        staging = normed.view(n_batch, channels, height, width).clone()
        scaled = staging * weight.float().view(1, -1, 1, 1)
        return (scaled + bias.float().view(1, -1, 1, 1)).to(x.dtype)

    def fixed(x, weight, bias, num_groups=8, eps=1e-5):
        n_batch, channels, height, width = x.shape
        grouped = x.float().view(n_batch, num_groups, -1)
        mean = grouped.mean(dim=-1, keepdim=True)
        var = grouped.var(dim=-1, unbiased=False, keepdim=True)
        normed = (grouped - mean) * torch.rsqrt(var + eps)
        out = normed.view(n_batch, channels, height, width)
        out = out * weight.float().view(1, -1, 1, 1) + bias.float().view(1, -1, 1, 1)
        return out.to(x.dtype)

    generator = torch.Generator().manual_seed(91099)
    x = (torch.randn((2, 16, 16, 16), generator=generator) * 1.8 + 0.7).bfloat16()
    weight = (torch.randn(16, generator=generator) * 0.5 + 1.0).bfloat16()
    bias = (torch.randn(16, generator=generator) * 0.2).bfloat16()
    prefix_replay = replay_call(
        capture_call(prefix, x, weight, bias, num_groups=4, eps=1e-5), prefix,
        mode="reference_check", reference_atol=_ATOL,
    )
    fixed_replay = replay_call(
        capture_call(fixed, x, weight, bias, num_groups=4, eps=1e-5), fixed,
        mode="reference_check", reference_atol=_ATOL,
    )
    assert prefix_replay["verdict"] == "real_numerical_bug"
    assert prefix_replay["reference_check"]["max_abs_err"] > _ATOL
    assert fixed_replay["verdict"] == "pass"
    assert fixed_replay["reference_check"]["max_abs_err"] <= _ATOL

    def sparse_mla_decode_step(block_table, topk_indices, seqused_k, guard):
        work_block_table = block_table.clone()
        slots = torch.arange(topk_indices.shape[1]).expand_as(topk_indices)
        if guard:
            safe_idx = torch.where(topk_indices < 0, 0, topk_indices)
            valid = (topk_indices >= 0).sum(-1).int()
            max_seqlen = 128
        else:
            safe_idx, valid = topk_indices, seqused_k.int()
            max_seqlen = int(seqused_k.max().item())
        work_block_table.scatter_(1, slots, safe_idx)
        return {"work_block_table": work_block_table, "seqused_k": valid,
                "max_seqlen_k": max_seqlen}

    seqused_k = torch.tensor([32, 12, 32, 16], dtype=torch.int64)
    slots = torch.arange(8).expand(4, 8).clone()
    topk_indices = torch.where(
        slots < ((seqused_k + 3) // 4).unsqueeze(1), slots, torch.full_like(slots, -1)
    )
    block_table = torch.arange(32, dtype=torch.int64).reshape(4, 8)
    blocked = sparse_mla_decode_step(block_table, topk_indices, seqused_k, guard=False)
    guarded = sparse_mla_decode_step(block_table, topk_indices, seqused_k, guard=True)
    blocked_negative = int((blocked["work_block_table"] < 0).sum())
    guarded_negative = int((guarded["work_block_table"] < 0).sum())
    assert blocked_negative > 0 and guarded_negative == 0
    assert blocked["max_seqlen_k"] == int(seqused_k.max()) == 32
    assert guarded["max_seqlen_k"] == 128

    ring = WatchRingBuffer(capacity=2, max_snapshot_numel=32, device="cpu")
    ring.advance_step()
    ring.watch((blocked["work_block_table"] < 0).float(), latch_threshold=0.5)
    blocked_latch = ring.read_latch()
    assert blocked_latch["latch_triggered"] is True
    clean_ring = WatchRingBuffer(capacity=2, max_snapshot_numel=32, device="cpu")
    clean_ring.advance_step()
    clean_ring.watch((guarded["work_block_table"] < 0).float(), latch_threshold=0.5)
    guarded_latch = clean_ring.read_latch()
    assert guarded_latch["latch_triggered"] is False
    return {"pre_fix_verdict": prefix_replay["verdict"],
            "pre_fix_max_abs_err": _rounded(prefix_replay["reference_check"]["max_abs_err"]),
            "post_fix_verdict": fixed_replay["verdict"],
            "post_fix_max_abs_err": _rounded(fixed_replay["reference_check"]["max_abs_err"]),
            "unguarded_negative_count": blocked_negative,
            "guarded_negative_count": guarded_negative,
            "unguarded_latch_triggered": blocked_latch["latch_triggered"],
            "guarded_latch_triggered": guarded_latch["latch_triggered"],
            "unguarded_max_seqlen_k": blocked["max_seqlen_k"],
            "guarded_max_seqlen_k": guarded["max_seqlen_k"]}


def _case_moe_healed_lockstep_and_w4a8_gate() -> dict:
    """Exercise slim MoE healing, W4A8 fake-quant triage and sampler drift."""
    torch = _torch()
    from mindie_debugger.precision import build_deterministic_stimulus

    stimulus = build_deterministic_stimulus(
        vocab_size=64, batch_size=2, seq_len=8, seed=7
    )
    cases = _w4a8_and_sampler(torch)
    with tempfile.TemporaryDirectory(prefix="mindie-precision-moe-") as temporary:
        cases.update(_slim_moe_lockstep(torch, Path(temporary), stimulus))
    return cases


def _run_all_cases() -> dict:
    """Run the five deterministic localization cases on CPU."""
    _torch()
    return {
        "k3_gdn_idle_dp_state": _case_k3_gdn_idle_dp_state(),
        "sfa_mtp_draft_seq_lens": _case_sfa_mtp_draft_seq_lens(),
        "kda_varlen_padded_workspace": _case_kda_varlen_padded_workspace(),
        "spatial_group_norm_bf16_truncation": (
            _case_spatial_group_norm_bf16_truncation()
        ),
        "moe_healed_lockstep_and_w4a8_gate": (
            _case_moe_healed_lockstep_and_w4a8_gate()
        ),
    }


def precision_localization_evidence(root: Path) -> dict:
    """Replay all five real defects twice and persist the deterministic proof JSON."""
    first = _run_all_cases()
    second = _run_all_cases()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first) == 5

    k3 = first["k3_gdn_idle_dp_state"]
    sfa = first["sfa_mtp_draft_seq_lens"]
    kda = first["kda_varlen_padded_workspace"]
    norm = first["spatial_group_norm_bf16_truncation"]
    moe = first["moe_healed_lockstep_and_w4a8_gate"]

    k3_caught = (
        k3["pre_fix_violations"] >= 2
        and k3["pre_fix_offending_module"].endswith("layers.0.self_attn")
        and k3["post_fix_violations"] == 0
    )
    sfa_caught = (
        sfa["pre_fix_shares_storage"] is True
        and sfa["post_fix_shares_storage"] is False
        and sfa["pre_fix_violations"] >= 2
        and sfa["post_fix_violations"] == 0
    )
    kda_caught = (
        kda["pre_fix_first_stomp_chunk"] == 74
        and kda["pre_fix_violations"] >= 1
        and kda["pre_fix_latch_triggered"] is True
        and kda["pre_fix_latch_epoch"] == 74
        and kda["uniform_control_violations"] == 0
        and kda["uniform_control_latch_triggered"] is False
        and kda["post_fix_violations"] == 0
        and kda["post_fix_latch_triggered"] is False
    )
    norm_caught = (
        norm["pre_fix_verdict"] == "real_numerical_bug"
        and norm["pre_fix_max_abs_err"] > _ATOL
        and norm["post_fix_verdict"] == "pass"
        and norm["post_fix_max_abs_err"] <= _ATOL
        and norm["unguarded_negative_count"] > 0
        and norm["guarded_negative_count"] == 0
        and norm["unguarded_latch_triggered"] is True
        and norm["guarded_latch_triggered"] is False
    )
    moe_caught = (
        moe["first_diverged_module"] == _MOE_BUG_MODULE
        and moe["naive_first_diverged_module"] != _MOE_BUG_MODULE
        and moe["healed_downstream_diverged"] == 0
        and moe["naive_downstream_diverged"] > 0
        and moe["healthy_verdict"] == "inherent_quantization_error"
        and moe["scale_bug_verdict"] == "real_numerical_bug"
        and moe["oversized_topk_rejected"] is True
        and moe["softmax_prefix_verdict"] == "real_numerical_bug"
        and moe["softmax_fixed_verdict"] == "pass"
    )
    caught = [k3_caught, sfa_caught, kda_caught, norm_caught, moe_caught]
    assert all(caught)
    assert moe["first_diverged_module"] == _MOE_BUG_MODULE
    post_fix_false_positives = (
        int(k3["post_fix_violations"] > 0)
        + int(sfa["post_fix_violations"] > 0)
        + int(kda["post_fix_violations"] > 0)
        + int(norm["post_fix_verdict"] != "pass")
        + int(moe["healed_downstream_diverged"] > 0)
    )
    assert post_fix_false_positives == 0

    summary = {
        "status": "pass",
        "deterministic": True,
        "cases_evaluated": len(first),
        "pre_fix_bugs_caught": sum(caught),
        "post_fix_false_positives": post_fix_false_positives,
        "k3_idle_dp_caught": k3_caught,
        "k3_offending_module": k3["pre_fix_offending_module"],
        "sfa_mtp_alias_caught": sfa_caught,
        "kda_workspace_caught": kda_caught,
        "kda_first_stomp_chunk": kda["pre_fix_first_stomp_chunk"],
        "norm_truncation_caught": norm_caught,
        "moe_healed_localized": moe["first_diverged_module"] == _MOE_BUG_MODULE,
        "moe_first_diverged_module": moe["first_diverged_module"],
    }
    payload = {**summary, "cases": first}
    evidence_path = root / "precision-localization.json"
    write_json(evidence_path, payload)
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == payload
    return {
        **summary,
        "evidence_path": str(evidence_path),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "cases": first,
    }
