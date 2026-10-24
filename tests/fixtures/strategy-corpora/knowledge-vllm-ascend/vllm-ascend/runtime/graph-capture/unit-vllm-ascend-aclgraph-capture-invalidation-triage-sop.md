---
category: sop
conditions:
  graph_mode:
  - aclgraph
  - piecewise
  - full_decode_only
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
domain_path: vllm-ascend/runtime/graph-capture
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: runtime_triage_sop
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L2
title: ACLGraph capture invalidation, dynamic shape bucketing, and stream sync triage SOP
triggers:
  error_signatures:
  - ACLGraph replay falls back to eager execution on decode steps
  - RuntimeError during npu_graph capture due to host-device synchronization or dynamic allocation
  - decode output diverges after batch size changes across padded graph buckets
  - eager execution passes on the same decode input while aclgraph or piecewise replay diverges
  keywords:
  - ACLGraph
  - graph capture
  - invalidation
  - padded batch
  - bucket
  - eager fallback
  - stream synchronization
  - persistent buffer
  - model_runner_v1
  - stale tail padding
---

# ACLGraph capture invalidation, dynamic shape bucketing, and stream sync triage SOP

## 1. Applicable Problem & When to Select

Select this SOP when `vllm-ascend` decode steps under `aclgraph`, `piecewise`, or `full_decode_only` mode exhibit graph-specific failure while the underlying operator math is valid in eager mode:
- **Observable Triggers**:
  - Eager execution (`--enforce-eager`) produces correct reference outputs on the identical decode step, whereas `aclgraph` / `piecewise` replay diverges numerically or corrupts subsequent steps;
  - `npu_graph` capture raises `RuntimeError` due to host scalar extraction (`.item()`, `.cpu()`) or stream switching inside custom operators;
  - Decode steps unexpectedly fall back to eager execution or repeatedly re-capture across padded batch buckets (`B_large -> B_small`).
- **Missing Facts & Minimal Probe (`condition_unknown`)**:
  - If `graph_mode` or eager baseline parity is not yet recorded, run a single-step comparison on the same input with `--enforce-eager` vs the active `aclgraph` configuration and log `num_scheduled_tokens`, `padded_num_tokens`, and `uniform_decode` before adopting this SOP.
- **When NOT to Adopt (Switch / Default Boundary)**:
  - Do **not** adopt if `--enforce-eager` already fails or diverges on the same single step (route to single-operator numerical or dtype overflow triage instead).
  - Do **not** adopt if multi-rank execution hangs in HCCL collective initialization before entering `model_runner_v1` graph capture (route to HCCL / DP collective triage).

## 2. Next Discriminating Action & Branch Routing

1. **Step 1 — Verify Bucket Coverage vs Scheduled Token Count**
   - Record `num_scheduled_tokens`, `num_reqs`, and `padded_num_tokens` at the step boundary in `model_runner_v1.py`.
   - Confirm whether `uniform_decode` holds (`max_query_len == 1` across active requests) and whether `padded_num_tokens` matches a pre-captured graph key.
   - **Branch A (Mixed prefill-decode chunk enters `full_decode_only` replay)**: Bypass full-graph replay for non-uniform batches rather than writing variable-length tokens into fixed-shape capture buffers.

2. **Step 2 — Isolate Host Synchronization & Dynamic Buffer Allocation Inside Capture**
   - Run one warm capture step with `ASCEND_LAUNCH_BLOCKING=0` while tracing host-device sync calls (`torch.npu.synchronize()`, `.item()`, `.to("cpu")`, or dynamic `torch.empty` allocations inside the captured region).
   - **Branch B (Host scalar or fresh tensor allocation inside captured region)**: Move scalar precomputation to host metadata preparation and copy all input metadata tensors (`positions`, `slot_mapping`, `block_table`, `seq_lens`, `query_start_loc`) into persistent static device buffers via in-place `.copy_()` on the capture stream before replay.

3. **Step 3 — Check Stale Tail Padding Across Padded Buckets (`B_large -> B_small`)**
   - Run a two-step bucket transition probe: Step A at `B_large` followed immediately by Step B at `B_small < B_large` reusing the same persistent static buffer.
   - **Branch C (Step B diverges only after Step A)**: Padded tail slots `[num_actual_tokens : padded_num_tokens]` retained stale metadata from Step A. Enforce `slot_mapping[num_actual_tokens:padded_num_tokens] = -1` and zero out padded `seq_lens` / `query_start_loc` deltas before replay.

## 3. Verification, Exit Criteria & Scoped Counter-Evidence

- **Verification & Exit**: Exit with `verified` only when (a) eager and `aclgraph` replay match within pre-registered tolerance across `B_large -> B_small -> B_large` transitions, (b) zero unplanned graph re-captures occur during steady-state decode, and (c) uninstrumented end-to-end decode throughput does not regress.
- **Scoped Counter-Evidence**: If resetting padded tail metadata (`slot_mapping = -1`, `seq_lens = 0`) and confirming in-place `.copy_()` on persistent buffers does not eliminate the `B_large -> B_small` divergence under `Ascend910B` / `Ascend910_93`, record a conditional counter-evidence finding against graph metadata buffering and switch to operator-level internal workspace / recurrent state life-cycle inspection.
