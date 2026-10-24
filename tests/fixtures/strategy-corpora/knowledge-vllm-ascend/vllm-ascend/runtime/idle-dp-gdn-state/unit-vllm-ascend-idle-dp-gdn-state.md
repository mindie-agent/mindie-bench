---
category: sop
conditions:
  affected_state:
    exact_set:
    - conv_state
    - ssm_state
  dp: 2
  idle_rank_state: no_scheduled_requests
  model:
  - Kimi K3 hybrid Gated DeltaNet/GDN plus MLA attention with MoE
  - Kimi K3 hybrid GDN/MLA + MoE
  - Kimi K3
  npu_count: 2
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  peer_rank_state:
  - active_requests
  - active
  repository:
  - vllm-ascend-idle-dp
  - vllm-project/vllm-ascend
  - vllm-ascend
domain_path: vllm-ascend/runtime/idle-dp-gdn-state
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-idle-dp-gdn-state
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-verification-and-state-parity-gates
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/pull/15655
  - https://github.com/vllm-project/vllm-ascend/pull/15510
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/tests/ut/worker/test_model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: https://github.com/vllm-project/vllm-ascend/pull/15655
  source_type: runtime_reproduction_with_source_diff
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L2
title: 'Idle-DP dummy run and cached GDN recurrent state: verified diagnosis'
triggers:
  error_signatures:
  - idle DP rank with no scheduled requests while peer is active
  - cached GDN conv_state or ssm_state corrupted after an idle step
  - later requests on the formerly idle rank diverge
  keywords:
  - idle DP
  - GDN
  - conv_state
  - ssm_state
  - dummy_run
  - skip_gdn_state_update
  - Kimi K3
  - PR15655
  - HCCL
  - state checksum
  - Ascend910_93
---

# Idle-DP dummy run and cached GDN recurrent state

## Symptom and scope

DP=2 Kimi K3 (hybrid GDN/MLA + MoE), 2×Ascend910_93, TP=1. One rank is idle (no scheduled requests) while the peer is active. The idle rank's cached GDN `conv_state`/`ssm_state` is overwritten and later requests on that rank diverge. Required outcome: idle state unchanged, active state updated, unrelated KV blocks unchanged, HCCL still synchronized, and GDN kernels not disabled.

## Procedure

1. **Confirm the branch.** In `execute_model`, zero scheduled tokens, `data_parallel_size > 1` and `external_launcher` must call `_dummy_run(1, skip_gdn_state_update=True)` and reach `coordinate_batch_across_dp`. A missing call or collective is an HCCL participation defect first.
2. **Prove metadata zero-effect (no NPU).** Builder spy: GDN gets zero-filled `gdn_query_start_loc`, `num_actual_tokens=0`, `max_query_len=0` and all-false `is_prefilling`; non-GDN keeps the dummy one-token shape; a real batch gets default `skip_gdn_state_update=False`.
3. **Prove idle invariance (one rank).** Snapshot per-row `conv_state`/`ssm_state` checksums. A zero-token dummy on the pre-fix path must move state by a synthetic one-token delta; the fixed path must leave both unchanged. Then a real decode must advance state and match the `fp32` logits/state.
4. **Prove collective symmetry (two ranks, DP2 TP1).** Bound every collective; all ranks enter and leave `coordinate_batch_across_dp` and exit `0`; idle state unchanged; active state advanced; the pre-fix negative control fails the idle assertion.
5. **Distill only verified results.** A mocked metadata test is Probe 1, not `reproduced_empirical`; distill after the same code state passes steps 3 and 4.

## Receipt (this topology only)

| Path | Idle rank | Active rank | HCCL |
| --- | --- | --- | --- |
| pre-fix | `[1.125, 1.25]` overwritten | `[1.5, 1.75]` | completes |
| post-fix | `[0.125, 0.25]` preserved | `[1.5, 1.75]` | completes |

The original idle state `[0.125, 0.25]` gained exactly one synthetic token unit per element pre-fix.

## Boundaries

The fix is in GDN metadata only; no kernel or state-tensor code changes. Pretrained weights, ACLGraph/full graph, other cache modes, TP>1 and full serving are outside. PR15510 is a rebase, not independent evidence. Re-run the checksum probe after any change to `num_actual_tokens`, `gdn_query_start_loc` or the state indices.
