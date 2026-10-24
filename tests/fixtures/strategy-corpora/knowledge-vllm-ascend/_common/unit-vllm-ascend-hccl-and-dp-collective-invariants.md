---
category: axiom
conditions:
  dp: '>1'
  idle_rank_state: no_scheduled_requests
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
domain_path: _common
evidence_strength: hardware_spec
id: unit-vllm-ascend-hccl-and-dp-collective-invariants
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  - https://github.com/vllm-project/vllm-ascend/pull/15655
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: runtime_architecture_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: HCCL and DP collective invariants for idle ranks
triggers:
  error_signatures:
  - idle DP rank hangs because a zero-token step skipped the DP collective
  - active ranks block in coordinate_batch_across_dp while one rank returns early
  - GDN conv_state or ssm_state changes after a zero-token dummy forward
  - a two-rank DP run hangs in HCCL before the idle-rank parity assertion
  keywords:
  - vllm-ascend
  - HCCL
  - DP
  - idle rank
  - dummy run
  - coordinate_batch_across_dp
  - collective
  - skip_gdn_state_update
  - conv_state
  - ssm_state
  - Ascend910_93
---

# HCCL and DP collective invariants for idle ranks

## Condition

`vllm-ascend` on Ascend 910B/910_93, `data_parallel_size > 1`, `external_launcher`, one idle DP rank. HCCL rank order and world size are fixed before the first collective.

## Constraint

1. **Rank symmetry:** all DP ranks enter the same collectives, in the same order and world size; a rank that skips one leaves peers blocked until timeout.
2. **Idle dummy:** zero-token `execute_model` calls `_dummy_run(1, skip_gdn_state_update=True)` so the idle rank joins `coordinate_batch_across_dp`; the dummy keeps graph/collective shape, with no request token.
3. **Zero-effect masking:** idle GDN metadata has `num_actual_tokens=0`, zero `gdn_query_start_loc`, `max_query_len=0` and all-false `is_prefilling`; non-GDN keeps the one-token dummy. Masked ops must not write `conv_state`/`ssm_state`.
4. **No reuse:** never pass the skip flag on a scheduled batch or reuse idle metadata for real decode.

## Rationale

DP collectives are all-rank barriers: the idle rank must participate or peers deadlock. Presence proves participation, not that masking left state unchanged.

## Evidence

Pinned `model_runner_v1.py` and the DP2 TP1 receipt. Covers participation and masking only.
