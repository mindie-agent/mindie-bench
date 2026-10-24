---
category: measurement
conditions:
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
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-verification-and-state-parity-gates
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/tests/ut/worker/test_model_runner_v1.py
  source_digest: 11e7ef8d243dcca647e6ed35e524a5f261606182e5a6deb39fa9c1dbd8acb6a3
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:tests/ut/worker/test_model_runner_v1.py
  source_type: verification_contract
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L1
title: Verification gates for eager/graph parity and recurrent-state invariance
triggers:
  error_signatures:
  - eager logits match but ACLGraph replay diverges or reads stale buffers
  - unit test passes with mocked metadata but no NPU state checksum was taken
  - prefill logits match while conv_state or ssm_state is wrong
  - idle-rank state changes after a zero-token dummy step
  keywords:
  - vllm-ascend
  - verification
  - eager
  - ACLGraph
  - numerical parity
  - state invariance
  - conv_state
  - ssm_state
  - checksum
  - fp32 reference
  - DP
  - HCCL
---

# Verification gates: eager/graph parity and state invariance

## Condition

A change to `vllm-ascend` step metadata, GDN state, graph capture or kernels on Ascend 910B/910_93. Freeze the stack, topology, graph mode, dtype/cache and tolerance first.

## Constraint

Ordered gates; a later gate is never implied by an earlier one.

1. **Metadata (no NPU):** the builder emits the intended GDN start-loc, token count, query length and prefilling flags; an idle zero-token step enters the DP collective.
2. **Eager parity (1 rank):** eager prefill/generation match CPU `fp32` on hidden states, logits and per-row `conv_state`/`ssm_state`; final logits alone fail.
3. **Graph parity:** ACLGraph replay of the same inputs must match eager logits and state rows at the same tolerance, deterministically.
4. **Idle invariance:** row-resolved `conv_state`/`ssm_state` snapshots stay unchanged across a zero-token dummy; aggregate cache hashes fail.
5. **Multi-rank:** all ranks enter/leave the DP/HCCL collective and exit 0; the pre-fix control fails the idle-state assertion.

## Rationale

Graph replay can read stale buffers while eager is correct; a correct logit can hide a corrupted recurrent row. Parity is per row.

## Evidence

Pinned `tests/ut/worker/test_model_runner_v1.py` and the DP2 TP1 receipt. Failed runs cannot establish `reproduced_empirical`; metadata-only tests are not gates.
