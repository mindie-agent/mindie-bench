---
category: axiom
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
evidence_strength: hardware_spec
id: unit-vllm-ascend-npu-runtime-and-graph-invariants
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/d8c9e3a673af918c94a7794ef6f1f69c024d9c62/vllm_ascend/worker/model_runner_v1.py
  source_digest: 61ee8bc56d4ac0e1a7f43d1804bfe41c1a6699181774938907c97174f2348343
  source_locator: vllm-project/vllm-ascend@d8c9e3a673af918c94a7794ef6f1f69c024d9c62:vllm_ascend/worker/model_runner_v1.py
  source_type: runtime_architecture_specification
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L1
title: Ascend NPU graph-capture and custom-kernel binding premises
triggers:
  error_signatures:
  - ACLGraph replay reads stale buffers or diverges from the eager result
  - torch.ops._C_ascend is missing or unbound at the first custom-op call
  - an NZ-fractal weight is read as ND or a non-contiguous view is indexed
  keywords:
  - vllm-ascend
  - ACLGraph
  - graph capture
  - stream
  - _C_ascend
  - custom op
  - NZ
  - contiguous
  - weight_nz_mode
  - Ascend910_93
---

# Ascend NPU graph-capture and custom-kernel binding premises

## Condition

Ascend 910B/910_93 with `vllm-ascend`, for ACLGraph-decoded steps or custom ops consuming weights/activations. Model, dtype, graph and stack are task facts.

## Constraint

1. **Stream/capture:** capture and replay share one device, stream and persistent addresses. The captured region allocates nothing, switches no stream, does not sync the default stream, and keeps shapes fixed; an address change or host sync replays stale data.
2. **Operator binding:** kernels exist via `torch.ops._C_ascend`, bound by the extension loaded at init. Meta registration is not a kernel. Check the binding before capture; a missing symbol is a runtime precondition failure.
3. **Layout/contiguity:** an NZ weight is not ND and must be consumed by an NZ op; never reinterpret NZ with an ND view. Selection is gated by dtype, device and `weight_nz_mode` (fp32/meta excluded). A flat token prefix must be contiguous.

## Rationale

Replay skips the Python launch path, so a stream, address or allocation mismatch is not repaired then. Layout and contiguity are value contracts; a wrong interpretation looks like a precision fault.

## Evidence

Pinned `model_runner_v1.py`; custom-op gate details are in `unit-vllm-ascend-custom-op-gates-import-retry-and-nz`. This axiom states preconditions only; parity comes from the measurement gate.
