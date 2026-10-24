---
schema: mindie-graded-unit/1
id: unit-ascend-operator-roofline-and-four-view-measurement
revision: 2
tier: L1
category: measurement
domain_path: _common
evidence_strength: reproduced_empirical
title: 'Ascend operator timing and roofline measurement baseline'
conditions:
  soc:
  - Ascend910B
  - Ascend910B1
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  - float32
inherits_from: []
provenance:
  source_type: measurement_contract
  source_locator: docs/evidence/c4-case-block-v2-gates.json
  evidence_refs:
  - docs/evidence/c4-case-block-v2-gates.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  supersedes:
  - mindie://operators/units/unit-ascend-operator-roofline-and-four-view-measurement@1
triggers:
  error_signatures:
  - a speedup is claimed without synchronized before/after timing or without warmup exclusion
  - one pooled IQR is used across different shapes while per-case block behavior is ignored
  - a correctness prerequisite is counted as a measured performance factor
  - kernel duration is reported as end-to-end operator latency
  keywords:
  - Ascend operator
  - measurement
  - OracleFreezer
  - torch.npu.synchronize
  - AblationGate
  - four-view
  - roofline
  - msprof
  - aicore_time
  - ub_to_gm
  - l2_cache_hit_rate
---
# Ascend operator timing and roofline measurement baseline

**Condition.** Any latency, speedup or bottleneck claim for an operator on Ascend 910B/910_93 under the declared shape, dtype and clock.

**Constraint.**
- **Reference accuracy first (`OracleFreezer`).** Freeze the fp32 reference and every input; require tolerance, exact `out_dtype` and unchanged inputs. Require two baseline runs to agree inside the recorded noise envelope before comparing a candidate.
- **Synchronized timing window.** Exclude two or more warmup rounds, then call `torch.npu.synchronize()` immediately before and after each timed block. Keep three or more blocks per case and twelve samples per block, with case identity on every sample; report per-case means and envelope.
- **Multi-view roofline attribution.** Attribute kernel time via msprof `aicore_time`, memory traffic via `ub_to_gm` and HBM counters, and cache behavior via `l2_cache_hit_rate`. Report kernel, host, memory/roofline, and end-to-end latency on the same workload and clock.
- **No cross-shape pooling.** Do not pool heterogeneous cases into one IQR, call the envelope a confidence interval, or report device kernel duration as end-to-end operator latency.

**Rationale.** Unmatched or unsynchronized runs mix queueing, cache state and clock noise; per-view attribution assigns a speedup to its factor.

**Evidence.** Contract in `docs/evidence/c4-case-block-v2-gates.json` and `docs/evidence/c4-case-block-v2-stage-acceptance.json`.
