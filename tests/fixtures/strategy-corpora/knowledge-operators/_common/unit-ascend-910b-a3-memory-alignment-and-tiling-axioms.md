---
schema: mindie-graded-unit/1
id: unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
revision: 2
tier: L1
category: axiom
domain_path: _common
evidence_strength: hardware_spec
title: 'Ascend 910B/910_93 memory alignment and reduction axioms'
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
  source_type: hardware_architecture_specification
  source_locator: docs/evidence/c4-case-block-v2-gates.json
  evidence_refs:
  - docs/evidence/c4-case-block-v2-gates.json
  supersedes:
  - mindie://operators/units/unit-ascend-910b-a3-memory-alignment-and-tiling-axioms@1
triggers:
  error_signatures:
  - misaligned UB DMA or L1 tile causes wrong results or a runtime fault
  - bf16 or fp16 RMSNorm/LayerNorm/Softmax loses accuracy because the reduction is not fp32
  - a strided gate view silently takes the contiguous fast path
  keywords:
  - Ascend910B
  - Ascend910_93
  - alignment
  - UB
  - L1
  - fp32 accumulation
  - RMSNorm
  - LayerNorm
  - Softmax
  - contiguous
  - strided
---
# Ascend 910B/910_93 alignment and reduction axioms

**Condition.** Ascend 910B (A2) or 910_93 (A3 dual-die), bf16/fp16/fp32. A violating kernel is out of scope even if one output looks correct.

**Constraint.**
- **UB/GM DMA: 32 B.** Base addresses and byte counts are 32 B aligned; pad the tail and mask it so padding is never data. Copy a misaligned view to an aligned buffer before DMA.
- **L1: 512 B.** Cube-tile bases and descriptor strides are 512 B aligned; re-tile, pad or stage a misaligned tile through UB. A misaligned L1 descriptor is undefined.
- **fp32 reductions.** RMSNorm, LayerNorm and Softmax accumulate sum and sum-of-squares in fp32, apply epsilon in fp32, and cast once to `out_dtype` at the store. A low-precision accumulator changes the numerical contract; an unrequested fp32 output is dtype drift.
- **Contiguous last-dim dispatch.** The contiguous path needs last-dimension stride 1 and 32 B UB / 512 B L1 compliant bases and strides. A non-unit stride or misaligned offset takes the strided kernel or explicit contiguous materialization; record the path.

**Rationale.** Misaligned DMA or L1 descriptors read undefined data; low-precision accumulation can shift the mean or square mean. Shape equality does not prove the contiguous native path ran.

**Evidence.** Hardware specification recorded in `docs/evidence/c4-case-block-v2-gates.json`.
