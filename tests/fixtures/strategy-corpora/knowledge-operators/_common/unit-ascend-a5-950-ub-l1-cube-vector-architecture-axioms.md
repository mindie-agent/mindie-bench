---
schema: mindie-graded-unit/1
id: unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
revision: 1
tier: L1
category: axiom
domain_path: _common
evidence_strength: hardware_spec
title: 'Ascend A5 (950/950DT/950PR) and 910B/910_93 UB, L1, workspace and AIC-AIV synchronization axioms'
conditions:
  soc:
  - Ascend950
  - Ascend950DT
  - Ascend950PR
  - Ascend910B
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  - float32
  - int8
  - fp8_e4m3fn
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  source_type: hardware_architecture_specification
  source_locator: 'CANN AscendC platform and cross-generation migration contracts; A5 bring-up topology and memory observations'
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/hardware/target-ascend950pr.md
  - https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/hardware/target-ascend910c.md
  - https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.md
  - https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md
  supersedes: []
triggers:
  error_signatures:
  - a kernel built with a 256 KB UB assumption runs out of UB or silently writes past the A3 192 KB budget
  - an A5 kernel underfills the device because it hardcodes 20 or 24 A2/A3 Cube cores
  - a pure AIV kernel writes the raw workspace pointer and fails with 507035 / MTE DDR out of range
  - a Mix kernel deadlocks after a CrossCoreSetFlag because the peer uses the wrong mode or never reaches the wait
  - an npu-smi or rank-mapping script treats A5 as two dies and parses Phy-ID 0..15
  keywords:
  - Ascend950
  - Ascend950DT
  - Ascend950PR
  - Ascend910B
  - Ascend910_93
  - A5
  - A3
  - dual-die
  - unified buffer
  - UB
  - L1
  - L0C
  - AIC
  - AIV
  - Cube
  - Vector
  - CrossCoreSetFlag
  - CrossCoreWaitFlag
  - CrossCoreSync
  - SyncAll
  - workspace
  - tiling ABI
  - GetBlockNum
  - GetCoreNumAiv
  - GetCoreNumAic
---

# Ascend A5 (950/950DT/950PR) and 910B/910_93 UB, L1, workspace and AIC-AIV synchronization axioms

**Condition.** A kernel is compiled for Ascend 910B (A2), Ascend 910_93 (A3 dual-die), or Ascend 950 / 950DT / 950PR (A5, arch35 / DAV_3510), with `bfloat16`, `float16`, `float32`, `int8` or `fp8_e4m3fn` data. The claim concerns Unified Buffer (UB) capacity, L1/L0 Cube staging, workspace, the AIC/AIV split, or cross-core flags. A kernel that violates a stated constraint is out of scope even if one output happens to look correct.

**Constraint.**

- **Package and logical-device topology.** A3 is two 910B dies in one package: two physical dies with a combined 40–48 AIC and 80–96 AIV, and `npu-smi` exposes the dies through `Phy-ID` values spanning the package.
  A5 is one logical chip per NPU in `npu-smi 26.2+`: there is no A3-style two-die `Phy-ID 0..15` enumeration to drive rank mapping.
  Ascend950DT and Ascend950PR are the same DAV_3510 silicon with different product roles; the runtime `_NNNN` suffix is a silicon bin, so branch on the queried SOC name rather than the marketing suffix.
  HBM capacity is SKU- and bin-dependent: the Ascend 950DT/PR bring-up SKU in this corpus reports **96 GB HBM**, while the CANN 950PR reference page lists 128 GB for a different 8×16 GB SKU.
  Read the capacity from `npu-smi info -t memory` or the platform API; never derive it from a model name or from an A3 fleet constant.
- **AIC:AIV topology.** Every AI Core is one Cube unit (AIC) plus two Vector units (AIV), so `AIV = 2 × AIC`.
  On the full A5 die this is 32 AIC / 64 AIV; a PG-binned A5 SKU reports 28 AIC / 56 AIV.
  A3 reports 40–48 AIC / 80–96 AIV.
  All core counts are platform facts, not compile-time constants.
  `GetBlockNum()` is the launch extent of the current kernel, not the physical core count; `PlatformAscendC::GetCoreNumAic()` and `GetCoreNumAiv()` are the platform queries.
  Derive `usedCoreNum` from the work and the queried resource, and make the host tiling value equal the kernel launch extent.
- **UB capacity and 32 B DMA.** A2/A3 UB is 192 KB per AIV.
  A5 UB is 256 KB physical, reported as 248 KB by `GetCoreMemSize(UB)` after the 8 KB compiler reserve.
  On A5 the SIMT/regbase path reserves a further 40 KB at the top of UB for SIMT DCache and thread state, so the SIMT-effective budget is about 208 KB.
  Every GM↔UB DMA base and byte count is 32 B aligned.
  A tile byte count that is not a multiple of 32 B must use `DataCopyPad` with an explicit mask or be staged through an aligned buffer; a plain `DataCopy` of a misaligned view is undefined.
- **L1 and L0 Cube alignment.** L1 is 512 KB per AIC on 910B, 910_93 and A5.
  Cube tile bases and L0A/L0B/L0C fractal descriptors are 512 B aligned; a 512 B-misaligned Cube tile descriptor is out of contract even though the generic L1 DMA granule is 32 B.
  L0A and L0B are 64 KB each.
  L0C is 128 KB on A2/A3 and 256 KB on A5.
  A2/A3 must move a Cube result through GM before Vector can consume it; A5 adds an L0C→UB Fixpipe path and a UB→L1 path, so a literal port of the A3 GM round-trip is correct but leaves the A5 fast path unused.
- **A5 memory hierarchy change.** A5 adds a Vector register file between UB and the vector ALU, an independent Scalar unit, and 6 pipelines plus Scalar; the A2/A3 memory-based model moves operands from UB directly.
  A3 has no SIMT path (arch22 is SIMD-only); A5 arch35 supports SIMD, SIMT and mixed SIMT/SIMD.
  A kernel that uses `Simt::` warp intrinsics, `ThreadBarrier`, or `LAUNCH_BOUND` is A5-only and must be rewritten to `TPipe`/`TQue`/`DataCopy` for A3.
- **Cross-core flag contract.** Intra-core ordering uses `PipeBarrier<pipe>()`, `SetFlag`/`WaitFlag` and, on A5, Mutex.
  Cross-core ordering uses `CrossCoreSetFlag<mode, pipe>(flagId)` and `CrossCoreWaitFlag<mode, pipe>(flagId)`; a `CrossCoreSync`-style helper still reduces to that set/wait pair and does not widen the participant set.
  On A2/A3 the supported Mix mode is mode 2: both AIVs of the pair must report before the AIC wait completes.
  A5 adds mode 4, where AIV0 and AIV1 can trigger the AIC wait independently.
  The mode, pipe, `flagId` and participant set must match on both sides; flag identifiers are a finite pool shared with Matmul and `SyncAll`.
  `SyncAll()` is not a universal barrier: its hardware/software form and its AIV-only form cover different participant sets, and it does not repair a missing producer/consumer pair.
- **Workspace contract.** Workspace is 32 B aligned.
  The arch35 system reserve is 16 MB, not the 2 MB of older targets: `GetUserWorkspace()` returns a pointer derived from the KFC workspace base, and a pure AIV kernel without `REGIST_MATMUL_OBJ`/`REGIST_CUBE_OBJ` does not establish that base.
  Writing the raw `workspace` argument in such a kernel can fail with `507035` and plog `errcode:(95) The DDR address of the MTE instruction is out of range`.
  Use the framework workspace accessor that matches the launch contract, or register the Cube object, and keep the workspace size in the tiling record.
- **Tiling struct ABI across 910B / 910_93 / 950.** The serialized `TilingData` is an ABI shared by host and kernel.
  Use fixed-width fields (`uint32_t`, `uint64_t`, `int32_t`), append new fields at the end, keep the existing offsets unchanged, and version the struct.
  Do not serialize `size_t`, raw pointers, or a struct whose padding differs between a 32-bit host build and the device build.
  A per-SOC `#if` may change a default value, but it must not change the field order or width of a shared struct.
- **Hardcoded core counts are an A2/A3 carry-over defect.** `20`, `24`, `40`, `48`, `28`, `32`, `56`, `64`, `80`, `96` are all values that appear on some SKU.
  A kernel that uses one of them as the launch extent either underfills an A5 full die (32/64) or oversubscribes an A3 package.
  Query `GetCoreNumAic()`/`GetCoreNumAiv()` on the host, pass the derived `usedCoreNum` in tiling, and use `GetBlockNum()`/`GetBlockIdx()` in the kernel.
  `GetBlockNum()` can equal 1 in a Mix kernel that still contains one AIC and two AIVs; never infer the AIC/AIV composition from the logical block count alone.

| Resource | 910B (A2) | 910_93 (A3) | 950 / 950DT / 950PR (A5) |
| --- | --- | --- | --- |
| Package | one die per NPU | two 910B dies per package | one logical chip per NPU in `npu-smi 26.2+` |
| AIC : AIV | 1 : 2 | 1 : 2 (40–48 AIC / 80–96 AIV) | 1 : 2 (32 AIC / 64 AIV full die; 28 / 56 on a PG bin) |
| UB per AIV | 192 KB | 192 KB | 256 KB physical, 248 KB reported, about 208 KB SIMT-effective |
| L1 per AIC | 512 KB | 512 KB | 512 KB |
| L0C per AIC | 128 KB | 128 KB | 256 KB |
| HBM | SKU-dependent | SKU-dependent | 96 GB on the observed SKU; SKU-dependent |
| Cross-core Mix mode | mode 2 | mode 2 | mode 2 plus mode 4 (per-AIV trigger) |
| Programming model | SIMD | SIMD only | SIMD + SIMT + mixed |

**Rationale.** These are hardware and ABI facts, not tuning preferences.
A 256 KB UB assumption overruns A3's 192 KB budget and can corrupt a neighboring buffer without a tiling error.
A hardcoded 20- or 24-core launch leaves most of an A5 die idle, while a hardcoded 32-core launch on a 28-core binned SKU is invalid.
A mode-2 flag sent to an A5 mode-4 peer, or a wait reached by only one AIV, deadlocks rather than producing a wrong number.
`GetUserWorkspace()` on arch35 resolves through the KFC base, so a pure AIV kernel that writes the raw workspace pointer can pass unit tests and still fault on the device.
A changed `TilingData` field width silently reinterprets every later field, so the kernel reads a valid-looking but wrong tile size.
All of these failures are avoidable at the host contract, before any performance tuning.

**Evidence.** Hardware specification in the CANN AscendC platform pages and cross-generation migration guide: A5 UB 256 KB physical / 248 KB reported, 40 KB SIMT reserve, L1 512 KB, L0C 256 KB, 32 AIC / 64 AIV full die, AIC:AIV 1:2, mode-4 cross-core flags; A3 dual-die 910B package, 192 KB UB, L0C 128 KB, mode-2 flags, SIMD-only.
The 950PR reference platform page records 128 GB for its 8×16 GB SKU; the A5 bring-up topology and **96 GB** HBM in this unit are field observations from the A5 (`a5-fia`, `a5-conv1d`, `a5-flash-comm`) fleet and are recorded as SKU-dependent, not as a universal A5 capacity.
The arch35 16 MB workspace reserve and the pure-AIV `507035`/errcode 95 failure are documented in the CANN AscendC runbooks.
This unit states contracts; it does not report a measured speedup.
