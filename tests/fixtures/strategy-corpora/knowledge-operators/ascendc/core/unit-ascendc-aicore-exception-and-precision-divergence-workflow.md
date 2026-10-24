---
schema: mindie-graded-unit/1
id: unit-ascendc-aicore-exception-and-precision-divergence-workflow
revision: 2
tier: L3
category: workflow
domain_path: ascendc/core
evidence_strength: reproduced_empirical
title: 'AscendC AICore exception, hang and numerical-divergence four-stage triage workflow'
conditions:
  kernel_language: ascendc
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  source_type: reproduced_debug_campaign_and_public_runbook_review
  source_locator: 'CANN AscendC crash, precision and sync-audit runbooks; field reproduction of 507015/507035, queue deadlock and bf16 tail divergence'
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-crash-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge
  - https://gitcode.com/Ascend/agent-skills
  - unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
  - unit-ascend-operator-roofline-and-four-view-measurement
  supersedes: []
triggers:
  error_signatures:
  - aclrtSynchronizeStream returns 507015 (AICore timeout or trap) or 507035 (vector core exception)
  - the kernel hangs in EnQue, DeQue, WaitFlag or CrossCoreWaitFlag and never reaches the next PRINTF
  - the first tile is correct and later tiles contain garbage, inf or NaN
  - the aligned prefix is correct but the tail is wrong near a 32 B or 256 B boundary
  - eager output looks correct but ACLGraph replay diverges or fails to capture
  - enabling DumpTensor or ASCEND_LAUNCH_BLOCKING makes the failure disappear while the uninstrumented run still fails
  - EZ9999 appears during tiling or graph compilation without a device-side crash
  keywords:
  - AscendC
  - AICore
  - '507015'
  - '507035'
  - EZ9999
  - hang
  - deadlock
  - EnQue
  - DeQue
  - WaitFlag
  - 507035 errcode 82
  - 507035 errcode 95
  - ICPU_RUN_KF
  - CPU twin
  - tikicpulib
  - AscendC::PRINTF
  - DumpTensor
  - observer perturbation
  - msaicerr
  - mssanitizer
  - msprof
  - PipeUtilization
  - MemoryUB
  - NaN
  - tail corruption
  - bf16 accumulation drift
---

# AscendC AICore exception, hang and numerical-divergence four-stage triage workflow

## Trigger and scope

Use when a custom AscendC kernel on Ascend 910B, 910_93 or 950 exhibits one of:

- a runtime error code `507015` (`ACL_ERROR_RT_AICORE_TIMEOUT` or an AICore trap) or `507035` (`ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`);
- a hang in `AllocTensor`, `EnQue`/`DeQue`, `WaitFlag`, `CrossCoreWaitFlag` or a `SyncAll`;
- a numerical divergence: `NaN`/`Inf`, tail corruption near a 32 B or 256 B boundary, bf16 accumulation drift, or a first-tile-correct/later-tile-garbage pattern;
- an observer-perturbation symptom where enabling `DumpTensor`, `PRINTF`, `mssanitizer`, or stream synchronization causes the error to vanish while the identical uninstrumented run still fails (or where a candidate patch branch execution count is `0` under a dump-enabled pass);
- `EZ9999` during tiling or graph compilation, where no device-side crash has occurred yet.

The workflow assumes the correctness oracle is frozen and the failing case is reproducible at least once. It does not cover a pure host Python error, a compile error, a link error, a device link-down, or a kernel whose mathematical contract is known to be wrong. Those are separate branches; mixing them into the AICore triage produces false root causes. Never treat a dump-enabled pass or zero-execution candidate branch as proof of a fix; always retain an uninstrumented baseline and an observer-only control before concluding.

The four stages are deliberately ordered: **stabilize and classify the first failure → shrink to one kernel and localize with the CPU twin and dumps → attribute the defect to memory, queue/sync or numerics → fix, regress and profile**. Do not skip from a 507035 directly to rewriting the math; the error code is a class, not a root cause.

## Facts to freeze before profiling

Record all of the following before changing code. A missing fact stays `unknown`; do not fill it from another SOC's constant.

| Fact | How to capture |
| --- | --- |
| SOC name and die/package topology | `npu-smi info`, `python3 -c "import acl; print(acl.get_soc_name())"` |
| Device id, CANN, driver, `torch_npu` | `npu-smi info -t board -i <id>`, `cat "$ASCEND_HOME_PATH/version.cfg"`, `python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"` |
| Kernel source identity | `sha256sum op_kernel/*.cpp op_kernel/*.h op_host/*.cpp` |
| TilingKey, blockDim, workspace size | plog `AIC_INFO`, or a host-side print of the tiling record |
| Input shape, stride, dtype, layout | a Python dump of `shape`, `stride`, `dtype`, `data_ptr() % 32` |
| Execution mode | eager, `torch.compile`, ACLGraph capture/replay; record the exact mode and the warmup count |
| Last known good case and first failing case | one row per case with input hash |
| Correctness tolerance and output dtype | pre-registered tolerance, exact expected dtype |
| Repetition count and seed | the same seed and at least three repeats |

```bash
npu-smi info
python3 - <<'PY'
import acl, torch, torch_npu
print("soc:", acl.get_soc_name())
print("device:", torch.npu.get_device_name(0), "available:", torch.npu.is_available())
print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)
PY
sha256sum op_kernel/*.cpp op_kernel/*.h | sort
```

For a hang, start with `timeout` so the process cannot consume the session indefinitely. For a crash, enable a core dump only when a host-side stack is needed: `ulimit -c unlimited`. Keep the original failing input and the plog; do not overwrite them while iterating.

## Stage 1 — Stabilize, reproduce and classify the first failure

The first goal is one recorded failure with a device-side identity, not a hypothesis. Force synchronous launch and verbose logging:

```bash
export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1
timeout 180s python3 test_op.py --case <failing_case> 2>&1 | tee /tmp/ascendc_fail.log
```

Locate the newest plog and extract the first error, not the last one:

```bash
LOG_ROOT="${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}"
LATEST=$(find "$LOG_ROOT/debug/plog" -type f -name 'plog-*.log' -mmin -10 | sort | tail -1)
echo "$LATEST"
grep -m5 -E 'extend info: errcode|errorStr|fault kernel_name|AIC_INFO|aicore exception|Bit map is' "$LATEST"
```

Use this differential before touching the kernel:

| Observation | First branch to take |
| --- | --- |
| `507035` + `errcode:(82, ...)` + `MTE2 to write UB ... out of bounds` | UB ledger or tile size, not the math. Recompute `sum(TQue_count × BUFFER_NUM × tile_bytes) + TBuf_bytes + fixedCost + headroom`. |
| `507035` + `errcode:(95)` + `DDR address of the MTE instruction is out of range` | Workspace pointer or GM extent. Check `GetUserWorkspace` vs raw `workspace`, arch35's 16 MB reserve, and the GM offset arithmetic. |
| `507035` + `errcode:(263, 82)` | Scalar/GM address out of range; check the source base and the 48-bit address arithmetic. |
| `507015` + `L0 read/write conflict in the MTE (same address)` on a Mix catlass kernel | Check `Arch::Resource` lifetime before geometry. A function-scoped Resource double-construction can corrupt the TPipe/event environment. |
| `507015` + `The configuration of FIXP is illegal` | Fixpipe layout/stride, not a hang. Audit degenerate `RowMajor` strides and zero extents. |
| `507015` + fast return, no watchdog expiry | A hardware trap, not a hang. It is statically inspectable; read the faulting instruction and the Fixpipe/L0/MTE configuration. |
| `507015` with a 90 s watchdog expiry | A true hang; go to the queue/sync branch. |
| `EZ9999` at tiling/compile, no device crash | Shape/attribute constraint. Check the operator's documented set (for example MLA graph-mode `num_heads/num_kv_heads ∈ {32,64,128}` or FIA MLA `group_num` power-of-two) before editing the kernel. |
| `core_id >= blockDim` or a core id outside the launch extent | Launch/config mismatch or a stale binary; verify the arch directory and the compiled binary. |
| `507035` immediately on every case, with an arch35/SIMT binary on an arch22 board | Arch mismatch. Rebuild for the actual SOC or change the implementation route; kernel math is not the cause. |

`AIC_INFO` contains the tiling fields expanded as integers, plus `tilingKey`, `blockDim` and the fault kernel name. Record those values verbatim; they are the join key between the host tiling and the device binary. The `Bit map is` line with `aicmap`/`aivmap` is the reliable discriminator between AIC and AIV faults; the generic `aicore exception` text does not distinguish them.

Exit Stage 1 with a row: `device`, `kernelName`, `TilingKey`, `blockDim`, `core_id`, `error_code`, `errorStr`, `first_failing_case`, `repetition`. If the failure cannot be reproduced, record `unreproduced` and stop; a one-off pass is not a root-cause proof.

## Stage 2 — Shrink to one kernel and localize with the CPU twin and dumps

Do not debug the full model. Extract one kernel invocation with the smallest failing shape and the same tiling record. Then separate math/indexing from timing/resource defects:

**CPU twin.** Compile the kernel against `tikicpulib` and run it on the host with the same inputs and the same tiling data:

```cpp
ICPU_SET_TILING_KEY(tilingKey);
ICPU_RUN_KF(my_kernel, blockDim, x, w, y, workspace, tiling);
```

The CPU twin exercises scalar and vector arithmetic, index arithmetic, masks and the tiling ABI. It does **not** reproduce UB bank conflicts, pipeline ordering, event pools, cross-core flags or real DMA timing. A CPU-twin pass with a device failure is evidence for a timing/resource/sync defect, not evidence that the math is correct on the device.

**Kernel-side dumps.** Use the uppercase macro and enable the dump switch:

```cpp
AscendC::PRINTF("blk=%d tile=%d off=%lld n=%d\n",
                static_cast<int>(GetBlockIdx()), static_cast<int>(t),
                static_cast<long long>(off), static_cast<int>(n));
DumpTensor(xLocal, 1001, 64);   // descriptor 1001, first 64 elements
```

```bash
export ASCENDC_DUMP=1
timeout 180s python3 test_op.py --case <failing_case> 2>&1 | grep -E 'blk=|DumpTensor|GM_ADDR'
```

`AscendC::PRINTF` is the macro name on current CANN; `AscendC::printf` and a standalone `kernel_printf.h` include are not portable. `PRINTF` inserts synchronization and changes timing, so it localizes a region but does not by itself prove the faulting instruction. Confirm with a bisect or a dump.

**Bisect by stage.** Disable in this order, rebuilding after each step:

1. the epilogue / final store, keeping the compute and the input copy;
2. the vector compute, keeping copy-in and copy-out (write a known pattern);
3. the copy-out, keeping copy-in and compute (dump the UB result);
4. the cross-core handoff, keeping a single-core path.

The last step that still fails defines the minimal trigger. If disabling a `PipeBarrier` changes the result, that is not proof the barrier was the root cause; it is proof the dependency was real and the barrier was covering it. If forcing a known-good path makes the output non-zero, the main data path is alive and the defect is in the branch-specific resource or code.

Exit Stage 2 with a single kernel, a single case, a minimal trigger stage, and a recorded CPU-twin result (`pass`/`fail`).

## Stage 3 — Attribute the defect: memory, queue/sync or numerics

Run the three branches in this order. Stop when one branch explains every observation; do not run all three and pick the most convenient.

**3A. Memory and resource attribution.** Recompute the UB ledger from the actual `InitBuffer` calls and the actual dtype sizes. Account for every queue depth and every `TBuf`, then add at least 32 KB of headroom. A ledger that fits exactly is a defect even if the first case passes.

```bash
grep -nE 'InitBuffer|BUFFER_NUM|TQue<|TBuf<' op_kernel/*.cpp op_kernel/*.h
python3 - <<'PY'
# paste the actual queue/tbuf sizes here; do not approximate with one dtype
tile = 16 * 1024
queues = 2 * 2 * tile          # 2 queues, BUFFER_NUM=2
tbuffers = 16 * 1024 + 2 * 1024 + 1024
print("ledger bytes:", queues + tbuffers + 32 * 1024)
print("A3 limit:", 192 * 1024, "A5 limit:", 248 * 1024)
PY
```

For a suspected out-of-bounds or multi-core overwrite, use `msaicerr` on the fault bundle and `mssanitizer` for an active check:

```bash
python3 msaicerr.py -p /tmp/fault_bundle -out /tmp/msaicerr_out -dev 0
grep -E 'Illegal|Misaligned|Multi-core|Leak|ERROR|WARNING' /tmp/msaicerr_out/* 2>/dev/null | head -40
```

`msaicerr` parses an existing exception dump; it does not prove numerical correctness. `mssanitizer` detects illegal read/write, multi-core overwrite, misaligned access, illegal free, leaks and unused allocations, but it changes timing and can hide a race. Treat a clean run as one piece of evidence, not a proof.

**3B. Queue and synchronization attribution.** The signature is a hang, or a first-iteration-correct/later-iteration-garbage pattern. Check the lifecycle first:

- every `AllocTensor` has a `FreeTensor` on every path in the same iteration;
- every `EnQue` has a matching `DeQue` on the same queue;
- a `WaitFlag` is after its producer's `SetFlag`, not before it;
- a `CrossCoreWaitFlag` has a reachable `CrossCoreSetFlag` with the same mode, pipe and `flagId`;
- every `SyncAll` participant reaches the barrier; idle cores do not return early;
- the loop runs more than `BUFFER_NUM` iterations, so a leaked slot actually exhausts the queue.

```bash
grep -nE 'AllocTensor|FreeTensor|EnQue|DeQue|SetFlag|WaitFlag|CrossCore|SyncAll' op_kernel/*.cpp op_kernel/*.h
timeout 180s env ASCEND_LAUNCH_BLOCKING=1 ASCENDC_DUMP=1 python3 test_op.py --case <mix_case>
```

A `PRINTF` at the loop head and tail is the cheapest discriminator: if `tile=0` prints both lines and `tile=1` prints only the head, the wait is inside the second iteration. If the value is correct on iteration 0 and garbage on iteration 1 with the same input, suspect a cross-iteration WAR race (the next copy-in overwrites a buffer the previous vector stage is still reading), not a numerical formula. The fix is a queue handoff or a double buffer, not a larger tolerance. `PipeBarrier<PIPE_ALL>` does not wait for an asynchronous `asc_vf_call` VF pipeline and is not a universal repair.

**3C. Numerical attribution.** Freeze an fp32 reference and compare per element. Then isolate the boundary:

- **Tail-only control:** run `N` just below and just above 32 B and 256 B boundaries. A jump localizes the defect to the tail copy or mask. Check `DataCopyPad` `blockLen`, `SetVectorMask`, and that the padded lanes carry the reduction identity (`0`, `-inf`, `+inf` as appropriate).
- **Accumulation drift:** compare a bf16 accumulator against an fp32 accumulator on the same input. A growing error after many reductions, or a bitwise difference that changes the argmax on borderline logits, is accumulation-order drift, not a random seed. Keep the fp32 accumulator and cast once at the store.
- **NaN/Inf:** find the first non-finite value with a dump, not with a downstream max. Check division by zero, `log(0)`, `sqrt` of a negative, a zero dynamic-quant scale, an uninitialized tail, and an out-of-range cast.
- **Determinism:** run the same input three times and compare bytes. A changing result on the same device and binary is a race or an unordered reduction, not noise.
- **Graph vs eager:** if eager passes and ACLGraph replay fails, compare the captured input addresses, the metadata buffer contents and the replay-time device values. A host list captured into the graph is a constant; a device tensor read at replay is not. Verify which one the operator consumed.

Exit Stage 3 with a named root cause and a falsifiable check: "errcode 82 because the ledger omitted the second output queue", "hang because iteration 2 reuses the input slot before its free", "tail wrong because `DataCopyPad` used the full tile byte count", "drift because the partial sum is cast to bf16 between levels". A hypothesis that failed is recorded too; it prevents the next run from repeating it.

## Stage 4 — Fix one variable, regress the boundary, and profile the retained path

Change one factor, rebuild, and rerun the original failing case before anything else. Then run the neighbor set:

1. the same case with `N ± 1` and with `N` just below and just above the next 32 B and 256 B boundaries;
2. a zero-length or empty-expert case, if the operator admits one;
3. a graph capture and two replays, if the operator is used in a graph;
4. the full unit/smoke set for the operator family;
5. a repeatability check on the fixed case.

Only after correctness is stable, profile the retained path:

```bash
msprof op --application="python3 test_op.py --case <case>" \
  --output=/tmp/prof_op --aic-metrics=PipeUtilization,Memory,MemoryUB --l2=on
python3 - <<'PY'
import glob, pandas as pd
path = sorted(glob.glob("/tmp/prof_op/**/op_summary_*.csv", recursive=True))[-1]
df = pd.read_csv(path)
keep = [c for c in ["op_name","op_type","aicore_time(us)","aiv_time(us)",
                    "aiv_vec_time(us)","aiv_mte2_time(us)","aiv_mte3_time(us)",
                    "aic_mte2_time(us)","aic_mac_time(us)"] if c in df]
print(path)
print(df[keep].to_string(index=False))
PY
```

Read the pipeline columns separately for AIC and AIV. `aic_mte2_time` high with `aic_mac_time` low is Cube starvation; `aiv_mte2_time` high is GM→UB input; `aiv_mte3_time` high is UB→GM output; `aiv_vec_time` high is the vector ALU. Do not merge AIC and AIV MTE2 columns. If the profiler does not export the columns, record `pipeline signal unavailable` and use the inherited measurement contract; do not fabricate zeroes.

## Verification and exit criteria

The workflow is complete only when all of the following hold:

- **Root cause is named and reproducible.** The original failing case is recorded with device, `kernelName`, `TilingKey`, `blockDim`, `core_id`, error code and `errorStr`, and the fix changes that exact observation.
- **Correctness gate passes.** Finiteness, max-abs and relative error against the frozen fp32 reference are inside the pre-registered tolerance; output dtype is exact; inputs are unchanged; three repeats are byte-identical; the tail-only control does not jump; a zero-length case does not produce `NaN`.
- **Neighbor boundaries pass.** At least one aligned, one 32 B-unaligned and one 256 B-unaligned case pass, plus the empty/tail case that the operator admits.
- **Sync gate passes.** The fixed Mix case runs three times under a timeout with no hang and no `507015`/`507035`; every producer/consumer pair is accounted for.
- **Graph gate passes.** Eager and ACLGraph replay agree within tolerance and the replay uses the current device metadata, not a captured host constant.
- **Profile is retained.** The `msprof` artifact, case identity and source hash are stored; the pipeline classification is recorded; a performance claim is not made from the failed-run baseline.

Exit with one of: `fixed_verified`, `mitigated_with_boundary`, `unreproduced`, or `invalid_measurement`. A passing case after a change is not proof that the original root cause was understood; the recorded differential and the failed hypotheses remain part of the result. A fix that only widens the tolerance, removes a barrier without a replacement, or disables the failing shape is a mitigation, not a root-cause fix.
