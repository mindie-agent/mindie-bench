---
schema: mindie-graded-unit/1
id: unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
revision: 1
tier: L2
category: sop
domain_path: ascendc/core
evidence_strength: software_contract
title: 'AscendC kernel SOP: host tiling, TPipe/TQue double buffering, queue lifecycle, cross-core handoff and custom-op packaging'
conditions:
  kernel_language: ascendc
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  source_type: public_software_contract_review_and_transcript_distillation
  source_locator: 'CANN AscendC programming and runbook contracts; AscendC operator-development and performance-optimization skill contracts; user AscendC transcripts'
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ut-guide/op-kernel-ut-guide.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/torch-ascendc-op-extension/routes/aclnn-registry.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/graph/torch-custom-ops-guide/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge
  - https://gitcode.com/Ascend/agent-skills
  - unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
  supersedes: []
triggers:
  error_signatures:
  - host tiling passes but the kernel fails with 507035 or writes an all-zero output because the UB ledger omitted double buffering
  - the last tile is copied with plain DataCopy and the shape is not a multiple of 32 B
  - a kernel hangs in AllocTensor, EnQue or DeQue because a Freed slot never returns
  - a Mix kernel deadlocks at CrossCoreWaitFlag because the flag mode, pipe or participant set differs from the producer
  - a custom op works in eager mode but fails torch.compile because no Meta implementation is registered
  keywords:
  - AscendC
  - tiling
  - TilingFunc
  - usedCoreNum
  - singleCoreLen
  - tileLen
  - DataCopyPad
  - SetVectorMask
  - TPipe
  - TQue
  - BUFFER_NUM
  - double buffer
  - AllocTensor
  - FreeTensor
  - EnQue
  - DeQue
  - PipeBarrier
  - SetFlag
  - WaitFlag
  - CrossCoreSetFlag
  - CrossCoreWaitFlag
  - SyncAll
  - aclnn
  - GetWorkspaceSize
  - TORCH_LIBRARY
  - Meta
  - FakeTensor
---

# AscendC kernel SOP: host tiling, TPipe/TQue double buffering, queue lifecycle, cross-core handoff and custom-op packaging

## Use when

A custom AscendC operator is being written or repaired for Ascend 910B, 910_93 or 950 and must be both correct and pipeline-overlapped. The SOP starts at the host tiling function and ends at a PyTorch custom op that survives `torch.compile`/ACLGraph capture. It assumes the alignment and UB/L1 axioms from the inherited L1 units. A kernel whose correctness oracle is not frozen, or whose target SOC is unknown, is not ready for this SOP.

Record the observed facts before editing: SOC string, CANN/`torch_npu` revision, input shape/dtype/stride, layout, output dtype, `BUFFER_NUM`, UB capacity from the platform API, launch core count, graph/eager mode, and the exact failing case. A missing fact stays `unknown`; do not substitute an A2/A3 constant.

## 1. Design the host tiling function

The host `TilingFunc` owns five numbers: `usedCoreNum`, `singleCoreLen`, `tileLen`, `tileCount`, `lastTileLen`. Compute them from the observed work and the queried platform, not from a constant.

```text
total          = number of independent elements, rows or tokens
coreNum        = PlatformAscendC::GetCoreNumAiv()   # Vector work
                 PlatformAscendC::GetCoreNumAic()   # Cube-only work
minWorkPerCore = a configured minimum, e.g. 256 elements or one tile
usedCoreNum    = min(coreNum, ceil(total / minWorkPerCore))
singleCoreLen  = ceil(total / usedCoreNum)
```

`minWorkPerCore` is a design parameter, not the final `tileLen`; after the UB budget produces `tileElems`, recompute `usedCoreNum` if the minimum would leave a core with less than one tile. The two numbers must be consistent in the tiling record.

For the UB tile:

```text
queueSlots     = sum over TQue of BUFFER_NUM
tileBytes      = floor((UB_limit - fixedCost - headroom - TBuf_bytes) / (queueSlots * elemBytes))
tileBytes      = floor(tileBytes / 32) * 32                 # 32 B UB DMA granule
tileBytes      = floor(tileBytes / 256) * 256               # widest vector repeat = 256 B
tileElems      = tileBytes / elemBytes
tileCount      = ceil(singleCoreLen / tileElems)
lastTileLen    = singleCoreLen - (tileCount - 1) * tileElems
```

`tileLen` is a byte count at the DMA boundary and an element count at the vector boundary; keep both names distinct in code. For `bfloat16`/`float16`, a 256 B repeat is 128 elements; for `int8`/`fp8_e4m3fn`, it is 256 elements. If `tileElems` rounds to zero, the budget is infeasible: reduce `queueSlots`, lower `BUFFER_NUM`, or increase the per-core work; do not emit a zero tile.

The tail is a first-class branch. Copy the aligned prefix with `DataCopy`, then copy the final `lastTileLen` elements with `DataCopyPad` and a `DataCopyExtParams` whose `blockLen` is the true byte count. Mask the vector tail with `SetVectorMask<T>(lastTileLen)` or the counter-mode equivalent. A plain `DataCopy` of a partial 32 B granule is undefined, and a `DataCopyPad` whose base address is not 32 B aligned is also out of contract: start every tile at an aligned element offset and let only the byte count be ragged.

Run the geometry check on the host before building:

```bash
python3 - <<'PY'
import math

def plan(total, core_num, ub_bytes, elem_bytes, queue_slots, tbuf_bytes=0,
         fixed=32768, headroom=32768):
    used = min(core_num, math.ceil(total / 256))
    per_core = math.ceil(total / used)
    tile_bytes = (ub_bytes - fixed - headroom - tbuf_bytes) // (queue_slots * elem_bytes)
    tile_bytes = (tile_bytes // 32) * 32
    tile_bytes = (tile_bytes // 256) * 256
    tile_elems = tile_bytes // elem_bytes
    assert tile_elems > 0, "infeasible UB ledger"
    tile_count = math.ceil(per_core / tile_elems)
    last = per_core - (tile_count - 1) * tile_elems
    assert last > 0
    print(dict(used=used, per_core=per_core, tile_elems=tile_elems,
               tile_bytes=tile_elems * elem_bytes, tile_count=tile_count, last=last))

plan(total=1 << 20, core_num=56, ub_bytes=248 * 1024, elem_bytes=2,
     queue_slots=4, tbuf_bytes=19 * 1024, fixed=16 * 1024, headroom=32 * 1024)
PY
```

**Invariants.** `usedCoreNum == kernel GetBlockNum()`; the per-core ranges are contiguous, non-overlapping and cover `[0, total)`; `singleCoreLen * usedCoreNum >= total`; `tileBytes % 32 == 0` and `tileBytes % 256 == 0` when the widest vector repeat is used; `lastTileLen > 0`; the tiling record uses fixed-width fields.

**Verification.** Print the five numbers and the derived ranges from the host, then print `GetBlockNum()` and `GetBlockIdx()` once in the kernel under `ASCENDC_DUMP=1`. A mismatch is a host/launch contract defect, not a numerical issue. Run at least one case with `N % 32 != 0` and one with `N % 256 != 0`.

## 2. Plan `TPipe` and `TQue` buffers before enabling double buffering

`TPipe` is created once in the kernel entry and passed by pointer into the kernel class. A `TPipe` member inside the class is legal but can inhibit scalar constant folding and adds scalar time; prefer the entry-owned object. Each `TQue<QuePosition::VECIN, BUFFER_NUM>` and `TQue<QuePosition::VECOUT, BUFFER_NUM>` allocates `BUFFER_NUM` slots of the configured tile, and every `TBuf` is a separate allocation.

The ledger is:

```text
sum(TQue_count * BUFFER_NUM * tile_bytes) + sum(TBuf_bytes) + fixedCost + headroom <= UB_limit
```

`UB_limit` comes from `GetCoreMemSize(UB)`: 192 KB on A2/A3, 248 KB on A5, about 208 KB on the A5 SIMT/regbase path after the 40 KB SIMT reserve. `fixedCost` covers masks, scale vectors, event tables and framework scratch. Reserve at least 32 KB of headroom for device stack and implicit allocations; a ledger that divides the available UB exactly to the last byte is a known cause of a silent all-zero output or a delayed `507035`. For a 16 KB tile with one input queue and one output queue at `BUFFER_NUM=2`, the queues alone consume `4 × 16 KB = 64 KB`; adding a 16 KB fp32 scratch, 2 KB mask and 1 KB scale gives 83 KB before headroom. That fits 192 KB and 248 KB, but a 96 KB A3 multibuffer slice would leave only 13 KB before the 32 KB headroom rule, so it fails the rule even though the raw 83 KB is below 96 KB.

```bash
grep -nE 'InitBuffer|BUFFER_NUM|TQue<|TBuf<' op_kernel/*.cpp op_kernel/*.h
```

**Invariants.** Every `InitBuffer` call appears in the ledger; the ledger includes every queue depth and every `TBuf`; `BUFFER_NUM=2` is enabled only when the ledger plus headroom fits; a dtype change re-computes the ledger because element size changes; the tile used by `DataCopyPad` is the same tile allocated by `InitBuffer`.

**Verification.** Put the ledger in a comment next to `Init` and print the computed sizes once. Build and run a case that exercises the largest tile. If the kernel returns all zeros with no error, reduce the budget by 32 KB and rerun before changing the math; that single-variable check distinguishes a UB budget defect from a numerical defect.

## 3. Keep the queue lifecycle strictly balanced

The canonical two-queue sequence for one tile is:

```cpp
TQue<QuePosition::VECIN,  BUFFER_NUM> inQ;
TQue<QuePosition::VECOUT, BUFFER_NUM> outQ;

for (int64_t t = 0; t < tileCount; ++t) {
    int64_t n = (t + 1 == tileCount) ? lastTileLen : tileElems;
    LocalTensor<T> x = inQ.AllocTensor<T>();
    DataCopyPad(x, xGm[off], copyParams, padParams);   // aligned base, true byte count
    inQ.EnQue(x);
    x = inQ.DeQue<T>();

    LocalTensor<T> y = outQ.AllocTensor<T>();
    Compute(y, x, n);                                  // SetVectorMask for the tail
    outQ.EnQue(y);
    y = outQ.DeQue<T>();
    DataCopyPad(yGm[off], y, copyParams);

    outQ.FreeTensor(y);
    inQ.FreeTensor(x);                                 // same iteration, no early continue
    off += n;
}
```

The middle `EnQue`/`DeQue` pair belongs to the output queue; it is not a second copy of the input pair. `AllocTensor` blocks when the queue has no free slot, so a missing `FreeTensor`, an early `continue`, or a `return` between `AllocTensor` and `FreeTensor` converts a data bug into a hang. With `BUFFER_NUM=2`, a slot is only returned by `FreeTensor`; the queue depth is not a garbage collector.

```bash
python3 - <<'PY'
import pathlib, re
src = "\n".join(p.read_text() for p in pathlib.Path("op_kernel").glob("*.[ch]*"))
for name in ("AllocTensor", "FreeTensor", "EnQue", "DeQue"):
    print(name, len(re.findall(rf"\b{name}\b", src)))
PY
timeout 120s env ASCEND_LAUNCH_BLOCKING=1 ASCENDC_DUMP=1 python3 test_op.py --case tail
```

Static counts are a smoke test, not a proof: branches can make the counts equal while one path still leaks a slot. Use the runtime check and a loop count greater than `BUFFER_NUM` (at least three iterations) so a missing free actually exhausts the queue.

**Invariants.** Every `AllocTensor` has exactly one `FreeTensor` on every control-flow path in the same iteration; every `EnQue` has a matching `DeQue` on the same queue; the input tensor is not freed before its last consumer; the output tensor is not freed before its `DataCopyPad` completes; no `return`/`break` skips a free.

**Verification.** Run a three-iteration minimum, a tail case, and a case with an early-exit branch. A hang under `timeout` plus a `DumpTensor` at the loop head and tail localizes the missing pair.

## 4. Make the barrier and cross-core contract explicit

Use the weakest primitive that covers the actual dependency:

- **Same core, same pipe, reused buffer:** `PipeBarrier<PIPE_V>()` after a vector producer before the next vector consumer. It orders the local vector pipe only; it does not publish GM and does not order another pipe.
- **Same core, different pipes:** `SetFlag<HardEvent::MTE2_V>(eventId)` / `WaitFlag<HardEvent::MTE2_V>(eventId)` for copy-in to vector, `MTE3_MTE2` for the reverse, `V_MTE3` for vector to copy-out. The event must be raised by the real producer instruction stream; pre-setting a flag before the loop to "unblock" the first wait breaks the pairing and can deadlock.
- **AIC↔AIV in a Mix kernel:** `CrossCoreSetFlag<mode, pipe>(flagId)` on the producer and `CrossCoreWaitFlag<mode, pipe>(flagId)` on the consumer. A2/A3 use mode 2, where both AIVs must report; A5 adds mode 4, where AIV0 and AIV1 can report independently. The `flagId` must be reserved from the pool shared with Matmul and `SyncAll`, and the producer pipe must match the data path (`PIPE_FIX` for a Cube result, `PIPE_V` for a Vector result).
- **All participating cores at a phase boundary:** `SyncAll()` only after every participant has reached the same point. The AIV-only form does not cover an AIC producer, and an idle core that returns early never reaches the barrier, so the remaining cores wait forever.

A minimal AIC→AIV handoff through GM workspace:

```cpp
if ASCEND_IS_AIC {
    // write the Cube partial to a 32 B-aligned workspace slot
    CrossCoreSetFlag<CROSS_MODE, PIPE_FIX>(FLAG_AIC_READY);
}
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<CROSS_MODE, PIPE_V>(FLAG_AIC_READY);
    // read the workspace slot, combine, write the final output
}
```

Use the target's supported `CROSS_MODE` and verify it, rather than copying a mode from another SOC. `SyncAll` is not a substitute for this pair: a `SyncAll` after the AIC write does not publish the write to an AIV that never reached the barrier.

```bash
grep -nE 'PipeBarrier|SetFlag|WaitFlag|CrossCoreSetFlag|CrossCoreWaitFlag|SyncAll' op_kernel/*.cpp op_kernel/*.h
timeout 180s env ASCEND_LAUNCH_BLOCKING=1 python3 test_op.py --case mix
```

**Invariants.** Every `CrossCoreWaitFlag` has a reachable `CrossCoreSetFlag` with the same mode, pipe, `flagId` and participant set; every `SyncAll` is reached by all declared participants; no `WaitFlag` is placed before its producer's `SetFlag`; a flag is not reused across phases without a matching completion.

**Verification.** Count producer and consumer sites, then run the Mix case under a timeout. A deadlock with a `PRINTF` before the wait and none after localizes the missing producer. A pass on one AIV but a hang on the pair indicates a mode-2 participant mismatch or an early return. Use `msprof` to confirm the AIC and AIV phases overlap rather than serialize.

## 5. Package the kernel as a custom op with a real two-stage aclnn and a Meta kernel

There are two packaging routes. Use the one the project already has:

- **Direct kernel launch:** `<<<...>>>` or the AscendC launch macro, `TORCH_LIBRARY` registration, `torch.ops.load_library`.
- **aclnn registration:** `op_host` (`OP_ADD` definition, tiling, infershape), `op_kernel`, and an `op_api` wrapper that exports `aclnnXxxGetWorkspaceSize` plus `aclnnXxx`. The PyTorch adapter calls the two-stage API inside `at_npu::native::OpCommand::RunOpApiV2` or `EXEC_NPU_CMD_V1`, using the current NPU stream.

For the aclnn route, the adapter owns the `aclTensor`/`aclIntArray` objects and must destroy them after the call. The two-stage contract is:

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnXxxGetWorkspaceSize(xAcl, ..., &workspaceSize, &executor);
void* workspace = nullptr;
if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnXxx(workspace, workspaceSize, executor, stream);
```

The PyTorch side needs both a backend implementation and a Meta implementation:

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("my_op(Tensor x, Tensor w, int tile) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("my_op", TORCH_FN(MyOpNpu));
}
TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("my_op", TORCH_FN(MyOpMeta));
}
```

The Meta kernel must forward the real arguments and derive shape/dtype/device, not duplicate one tensor as a stand-in for another. A Meta that validates the wrong tensor passes eager tests and fails only under `torch.compile`/ACLGraph capture. Multi-output ops return `std::tuple`, not `std::vector`. For the Python path, `torch.library.custom_op` plus `register_fake` is the equivalent pair.

```bash
OPP_ROOT="${ASCEND_CUSTOM_OPP_PATH:-$ASCEND_HOME_PATH/opp}"
find "$OPP_ROOT" -name 'libcust_opapi.so' -print | head
find "$OPP_ROOT" -name 'libcust_opapi.so' -print -quit | xargs -r nm -D 2>/dev/null | grep -E 'aclnn.*(GetWorkspaceSize)?$' | head
python3 - <<'PY'
import torch, torch_npu
print(torch.ops.npu.my_op)
x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
y = torch.ops.npu.my_op(x, torch.randn(128, 128, dtype=torch.bfloat16, device="npu"), 128)
torch.npu.synchronize()
print(y.shape, y.dtype, y.data_ptr() % 32)
PY
```

**Invariants.** The Python schema, the `OP_ADD` signature, the tiling parser and the kernel entry consume the same argument order and widths; the installed aclnn header name matches the exported symbol, not the source filename; `GetWorkspaceSize` and the execution call use the same executor; every `aclTensor`/`aclIntArray` is destroyed; the output dtype is exactly the declared dtype; the input is not mutated.

**Verification.** Check the exported symbol, run eager, then capture with `torch.compile` or ACLGraph and replay twice. Compare output shape/dtype and `data_ptr` stability, and compare the Meta-derived shape with the eager shape. A graph-only failure with an eager pass is a Meta or stream/queue-lifetime defect, not a kernel-math defect.

## Claim boundaries

This SOP is a build-and-correctness procedure. It does not assert a speedup, a universal tile size, or a UB number for every SKU. The 32 B/256 B geometry, the UB ledger and the flag mode are target facts; the correct values must be read from the target platform and the installed CANN. A passing tail case does not establish a full shape envelope, and a passing eager custom op does not establish graph capture.
