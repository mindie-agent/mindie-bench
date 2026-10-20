# A3 same-package dual-Die TP=2 MatmulAllReduce placement

You are given an isolated A3 sub-cluster of **four leased Dies** (`/dev/davinciN`
for the four `N` in `ASCEND_RT_VISIBLE_DEVICES`). Two Dies form a sibling
package pair; a different package contributes the cross-package control Die.
The candidate sibling pair must occupy consecutive even two-Die TP slots in the
visible order (for example `0,1` in `0,1,2,3`), so the production rank slot is
unambiguous. The production failure is a TP=2 group placed on Phy-IDs `(1,2)`:
it straddles two A3 NPU packages instead of using sibling Dies `(2k, 2k+1)`.

## Required candidate change

Add `vllm_ascend/distributed/placement.py` to the candidate checkout and wire
its TP=2 guard into the pinned `init_ascend_model_parallel` path. The verifier
imports the candidate module in each real rank and executes this interface
before `torch`/`torch_npu`/HCCL initialization:

```python
class CrossPackagePlacementError(RuntimeError): ...

def validate_tp_placement(device_phy_ids: Sequence[int]) -> dict:
    """Return visible_phy_ids/device_phy_ids, packages and sibling_pairs.
    Raise CrossPackagePlacementError when the IDs span A3 packages."""

def validate_rank_tp2_placement(rank: int, visible_phy_ids: Sequence[int]) -> dict:
    """Resolve the two-Die slot for this TP=2 rank and validate that pair."""

def enforce_visible_tp_placement(visible: str | None = None) -> dict:
    """Parse ASCEND_RT_VISIBLE_DEVICES (or the explicit value) and validate it."""
```

A3 packages own exactly `Phy-ID 2k` and `2k+1`; `phy_id // 2` is the package.
The production trigger must be narrow: **A3, single-node, TP=2 only**, using the
current rank's two-Die slot and the real `ASCEND_RT_VISIBLE_DEVICES` list. Do
not blanket-reject DP/EP/PP or larger visibility lists.

## Real workload and acceptance

The verifier runs read-only `npu-smi info`, selects a candidate sibling pair and
an explicit cross-package negative-control pair from the four leased Dies,
checks that every leased Die is idle and `/dev/shm/hccl*` is clean, then runs
two sequential two-rank HCCL worlds:

- **candidate arm:** each child imports the candidate guard, calls
  `enforce_visible_tp_placement()` before `torch`/HCCL, verifies the returned
  actual pair equals the requested pair, and only then calls the real fused
  operator:

  ```python
  torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, reduce_op="sum")
  ```

- **negative-control arm:** the child executes the guard only to observe the
  expected `CrossPackagePlacementError`, then runs an explicitly labelled
  diagnostic HCCL arm with `production_path=False`. This arm exists only under
  the full four-Die lease and can never become a candidate pass.

The expected fused result is recomputed independently on CPU. Every measured
iteration contributes to the relative-error maximum; the last sample is not
used as the result. A candidate pass requires:

1. the candidate guard executes inside both candidate rank launches, accepts
   the actual sibling pair, and the negative-control child observes rejection;
2. both arms return finite samples (at least 8) with the maximum relative error
   over **all** iterations at most `2e-2`;
3. the candidate median latency is no more than `2.0x` the labelled negative
   control median (the actual ratio is recorded);
4. every measured rank observes successful `destroy_process_group`; a cleanup
   failure fails the run and remains visible in the receipt;
5. `/dev/shm` is available and has no residual `hccl*` entry after both runs.

There is no model-serving claim and no prefilled evidence. If `npu-smi`,
`torch_npu`, `npu_mm_all_reduce_base`, the device nodes, or `/dev/shm` are
unavailable, the verifier exits nonzero with `unsupported:...`; that is not an
acceptance. Do not reset the server, kill peer jobs, or widen the lease.

## Observed runtime limitation (managed A3 container, 2026-10-21)

A real two-rank execution on the managed A3 container `mindie-qwen35hist`
(CANN 8.5.1, `Ascend-cann-A3-ops` 8.5.1) initialized HCCL, and both candidate
ranks accepted the guard, but the fused operator was rejected by CANN:

```text
aclnnMatmulAllReduce failed, error code is 161001
The binary_info_config.json of socVersion [ascend910_93]
does not support opType [MatmulAllReduce].
```

`aclnnMatmulAllReduce` is exported by the installed `libopapi.so` and its
header is present, but the installed `ascend910_93` kernel configuration has no
`MatmulAllReduce` entry. The active `ASCEND_CUSTOM_OPP_PATH` vendor contains no
such operator, and a read-only 9.1.0 A3 operator tree on the same host (x86_64
userland) also lacks it. This is a runtime capability limitation, not a missing
environment variable or an omitted installed package.

No hardware acceptance has been completed for this task. Do not report a
measured pass, and do not replace the fused operator with a different operator
to manufacture one. The verifier preserves the raw ACLNN error and reports this
specific condition as `unsupported:matmul_all_reduce_kernel_unavailable`.

## Operator acceptance command (A3 host, non-destructive probe first)

```bash
ssh root@192.168.13.165 'npu-smi info -l | head'
# Parent leases four idle Dies (for example 0,1,2,3) and runs the frozen task
# through the production runner; the private oracle receives only verifier/,
# oracle/ and task.toml and executes with cwd = candidate workspace.
```
