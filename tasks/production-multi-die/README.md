# Production multi-Die tasks — execution status

This directory is **not** a claim that five production tasks are available.
The dual-Die canary has a bounded executable verifier, and a real parent-owned
A3 execution was attempted: it reached the fused runtime, but the installed
CANN 8.5.1 A3 operator package has no `MatmulAllReduce` kernel for
`ascend910_93`. There is no hardware acceptance. The four model/network tasks
are **incomplete and blocked** on real model/network prerequisites and on an
unwritten serving harness; their verifiers report explicit blocked-prerequisite
status instead of inventing a result.

| Task | Status |
|---|---|
| `a3-dual-die-tp2-allreduce-placement` | **RUNTIME-UNSUPPORTED / not hardware-accepted.** A real A3 execution ran the candidate guard inside both sibling HCCL ranks and observed the expected cross-package rejection; the fused `npu_mm_all_reduce_base` call was then rejected because CANN 8.5.1 `ascend910_93` has no `MatmulAllReduce` kernel entry. The cross-package control remains diagnostic only. |
| `qwen35-gdn-dp2-tp2-asymmetric-aclgraph` | **BLOCKED / incomplete.** No real Qwen3.5 snapshot or executed DP=2/TP=2 serving harness. |
| `qwen3-moe-tp2-ep4-mc2-eplb-skew` | **BLOCKED / incomplete.** No real Qwen3-MoE snapshot or executed EP/MC2/EPLB harness. |
| `a3-16die-pd-disaggregated-mooncake-handoff` | **BLOCKED / incomplete.** No model, Mooncake setup, or executed PD handoff. |
| `a3-2node-roce-hccl-dp4-ep8-canary` | **BLOCKED / incomplete.** The standard single-host L3 slot is network-isolated; no cross-node acceptance exists. |

No `soundness_receipt.json`, `MANIFEST.sha256`, or synthetic before/after reward
is shipped. The four blocked tasks have no gold patch and no production
acceptance path. Do not promote them or describe them as available.

## Dual-Die canary path

**Lease:** four idle A3 Dies, for example `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`,
with `/dev/davinci0..3` bound into the slot, no `npu-smi` process row on those
NPUs, and a clean `/dev/shm/hccl*`. The candidate sibling pair must occupy
consecutive even two-Die slots in the visible order; otherwise the verifier
fails with `unsupported:lease_visible_order_not_tp2_aligned` rather than
guessing a production rank mapping. The verifier selects candidate sibling
`(0,1)` and negative-control `(1,2)` from that four-Die lease.

**Runtime:** `torch_npu` with `npu_mm_all_reduce_base`, HCCL, `npu-smi`, and a
writable `/dev/shm`; no model or network download. The fused operator
additionally requires a CANN A3 ops package that includes the `MatmulAllReduce`
kernel for `ascend910_93`; the managed CANN 8.5.1 package does not.
`MINDIE_BENCH_WORKSPACE` is set by `runner.verifier` for local and remote
verifiers.

**Candidate change:** create `vllm_ascend/distributed/placement.py` and integrate
its narrow single-node A3 TP=2 guard into `init_ascend_model_parallel`. Each
candidate rank child imports the guard and calls
`enforce_visible_tp_placement()` **before** `torch`/HCCL initialization; the
returned actual pair must equal the requested candidate pair. The cross-package
arm is explicitly labelled `production_path=False` and is never a candidate
pass.

**Observed blocker (2026-10-21, managed A3 `mindie-qwen35hist`):** the candidate
guard was accepted on sibling pair `(4,5)` in both candidate ranks, the
negative-control ranks observed the expected cross-package rejection, HCCL
groups initialized and were destroyed, and `/dev/shm` was clean. The fused call
then failed in CANN:

```text
aclnnMatmulAllReduce failed, error code is 161001
The binary_info_config.json of socVersion [ascend910_93]
does not support opType [MatmulAllReduce].
```

The installed `Ascend-cann-A3-ops 8.5.1` exports the `aclnnMatmulAllReduce`
symbol and header but has no `MatmulAllReduce` entry in the `ascend910_93`
kernel configuration. The active custom OPP vendor contains no such operator,
and a read-only CANN 9.1.0 A3 tree on the same host also lacks it. This is a
runtime capability limitation, not an environment-variable selection bug. No
hardware acceptance is claimed, and the fused operator must not be replaced
with a different operator. The verifier reports this condition as
`unsupported:matmul_all_reduce_kernel_unavailable` and preserves the raw error.

**Direct private-oracle canary from the candidate workspace:**

```bash
cd <candidate-vllm-ascend-worktree>
MINDIE_BENCH_WORKSPACE="$PWD" \
MINDIE_BENCH_ORACLE_ROOT="<repo>/tasks/production-multi-die/a3-dual-die-tp2-allreduce-placement" \
  bash "<repo>/tasks/production-multi-die/a3-dual-die-tp2-allreduce-placement/verifier/test.sh"
```

The private bundle contains only `verifier/`, `oracle/` and `task.toml`; it does
not need an `environment/` directory. With no real NPU, the same command returns
`unsupported:npu_smi_unavailable`.

**Parent-owned canary checks (no promotion is performed by this worker):**

```bash
.venv/bin/python -m mindie_bench.cli validate --tasks tasks/production-multi-die

.venv/bin/python -m mindie_bench.cli soundness \
  --task tasks/production-multi-die/a3-dual-die-tp2-allreduce-placement \
  --config "$L3_CONFIG" --tier l3 --output /tmp/mindie-dual-soundness

# Only after the fused kernel is available and the parent observes a real
# before=0 / after=1 proof:
.venv/bin/python -m mindie_bench.cli run \
  --tasks tasks/production-multi-die/a3-dual-die-tp2-allreduce-placement \
  --config "$L3_CONFIG" --tier l3 --groups G0 --repetitions 1 \
  --output /tmp/mindie-dual-run
```

`$L3_CONFIG` must provide `[remote]` with `container`, the container-side
`repo_path` of `vllm-ascend-idle-dp`, and `devices = [0,1,2,3]` (or the
equivalent four idle Dies). The task declares `npu_count = 4`. A real soundness
run may update the task's evidence only in the parent's checkout; this worker
does not create that receipt. While the fused kernel is absent, the soundness
and run paths cannot produce a hardware pass; they must report the explicit
unsupported condition rather than a synthetic pass.

## Blocked prerequisites for the four incomplete tasks

The four blocked tasks require, respectively:

- Qwen3.5 GDN snapshot at `MINDIE_BENCH_QWEN35_MODEL` (or
  `MINDIE_BENCH_MODEL_PATH`), four idle Dies, and an executed
  `--tensor-parallel-size 2 --data-parallel-size 2` serving harness.
- Qwen3-MoE snapshot at `MINDIE_BENCH_QWEN3_MOE_MODEL` (or
  `MINDIE_BENCH_MODEL_PATH`), eight idle Dies, and an executed
  `--tensor-parallel-size 2 --data-parallel-size 4 --enable-expert-parallel`
  harness.
- A model snapshot, all 16 Dies, Mooncake, and disjoint Prefill (`kv_producer`)
  / Decode (`kv_consumer`) endpoints; `MINDIE_BENCH_MOONCAKE_READY=1` only
  after provisioning and an executed handoff.
- Two reachable hosts, `MINDIE_BENCH_PEER_HOST`/`MINDIE_BENCH_HCCL_PEER`, and a
  direct multi-node lease (`MINDIE_BENCH_MULTINODE_DIRECT=1`). The standard
  single-host `--unshare-net` slot is insufficient.

These are blocked prerequisites, not evidence that the corresponding hardware
run happened. The current verifiers preserve the report and award no reward.
The dual-Die task is also not a completed hardware acceptance: its fused
runtime is unsupported by the installed A3 CANN 8.5.1 operator package. None
of the five tasks should be described as hardware-accepted.
