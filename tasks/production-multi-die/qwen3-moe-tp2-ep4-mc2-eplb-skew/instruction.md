# Qwen3-MoE TP=2 x EP=4 MC2 + EPLB under skewed expert routing

Run the **8-Die full-sub-cluster** Qwen3-MoE workload with `TP=2, EP=4`,
`Dispatch`/`Combine` AlltoAll and MC2 fused communication. The scoring input
is an expert-routing skew distribution: a small set of hot experts receives
most tokens while the remaining experts are cold. This exposes EPLB lag,
AlltoAll tail latency and a single straggler rank.

Reproduce the uniform and skewed distributions on the same lease. Collect
per-rank step time, communication time, overlap time and expert load. Rebalance
with the production EPLB path; do not drop experts, change the model, or widen
the lease to hide a straggler. Keep a uniform-distribution guard workload.

Acceptance requires: (1) logits and routing decisions remain equivalent to the
Eager reference within tolerance; (2) `rank_skew_ratio =
max_rank_step_ms / median_rank_step_ms` stays below the gate; (3) `comm_ratio`
and `overlap_ratio` meet the declared targets after EPLB; (4) the uniform guard
workload does not regress; and (5) every rank stays inside the workspace budget.
Report the skew input, EPLB decision, per-rank timings and the guard result.

## Runtime prerequisites and current status

The verifier consumes `MINDIE_BENCH_WORKSPACE` and ignores any evidence JSON in
it. It requires a real Qwen3-MoE snapshot via `MINDIE_BENCH_QWEN3_MOE_MODEL` or
`MINDIE_BENCH_MODEL_PATH`, eight leased Dies visible in
`ASCEND_RT_VISIBLE_DEVICES`, and importable `torch_npu`/`vllm`. The real workload
is an expert-parallel server plus skewed and uniform prompt sets:

```bash
vllm serve "$MINDIE_BENCH_QWEN3_MOE_MODEL" --host 127.0.0.1 --port 18000 \
  --tensor-parallel-size 2 --data-parallel-size 4 --enable-expert-parallel \
  --trust-remote-code
```

Without the model and a bounded skew client the verifier exits nonzero with
`unsupported:model_cache_unavailable` or
`unsupported:task_blocked_real_workload_harness_incomplete`; no synthetic rank-skew or EPLB
metric is produced.

## Blocked / incomplete

This task has no production acceptance path yet. The verifier preserves the
missing model/serving prerequisite as an explicit report; it must not be
promoted, described as available, or counted as a completed benchmark.
