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
