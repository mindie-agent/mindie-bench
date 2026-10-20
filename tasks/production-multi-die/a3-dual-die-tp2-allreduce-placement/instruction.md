# A3 same-package dual-Die TP=2 AllReduce placement

You are given one isolated A3 sub-cluster of **2 Dies** on a single NPU package
(`Phy-ID 0,1` or the next free sibling pair). The production node runs a
`TP=2` worker with `MatmulAllReduce` / MC2 fused communication. A previous
deployment picked Phy-IDs `(1,2)`, which straddles two NPU packages: the
all-reduce falls back from the sibling-Die HCCS path and its bandwidth/latency
diverges from the same-package baseline.

Reproduce the placement difference on the assigned lease. Keep
`ASCEND_RT_VISIBLE_DEVICES` restricted to the allocated sibling pair, run the
real TP=2 fused `MatmulAllReduce` microbenchmark, and compare it with an
explicit cross-package `(1,2)` control that is measured only as a degraded
fallback. Do not reset the server, kill peer jobs, or widen the lease.

Acceptance requires: (1) the same-package `TP=2` run is numerically equivalent
to a single-Die eager reference within the declared tolerance; (2) the
`MatmulAllReduce` latency and throughput meet the same-package target; (3) the
cross-package control is either rejected by the placement guard or clearly
reported as `cross_package_degradation` with its measured penalty; and (4) no
rank leaves residual `/dev/shm/hccl*` state after the run.
