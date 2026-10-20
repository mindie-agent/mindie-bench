# Qwen3.5 GDN DP=2 x TP=2 asymmetric batches under ACLGraph

Use the **4-Die intra-node HCCS lease** to run Qwen3.5 GDN in `DP=2, TP=2`
with piecewise ACLGraph capture. Production sends asymmetric decode batches:
`[B_max, 0]`, `[0, B_max]`, `[1, B_max]` and `[0, 0]`. When one DP rank has
no scheduled request, the empty rank must still join every DP collective. The
historical failure is an `HCCL_EXEC_TIMEOUT` on the idle rank followed by
`conv_state` / `ssm_state` corruption on the next request.

Reproduce with the four batch matrices against the allocated ranks. Compare
logits and per-rank recurrent-state checksums with the `TP=1, DP=1` Eager
reference, and compare piecewise ACLGraph against Eager for the same parallel
layout. Include a guard workload that exercises the active-rank path.

Acceptance requires: (1) no `HCCL_EXEC_TIMEOUT` and all ranks participate in
every matrix, including `[0, 0]`; (2) KL divergence and Top-1 agreement satisfy
the invariance gate; (3) idle ranks leave `conv_state` / `ssm_state` unchanged
while active ranks update normally; (4) warm-server TTFT and TPOT stay within
the declared SLA; and (5) no rank exceeds the workspace or guard-workload
budget.
