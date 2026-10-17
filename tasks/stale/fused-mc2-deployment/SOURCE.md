# Provenance and stale-knowledge relationship

The runtime source is the complete vLLM-Ascend module tree at
`c3394206bdd00b132580ae439555b4f1a70d5bf6`, running against the existing
vLLM `0.27.1+empty` environment. No upstream runtime function is replaced or
extracted for this task. `deployment.py` is a benchmark-constructed launcher
fixture with a faulty compatibility override, not a historical source patch.

The actual retained corpus package is
`adc00ab1ad0d5b4d05cecca47bdcef318d9eda10405d667b6e976d21bf59a11e`,
block `05f4d9d87d7c0f07595dfc310d32e2408af0dd9cd2984796a21d99bab95e082d`.
It recommends `VLLM_ASCEND_ENABLE_FUSED_MC2=0` as a workaround for the affected
historical optional fused path. That block explicitly discusses v0.23.0,
whose environment module is at
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`. The current configuration migration
is represented by upstream commit
`80c833fb86b105391d164c9e3fc60d0bfecbf85e`.

The stale relationship is established by current execution: the fixture's
old environment override leaves the current configured flag enabled and
real BF16 fused MoE operators execute. Updating the current configuration
disables that path without bypassing model arithmetic or HCCL. The historical
block is retained as potentially misleading context, not excluded as a
same-origin answer. The private reference uses current configuration and
compares complete tiny-model outputs on two input cases and two ranks.

The operator-level prototype used the same existing image and both real
paths; it observed finite nonzero outputs and zero maximum absolute difference
for its 64-token case. The canonical task has its own independent
fail-before/pass-after proof; prototype output is not its proof.
