# Real Ascend diagnostic replay with independent CPU grading

`sampler-topk` binds a saved actual NPU error to the complete historical
`vllm_ascend/sample/sampler.py` module. It adds replay-layer coverage for the
existing sampler defect family; it is not another independent L3 family or
an additional flagship task.

The source manifest records exact Git base/fix commits, full module and patch
hashes, original stderr bytes, the independently completed hardware proof hash
and its actual image/CANN/driver/torch_npu identities. The retained stderr came
from the first actual source probe whose judge initially misclassified the
uppercase NPU message as infrastructure failure. A subsequent fresh hardware
proof correctly observed source reward 0 and gold reward 1. Neither original
record is rewritten by this task.

The saved failure is `aclnnTopk` 161002 / EZ1001 with local width 8 and k=4096.
The public CPU counterpart imports the **whole original module**, including
its class/function definitions and fallback selection. It replaces unavailable
framework imports/configuration/base classes and a one-rank collective with
explicit fixtures. Tensor creation, top-k, sorting, softmax and masking execute
real CPU torch operations. Unsupported stream/AscendC paths fail visibly.
This does not emulate NPU kernels, streams, HCCL, timing or an entire serving
runtime, and it does not extract target functions with AST.

The existing `npu-replay-shim` remains a thin HTTP client to the parent gateway.
The parent holds `oracle/replay_rules.json` and returns the saved real stderr
when the historical unsafe source-call pattern remains. That intentionally
limited source-pattern replay is **diagnostic only**. The external hidden
judge imports its own immutable boundary helper, executes the actual source
on real CPU tensors and verifies candidate/index correspondence, candidate-set
completeness, filtering and non-reduce behavior. It never grades a gold string,
the replay exit status or the public helper. A formatting-only source change
that fools the replay rule must still fail the tensor judge.

## Reproduce

Prepare the separate locked torch CPU runtime using
[`../cpu/README.md`](../cpu/README.md); do not install torch into a shared
benchmark/model environment. Then:

```sh
uv run python tasks/replay/materialize.py \
  --source-repo /owned/vllm-ascend-with-historical-objects \
  --runtime-manifest /owned/cpu-runtime/runtime-manifest.json \
  --output /owned/new-replay-task-revision

uv run mindie-bench validate \
  --tasks /owned/new-replay-task-revision/replay-sampler-topk

uv run mindie-bench soundness --tier l2 \
  --task /owned/new-replay-task-revision/replay-sampler-topk \
  --config /owned/explicit-local-config.toml \
  --output /owned/new-replay-proof --promote
```

The source repository must already contain the exact base and fix. The
materializer compares its base Git blob with the tracked full source snapshot
and verifies the frozen CPU runtime lock and binary. It performs no fetch,
installation, model call, verifier execution or proof inheritance. Give the
local verifier read access only to the explicit CPU venv/interpreter and its
stdlib, alongside the existing benchmark grants.

Only `environment/` is flattened into the agent's workspace. Gold, the external
judge, replay rules, proof and source provenance stay private. Never grant the
task parent, fixed source checkout or operator report root to the foreground.
Changed bytes/runtime/paths require a fresh materialization and soundness.

Use the existing runner to install the declared `npu-replay-sampler` executable
into a fresh L2 profile. A zero-model gateway/shim control can also exercise
the same real command without claiming a native model or Stop lifecycle.
Replay budget counts are diagnostic command attempts, not physical kernels.

The tracked regression controls are executable without a model or NPU:

```sh
MINDIE_BENCH_CPU_PYTHON=/owned/cpu-runtime/venv/bin/python \
  uv run pytest -q tests/test_npu_replay_task.py
```

These controls check the whole source module before/after its actual patch,
the saved-error hash, the existing installed shim, a source-format bypass,
an edited public reproduction/helper and a changed runtime identity. The
semantic judge covers 37 tensor/candidate/filtering cases; its terminal reward
uses the existing verifier schema. Without the separately selected CPU
interpreter, only tensor-runtime-dependent controls are explicitly skipped.

`no_hit` applies only when the operator freezes and actually mounts an empty
`replay-empty` corpus. No claim about absence of semantic analogues across the
full corpus is made. There is no evidenced stale-version recommendation for
this task, and no stale label is fabricated. The corpus-first L2 protocol suite
and this hardware-grounded diagnostic replay remain distinct scopes.
