# A version-evidenced stale-knowledge configuration task

The sibling [fused-mc2-deployment](fused-mc2-deployment/SOURCE.md) adds the L3
business counterpart: the complete current tiny Qwen3-MoE runs on two physical
NPUs with TP2/EP2 and HCCL. Real operator traces distinguish the obsolete
environment override from the supported current configuration. Its deployment
fixture is explicitly constructed; the retained historical advice is real.
The sections below describe the separate CPU reporting migration.

`fused-mc2-report` is a real source-bound CPU configuration/reporting migration
task. It deliberately preserves the old knowledge body and checks its version
conditions against an executed current-source counterexample. Old source alone
is not the reason for the `stale` label.

The read public package is
`adc00ab1ad0d5b4d05cecca47bdcef318d9eda10405d667b6e976d21bf59a11e`,
block `05f4d9d87d7c0f07595dfc310d32e2408af0dd9cd2984796a21d99bab95e082d`,
from corpus commit `cec3249ccd75e7a73cf2f2f776117e0ec385795b`.
It explicitly discusses 0.23 and recommends the legacy environment knob as a
workaround. The recipe binds the actual read body hash and recommendation.
Its broader NPU bug narrative is not treated as independently verified here.

The executable applicability boundary is:

| Source revision | Complete module | Actual CPU observation |
| --- | --- | --- |
| Tag `v0.23.0`, `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` | `vllm_ascend/envs.py` | The old knob is recognized and its values 0/1 are read. |
| vLLM 0.27.1 configuration migration, `80c833fb86b105391d164c9e3fc60d0bfecbf85e` | `vllm_ascend/envs.py` | The same knob is no longer recognized for either value. |

Both complete, unmodified API modules are public counterexample assets and are
executed side by side. A current default may accidentally agree with an old
recommendation; that coincidence does not establish that the removed knob
still controls anything. This task does not assert a kernel failed because
someone set the old knob to zero.

The actual repair pair is
`e8600b7f6198fb6bd07116fa0a77792f9867f86d` to the migration commit above,
targeting the complete
`tests/e2e/nightly/multi_node/external_dp/scripts/utils.py`.
The historical reporting module still attributes an active feature to the old
input channel and omits current configured features. Its actual upstream fix
is supplied privately as a gold patch. The CPU judge imports the complete
reporting module, real `ServerCommand` class and all real dependency modules.
No AST extraction, replacement business implementation, model, NPU, network
publication or server launch is involved.

The public instruction identifies the current runtime contract, reporting
symptoms and engineering entry point. It does not publish the hidden judge or
gold patch. The judge checks 32 actual configuration/feature cases, including
both supported argument spellings, enabled/disabled settings, conflicting
legacy inputs, unrelated tags, deduplication and visible malformed-JSON errors.
Deleting the obsolete tag alone still fails the current-configuration cases.
Runtime/import failures remain infrastructure failures.

## Prepare a private runtime and revision

The runtime has only locked PyYAML and regex dependencies. It does not install
vLLM, torch or anything into an existing benchmark/model environment.

```sh
UV_PROJECT_ENVIRONMENT=/owned/stale-runtime/venv \
  uv sync --project tasks/stale/runtime --locked --no-dev

uv run python tasks/stale/materialize.py \
  --corpus /owned/fixed-public-corpus \
  --source-repo /owned/vllm-ascend-with-historical-objects \
  --python /owned/stale-runtime/venv/bin/python \
  --output /owned/new-stale-task-revision

uv run mindie-bench validate --tasks /owned/new-stale-task-revision/compiled
uv run mindie-bench soundness --tier l2 \
  --task /owned/new-stale-task-revision/compiled \
  --config /owned/explicit-local-config.toml \
  --output /owned/new-stale-proof --promote
```

The materializer verifies every tracked asset, complete original source blob,
actual knowledge quote/body and locked dependency versions before compiling a
candidate with the existing `compile_recipe` API. It does not execute soundness,
fetch source or inherit proof. Its `materialization.json` records the exact
Python/stdlib read paths needed by the existing verifier and runner. Give
those precise capabilities, rather than a task/report/source-parent directory.
Relocated or changed inputs require a new task identity and proof.

Only `environment/` is flattened into the source workspace; the instruction
correctly addresses source-root `version_probe.py` and `reproduce.py`.
The independent judge, gold and proof stay private.

## Keep the actual stale body available

The materializer freezes a derived `stale-knowledge-corpus` containing the
actual old package, unchanged, with its own Git commit and tree hash.
Configure the task's `stale-fused-mc2` corpus ID to that exact scope for a
controlled stale-input experiment. This selection is explicit, not an
exhaustive relevance judgement over all public packages.

The old package has the `stale` evidence relation and is **not** put in
`same_origin_task_hashes`; otherwise ordinary cold-profile filtering would
remove the very stale advice being evaluated. A bounded exact fix-SHA/PR scan
found no declared exact-origin package for the later reporting migration.
That scan does not prove absence of every possible semantic analogue.

This is a standalone eval task; it has no invented G3 seed. Natural and
explicit knowledge activation are both legitimate study protocols and must be
recorded separately. A source/gold proof establishes this CPU reporting
contract, not model benefit, true runtime feature execution, NPU precision,
HCCL or performance.

The tracked regression entry is:

```sh
MINDIE_BENCH_STALE_PYTHON=/owned/stale-runtime/venv/bin/python \
MINDIE_BENCH_STALE_SOURCE_REPO=/owned/vllm-ascend-with-historical-objects \
  uv run pytest -q tests/test_stale_config_task.py
```

Without those two explicit owned inputs, source/runtime-dependent controls are
marked skipped; the complete old/current API counterexample remains runnable.
