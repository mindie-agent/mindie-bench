# Strategy continuation - independent local verifier

The selector-only default remains unchanged: `continuation="not_executed"`, no
continuation outcome, and `None` for unmeasured continuation metrics. The explicit
`StrategyContinuationExecutor` now runs a real local command and an independent
behavior verifier. This is Tier 1 execution plumbing, not evidence of model or
Ascend performance benefit.

## Callable API

```python
from pathlib import Path
from mindie_bench.models import SourceEnvironment
from mindie_bench.evaluation.strategy_selection import (
    StrategyContinuationExecutor, StrategyContinuationTask,
    run_strategy_selection_pilot,
)

# case is an independently authored StrategyCase. Its source pin contains the
# public decision-point workspace; task_root keeps the private behavior oracle.
executor = StrategyContinuationExecutor(
    tasks={case.case_id: StrategyContinuationTask(
        task_root=Path("/tasks/my-continuation"),
        source=SourceEnvironment(repo_path="/sources/module", base_commit=commit),
    )},
    command=[python, "-I", "/adapters/continue_agent.py"],
    read_paths=[Path("/adapters/continue_agent.py")],
    run_root=Path("/results/strategy-continuations"),
    verifier_timeout_sec=120,
    knowledge_token_budget=4096,
)
report = run_strategy_selection_pilot(
    corpus_roots=corpora, cases=[case], repetitions=2, executor_fn=executor,
)
```

The adapter receives one JSON document on stdin and edits its current workspace.
The context includes the public task, decision-point observations, selected
strategy reference, complete selected body and dependency closure, and remaining
execution/model-call budgets. Arm A receives the same task observations with no
selected card. B/C selection isolation remains unchanged. A deliberately mistaken
selection can still be executed for negative-transfer comparison; loading a card
is not evidence of adoption. The command's stdout is retained as a log, never
trusted as a task verdict or a model-usage receipt.

`task_root/verifier/test.sh` launches the independently authored `verify.py` using
`MINDIE_BENCH_ORACLE_ROOT` and `MINDIE_BENCH_WORKSPACE`. It returns the normal
verifier JSON/exit-code contract. `WorktreeSlot` starts each arm from the same
exact source commit. The existing native sandbox hides the source repository and
private oracle from the foreground command; broad read grants crossing either
are rejected. The existing oracle freezer runs before foreground execution and
`local_verifier` checks it before and after the behavior test. No new verdict
attestation or shadow execution state machine is introduced.

## Report semantics

- `continuation_completed` records a successful foreground process exit; an agent
  crash can still leave a behaviorally correct candidate. Its independent
  `continuation_passed` remains true, while completion remains false.
- `continuation_passed` comes only from the independent verifier. Selector labels
  such as `false_adoption` cannot overwrite a real task pass.
- `continuation_regressed` compares actual A and B/C verdicts for the same case
  and repetition: A passed and the selected continuation failed. Missing either
  verdict leaves regression `None`; failure by itself is not a regression.
- `selected_and_completed_rate` counts successful foreground completion plus a
  verifier pass among non-default selections with an observed verifier verdict.
  `regressed_rate` counts regressions among those selections with an observed A
  comparison. Both ratios expose numerator, denominator and missing counts.
- `continuation_evidence` points to the retained JSON context, command logs,
  final patch and verifier record. Each run has a distinct workspace. Cleanup
  and patch-accounting faults remain visible without erasing a known verdict.
- Model tokens, model calls, NPU runs and device time stay `None` because the local
  command adapter does not independently meter them. Costs are never copied from
  the case's declared shared-prefix estimates or a command's stdout claims.

The concrete executor's verifier and knowledge budgets must fit the pilot's
budgets. Source preparation is charged against the remaining execution window.
The runner fixes absolute business, verifier and outer deadlines at entry.
`WorktreeSlot` receives the current phase deadline and caps every Git subprocess
and metadata-lock wait by the remaining phase allowance. Agent cleanup reduces
the remaining verifier allowance, and the verifier's possible process drain
reserves another 5 seconds before the verifier phase boundary. The outer
deadline reserves the pilot's cleanup grace. Patch capture reserves half that grace for worktree removal,
which is attempted even when patch accounting fails. No phase restarts a fresh
full Git timeout. Actual elapsed cost and deadline overruns remain reported.

## Remaining native acceptance work

Legacy `executor_fn` callables returning dictionaries remain explicitly marked
`executor_self_reported_verifier_passed_unverified`; their claims never populate
the new independently verified completion/regression metrics. Real local runs
are marked `independent_local_verifier`. Neither path authorizes a rollout:
local execution returns `local_behavior_only` once lineage readiness holds.

A real model/NPU study still needs independently authored continuation tasks,
source pins, behavior-oracle soundness, matched harness configuration, native
usage and observed adoption/device occupancy. Such a study should bind the
existing `run_trial` receipts and `observation_from_receipt` cost mapping. CPU
sandbox tests do not establish any Ascend speedup or cross-model transfer.
The pilot still accumulates its finite report in memory; an outside crash before
report publication can lose the aggregate, though executor artifacts remain.
