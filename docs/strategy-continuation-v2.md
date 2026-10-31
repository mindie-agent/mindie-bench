# Strategy Continuation & Independent Local Verifier

By default, strategy selection evaluation runs only the selector step (`continuation="not_executed"`), leaving continuation outcomes and unmeasured continuation metrics as `None`. When you want to verify whether a selected strategy card actually leads to a working fix in a local workspace, pass `StrategyContinuationExecutor` to `run_strategy_selection_pilot`.

## Usage

```python
from pathlib import Path
from mindie_bench.models import SourceEnvironment
from mindie_bench.evaluation.strategy_selection import (
    StrategyContinuationExecutor,
    StrategyContinuationTask,
    run_strategy_selection_pilot,
)

executor = StrategyContinuationExecutor(
    tasks={
        case.case_id: StrategyContinuationTask(
            task_root=Path("/tasks/my-continuation"),
            source=SourceEnvironment(repo_path="/sources/module", base_commit=commit),
        )
    },
    command=[python, "-I", "/adapters/continue_agent.py"],
    read_paths=[Path("/adapters/continue_agent.py")],
    run_root=Path("/results/strategy-continuations"),
    verifier_timeout_sec=120,
    knowledge_token_budget=4096,
)
report = run_strategy_selection_pilot(
    corpus_roots=corpora,
    cases=[case],
    repetitions=2,
    executor_fn=executor,
)
```

### How It Works

1. **Workspace & Input Isolation**:
   - Each arm starts from an isolated `WorktreeSlot` checked out at `base_commit`.
   - The adapter receives a JSON payload on `stdin` containing the task description, decision-point observations, the selected strategy card (body + dependency closure for Arms B/C; omitted for Arm A), and remaining execution/model-call budgets.
   - The native sandbox hides the source repository and private oracle directory from the foreground command.
2. **Independent Verifier**:
   - After the foreground command exits, `task_root/verifier/test.sh` runs `verify.py` against the workspace (`MINDIE_BENCH_WORKSPACE`) and private oracle (`MINDIE_BENCH_ORACLE_ROOT`).
   - Oracle files are frozen before execution and checked before and after verification.

## Report Fields

- `continuation_completed`: `True` if the foreground adapter exited cleanly (`returncode == 0`).
- `continuation_passed`: Determined solely by the independent verifier (`verify.py`), regardless of whether the foreground process crashed after writing a valid fix or what the selector label was.
- `continuation_regressed`: `True` when Arm A passed verification for the same case and repetition while the selected continuation (Arm B/C) failed. `None` if either verdict is missing.
- `selected_and_completed_rate` & `regressed_rate`: Aggregate ratios (with explicit numerator, denominator, and missing counts) over non-default selections that have an observed verifier verdict.
- `continuation_evidence`: Paths to the saved context JSON, command stdout/stderr logs, final git diff, and verifier output.
- **Unmeasured Hardware Metrics Stay `None`**: Model tokens, model calls, NPU runs, and device time remain `None` unless supplied by a native harness trajectory, rather than being copied from static prefix estimates or adapter stdout.

## Attaching Native Trajectory & Broker Observations

When wrapping a real `HarnessDriver` or `run_trial`, attach the native `Trajectory` and `summarize_broker_knowledge` output to `ContinuationExecution`:

```python
from dataclasses import replace
from mindie_bench.evaluation.broker_knowledge import summarize_broker_knowledge

broker_summary = summarize_broker_knowledge(
    trial_dir / "broker.jsonl",
    process_origin_monotonic_sec=process_started_monotonic_sec,
    preflight_finished_monotonic_sec=preflight_finished_monotonic_sec,
)
execution = replace(
    execution,
    trajectory=driver.trajectory(),
    broker_knowledge=broker_summary,
)
```

Each cell records `adoption_observation`:
- `loaded_strategy_refs`: Populated only from successful `mindie_knowledge_explain` responses that returned a non-empty unit body with a pinned revision (post-preflight).
- `declared_adoption`: Populated when `mindie_knowledge_feedback` records `event_type="applied"` with a singular `ref`. `applied_in_actions` remains `None` because neither loading nor feedback alone proves behavioral adoption—that is measured by `continuation_passed`.
- `continuation_verdict_basis`: Distinguishes `independent_local_verifier` (local command + verifier), `independent_verifier` (typed native callback), and `executor_self_reported_verifier_passed_unverified` (legacy dict callback).
