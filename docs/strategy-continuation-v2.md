# Strategy continuation v2 — receipt-owned task outcomes

Target design only. **Nothing below is implemented yet, and nothing below constitutes model or NPU acceptance.** No new runner and no receipt hash, signature, or attestation chain is introduced.

## Source status today

`evaluation/strategy_selection.py` judges real, isolated Arm B / Arm C selector decisions. That half
is sound. The continuation half is not an outcome owner: `evaluate_strategy_continuation` reads
`verifier_passed` from the executor's returned mapping, so whoever supplies `executor_fn` also
supplies the task verdict that feeds `pass_rate`, Cost-of-Pass and the Pareto frontier.

This is a contract gap, not an exploit: `executor_fn` is a trusted in-process callable with a
documented contract, and the shipped default supplies none at all. The problem is that the pilot has
no independent owner for the outcome, so its continuation numbers cannot be acceptance evidence.

What the current source therefore does:

- the selector-only default is unchanged (`continuation = "not_executed"`) and reports no task outcome at all;
- an executed pilot sets `continuation_verdict_basis =
  executor_self_reported_verifier_passed_unverified`;
- `decision_gate` returns `continuation_unverified` instead of a rollout verdict, so the A/B/C
  comparisons stay diagnostic and cannot authorize `continue_expanding_families`;
- a continuation timeout records `terminal_status = "unknown"` and any other continuation exception
  records `terminal_status = "infra"`, preserving the selector decision, its spend, and the measured
  elapsed overrun without voiding other cells;
- device time (`npu_card_hours`, `npu_die_hours` via `device_time_accounting`) comes only from
  executor-measured `npu_occupancy_sec` paired with executor-observed `npu_cards` / `npu_dies`
  (`npu_accounting_basis`), never from the case's planned device counts.

## Target owner boundary

`run_trial` already owns an attempt, its verdict and its cost, and `runner/product_executor.py` is
the existing precedent for binding a planned cell to it. v2 reuses both:

- `runner/` owns execution. A continuation executor binds one case + arm to a `TaskSpec` and
  `ExperimentConfig`, calls `run_trial`, and hands back the resulting `TrialReceipt`.
- `evaluation/` keeps judgement only: `selection_acceptable`, `false_adoption`,
  `selection_reasonable`. It stops deciding whether the task succeeded.
- Cost mapping reuses `runner/matrix.py::observation_from_receipt`, which already yields wall clock
  from `delivery_wall_clock_sec`, `model_calls`, tokens, cost (only when `cost_complete`) and
  `npu_runs`. The pilot consumes that instead of keeping its own ledger.

`task_succeeded` becomes `receipt.terminal_status is TerminalStatus.PASS and
receipt.primary_metrics["reward"] == 1.0`; `infra_failed` / `unknown` map to the existing
`AblationOutcome.INFRA` / `UNKNOWN` the cell layer now produces.

The value is that `run_trial` runs the oracle the pilot does not control. A typed receipt handed in
by a caller is no more trustworthy than today's boolean, so **no** receipt hash, signature or
attestation chain is added. `receipt.trial_id` plus `artifacts["private_trial_dir"]` remain the only
identity, exactly as the product matrix already does it.

## Remaining mapping gap

Each row is a decision someone has to make, not a shim to write.

| Gap | `StrategyCase` has | `run_trial` needs |
|---|---|---|
| Task root | `problem_statement` prose | on-disk `task.toml`, `instruction.md`, `verifier/test.sh` |
| Source pin | nothing | `SourceEnvironment.repo_path` + 40-hex `base_commit`; `WorktreeSlot` needs a real git worktree |
| Topology | `hardware_tier`, `topology_tier`, planned `npu_cards`/`npu_dies` | `topology_scale` + `placement` + `die_count` declared together, plus `visible_phy_ids` |
| Oracle | `private_judge` mapping of acceptable/forbidden refs | an executable oracle with exit codes, frozen by `oracle_freezer` |
| Soundness | `reference_solution` / `known_bad_solution` mappings | real patches proving `before_reward=0`, `after_reward=1` for `evidence="verified"` |
| Corpus | `corpus_roots` → ad-hoc `RoutingIndex` | `CorpusConfig` (path + 40-hex commit) and a group `knowledge_mode` |
| Config | none | an `ExperimentConfig` with a harness, provider endpoint and `component_revisions` |
| Adoption | executor-declared `applied_strategy_ref` | observation from `Trajectory.calls` / broker `explain` targets; no unit-id → strategy-ref mapping exists |
| Device time | planned counts | observed counts; `gates.TelemetryEvidence.devices` via `telemetry_from_c2` is the natural source |
| Budget | `StrategySelectionBudget` | `Budget.timeout_sec` / `verifier_timeout_sec` and `experiment_context["max_wall_clock_sec"]`, split by the existing `product_execution_window` + `lifecycle_overrun` |

The oracle and soundness rows block everything else: a decision point is not an executable task.
Until a pilot family has a real failing-to-passing task, no receipt can own its outcome, and
inventing one to satisfy the new signature would reproduce today's defect behind a stricter type.

## Phases

1. **Contain and label** — done in this change: per-cell infra/unknown, observed-only device time,
   `continuation_unverified` gate, selector-only default untouched.
2. **Author one real continuation task** for a single pilot family: task root, pinned source,
   executable oracle, and a genuine `soundness_receipt.json` from the existing `mindie-bench
   soundness` runner. No pilot code changes.
3. **Move the verdict to the receipt** once that task exists, and delete `verifier_passed` with the
   legacy dict path rather than keeping two outcome sources.
4. **Add the `run_trial`-backed executor** behind an opt-in CLI flag; the default stays selector-only.
5. **Observe adoption** from the trajectory and retire executor-declared `applied_strategy_ref`.

## Validation

- **Local, CPU.** Phases 1 and 3 are checkable with `backend="fixture"` at tier `l2`: a receipt with
  `reward=0.0` must override a `verifier_passed=True` claim; infra and unknown must reach the
  ablation denominators; budgets must satisfy `outer_timeout_sec >= selector_timeout_sec +
  business_window_sec + verifier_timeout_sec + cleanup_grace_sec`. This proves plumbing only and says
  nothing about Ascend behaviour.
- **Live.** Only a tier `l3` cell on the standing A2 container, with a real `verifier/test.sh` and
  npu-top C2 telemetry supplying the device window, can produce publishable `npu_die_hours` or any
  claim of model/NPU benefit.
- **Gate.** `continuation_unverified` holds until a receipt owns the verdict. A rollout gate returns
  only when at least `min_independent_families=3` families × 2 repetitions × 3 arms have settled
  receipts; fixture runs never move it.

## Known boundary

The pilot holds results in memory and writes them once at the end. Ordinary executor faults no
longer lose the other cells, but an interrupt or a crash outside the executor still discards the run.
No per-cell journal is added for that; it stays an honest limitation until someone needs durability.
