# Design and research review

The input was `mindie_agent_benchmark_strategy.md`, reviewed on 2026-10-16.
The product design principles, architecture, harness lifecycle, and native
runtime reuse documents in the parent `mindie-agent` checkout are authoritative.
This independent repository contains evaluation tooling, not a new business
workspace manager, plugin, service, or mandatory user workflow.

## Adopted scope

- L0: deterministic contracts and component observations, with no model or NPU.
- L1: bounded native-harness tasks, including external trajectory graders.
- L2: fresh source worktrees, CPU replay feedback, and an external final judge.
- L3: SSH into an existing dedicated Linux container, with per-device leases,
  private worktrees, a persistent isolated worker, and memory/process readback.
- G0: native harness; G1: remote tools; G2: remote tools plus filtered public
  knowledge; G3: independent seed/capture/eval sequences with local knowledge.
- Each trial has one immutable terminal receipt. There are no automatic retries,
  daemon, external telemetry backend, or automatic GitHub/Hugging Face writes.

Completion and total cost are primary. Routing, retrieval, tool errors, and hook
or schema checks are diagnostics. Fast component tests do not establish product
usefulness or hardware correctness.

## Corrections to the proposal

1. An isolated HOME does not isolate filesystem access. The agent also runs
   under a default-deny macOS Seatbelt profile or a Linux bubblewrap namespace.
   Solutions, verifier code, private journals, source Git history, and production
   profiles are outside its read capabilities. User runtime read grants that
   expose these protected roots are rejected. Snapshots have no `.git` pointer.
2. A PATH shim cannot stop HTTP requests or absolute-path executables. Network
   access is restricted by the OS. A parent broker permits only configured model
   inference routes, replay, and the assigned remote slot. It holds real model
   credentials; the agent gets a dummy broker credential. No route to GitHub's
   REST/GraphQL APIs is exposed. Native profile hooks are disabled; G3 capture is
   an explicitly configured local adapter, outside the foreground agent. Optional
   indexer inference has its own pinned model, call cap and provider usage ledger.
3. Replay rules never enter an agent environment variable or mounted file.
   A shim sends a command to the parent, which evaluates external rules and
   returns only test feedback. A text-pattern replay is a surrogate, not proof
   of hardware semantics; the final external verifier remains independent.
4. Harbor's current native layout uses `solution/solve.sh` and `tests/test.sh`.
   This implementation deliberately uses `oracle/` and `verifier/` with MindIE's
   own versioned schema. A similar directory layout is not seamless Harbor
   compatibility. No Harbor integration has been accepted.
5. The real public knowledge feed uses `index.md` / `blocks/*.md`, whereas the
   proposal describes local `PACKAGE.md` / `materials/*.md`. Both explicit
   formats are recognized. Published block inventories, IDs, and SHA-256 values
   are checked. Unknown layouts and corruption raise errors. This is a
   structural reader, not a substitute for upstream's full publication review.
6. A manifest's `complete` status is not evidence that a historical engineering
   task succeeded. The first-party material schema also recognizes `pending`,
   `incomplete`, `failed`, and `uncertain`. Candidate records preserve this
   source status and list missing acceptance evidence. Mining never certifies a
   historical narrative as a verified task.
7. G3 needs temporal evidence, not just a preloaded warm snapshot. Each
   repetition/cluster starts with a new physically filtered corpus; all selected
   evaluation origins are excluded initially. Only actual seed captures flow
   into dependent evaluations. Different clusters and repetitions do not share
   generated experience. A capture failure stops dependent trials.
8. Frozen baselines depend on exact harness version/binary, model identifier,
   endpoint, budget, source commit, task digest, platform, and shared component
   revisions. They are not reusable forever or across a major-version label
   alone. Hosted model aliases can still drift; use pinned deployment identities.
9. `(c/n)^k` is a plug-in reliability estimate, not observed consecutive success.
   We report it per task only when at least `k` valid repetitions exist, and also
   report the combinatorial estimator `choose(c,k)/choose(n,k)`. Cubing a mixture
   of easy and hard tasks would distort reliability. Neither estimate is a
   guarantee of future performance.
10. Tool durations can overlap, and not every harness exposes both boundaries.
    The requested sums are retained; the derived LLM/exploration remainder is
    null when missing or negative. Missing localization is penalized by the
    task's max-turn budget in comparison. Missing metering is not zero cost.
11. Verifier exit 1 alone cannot distinguish a wrong patch from a broken test
    environment. A nonzero result requires an explicit JSON `reward: 0`; a
    nonzero result without that contract is infrastructure failure. Contradictory
    exit status/reward is also an infrastructure failure.
12. "Compositional reuse" and "no relevant knowledge" are different experiments.
    The proposal's seed blueprint used tier 3 for both. The schema gives them
    separate `compositional` and `no_hit` labels.
13. The proposal's time, cost, sample-count, success-rate and improvement numbers
    are estimates or targets. None are substituted for observations. There is
    no claim that all public packages represent successful NPU tasks.

## Receipts and failure semantics

`completed_pass` requires a clean native completion and passing external judge.
`completed_fail` is a completed wrong answer, or a deliberate turn/test/time cap
with a valid external judge. A capped trial cannot earn a pass; its actual
verifier result is preserved separately.

`infra_failed` covers preparation, required component initialization, verifier,
and required postflight failures. `unknown` covers uncertain model/remote
effects, malformed/incomplete native evidence, and native crashes after launch.
An unfinished `attempt_started.json` without a terminal receipt blocks scoring;
it cannot disappear from a denominator or trigger automatic resubmission.

If cleanup fails after a business operation, both facts are retained.
Local artifact cleanup errors do not overwrite the business verdict. Failed
required remote postflight produces an infrastructure receipt with the original
business status preserved. CLI exit 2 surfaces any infrastructure/unknown state
or recorded maintenance failure.

Reports show all four terminal counts and coverage alongside valid-attempt
Pass@1. They withhold cost per resolved task when any attempt's metering is
incomplete. Comparison requires matched task/tier/repetition cells, computes
completion lift only on mutually valid pairs, and measures time speedup only
on mutually solved pairs. It reports the number of such pairs.

`total_wall_clock_sec` retains the proposal's prompt-to-agent-exit measurement.
`delivery_wall_clock_sec` additionally includes preparation, judging, capture
and cleanup; delivery speedup uses this complete measurement. Provider SSE usage
is metered in the parent, including requests initiated by component tools, so
their cost cannot disappear from the foreground CLI's narrower usage summary.
G3 seed outcomes are separate from evaluation scores; the report also exposes
cost per resolved evaluation including seed setup.

## Process and hardware boundaries

macOS uses the product's owner-pipe/process-group pattern. Linux uses
bubblewrap's PID/network namespaces and owner-death handling. Python ownership
helpers run with `-I`, so a business file named `operator.py` or `json.py`
cannot execute before the sandbox by shadowing a standard-library module.
Native Windows execution is rejected; scoring/mining are portable, but no
unverified Windows sandbox or Job Object substitute is advertised.

The remote worker is a single persistent PID namespace per trial, so remote-dev
jobs survive individual tool requests. The SSH facade supports one-shot commands
to `bench-slot`; tunnels, interactive SSH, SFTP and SCP are not implemented.
Namespace jobs are quiesced before final judging. Self-contained verifier code
is supplied only after foreground agent termination. Leases remain held through
remote teardown, exact slot-token cleanup, patch capture and memory readback.
Only this trial's worktree is removed; unrelated jobs and worktrees are untouched.

Remote NPU selection uses cooperative `flock` plus
`ASCEND_RT_VISIBLE_DEVICES`. This is not a hostile multi-tenant device-security
boundary: the benchmark requires a dedicated evaluation container with the
correct device mappings. It never resets a device or kills a process merely
because it uses an assigned NPU. Hardware/topology behavior still needs live
acceptance on the intended Ascend machine.

## Publication boundary

`export-hf` stages data locally. It excludes raw prompts, reasoning, commands,
MCP environments, exception text and private paths. The installed first-party
`mindie_knowledge` deterministic redactor is required, followed by a fresh
Gitleaks gate with repository allowlists disabled. The entire JSON payload is
scanned before Parquet encoding; Parquet is read back before atomic publication
to the requested local directory. No network upload is performed.

## Primary references checked

- Codex noninteractive JSONL:
  https://developers.openai.com/codex/noninteractive
- Harbor task structure:
  https://harborframework.com/docs/task-format
- Native CLI `--help` / isolated `--version` on the actual installed binaries.
- Public corpus `publication-contract.json`, `docs/package-format.md`, and the
  first-party `materials/store.py` validator for actual accepted statuses.

The survey's broader literature claims and headline percentages are not an
implemented acceptance criterion or an independently reproduced research result.
