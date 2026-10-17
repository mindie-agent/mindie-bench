# Design and research review

The input was `mindie_agent_benchmark_strategy.md`, reviewed on 2026-10-16.
The product design principles, architecture, harness lifecycle, and native
runtime reuse documents in the parent `mindie-agent` checkout are authoritative.
This independent repository contains evaluation tooling, not a new business
workspace manager, plugin, service, or mandatory user workflow.

## Adopted scope

- L0: deterministic contracts and component observations, with no model or NPU.
- L1: bounded native-harness tasks, including external trajectory graders.
- L2: fresh source worktrees with actual CPU execution or explicitly labelled
  replay feedback, and an external final judge. Real CPU tasks have their own
  locked runtime, task identity and source-fail/gold-pass proof.
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

1. An isolated HOME does not isolate filesystem access. The local agent runs
   under a default-deny macOS Seatbelt profile or a Linux bubblewrap namespace.
   Solutions, verifier code, private journals, source Git history, and production
   profiles are outside its read capabilities. User runtime read grants that
   expose these protected roots are rejected. Snapshots have no `.git` pointer.
   Root native-client configuration is removed only from owned snapshots:
   `.claude`, `.codex`, `.kimi`, `.gemini`, `.agents`, and `.mcp.json`.
   `AGENTS.md` and the source repository remain intact; final patches exclude
   projection deletions and fingerprints record the projection rules.
   Fresh Codex profiles use `project_root_markers = []` so instruction discovery
   retains the workspace's `AGENTS.md` without walking into private artifact
   ancestors. This keeps the filesystem access policy intact.
2. A PATH shim cannot stop HTTP requests or absolute-path executables. Network
   access is restricted by the OS. A parent broker permits only configured model
   inference routes, replay, and the assigned remote slot. It holds real model
   credentials; the agent gets a dummy broker credential. No route to GitHub's
   REST/GraphQL APIs is exposed. Personal hooks are never copied. G3 Codex alone
   enables a native Stop recorder in its fresh profile; capture remains an
   explicitly configured local adapter outside the foreground agent. It requires
   the actual thread identity, transcript and matching Stop event. Optional
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
   repetition/cluster/corpus lineage starts with a new physically filtered
   corpus; all selected evaluation origins are excluded initially. Only actual
   seed captures flow into dependent evaluations. A completed failed seed can
   contribute failure evidence if capture completes. Unknown seed effects or
   failed capture stop dependent trials. Held-out evaluations never advance the
   warm corpus, even when their own capture succeeds.
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
    null when missing or negative. Localization recognizes typed source access;
    shell-only reads or edits may be unobserved. Missing localization retains
    the task's max-turn penalty in the descriptive
    `localization_penalized_turns_ratio`. Comparison reports
    `localization_observed_pairs` and emits `localization_speedup` only when
    both sides are observed for every matched valid pair. Unobserved source
    access is not evidence that an agent failed to locate the defect.
    Missing metering is not zero cost.
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
14. NPU run counts use recognized test-command patterns. They measure attempts
    such as `pytest` or `torchrun`, not kernel launches or complete device
    utilization. A task verifier must independently establish NPU correctness.
    All remote admission paths inspect the command field. Stdin and uploaded
    source/patch bodies are opaque input; mentioning `pytest` inside a patch
    must not consume a test attempt. Native-reported attempts and gateway
    admissions are distinct observations, and neither certifies completion.
15. Privacy redaction can also destroy data identity. Public exports first
    replace native trial identifiers with deterministic SHA-256 IDs, then run
    the unchanged first-party redactor. The cleaned rows must retain distinct
    IDs, row counts and core task/outcome identity before publication staging.
    A Gitleaks pass alone cannot certify a usable dataset. Raw identifiers are
    neither restored after redaction nor included as a mapping in the export.

## Receipts and failure semantics

`completed_pass` requires a clean native completion and passing external judge.
`completed_fail` is a completed wrong answer, or a deliberate turn/test/time cap
with a valid external judge. A capped trial cannot earn a pass; its actual
verifier result is preserved separately.

`infra_failed` covers preparation, required component initialization, verifier,
and required postflight failures. `unknown` covers uncertain model/remote
effects, malformed/incomplete native evidence, and native crashes after launch.
Uncertain effects take precedence over a deliberate budget stop. The broker
seals model admission and allows up to ten seconds for already admitted provider
requests to settle. If the native client disconnects, delivery failure is
recorded separately while the same upstream response is drained and metered,
also with a ten-second bound. Missing usage, truncated provider output or an
expired settlement remains unknown. Socket shutdown cannot implicitly reconnect
and submit the request again. The grace period does not initiate inference.
An unfinished `attempt_started.json` without a terminal receipt blocks scoring;
it cannot disappear from a denominator or trigger automatic resubmission.
After an uncertain remote operation, further remote commands are blocked. Late
business results are retained separately and cannot stand in for the required
cleanup receipt.

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
cost per resolved evaluation including seed setup, observed tokens including
seed setup, and delivery seconds including seed setup. These totals retain
their own metering/coverage flags. Capture has a separate wall-clock observation;
it is not folded into the foreground agent's exploration remainder.
Comparison retains full per-group inputs and applies `role="eval"` only to
matched performance outcomes. Required G3 seed resources are added separately;
G0 seed-role work is not setup. Missing seed receipts or incomplete/unknown
capture withhold resource completeness, and absent dollar prices stay null.
Seed coverage uses the recorded cluster/repetition; a complete experiment
manifest or frozen receipt inventory must still detect missing individual
seed cells. Receipts do not record a corpus ID or expected seed-task inventory.

G0/G1/G2 cells and independent G3 lineages share one thread pool, cancellation
signal and NPU allocation pool. Work is interleaved at submission. Within each
lineage, seed execution, capture and dependent evaluation remain ordered.
Receipts are retained as cells finish; a failed required seed capture leaves a
partial summary and raises an error rather than making an incomplete matrix
appear complete. Scoring still rejects missing matrix cells.

## Process and hardware boundaries

macOS uses the product's owner-pipe/process-group pattern. Linux uses
bubblewrap's PID/network namespaces and owner-death handling. Python ownership
helpers run with `-I`, so a business file named `operator.py` or `json.py`
cannot execute before the sandbox by shadowing a standard-library module.
The macOS policy supports native PTY allocation and termios through the
`com.apple.sandbox.pty` extension: ptmx and extension-owned slave terminals
receive read/write/ioctl permission. It does not grant blanket access to
preexisting slave terminals. Real OS regressions preserve denials for an owned
preexisting PTY, an outside secret and unauthorized loopback.
Native Windows execution is rejected; scoring/mining are portable, but no
unverified Windows sandbox or Job Object substitute is advertised.

The remote worker is a single persistent PID namespace per trial, so remote-dev
jobs survive individual tool requests. The SSH facade supports one-shot commands
and persistent noninteractive stdio RPC streams to `bench-slot`. Opening a stream
does not wait for stdin EOF. A bounded input queue and background pipe writer let
the worker drain stdout/stderr with nonblocking select while input is pending.
Each exchange limits output to 8 MiB, and each stream limits pending input to
8 MiB. An exit code becomes terminal only after process exit and both output
pipes reach EOF. Streams are trial-local process state, not a cross-trial service.
All control frames use the same locked acknowledgement channel; an uncertain
operation blocks further execute, stream and quiesce writes. Tunnels,
interactive SSH, SFTP and SCP are not implemented.
Namespace jobs are quiesced before final judging. Self-contained verifier code
is supplied only after foreground agent termination. Leases remain held through
remote teardown, exact slot-token cleanup, patch capture and memory readback.
Only this trial's worktree is removed; unrelated jobs and worktrees are untouched.

The tested Ascend driver requires a dedicated privileged evaluation container.
Its L3 worker uses separate PID, network, IPC and UTS namespaces without a user
namespace, retains container capabilities, and exposes the container's writable
`/sys`. Runtime paths are read-only and `/tmp` and `/var` are private tmpfs
mounts. The worker binds Ascend device nodes from that container rather than
host block devices.

Remote NPU selection uses cooperative `flock` plus
`ASCEND_RT_VISIBLE_DEVICES`; it is not a hostile multi-tenant device-security
boundary. L3 is for trusted evaluation code in a dedicated container, separate
from unrelated workloads. It never resets a device or kills a process merely
because it uses an assigned NPU. Local profile/oracle restrictions remain
separate from this privileged remote execution boundary. Hardware/topology
behavior still needs live acceptance on the intended Ascend machine.

Fresh workspaces alone cannot prevent a solution leaking through granted runtime
paths. For the RMSNorm tasks, a dedicated derived image replaces the installed
target class and its tests with blobs from the failing revision, removes visible
Git history and stale bytecode, and records an inspected projection manifest.
Generic guards in other classes are reviewed separately. This task-specific
projection does not establish that arbitrary runtime dependencies are free of
contamination; image identity and soundness proofs must follow each change.
The three-family A2/A3 tasks additionally project the exact failing RoPE cache
and sampler functions while preserving modern runtime interfaces. Their source,
bytecode, relevant archives and symlink observations have explicit adjudications;
opaque vendor binaries remain an inspection limit. A3 canonical RMSNorm tasks
were separately revalidated against the final image without rewriting old proofs.

## Native capture and local knowledge delivery

The first-party adapter reuses the selected native Codex transcript parser,
knowledge admission/capture engine, deterministic redactor, native small-model
worker with LangMem, and `MaterialStore`/ReMe retrieval. It preserves redacted
Markdown bodies as authoritative; summaries and indexes remain derived
navigation. It records indexing usage before reporting completion.

For native model aliases, the parent supplies full Codex `ModelInfo` records
exported by an explicitly recorded native CLI. The execution CLI and metadata
export CLI identities can differ and must both be retained. The selected Codex
adapter projects only the unique
summary-model record into its fresh native home as `{"models":[...]}` and sets
top-level `model_catalog_json`. The selected bytes enter route and worker-policy
hashes. Missing, unreadable, malformed or ambiguous supplied metadata fails
explicitly; native warning/error events remain visible. This does not copy a
personal profile or unrelated model records.

Each seed publishes only to its isolated local bare Git repository. An
independent clone verifies the published commit and package hashes, rebuilds
retrieval, and reads actual references before the material enters that lineage's
warm corpus. This local delivery does not exercise GitHub PR creation, Grok
review, or public merge. Consumer readback also does not establish that a later
foreground agent consulted or benefited from the material. Those observations,
and full G0-G3 acceptance, need their own receipts.

## Publication boundary

`export-hf` stages data locally. It excludes raw prompts, reasoning, commands,
MCP environments, exception text and private paths. The installed first-party
`mindie_knowledge` deterministic redactor is required, followed by a fresh
Gitleaks gate with repository allowlists disabled. The entire JSON payload is
scanned before Parquet encoding; Parquet is read back before atomic publication
to the requested local directory. No network upload is performed.
The redaction subprocess also uses Python isolated mode (`-I`), so a module in
the caller's current directory cannot shadow the installed redactor's imports.

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
