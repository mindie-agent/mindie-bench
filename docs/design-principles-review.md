# Design-principles review - 2026-10-16

Reviewed implementation commit:
`1b27eff2b2c73c4376e600d79bcf798a5c68751a`.
Reviewed Python source SHA-256:
`0d03c31600cce24f80f931de832aa9c69aa39317f7d4c935b0d0c78cb5731c29`.

Principles were read from the parent checkout at commit
`240248b18996904490cca1e1ecceb8d9ecd83e03`, including its existing working-file
changes. The exact reviewed Chinese principles file has SHA-256
`642bb77daf460dd9c15931b70983fd972f5f8ee352863c2158432c1ca96b470a`.
The architecture, harness lifecycle, and native runtime reuse documents were
also read. No parent document, personal plugin, skill, MCP configuration or
production native profile was changed by this implementation.

## Nine concrete conclusions

| Principle | Applicability, findings and resulting behavior | Evidence and limits |
| --- | --- | --- |
| 1. Total cost | Completion and delivery cost lead the report. Added full delivery timing beyond foreground inference, all-attempt cost coverage, matched-success speedup, and G3 seed setup accounting. Fixed double-counting risk in cached input; cache writes require their own declared price. Frozen baselines reject changed controls instead of causing unnecessary reruns. | Metric, usage and comparison tests; the three-trial synthetic report is an identity comparison. No real MindIE gain is claimed. |
| 2. Closed tool boundaries | Runner, source staging, external judge, replay feedback, scoring, candidate discovery and export are mechanical operations with typed inputs and explicit failure. Task solving and applicability judgment remain with the native agent. Replay never certifies physical NPU semantics. | Strict task/receipt/verifier validation and corruption tests. Real engineering verifiers remain a dataset-author responsibility. |
| 3. Cognitive load | The operator CLI belongs to an independent evaluation repository. Normal MindIE users acquire no activation, benchmark, status, configuration or closing ritual. Required experiment settings are explicit rather than silently read from personal profiles. | Isolated profile construction and missing-component tests. Native component adapter configuration is documented, not hidden behind a claimed one-click product acceptance. |
| 4. Single responsibility | Separate modules own profiles, worktrees, native stream adapters, metering, remote ownership, judges, scoring and publication staging. The driver protocol stays small; no global dispatcher, business workspace manager, persistent evaluation service or second production knowledge database was added. | Final module/dependency/CI diff inspected. G3 uses the selected component adapter and copied Markdown view, not a new production capture stack. |
| 5. Skills give a map | Skill snapshots are optional native skill directories. There is no mandatory retrieval instruction, fixed debugging workflow, new skill installation or global instruction edit. | G0/G1 reject knowledge/skill contamination; configured skills are confined to the trial profile. Real skill triggering still needs native L1 observations. |
| 6. Historical knowledge is advisory | Corrected the proposal's public-feed layout assumption. Actual `index.md/blocks` and local `PACKAGE.md/materials` formats have explicit readers. Public block identity/hash corruption is an error. Source statuses remain historical metadata; all mined records stay candidates. G2 physically excludes origins; G3 excludes evaluation origins before seeding and isolates each repetition/cluster. | Physical mask, stale-content preservation, symlink and temporal G3 tests. Actual stale-knowledge behavior and negative-transfer rates are unmeasured. |
| 7. On-demand intervention | Only an explicit CLI run starts an experiment. Every profile has a fresh HOME/XDG/native data root; credentials remain in the parent broker. OS rules prevent direct HTTP publication, source-history access and oracle reads. No public PR, issue, vote or upload is emitted by a trial. | Real Seatbelt oracle/network tests and gateway route tests; synthetic G3 local capture. Native full plugin-loop acceptance is still pending. |
| 8. Reuse mature mechanisms | Reused Git worktrees, uv locking, Pydantic, pytest, OS Seatbelt/bubblewrap and owner-pipe/process-group ownership. Required MCP servers must initialize and expose tools before native inference. The installed first-party deterministic redactor and Gitleaks remain mandatory. No replacement secret detector, universal process daemon or legacy installer was added. | Native version probes, process tests, existing redactor/Gitleaks export, packaging and wheel CLI smoke. macOS guarantees the owned group; detached sessions and native Windows are not advertised as accepted. SSH/NPU execution remains hardware-unaccepted. |
| 9. Visible errors | Four terminal states are exclusive; raw native lines are saved before parsing. Missing data stays null. Required component, corpus, judge and export errors do not become empty success. Unknown paid effects prohibit retries; provider usage survives interruption. Cleanup errors preserve the original business/unknown verdict, and the CLI surfaces recorded failures. | Crash, malformed JSON, missing component, verifier outage, unknown-provider no-retry, interrupted-ledger, tampered-baseline and fail-closed export tests. Original private receipts remain inspectable. |

## Specific findings fixed during implementation

- Temporary HOME alone and `gh` interception were insufficient isolation.
  Added OS file/network boundaries and a model/replay/SSH broker.
- A business file named `operator.py` could shadow Python's standard library
  before the sandbox. Ownership helpers, bridge scripts and shims now use `-I`;
  the regression test proves that an outside sentinel is not written.
- Concurrent process-group cleanup could race the EOF guardian. The owner keeps
  its writer until the explicit group kill, then closes it.
- A pooled `(successes/attempts)^3` would mix task difficulties and overstate
  sparse evidence. Reliability is stratified and requires enough repetitions;
  the unbiased combinatorial estimate is reported alongside the plug-in power.
- G3 seed successes could inflate evaluation scores. Seed rows are separate,
  with setup costs and an inclusive cost-per-resolved-evaluation field.
- Missing native usage, overlapping timing, absent cache-write prices, or
  incomplete capture metering could look like free/fast work. Such metrics are
  withheld; provider usage is metered independently of foreground CLI summaries.
- A public corpus's `complete` status was being confused with success evidence.
  The corpus reader and mining output now retain actual supported statuses and
  explicitly list missing reproduction evidence.

## Validation and remaining acceptance

Actual local validation: **79 tests passed**, Ruff passed, three independent
synthetic trials completed, baseline hashes and Parquet rows were read back,
the existing `mindie-knowledge 0.11.0` redactor and Gitleaks 8.30.1 passed, and
wheel/sdist builds plus an isolated wheel CLI smoke succeeded. Source hashes in
the final receipts matched the reviewed implementation.

The paired-release rule is applicable at publication time: this is an independent
development package with no plugin lock changes, and no formal Release, tag or
public PR was published. The error/unknown rule applies to benchmark model and
remote calls, including the optional indexer; there is no automatic retry.

This review does not certify live SSH/NPU/HCCL behavior, native full solves,
production Stop-hook/capture adapters, native Windows execution, GitHub CI, a
verified 12-task dataset, Harbor compatibility, or Hugging Face publication.
See `acceptance.md` for the concrete evidence and required inputs.

The follow-up documentation change adds README navigation and this review.
It was checked again against principles 3 and 9 for clear entry points and
honest acceptance boundaries. Runtime source and the validated source digest
remain unchanged.
