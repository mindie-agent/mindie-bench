# Design-principles review

## Native inference lifecycle refactor (2026-10-20)

- Scope: internalize lifecycle contracts, counterbalanced paired workloads,
  required regression guards and task-balanced aggregation in the native harness.
  The external RSI/Pier launcher and importer are removed. No new runtime
  dependency, background process or paid model call is introduced.
- Compatibility: optional task/verifier fields preserve existing schema 1.0
  tasks and receipts. Existing binary rewards, four terminal states, source/gold
  soundness, ACLGraph/workspace/warm-server gates and G0-G3 accounting remain.
- Responsibility: data contracts, judge measurement and receipt aggregation are
  separate small modules with invariant headers. Local and remote judges share
  validation; soundness passes the same contract. The existing runner owns all
  timeouts, isolation, cleanup and no-retry policy.
- Evidence: baseline source/runtime/workload identities, run nonce and frozen
  oracle bind measurements. Correctness and guard failures are not missing data.
  Slowdowns remain below 1; unchanged source cannot gain from timing noise.
  Complete scores require the original plan, not the surviving receipt subset.
- Deliberate differences: every declared guard is required, evaluated with the
  paired median. Task acceptance stays binary; measured gain stays separate.
  Group, role, tier, evidence and model/config cohorts remain distinct.
- Review corrections: route original matrix cells into run/score/audit/export
  summaries; preserve postflight invalid outcomes and additional native gates;
  revalidate existing performance evidence even for subthreshold inference gains;
  retain native budget stops; use a high-resolution monotonic counter for short
  host measurements. The matrix integration test supplies explicit concurrency.
- Validation: native contract/scoring/accounting tests, actual CPU subprocess
  judges, old scorer regression tests, Ruff and packaging are checked. A native
  source/gold plus three-task matrix test runs on Linux/macOS CI; Windows does
  not bypass the existing OS sandbox. The PR validation record identifies the
  exact passing checks and remaining platform/hardware limits.
- No claim of production GPU/NPU task acceptance is made by synthetic examples.
  No frozen historical NPU task or its soundness receipt is rewritten.

## Current benchmark improvements

Reviewed final implementation commit: `52466b4`; its 51-file runtime digest is
`92f13ff4fb138a1a69818e1c5b5b2cfafe63a3688623b53e8610113d512031cf`.
The actual O model experiments retain implementation `d5e546a`, bound by the
51-file implementation digest:
`b1634bfd4f181b07b4c3af248234008bb44eee7c052b79b56d992f143180dfcf`.
Principles revision remains `240248b18996904490cca1e1ecceb8d9ecd83e03`,
file SHA-256 `642bb77daf460dd9c15931b70983fd972f5f8ee352863c2158432c1ca96b470a`.

1. Total cost: interval unions replace summed concurrent waits; first/last edits,
   validation tails and seed-inclusive costs separate repair progress from later work.
   Program-position parsing removes textual false test matches without claiming
   to count every opaque program or actual kernel execution.
2. Closed boundaries: the broker transports actual MCP exchanges; recipe compilation
   validates source/assets and leaves engineering diagnosis to the native agent.
   The final scorer scans bounded literal quotes, escapes and comments before
   interpreting a heredoc. Only existing shell/SSH recursion interprets quoted
   command arguments; cat/echo payloads remain data and no command is executed.
3. Cognitive load: existing run/mine/compare commands remain the maintainer entry.
   Required MCP startup removes the need for agents to construct manual clients.
   Actual local Git publication is checked automatically before model admission,
   without a user-facing recovery command or an unsandboxed capture fallback.
4. Single responsibility: profiles own grants, the broker owns transport observations,
   external judges own correctness and the scorer owns derived measurements.
   One shared command predicate now serves all three gateway admission paths
   and the scorer instead of four independently interpreted regex checks.
5. Skills give a map: domain guidance remains advisory; explicit retrieval is a
   separately declared experiment condition rather than an implicit product mandate.
6. Advisory knowledge: actual stale advice is tested against current source;
   origin packages are physically excluded and G3 consumers receive their own seed.
7. On-demand scope: each requested trial uses fresh profiles and assigned slots.
   Personal configuration and public publication remain outside these experiments.
8. Reuse: native MCP readiness, real MindIE capture/retrieval, Git, OS sandboxes,
   locked CPU libraries and actual distributed model modules are used directly.
   G3 materialization copies the existing task as explicit same-defect candidates;
   it never inherits their source/gold proofs or invents a second task domain.
   macOS capture resolves the installed Git executable before sandboxing and
   grants only that binary and its existing helper directory, avoiding xcrun's
   blocked loader without granting the complete developer-tools installation.
   The new C++ task builds the complete current operator source and imports a
   hash-bound historical test module for eight numerical cases.
   Its A3 source-fail/gold-pass proof is separate from model repair results.
9. Visible errors: caller argument errors remain visible without poisoning a healthy
   knowledge backend; provider/remote uncertainty is retained. Remote modification
   inventories now describe the judged workspace and exclude unchanged provided inputs.
   Conservative literal-container access analysis invalidates escaped or mutated
   aliases. Later job receipts establish success only for an unambiguous matching
   target/job; original pending, conflicting and unknown facts remain intact.
   Both native job-identity aliases are accepted only when consistent with each
   other and the result. Git failures retain the original exception and stderr.
   The final quote-aware correction retains missing heredoc boundaries as
   unresolved. Its observation-only replay never changes original trial statuses,
   budgets, provider usage or receipts.

The final source suite reports 666 passed and zero skips, with all optional
native/MCP/CPU runtimes explicitly selected. The earlier O suite reported 645;
the first final-task software snapshot reported 648 before the nested-SSH
correction. These source revisions remain distinct from their model experiments.
Wheel/source/install byte and dependency checks are recorded in their separate
runtime inventories; six base dependencies use hash-locked entries.
The prior remote-inventory correction passed separate real zero-model canaries
on both A2 and A3. N's installed foreground/MCP preflight returned the exact
knowledge body while denying four existing private files to Python and `cat`;
both backend sessions closed cleanly, with zero model or remote executions.
Frozen L model trials retain their original `59a53f99…` implementation and
original counters. N's read-only command audit records 13 textual false matches
and six unresolved inline-Python commands rather than relabelling all 19
differences as false tests. The real N seed passed its six state checks but failed
capture before any summary-model request; its eval remains blocked. Same-policy
zero-model diagnosis established the xcrun loader failure. O's installed sandbox
then passed the complete owned init/commit/push/clone/body-readback route and the
private-bare check. O's MCP privacy preflight also passed with unchanged component
trees and zero model/remote executions. The subsequent actual O study completed
both planned DP2 cells and both captures: independent readback confirms exact
Stop identities, twelve state judgments, local Git/consumer identity and the
eval's actual native receipt of its own seed body. Fifty requests have complete
provider usage; original N failure/blocked outcomes remain unchanged. This is
one repeated incident, with seed-inclusive cost and no causal lift claim.
The subsequent four-condition C++ study used the unchanged installed O runtime:
all four final source snapshots passed eight numerical cases, while three native
sessions exhausted their fixed 40-call budget and remain completed failures.
The first G0 target edit was absent from the original O metrics; a separate
read-only final-parser derivation recovers the actual literal edit without
rewriting the original experiment. Independent review found no material issue
in the bounded parser change and its opaque-payload/incomplete-boundary controls.
The actual empty-replacement failure also produced remote-dev fix `34bb6e5`
in an isolated worktree based on the exact frozen component revision. Its
declared string schema accepts deletion, while invalid input remains a visible
caller error before endpoint resolution. Forty-seven targeted tests plus
54 subtests execute the real file-operation program locally, including stale
read protection. This component change has its own concrete nine-principle
review and does not replace the original experiment's frozen dependency.
Actual task outcomes and environment differences are in
[acceptance](acceptance.md), rather than inferred from passing software tests.

## Frozen G/H/I review history

Current reviewed runtime commit:
`620949dab43cf3e87c021d49c608edb8f18dde9e`, with 40-file runtime SHA-256
`ddf64cba66341dbcf87d1ef8c87265fecb54c31a6fbf927f3a84778ebae4c278`.
The final delivery may add documentation while retaining those exact runtime
bytes. Hardware results from another runtime remain separately identified.

The G runtime reviewed below is `2b2ea792deafaf7c18820a01749127642252234d`.
The independently installed 40-file runtime SHA-256 is
`73f7c9d1b4f79e191e0e1bb219075b882ec33af27cfa982e482c75cd1bca59d4`.
The authoritative product checkout was read at
`a569fc6a26ef7318193e1e04bddb80b4ac5e477d`; its principles file last changed at
`240248b18996904490cca1e1ecceb8d9ecd83e03`, SHA-256
`642bb77daf460dd9c15931b70983fd972f5f8ee352863c2158432c1ca96b470a`.
Architecture, harness lifecycle and native runtime reuse were also read.

The [earlier review](design-principles-pre-g-history.md) retains prior source
reviews and observed defects. It is historical evidence, not G's acceptance.
Current experiment outcomes are maintained in [acceptance](acceptance.md).
Report paths here use `reports/acceptance-a2-a3-20261017/`.

## I: late remote-RPC settlement

Actual H evidence exposed a race after a native 600-second timeout. Ten model
requests had complete provider usage, but an already admitted remote RPC
reported uncertain effects after the runner's first outcome check. The original
receipt retained an independent passing judge, capped business failure and
missing-Stop capture failure, but did not promote that late remote uncertainty
to the overall status. Clean postflight cannot establish the missing RPC result.
The H receipt and journal are preserved; I does not rewrite or resume them.

I seals new remote admission and waits for admitted HTTP handlers before
quiescence and judging, using the existing bounded settlement mechanism.
Expired settlement prevents judging and capture. Final receipt construction
checks remote uncertainty again and retains prior business, capture and cleanup
facts alongside `unknown`. An uncertain operation is never resubmitted.

| Principle | Concrete I review conclusion | Actual validation and boundary |
| --- | --- | --- |
| 1. Total cost | Settling the existing request prevents false completion and subsequent duplicate work. The bounded wait adds no inference or user interaction. | The real H failure is tied to its original ten metered requests. I's 252-test suite passed in 68.50 seconds. Four separately frozen native RoPE seed/eval cells have passed on A2/A3; they do not replace H's planned cells or establish a speedup. |
| 2. Closed tools | Admission, handler completion and uncertain transport delivery are deterministic broker responsibilities. The runner still leaves diagnosis and repair to the native agent. | Actual loopback HTTP tests cover an admitted late error, client disconnect and a partially submitted request; no arbitrary command interpretation is added. |
| 3. Cognitive load | The internal runner performs settlement without a new end-user command, setting or recovery ritual. | Existing experiment budgets and maintainer entry commands remain sufficient; source and failure references stay in receipts. |
| 4. Single responsibility | Gateway owns admission, in-flight counts and transport facts. The runner orders judging and merges those facts into its final receipt. | Two runtime files changed; no second scheduler, durable queue or component state machine was added. |
| 5. Skills give a map | RPC lifetime handling adds no Skill or prompt instruction. Native engineering decisions retain the same advisory context. | The changed source and test commit includes no Skill files or model prompts. |
| 6. Historical knowledge is advisory | I changes transport accounting, not knowledge authority or G3 corpus advancement. A failed seed still needs known business state and complete capture. | Existing lineage tests remain in the unified suite; direct reads, MCP retrieval and consumer readback retain separate evidence contracts. |
| 7. On-demand scope | Only an already requested trial's admitted operations are settled. New calls after sealing receive 409 without executing an RPC. | Both SSH transports and an HTTP body completed after sealing are tested. No personal configuration, public write or foreign-process cleanup is added. |
| 8. Reuse mature mechanisms | Uses the broker's existing lock/condition, bounded settlement, journal and owned remote teardown. It preserves the frozen H installation. | The I wheel, sdist, source commit and non-editable installation have identical runtime bytes; dependency RECORD and lock checks pass independently. |
| 9. Visible errors | Late remote uncertainty overrides a nominal business or infrastructure result while preserving that earlier result and its original errors. Clean device state does not imply reconciliation. | Four pre-fix failures were retained. Ten new cases cover ordering, expiry, refusal and error preservation; 36 focused tests pass. No fake Stop or retry is introduced. |

`model-review/late-remote-settlement-fix/review.json` records the actual race,
RED/GREEN evidence and affected-principle review.
`i-source-validation/observed.json`, `i-installation/installed-identity.json`
and `model-review/i-artifact-audit/bytehash-and-dependency-audit.json` bind the
source and installed artifacts. Source/package checks are not substitutes for
the separate native and real-remote canary evidence in [acceptance](acceptance.md).
The four-cell native plan is `i-controlled-canary-plan/plan.json`.
Both hardware classes also completed their two-case real SSH/RemoteSlot canary
in `model-review/i-remote-settlement-canary/{a2,a3}-attempt-1/observed.json`.
Normal settlement drains the actual admitted calls; the deliberate short
deadline preserves sticky unknown after those original calls return and after
owned cleanup. Post-seal and partially submitted requests receive 409 without
executing a new RPC. Those transport canaries use zero model calls and zero NPU
tensor commands; native Stop and NPU correctness require the separate native
cells.

`model-review/i-{a2,a3}-terminal-audit/cohort-audit.json` now verifies two of
two planned native cells per hardware, exact source/task/config/image identity,
complete provider usage and 36 external NPU cases per hardware.
The corresponding `i-{a2,a3}-full-loop-audit/readback.json` validates actual
Stop/session/transcript, local Git producer/bare/consumer identity, package/body
hashes and each held-out view's own seed lineage. A2's independent final resource
readback is `a2-i-final-owned-resource-readback/readback.json`; A3's independent
native and resource audit is `model-review/i-a3-native-audit/readback.json`.
Owned processes and worktrees are absent, and assigned-device DCMI PID lists
are empty; driver-resident memory is retained as measured rather than called zero.

The A3 held-out native trajectory also returned its own new seed Markdown:
2,469 UTF-8 bytes match the actual block file, with separate canonical block and
body hashes. This is direct foreground shell reading, not a query/explain event,
IR ranking observation or proof of performance benefit. The new package was
absent from the initial corpus. A2's foreground observation is only a path
listing; it is not promoted to a body read. These bounded observations preserve
principles 1, 6 and 9 without changing prompts to force retrieval.

Final documentation review also found an overbroad earlier summary-worker
inference: no top-level `tools` field does not mean no tool descriptions in
`input.additional_tools`. The original mock does not prove those described
capabilities were executable. The old interpretation is explicitly withdrawn
in `model-review/actual-summary-worker-tool-boundary-corrigendum-20261017.json`,
with all twelve original evidence files unchanged. This correction applies
principles 2, 6 and 9: distinguish observed request fields from execution
capability and unknown original cause, without new inference, retries,
undocumented feature flags or a budget increase.

## G and H implementation review

The H runtime is `18e854e`, digest
`f922f0dda22d052679523830782dca52736054978d5922d0233926fe62bb8bc9`.
The final delivery suite has 242 passing zero-model tests, including the
Ascend runtime preparation recipe. H changes only capture-process diagnostic
retention and hashed admission-refusal logging in the runtime.
The sampler seed task has its own revised proof after a case-insensitive
expected-error correction.

H's affected-principle review found that missing capture stop reasons created
avoidable investigation cost (1), obscured a closed admission boundary (2),
and weakened error visibility (9). Diagnostics now stay with their owning
broker/runner (4), reusing existing ProcessResult, SHA-256 and durable journals
(8). No new user ceremony, Skill rule, forced knowledge call, personal
configuration or public write was added (3, 5, 6, 7). Tests verify one actual
loopback admission plus a rejected distinct request, and separate complete
provider usage from incomplete capture. Budgets and unknown semantics remain
unchanged. Real native/summary-worker zero-paid injection reproduces the
controlled rejection chain without establishing the missing original A2
response content. Original G receipts and installed bytes are unchanged.

The changed sampler seed was explicitly returned to candidate, its old proof
preserved, and then independently reproved on A2/A3 before promotion. Actual
case-insensitive target errors remain business failure while unavailable
runtime, driver and allocation errors remain infrastructure. This reduces
false diagnosis (1), keeps the task judge's boundary precise (2, 4), reuses the
real handler and external judge (8) and preserves honest errors/history (9).
There is no change to the user/Skill/knowledge/scope boundaries (3, 5, 6, 7).

| Principle | Concrete review conclusion | Validation and remaining boundary |
| --- | --- | --- |
| 1. Total cost | A finite controlled matrix includes completion, delivery time, foreground/provider usage and required G3 seed/indexer resources. G rejects actual sandbox import/interpreter/digest mismatches before expensive preparation or inference. Baselines are reused only under exact compatible controls; missing prices remain null. | 242 final delivery tests passed; actual installed preflights used zero model calls. G's full planned ledger was audited with blocked cells and unknown costs visible. H requires its own terminal audit; no desired speedup is filled in. |
| 2. Closed tools | Import probing, source projection, budget admission, remote execution, external verification and scoring have mechanical inputs and outputs. Agents retain diagnosis/repair decisions. Recognized NPU command counts are attempt observations, not kernel or successful-test counts. | Import guard tests cover wrong digest/interpreter/module identity, timeout and malformed output. Real source-fail/gold-pass verifiers separately establish bounded NPU behavior. |
| 3. Cognitive load | This remains an explicit maintainer benchmark with fresh native profiles, not a new end-user activation or closing ceremony. Runtime identity checks run internally and report their paths/errors. | No new product setting, personal login copy or approval queue. Independent local CLI help, validation and zero-model demo passed. Product documentation points to the independent component. |
| 4. Single responsibility | Runner, inference broker, remote slot, capture adapter, scorer and exporter retain their existing boundaries. G adds an import prerequisite and a narrowly scoped `/proc` disappearance handler; no parallel scheduler or retry queue. | Wrong imports fail before workspace/remote/Gateway admission. The postflight fix catches normal process disappearance but leaves permission/I/O failures visible. |
| 5. Skills give a map | Frozen native Skills remain advisory. Source snapshots preserve their `AGENTS.md`; private ancestors, answers and root client configuration are excluded. Explicit L1 contract tests are separate from engineering tasks. | Actual G3 engineering prompts do not force knowledge calls. Natural non-use is preserved, not corrected by changing prompts after seeing results. General Skill quality remains outside a finite slice. |
| 6. Historical knowledge is advisory | G2 removes known origins physically; each G3 seed/eval lineage receives only its own captured seed corpus. Held-out captures do not advance later evaluations. Summary navigation never replaces original body identity. | Corpus origin inventory and local consumer package/ref/body checks are explicit. Foreground MCP use requires actual query/explain events; observed native shell body reads have their own exact byte and lineage evidence, outside the MCP ranking denominator. Three defect families do not establish stale-knowledge or universal transfer benefit. |
| 7. On-demand scope | Only requested finite benchmark runs activate fresh profiles, assigned remote slots and local G3 capture. True credentials remain with the parent relay/broker. No public delivery occurs. | No personal Skill/plugin/MCP/auth changes, no public PR/Issue/HF writes, and no foreign process cleanup. Actual preflight admission is repeated per trial rather than inferred from earlier idle cards. |
| 8. Reuse mature mechanisms | Uses native Codex events/catalogs, MindIE extraction/redaction/capture, LangMem, MaterialStore/ReMe, Git, uv, Seatbelt/bubblewrap and process ownership. CPU tasks use actual locked PyTorch. | Installed G runtime bytes match the frozen commit/package; helper probes use the real interpreter with `-I` and the same grants. Leakage review retains native binary/vendor limits. No serving/HCCL/hostile-tenant claim. |
| 9. Visible errors | Business results, required cleanup, capture, usage and uncertain external effects remain separate. Unknown requests are never automatically replayed. G's `/proc` fix suppresses only a proven normal exit race; wrong grants and genuine read failures stay visible. | Regression tests distinguish FileNotFoundError from permission/EIO. Original E/F and prior unknown/infra receipts remain untouched. Artifact identity/redaction checks reject unusable exports instead of emitting success. |

## Findings and fixes included in G

The E operator used an editable interpreter to launch frozen code; sandboxed
helpers could not import it and a native Agent spent model turns trying to
recover. F installed the wheel but supplied an integration parent path outside
its read grants. The final configuration points to installed G modules and
grants the actual runtime. G additionally probes helper imports under the
real sandbox and interpreter before model or remote admission. It neither
widens grants automatically nor substitutes an import fallback.

A2's earlier postflight encountered a short-lived process disappearing between
`/proc` enumeration and stat/pidfd access. The narrow FileNotFoundError handling
now treats that normal disappearance consistently across the scan. Permission
and device I/O failures remain visible. Old cleanup uncertainty is unchanged.

Task development found a broken whole-file historical rotary projection and
an expected NPU business error with different capitalization. The exact failing
functions are now projected while preserving modern runtime interfaces; the
sampler verifier recognizes the expected error without hiding driver failures.
Source/gold proof and runtime answer-exposure audits were repeated on the final
images. Canonical RMSNorm task bytes received a separate exact-image proof.

Earlier fixes remain included: PTY ownership restrictions, native instruction
scope, complete indexer metadata, opaque patch input budget handling, bounded
provider drain without implicit reconnect, failed-seed diagnostics, seed-only
lineage advancement, seed-inclusive resource accounting and unique public
export IDs. Their old observations remain tied to their original revisions.

## Actual validation

`g-source-validation/validation-observed.json` records 214 passing tests in
52.33 seconds, Ruff and format checks for 94 files. The suite includes actual
OS isolation and loopback protocol tests without paid inference or NPU use.
`model-review/runtime-import-principles-review.json` records the focused
RED/GREEN guard checks and all nine conclusions.
`model-review/g-artifact-audit/nine-principles-review.json` provides the
independent package/runtime review.

A2/A3 final-image task soundness passed for six task variants per hardware.
The G 90-cell plan has terminated: each hardware class has 42 actual receipts
and three blocked evaluations. The full ledger, original incomplete inputs,
refused freezes and explicitly partial exports were independently audited in
`model-review/g-final-independent-readback.json` and the accompanying
`g-{a2,a3}-terminal-audit/` and `g-{a2,a3}-terminal-operations/` records.
This is not acceptance of a complete G matrix. H's replacement matrix requires
its own evidence: both hardware plans have now terminated with 44 actual
receipts and one blocked held-out evaluation each. Their independent audits
are in `model-review/h-{a2,a3}-terminal-audit/cohort-audit.json`. A2 retains
39 pass, three business fail and two unknown; A3 retains 39 pass, two business
fail, one infrastructure failure and two unknown. A3's infrastructure receipt
also has the late remote uncertainty described in I's review. Its original
status is preserved and the independent adjudication marks that uncertainty
explicitly; H is not endorsed as a clean or complete matrix.
The all-actual JSON/Parquet/HTML exports retain every seed and original status,
with separately redacted and scanned scope reports showing 45 planned versus
44 actual cells and the known classification discrepancy. No missing receipt
or successful subset is substituted for the original plan.
`h-delivery-source-validation/observed.json` records the 242-test final source
check. Final documentation and CPU/runtime recipes do not change H's 40 runtime
files, but their archive and installed readback must record the later delivery
commit and artifacts.

No public development PR, component Release or plugin Release is created by
this work. Any future submission must review its final diff again against the
then-current principles and paired-release requirements.
