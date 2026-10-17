# Acceptance evidence - 2026-10-17

This is development version `0.1.0`, not a published component or plugin Release.
Implementation, task soundness, native experiment completion and measured
engineering benefit are separate conclusions. All paths below are relative to
`reports/acceptance-a2-a3-20261017/`, unless another root is stated.

## Current improvements and measured comparisons

The initial new experiments below use the independently installed runtime
`59a53f994306c8fbee36dfe2a42a675b4533270eaed4c8991137f73037e59379`
and raw evidence under `reports/benchmark-improvement-20261017/`.
Foreground corpus paths/grants were removed: actual knowledge processes own
their private files, while the foreground uses the real MCP transport.
Native servers are required and initialized before the first model request.
Broker accounting includes manual MCP clients, uses the exact native clock
origin, and separates foreground retrieval from capture-consumer checks.

The scorer now parses shell source access, merges overlapping tool intervals,
records first/last target edits and validation tails, and preserves measurement
reasons. The subsequent source-inventory correction uses the actual judged
workspace, excludes unchanged provided inputs and retains local/remote lists
separately. Its installed wheel passed actual zero-model SSH/worktree/postflight
checks on A2 and A3 (`remote-inventory-fix/{a2,a3}-canary/receipt.json`).
Neither canary changes or retroactively rescores a frozen model trial.

The source-bound datasets now include six executable corpus-first CPU recipes,
a whole-module Ascend error replay with an independent 37-case CPU tensor
judge, a stale-advice task with executed old/current API counterexamples and
a real dual-card model-state task. Each admitted task has its own source-fail /
gold-pass proof. Public instructions and reproduction assets remain separate
from gold, hidden judges, replay rules and source history.

### Real A2/A3 native cells

One attempt per condition, same foreground model and task budget within each
hardware study; all eight independent verifiers passed:

| Task / hardware | Condition | Native time (s) | Delivery time (s) | Original regex / parsed command attempts | Foreground query / explain |
| --- | --- | ---: | ---: | ---: | ---: |
| Legacy RMSNorm / A2 | G0 | 275.80 | 310.36 | 11 / 3 | 0 / 0 |
| Legacy RMSNorm / A2 | G1 | 174.02 | 208.85 | 4 / 2 | 0 / 0 |
| Legacy RMSNorm / A2 | G2 natural | 188.34 | 233.33 | 8 / 5 | 0 / 0 |
| Legacy RMSNorm / A2 | G2 explicit | 215.69 | 260.32 | 5 / 3 | 1 / 1 |
| Kimi K3 idle state / A3 | G0 | 298.26 | 338.00 | 6 / 6 | 0 / 0 |
| Kimi K3 idle state / A3 | G1 | 237.76 | 280.16 | 7 / 6 | 0 / 0 |
| Kimi K3 idle state / A3 | G2 natural | 179.07 | 231.38 | 6 / 4 | 0 / 0 |
| Kimi K3 idle state / A3 | G2 explicit | 244.99 | 295.84 | 6 / 5 | 1 / 1 |

The A3 task executes complete historical runner/model modules with actual
DP2/TP1, HCCL, KDA/MLA/MoE and NPU state updates. Its six hidden state checks
cover idle, busy-rank and unaffected-row behavior. It is eager state correctness,
with tiny dummy-weight model execution, rather than ACLGraph, pretrained-model
accuracy or serving throughput. A2 remains the original functional regression.
Both explicit conditions returned exact canonical knowledge bodies; natural
G2 did not query. Baselines passed, so these cells show no completion-rate lift.
Explicit retrieval took longer than G1 in both studies. The parsed counts are
a separate read-only audit, not rewritten trial results or replayed budgets:
13 original matches were textual/compiler commands rather than test-program
execution; six further inline-Python commands are unresolved by the conservative
parser. Parsed counts are lower-bound command observations, not measured kernel
executions or guaranteed hardware-test counts. A2's 55 no-input job observations
all requested 10-second waits; only one was both pending and without new output.
No-input calls are not automatically idle polls.

An independent retrieval diagnostic generated six new bytecode caches while
the natural cohorts were active. Excluding precisely those files reproduced
the frozen component tree hash: all other bytes, including source, were
unchanged. The first explicit launch was rejected before model admission;
the six caches were archived and removed, and the exact tree was restored
before either explicit cell. This environment difference remains in
`diagnostic-cache-restoration/restoration.json`; comparisons are descriptive,
not clean causal estimates. Original trials and the refusal are preserved.

### CPU, replay and retrieval

The earlier 24-cell CPU/replay study completed with 23 passes and one
infrastructure failure. It exposed missing native first-turn knowledge tools,
manual-client undercounting and malformed-client input poisoning the broker.
Those three implementation defects were fixed before the new installed-runtime
study. Original outcomes remain unchanged.
The twelve ordinary G0/G1/G2 CPU cells took 434.35 scheduler seconds; the two
G3 seed/eval pairs took 348.02 seconds. The separate three-condition Ascend
replay took 245.27 seconds, and its explicit G2 cell took 93.55 seconds.
These are actual local slice durations, rather than a five-minute claim for
the complete dataset.

The new ten-cell CPU/replay study has five passes, four failures and one unknown:
two passing seeds; one passing and one unknown G3 evaluation; four failing stale
conditions; one passing explicit placeholder evaluation and one passing replay.
Both G3 evaluations actually queried and explained their own new seed bodies.
The unknown evaluation passed its business judge but its capture worker made a
second native request beyond the declared one-call budget. The first request
has complete provider usage; the refused request was not admitted. Its local
capture remains unknown and was not retried. Evidence does not establish why
the native worker requested a continuation, so the budget was not increased.

All four stale conditions missed the supported `--additional_config` spelling,
whose usage exists in the public historical source. The explicit condition
performed real query/explain but still failed; this is zero observed completion
benefit, not evidence that stale advice caused the failure. Independent
readbacks are in `independent-native-l-review/`.

The fixed retrieval component slice uses 12 distinct known-positive queries
over the frozen 661-package corpus: Recall@3 = 11/12, MRR@3 = 0.83333 and
nDCG@3 = 0.855155. Query median/p95 are 436.59/519.53 ms; explain median/p95
are 6.02/69.29 ms. The preserved miss is a Chinese paraphrase ranked below
the top three by lexical BM25, with correct source labels and indexed bodies.
This slice is not an exhaustive relevance evaluation or general recall claim.

The original twelve-task/five-domain flagship, repeated reliability estimates,
actual graph execution and performance-regression tasks still need additional
source-bound tasks and trials. Current passing eager/CPU/replay cases do not
complete that broader matrix. Private detailed receipts remain under reports;
the product documentation retains architecture and maintainer contracts only.

### Further defects found by real NPU capture

The subsequent N same-defect G3 pair accounts for one failed seed prerequisite
and one blocked eval. Its seed passed all six distributed state checks, used
actual foreground query/explain once each, and recorded the exact native Stop.
Capture failed before any summary-model admission: macOS PATH Git invoked
xcrun, whose loader was denied by the capture sandbox. A same-policy read-only
canary established the original stderr and successful direct Git invocation.
The original receipt remains `infra_failed`, rather than a completed flywheel.

This trial also exposed a scorer omission: native job observations used
`session_id`, while later-completion reconciliation accepted only `job_id`.
The fix accepts either only with consistent scoped result identity, preserving
conflicting or unconfirmed commands as unknown. It does not rewrite the N
null metrics. The command parser now also recognizes the public `reproduce.py`
entry at a literal program position, without counting reads, compiler arguments
or quoted text as executions.

The installed O runtime is
`b1634bfd4f181b07b4c3af248234008bb44eee7c052b79b56d992f143180dfcf`.
It selects actual Git and its helper directory before sandboxing, retains Git
stderr, and verifies local init/commit/push/clone/body-readback before workspace,
remote or model admission. Its full suite has 645 passes and zero skips; all
51 runtime files match source, wheel and installation. The independent
installed Git and MCP/privacy preflights passed with zero model/remote/NPU
executions. Detailed original outcomes and zero-model diagnosis are in
`independent-native-n-review/` and `native-npu-flywheel-n/reconciliation/`.

### Actual O NPU experience reuse

The new O study completed both planned cells without blocked, failed or unknown
outcomes. They repeat one historical defect and the same verifier in separate
native sessions; they are not two independent defects.

| Measurement | Seed session | Session using its captured experience |
| --- | ---: | ---: |
| Native wall time (s) | 303.09 | 250.09 |
| Complete delivery time (s) | 368.55 | 315.38 |
| First observed successful target access, turn / s | 20 / 207.20 | 7 / 53.88 |
| First target edit, turn / s | 20 / 207.20 | 12 / 104.25 |
| Last target edit, turn / s | 20 / 207.20 | 13 / 110.89 |
| Recognized attempts before last edit / total | 3 / 7 | 1 / 5 |
| Foreground query / explain | 1 / 1 | 1 / 1 |
| Foreground / summary-model requests | 26 / 1 | 22 / 1 |

Both cells passed all six independent DP2/HCCL state judgments. Each has a real
native Stop, deterministic redaction, completed one-call model navigation, local
Git publication and an independent consumer. Before eval capture began, its
initial corpus tree exactly matched the seed's captured tree. Its foreground
query selected its own new seed package, and its actual native explain result
returned the canonical 5,065-byte body, SHA-256
`d22360fa0c86b8b11f6eae9ffa5baf5b27ae9aba4621afe42d574abee189a0ea`.
This is foreground MCP evidence, separately verified from consumer readback.

All 48 foreground and two summary-model requests have complete usage:
3,426,146 tokens including the seed and both captures. The scheduler took
686.14 seconds; summed delivery time was 683.92 seconds. There are 64 structured
remote results with zero non-business errors and complete observation coverage.
Both final remote settlements are sealed with zero active/unknown operations,
and assigned-card PID lists are empty.

The edit/access fields are successful command proxies with 2/2 observation
coverage; complete initial Git-blob visibility remains null because only
excerpts were returned. One repeated incident and one eval do not establish
Pass³, completion lift or causal efficiency improvement. The seed cost is
included, and neither this study nor its timing ratios replace the earlier
G0/G1/G2 cohorts. Actual metrics and CLI accounting are in
`independent-native-o-review/reportable-terminal-v1.{json,md}`; closed-loop
identity/body/resource evidence is in
`native-npu-flywheel-o/reconciliation/full-loop-terminal.json`.

### Native C++ numerical repair: four conditions

The source-bound `kda-chunk-varlen-padded-workspace` task adds an actual CANN
operator workspace defect. Its exact A3 soundness proof observes four failing
variable-length cases and four passing dense controls before repair, then eight
passing cases after gold. Each model trial's private judge rebuilds its complete
current 1,626-file C++ source tree, loads a fresh operator prefix and imports the
hash-bound historical test module to execute those eight numerical cases.
It checks finite values and independent FP32 output/final-state references;
it does not run every test in that historical module.

The four predeclared single-attempt conditions used the unchanged installed O
runtime, the same model and a 40-request foreground limit:

| Condition | Original overall outcome | External numerical cases | Foreground / verifier / delivery (s) | Foreground query / explain | Recognized NPU attempts |
| --- | --- | ---: | ---: | ---: | ---: |
| G0 | completed_pass | 8/8 pass | 456.11 / 306.74 / 771.13 | 0 / 0 | 3 |
| G1 | completed_fail: max_turns | 8/8 pass | 643.22 / 308.41 / 960.05 | 0 / 0 | 2 |
| G2 natural | completed_fail: max_turns | 8/8 pass | 487.40 / 306.11 / 811.46 | 0 / 0 | 2 |
| G2 explicit | completed_fail: max_turns | 8/8 pass | 632.29 / 306.04 / 955.91 | 1 / 1 | 3 |

Correct final source and native completion within budget are distinct results:
the latter is 1/4 under this declared protocol. Explicit retrieval succeeded
and took 12.26 seconds, but showed no budget-completion improvement here.
G0 still passes this one task; no observed score was used to force its baseline
into a desired 20-60% band. There are four singleton conditions on one defect,
not a completion-lift or reliability estimate. Natural conditions used a
two-device pool with concurrency two, followed by explicit concurrency one;
these timings are descriptive, not a randomized hardware comparison.

All 158 admitted foreground requests have complete usage: 423,009 uncached
input, 12,216,576 cached input and 34,524 output tokens; no capture requests.
Three subsequent calls were refused locally before paid admission. The actual
scheduler duration was 2,538.52 seconds. Each assigned card's postflight has no
PID, all remote settlements are sealed with no active or uncertain operations,
and all cleanup-error lists are empty. The three-cell natural manifest and
one-cell explicit manifest are accounted separately, then joined by arm ID.

The real G0 trace also validates interval merging: 949.83 cumulative remote
seconds become a 363.06-second union within its 456.11-second foreground wall.
All four exploration-time residuals and first-access measurements are observed.
Original O first/last-edit coverage is 3/4: a nested quoted SSH/Python heredoc
caused G0's literal edit to be missed. Final source commit `52466b4` fixes that
bounded parser and passes 666 tests with zero skips. Its separate observation
replay recovers G0's turn-14 edit at 63.08 seconds and measures all seven
access/edit fields in all four conditions. The other three conditions retain
their original values. The installed-runtime derivation is
`nested-ssh-heredoc-parser-fix-1/p-installed-four-arm-shadow.json`;
original receipts, trajectories, budgets, provider usage and outcomes remain
unchanged.

One natural-G2 `remote_edit` call failed before submission when deleting text
with `new_string=""`. Its native error retains caller/not-sent metadata, but
the original normalized observation lacks structured attribution. Therefore
the original total non-business error count stays unmeasured with known lower
bound zero; it is not reported as zero errors. The component defect is tracked
separately from the final numerical result. Remote-dev commit
`34bb6e514c6b7fe1ebb7eb0527f98bd527d23695`, based on the exact frozen `f72c96c`,
now accepts empty string replacements while rejecting missing, null and
non-string arguments before endpoint resolution. Its schema-to-MCP-dispatch
regression executes the actual file-operation program against temporary files:
47 targeted tests and 54 subtests pass, with no model, SSH or NPU operations.
Old-string and stale-read guards remain exercised. The original frozen
component, model results and failure observation were not replaced by this fix.

The compact measured readout is
`independent-native-kda-o-review/final-reportable-readout-v1.{json,md}`;
fresh-source, eight-case and resource identity are independently checked in
`native-kda-study-o/reconciliation/verifier-terminal-readback.json`.
These are numerical correctness results on A3, alongside the earlier real DP2
state task. They do not complete the original flagship's graph, performance,
cross-model analogical-transfer or repeated-reliability matrix.

## Historical G/H/I cohorts

The source-matched H dual-cluster matrix has terminated with 88 actual receipts
and two blocked cells across 90 planned cells. It did not pass complete matrix
acceptance. The later I runtime passed a separate four-cell native canary and
real zero-model RPC boundary cases on both clusters. These finite checks do not
replace the H matrix or erase its failures and unknown effects.
The [pre-G record](acceptance-pre-g-history.md) preserves prior source
identities, failed/unknown cohorts and the node 153 uncertainty.

## H corrections and terminal outcomes

The later runtime commit `18e854ec15c9bce8408075f0f8e6786aaccc3db4` has SHA-256
`f922f0dda22d052679523830782dca52736054978d5922d0233926fe62bb8bc9`.
Its initial unified suite passed 235 tests in 56.09 seconds. After reusable
environment recipes were added, the delivery suite passed 242 tests in 58.24
seconds, Ruff and formatting for 122 files
(`h-delivery-source-validation/observed.json`). Capture receipts now retain
the actual child return code, stop reason, errors, elapsed time and launch
state before judgment. The inference broker journals a budget rejection before
returning HTTP 429, including only request hash/counters and no prompt.
Unknown classification, budgets, no-retry behavior and provider forwarding
are unchanged. The original G installed runtime is preserved frozen.

The sampler seed judge had the same capitalization omission previously fixed
in the oversized variant. A partial wrong solution can pass its public
reproduction yet raise `Selected index k out of range` in the external judge.
The new seed task recognizes that target error as reward 0 while retaining
driver/unavailable failures as infrastructure. Both A2 and A3 independently
reproved the changed task: source reward 0, gold reward 1, 18 real NPU cases,
zero model calls and clean assigned devices. Its new task hashes and proofs
do not rewrite G's infrastructure results.

`model-review/capture-diagnostics-principles-review.json` and
`task-expansion/sampler-seed-casefold/README.md` record the concrete reviews and
RED/GREEN checks. A real native CLI and the actual first-party summary worker
were also exercised against a local mock provider without paid calls. A
controlled out-of-contract tool response can trigger a distinct continuation
request, cap-1 rejection, and `native_event_error` with complete first-provider
usage but no completed local summary. H records this chain accurately.
The captured summary request omits the top-level `tools` field and requests
strict JSON Schema output. Its `input` also contains an `additional_tools`
description, so the top-level count alone does not establish absence of tool
descriptions or executable native capabilities. This mechanism reproduction
does not establish the missing original A2 response body or justify retrying it.
`model-review/actual-summary-worker-tool-boundary-corrigendum-20261017.json`
withdraws the older top-level-only interpretation and binds the unchanged
original request and mock evidence; successful execution of the described
capabilities remains unverified.

G also encountered foreground provider EOF. Read-only local relay usage shows
three matching-duration failed streams in the same time window and successful
neighboring streams longer than 60 seconds. There is no evidence for a universal
60-second timeout; request hashes are unavailable for exact cross-log joining.
No shared relay setting was changed and no uncertain call was replayed.
H does not claim to fix a provider or network fault.

H's installed experiment source is
`b8f22961989de31bbe979a3d24d1e4af9edb1ece`, with the same H runtime digest.
`a3-h-frozen-manifest.json` and `a2-h-frozen-manifest.json` bind their exact
45-cell plans, six tasks, configuration hashes, actual installed interpreter,
model/catalog/components/corpus and unchanged projected images. A3 uses fixed
baseline devices 11/12 and G3 device 13; A2 uses baseline 3/4 and G3 5.
These are disjoint from the preceding G pools and are not randomized device
matching. Both H actual sandbox/MCP canaries passed with zero model calls.
Global admission included the finite L1/CPU jobs and never exceeded six native
profiles. Later recipe/documentation commits retain the H runtime bytes.

All four H phase supervisors exited. Each hardware accounts for 44 original
terminal receipts and one explicitly blocked eval, without a fabricated receipt.

| Hardware / group and role | Planned | Pass | Fail | Infra | Unknown | Blocked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 G0 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A3 G1 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A3 G2 eval | 9 | 8 | 0 | 0 | 1 | 0 |
| A3 G3 seed | 9 | 6 | 2 | 0 | 1 | 0 |
| A3 G3 eval | 9 | 7 | 0 | 1 | 0 | 1 |
| A2 G0 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A2 G1 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A2 G2 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A2 G3 seed | 9 | 5 | 3 | 0 | 1 | 0 |
| A2 G3 eval | 9 | 7 | 0 | 0 | 1 | 1 |

These are raw receipt statuses: A3 has 39 pass / 2 fail / 1 infra / 2 unknown;
A2 has 39 pass / 3 fail / 2 unknown. The A3 infra receipt additionally contains
an independently adjudicated unknown remote effect, described below. It is
neither a fifth raw unknown receipt nor a resolved effect. Thus the raw H
outcome classification is materially incomplete even though every planned
identity is accounted for. Original receipts are unchanged.

The corrected sampler judge exposes real reward-0 business failures. A
completed failed seed whose capture succeeds can provide reference knowledge
to its eval; its failed outcome is retained. Unknown seed prerequisites blocked
one eval per hardware. Neither task prompts nor the missing cells were altered
to obtain passing results.

Independent audits observed 888 external-verifier NPU case executions on A3 and
876 on A2, including successful stages of affected trials. They verified 15
completed capture/local-Git/independent-consumer loops on each hardware, 15
actual native Stops on A3 and 16 on A2, and eight seed/eval corpus lineages per
hardware. These are repeated case and stage counts, not a clean 90-cell pass.

Actual CLI score/freeze/compare reject both incomplete 17/18 G3 roots.
A2's whole 27-cell baseline and each declared nine-cell baseline group froze;
A3's whole baseline and G2 freeze were refused because of the retained unknown,
while its G0/G1 groups froze. Group views select declared conditions, never
successful outcomes. A3 CLI output is explicitly raw-receipt accounting and
does not supersede the independent late-RPC adjudication.

Each hardware has a 44-row local export containing all actual seed and eval
receipts, with first-party redaction, Gitleaks, thirteen typed core fields and
distinct opaque trial IDs checked against JSON and Parquet. Open the
self-describing scope reports:
`model-review/h-a3-actual44-export-audit/scoped-report/index.html` and
`model-review/h-a2-actual44-export-audit/scoped-report/index.html`.
Their independently scanned scope states 45 planned / 44 actual / 1 blocked
and the additional A3 unknown effect, while preserving original raw statuses.
Nothing was uploaded. Full ledger, loop and CLI evidence is under
`model-review/h-a{2,3}-terminal-audit/`,
`model-review/h-a{2,3}-full-loop-audit/` and
`model-review/h-a{2,3}-terminal-operations/`.

H foreground knowledge MCP query/explain counts are zero. Bounded explicit
corpus-path inventories found three A3 references without a proven body read
and zero A2 references; they do not establish absence of all possible reads.
Consumer readbacks alone are not foreground retrieval.
See `a3-h-native-corpus-access-final-audit/audit.json` and
`a2-h-native-corpus-access-final-audit/audit.json`.

## I remote-settlement correction

H exposed a real outcome-classification race in A3 RMSNorm eval repetition 2
(`rmsnorm-null-description_G3_2_98e7af6bab56`). The native process exhausted
its 600-second wall-clock budget after ten foreground requests, all with complete
provider usage. Its external judge returned reward 1, but the native timeout
made the business outcome fail. No natural Stop was recorded; the subsequent
required capture failed before any capture-model admission. Its original
receipt records infrastructure failure.

The same original broker journal also contains a late RPC `remote_unknown`
after the last model usage settled. H checked gateway uncertainty before the
judge, while an already admitted remote HTTP handler could still complete later.
Required quiescence could race with that handler, and a later capture error
could hide the uncertain RPC result. Clean device cleanup proves resource
release, not the outcome of a missing RPC acknowledgement. The original receipt
is preserved; the independent H audit reports this discrepancy separately
from its raw terminal status.

Commit `620949dab43cf3e87c021d49c608edb8f18dde9e` changes the runtime digest to
`ddf64cba66341dbcf87d1ef8c87265fecb54c31a6fbf927f3a84778ebae4c278`.
The broker now closes remote admission after foreground termination and waits
up to the existing ten-second settlement grace for admitted HTTP handlers,
including result delivery, before worker quiescence or judging. Expiry remains
unknown. The runner checks remote uncertainty again before saving its receipt
and preserves any preceding business, capture or cleanup errors. Sealed
requests, including bodies completed after sealing, cannot admit remote work.
There is no new paid retry, expanded task budget or fabricated Stop.

Real local HTTP regression tests first reproduced the wrong fail/infra outcomes.
The focused related suite then passed 36 tests, including late acknowledgements,
settlement expiry, client disconnect, both remote transports, blocked new
admissions and original-error preservation. The unified I delivery suite passed
252 tests in 68.50 seconds with Ruff, formatting for 123 files and diff checks
(`i-source-validation/`). The independently installed I wheel/sdist match all
40 runtime files and eight installed distribution RECORDs
(`i-installation/`, `model-review/i-artifact-audit/`).

After all H phase supervisors exited, the separate I plan
`i-controlled-canary-plan/plan.json` admitted one new partial-RoPE seed/eval
lineage per hardware, one repetition each. The plan SHA-256 is
`37be43442fa272abb6d29ec6679d69fe96a7c6daaa6475a75ec6baa3d7ef87fd`.
Both native supervisors exited successfully.

| Hardware | Planned / actual | Native passes | External NPU cases | Foreground / capture provider requests |
| --- | ---: | ---: | ---: | ---: |
| A3, assigned device 12 | 2 / 2 | 2 | 36 | 27 / 2 |
| A2, assigned device 4 | 2 / 2 | 2 | 36 | 47 / 2 |

All four receipts match I's source/task/config/model/image identities. All 74
foreground and four capture requests have complete usage and matching broker
receipts. Each cell has an actual Stop tied to its native session, completed
Luna indexing, local producer/bare Git publication and an independently
validated clean consumer. Each eval received exactly its own newly captured
seed corpus, with the new package absent from the initial corpus. All four
remote snapshots are sealed, active zero and not unknown; required postflight
was clean. No old unknown request was replayed.

A3's held-out native Agent actually read its own new seed Markdown with `cat`:
the returned 2,469 UTF-8 bytes exactly match the published canonical block
SHA-256 `bd412f9a9c2784b8d8a6a1c14494d123a51d3aed59314e4cc1b94ae1919deb24`.
This is whole Markdown, distinct from its parsed content-body hash. A2's bounded
native-access audit found a filename-listing call, without a proven body read.
Both have zero MCP query/explain calls. Direct body access establishes an
observed foreground read, not a ranking sample or causal knowledge benefit.
See `model-review/i-a3-native-audit/readback.json` and
`a2-i-final-diagnostic/diagnostic.json`.

Actual installed-I CLI score and whole two-cell freeze passed on each hardware.
Each two-row JSON/Parquet export retained all thirteen typed core fields and
distinct SHA-256 public IDs, passed the first-party privacy gate and Gitleaks,
and retained original receipt hashes. Their independently scanned scope reports
are `model-review/i-a3-cli-export-audit/scoped-report/index.html` and
`model-review/i-a2-cli-export-audit/scoped-report/index.html`.
Full independent ledger and loop evidence is under
`model-review/i-a{2,3}-terminal-audit/` and
`model-review/i-a{2,3}-full-loop-audit/`.

Separate real RemoteSlot/Gateway canaries ran two fresh owned cases per cluster
on A3 device 11 and A2 device 6, with zero model requests and zero NPU tensor
commands. A normal admitted pair settled before quiescence. An intentionally
expired pair kept sticky unknown even after both original HTTP results arrived;
no operation was repeated. Sealed requests, including a partially read body
completed after sealing, returned 409 without new remote work. Owned cleanup,
PID/worktree removal and assigned-device readbacks passed. These are remote
protocol boundary checks, not native Stop or tensor-compute evidence
(`model-review/i-remote-settlement-canary/`).

The declared four-cell native canary and both clusters' RPC boundary checks
passed. They cover one defect family with one repetition, no G0/G1/G2 controls
and no three-repeat reliability estimate; Pass^3 remains null. They establish
neither a 90-cell rerun, four independent defect families, controlled speedup
nor general provider reliability. H's installed runtime and original receipts
remain unchanged; no H hardware evidence is relabelled as I acceptance.
`model-review/i-final-independent-readback.json` binds the terminal ledger,
full-loop, provider, native-access, RPC and actual CLI/export readbacks.

## Preserved G terminal outcomes

All original G runner processes have exited. Each hardware class has 42 original
terminal receipts and three explicitly blocked G3 sampler evals, accounting
for all 45 planned cells. No blocked receipt was fabricated.

| Hardware / group and role | Planned | Pass | Fail | Infra | Unknown | Blocked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 G0 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A3 G1 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A3 G2 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A3 G3 seed | 9 | 6 | 0 | 3 | 0 | 0 |
| A3 G3 eval | 9 | 5 | 0 | 0 | 1 | 3 |
| A2 G0 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A2 G1 eval | 9 | 7 | 0 | 0 | 2 | 0 |
| A2 G2 eval | 9 | 9 | 0 | 0 | 0 | 0 |
| A2 G3 seed | 9 | 6 | 0 | 3 | 0 | 0 |
| A2 G3 eval | 9 | 4 | 0 | 0 | 2 | 3 |

Thus A3 has 38 pass / 1 unknown / 3 infra and A2 has
35 pass / 4 unknown / 3 infra. All three sampler seeds per hardware hit the
old judge classification omission and blocked their dependent evals. A3's
unknown and A2's two G1 unknowns are foreground provider streams; A2's two
G3 eval unknowns occurred during capture. Their successful business phases
do not override the overall unknown state. Original required cleanup was clean
for all 42 A2 receipts and the old G resource readback found no old owning PIDs;
H resources running at the time of that readback were identified separately.

The independent G audits record 864 observed external-verifier NPU cases for
A3 and 846 for A2, including stage successes in affected trials. These are
repeated case executions, not independent tasks or full-cohort passes.
The source/task/config/model/image identities match their own frozen G plans.

Actual CLI score/freeze/compare reject the incomplete 15/18 G3 roots.
A3's complete baseline has all 27 passing cells, so its whole baseline and all
three declared nine-cell groups froze successfully. A2's whole 27-cell baseline
and complete G1 group retain unknowns and were refused as frozen baselines;
its complete G0/G2 groups froze. Views select declared groups, never successful
outcomes. Valid-pair descriptive comparisons keep unknown coverage visible.
The clean A3 G0/G1 comparison has nine matched solved pairs and delivery ratio
1.2075; G0/G2 is 0.8703. G2 made no knowledge calls. These finite G observations
do not establish knowledge benefit or provide H baselines.

Each hardware's full 42 actual receipts, including all nine seeds, also has a
separate explicitly partial local export. Its input-boundary manifest records
45 planned / 42 actual / 3 blocked, all original byte hashes, and no fabricated
experiment manifest. JSON/Parquet identity, status counts and all thirteen typed
core fields match; mandatory privacy gates passed and nothing was uploaded.
See `model-review/g-a3-terminal-audit/`,
`model-review/g-a2-terminal-audit/`,
`model-review/g-a3-actual42-partial-export-audit/` and
`model-review/g-a2-actual42-partial-export-audit/`.

G2 and G3 actual foreground knowledge MCP calls were zero. A2 nevertheless has
one actual native `cat` of its own newly captured RoPE seed block: returned
2,812 bytes match the seed/eval canonical Markdown SHA-256
`01e1b960f82b10401df3ebd35cdb3f3a9073adf6f8b8084cde7ffef06eae067e`,
and the package was absent from the initial corpus. The final native-access
inventory distinguishes this read from four path/filename discovery calls.
That held-out trial completed pass, including its business judge and capture.
This proves a real foreground read via native tools, not an MCP event or
counterfactual effectiveness gain. Consumer readbacks remain separate.
Details are in `a2-g-final-diagnostic/` and
`a2-g-native-corpus-access-final-audit/audit.json`. This inventory covers
commands explicitly referencing the profile corpus path; it does not prove
absence of every possible relative-path read.

## Historical G implementation and shared native dependencies

| Item | Measured identity |
| --- | --- |
| G source commit | `2b2ea792deafaf7c18820a01749127642252234d` |
| 40-file runtime SHA-256 | `73f7c9d1b4f79e191e0e1bb219075b882ec33af27cfa982e482c75cd1bca59d4` |
| Local validation | 214 tests passed in 52.33 s; Ruff passed; 94 files format checked |
| Execution harness | Codex `0.153.4`, actual binary hash in each fingerprint |
| Foreground route | `home/codex/gpt-6.1-sol`, medium |
| Native indexer route | `home/codex/gpt-5.6-luna` |
| Model catalog SHA-256 | `dd856b38c128a40934350e6d5738e37a14389a3fee4b719090cb40e09f112769` |
| Knowledge component | `f81b90a02d27ee5df0b250a517b0df8a31b2ddd6`, version `0.11.0` |
| Remote-dev component | `f72c96c75afc5e35401f907b80296bc221bf420a`, version `0.9.5` |
| Native Codex adapter | Local fix `176518026d04a82958350fa1e7230cc61f4417a0`; not a published Release |
| Initial public corpus | `cec3249ccd75e7a73cf2f2f776117e0ec385795b`, 661 packages / 2,382 Markdown files |

`g-source-validation/validation-observed.json` records the unified source checks.
The complete native model records were exported from the actual installed
Codex `0.163.0-alpha.10.2` binary and selected for the 6.1-Sol/Luna routes.
Only routing prefixes were added; metadata was not fabricated from another
model. The actual `0.153.4` execution binary accepted this catalog under the
trial's sandbox. `model-review/route-6-1-sol/final-route-readback.json` retains
its provenance and health observation.

These are routed model identities, not immutable provider weight hashes.
No authoritative dollar prices were supplied, so USD metrics remain `null`.
The single successful route health request does not establish a provider
reliability guarantee or repair old incomplete provider streams.

G guards actual sandbox helper imports, interpreter identity and runtime digest
before source/remote preparation or model admission. It also handles the normal
disappearance of a short-lived `/proc` process while retaining permission and
I/O errors. The actual installed G MCP preflights passed with zero model calls.
The earlier E/F import failures and `/proc` race remain separate diagnostic
observations; their old unknown receipts were not changed.

## Independent hardware and task soundness

| Hardware | Dedicated projected image | Frozen plan |
| --- | --- | --- |
| A2, measured DCMI `910B4` | `sha256:bd78c94305dc7f9f5dc3f225450ec00b73b8f00a1e0829469c362f60441abaf4` | `a2-g-frozen-manifest.json` |
| A3 node 156, measured DCMI `Ascend910` | `sha256:90d743487f574f50d9baefbb4438b6590f4a1ddea2676f26de50c01da0b5da27` | `a3-g-frozen-manifest.json` |

Both use CANN 9.1 and separately measured driver/Python/torch_npu/runtime
identities. Idle card admission and required postflight are actual observations,
not assumptions from a previous run. A2 and A3 task bytes and hardware labels
are explicit, with separate hashes and proofs. No A2 result is relabelled A3.

| Defect family | Upstream failing base | Upstream fix | Real NPU cases per variant |
| --- | --- | --- | ---: |
| RMSNorm optional metadata | `740e7b9f` (full identity in each task) | `fb5533a` / PR 14704 | 30 |
| Partial-RoPE explicit dimension | `d060c797ed8fd22a1ecdf86c5e26f6eeaf7531f2` | `80e5812b39bd3b19efa8a6af65658f599bd6cae0` / PR 6581 | 18 |
| Reduce-sampler candidates | `d552d03a4851aa4d75420377952b0177c94cd4a5` | `80cd5964286f2a394e3e6b3d1f7581ce901c812a` / PR 10004 | 18 |

Each family has one seed and one held-out trigger. All six tasks were
independently shown to fail at source and pass with the upstream gold patch on
each hardware class: 132 positive NPU cases per hardware, with zero model calls.
Canonical RMSNorm task bytes were additionally revalidated against the exact
final A3 image; `canonical-rmsnorm-runtime-revalidation/summary.json` records
this without overwriting the older canonical soundness receipt.

These are historical source functions with actual NPU tensor execution, not
mock NPU operations. RoPE/sampler project the required failing functions while
preserving modern runtime interfaces. Sampler uses an explicit TP=1 collective
adapter; it does not exercise HCCL. This is not whole historical vLLM
compatibility, loaded-model serving, custom AscendC or performance acceptance.

An initial oversized-top-k verifier treated the NPU's capitalized business
error as infrastructure. Its retained failed proof led to a case-insensitive
expected-error check; the verifier still rejects driver/unavailable errors.
The obsolete full-file rotary projection and import failure are also retained.

## Runtime and corpus answer exposure

Dedicated images remove task-specific fixed helpers/tests/Git history and
project the five relevant functions to exact failing source ASTs. Both runtime
inventories were independently reviewed across Python source, bytecode,
symlinks and relevant archives:

- `task-expansion/a3-leakage-adjudication.json`: 191,705 files, 1,883 symlinks
  and 41 archives; five target functions match the failing bases.
- `a2-infrastructure/strict-runtime-leakage-adjudication.json`: 222,280 files,
  1,960 symlinks and 45 archives; five target functions match the failing bases.

Raw inventory anomalies remain present. Audited vendor wrappers, dangling
symlinks and opaque native binaries are explicit boundaries; this is not a
claim about arbitrary compiled binary contents. Apparent `.pyc` differences
were reproduced as isolated/full-module compilation context differences and
compared against the complete failing modules.

`task-expansion/corpus-origin-review.json` records the exact PR/fix/target
semantic scan of all 661 packages and the review of 13 initial broad matches.
There was no same-origin hit for the two new families in that bounded scan.
Bare digits inside hashes are not source-origin matches. The RMSNorm tasks
retain their three known same-origin exclusions, which G2 physically removes.
Absence of an exact origin does not establish general semantic retrieval quality.

## Declared native matrix and terminal audit contract

Each hardware class has the same finite shape:

| Group and role | Task variants | Repetitions | Planned cells |
| --- | ---: | ---: | ---: |
| G0 eval | 3 | 3 | 9 |
| G1 eval | 3 | 3 | 9 |
| G2 eval | 3 | 3 | 9 |
| G3 seed | 3 | 3 | 9 |
| G3 eval | 3 | 3 | 9 |
| Total | 6 | | 45 |

This gives 90 cells across A2/A3, including 72 held-out evaluations and 18
required seeds. It is three independent defect families, not 90 independent
problems or the proposal's initial twelve-task flagship set. Baselines use two
concurrent slots and G3 one per hardware, six native profiles in total.
Fixed device pools are disclosed and are not randomized device matching.

Terminal audit must account for every planned identity, source/task/config/
model/image hash, original receipt byte hash, external verifier, native Stop,
provider/indexer usage, required cleanup, seed corpus lineage and actual
foreground retrieval through MCP or exact native body reads. Blocked cells
have no fabricated receipt. Unknown and
infra outcomes retain their strict comparison/freeze boundaries.
G3 resources include every required seed. Per-hardware comparisons must not
mix A2/A3 or older attempts. A consumer's successful readback is separate from
the foreground Agent's decision to retrieve knowledge.

## L1 and real CPU evidence

The separate L1 observations used G's actual installed native harness/model,
the same G2 tool/Skill configuration and finite independent budgets. Four
revised clear-negative local tasks completed pass, with zero observed foreground
knowledge/remote/Skill/SSH triggers. The actual routing scorer reports
0/4 false triggers; this is not a population false-trigger probability.
A single explicit warm-corpus canary also completed pass, with one successful
native knowledge query followed by one explain. An independent readback matched
the newly captured seed package revision, block/ref and returned UTF-8 body hash.
This is a separate retrieval contract, not autonomous NPU effectiveness.

The first four negative native executions completed their model work, but
their here-document judge could not create a temporary file in the actual
sandbox. Their original four infrastructure receipts remain unchanged.
After a direct Python judge was validated with zero-model source/gold checks,
four new task IDs and a new finite native cohort were run; all passed.
The old four and new four represent two versions of four request types, not
eight independent negative scenarios. All 41 actual provider ordinals across
old negatives, corrected negatives and canary have complete usage
(18 + 19 + 4); no unknown request was retried. See
`model-review/l1-final-readback/readback.json`.

Background Stop and MCP initialization are not foreground triggers. Natural
non-use of optional knowledge is preserved rather than labelled a false
negative. Direct body reads have their own inventory, outside MCP ranking
denominators.

The four reusable RoPE/sampler CPU recipes live under `tasks/cpu/`. Each uses
the exact historical failing/gold source, real CPU PyTorch and its own L2
soundness proof; no NPU mocking or text-pattern judge is substituted.
Portable templates remain candidates until materialized and proven against
the selected interpreter. A native smoke is a separate experiment, not a
completed CPU G0-G3 effectiveness matrix. The actual one-task G0 CPU smoke
completed pass on the RoPE held-out task: 18 real CPU cases, seven fully
metered foreground calls, 60.27 seconds delivery and zero remote/replay/NPU
calls. Its public reproduction was unchanged and the source was actually
repaired. The repair omits the upstream gold's `int(rotary_dim)` cast, so it is
not byte-equivalent or proof of metadata types outside the tested contract.
All 16,796 CPU runtime files retained their hashes after execution.
`cpu-l2/native-smoke-readback.json` records the exact receipt and source/body
hashes. RMSNorm has no claimed CPU twin.
See `tasks/L1-L2-acceptance.md` for observation definitions and limits.

## Local distribution and final readback

The G wheel/sdist were built and installed independently using exact locked
dependency hashes. `model-review/g-artifact-audit/` verifies all 40 runtime
files across source commit, wheel, sdist and installed package, wheel/dependency
RECORDs, seven exact dependency versions, actual CLI help/validation/demo, and
an extracted-artifact Gitleaks scan with default rules. Its sdist has 139
tracked G files and no raw reports, profiles, auth, `.git` or `.venv`.

The separately installed H wheel/sdist passed the same 40-file runtime,
RECORD/dependency and privacy checks against H source
`b8f22961989de31bbe979a3d24d1e4af9edb1ece`; its sdist contains 177 tracked
source files. `model-review/h-artifact-audit/` records these checks and the
runtime change from G in only `gateway.py` and `orchestrator.py`. Component
payloads were unchanged. Package checks do not establish H hardware acceptance.

The installed I wheel/sdist separately passed runtime byte and RECORD checks
against commit `620949dab43cf3e87c021d49c608edb8f18dde9e`
(`model-review/i-artifact-audit/`). Its hardware evidence is only the bounded
I plan described above.

Documentation and CPU recipe delivery may have a later commit and artifact
hash than the frozen experiment source. Each artifact must identify and retain
its corresponding runtime bytes; H delivery retains H bytes and does not turn
G evidence into H acceptance. Final local JSON/Parquet/HTML exports require
the native deterministic redactor and Gitleaks, unique SHA-256 public trial IDs,
typed core-field checks and independent readback. Local staging is not an
upload, public PR/merge, plugin update or formal Release.

## Preserved failures and remaining scope

Old E/F diagnostics, prior A/B/C cohorts and paid unknown requests retain their
original bytes and source identities. The failed E editable import path, F MCP
read grants, transient process cleanup race, occupied cards and incomplete
provider EOF were not erased by launching G. No unknown call was replayed.
Node 153's earlier Docker/namespace uncertainty remains unresolved; using node
156 does not repair that operation.

Kimi Code and Grok were actually invoked for independent snapshot observations.
Their bounded processes were interrupted before final review and uncertain
calls were not restarted. `model-review/peer-disposition.json` records this;
no final peer approval is claimed. Kimi's missing optional PyArrow test
dependency was separately diagnosed with the same 61 tests passing under the
locked export-capable interpreter; its original failed run remains preserved.

The proposal's initial twelve-task flagship set, four knowledge reuse gradients,
stale-knowledge benefit, full serving, AscendC, multi-host HCCL, native Windows,
other harness live acceptance and public leaderboard publication remain
separate work. This benchmark must report observed costs and outcomes even
when a plugin produces no gain or makes a small task slower.
