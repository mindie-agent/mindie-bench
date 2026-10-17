# Design-principles review - 2026-10-17

Reviewed final runtime commit:
`7691a4810eebdebc6fcfcc99b6feb1eceb3097fa`.
Independently recomputed Python source SHA-256:
`a4e6245daabd3b132ac209a9286f692811b61634dcb322359e09408ab2bb5995`.
`final-e-implementation-source.json` records its source/test validation.
The last hardware experiment C remains bound to commit
`d259695883857646f02c37c89677a82beeffb4ee`, digest
`fd7c8fed8fcccabc754b3274df2a02d53720644195a515e0bd22588cab8c477b`.
Its runtime/task/configuration hashes matched `controlled-final-c-source.json`
at terminal readback and are independently audited from its archived source.
Subsequent runtime changes correct the seed-prerequisite diagnostic and export
identity integrity. They were not followed by another paid or NPU experiment.
README, design, usage and acceptance documentation were read for consistency;
documentation does not change the Python implementation digest.

The parent snapshot read at the earlier review was
`d83e94a4b4dab02fbd7bb1d12a6e556f3c90649f`.
The parent later advanced independently; the verified principles revision remains
`240248b18996904490cca1e1ecceb8d9ecd83e03`, with exact file SHA-256
`642bb77daf460dd9c15931b70983fd972f5f8ee352863c2158432c1ca96b470a`.
The architecture, harness lifecycle and native-runtime reuse documents were
also read. This benchmark changes no parent document, personal plugin, Skill,
MCP configuration, authentication or production native profile. Unprefixed
report paths below are relative to `reports/a3-153-20261016/`.

## Nine concrete conclusions

| Principle | Applicability, findings and resulting behavior | Evidence and limits |
| --- | --- | --- |
| 1. Total cost | Completion, complete delivery time, foreground/provider usage, native indexer usage and required G3 seed resources remain separate observations. Comparison retains complete inputs before selecting matched evaluation outcomes; missing seeds/capture or unknown effects withhold completeness, and absent prices stay null. Final C fixes patch input consuming the command budget and avoids dropping billable usage solely because the native client disconnected. Settlement/drain is bounded rather than an unconditional 120-second wait. | Seed-cost, metric, command-budget and provider-settlement tests cover these paths. Ordinary cells and independent lineages share the scheduling/device pool. Observed delivery sums are resource totals, not concurrent makespan; no clean improvement or dollar saving is claimed. |
| 2. Closed tool boundaries | Source projection, remote RPC, command admission, transport metering, external judging and scoring remain mechanical tools. Final C inspects the command field on one-shot SSH, direct stream open and stream RPC; stdin/patch bodies and uploaded worker source are opaque input. Agents decide diagnosis and repair. | Budget tests verify refusal before forwarding and that failed/unfinished reported attempts still count. Recognition is a regular-expression attempt metric, not a shell interpreter, successful-test count, kernel count or complete hardware-use detector. Actual correctness remains the external verifier's responsibility. |
| 3. Cognitive load | The system remains an independent benchmark operator tool, with no activation or closing ceremony added to normal MindIE use. Explicit components/models and fresh profiles replace accidental dependence on personal settings. Delivery failures are exposed in auxiliary receipts without requiring a recovery command. Native instruction discovery retains cwd `AGENTS.md` without reading private ancestors. | Profile/driver and gateway tests cover the boundaries. The earlier zero-model native instruction canary verifies that specific checkpoint; it is not a final-C paid-model result. No new user-facing setting or approval queue was introduced by the C fixes. |
| 4. Single responsibility | The runner owns trial scheduling, isolation and immutable receipts; the gateway owns admission, forwarding, usage and transport outcomes; the remote slot owns its trial namespace. The integration adapter composes the selected knowledge/capture components and local Git. Final C uses one socket-closing helper for settlement and context exit; the orchestrator copies foreground/capture delivery observations. | No second model scheduler, process manager, knowledge database, persistent service or retry queue was added. Socket shutdown/close occurs outside the Condition/RLock so handlers can retain usage and notify completion. |
| 5. Skills give a map | Native Skill snapshots remain optional reference material. Source projection removes only the declared root client configuration from owned snapshots, preserves `AGENTS.md`, and does not edit the source checkout or users' installed Skills. The benchmark does not force a retrieval call or fixed repair procedure. | Source-projection/profile tests cover symlinks, untracked metadata and reserved configuration. Actual native trials choose their tools. General Skill-trigger quality remains unmeasured; final C changes no Skill. |
| 6. Historical knowledge is advisory | Markdown bodies remain authoritative, summaries derived, and retrieval/status/hash checks separate from engineering correctness. Each G3 lineage starts from a physically filtered corpus. A completed failed seed can provide failure evidence if capture completes; only seeds advance the corpus and held-out captures cannot feed later evaluations. | Temporal-lineage tests and historical real seed query/explain readbacks establish these narrow mechanisms. One upstream defect family with two triggers does not establish general transfer, retrieval quality or negative-transfer rates. Old A/B receipts retain their own source identities. |
| 7. On-demand intervention | Explicit benchmark execution authorizes only fresh trial profiles, assigned remote slots and G3 Stop recording. Credentials stay in the parent broker. Capture delivery is limited to a fresh local bare Git repository and independent clone; there is no public PR, Issue, merge, vote or upload. NPU admission checks current PID/memory inventories rather than relying on earlier idle observations. | The first C capacity check refused occupied devices before any model request. Later fresh checks admitted the finite six-slot cohort; renewed foreign occupancy then failed preflight/postflight. Six G1/G2 cells made zero model calls, and three dependent G3 evaluations were not admitted. Foreign business processes were not killed to make room. |
| 8. Reuse mature mechanisms | Uses native Codex events/catalogs, MindIE transcript/redaction/capture, LangMem, MaterialStore/ReMe, Git, uv, Seatbelt/bubblewrap and native process ownership. The PTY fix uses the sandbox extension and restricts slave access to extension-owned terminals; the redactor runs with Python `-I`. Final C reuses HTTPConnection/HTTPResponse, Condition, monotonic time and socket shutdown, retains the original transport after response ownership transfer, and sets `auto_open=0` to stop the demonstrated implicit reconnect race. | Final-C local tests include actual OS PTY isolation and zero-paid HTTP/SSE/TCP-reset tests. Standalone native PTY RPC and real shadow-CWD export evidence belong to B, not C's exact digest. L3 is a trusted-task privileged-container boundary, not hostile tenancy. Native Windows and HCCL remain unaccepted. |
| 9. Visible errors | Original native lines, remote results and unknown receipts remain intact. Lost acknowledgements block later writes; late business results cannot substitute for cleanup. Final C records downstream model delivery failure separately while draining the same admitted provider request. Incomplete usage, truncation or expiry remain unknown; known 400/401/403/404/422 rejection remains infra. Sealing blocks new admission, duplicate payloads are rejected, and `auto_open=0` prevents implicit reconnect after settlement closes a socket. | Real loopback regressions cover final usage after disconnect, missing usage, truncation with or without complete usage, bounded shutdown, no new request during drain, known rejection and pre-submission close. C correctly retains upstream HTTP 200 EOF without final usage as unknown, with no downstream delivery failure; five successful business verifiers and three completed captures do not override failed required postflight. B ordinal 32, C ordinal 16 and the unresolved 153 operation remain separate unknowns and were not replayed. |

## Concrete findings and fixes

The source review covers both the earlier implementation mechanisms and the
final C delta. Persistent remote-dev RPC now uses bounded open/exchange streams
rather than waiting for stdin EOF; partial command frames are inspected before
submission. Foreground prompt and remote frame writes remain cancellable under
backpressure. Owned namespace cleanup, durable one-shot results, late stream
results and uncertain remote effects remain distinct.

Earlier corrections also confine source snapshots and native instruction
lookup, project complete model metadata into the native indexer's fresh home,
remove audited fixed-answer/runtime-history exposure from the dedicated image,
interleave independent G3 lineages, preserve failed seed evidence and seed-only
corpus advancement, withhold localization speedup for asymmetric observations,
include required seed resources in comparison, grant only native-owned PTYs,
and isolate the export redactor's imports. Their tests and original RED
observations remain inspectable; their historical live evidence is not relabelled
as final-C acceptance.

B's subsequent real cohort found two additional defects corrected in C:

- Five patch/dry-run submissions consumed the broker's recognized-command
  budget because one-shot SSH scanned opaque stdin. Final C inspects only the
  command field, consistent with the stream paths. The regression keeps failed
  and unfinished native attempts observable while refusing over-budget remote
  submissions before execution.
- Native cancellation closed the downstream response and stopped the handler
  before upstream provider usage was settled. Final C journals delivery failure
  independently and drains/meters that same request, with a ten-second deadline.
  Overall inference settlement seals admission, waits at most ten seconds by
  default and shuts down owned transports outside the lock if still unresolved.
  Missing usage and truncated responses remain unknown even if the business
  process stopped deliberately. A barrier test reproduced HTTPConnection's
  implicit reconnect after a settlement close; explicit connect plus
  `auto_open=0` prevents that new submission. No paid retry was added.

Final result review also found that the matrix error called every failed seed
prerequisite "capture failed", including completed capture with failed
postflight. The final diagnostic now reports failed seed prerequisites and
both actual terminal/capture states. The existing no-retry/dependent-admission
regression adds an infrastructure-failed seed with completed capture. This
fix improves principles 1, 2, 3, 4 and 9 without changing scheduling or the
boundaries reviewed under principles 5-8. Original C receipts are unchanged.

Actual export readback then exposed a separate data-integrity defect: mandatory
high-entropy redaction collapsed twelve native trial IDs to one marker despite
successful privacy/row-count gates. The rejected output and affected historical
exports are recorded at `export-redaction-collision-20261017/`; private receipts
were not changed. The final exporter uses standard SHA-256 public IDs before
the unchanged redactor, then verifies row count, uniqueness and thirteen typed
identity/outcome fields. Invalid output fails before publication, without
restoring any private value. Its manifest describes the public ID scheme.
This addresses total rework cost (1), a closed validation boundary (2), exporter
ownership (4), mature hashing plus the existing privacy engine (8), and visible
integrity failures (9). It adds no user workflow, Skill constraint, knowledge
authority or public write under principles 3, 5, 6 and 7. RED/GREEN and concrete
review are in `export-identity-fix-20261017/`.

The source-only review details are retained at
`final-c-principles-review-source.md`; its admission discussion records the
initial blocked observation, before later launch and the terminal readback
below. The preceding B review is preserved
verbatim at `final-c-review/b-source-review-snapshot.md`, SHA-256
`127139439fa2db913a1cbc8b77d6935c560cfef592a12bf32af3ac291c1fde82`.

## Actual validation and final C experiment

The final implementation passed **197 tests in 44.52 seconds**, Ruff check and
format checks for 78 files. Logs are `final-e-source-pytest.stdout`,
`final-e-source-ruff.stdout` and `final-e-source-format.stdout`; pytest stderr
is empty. The suite covers command-budget fairness, real loopback provider
settlement/reconnect behavior, remote receipts/streams, native OS PTY
permissions, capture batches/cleanup, seed-inclusive comparison, export
isolation and runner contracts. These tests make no paid model or NPU calls.

`controlled-final-c-source.json` freezes the two finite plans: nine G0/G1/G2
held-out cells and three G3 seed/evaluation pairs, for 15 planned cells, 12
held-out cells and one defect family. The first capacity check in
`controlled-final-c-preflight.json` was blocked by foreign owning PIDs and
approximately 46 GiB used per device. Its zero plans/model calls describe that
check only. After a later idle readback, fresh
`controlled-final-c-baselines-admission.json` and
`controlled-final-c-g3-admission.json` checks passed. The runners started at
concurrency three each on disjoint pools 8/9/10 and 11/12/13. Both runners
terminated with exit 2. Runtime, task and configuration hashes still match the
frozen manifest.

| Group and role | Planned | Receipts | Infra | Unknown | Not started |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 eval | 3 | 3 | 2 | 1 | 0 |
| G1 eval | 3 | 3 | 3 | 0 | 0 |
| G2 eval | 3 | 3 | 3 | 0 | 0 |
| G3 seed | 3 | 3 | 3 | 0 | 0 |
| G3 eval | 3 | 0 | 0 | 0 | 3 |

There are **12 original terminal receipts: 11 infrastructure failures and
1 unknown**, with no valid completed-pass/failed denominator. Pass@1 and
Pass^3 are null. All three dependent G3 evaluations were blocked by their
seeds' failed required postflight; they were never started and have no
fabricated receipts. **The final C controlled acceptance did not pass.**

Five independent business verifiers passed: G0 repetitions 1/2 and the three
G3 seeds, each with 30 real NPU RMSNorm cases, for **150 cases**. All three
seeds also completed actual native Stop, deterministic capture, one native
Luna/LangMem indexer call, local Git publication and independent consumer
readback. These successful stages are preserved separately from their overall
infrastructure states: required postflight found foreign NPU occupancy.
Six G1/G2 cells failed preflight before any model call. No C held-out G3
evaluation ran, so the C seed readbacks do not establish held-out experience
reuse; the earlier B/smoke held-out observations remain their own evidence.

G0 repetition 3 retains an upstream provider failure: ordinal 16 received
HTTP 200 and then EOF without final usage, recording `IncompleteProviderUsage`.
Its native transcript reports unexpected EOF, and
`model_delivery_failures=[]`; this is distinct from B's downstream BrokenPipe
defect. The original unknown receipt and incomplete metering remain intact.
It was not replayed. The observed provider failure and renewed foreign
occupancy are visible external errors under the intended contracts; they do
not indicate an additional runtime fix or justify weakening admission,
postflight or unknown handling.

`final-c-package-preflight-20261017/summary.json` records a successful C
wheel/sdist preflight: an isolated installed wheel reproduced the exact C
implementation digest, CLI help and two-task validation passed, three
zero-model synthetic demo receipts passed, and archive privacy/Gitleaks checks
passed. Its README metadata predates the final cohort readback.
`final-artifact-readback.json` separately records the final revision's
wheel/sdist hashes, isolated install and exact installed implementation digest.
`final-export-readback.json` records local redaction/Gitleaks and a 12-row
Parquet readback of C's actual failure/unknown receipts, with no upload.
The exporter revision and the experiment revision remain separate identities;
no A/B artifact supplies a C outcome or a final-revision hardware result.

## Preserved real evidence and its limits

The B checkpoint was `301f9772027a91805af702e62aa54cf93ab7d3c4`, digest
`71f7aa54a01977e27b8b3623d74b13db9085e8ca5696d39b49c2867a384bb260`.
Its source validation was 161 passing tests; its separate actual Codex 0.153.4
standalone PTY RPC check made zero model/thread/turn/NPU requests and succeeded
under the B sandbox. Its shadow-CWD export check used B exporter code on six
older real receipts and passed native redaction/Gitleaks without executing
`json.py`. Those checks do not have C's digest.

B's 15-cell plan produced **14 immutable receipts: 10 completed pass,
3 infrastructure failures and 1 unknown**, with one dependent G3 evaluation
not admitted. Twelve independent verifier executions covered **360 real NPU
RMSNorm cases**; verifier success in a trial whose required postflight failed
does not turn that trial into a pass. G0 provider ordinal 32 has incomplete
usage after BrokenPipe and remains unknown. The new predicate/drain fixes do
not settle or replay it. G0 has two valid repetitions, so Pass^3 is null;
coverage is 2/3. The affected budget, incomplete paid usage, failed postflights
and preflight failure prevent a clean effectiveness claim.

B's two admitted G3 held-out evaluations remain `completed_pass`. Each made
one actual knowledge query and one explain call, then completed native Stop,
one native indexer call, local Git delivery and independent consumer readback.
These two real held-out closed-loop observations belong to B; they are not
substitutes for C's three blocked held-out cells.

Actual hardware evidence uses **node 156 in the requested A3 cluster**, not
node 153 itself. The two current soundness proofs each record FAIL before and
PASS after the gold fix, 30 real NPU RMSNorm cases, zero model calls and clean
owned-PID readback on the projected image. Historical smoke 4 records real
native Stop, deterministic body extraction/redaction, one native Luna/LangMem
indexing call, local Git delivery and independent consumer clone readback for
both seed and held-out evaluation. The held-out foreground also queried and
explained the seed's exact block. Its seed/eval source digests differ, so this
is mechanism evidence rather than a frozen treatment-effect comparison.

The earlier A cohort binds `ae9d2feb653321992e1d6a552bf50c4fff60a688`, digest
`59688ee76c5d674def5f058255fea4b13581aaa90663a964cebc544a246617f4`:
24 pass receipts, 720 verifier cases, complete token metering and clean
postflight, qualified by the runner's G0 PTY denial. Its three G3 evaluations
read their own seed blocks; near-exact retrieval scores of 1.0 on these three
observations are narrow diagnostics. Neither this cohort nor synthetic demos
certify the final C source or broad MindIE gains. Exact historical receipts,
component revisions, hashes and pilot reconciliation remain in
`docs/acceptance.md`.

Node 153's original Docker/namespace request remains unknown. The bounded
read-only snapshot still found tracked stuck processes; exact-name inspect
absence did not recover its original completion receipt. It was not replayed,
reset or declared cleaned up. Success on node 156 does not reconcile node 153.

## Review conclusion and remaining boundaries

No unresolved blocking principle violation was found in the reviewed final
source and affected behavior. Final source tests and C package preflight passed;
the C controlled hardware acceptance did not pass. Its partial business and
capture successes remain separately evidenced, with all infrastructure and
unknown states preserved. Final document-version packaging/export remains
recorded separately; general retrieval/transfer quality, comparative cost and
performance remain unmeasured. One defect family with two trigger variants is not a
12-task verified dataset. Hosted aliases do not establish immutable provider
weights.

Local Git closure does not validate public GitHub PR creation, Grok Bot review
or merge. Kimi and Grok were actual development collaborators in independent
snapshots; their patches/reviews are retained in `parallel-review/`, separate
from the tested Codex harness and public knowledge review. GitHub CI, Harbor,
native Windows, multi-host HCCL and Hugging Face hosting/upload are not accepted.

No public development PR, tag or Release was created. The paired-release rule
still applies before publishing a plugin/component; the separate local Codex
adapter fix and frozen payload do not constitute a published dependency
Release. Any later submission must retain the actual distribution/export
readbacks and unresolved hardware/publication boundaries.
