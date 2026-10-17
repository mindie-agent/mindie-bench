# Design-principles review - G and H implementations

Reviewed runtime commit: `2b2ea792deafaf7c18820a01749127642252234d`.
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

The H runtime is `18e854e`, digest
`f922f0dda22d052679523830782dca52736054978d5922d0233926fe62bb8bc9`.
Its three added tests and the two-variant verifier classification tests bring
the unified suite to 235 passing tests. H changes only capture-process
diagnostic retention and hashed admission-refusal logging in the runtime.
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
| 1. Total cost | A finite controlled matrix includes completion, delivery time, foreground/provider usage and required G3 seed/indexer resources. G rejects actual sandbox import/interpreter/digest mismatches before expensive preparation or inference. Baselines are reused only under exact compatible controls; missing prices remain null. | 214 tests passed; actual installed preflights used zero model calls. All planned cells and seed resources still require terminal audit; no desired speedup is filled in. |
| 2. Closed tools | Import probing, source projection, budget admission, remote execution, external verification and scoring have mechanical inputs and outputs. Agents retain diagnosis/repair decisions. Recognized NPU command counts are attempt observations, not kernel or successful-test counts. | Import guard tests cover wrong digest/interpreter/module identity, timeout and malformed output. Real source-fail/gold-pass verifiers separately establish bounded NPU behavior. |
| 3. Cognitive load | This remains an explicit maintainer benchmark with fresh native profiles, not a new end-user activation or closing ceremony. Runtime identity checks run internally and report their paths/errors. | No new product setting, personal login copy or approval queue. Independent local CLI help, validation and zero-model demo passed. Product documentation points to the independent component. |
| 4. Single responsibility | Runner, inference broker, remote slot, capture adapter, scorer and exporter retain their existing boundaries. G adds an import prerequisite and a narrowly scoped `/proc` disappearance handler; no parallel scheduler or retry queue. | Wrong imports fail before workspace/remote/Gateway admission. The postflight fix catches normal process disappearance but leaves permission/I/O failures visible. |
| 5. Skills give a map | Frozen native Skills remain advisory. Source snapshots preserve their `AGENTS.md`; private ancestors, answers and root client configuration are excluded. Explicit L1 contract tests are separate from engineering tasks. | Actual G3 engineering prompts do not force knowledge calls. Natural non-use is preserved, not corrected by changing prompts after seeing results. General Skill quality remains outside a finite slice. |
| 6. Historical knowledge is advisory | G2 removes known origins physically; each G3 seed/eval lineage receives only its own captured seed corpus. Held-out captures do not advance later evaluations. Summary navigation never replaces original body identity. | Corpus origin inventory and local consumer package/ref/body checks are explicit. Foreground use requires actual query/explain events. Three defect families do not establish stale-knowledge or universal transfer benefit. |
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
The G 90-cell native matrix must be independently audited after termination;
current successful stages cannot predeclare completion. Final documentation
and CPU recipes do not change runtime bytes, but their own archive and installed
readback must record the later delivery commit and artifacts.

No public development PR, component Release or plugin Release is created by
this work. Any future submission must review its final diff again against the
then-current principles and paired-release requirements.
