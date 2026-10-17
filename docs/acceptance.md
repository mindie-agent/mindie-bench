# Acceptance evidence - 2026-10-17

This is development version `0.1.0`, not a published component or plugin Release.
Implementation, task soundness, native experiment completion and measured
engineering benefit are separate conclusions. All paths below are relative to
`reports/acceptance-a2-a3-20261017/`, unless another root is stated.

The source-matched G dual-cluster matrix is currently running. Its 90 planned
cells must reach terminal audit before this document can call hardware
acceptance complete. No earlier successful cell substitutes for a missing G
cell. The [pre-G record](acceptance-pre-g-history.md) preserves prior source
identities, failed/unknown cohorts and the node 153 uncertainty.

## H corrections under validation

The later runtime commit `18e854ec15c9bce8408075f0f8e6786aaccc3db4` has SHA-256
`f922f0dda22d052679523830782dca52736054978d5922d0233926fe62bb8bc9`.
Its unified suite passed 235 tests in 56.09 seconds. Capture receipts now retain
the actual child return code, stop reason, errors, elapsed time and launch
state before judgment. The inference broker journals a budget rejection before
returning HTTP 429, including only request hash/counters and no prompt.
Unknown classification, budgets, no-retry behavior and provider forwarding
are unchanged. The running G installed runtime remains frozen.

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
The real summary request exposes zero tools and uses strict JSON Schema output.
This mechanism reproduction does not establish the missing original A2
response body or justify retrying it.

G also encountered foreground provider EOF. Read-only local relay usage shows
three matching-duration failed streams in the same time window and successful
neighboring streams longer than 60 seconds. There is no evidence for a universal
60-second timeout; request hashes are unavailable for exact cross-log joining.
No shared relay setting was changed and no uncertain call was replayed.
H does not claim to fix a provider or network fault. Any fresh H hardware cohort
must have its own finite plan, installed source identity and terminal audit.

## Frozen implementation and actual dependencies

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

## Planned native matrix and required terminal audit

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
foreground retrieval. Blocked cells have no fabricated receipt. Unknown and
infra outcomes retain their strict comparison/freeze boundaries.
G3 resources include every required seed. Per-hardware comparisons must not
mix A2/A3 or older attempts. A consumer's successful readback is separate from
the foreground Agent's decision to retrieve knowledge.

## L1 and real CPU evidence

The separate L1 negative request observations and optional explicit warm-corpus
retrieval canary have finite independent budgets. They do not enter the L3
matrix or establish a population false-trigger probability. Background Stop
and MCP initialization are not foreground triggers. Natural non-use of optional
knowledge is preserved rather than labelled a false negative.

The four reusable RoPE/sampler CPU recipes live under `tasks/cpu/`. Each uses
the exact historical failing/gold source, real CPU PyTorch and its own L2
soundness proof; no NPU mocking or text-pattern judge is substituted.
Portable templates remain candidates until materialized and proven against
the selected interpreter. A native smoke is a separate experiment, not a
completed CPU G0-G3 effectiveness matrix. RMSNorm has no claimed CPU twin.
See `tasks/L1-L2-acceptance.md` for observation definitions and limits.

## Local distribution and final readback

The G wheel/sdist were built and installed independently using exact locked
dependency hashes. `model-review/g-artifact-audit/` verifies all 40 runtime
files across source commit, wheel, sdist and installed package, wheel/dependency
RECORDs, seven exact dependency versions, actual CLI help/validation/demo, and
an extracted-artifact Gitleaks scan with default rules. Its sdist has 139
tracked G files and no raw reports, profiles, auth, `.git` or `.venv`.

Documentation and CPU recipe delivery may have a later commit and artifact
hash than the hardware G source. Such artifacts must retain the exact G runtime
bytes and record both identities. Final local JSON/Parquet/HTML exports require
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
