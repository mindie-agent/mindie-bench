# 0.2.0 acceptance

This stage implements the existing Codex/Claude Code and five-component design.
It introduces no Kimi driver, public publication, PR or Release. The experiments
below use real models unless explicitly marked zero-model. Raw trials remain
under `reports/`; development configuration errors are repaired in place.

Budget classifications below preserve the historical runner policy. Since
2026-10-20, a settled budget stop no longer rejects an independently verified
artifact: business acceptance and budget compliance are reported separately.
Original receipts and result tables are not rewritten or silently rescored.

## Implemented contracts

| Requirement | Implementation and acceptance |
| --- | --- |
| NPU Infra tasks | Existing six CPU and six Ascend smoke cases retained. Added cross-model quantization L2, real TP2/EP2 stale deployment L3, and gated RMSNorm performance L3; retained real DP2 state and native KDA cases. |
| Two native harnesses | Codex and Claude Code support G0-G3. Both have actual Stop/transcript, native organizer, redaction, summary, local Git and independent consumer evidence. |
| Oracle integrity | Protected-file SHA-256 checks around the private judge, negative controls, explicit `oracle_tampered`/reward zero. Performance evidence binds the run nonce and oracle identity. |
| Performance transfer | Synchronized host monotonic timing, two private shapes, four buffers per shape, actual ACLGraph capture/replay, workspace budget and warm eight-step HTTP service. |
| G3 quality | CorpusLinter rejects unconditional bans, absolute performance claims, missing assistant-authored causal sections and native indexes over 50 lines, before publication and consumption. |
| Physical exclusivity | Coordinator device/port leases, live npu-top process/HBM/AICore checks, diagnostics, separate compiler caches and short private TMPDIR aliases. |

The real tests also fixed SSH RPC standard-library shadowing, A2 single-chip
occupancy parsing, TP2 Unix-socket path limits, Claude beta-query forwarding,
Claude temporary/timezone grants, typed multi-edit metrics and public-asset
initial-source visibility. Component fixes remain in the current `remote-dev`
and `coordinator` source checkouts; no component Release is implied.
Timing integrity tests also exposed candidate monkey patches of synchronization
and monotonic clocks that comparator checks alone missed. The performance judge
now captures the original bindings before candidate import and rejects changes
after execution.

## Real model results

All following new foreground cohorts use `home/codex/gpt-6.1-sol`, low effort,
except the quantization L2 cohort, which used medium effort. Harness, task,
model, budget and source identities remain in each receipt. Each condition has
one repetition: these results establish concrete behavior, not population
completion lift or repeated reliability.

### Codex, A3 distributed state

The complete historical model runner executes a tiny real Kimi K3 model with
DP2/TP1, native state updates and HCCL. Six private checks cover idle state,
ordinary-token updates and unaffected rows. Dummy weights and eager execution
are explicit; this is not pretrained accuracy or full serving throughput.

| Condition | Overall | Foreground s | Delivery s | Turns | Recognized NPU commands | Query / explain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0 | PASS | 188.44 | 257.48 | 17 | 2 | 0 / 0 |
| G1 | PASS | 198.31 | 267.01 | 25 | 5 | 0 / 0 |
| G2 explicit | PASS | 237.82 | 316.91 | 24 | 4 | 1 / 1 |
| G3 seed | PASS | 243.92 | 336.68 | 20 | 5 | 1 / 1 |
| G3 evaluation | PASS | 204.54 | 295.47 | 15 | 4 | 1 / 1 |

G3 used two independent native sessions repeating the same incident. Both
captures passed CorpusLinter and delivered through local Git. The evaluation's
actual foreground query/explain returned its own new seed package. The seed
cost is additional; this table does not establish knowledge completion lift.
G0-G2 also provide no observed completion lift, and explicit retrieval took
longer. Source edits during development are fingerprinted rather than hidden.

A typed `remote_multi_edit` measurement omission was fixed after this cohort.
Reparsing the evaluation's existing trajectory gives first/last target edit at
turn 10, 121.334/121.480 seconds, with one recognized NPU command before the last
edit. Original receipts remain unchanged. The seed's unconfirmed pending edit
retains a missing success-only timestamp; the parser does not invent completion.

Evidence: `reports/v0.2.0-native-dp2-low-effort/{g0,g1,g2,g3}/`.

### Claude Code, complete local matrix

The same model and low effort ran through the real Claude Code binary and its
Anthropic protocol. Capture used the original mindie-agent-cc organizer and
GPT-5.6 Luna, not a Codex transcript substitute.

| Condition | Overall | Foreground s | Delivery s | Foreground calls | Query / explain |
| --- | --- | ---: | ---: | ---: | ---: |
| G0 | PASS | 99.43 | 100.55 | 15 | 0 / 0 |
| G1 | PASS | 71.66 | 72.86 | 14 | 0 / 0 |
| G2 explicit | PASS | 99.74 | 110.63 | 14 | 1 / 1 |
| G3 seed | PASS | 185.67 | 209.66 | 21 | 1 / 1 |
| G3 evaluation | PASS | 121.18 | 144.72 | 17 | 1 / 1 |

All five cells have complete foreground usage and empty error lists. Each G3
capture used one metered summary request, passed lint and completed local Git
and independent consumption. The evaluation foreground read its own seed block.
This is functional acceptance; G3 did not improve evaluation time over G0 and
its seed cost must also be counted.

Earlier independent Opus conditions remain separate: some repaired code passed
the judge but exceeded the foreground budget; a stalled stream and a 429 left
unknown outcomes and blocked their dependent evaluations. They were not replayed
or silently converted to passes.

Evidence: `reports/v0.2.0-claude-l2-smoke/` under
`g0-g1-g2-claude-codex-route-low` and `g3-claude-codex-route-low`.

### NPU operator performance

| Condition | Overall | Judge | Operator speedup | Warm eight-step speedup | Workspace delta MB | Query / explain |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| G0 | PASS | PASS | 1.225x | 1.139x | -3.566 | 0 / 0 |
| G2 explicit | FAIL: max turns | PASS | 1.799x | 1.297x | -3.559 | 1 / 1 |

Both candidates passed actual graph capture/replay and numerical checks. G2
produced good final code but exhausted the 40-turn budget. Overall failure is
preserved. All 77 admitted provider requests have complete usage; the next
G2 request was refused before provider admission.

The unmodified historical reference failed the warm-service gate at 0.926x,
despite isolated speedup. Profiling found gate packing, generic host dispatch
and unused intermediate allocation overhead. A benchmark-specific native
RMSNorm reference then reached 1.983x isolated and 1.691x warm, while adding
13.218 MB workspace. Its explicit extra allowance is 16 MiB and total service
scratch cap 32 MiB. These memory budgets were added after the initial measurement
and validated against the preserved raw evidence. This tiny service has no real
KV cache; no KV-capacity or full-model throughput claim is made.
After adding the synchronization/clock identity guard, a final real NPU
source/gold control passed again: 1.985x isolated, 1.635x warm, actual graph
capture/replay and the same workspace bounds. The final task proof binds the
current guard; the original paid cohorts keep their original task identity.

Evidence: `reports/v0.2.0-performance-5312b32d/native-results/results.md`.

### Stale advice and cross-model transfer

`tasks/stale/fused-mc2-deployment` uses complete Qwen3-MoE modules, TP2/EP2 and
HCCL. A constructed deployment retains a now-ineffective legacy override while
the current configuration enables optional fusion. The actual old knowledge
body remains available. Source/gold proof measured the fused/non-fused operator
paths and numerical outputs on two inputs and both ranks: source 0, gold 1.

G2 explicit passed all four checks, read the canonical stale body through actual
query/explain, identified the old advice's version boundary and repaired the
current configuration. It took 191.80 seconds foreground and 320.95 delivery.
G0 also passed the business judge, but its original overall receipt is an
infrastructure failure: initial-source visibility did not support the public
deployment fixture. That scorer defect is fixed; the original result was not
rewritten and cannot be used as a clean completion-lift comparison.

The quantization L2 task executes complete historical EAGLE3 registration/loader
modules and real CPU tensors. It physically excludes the same-origin EAGLE3
package and retains a K3 weight-basis analogue. G0/G1/G2 all passed in
43.56/50.29/108.41 foreground seconds; explicit G2 made two queries and two
explains. This case measured no knowledge speedup. Registry and weight-I/O
boundaries are declared CPU fixtures, not an NPU accuracy claim.

Evidence: `reports/v0.2.0-stale-mc2/` and
`reports/v0.2.0-infra-l2-quarot/`.

## Five-component measurements

The five A3 distributed-state cells above completed five allocation pairs
(ten leases): ten acquisitions, ten activations and ten releases, all acknowledged.
There were 268 acknowledged heartbeats and 149 accepted npu-top samples with
zero rejections. Fifteen diagnostics events were recorded with no logging
failures or dropped records. Component call latency includes SSH and startup:

| Component / observation | Measured result |
| --- | --- |
| coordinator | All 303 requests acknowledged; mean 0.771 s, p95 1.101 s |
| npu-top | 149 samples, mean 3.553 s, p95 3.985 s |
| diagnostics | 15 events, mean 0.0736 s, p95 0.0823 s; zero drops |
| knowledge | Three foreground queries: 12.915-15.396 s including startup; three explains: 0.0061-0.0137 s |
| remote-dev | Real foreground tool/job observations and latency by operation in each receipt; pending requests are distinct from confirmed completions |

All five cells ended with released leases, clean assigned-device PID lists and
sealed remote settlements. Brief NPU activity can fall between observer samples;
actual model/verifier tensor execution is separate evidence from a sampled
AICore percentage. Recognized NPU command counts are bounded command-attempt
proxies, not kernel counts. Tool durations use interval unions.

A2 source/gold smoke also passed through the current five-component controller:
25.93 seconds source failure, 29.77 seconds gold success, intact oracle, ten
accepted samples, zero rejections, both leases released and three delivered
diagnostics events. The two component parser/bootstrap fixes were exercised
here. Its real model smoke comparison also completed:

| A2 smoke condition | Overall | Foreground s | Delivery s | Turns | NPU command attempts | Query / explain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0 | PASS | 161.86 | 220.65 | 15 | 4 | 0 / 0 |
| G2 explicit | PASS | 195.90 | 264.79 | 20 | 5 | 1 / 1 |

Both have empty error lists, released leases, clean NPU postflight and zero
monitor rejections. Their 22/26 monitoring samples and actual NPU verifier runs
cover the A2 process-table layout. This retained RMSNorm smoke is separate from
the new A3 infrastructure cases. Evidence: `reports/v0.2.0-a2-runtime/`.

## Validation and retained history

The daily local quickcheck completed all five G0/G1/G2/G3 seed/evaluation cells.
Single-task G2 delivered in 98.13 seconds; the G3 two-round phase took 251.71
seconds. Whole-cohort development time, including configuration debugging, was
1318.50 seconds; the full matrix is not a five-minute result. The 319.69-second
G1 cell spent 315.15 seconds in model-request intervals and made no remote MCP
calls, so its delay cannot be attributed to remote-dev.

All 58 provider requests have complete usage. Both captures passed lint and
the held-out foreground read its own seed's exact 3351-byte body. Evidence:
`reports/v0.2.0-l2-quickcheck/results.md`. These CPU publication contracts test
the daily engineering loop; the A2/A3 studies provide physical NPU evidence.

The single final full regression returned 810 passed, one failed and one
skipped in 125.71 seconds. The failure was an obsolete remote-settlement test
double; it now checks drain/quiesce/judge ordering at the verifier boundary.
The targeted settlement and remote-verifier recheck passed all 18 tests in
11.57 seconds. The full suite was not repeated. The Linux `/proc` test is the
one macOS skip; the A2/A3 runs exercised the real process-table path.

The final repository-wide Ruff check passed; the repaired test file also
passed its targeted Ruff check. The remote-dev bootstrap regression passed
17 tests with four Linux-only skips, and coordinator's NPU topology parser
passed seven tests including ten subtests. Both component fixes were also
exercised on A2.

One final `uv build --out-dir dist` produced the 0.2.0 wheel and sdist.
Their version metadata and six new implementation modules were checked;
the sdist contains all 14 task manifests and excludes private reports and
virtual environments. CLI help loaded directly from the wheel with the
existing Python environment. This is a packaging check, separate from the
source-based physical NPU acceptance above.

The original 0.1.0 experiments and their failed/unknown attempts remain under
`reports/acceptance-a2-a3-20261017/` and
`reports/benchmark-improvement-20261017/`. Their detailed intermediate build
and reconciliation narratives are not repeated in this document.

## Target-state v0.2.0 systemic closure (`2026-10-18`)

To close the attribution loop across all five MindIE components (`remote-dev`,
`knowledge`, `coordinator`, `npu-top`, `diagnostics`), four systemic confounds
identified in the initial 0.2.0 cohorts were resolved and re-evaluated on real
models (`home/codex/gpt-6.1-sol` foreground, `home/codex/gpt-5.6-luna` capture
summary) and physical Ascend910 A3 NPUs (the A3 cluster, devices `8..15`):

1. **`remote-dev` subreaper exit race & premature first-byte break** (`mindie-agent/remote-dev` PR `#32`):
   fixed `job_status` counting the exiting worker supervisor as a live child after
   `result.json` with `descendants_drained=True`, lowered supervisor polling intervals
   (`5ms` / `10ms`), and added a completion/coalescing grace window in
   `control_job(action="exchange")`.
2. **`coordinator` A2 single-chip `npu-smi info` parser** (`mindie-agent/coordinator` PR `#45`):
   explicitly supports single-chip tables without `Phy-ID` while failing closed on
   ambiguous/mixed layouts.
3. **Symmetric L3 workspace boundary & pre-warmed `knowledge` ReMe index**:
   L3 trials no longer materialize a local `WorktreeSlot` mirror on the client host
   (`local_files_modified = []`), eliminating G0's local-filesystem search shortcut;
   `first_party.consumer()` pre-warms and checkpoints `.reme-index` during
   `mcp_probe` preflight before the foreground clock starts.
4. **Task de-spoilering & natural G3 causal-chain linting**:
   removed direct file-path/formula/legacy-key spoilers from public `instruction.md`
   and `environment/` across `quarot-draft`, `fused-mc2-deployment`,
   `k3-gdn-idle-dp-state`, and `fused-rmsnorm-gate`, and removed the G3-only
   foreground prompt injection while extending `CorpusLinter` to recognize both
   explicit headings and natural assistant-authored causal chains.

### 1. A3 distributed state (`k3-gdn-idle-dp-state`, DP=2 real Ascend NPU)

| Condition | Overall | Foreground s | Delivery s | 1st localize s (turn) | 1st edit s (turn) | Last edit s | Empty polls | `knowledge_query` mean s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 (SSH baseline) | PASS | 174.03 | 243.49 | 66.15 (9) | 113.24 (12) | 113.24 | 0 | — |
| G1 (`remote-dev`) | PASS | 143.17 | 208.25 | 33.00 (5) | 93.12 (8) | 93.12 | 0 | — |
| G2 (`explicit`) | PASS | 183.14 | 252.97 | 20.70 (4) | 127.03 (12) | 127.03 | 0 | 0.699 |
| G3 (`warm-seed`) | PASS | 265.15 | 361.78 | 41.96 (6) | 98.70 (9) | 177.35 | 0 | 1.281 |
| G3 (`warm-eval`) | PASS | 197.95 | 298.83 | 21.90 (4) | 75.12 (7) | 127.45 | 0 | 1.186 |

Key closed-loop findings (`reports/v0.2.0-target-state-20261018/l3-idle-dp/`):
- **G1 vs G0 (`remote-dev` lift)**: With the local worktree leak removed and
  `remote-dev` polling fixed, G1 beat G0 by **-17.7% foreground time**
  (`174.03s → 143.17s`), **1.169x delivery speedup** (`243.49s → 208.25s`),
  and **2.0x faster localization** (`66.15s / turn 9 → 33.00s / turn 5`), while
  `total_empty_job_polls` dropped from **13 (prior v0.2.0 G1) → 0**.
- **G2 localization & pre-warmed retrieval**: First localization improved **3.2x
  over G0** (`66.15s → 20.70s`, turn `9 → 4`), and `knowledge_query` latency
  dropped **21.6x** (`15.1s → 0.699s`).
- **G3 natural experience flywheel (`warm-seed -> warm-eval`)**: Without any
  foreground G3 prompt injection, `warm-seed` passed `CorpusLinter` (`passed`),
  published commit `7446750f...` to local bare Git, and `warm-eval` retrieved the
  captured package (`95567b4d...`) in `1.186s`—reducing foreground wall-clock by
  **-25.3%** (`265.15s → 197.95s`), localization time by **-47.8%**
  (`41.96s → 21.90s`), last-edit time by **-28.1%** (`177.35s → 127.45s`), and
  pre-edit NPU test runs by **-50%** (`2 → 1`).

### 2. A3 stale-knowledge negative-transfer gate (`stale-fused-mc2-deployment`, TP2/EP2 real Ascend NPU)

| Condition | Overall | Foreground s | Delivery s | 1st edit s (turn) | Last edit s (turn) | Exploration s | Empty polls | `knowledge_query` s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 (`remote-dev` only) | PASS | 152.37 | 225.73 | 72.10 (6) | 77.19 (7) | 149.20 | 0 | — |
| G2 (`explicit` knowledge) | PASS | 109.71 | 194.28 | 26.44 (5) | 37.24 (6) | 100.22 | 0 | 0.477 |

After removing the legacy-env spoiler from `instruction.md` and `reproduce.py`,
G2 (`explicit`) retrieved the package in `0.477s`, verified the `v0.27+`
applicability boundary against `ascend_runtime_config.py`, and beat G1 by
**-28.0% foreground wall-clock** (`152.37s → 109.71s`), **-63.3% time to first
target edit** (`72.10s → 26.44s`), and **-51.8% time to last target edit**
(`77.19s → 37.24s`). Evidence: `reports/v0.2.0-target-state-20261018/l3-stale-mc2/`.

### 3. De-spoilered cross-model quantization L2 (`l2-quarot-draft-dispatch`)

| Condition | Overall | Foreground s | Delivery s | Turns | 1st localize s (turn) | 1st edit s (turn) | `knowledge_query` / `explain` s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | PASS | 85.19 | 86.25 | 9 | 7.12 (3) | 40.91 (5) | — |
| G1 | PASS | 82.45 | 83.27 | 8 | 5.21 (2) | 36.22 (4) | — |
| G2 (`explicit`) | PASS | 99.17 | 126.75 | 10 | 27.65 (4) | 65.72 (6) | 1.113 / 0.041 |

Removing the closed-form formula from `quarot_boundary.py` eliminated G0's
43.56s copy-paste shortcut (`85.19s / 9 turns` G0 vs `82.45s / 8 turns` G1), and
ReMe pre-warming reduced G2 first `knowledge_query` latency **13.2x**
(`14.74s → 1.113s`). Evidence:
`reports/v0.2.0-target-state-20261018/infra-l2-quarot/` and unified Hugging Face
dataset export `reports/v0.2.0-target-state-20261018/hf-dataset/` (`10` trial
rows in `trials.parquet`, `14` public tasks in `tasks.parquet`).

Usage is completely metered for the completed new cohorts. Dollar cost remains
unmeasured where no price schedule is configured. None of these results supports
a population Pass^3 estimate or a general causal knowledge-benefit claim.

