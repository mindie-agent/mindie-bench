# Acceptance evidence - 2026-10-17

This is development version `0.1.0`, not a published component or plugin Release.
The evidence below covers real-machine checks, a successful native seed capture,
local delivery and held-out solve, and preserved failed/unknown pilots. It does
not establish a clean native G0-G3 effectiveness estimate or MindIE improvement.
The original 24-cell cohort is qualified historical evidence. A subsequent
15-cell plan exposed two more runner defects and interference from another NPU
workload; its original partial results remain intact. The final C source fixes
those defects and passes local tests. Its first hardware admission was blocked;
after a later clean readback both runners passed fresh admission and started.
The final cohort then encountered recurring foreign occupancy and an upstream
model EOF. Full hardware acceptance for the final source has not passed.
Source-specific evidence is
kept separate from earlier synthetic and native checkpoints.
Unless prefixed otherwise, real-run report paths below are relative to
`reports/a3-153-20261016/`.

## Final implementation and C hardware outcome

Current runtime commit:
`7691a4810eebdebc6fcfcc99b6feb1eceb3097fa`.
Python source SHA-256:
`a4e6245daabd3b132ac209a9286f692811b61634dcb322359e09408ab2bb5995`.
The reviewed source passed **197 tests in 44.52 seconds**, Ruff check and
formatting for 78 files. Logs are `final-e-source-pytest.stdout`,
`final-e-source-ruff.stdout` and `final-e-source-format.stdout`.
`final-e-implementation-source.json` records this identity separately from hardware C.
One runtime change after the C experiment is diagnostic: a seed whose capture completed
but postflight failed is now reported as failed seed prerequisites with the
actual terminal and capture states, rather than "capture failed".
The regression covers completed capture plus infrastructure failure.
Another fixes export identity loss, described in the distribution section;
neither correction was followed by another paid or NPU cohort.

The latest hardware cohort C used commit
`d259695883857646f02c37c89677a82beeffb4ee`, digest
`fd7c8fed8fcccabc754b3274df2a02d53720644195a515e0bd22588cab8c477b`;
its source validation was 182 tests in 44.83 seconds.
`controlled-final-c-source.json` freezes runtime/task files and two configuration
hashes for a new independent 15-cell experiment, at concurrency 3 per runner
and disjoint device pools 8/9/10 and 11/12/13.

`controlled-final-c-preflight.json` records the first blocked admission before either
runner was launched: all selected devices had live owning PIDs and approximately
46 GiB in use, above the explicit 3600 MiB idle allowance. It records zero
launched plans, zero trial receipts, zero model calls and zero NPU test commands
at that check.
Earlier empty-PID observations did not authorize starting on newly occupied
devices. No business process was terminated to make room. This preflight is
neither a failed model attempt nor a successful trial.

`final-owned-resource-readback-20261017/` subsequently found no owning slot
processes and empty device PID inventories. Each runner then performed another
fresh admission check (`controlled-final-c-baselines-admission.json` and
`controlled-final-c-g3-admission.json`) before starting its finite plan.
Both finite runners exited 2. Source, task and configuration hashes still
matched the frozen manifest after termination.

| Group and role | Planned | Receipts | Pass | Infra | Unknown | Not started |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 eval | 3 | 3 | 0 | 2 | 1 | 0 |
| G1 eval | 3 | 3 | 0 | 3 | 0 | 0 |
| G2 eval | 3 | 3 | 0 | 3 | 0 | 0 |
| G3 seed | 3 | 3 | 0 | 3 | 0 | 0 |
| G3 eval | 3 | 0 | 0 | 0 | 0 | 3 |

The 12 original receipts contain 11 infrastructure failures and 1 unknown.
All six G1/G2 cells failed preflight before model admission. Two G0 cells and
all three G3 seeds passed their independent business verifiers, totaling
**150 real NPU cases**, but required postflight failed on foreign occupancy.
Each seed separately delivered actual native Stop, completed one Luna indexer
call, published to local Git and passed independent consumer readback. These
phase successes are preserved without turning overall infrastructure failures
into passes. Their dependent held-out evaluations were never started.

G0 repetition 3 preserves provider ordinal 16: accepted HTTP 200 followed by
EOF without complete usage, journaled as `IncompleteProviderUsage`. The native
client reported unexpected EOF. There was no downstream delivery failure in
this attempt; it is distinct from B's downstream BrokenPipe defect. The
unknown receipt, partial usage and failed postflight remain intact, and no
request was replayed.

`post-c-owned-resource-readback-20261017/` subsequently found no process with
the benchmark slot marker and empty physical-device PID inventories on 8-15,
with 2879-3139 MiB used. The dedicated container/image remained unchanged.
This is a separate final cleanup observation, not a rewrite of the original
failed trial postflights or a guarantee of a stable idle allocation.

There are zero valid final-C attempts: Pass@1 and Pass^3 are null, not zero or
100 percent. Strict comparisons can reject missing preflight fingerprints or
absent G3 evaluations; no control is filled in to manufacture an estimate.
The system's implementation/local validation and these observed real business
phases do not establish a clean final-source G0-G3 result. A stable idle device
allocation and a reliable provider stream are still needed for that acceptance.

## Preserved B cohort: partial results and defects

The preceding runtime was commit `301f9772027a91805af702e62aa54cf93ab7d3c4`,
digest `71f7aa54a01977e27b8b3623d74b13db9085e8ca5696d39b49c2867a384bb260`.
All runtime/task file hashes still matched `controlled-final-r3-source.json`
after both runners terminated with exit 2. This cohort used four slots:
G0/G1/G2 on devices 10/11, and G3 on devices 8/9, each at concurrency 2.

| Group and role | Planned | Receipts | Pass | Infra | Unknown | Not started |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 eval | 3 | 3 | 2 | 0 | 1 | 0 |
| G1 eval | 3 | 3 | 3 | 0 | 0 | 0 |
| G2 eval | 3 | 3 | 1 | 2 | 0 | 0 |
| G3 seed | 3 | 3 | 2 | 1 | 0 | 0 |
| G3 eval | 3 | 2 | 2 | 0 | 0 | 1 |

There are 14 original receipts: 10 pass, 3 infrastructure failures and 1 unknown.
The absent G3 evaluation was blocked by its seed's failed required postflight;
it was never admitted and has no fabricated trial receipt.

G0 repetition 2 reached the broker's 12-command limit. Audit found five patch
or dry-run submissions miscounted because their opaque stdin mentioned tests.
The native trajectory's ten recognized attempts are a separate observation:
seven shared with the broker, two failures before gateway admission, and one
unfinished tool call. They are not ten completed NPU tests. Provider ordinal 32
then encountered downstream `BrokenPipeError` before final usage was read;
the receipt remains unknown with incomplete token metering and no independent
verifier result. `g0-budget-audit-20261017/` retains the original mapping and
RED/GREEN budget regression evidence. The source C predicate now checks only
the command field, consistent with stream/RPC paths.

G2 repetition 1 and G3 seed repetition 3 passed their independent business
verifiers, but required postflight found another workload's NPU processes;
their final infrastructure states preserve the separate business verdict.
That G3 seed also completed native Stop, one Luna indexer call, local Git
publication and independent consumer readback before cleanup failed. G2
repetition 3 failed preflight, before model admission. The other workload
belonged to a different Docker cgroup; `foreign-workload-156-20261017/` retains
the exact PID, parent, cgroup and NPU observations. Its processes were not killed.

Source C also separates downstream delivery failure from the original upstream
request, drains and meters that request within explicit ten-second bounds, and
disables HTTPConnection implicit reconnect after settlement closes a socket.
Provider truncation, incomplete usage or expiry remains unknown. Accepted
usage is retained even if delivery fails; known pre-inference provider
rejections remain infrastructure errors. `provider-drain-fix-20261017/`
preserves real loopback HTTP/TCP-reset RED/GREEN tests; these make zero paid
model or NPU calls and do not retroactively settle B ordinal 32.

G0 has only two valid repetitions: valid-attempt Pass@1 is 2/2, coverage and
all-attempt success are 2/3, and Pass^3 is null. An affected cohort cannot be
made clean by freezing only its successes. Missing runtime identity in a
preflight-failed cell can also make strict comparison reject its full group.
These effects, external interference, one defect family and fixed physical
device pools preclude a clean effectiveness claim from this cohort.

## Real Ascend task soundness

The current task revision uses `max_turns=40`, 12 recognized test-command
attempts, a 600-second trial limit and a 180-second verifier limit. Earlier task
bytes and proofs are retained under `reports/a3-153-20261016/task-revisions/`.
Changing these contracts required new soundness proofs; the old pilots remain
bound to their original task digests.

Both tasks exercise the same upstream RMSNorm defect family, with missing and
null `quant_description` triggers. They use failing source commit
`740e7b9f821230fb1ead96c2ab805cb6018928a9` and the gold fix from
`fb5533a6869f8b02e61811933569661fc055850c` (#14704).

| Current proof | Device on A3 node 156 | Actual result | Cleanup readback |
| --- | --- | --- | --- |
| `soundness-missing-5/` | 8 | Before: exit 1/reward 0; after: exit 0/reward 1; 30 real NPU RMSNorm cases | 3140 to 3141 MiB, no owning PIDs |
| `soundness-null-4/` | 9 | Before: exit 1/reward 0; after: exit 0/reward 1; 30 real NPU RMSNorm cases | 2881 to 2883 MiB, no owning PIDs |

Both runs report zero model calls, no soundness errors, and unchanged runtime
fingerprints. Each passing verifier covers five quantization configurations,
three dtypes, and residual/no-residual behavior. The observed maximum absolute
error is `0.03125`; neither proof reports a speedup.

The task-bound SHA-256 values are:

- Missing trigger: `f54c36bd398cebcbd11c5af8f9995379bf2420f921e971babbedf97bc4360bbd`.
- Null trigger: `cbc15b88d7659a2f223c42d899ea6baca432ae47630af62a722c520fcd1591c8`.

These are two trigger variants of one defect, not two independent engineering
families or a 12-task verified dataset. They load the historical RMSNorm source
class into the pinned installed runtime; this is not full historical vLLM
compatibility, model-weight serving, throughput or multi-host HCCL acceptance.

## Runtime contamination correction

The earlier image
`sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83`
exposed a newer fixed target class, related helper tests and Git history through
its granted runtime paths. Earlier pilots retain their observations and terminal
states, but cannot establish an uncontaminated comparison.

The dedicated derived image is now
`sha256:b79ae5d9b5a63ed16f92efef70e47a55c8b3c38bf17bd7ab988784e7eae9d187`,
in container `mindie-bench-156-verified-20261016`. It replaces the runtime's
`vllm_ascend/ops/layernorm.py` and target unit test with the failing revision's
actual blobs, removes agent-visible Git metadata and stale bytecode, and leaves
the original business checkout intact. The final inventory found one target
class with `fixed_guard=false`, no direct copies, and no remaining target-guard
candidates or visible Git metadata. Generic defensive guards in other runtime
classes were reviewed and retained.

`runtime-projection-1/projection-manifest-final.json` has SHA-256
`068140e12d9ba17c301afa46c372fc2834cba36a7bb3d26c4f07629d8267165e`.
Its `verified-runtime-receipt.json` records zero-model FAIL-before/PASS-after
checks on devices 12 and 13, each with 30 real NPU cases, clean PID readbacks and
unchanged runtime fingerprints. The current proofs above were also rerun on
this image. This is a task-specific inspected projection, not a general
guarantee that arbitrary dependencies are free of solution leakage.

## Hardware and component checks

| Check | Actual result | Evidence boundary |
| --- | --- | --- |
| Device 10 slot canary | Real NPU tensor round-trip returned `[0,1,2,3,4,5,6,7]`; one visible device; exit 0 | `slot-canary-device10-1/receipt.json`; small tensor correctness, not model serving |
| Device 10 cleanup | 3140 MiB before and after, no owning PIDs, unchanged runtime | Complete slot/quiesce/close path exercised |
| Knowledge MCP | Real initialize, tool discovery, query and explain succeeded in an isolated G2 profile; zero model calls | `knowledge-mcp-canary-1/attempt/observed.json`; scripted consumer, not foreground model retrieval |
| Knowledge body readback | Exact referenced block file and 12,439-byte body hashes checked | `knowledge-mcp-canary-1/attempt/verified-readback.json`; retrieval does not certify the historical engineering claims |
| Remote-dev MCP | Version 0.9.5 completed 12 real tool calls over persistent RPC streams; connection reuse observed | `remote-mcp-canary-1/attempt-3/observed.json`; file write/read, short job completion, background status/tail/stop |
| Remote job cancellation | Background job survived earlier requests, then reached `cancelled` with `quiet=true` and final cancelled status | Same MCP canary; server exit 0, errors empty, gateway unknown false |
| Device 11 cleanup | 2883 MiB before and after, no owning PIDs, unchanged runtime | Zero model calls and zero NPU test-command attempts in this MCP canary |

Knowledge readback recorded block-file SHA-256
`d51941474e7ee2604b41b6e99fe20a77f3d41d02a760ccd237f4f602fbb5dd87`
and body SHA-256
`9e31e3a230abb164ca451178daebccf5bbe489608d3f3942d94df4b0715794fe`.
The corpus commit was `cec3249ccd75e7a73cf2f2f776117e0ec385795b`; the
physically filtered view excluded the three declared same-origin packages.

These earlier canaries used node 156 in the requested A3 cluster and the
original `090eeee...` image; they establish the listed mechanisms rather than
an uncontaminated task comparison. Both images report CANN 9.1.0 (compiler
timestamp `20260730_231653901`), torch_npu `2.10.0.post4`, driver `25.5.3`,
Python 3.12.13 and kernel `5.10.0-216.0.0.115.oe2203sp4.aarch64`.

Node 153 itself has not passed acceptance. Docker exec and namespace/umount
operations stalled there. The original request to create
`mindie-bench-153-20261016` remains unresolved; its stdout/stderr are retained
and it must not be reissued as though it never happened. Successful node 156
checks do not establish node 153 recovery. The final bounded read-only snapshot
`153-final-readonly-20261017/summary.json` found the four original tracked
processes still present: the Docker client and runc exec were sleeping; runc
init and bubblewrap remained in D state at `exp_funnel_lock`. Exact-name Docker
inspect returned "No such container" without transport timeout, but no original
completion acknowledgement or exit status was recovered. The receipt keeps
`original_outcome=unknown` and `reconciliation_complete=false`. Both clocks are
retained because the remote snapshot clock was about 127-128 seconds ahead of
local capture. No request was replayed and no cleanup completion is claimed.

L3 requires a dedicated privileged Ascend container for trusted evaluation code.
The tested driver needs capabilities and writable sysfs; device selection uses
environment configuration and cooperative flock leases. PID/network/IPC/UTS
namespaces do not make this a hostile-tenant security boundary. Local profiles
remain subject to their separate Seatbelt access policy.

## Current native model and component identity

The model endpoint and selected component runtimes are already configured for
fresh profiles. The current foreground model is `home/codex/gpt-5.6-sol` with
medium reasoning; the native indexer is `home/codex/gpt-5.6-luna`, through the
local relay with Codex CLI 0.153.4. Personal CLI login files were not copied.
Current component revisions are knowledge
`f81b90a02d27ee5df0b250a517b0df8a31b2ddd6`, remote-dev
`f72c96c75afc5e35401f907b80296bc221bf420a`, and native Codex adapter
`176518026d04a82958350fa1e7230cc61f4417a0`.

`component-v2-manifest.json` binds the private frozen payload at
`/tmp/mindie-bench-components-20261016-v2`. The Codex change projects the unique
complete native summary-model record into the indexer's fresh profile and
hashes the selected bytes into route/policy identity. It does not filter native
warning/error events or copy a personal profile. The isolated component commit
is local; no PR, public merge or component Release is claimed.

Two zero-model native checks passed:

- `native-metadata-profile-preflight-1/observed.json` contains the exact two
  selected model slugs and parent catalog SHA-256
  `89890cad2f32052a8b32b02988372fa6a62e8d955480741bb9daed6158783a65`.
  `metadata-preflight/component-native-projection/observed.json` records native
  `debug models` exit 0, empty stderr, one Luna alias, and full-record equality
  after projection into the summary home.
- `native-instructions-preflight-1/observed.json` records `debug prompt-input`
  under the actual sandbox with `project_root_markers=[]`: exit 0, empty stderr,
  cwd `AGENTS.md` present, and private ancestor instructions absent. This
  resolves the startup discovery warning without relaxing sandbox access.

## Earlier native pilots and reconciliation

Pilots 1-3 used `home/codex/gpt-6.1-sol`, Codex adapter `4e4efca...` and the
original runtime image. They are not cells of the new matched comparison.

These terminal states retain the policy used by the historical runner. Since
2026-10-20, settled budget stops are assessed by the independent judge and budget
compliance is reported separately; the archived receipts are unchanged.

| Original attempt | Preserved terminal state | What the evidence establishes |
| --- | --- | --- |
| `native-g0-smoke-1/` | `completed_fail`, `max_turns` at 20 | External final judge passed, but the capped trial earns no pass |
| `native-g3-smoke-1/` | `infra_failed` in capture | Foreground reached its cap and final judge passed; capture failed during local bare-Git validation before indexer inference (`capture_model_calls=0`) |
| `native-g0-smoke-2/` | `completed_fail`, `max_npu_runs` | The configured four-attempt command budget was exhausted; final judge passed but the capped trial remains failed |
| `native-g3-smoke-2/` | `unknown`, `cancelled` | The persistent-RPC facade deadlocked before remote business submission; original native attempt and reconciliation remain intact |
| `native-g0-smoke-3/` | `unknown`, `uncertain_external_effects` | No final verifier result; remote postflight was clean, which does not turn uncertain business effects into failure or success |
| `native-g3-smoke-3/` | `unknown` in capture | Foreground business and verifier passed on the old runtime; actual native Stop arrived, but strict native event validation stopped the indexer on a metadata error before capture inference (`capture_model_calls=0`) |

For G3 pilot 2, `operator-abort-reconciliation.json` records model-start ordinals
`[1,2]` matched by model-end ordinals `[1,2]`, with no remote business submissions.
The operator stopped that known defective attempt after read-only
reconciliation. Its original `unknown` receipt was not rewritten as success or
treated as a retryable unpaid attempt.

For G3 pilot 3, `capture-readonly-reconciliation.json` confirms actual Stop
delivery and zero capture-broker requests before the metadata error. The
original `unknown` receipt and component ledger remain intact; inference was
not replayed. The metadata projection correction is a new component revision
used by a fresh attempt.

## Real native seed capture and held-out solve

`native-g3-smoke-4/rmsnorm-missing-description_G3_1_a3a45e73cb17/trial_receipt.json`
is `completed_pass`, with no receipt errors, a clean remote postflight and
external verifier reward 1. The verifier covered 30 real NPU cases with maximum
absolute error `0.03125` on the projected image. The foreground used
GPT-5.6-Sol and the new component payload. It made a real knowledge query and
remote-dev source accesses; that seed's query does not demonstrate reuse of its
own later capture.

Its `profile/full-loop-receipt.json` records:

- Native Stop delivered for session `01a14562-f023-7a43-ae3d-7ec920821176`.
- Deterministic public-transcript body capture with zero body-model calls.
- One completed native Luna/LangMem indexer call: 11,234 input and 393 output
  tokens, 11,627 total, with zero unknown calls.
- Local Git publication revision
  `1fbecf935f2d4ac2be64fbd3bf46f65689ece43b`; an independent consumer checked
  that same revision, found the package and read one returned body reference.
- Package ID `2086c34558c9e9a66fdcf84ce1718af9dea7d2b0334aa859891fcfe9fa100ae8`
  and package hash
  `855c79202c1e872ab3edf331887874a8e03d8268f755715942198a906973ad4a`.

The full-loop receipt SHA-256 is
`de90bffb9397df76b4cff456e0bdec1e30327d237cdf2843106370eeb308a020`.
This proves native seed capture, indexing and local delivery for this attempt.
Independent consumer readback alone does not establish foreground use of the
material; the dependent evaluation provides separate observations below.

The seed receipt binds benchmark source digest
`ba47843a2cb67bb618777185afd2691a4da3d35d065b844e751b4229cf562ea3`,
an intermediate smoke checkpoint. Its dependent held-out evaluation
`native-g3-smoke-4/rmsnorm-null-description_G3_1_ae4eb0991e42/trial_receipt.json` is also
`completed_pass`, with no receipt errors, reward 1, maximum absolute error
`0.03125` and clean NPU postflight. The foreground made one real knowledge query
and one explain request. Its `profile/knowledge-store/retrieval-observations.jsonl`
records the seed package and exact block/body hash returned and read:
`3379363d0ee2a1a6e4013c4e640c33f6d5dc234a738a462b752c32574c21dd2c`
at `f2feb099ffc23c011fdc4312177ff0687da87c7f094bd0f96728ed6e3f184345`.
The evaluation's input corpus hash equals the seed's captured corpus hash.

The evaluation then completed its own native Stop capture and one Luna indexer
call (11,201 input plus 332 output tokens). Its private local Git revision
`b52cf20f5de07d3bd05e9cb0ff9da9b69da1146a` was read by an independent consumer;
the evaluation full-loop receipt SHA-256 is
`59685001c7b5e9338217151ffcf10ec73faf749e50989dc5cf4c76990eb43da6`.
This evaluation capture does not advance the seed corpus.

The evaluation source digest is
`4d2421e983702df0b45df44951adfd11b4c3834ab0b183847185b7488d53d371`,
different from the seed's checkpoint because development continued. These
receipts establish one real native mechanism sequence, not a frozen controlled
G0-G3 matrix or a causal improvement from knowledge. The summary has one seed
and one evaluation; Pass^3 remains null. Dollar costs remain unmeasured without
explicit model prices.

## Historical qualified 24-cell cohort

`controlled-matrix-r3-1-readback.json` records 24 terminal receipts, all
`completed_pass`, with 720 real NPU verifier cases, clean NPU postflight and
complete token metering for all attempts. It binds the previous source commit
`ae9d2feb653321992e1d6a552bf50c4fff60a688` and digest
`59688ee76c5d674def5f058255fea4b13581aaa90663a964cebc544a246617f4`.
Complete token metering does not supply missing dollar prices.

One G0 seed,
`rmsnorm-missing-description_G0_1_1396ad42e789`, encountered native openpty
denial from the runner's Seatbelt policy. The model later completed and the
external judge passed; this does not remove the runner defect or its effect on
the trajectory. Two G0 evaluation stderr observations were separately classified
as recoverable local patch attempts on a remote path. Original receipts remain
unchanged. This cohort is retained for mechanism/repetition evidence and is not
advertised as a clean native-harness effect estimate.

`controlled-matrix-r3-1-g3-audit.json` verifies all three independent G3
lineages: each held-out foreground made one query and one explain call reading
its own seed's exact block/hash, and each seed/eval pair delivered actual Stop
events, completed native indexing/local Git delivery and had clean NPU
postflight. The corpus handoff was verified for every lineage.
`observed-g3-retrieval-r3-score.json` reports recall@3, reciprocal rank and
NDCG@3 of 1.0 for those three observations. These same-family near-exact queries
are a narrow secondary diagnostic, not general retrieval quality, three
independent defect families or proof of causal engineering improvement.

## Earlier A/B fixes and bounded checks

- Native PTY: `pty-seatbelt-1/diagnosis.md` preserves both RED stages: initial
  openpty denial and later tcsetattr denial after the first policy delta.
  The final policy grants read/write/ioctl on ptmx and only extension-owned
  slave terminals. `final-fixes-targeted-2.stdout` records 33 passing tests,
  including openpty, tcgetattr/tcsetattr, isatty and real bytes, while an owned
  preexisting PTY, an outside secret and unauthorized listening loopback remain
  denied. This OS check alone is not corrected-source native-model acceptance.
  `native-pty-rpc-1/rpc-observed.json` adds actual Codex 0.153.4 host evidence:
  a fresh G0 profile under the final Seatbelt used standalone
  `command/exec` with `tty=true`, then `command/exec/write` after ready.
  All three streams were terminals; input echo and child readback succeeded.
  Child/server exited 0, native stderr and errors were empty. No thread/turn,
  model, remote or NPU operation was requested. This verifies the native PTY
  path at source B, separately from model-driven acceptance of later sources.
- Required seed resources: compare now receives full per-group rows and applies
  the selected role internally. Matched outcomes remain eval-only by default;
  required G3 seed token, delivery and cost observations appear separately with
  coverage/completeness. G0 seed work is not setup. Missing seed/capture or
  unknown outcomes withhold completeness; unpriced comparison dollars stay
  null. Complete manifests or frozen inventories still enforce individual-cell
  completeness beyond the observed cluster/repetition seed coverage.
- Export import isolation: the redaction subprocess now runs with Python `-I`.
  The original actual export from a directory containing `json.py` executed that
  module and failed; it was retained as a visible import-isolation defect.
  `export-import-preflight-1/final-export-observed.json` records a successful
  B-checkpoint export of six real prior-cohort receipts from the same shadow
  directory: canary not executed, native component redactor used, Gitleaks
  passed, zero model calls and no upload. This checks B's import isolation using
  older receipt data; it is not a final-cohort result or final wheel build.
  The later identity audit separately rejects this export's collapsed IDs.

The wheel/sdist and isolated-wheel smoke bound by
`frozen-source-build-hashes.json`, and the six-row Parquet readback at
`g3-export-readback.json`, belong to the previous `ae9d2fe` checkpoint.
They do not establish packaging of a later source. The separate C package
preflight and final distribution records below retain their own identities.

## Final distributions and staged export

Final wheel/sdist artifacts are in `dist/mindie-bench-final-delivery-20261017/`.
`final-artifact-readback.json` records their actual SHA-256 values, archive
privacy checks, isolated installation, CLI validation and installed Python
implementation digest. The final digest is
`a4e6245daabd3b132ac209a9286f692811b61634dcb322359e09408ab2bb5995`;
a package version alone does not establish this match. The separate earlier
C preflight exercised three passing zero-model synthetic replay trials.

`dist/mindie-bench-c-evidence-20261017/` contains local redacted JSON, a
12-row Parquet file and a static leaderboard for the **C cohort's actual
receipts only**. It preserves eleven infrastructure failures and one unknown;
the three blocked evaluations have no invented rows. The separate
`controlled-final-c-audit/audit-inventory.json` records all 15 planned cells and
the blocked coverage. `final-export-readback.json` binds the installed exporter,
first-party redactor, Gitleaks gate, Parquet round-trip, file hashes and
`uploaded=false`. It is a failure/coverage evidence export, not a successful
baseline, a final-implementation hardware result or a hosted dataset.

The first D-source export passed redaction/Gitleaks and Parquet row-count
checks, but independent readback found all twelve native trial IDs collapsed
to `[REDACTED_SECRET]`. It is preserved as rejected output at
`export-redaction-collision-20261017/`. The historical six-row G3 exports and
three-row synthetic export have the same identity defect: their privacy and
row-count checks remain observations, but those outputs must not be used as
analysable datasets. Original private receipts remain intact.

The corrected exporter generates stable SHA-256 public trial IDs before the
unchanged first-party redactor and validates row count, unique IDs and thirteen
typed task/outcome identity fields afterward. It refuses damaged output
without restoring redacted values. `export-identity-fix-20261017/` retains the
original exporter and RED/GREEN tests. The final dataset readback checks all
twelve anonymous IDs against the private originals without exporting a raw-ID
mapping. No rejected or final dataset was uploaded.

## Parallel development validation

The scheduler now interleaves ordinary cells and independent G3 lineages in one
thread pool. All use the same device allocator and cancellation signal.
Seed capture must complete before dependent evaluation; a completed failed seed
can contribute failure evidence. Only seed corpora advance the lineage;
held-out evaluation captures do not flow into later evaluations.

Localization is observable only through recognized typed source-access events;
shell-only `sed`, `cat` or Python reads can leave it null. Comparison retains
the max-turn penalty as `localization_penalized_turns_ratio`, reports
`localization_observed_pairs`, and withholds `localization_speedup` unless every
matched valid pair has observations on both sides. Asymmetric visibility must
not become an effectiveness claim.

Focused local stream/receipt checks passed 15 tests, including real local
subprocess ready-before-stdin handshakes, repeated RPC, stderr/EOF, 8 MiB
backpressure, lost-acknowledgement blocking, and late-result/cleanup separation.
These tests make no model or NPU calls. The real MCP canary separately validates
the installed remote-dev contract on node 156.

Kimi supplied the local snapshot patch for symlink-resolved worktree metadata
and NUL-delimited untracked-file reporting. Its three projection regression
tests passed; the patch and transcript remain in `parallel-review/`.
Grok's recorded code review identified six implementation issues and two native
host questions requiring real evidence. The review drove corrections and is
retained at `parallel-review/grok-review.md`; it is not native-loop acceptance
or the public knowledge repository's Grok review.

## Historical synthetic checkpoint

The earlier synthetic release-candidate artifacts bind Python source digest
`0d03c31600cce24f80f931de832aa9c69aa39317f7d4c935b0d0c78cb5731c29`.
At that checkpoint, 79 tests passed in 25.89 seconds, Ruff check/formatting
passed, three synthetic L2 G0 trials passed, and their frozen identity comparison
was produced. Those source-bound results are not the final validation of the
current working tree.

Inspectable historical artifacts include:

- `reports/release-candidate/run/`, `reports/release-candidate/baseline/`, and
  `reports/release-candidate/comparison.md`.
- `reports/native-version-smoke-final.json`: isolated version checks for
  `codex-cli 0.153.4` and Claude Code `2.1.269`.
- `dist/hf-release-candidate/`: local redacted JSON, three-row Parquet readback
  and static HTML; first-party knowledge 0.11.0 redaction plus Gitleaks 8.30.1.
- Original synthetic wheel/sdist CLI-help and archive privacy checks.
  Later artifacts with the same version require their own source-bound hashes.

These artifacts contain synthetic receipts and were not uploaded. They must not
be presented as packages or baselines built from later implementation changes.
Public source discovery scanned 661 packages, matched 497 domain heuristics and
produced 12 candidate dossiers; those dossiers are not verified executable tasks.

## Remaining acceptance boundaries

1. Obtain stable device availability and reliable provider completion before a
   new controlled hardware acceptance. The final C cohort did not pass; its
   observed phase successes and later empty device inventory do not repair
   its original failures. Preserve the historical cohorts' qualifications.
2. Expand beyond this single defect family before claiming general engineering
   gains. General retrieval quality, comparative indexing overhead, hardware
   performance gains and comparative total cost need additional measurements.
3. Reconcile node 153's uncertain operations before reusing their identities.
   Do not reset devices, reboot hosts or disturb unrelated workloads as an
   implicit benchmark recovery step.
4. The final diagnostic revision has local tests/distribution checks; its own
   full hardware cohort has not been accepted. Do not attribute the C or older
   cohorts to its different implementation digest.
5. Local Git delivery does not validate GitHub PR creation, public Grok review or
   merge. GitHub CI, Harbor integration, native Windows execution, multi-host
   HCCL and Hugging Face hosting/upload are not accepted.

Parent design principles revision:
`240248b18996904490cca1e1ecceb8d9ecd83e03`. The final implementation review
belongs in `docs/design-principles-review.md` with its actual source revision
and remaining evidence gaps.
