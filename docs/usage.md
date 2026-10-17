# Running the system

## Zero-model demonstration

```sh
uv sync --locked --all-extras
uv run mindie-bench demo --output reports/demo
uv run mindie-bench score --input reports/demo/run
```

This creates a small source repository, runs three independent synthetic G0
trials in real OS sandboxes, exercises parent-side CPU replay and an external
verifier, freezes receipts, and writes JSON/Markdown identity comparisons.
The fixture is prominently labelled synthetic. It measures no MindIE lift.
Existing run directories are never overwritten or resumed.

## Native experiment

The real task runbooks are `tasks/ascend/README.md` and `tasks/cpu/README.md`;
the routing/retrieval evidence contract is `tasks/L1-L2-acceptance.md`.
Copy an experiment template from `examples/` and fill in the exact installed
binary version, pinned model, model endpoint, and component commands. The model
API key must already exist in the explicitly named environment variable. The
runner neither reads nor copies personal CLI login files.

```sh
uv run mindie-bench validate --tasks /absolute/tasks --config experiment.toml
uv run mindie-bench run --tasks /absolute/tasks --config experiment.toml \
  --tier l2 --groups G0,G1,G2 --repetitions 3 --concurrency 4 \
  --output reports/controlled-run
```

Use exact model deployments, not moving aliases. Record provider-side sampling
or reasoning settings in the deployment identity. No universal `temperature`
flag is invented for CLIs/models that do not support one. Explicit prices are
optional: without them, dollar metrics remain unmeasured.
Model endpoints require HTTPS, except numeric loopback HTTP for a local relay.
The parent broker holds the selected credential; trial processes receive a
broker-only placeholder.
For native Codex aliases that require a custom catalog, set
`harness.model_catalog_path` to a UTF-8 JSON file with native
`{"models":[...]}` records. Export full records using
`codex debug models --bundled`; record the actual export CLI binary/version
as well as the execution CLI, and verify the latter accepts the catalog under
the real sandbox. A slug-only list is insufficient. The selected
model slugs must match the configured foreground and capture routes. The
catalog is copied into the fresh profile and its hash enters the fingerprint.
Cache writes are a subset of non-read input tokens; when present, they require
an explicit `cache_creation_usd_per_million` rate. Missing cache-write pricing
withholds the dollar estimate instead of silently charging the base input rate.

Native G1 requires a real remote-dev MCP server. G2/G3 also require a real
knowledge server; native trials initialize each required server and inspect its
tool catalog before the model starts. Declare exact component revisions.
`runtime_read_paths` must grant only the installed interpreters, libraries and
component source needed by those commands. Do not grant a whole home directory,
business source repository, or benchmark task tree.
Inspect granted runtimes for the task's fixed source, helper tests, Git history
and stale bytecode as well: a clean source snapshot cannot hide a solution in an
installed runtime. Record any task-specific image projection and rerun soundness
against that exact image.

MCP server args/env support `{corpus}`, `{profile}`, and `{local_remote}`.
These are explicit integration points for the selected component's native
configuration, not environment variables claimed to be understood by every
MindIE version. For example, a knowledge adapter must bind its actual retrieval
configuration to `{corpus}` and rebuild its index from this filtered view.
The benchmark does not install plugins or change your saved production settings.
On macOS the runner's Seatbelt policy supports native PTY allocation and
termios using the sandbox's PTY extension. The regression checks actual terminal
I/O and keeps an owned preexisting PTY, an outside secret and unauthorized
loopback inaccessible.

`skills_path` is a directory of native skill folders; each contains `SKILL.md`.
It is copied into the fresh Codex/Claude skill directory. Skill use remains
advisory. Put task-specific trigger expectations in L1 tasks rather than forcing
retrieval in all engineering tasks.

## Task contract

```toml
schema_version = "1.0"
[task]
instance_id = "a-real-bug"
domain_category = "graph_and_kernel_debugging"
evidence = "candidate"
[mindie_knowledge]
corpus_id = "ascend"
knowledge_gradient = "analogical"
same_origin_task_hashes = ["EXACT_SOURCE_PACKAGE_ID"]
analogical_task_hashes = ["RELATED_PACKAGE_ID"]
target_files = ["vllm_ascend/worker/model_runner.py"]
[budget]
max_turns = 25
max_npu_runs = 4
timeout_sec = 1080
[environment.l2]
repo_path = "../repositories/vllm-ascend"
base_commit = "REPLACE_WITH_EXACT_40_HEX_SOURCE_COMMIT"
replay_commands = ["npu-replay-test"]
```

Relative `repo_path` resolves against the task directory. `instruction.md` and
external `verifier/test.sh` are required. Only `environment/` public assets are
copied into the source snapshot; they cannot overwrite tracked source.
`oracle/gold_patch.diff` and `oracle/replay_rules.json` stay outside it.
The runner removes root `.claude`, `.codex`, `.kimi`, `.gemini`, `.agents`, and
`.mcp.json` configuration from its owned snapshot and records this projection in
the trial fingerprint. It preserves `AGENTS.md` and the original source
repository. Final patches exclude these projection deletions. Public assets
cannot reintroduce the reserved configuration paths.
Fresh Codex config sets top-level `project_root_markers = []`; workspace
`AGENTS.md` still applies, while discovery does not read private ancestor
instructions. Sandbox access remains restricted.

Replay rule JSON has `schema_version: "1.0"`, `checks` containing confined
`path` plus `contains`, `not_contains`, or `regex`, and `pass`/`fail` objects with
`stdout`, `stderr`, and `exit_code`. The external verifier should test actual
CPU-observable behavior, not just repeat the replay string matcher.

The verifier runs in the final workspace. It can use
`MINDIE_BENCH_WORKSPACE`, and for local L1/L2 tasks,
`MINDIE_BENCH_TRAJECTORY` for a read-only normalized tool trajectory.
Its last nonblank line may be:

```json
{"reward": 1.0, "speedup_ratio": 1.38, "max_abs_error": 0.0001, "cosine_similarity": 0.9999}
```

Return exit 0 for reward 1; return nonzero with explicit reward 0 for a wrong
solution. Exit nonzero without a valid result for a broken environment.
Performance numbers are verifier observations; none are inferred from logs.

## Soundness and mining

```sh
uv run mindie-bench mine --corpus /absolute/public-corpus \
  --output tasks/new-candidates --limit 12
uv run mindie-bench soundness --task /absolute/candidate-task \
  --config experiment.toml --tier l2 --output reports/soundness-attempt --promote
```

Mining records the exact public source commit, package/block hashes, original
source status, candidate domains, and missing evidence. It produces
`candidate.json`, not an executable task that pretends to be verified.
Supply a failing source revision, a minimal public instruction, independent
verifier, gold fix, and transfer labels before soundness.

Successful soundness records FAIL before and PASS after the gold patch, with
zero model calls. Promotion is explicit. The proof binds the task bytes and tier;
editing the task invalidates it. Synthetic fixtures stay synthetic.

## G3 local sequence

Declare `task.role = "seed"` or `"eval"` and a shared `task.cluster`.
Each repetition gets its own initial corpus and seed lineage. G3 requires
`groups.G3.capture_command`, a local adapter for the selected installed native
capture implementation. Its only optional network access is the inference
broker; it cannot publish publicly. `capture_model` pins an indexer model
(default: the foreground model), and `capture_max_model_calls` defaults to one.
The adapter's native model alias, catalog entry, and capture broker model must
agree; the installed native indexer may require its own small-model alias.
The selected Codex component must project the unique complete summary-model
record from the parent catalog into its own fresh home and declare
`model_catalog_json` there. Changing only the foreground profile's catalog does
not configure the indexer. Supplied unreadable, malformed or ambiguous metadata
fails explicitly; native metadata warnings and error items remain visible.

G0/G1/G2 cells and independent G3 lineages share the `--concurrency` thread pool,
cancellation signal, and L3 device allocator. Each G3 lineage is keyed by
repetition, cluster, and corpus ID. Its seeds run before its held-out evaluations;
different lineages can run concurrently. A seed may be a completed engineering
failure: its captured failure evidence remains useful reference material.
Advancing the lineage requires a completed seed capture and a known business
outcome, not a passing solution. Failed or unknown seed capture stops dependent
evaluations and raises an error while preserving receipts and a partial summary.
The final seed receipt must be `completed_pass` or `completed_fail`; a required
postflight infrastructure failure blocks its evaluations even when capture
completed. The matrix error reports both terminal and capture states.
Only seed captures advance the warm corpus. An evaluation's own capture remains
in its private trial and never becomes input to the next held-out evaluation.

The adapter reads one JSON object on stdin:

```json
{"messages":[{"role":"user","text":"..."}],"destination":"/isolated/corpus",
 "local_remote":"/isolated/local_knowledge.git","instance_id":"seed","cluster":"family",
 "native":{"harness":"codex","session_id":"actual-native-id",
 "transcript_path":"/isolated/native-session.jsonl","cwd":"/isolated/workspace",
 "stop_event_path":"/isolated/native-stop.json"}}
```

For Codex, the runner enables only its event recorder in the fresh G3 profile.
It uses the actual native thread ID to locate the transcript and requires a
matching native `Stop` event. Personal hooks and profiles are not copied.
The `native` object accompanies the general adapter payload; the bundled
first-party capture adapter requires it and currently supports Codex only.

It must deterministically redact before saving, invoke the intended installed
capture/indexing code, write actual Markdown into `destination`, use only
`local_remote` for any Git publication, and return:

```json
{"status":"completed","tokens":{"total":0,"complete":true},"estimated_cost_usd":0}
```

The runner checks that the corpus actually changed and includes summary/indexer
usage in the trial's cost. Missing metering withholds cost per resolved task.
The adapter drains the native indexer's due batches rather than treating the
first batch as completion. Failed, cancelled, or unknown indexing cannot be
silently skipped or resubmitted.

`mindie_bench.integrations.first_party` exposes `knowledge-mcp` and `capture`.
Run it with the selected component interpreter and frozen runtime payload.
`knowledge-mcp --corpus ... --state ...` installs the physically filtered
Markdown view with the first-party `MaterialStore` and rebuilds ReMe retrieval.
It records actual query and explain references in the private profile.
`capture` takes `--codex-scripts`, `--codex-bin`, `--summary-model`, `--gitleaks`,
and a consumer `--query`. It validates native event/transcript identity, invokes
the native transcript extractor and knowledge engine, runs the native small-model
worker with LangMem, and retains deterministic redacted bodies plus indexing
usage. It then pushes to the profile's fresh local bare Git repository, clones
it independently, verifies the commit and package hashes, rebuilds retrieval,
and reads the returned references before updating the warm corpus.

These are separate evidence boundaries: local Git delivery is not a GitHub PR,
Grok review, or public merge; the adapter's consumer readback is not evidence
that a foreground evaluation queried knowledge. A custom offline adapter also
does not establish native Stop delivery. Read `docs/acceptance.md` for completed
real runs rather than inferring acceptance from adapter availability.

## L3 remote prerequisites

Use an existing dedicated privileged Linux Ascend container for trusted
evaluation tasks, with:

- Python 3.11+, Git, bash, flock, bubblewrap, PID/network namespaces, and pidfd
  support. Missing primitives fail before any model request.
- Source at `remote.repo_path` containing each exact task base commit.
- Explicit runtime bind paths for the installed CANN/Python/NPU dependencies.
- The driver-compatible device and writable sysfs view. The worker uses separate
  PID, network, IPC and UTS namespaces without a user namespace, retains container
  capabilities, and binds the container's `/sys` read-write. Its `/tmp` and `/var`
  are private tmpfs mounts; runtime paths are read-only.
- An operator-validated, read-only `remote.probe_command` returning JSON:

```json
{"0":{"used_mb":0,"pids":[],"model":"Ascend910B"},
 "_environment":{"image_digest":"sha256:...","cann":"exact version",
 "torch_npu":"exact version","driver":"exact version","python":"exact version","kernel":"exact version"}}
```

No guessed `npu-smi` text parser substitutes for measured device allocation.
The probe must report actual allocated-device memory and owning PIDs. Native
NPU topology, mapping and probe correctness need their own real-machine evidence.
Set the admission threshold from the measured idle driver baseline, and still
require no owning PIDs. For example, some A3 environments retain several GiB of
driver memory while idle; the example's 500 MiB is not a universal threshold.

```toml
[remote]
host = "evaluation-host"
container = "dedicated-benchmark-container"
repo_path = "/repositories/vllm-ascend"
slots_root = "/tmp/mindie-bench-slots"
devices = [0, 1, 2, 3]
memory_limit_mb = 500
probe_command = ["/opt/bench/bin/npu-probe-json"]
runtime_read_paths = ["/usr/local/Ascend", "/opt/python"]
```

Declare `[environment.l3]` with the same exact source commit and the task's
`npu_model` / `npu_count`. Run with `--tier l3`; the device pool admits only
nonoverlapping allocations. Remote `flock` protects against other runner
instances. Agent/MCP endpoints must address `bench-slot` directly and use the
provided remote workspace. The SSH facade supports one-shot commands and
persistent noninteractive stdio RPC streams into that slot, so installed
remote-dev connections can send several requests before stdin EOF. These streams
are ephemeral state owned by the current trial. Input and each exchange's output
are bounded to 8 MiB; output is drained while input is written to avoid pipe
backpressure deadlocks. A terminal stream exit is reported only after both
stdout and stderr have been drained. Lost acknowledgements or worker errors stop
subsequent execute, stream, and quiesce writes; cleanup retains late frames
separately.
Device nodes come from the dedicated container, while device selection uses
`ASCEND_RT_VISIBLE_DEVICES` and cooperative leases. This is not a hostile-tenant
device-security boundary. Do not run untrusted code or share this container with
unrelated workloads.
L3 verifiers are self-contained bash scripts sent after agent quiescence.
Interactive SSH, tunnelling, SCP/SFTP, multi-host HCCL and native Windows workers
are outside this implementation.

`max_npu_runs` caps recognized test-command attempts using command patterns such
as `pytest`, `torchrun`, and `python ...test_...`. It does not count hardware
kernel launches or detect every possible NPU program. Reports retain this limit
and the observed attempt count rather than claiming exact hardware utilization.
One-shot SSH, direct SSH stream commands and remote RPC command fields use the
same command-pattern contract. Stdin, uploaded worker code and file/patch bodies
are opaque data and do not consume the budget merely by mentioning a test.
Programs executed through stdin are not inferred to be tests when their command
does not match this contract. A recognized command still counts as an attempt
if an earlier shell step fails or pytest only collects tests.

`remote_test_budget_invocations` counts admitted recognized remote commands.
`remote_test_invocations` counts native-reported matching tool attempts,
including failed or unfinished calls. These values can differ when one tool
call contains multiple SSH commands, fails before reaching the gateway, or has
an unfinished/rejected remote submission; neither means completed NPU tests.

## Offline scoring, freezing and comparison

```sh
uv run mindie-bench freeze --input reports/g0 --output baselines/exact-g0
uv run mindie-bench compare --baseline baselines/exact-g0 \
  --candidate reports/g2 --output reports/comparison
uv run mindie-bench score-slices --kind retrieval --k 3 --input observations.jsonl
```

Select one group per comparison input and the same task/tier/repetition matrix.
An identity comparison is allowed but establishes no improvement. Frozen
receipt hashes are verified on read. Do not collect unrelated experiment
directories into one scoring input.
`freeze` rejects unknown or infrastructure-failed attempts, and receipt loading
checks a supplied experiment manifest for missing cells. Synthetic/replay
baselines retain their evidence label. Preserve the complete original experiment
and its coverage; selecting only successful receipts does not turn an affected
cohort into a clean baseline.
Keep full per-group inputs, including G3 seed receipts. `compare` defaults to
`--role eval` internally; `--role seed` compares seed outcomes separately.
Pass rates, paired lift and the selected-role time ratio retain that role's
matched cells. The resource section additionally includes required G3 seed
tokens, delivery seconds and dollar costs; G0 seed work is not setup.

Inspect `seed_setup_complete`, `token_metering_including_seed_complete`,
`delivery_timing_including_seed_complete` and `cost_including_seed_complete`
alongside the observed lower bounds and `cost_per_resolved_task_including_seed`.
Missing required seeds or failed/unknown capture keep completeness false;
unpriced dollar fields stay null. Cluster/repetition seed coverage cannot infer
every missing individual seed task, so keep the complete run manifest or frozen
receipt inventory. The sum of delivery seconds includes each trial's work and
is not the concurrent scheduler's elapsed wall time.

Localization uses typed read/grep/edit observations. Shell-only source access
can leave `turns_to_first_localization` null even when the source was inspected.
Comparison retains max-turn penalties in
`localization_penalized_turns_ratio`, reports `localization_observed_pairs`,
and leaves `localization_speedup` null unless every matched valid pair has
observations on both sides. Do not interpret asymmetric visibility as a
localization improvement.

Slice rows require unique `sample_id` and consistent `evidence`. Retrieval rows
contain `ranked_ids` and nonempty `relevant_ids`; routing rows contain explicit
boolean `expected_trigger` and `actual_trigger`. These are secondary metrics.

## Local publication staging

```sh
uv run mindie-bench export-hf --input reports/demo/run --output dist/hf-demo \
  --knowledge-python /absolute/installed-knowledge-runtime/bin/python \
  --gitleaks /absolute/bin/gitleaks
```

The output includes JSON receipts, leaderboard data, Parquet, and an evaluation
manifest. Both first-party redaction and the fresh Gitleaks gate are required.
Native trial IDs become stable SHA-256 public identifiers before redaction;
private receipts retain their original IDs. The manifest describes this scheme.
After redaction, row count, unique public IDs and core task/outcome identity
must still match; a scanner pass does not establish data integrity. Missing
dependencies, identity corruption, scan findings, corrupt input or failed
Parquet readback leave no published output directory. This command does not
upload anything or restore a private value removed by the redactor.
The installed component redactor runs with Python `-I`; export does not import
modules from the invocation directory. The actual shadow-directory export
preflight and its six real input rows are recorded in
`docs/acceptance-pre-g-history.md`; that checkpoint is not relabeled as G
runtime acceptance.
