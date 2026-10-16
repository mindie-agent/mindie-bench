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
Cache writes are a subset of non-read input tokens; when present, they require
an explicit `cache_creation_usd_per_million` rate. Missing cache-write pricing
withholds the dollar estimate instead of silently charging the base input rate.

Native G1 requires a real remote-dev MCP server. G2/G3 also require a real
knowledge server; native trials initialize each required server and inspect its
tool catalog before the model starts. Declare exact component revisions.
`runtime_read_paths` must grant only the installed interpreters, libraries and
component source needed by those commands. Do not grant a whole home directory,
business source repository, or benchmark task tree.

MCP server args/env support `{corpus}`, `{profile}`, and `{local_remote}`.
These are explicit integration points for the selected component's native
configuration, not environment variables claimed to be understood by every
MindIE version. For example, a knowledge adapter must bind its actual retrieval
configuration to `{corpus}` and rebuild its index from this filtered view.
The benchmark does not install plugins or change your saved production settings.

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

The adapter reads one JSON object on stdin:

```json
{"messages":[{"role":"user","text":"..."}],"destination":"/isolated/corpus",
 "local_remote":"/isolated/local_knowledge.git","instance_id":"seed","cluster":"family"}
```

It must deterministically redact before saving, invoke the intended installed
capture/indexing code, write actual Markdown into `destination`, use only
`local_remote` for any Git publication, and return:

```json
{"status":"completed","tokens":{"total":0,"complete":true},"estimated_cost_usd":0}
```

The runner checks that the corpus actually changed and includes summary/indexer
usage in the trial's cost. Missing metering withholds cost per resolved task.
A custom offline adapter is an integration seam, not evidence that production
Stop-hook delivery or public publication has been accepted.

## L3 remote prerequisites

Use an existing dedicated Linux Ascend container with:

- Python 3.11+, Git, bash, flock, bubblewrap, PID/network namespaces, and pidfd
  support. Missing primitives fail before any model request.
- Source at `remote.repo_path` containing each exact task base commit.
- Explicit runtime bind paths for the installed CANN/Python/NPU dependencies.
- An operator-validated, read-only `remote.probe_command` returning JSON:

```json
{"0":{"used_mb":0,"pids":[],"model":"Ascend910B"},
 "_environment":{"image_digest":"sha256:...","cann":"exact version",
 "torch_npu":"exact version","driver":"exact version","python":"exact version","kernel":"exact version"}}
```

No guessed `npu-smi` text parser substitutes for measured device allocation.
The probe must report actual allocated-device memory and owning PIDs. Native
NPU topology, mapping and probe correctness need their own real-machine evidence.

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
provided remote workspace; one-shot SSH is brokered into that isolated slot.
L3 verifiers are self-contained bash scripts sent after agent quiescence.
Interactive SSH, tunnelling, SCP/SFTP, multi-host HCCL and native Windows workers
are outside this implementation.

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
Missing dependencies, scan findings, corrupt input or failed Parquet readback
leave no published output directory. This command does not upload anything.
