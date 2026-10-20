# RSI-Inference integration

MindIE Bench can discover and launch Apple's original RSI tasks through their
bundled Pier runtime, then import the original verifier receipts into a separate
task-balanced report. The original CUDA tasks remain GPU tasks. This backend
does not claim Ascend acceptance or G0–G3 knowledge/remote-tool ablation.

The reference source is
[`apple-aiml-research/ml-rsi-inference-bench@a6b10af`](https://github.com/apple-aiml-research/ml-rsi-inference-bench/tree/a6b10af282014412b4eae14f3105d68555889821).
See [the architecture, comparison and migration analysis (中文)](rsi-analysis.zh-CN.md).

## Install and inspect

Clone the source once alongside MindIE Bench. Do not copy its tests into the
agent's native MindIE workspace or knowledge corpus.

```sh
git clone https://github.com/apple-aiml-research/ml-rsi-inference-bench.git ../ml-rsi-inference-bench
git -C ../ml-rsi-inference-bench checkout a6b10af282014412b4eae14f3105d68555889821
uv run mindie-bench rsi-catalog --source ../ml-rsi-inference-bench \
  --output reports/rsi-catalog.json
```

The catalog records the source commit and dirty state, task TOML hashes, target
engine revision, model, GPU count/type, memory/storage/shared-memory requests,
offline setting, original budgets, verifier environment mode, workload and
scoring metadata. It supports both metadata layouts used by the 37 released
tasks. The task metadata's target revision can be a tag; it is not falsely
presented as a resolved commit SHA. The checkout commit binds the full upstream
tree; the catalog hash covers the extracted metadata, not every source byte.

Discovery/scoring require Python 3.11+ and Git, and work without GPU libraries.
Execution uses upstream Python 3.12+, `uv`, Docker with NVIDIA GPU support or a
configured Modal environment, the task's H100/B200 resources, model weights and
agent credentials. Builds can consume hundreds of GB or several TB; the catalog
reports the task's actual requirements. Installation/network access during image
preparation is separate from the task's offline agent phase.

## Plan and run

```sh
uv run mindie-bench rsi-run --source ../ml-rsi-inference-bench \
  --task optimize_qwen3_6_27b_1xh100_open \
  --agent codex --model PROVIDER/PINNED_MODEL \
  --attempts 1 --backend docker --output reports/rsi-qwen --dry-run
```

Remove `--dry-run` to execute. An optional `--env-file /absolute/path/to/.env`
is passed to Pier; its contents are never copied into the plan. `--uv` selects
the executable when it is not on PATH. Use a fresh output path for each job.
Dry-run prints JSON without creating a run directory or making model calls.

Execution saves `run-plan.json`, then replaces the launcher process with:

```text
uv run --locked --directory <source> pier run --path <source>/tasks/<task>
  --env docker --agent <agent> --model <model>
  --jobs-dir <output>/jobs --job-name rsi
  --n-attempts <attempts> --n-concurrent 1 --max-retries 0
```

Pier owns image construction, agent installation, network allowlists, timeouts,
candidate capture, separate-verifier environments where declared, and cleanup.
There is no second scheduler or enclosing timeout that cuts off its verifier.
The adapter requires a clean checkout and refuses existing output directories.
It does not automatically retry interrupted jobs. If launch fails, the saved
plan remains evidence; use a new output path after correcting the cause.

One task runs per invocation and attempts run serially. Allocate GPU resources
externally before launching several jobs. To configure custom providers,
reasoning or advanced agent kwargs beyond the exposed options, use the upstream
Pier job configuration interface and import the resulting jobs with `rsi-score`.
Native MindIE profile isolation, credential brokering and G3 capture are not
implicitly applied to this external backend.

## Import and score

```sh
uv run mindie-bench rsi-score --source ../ml-rsi-inference-bench \
  --input reports/rsi-qwen/jobs --task optimize_qwen3_6_27b_1xh100_open \
  --expected-attempts 1 --output reports/rsi-qwen-score.json
```

Omit `--task` when scoring the entire catalog. Repeat it to declare a subset.
Input containing other tasks is rejected instead of silently filtered. A subset
is labelled `selected_tasks`; it is never presented as a full benchmark result.
The input is a job tree or one trial's **`result.json`** (singular), with its
original adjacent `verifier/` files. Job-level result objects are not trials.

The report separates agent/version/model/execution-config cohorts, retains each original task
checksum, hashes the imported result/reward/metrics files, and records the
current catalog revision. This is local provenance, not a cryptographic
attestation of historical execution. Agent, environment, verifier and timeout
configuration fields are hashed into the cohort identity without copying their
contents or credentials into reports. This separates different recorded budgets,
reasoning settings and environment overrides. It cannot detect unrecorded mutable
deployments or hardware drift; retain those conditions in separate input trees.

| Evidence | Treatment |
| --- | --- |
| Support reward exactly 1 | A full pass |
| Support reward in [0, 1) | A failure; partial test fraction retained |
| Correct optimization with `speedup_geomean` | Use the measured ratio, including values below 1 |
| Failed correctness / acceptance with a zero reward and no scored ratio | Neutral 1×; record a completed failure |
| `infra_void` or upstream `reward=-1` | `infra_failed`, no score |
| Pier exception, missing finish/reward, nonstandard GPU flag | `unknown`, no score |
| Partial diagnostic ratio only in `metrics.json` | Never substitute it for the scored ratio |
| Duplicate trial ID, conflicting reward copies, mixed task checksums | Explicit error |

Pier gives `reward.txt` precedence over `reward.json`; this matters when the text
file carries a void sentinel. Optimization `reward=max(0, speedup-1)` loses
information, so it is **never** inverted to obtain a measured ratio.

For support, compute the fraction of full passes within each task, then the
arithmetic mean across tasks. For optimization, compute a geometric mean within
each task and across tasks. The repeated-attempt aggregation is explicitly
defined here; the published HTML exposes preaggregated results and is not an
executable specification of every upstream aggregation choice. No BT/Elo rating
or leaderboard rank is inferred from imported trial receipts.

Every task retains attempted, valid and expected counts plus all four terminal
statuses. Track `score` is withheld unless every selected task has exactly the
declared number of valid attempts, with no extra or invalid attempts. Partial
`observed_score` remains available with coverage. This conservative publication
gate preserves MindIE's missing-evidence semantics; it is stricter than merely
averaging the surviving results. Different per-task repetition plans currently
need separate scoring selections. Original failures and uncertainty are never
deleted to make the coverage complete.

## Boundary and attribution

This is an independently written adapter. No Apple task, model weight, reference
patch, verifier implementation or Pier source is copied into MindIE Bench.
Upstream remains under its own [LICENSE](https://github.com/apple-aiml-research/ml-rsi-inference-bench/blob/a6b10af282014412b4eae14f3105d68555889821/LICENSE)
and [ACKNOWLEDGEMENTS](https://github.com/apple-aiml-research/ml-rsi-inference-bench/blob/a6b10af282014412b4eae14f3105d68555889821/ACKNOWLEDGEMENTS);
the bundled Pier/Harbor notices and model/dataset terms remain in that checkout.
Integration does not imply Apple endorsement.

Ascend migration requires new task identities, pinned vLLM-Ascend/CANN/container
versions, NPU-native reference measurements, numerical/graph/memory/service
verification and source-fail/gold-pass evidence. Replacing `gpu_types` with
`npu_model`, or importing RSI scores as native MindIE receipts, cannot establish
any of those properties.
