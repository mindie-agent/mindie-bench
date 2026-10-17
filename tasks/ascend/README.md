# Ascend task families and operator runbook

These tasks are historical-source repair tasks. A candidate dossier, an
executable reproduction and a verified hardware task are distinct states.
Only an actual source-fail/gold-pass receipt bound to the exact task bytes,
tier and projected image permits `evidence = "verified"`.

| Defect family | G3 seed | Held-out evaluation | Source scope |
| --- | --- | --- | --- |
| RMSNorm optional quantization metadata | `rmsnorm-missing-description` | `rmsnorm-null-description` | Historical RMSNorm class; real NPU forward |
| Partial-RoPE explicit dimension | `rope-rotary-dim-64` | `rope-rotary-dim-32` | Historical cache functions; real NPU tensor allocation/update |
| Reduce-sampler candidate contract | `sampler-unfiltered-candidates` | `sampler-oversized-topk` | Historical PyTorch filter function; real NPU tensors, explicit TP=1 adapter |

Two trigger variants count as one defect family. The focused RoPE/sampler
coverage does not establish full-model serving, whole historical vLLM
compatibility, custom AscendC kernels or HCCL.

## Prepare and prove each hardware environment

Use separate A2/A3 experiment configurations, task revisions, projected images,
run directories and comparison inputs. The exact DCMI model label in each
task must match its measured probe output; changing a hardware label creates
a new task digest and requires its own proof. Do not relabel an A2 proof as A3.

The task's source repository must contain its exact base revision. The owned
source snapshot exposes the failing source, while private gold patches,
verifiers, Git history and personal profiles remain outside agent read grants.
Inspect every granted runtime root for newer fixed copies, helper tests,
bytecode, source archives and symlink targets. Merely stripping the task's
`.git` directory does not remove an answer in installed dependencies.

The reusable owned-image preparation recipe, exact core-image/blob hashes,
function projection, existing DCMI probe and bounded leakage scanner are in
[`runtime/README.md`](runtime/README.md). These require a pre-existing exact
local base image; a new build needs a new identity and fresh soundness.

With a configured interpreter/component/model deployment and an admitted idle
device, the existing CLI provides the validation and soundness entry points:

```sh
uv run mindie-bench validate --tasks /absolute/hardware-specific-task-root \
  --config /absolute/hardware-experiment.toml

uv run mindie-bench soundness \
  --task /absolute/hardware-specific-task-root/rope-rotary-dim-64 \
  --config /absolute/hardware-experiment.toml --tier l3 \
  --output /absolute/new-evidence-root/rope64-soundness --promote
```

Repeat soundness for every selected variant on that hardware class. Each
output directory is a new explicit attempt; never resume or overwrite an
unknown attempt. Preserve original failed/unknown receipts. A missing
completion acknowledgement is not permission to repeat a paid/remote write.

Promotion does not assert general effectiveness. Its receipt must retain the
actual source/gold outcomes, task and runtime hashes, real verifier cases,
zero model calls and preflight/postflight PID/memory observations.

## Run the controlled matrix

The baseline evaluates only each held-out task. For example:

```sh
uv run mindie-bench run \
  --tasks /absolute/hardware-specific-task-root/rope-rotary-dim-32 \
  --config /absolute/hardware-experiment.toml --tier l3 \
  --groups G0,G1,G2 --repetitions 3 --concurrency 3 \
  --output /absolute/new-evidence-root/rope-baselines
```

Run the other two held-out tasks in their own named baseline roots. A G3 run
uses a task root containing exactly the three selected seed/eval pairs:

```sh
uv run mindie-bench run --tasks /absolute/hardware-specific-task-root \
  --config /absolute/hardware-experiment.toml --tier l3 --groups G3 \
  --repetitions 3 --concurrency 3 \
  --output /absolute/new-evidence-root/g3
```

Within each cluster/repetition, only the seed's completed capture advances
the corpus. Failed/unknown prerequisites block dependent evals. Held-out
captures never feed subsequent evaluations. The foreground agent chooses
whether to query knowledge; consumer readback does not prove foreground use.

The full three-family plan has 27 baseline eval cells, nine G3 seeds and nine
G3 evals per hardware class: 45 planned cells, including 36 held-out evals.
A smaller pilot must disclose its scope. Fixed device pools are not randomized
device matching, and repeated triggers do not become independent defects.

## Audit and export

Keep a machine-readable planned-cell inventory alongside raw receipts. Scan
only the current named runner roots; do not pool unrelated historical reports.
Hash original bytes, include unknown/infra attempts, and record blocked cells
without inventing receipts. Do not give a partial raw-receipt join an ordinary
complete-matrix manifest.

Record business verification, Stop/capture, metering and cleanup independently.
For observed G3 use, check the exact seed corpus/package/block/ref/body hashes
through the held-out query and explain result. Then use existing `score`,
`compare`, `score-slices` and `export-hf` entry points with their required
completeness and privacy gates. A full directory comparison must still reject
missing cells or incompatible fingerprints.

No valid completed denominator means Pass@1/Pass^3 remain null. Include all
required seed resources, including seeds whose eval never started. Missing
provider usage is not zero cost; absent USD prices remain unmeasured. Public
export IDs use the manifest's SHA-256 scheme while original raw IDs stay private.
Local Git closure and local export do not establish public PR, merge, Release
or Hugging Face upload acceptance.
