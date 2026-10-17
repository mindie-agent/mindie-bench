# Real CPU source-repair twins

These are four candidate templates for two upstream defect families. They
exercise actual historical Python functions on real CPU torch tensors. The
runtime-bound task revision, its independent L2 soundness receipt and its
frozen configuration form the executable evidence.

| Family | Seed template | Held-out template |
| --- | --- | --- |
| Partial-RoPE cache dimension | `cpu-rope-rotary-dim-64` | `cpu-rope-rotary-dim-32` |
| Reduce-sampler candidate contract | `cpu-sampler-unfiltered-candidates` | `cpu-sampler-oversized-topk` |

Two variants still count as one family. These CPU twins share their source
defects with the related L3 tasks; they are not four additional independent
defect families. They do not prove NPU numerics, CANN/custom kernels, HCCL,
full-framework imports, model serving or broad MindIE effectiveness.

## Create the isolated runtime

The current recipe locks CPython 3.13.12, torch 2.10.0, NumPy 2.3.5 and their
transitive packages for macOS arm64. `runtime/uv.lock` includes the artifact
hashes. It deliberately does not install into the benchmark/component/native
host environment.

Use an explicitly installed Python 3.13.12 and a new owned destination:

```sh
python3 tasks/cpu/setup_runtime.py \
  --python /absolute/cpython-3.13.12/bin/python3.13 \
  --destination /absolute/new-owned-cpu-runtime
```

This runs `uv sync --locked`, records stdout/stderr, probes a genuine CPU
tensor and saves the interpreter/package/lock identity in
`runtime-manifest.json`. A destination that already exists is rejected; failed
setup evidence is retained. No global Python, personal profile or existing
benchmark environment is modified.

This lock is specific to macOS arm64. Other platforms need an explicitly
reviewed CPU-wheel lock and a new runtime/proof; do not silently select a
different wheel backend or relabel this proof.

## Materialize and prove a task revision

`local_verifier` has no separate Python-environment injection. The templates
therefore fail visibly until a fresh private revision binds its public
reproduction command and private judge launcher to the exact CPU interpreter:

```sh
python3 tasks/cpu/materialize.py \
  --runtime-manifest /absolute/new-owned-cpu-runtime/runtime-manifest.json \
  --output /absolute/new-private-cpu-task-revision
```

The script resolves each existing owned source repository, copies the four
templates, records the runtime binding and inherits no soundness receipt.
Gold patches, judge code and full Git history must remain outside the native
agent's read grants. Only `environment/` is copied into its source workspace.

Create a separate native experiment TOML with the chosen benchmark/harness
source, exact model metadata and explicit provider configuration. For G0,
omit remote/MCP/skills/corpora and grant only the additional CPU venv directory:

```toml
schema_version = "1.0"
backend = "native"
runtime_read_paths = ["/absolute/new-owned-cpu-runtime/venv"]

[harness]
name = "codex"
executable = "/absolute/native/codex"
expected_version = "codex-cli VERSION"
model = "EXPLICIT_MODEL"
model_catalog_path = "/absolute/full-native-model-catalog.json"
reasoning_effort = "medium"
provider_url = "http://127.0.0.1:PORT/v1"
api_key_env = "EXPLICIT_PROVIDER_KEY_ENV"
```

The runner already grants its own interpreter/package and native binary.
Do not grant the whole task revision, source repository, private report root
or personal home merely to make an import succeed.

```sh
mindie-bench validate --tasks /absolute/new-private-cpu-task-revision \
  --config /absolute/cpu-experiment.toml

mindie-bench soundness --tier l2 \
  --task /absolute/new-private-cpu-task-revision/cpu-rope-rotary-dim-64 \
  --config /absolute/cpu-experiment.toml \
  --output /absolute/new-proof/rope64 --promote
```

Repeat soundness for each task using new proof destinations. Each L2 task has
`npu_count = 0` and `replay_commands = []`. Its final judge executes CPU tensor
operations from the actual source file; no string-pattern replay is involved.
Each verifier checks 18 cases across float32, float16 and bfloat16. Runtime
imports/version failures remain infrastructure failures rather than reward0.

After promotion, a finite native smoke uses the existing runner:

```sh
mindie-bench run --tier l2 --groups G0 --repetitions 1 --concurrency 1 \
  --tasks /absolute/new-private-cpu-task-revision/cpu-rope-rotary-dim-32 \
  --config /absolute/cpu-experiment.toml --output /absolute/new-native-smoke
```

One smoke establishes that task's native repair/judge path only. A full L2
controlled cohort, knowledge/capture groups and clear-negative routing samples
require their own budget, configuration and observations.

## Provenance and relocation

Preserve template/revision task hashes, actual source/gold commits, original
receipt bytes, source implementation digest, CPU runtime manifest and lock
hash, final configuration, public reproduction and verifier results. Canonical
templates remain candidate templates; only the separately proven revision may
be marked verified.

Moving the runtime, changing its interpreter/dependencies, changing a task
byte or changing the source execution contract creates a new revision. Re-run
L2 soundness and refresh task/config/runtime fingerprints instead of copying
an old proof. A path-bound receipt is not portable acceptance.

Private CPU development diagnostics and L3 hardware proofs are supporting
context, not substitutes for this independent L2 proof. Unknown paid attempts
remain unknown and are never automatically resumed or repeated.
