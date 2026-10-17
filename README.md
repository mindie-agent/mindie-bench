# MindIE Bench

An independent evaluation component for MindIE Agent completion rate and total
engineering cost. It holds the native harness, model, task and budget constant
while comparing remote tools, public knowledge and cross-task experience.

The runner uses fresh native Codex or Claude Code profiles, OS sandboxes, plain
source snapshots, external verifiers and finite G0-G3 experiment matrices.
Knowledge lives outside foreground file grants and is accessed through the real
MCP service. G3 connects the actual Codex or Claude Code Stop and transcript
to deterministic redaction, model-generated navigation, a corpus quality gate,
local Git publication and an independent consumer.

## Run

```sh
uv sync --locked --all-extras
uv run mindie-bench --help
uv run mindie-bench demo --output reports/new-demo
uv run pytest tests/test_performance_gates.py -q
```

The demo is synthetic and makes no model, SSH or NPU calls. Native runs require
an explicitly selected endpoint, credential environment variable, component
revisions and task configuration. Personal client logins are not copied.
Existing attempts are never overwritten or automatically resumed.
During development, test only the changed modules. Run the complete test suite
and packaging check once when the stage is ready for delivery.

## Datasets

| Dataset | Execution and scope |
| --- | --- |
| [Corpus-first L2 recipes](tasks/corpus-first-l2/README.md) | Four real CPU repair contracts and two distinct seed triggers, with complete historical modules, separate public reproduction and hidden judges. |
| [Ascend diagnostic replay](tasks/replay/README.md) | Saved real NPU errors, the complete historical sampler module and an independent CPU tensor judge; replay feedback alone cannot earn a pass. |
| [Ascend infrastructure L2](tasks/infra-l2/README.md) | Complete historical quantization loader modules, real CPU tensors and physically filtered cross-model knowledge. |
| [Versioned stale-knowledge task](tasks/stale/README.md) | Actual old advice, executed old/current configuration APIs and a separately judged reporting migration. |
| [Dual-card idle-DP task](tasks/corpus-first-l3/k3-gdn-idle-dp-state/SOURCE.md) | Complete historical model runner, actual Kimi K3 model modules, DP2/HCCL and independent state checks on physical NPUs. Scope is eager state correctness. |
| [Native KDA numerical task](tasks/corpus-first-l3/kda-chunk-varlen-padded-workspace/SOURCE.md) | Complete historical C++ operator source, fresh CANN builds, four failing variable-length cases and four dense controls, judged against independent FP32 references on A3. |
| [Fused RMSNorm performance](tasks/performance/README.md) | Real NPU synchronized measurements, hidden shapes, rotating buffers, ACLGraph, workspace and warm eight-step service gates. |
| [Frozen retrieval questions](benchmarks/retrieval/README.md) | Twelve known-positive questions over twelve real packages in the fixed 661-package corpus; component retrieval quality and latency. |
| [Ascend smoke suite](tasks/ascend/README.md) | Three historical defect families and six seed/eval variants with independent A2/A3 source-fail/gold-pass proofs. |
| [PyTorch CPU recipes](tasks/cpu/README.md) | Four separate CPU source-contract tasks with explicit runtime and soundness requirements. |
| [Routing slices](tasks/L1-L2-acceptance.md) | Separately labelled routing, retrieval and native protocol observations. |
| [Mining candidates](tasks/candidates/README.md) | Source-grounded historical dossiers awaiting reproducible task proof. |

`mine --recipe` compiles hashed source, knowledge evidence, reproduction,
verifier, gold and replay assets into an executable L2 candidate. Actual
fail-before/pass-after soundness controls promotion. CPU execution, deterministic
replay and physical NPU execution retain distinct identities.

## Measurements

Report independently judged completion, delivery time, all-attempt cost, source
access and edit progression, recognized NPU attempts, tool errors, polling,
retrieval calls and capture stages. Tool time uses interval unions. Missing
observations retain reasons and sample counts. Natural, explicit and oracle
knowledge conditions are separately fingerprinted.

Every report binds source/runtime, task, corpus, model, component, hardware and
planned-cell identities. `completed_pass`, `completed_fail`, `infra_failed` and
`unknown` remain distinct. Original failed and uncertain trials remain intact;
parser improvements do not relabel old hardware acceptance.

The verifier fingerprints its protected files before foreground admission and
checks them around execution. Performance acceptance requires numerical
correctness and actual graph, memory and warm-service evidence; isolated speedup
alone cannot earn a pass. Native L3 runs use coordinator leases, live npu-top
occupancy checks, diagnostics and per-slot compiler caches and ports.

See [usage and contracts](docs/usage.md) for maintainer commands,
[actual acceptance](docs/acceptance.md) for measured results and raw evidence,
and [architecture](docs/design.md) for evaluation boundaries. L3 uses trusted
evaluation code in dedicated Ascend containers with cooperative card leases,
with an empty client scratchpad so G0 and G1–G3 operate under symmetric remote
workspace boundaries.

`export-hf` applies deterministic `mindie_knowledge` redaction and Gitleaks
gates before writing dual-split Hugging Face Dataset artifacts (`trials.parquet`
and `tasks.parquet`, JSON summaries, Dataset Card `README.md`, and `index.html`).
By default it exports locally without network calls; pass `--upload-repo <org/dataset>`
when an authenticated push to Hugging Face Hub is explicitly desired. Public corpus
metadata is attributed under CC-BY-4.0; implementation code is MIT.

