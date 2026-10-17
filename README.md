# MindIE Bench

An independent evaluation component for MindIE Agent completion rate and total
engineering cost. It holds the native harness, model, task and budget constant
while comparing remote tools, public knowledge and cross-task experience.

The runner uses fresh native Codex or Claude Code profiles, OS sandboxes, plain
source snapshots, external verifiers and finite G0-G3 experiment matrices.
Knowledge lives outside foreground file grants and is accessed through the real
MCP service. G3 connects an actual native Codex Stop to public transcript
capture, deterministic redaction, model-generated navigation, local Git
publication and independent consumer readback.

## Run

```sh
uv sync --locked --all-extras
uv run mindie-bench --help
uv run mindie-bench demo --output reports/new-demo
uv run pytest
uv run ruff check .
```

The demo is synthetic and makes no model, SSH or NPU calls. Native runs require
an explicitly selected endpoint, credential environment variable, component
revisions and task configuration. Personal client logins are not copied.
Existing attempts are never overwritten or automatically resumed.

## Datasets

| Dataset | Execution and scope |
| --- | --- |
| [Corpus-first L2 recipes](tasks/corpus-first-l2/README.md) | Four real CPU repair contracts and two distinct seed triggers, with complete historical modules, separate public reproduction and hidden judges. |
| [Ascend diagnostic replay](tasks/replay/README.md) | Saved real NPU errors, the complete historical sampler module and an independent CPU tensor judge; replay feedback alone cannot earn a pass. |
| [Versioned stale-knowledge task](tasks/stale/README.md) | Actual old advice, executed old/current configuration APIs and a separately judged reporting migration. |
| [Dual-card idle-DP task](tasks/corpus-first-l3/k3-gdn-idle-dp-state/SOURCE.md) | Complete historical model runner, actual Kimi K3 model modules, DP2/HCCL and independent state checks on physical NPUs. Scope is eager state correctness. |
| [Native KDA numerical task](tasks/corpus-first-l3/kda-chunk-varlen-padded-workspace/SOURCE.md) | Complete historical C++ operator source, fresh CANN builds, four failing variable-length cases and four dense controls, judged against independent FP32 references on A3. |
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

See [usage and contracts](docs/usage.md) for maintainer commands,
[actual acceptance](docs/acceptance.md) for measured results and raw evidence,
and [architecture](docs/design.md) for evaluation boundaries. L3 uses trusted
evaluation code in dedicated Ascend containers with cooperative card leases.

`export-hf` applies the privacy gates before writing local JSON, Parquet and HTML;
it does not upload or create public issues, PRs or Releases. Public corpus
metadata is attributed under CC-BY-4.0; implementation code is MIT.
