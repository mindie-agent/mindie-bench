# Native inference lifecycle evaluation

MindIE Bench incorporates ideas from RSI-Inference into its own task, judge,
receipt and scoring pipeline. It needs no Apple checkout, Pier installation,
Harbor receipt import or additional runtime dependency. Use the existing
`validate`, `soundness`, `run`, `score`, `audit-report`, `compare` and `export-hf`
commands. The previously proposed external `rsi-*` adapter is removed from PR #6.

The [Chinese source analysis](rsi-analysis.zh-CN.md) explains the original systems,
the gaps, the native implementation and the deliberate differences from RSI.

## A backwards-compatible task contract

Existing schema 1.0 tasks remain valid without changes. New tasks opt in with
`[inference]`. Existing `PerformanceContract` requirements are still enforced;
an inference contract cannot bypass ACLGraph, workspace or warm-server gates.

```toml
[inference]
category = "regression" # model_support | optimization | regression
engine = "vllm-ascend"
model = "YOUR_PINNED_MODEL"
baseline_commit = "0123456789abcdef0123456789abcdef01234567"
baseline_source_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
runtime_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
device_class = "Ascend-910B"
required_checks = ["numerics", "serving"]
warmup_rounds = 1
measurement_rounds = 4
min_speedup_ratio = 1.03

[[inference.workloads]]
name = "decode"
metric = "tokens_per_sec"
direction = "higher"
role = "scored"
workload_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

[[inference.workloads]]
name = "prefill_guard"
metric = "ttft_ms"
direction = "lower"
role = "guard"
min_ratio = 0.95
workload_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
```

The hashes above are **placeholders**, not verified runtime identities. The
baseline commit must equal the existing `environment.<tier>.base_commit`.
The judge must derive and check runtime, model/configuration and workload hashes
from its actual frozen inputs. Hashing a manifest is not evidence that a running
engine uses those settings. The existing task fingerprint binds source, budget,
harness, model, device/remote context and the entire inference contract.

`model_support` declares required checks and no performance workloads. Full
support requires every check; a partial fraction is diagnostic only. Optimization
and regression require at least one scored workload. Guards require an explicit
minimum ratio and never contribute positively to the headline geometric mean.

## Implement a trusted judge

The existing `verifier/test.sh` runs in the local or remote private judge. It emits
the existing binary `reward` plus `inference_evidence`. That evidence contains:

- The runner-provided `MINDIE_BENCH_PERFORMANCE_NONCE` and
  `MINDIE_BENCH_ORACLE_SHA256`, binding observations to this run and frozen oracle.
- Baseline commit/source hash, runtime hash, actual device class, candidate source hash and
  an explicit baseline health outcome.
- An actual boolean outcome for **every** required correctness check.
- For each workload: its input hash, warmup count and all paired measurement
  rounds, including order, baseline/candidate values and explicit candidate errors.

Use `evaluation.inference.measure_workloads(contract, measure)` inside that judge.
The callback receives an arm (`baseline` or `candidate`) and a declared workload.
It must perform the actual measurement, including server readiness and device
synchronization. Both arms use identical requests/settings. The helper warms
both arms, then alternates AB/BA over an even number of rounds. Warmup results are
excluded. `measure_latency_ms` measures completed operations with a monotonic,
high-resolution host counter and caller-supplied synchronization.

A callback may raise `CandidateMeasurementFailure` only for an established
candidate business failure. Baseline failures and unexpected exceptions propagate
to native infrastructure handling. The helper does not retry trials or create a
second scheduler. The existing verifier timeout encloses all work and cleanup.

Call `evaluate_inference(contract, evidence)` to derive acceptance and diagnostics.
The runner independently repeats that evaluation in `parse_verifier`, checks the
run/oracle identity and rejects successful rewards that contradict the gates.
Arbitrary JSON from agent code is not trusted measurement: the judge and frozen
baseline must remain outside candidate write/import authority, as for existing
MindIE performance probes. The CPU example is a synthetic protocol example, not
an adversarial isolation proof for a production engine.

## Scoring inside native receipts

For each paired round, the ratio is candidate/baseline for a higher-is-better
metric and baseline/candidate for a lower-is-better metric. Use the **median of
paired ratios** per workload, then the geometric mean across scored workloads.
Every guard must be measured; a guard passes when its paired median reaches its
declared floor. This is deliberately stricter than RSI's regression implementation
that can admit an unmeasured guard or use the best guard round.

| Observation | Binary acceptance | Optimization track value |
| --- | --- | --- |
| Correct, guards pass, gain reaches threshold | 1 | Measured ratio |
| Correct slowdown or gain below threshold | 0 | Measured ratio, including below 1 |
| Correctness or measured guard failure | 0 | Neutral 1×; diagnostic ratios retained |
| Unchanged candidate source | 0 | Neutral 1×; timing noise cannot establish improvement |
| Established candidate measurement failure | 0 | Neutral 1× |
| Existing performance gate or native process budget rejects candidate | 0 | Neutral 1×, even below inference threshold |
| Other native gate rejects an otherwise accepted candidate | 0 | Neutral 1× |
| Missing workload/round, unhealthy baseline, wrong runtime/nonce | No accepted verifier result | No score |
| Native `infra_failed` or `unknown`, even with prior successful observations | Unchanged native status | No score |

There is no clipped continuous `reward` to invert. The native binary outcome and
continuous measured ratio remain separate. Source-fail/gold-pass soundness,
G0–G3 isolation/capture, cleanup, cost accounting and existing completion metrics
retain their meaning.

Native `summary.json`, `score`, audit, comparison JSON and HF-compatible export
add an `inference` section. Scoring recomputes observations from typed evidence.
When both contracts are declared, the receipt also retains the native performance
contract and the runner's process stop reason. Scoring revalidates the existing
performance evidence; a failed workspace/graph/server gate or exhausted process
budget cannot contribute a gain just because inference was below its threshold.
Cohorts preserve group, role, tier, evidence class, experiment and harness/model
controls. Repeated attempts aggregate within each task before equal task weighting:
arithmetic for full-support pass rates, geometric for optimization/regression.
Different task revisions or duplicate native matrix cells are errors.

`score` and `audit-report` read the original `eval_manifest.json` when present.
Complete scores require exactly its planned valid lifecycle cells. Missing,
blocked, unknown and extra cells withhold the score while retaining observed
values and counts. Receipt-only/frozen inputs without that plan report observed
values without claiming full coverage. Nothing reconstructs missing G3 seeds.

## Run the executable examples

The repository includes three synthetic CPU tasks: support a new request mode,
optimize shared-prefix computation, and repair repeated-prefix work. They execute
real subprocesses and measure actual CPU elapsed time, with throughput, latency
and a small-request guard. They do not claim vLLM/SGLang or NPU benchmark results.

```sh
uv sync --locked --all-extras
uv run python tasks/inference/materialize.py --output /absolute/path/inference-example
uv run mindie-bench validate --tasks /absolute/path/inference-example/tasks
uv run mindie-bench soundness --task /absolute/path/inference-example/tasks/model_support \
  --config /absolute/path/inference-example/experiment.toml --tier l2 \
  --output /absolute/path/inference-example/proof
uv run mindie-bench run --tasks /absolute/path/inference-example/tasks \
  --config /absolute/path/inference-example/experiment.toml --tier l2 --groups G0 \
  --repetitions 1 --output /absolute/path/inference-example/run
uv run mindie-bench score --input /absolute/path/inference-example/run
```

Full native execution requires the existing Linux/macOS sandbox. Materialization,
pure scoring and direct synthetic judge contract checks also work on Windows.
The example runtime is pinned to the materializing Python version. Re-materialize
after changing interpreters. Its fixture agent performs a known synthetic edit;
it is not a paid-model evaluation. Baseline-versus-itself timing may fluctuate,
so the deterministic soundness example uses the model-support task.

For real Ascend tasks, supply new native task IDs, actual model/engine/runtime
identities, private numerical and workload probes, and source-fail/gold-pass
evidence. Combine the lifecycle contract with existing device performance gates
where needed. No original Apple GPU task has been relabelled as NPU acceptance.

## Source attribution

Design reference:
[`apple-aiml-research/ml-rsi-inference-bench@a6b10af`](https://github.com/apple-aiml-research/ml-rsi-inference-bench/tree/a6b10af282014412b4eae14f3105d68555889821),
especially its throughput `scoring.py`, regression `regression_scoring.py`,
multi-workload task manifests and frozen-baseline/verifier structure.
This implementation independently expresses those methods in MindIE's native
contracts. No Apple task, verifier, Pier source, model weights or datasets are
copied or required at runtime. Their original licenses remain applicable to
their own repositories; this work does not imply Apple endorsement.
