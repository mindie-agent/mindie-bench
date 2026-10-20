# Native inference lifecycle examples

`materialize.py` creates three schema 1.0 MindIE tasks, each with an optional
`inference` contract, pinned source repository, private baseline, external judge,
gold patch and a synthetic fixture-agent configuration. `judge.py` uses the
reusable `mindie_bench.evaluation.inference` measurement and acceptance helpers.

The CPU fixture computes the sum of squared shared-prefix values for a batch of
integer requests. The support task adds a new request mode. Optimization and
regression tasks remove redundant prefix computation while preserving exact
outputs and a small-request latency guard. Baseline and candidate execute in
separate Python child processes; measurements include their process startup.
Warmup and counterbalanced rounds are real executions, not preset speedup values.

These are **synthetic protocol examples**, not production LLM, GPU/NPU acceptance
or adversarial candidate-isolation proofs. Timing may fluctuate. The fixture
agent knows the intended edit and makes no model calls. Source-fail/gold-pass
soundness is deterministic on the support task; production performance tasks
need stable private workloads and hardware-specific acceptance calibration.

See [native usage, contracts and limits](../../docs/rsi-inference.md) for commands.
Use a fresh output directory. The materializer never clones an external repo,
installs a model, starts a remote job or modifies an existing task collection.
