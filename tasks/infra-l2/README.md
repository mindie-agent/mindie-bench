# Ascend infrastructure on CPU

`quarot-draft` executes the complete historical Ascend registration and EAGLE3
loader modules against real CPU tensors. It reproduces upstream fix `b9bd3ee01`:
one checkpoint architecture selects the upstream loader and loses the required
weight-basis transformation. Hidden cases cover both names, rotated and ordinary
weights, and three tensor dimensions. Registry resolution and the upstream weight
sink are explicit fixtures; this task makes no NPU or full-model accuracy claim.

The existing recipe compiler binds real source blobs and read corpus blocks.
The profile physically excludes the EAGLE3 registration family and retains the
K3 DSpark weight-basis block as cross-model advice. The external judge uses its
own boundary helper and a separately executed torch comparator negative control.

```sh
uv run python tasks/infra-l2/materialize.py \
  --corpus /owned/knowledge-vllm-ascend \
  --source-repo /owned/vllm-ascend \
  --python /owned/existing-torch-runtime/bin/python \
  --output /owned/quarot-task
uv run mindie-bench soundness --task /owned/quarot-task/compiled \
  --config /owned/cpu-config.toml --tier l2 \
  --output /owned/quarot-task/proof --promote
uv run mindie-bench run --tasks /owned/quarot-task/compiled \
  --config /owned/cpu-config.toml --tier l2 --groups G0,G1,G2 \
  --canary-first --output /owned/quarot-task/results
```

Use an existing CPU torch 2.10 runtime. The materializer installs nothing and
the ordinary source/gold verifier must establish fail-before/pass-after before
any model cohort. This is one loader regression, separate from the six CPU
knowledge-component tasks and six Ascend smoke tasks.
