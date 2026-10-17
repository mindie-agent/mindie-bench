# NPU performance and transfer

`fused-rmsnorm-gate` uses the complete historical vLLM-Ascend source at
`f02da291ec0bdbe31af248deffa69f7a97e8f259`. Its public reproducer exercises
CustomOp dispatch on a real NPU. The private judge checks numerical correctness,
two hidden shapes, four independent buffers per shape, synchronized host
monotonic timing, actual ACLGraph capture/replay and allocator peaks.

The end-to-end workload is a persistent HTTP service running eight gated
RMSNorm steps. It requires at least 1.05x isolated speedup and 1.01x warm-service
speedup. The baseline scratch reserve is 16 MiB, extra candidate allowance
16 MiB and total service scratch budget 32 MiB. This service has no KV cache;
its result does not establish full-model throughput or KV-cache capacity.

The unmodified historical fix `5312b32d` improved isolated execution but failed
the warm-service gate. The reference patch adds a measured native RMSNorm host
path improvement to that historical repair. It is explicitly a benchmark
reference, not an unmodified upstream commit. The private numerical reference
uses CPU arithmetic, and comparator negative controls run both independently
and in the actual judge process after candidate code executes.

Use an existing NPU runtime with its compiled extensions, while granting only
the necessary shared libraries and metadata from any already repaired backend.
Granting the whole installed backend Python tree would expose the solution.
The ordinary commands apply:

```sh
uv run mindie-bench validate --tasks tasks/performance/fused-rmsnorm-gate \
  --config /owned/npu-experiment.toml
uv run mindie-bench run --tasks tasks/performance/fused-rmsnorm-gate \
  --config /owned/npu-experiment.toml --tier l3 --groups G0,G2 \
  --canary-first --output /owned/performance-results
```

For a new runtime, establish source failure and reference success using the
existing `mindie-bench soundness` command before a model cohort. Measurement
details belong to the verifier receipt; current real results are summarized
in `docs/acceptance.md`.
