# Task Provenance: vllm-ascend-profiler-straggler-and-hbm-reconcile-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Target Files**: `vllm_ascend/profiling_config.py`, `vllm_ascend/worker/worker.py`
- **Trace Evidence**: `traces/tp2_decode_msprof/rank_0/ascend_pytorch_profiler_0.db`, `traces/tp2_decode_msprof/rank_1/ascend_pytorch_profiler_1.db`, `traces/tp2_decode_msprof/hbm_snapshot.json`
- **Domain**: Multi-stream interval-union exposed communication vs barrier wait straggler attribution and non-additive `P0–P4` HBM memory reconciliation.
