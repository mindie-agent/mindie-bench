# Task Provenance: vllm-ascend-mla-decode-profiler-sim-structure-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Target Files**: `vllm_ascend/ops/mla.py`, `vllm_ascend/worker/model_runner_v1.py`
- **Trace Evidence**: `traces/vllm_mla_decode_a5.db`
- **Domain**: Cross-component `mindie-sim` structure export + `mindie-profiler` CANN `msprof` SQLite analysis identifying unattributed `aclnnCast` and `aclnnInplaceCopy` kernels in MLA decode forward and per-step metadata preparation.
