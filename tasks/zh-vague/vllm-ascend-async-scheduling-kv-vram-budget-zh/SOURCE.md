# Task Provenance: vllm-ascend-async-scheduling-kv-vram-budget-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Target Files**: `vllm_ascend/core/kv_cache_interface.py`, `vllm_ascend/worker/model_runner_v1.py`
- **Log Evidence**: `logs/a5_async_sched_oom.log`
- **Domain**: Ascend 950PR (A5) async scheduling double-buffered workspace, in-flight lookahead KV padding, delayed KV block release retention, and joint HBM + TPOT SLA concurrency solver.
