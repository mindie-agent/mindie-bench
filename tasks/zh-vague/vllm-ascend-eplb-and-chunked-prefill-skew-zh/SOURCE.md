# Task Provenance: vllm-ascend-eplb-and-chunked-prefill-skew-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Target Files**: `vllm_ascend/eplb/eplb_updator.py`, `vllm_ascend/ascend_config.py`
- **Log Evidence**: `logs/dsv3_dp_skew_sla.log`
- **Domain**: Multi-DP `AllToAllV` straggler barrier wait, EPLB routing imbalance ratio + D2D migration net benefit gate, and SLA-constrained chunked-prefill token budget.
