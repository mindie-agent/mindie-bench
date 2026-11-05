# Task Provenance: `vllm-ascend-aclgraph-capture-and-replay-zh`

- **Repository**: `vllm-project/vllm-ascend`
- **Branch**: `bench/v2-d4-aclgraph`
- **Base Commit**: `f9705652efb0ed4d9fad25aff1c1372c3edbd38d`
- **Fix Commit**: `9838a607eaf745eb2e61d4c290574352f94378e3`
- **Target Files**:
  - `vllm_ascend/compilation/acl_graph.py`
  - `vllm_ascend/worker/model_runner_v1.py`

## Root Cause Summary
1. **Capture Stream Synchronization Hazard (`vllm_ascend/compilation/acl_graph.py`)**:
   Inside `with torch.npu.graph(aclgraph, pool=self.graph_pool):`, `ACLGraphWrapper.__call__` executes `_ = int(arg.reshape(-1)[0].item())` on the first tensor argument, forcing a host-device synchronization during stream capture.
2. **Static Input Pointer Rebinding (`vllm_ascend/worker/model_runner_v1.py`)**:
   In `NPUModelRunner._model_forward`, `positions = positions[:num_tokens_padded].clone()` allocates a fresh tensor during replay instead of passing the pre-allocated static `positions[:num_tokens_padded]` slice bound during graph capture. Similarly, in `_get_block_table_and_slot_mapping`, `slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded].clone()` rebinds `slot_mapping` to a new `data_ptr()` when `num_tokens < num_tokens_padded`.
3. **Unmasked Padded `slot_mapping` Tail (`vllm_ascend/worker/model_runner_v1.py`)**:
   In `_get_block_table_and_slot_mapping`, `slot_mapping[num_tokens:num_tokens_padded].fill_(-1)` is missing, leaving stale non-negative slot IDs in the padded tail `slot_mapping[num_tokens:num_tokens_padded]` during variable-batch `ACLGraph` replay and overwriting live KV cache entries.
