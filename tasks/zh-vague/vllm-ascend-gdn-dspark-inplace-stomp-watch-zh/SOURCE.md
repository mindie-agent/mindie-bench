# Source Provenance: vllm-ascend-gdn-dspark-inplace-stomp-watch-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Base Commit**: `adbea376b521c8028fa8380a8d1c52d92a9b61c3`
- **Fix Commit**: `27bf736f01de587164de94b150b8922eb0b2afe2`
- **Target Production Files**:
  - `vllm_ascend/ops/gdn_attn_builder.py`
  - `vllm_ascend/spec_decode/dspark_proposer.py`
- **Root Cause**:
  1. In `vllm_ascend/ops/gdn_attn_builder.py`, `_build_actual_seq_lengths` aliases `query_start_loc[:num_sequences + 1]` when `out is None` and overwrites `query_start_loc[1:]` in-place; meanwhile `AscendGDNAttentionMetadataBuilder.build()` slices `block_table_tensor[:num_spec_decodes, :self.num_spec + 1]` without `torch.index_select` isolation and `_attach_spec_decode_metadata` calls `masked_fill_` in-place on `cache_indices`, stomping the caller's live `block_table_tensor`.
  2. In `vllm_ascend/spec_decode/dspark_proposer.py`, `AscendDSparkProposer.set_inputs_first_pass` aliases `self._per_group_block_tables[gid]` into `self._per_group_block_table_buffers[gid]` without cloning and fills trailing rows with `-1` in-place, and mutates `cad.seq_lens` in-place via `.sub_(num_rejected_tokens_gpu)` and `.add_(self.num_query_per_req)`.
