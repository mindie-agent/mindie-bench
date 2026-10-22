我们在开启 DCP4 + MTP5（`num_speculative_tokens = 5`）跑投机解码服务时，一个 batch 展开后的 draft slot 数量超过了 `slot_mapping` 缓冲区长度，在 CPU 赋值和 Eagle3 全图 warmup 的 `aclnnInplaceCopy` 阶段直接越界崩了。排查发现 `BlockTable` 和 `AscendSpecDecodeBaseProposer.slot_mapping_group` 还在用旧的 `max_num_batched_tokens + 2 * max_num_reqs` 固定冗余，在 `num_speculative_tokens > 3` 时容量不够。

报错如下，帮我修一下（需要将 `BlockTable` 的 `slot_mapping_cpu` / `slot_mapping` 以及 `AscendSpecDecodeBaseProposer` 的 `slot_mapping_group` 容量统一改为 `max_num_batched_tokens + max(num_speculative_tokens - 1, 0) * max_num_reqs`，同时保留 proposer 里 `seq_lens_group` 和 `query_start_loc_group` 原有的 `2 * max_num_reqs` 大小不变）：

```text
ValueError: could not broadcast input array from shape (110,) into shape (104,) in BlockTable.compute_slot_mapping
RuntimeError: aclnnInplaceCopy failed during Eagle3 full-graph warmup: BlockTable slot_mapping (8208) vs AscendSpecDecodeBaseProposer.slot_mapping_group (8200)
```
