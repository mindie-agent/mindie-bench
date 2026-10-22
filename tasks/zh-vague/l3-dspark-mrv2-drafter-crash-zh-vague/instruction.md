在昇腾上用 ModelRunner V2 跑 DeepSeek-V4 的 DSpark / EAGLE 投机解码时服务启动后直接崩了，出现了两个连环报错：一是 DeepSeek-V4 模型初始化时没有在 PP 最后一级（`is_last_rank`）且开启 `use_eagle()` 或 `uses_draft_model()` 时初始化 `_mtp_hidden_buffer`，前向报 `AttributeError`；二是 DSA attention 的 `build_for_drafting` 在计算 `block_table` 步长时错误地除以了 `self.block_size` 而不是 `self.storage_block_size`。

报错日志如下，帮我把这两处修一下：

```text
AttributeError: 'DeepseekV4ForCausalLM' object has no attribute '_mtp_hidden_buffer' (when speculative_config.use_eagle() or uses_draft_model() is enabled on PP last rank)
RuntimeError: DSA build_for_drafting block_table stride mismatch: divided by self.block_size instead of self.storage_block_size
```

（注：请在远端 NPU 工作区中直接修改 `vllm_ascend/` 源码并在分配的 NPU 上验证，不要修改本地工作区或 `tests/` 目录，修复验证通过后立即收敛总结。）
