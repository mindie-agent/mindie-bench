我们在跑 GLM-5.2 开启多步 MTP（`MTP > 1`）并搭配 SFA 注意力后端时，发现投机解码接受率异常偏低。排查发现 SFA metadata builder 在为每个 draft step（`draft_index = 1, 2, ...`）构建 attention metadata 时，普通的 `_build` 路径每次都把 `actual_seq_lengths_query` 和 `actual_seq_lengths_key` 写进了同一对主图缓冲区（`self.actual_seq_lengths_query` / `self.actual_seq_lengths_key`），导致前序 draft step 的序列长度张量被后续 step 原地覆盖。

帮我修一下 SFA 注意力后端的 `_build` 方法（当传入 `draft_index is not None` 时，参照同文件中 SFA-CP 的写法选择 `self.spec_actual_seq_lengths_query[draft_index - 1]` 和 `self.spec_actual_seq_lengths_key[draft_index - 1]`，非 draft 调用保持原有缓冲区不变）：

```text
AssertionError: SFA attention metadata for draft_index=1 was overwritten by draft_index=2 because _build reused self.actual_seq_lengths_query instead of per-step speculative sequence length buffers
```

（注：当前为本地源码级修复验证环境，无需连接远端 NPU，直接在本地工作区完成代码修改并用 `/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python -m py_compile` 校验通过后即可直接收敛总结。）
（不要修改 `tests/` 目录，只修改 `vllm_ascend/` 源码后立即收敛总结。）
