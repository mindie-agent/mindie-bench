开启 SFA 注意力后端跑多步 MTP 投机解码时，我们发现 `draft_index=1` 和 `draft_index=2` 生成出来的 `AscendSFAMetadata` 共享并互相覆盖了同一个 `actual_seq_lengths_query` / `actual_seq_lengths_key` 缓冲区，导致每个投机步的序列长度全部串掉，报错如下，帮我修一下：

```text
AssertionError: AscendSFAMetadataBuilder._build overwrote draft step 1 actual_seq_lengths_query and actual_seq_lengths_key when building draft step 2 (expected per-draft-step speculative buffer selection for draft_index >= 1)
```

（注：当前为本地源码级修复验证环境，无需连接远端 NPU，直接在本地工作区完成代码修改并用 `/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python -m py_compile` 校验通过后即可直接收敛总结。）
（不要修改 `tests/` 目录，只修改 `vllm_ascend/` 源码后立即收敛总结。）
