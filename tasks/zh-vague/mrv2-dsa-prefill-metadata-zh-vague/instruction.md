我们在开启 ModelRunner V2（MRV2）跑 DeepSeek DSA 注意力架构并打开 MTP 投机解码时，prefill 和 draft 阶段的 attention metadata 状态不对，KV cache shape 接口签名也报参数不匹配，甚至在投机步更新 `seq_lens` 时把 DSA 的序列长度重复加了 1，导致推理直接崩掉。

现场报错如下，帮我把 MRV2 下 DSA 的 metadata 链路修好（需要把 `input_batch.is_prefilling_np[:num_reqs]` 切片转为 bool 张量后通过 `build_attn_metadata` 和 `build_draft_attn_metadata_factory` 透传给 `CommonAttentionMetadata` 的 `is_prefilling` 字段，在 `_ascend_update_seq_lens` 中对 `DSA` 架构跳过通用的 `seq_lens += 1` 更新，并对齐 DSA backend `get_kv_cache_shape` 的 `cache_dtype_str: str | None = None` 参数签名）：

```text
TypeError: AscendDSABackend.get_kv_cache_shape() takes 5 positional arguments but 6 were given (missing cache_dtype_str: str | None = None)
AssertionError: CommonAttentionMetadata.is_prefilling is None in MRV2 prepare_attn / build_draft_attn_metadata_factory, and _ascend_update_seq_lens unexpectedly mutated seq_lens for DSA architecture
```

（注：当前为本地源码级修复验证环境，无需连接远端 NPU，直接在本地工作区完成代码修改并用 `/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python -m py_compile` 校验通过后即可直接收敛总结。）
