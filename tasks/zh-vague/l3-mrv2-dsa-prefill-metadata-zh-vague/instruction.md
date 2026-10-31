我在昇腾上切到 Model Runner V2 跑 DeepSeek 稀疏注意力（DSA）模型并开启多步投机解码时，初始化 KV cache、构造注意力元数据和跑多步草稿更新的时候连着报错挂了，日志如下，帮我解一下这个问题：

```text
TypeError: AscendDSABackend.get_kv_cache_shape() got an unexpected keyword argument 'cache_dtype_str'
AssertionError: assert common_attn_metadata.is_prefilling is not None
RuntimeError: DSA persistent seq_lens tensor was modified in-place during speculative draft step
```
