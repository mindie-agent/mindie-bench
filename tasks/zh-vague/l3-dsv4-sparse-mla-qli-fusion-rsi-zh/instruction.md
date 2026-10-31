我在昇腾 910B 上调优 DeepSeek-V4（W8A8）的 SparseMLA 解码算子，前一阶段尝试记在 `logs/dsv4_stage1_l3_attempt.json` 里了，现在跑长序列还是报下面的错而且搬运和同步开销太大，帮我继续优化把这个问题解一下：

```text
RuntimeError: AICore invalid GM address: negative block id (-1) reached npu_sparse_flash_attention
```
