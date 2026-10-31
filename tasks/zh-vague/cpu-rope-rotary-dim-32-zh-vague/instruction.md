换了一组模型头维度配置（`rotary_dim=32`）之后，在昇腾上跑 RoPE 缓存更新报错挂了，帮我解一下这个问题：

```text
RuntimeError: The expanded size of the tensor (128) must match the existing size (32) at non-singleton dimension 3.  Target sizes: [1, 3, 1, 128].  Tensor sizes: [3, 1, 32]
```
