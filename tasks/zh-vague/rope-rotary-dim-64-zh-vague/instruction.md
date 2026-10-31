我在昇腾上跑一个自定义头维度的模型（配了 `rotary_dim=64`），更新 RoPE 缓存的时候报错挂了，帮我解一下这个问题：

```text
RuntimeError: The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 3.  Target sizes: [1, 3, 1, 128].  Tensor sizes: [3, 1, 64]
```
