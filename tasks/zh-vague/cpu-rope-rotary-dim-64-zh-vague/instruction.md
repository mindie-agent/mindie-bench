我们在昇腾 NPU 上跑一个自定义头维度的模型（`head_size=128`，`hf_text_config` 只配了显式的 `rotary_dim=64`，没有配 `partial_rotary_factor`），在更新 RoPE 旋转位置编码缓存（`set_cos_and_sin` / `update_cos_sin`）时直接报张量维度不匹配崩了。

报错堆栈如下，帮我定位并修一下源码（要求保持现有 `partial_rotary_factor` 最高优先级与完整 `head_size` 默认行为不变，当未配置 `partial_rotary_factor` 但 `hf_text_config` 存在有效 `rotary_dim` 时使用 `rotary_dim`）：

```text
RuntimeError: The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 3.  Target sizes: [1, 3, 1, 128].  Tensor sizes: [3, 1, 64]
```
