我在昇腾 A5（Ascend 950DT）上跑 `vllm-ascend` 的 4D 空间分组归一化算子（单算子复现命令 `python3 benchmarks/reproduce_spatial_group_norm.py`，报错日志见 `logs/spatial_group_norm_a5.log`）时，算子耗时偏高而且数值精度校验不过，报错如下，帮我把这个问题解一下：

```text
AssertionError: [Ascend950DT] Numerical check failed on shape (2, 16, 16, 16): max_abs_err=0.046875 > 5e-3
```
