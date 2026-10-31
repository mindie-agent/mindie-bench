我在昇腾 910B 上调优 4D 空间分组归一化规约算子，前一阶段尝试记在 `logs/operator_stage1_l3_attempt.json` 里了，现在精度还是超阈值而且性能卡住下不去，报错如下，帮我继续优化把这个问题解一下：

```text
AssertionError: Numerical check failed on shape (2, 16, 16, 16): max_abs_err=0.046875 > 5e-3
```
