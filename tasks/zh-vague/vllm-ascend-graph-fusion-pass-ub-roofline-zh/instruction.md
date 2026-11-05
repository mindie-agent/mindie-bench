`GraphFusionPassManager` 在选择算子融合方案时触发了 192KiB UB 溢出且实际端到端耗时反而变慢（见 `logs/fusion_pass_ub_overflow.log`），请修复其双缓冲 UB 容量校验和因果 Roofline 关键路径优选逻辑。
