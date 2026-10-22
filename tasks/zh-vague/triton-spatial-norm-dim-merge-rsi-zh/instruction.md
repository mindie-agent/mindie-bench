我们在昇腾 910B (A2, 48 VectorCores, 192KB UB) 上调优 4D 空间分组归一化规约算子 `vllm_ascend/ops/spatial_group_norm_reduce.py`（输入形状 `[N, C, H, W]`，`bfloat16`）。

前期的第一阶段调优记录在 `logs/operator_stage1_l3_attempt.json`：我们已经按照常规 `L3` Tiling 工作流将 1D Persistent Grid 钳制到了物理 VectorCore 数量 `grid = (min(total_tiles, NUM_VECTOR_CORES),)`，但连续两轮尝试仍然停滞（`verdict: "unsupported"`，耗时 `51.4 ms`，且最大绝对误差 `0.0184` 超出 `5e-3` 门禁）：
1. 算子内部仍保留 `c -> h -> w` 三层嵌套循环（`nested_loop_depth = 3`），在内层反复做标量 `sum` 累加与地址重算；
2. `block_size = 512` 固定写死，未根据 `192KB` UB 预算与活跃 `fp32` 缓冲区数量做 2 的幂次动态放大；
3. 在多趟规约（mean/var + normalize）路径上盲目开启了 `enable_multibuffer = True`（存在 UB 溢出风险），且中间部分和被截断回 `bfloat16` 并落盘了额外的 `fp32` GM 临时张量（`intermediate_gm_copies = 2`）。

请你按以下步骤完成定位与深度探索重构：
1. 先使用 `mindie_knowledge_query`（指定 `domain="operators"`，默认 `mode="reuse"`）检索算子知识库中关于 `Triton-Ascend persistent grid tiling UB budget` 的常规经验，并结合 `logs/operator_stage1_l3_attempt.json` 确认常规 `L3` 工作流为何停滞；
2. 确认常规 `L3` 工作流停滞后，调用 `mindie_knowledge_query(..., domain="operators", mode="explore", bottleneck_summary=..., ledger_entries=...)` 切换到 **RSI 独立探索模式**（剥离 `L3` 工作流，挂载 `L4` 代码优化变换 `mutator`），并用 `mindie_knowledge_explain` 精读命中的 `L4` Mutator 全文；
3. 严格按照该 `L4` Mutator 定义的 `select_optimal_block_size`、三项可分离变换因子及留一法消融规范重构 `vllm_ascend/ops/spatial_group_norm_reduce.py`，实现 `run_ablation_suite()` 函数（返回四组消融实验结果）；
4. 运行消融验证并在仓库根目录生成实验台账 `experiment-ledger.jsonl`（记录 baseline、完整候选及三组留一法消融的 `run_id`、`arm`、`verdict` 与关键指标）和总结报告 `PROGRESS.md`。

（注：当前为本地验证环境，请使用 `/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python` 执行本地 Python 脚本与消融验证，无需连接远端 NPU。）
