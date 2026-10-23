我们在昇腾 A3 (`Ascend910_9362`) 上评估算子融合与整网性能预测脚本 `vllm_ascend_diag/a3_sim_planner.py`（配套公开输入位于 `vllm_ascend_diag/fixtures/sim_four_layer.json`、`qwen2.5-7b-instruct.json` 与 `mixed_operator_plan.json`）。

当前 `evaluate_a3_simulation_suite()` 的实现存在五处仿真建模缺陷：
1. **Layer 1 Roofline 与 HAP 评分公式错误**：未按 `Ascend910_9362` 单 Die 硬件规格（20 AIC, 8192 bf16 ops/cycle/AIC, 1650 MHz -> `270.336 TFLOP/s`）计算 `t_hw_us` (`508.40048484848484 us`)，且未使用 Hardware-Anchored Performance 公式 `HAP = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))` 计算 `hap_score` (`0.8429360983266381`)；
2. **Layer 2 周期流水线回边 (`distance=1`) 求解错误**：直接取单级最大耗时 `200.0 ns`，而未按 `load (100ns) -> mma (200ns) -> store (50ns) -> load (distance=1)` 环路求得 `minimum_period_ns = minimum_tile_ii_ns = 350.0 ns`；
3. **Layer 3 UB 子图融合的 GM 读写双倍消除与 UB 容量校验缺失**：消除 `producer` (`matmul`) 与 `consumer` (`aiv_cast`, `5.0 us`) 之间的 `1 MiB` GM 边时，会同时省去 producer 的 GM 写出与 consumer 的 GM 读入（`removable_gm_bytes = 2 * MiB`、`critical_path_saving_us = 5.0 us`），且必须校验 `live_buffers` 是否超出 `UB` 容量上限 (`248 KiB`) 而非 `L1`；
4. **Layer 4 A3 双 Die 集合通信层级缺失**：16 卡 A3 `all_reduce` 必须完整包含 `["d2d_intra_package", "d2d_intra_node", "d2d_inter_node"]` 三层拓扑；
5. **整网 `PhysicalCalibration` 的 `e2e_ms` 步数与算子 DAG `not_npu_measured` 标记错误**：生成 `N = generated_tokens` 个 token 时，首个 token 由 prefill `ttft_ms` 产出，后续仅有 `N - 1` 步 decode `tpot_ms`（即 `e2e_ms = ttft_ms + (generated_tokens - 1) * tpot_ms`），且 `physical_calibration` 状态应为 `"configured"` (`timing_scope = "device_event_warm"`，`applied_op_count = 452`)；`operator_plan` 的 `elapsed_ms` 应等于 `execution.makespan_ms` 且所有算子 `not_npu_measured = True`。

请修复 `vllm_ascend_diag/a3_sim_planner.py` 中的 `evaluate_a3_simulation_suite()`，使其在默认输入与 UB 容量超限扰动输入 (`ub_scratch_override_kib=300`) 下均通过校验。
（注：当前为本地离线复现环境，请优先检索 MindIE 领域知识库或直接复用环境中已安装的 `mindie_sim` 组件完成修复与验证；若当前环境未安装对应组件，请按上述规范直接修复模块代码并用 `python3` 自测，无需连接远端 NPU。）
