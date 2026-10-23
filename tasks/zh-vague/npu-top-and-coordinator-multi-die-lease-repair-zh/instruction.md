我们在昇腾 A2 (`910B4`) 与 A3 (`Ascend910` 双 Die) 集群上排查设备观测与多 Die 租约分配模块 `vllm_ascend_diag/multi_die_lease_allocator.py`（配套采集数据位于 `vllm_ascend_diag/fixtures/npu_coordinator_case.json`）。

当前 `reconcile_multi_die_and_leases()` 的实现存在四处硬件拓扑与调度缺陷：
1. **混淆 A3 双 Die 卡内 `Chip ID` 与全局物理 `Phy-ID` (`a3_parsed` 与 `a2_parsed`)**：在解析 A3 `dual_die_smi` 时，把每张卡的两个 Die 的 `phy_id` 都错写成了卡内局部 `0, 1`（正确应为 Card 0 `[0, 1]`、Card 1 `[2, 3]`、Card 2 `[4, 5]`），未把卡内双 Die 的 HBM 汇总到卡级（Card 0 应为 `3343 + 40000 = 43343 / 131072 MB`），未将 `(npu_id=0, chip_id=1)` 上的进程 `777` (`VLLMWorker_0`) 映射到 `phy_id = 1`，且未应用 `usages_text` 对单 Die/双 Die `aicore_percent` 的覆盖 (`usage_override_probe` 应更新为 `55` 与 `70`)；
2. **跨半忙卡错误分配同卡双 Die 租约 (`s1_dual_die` 与 `s1_pd_pair`)**：在 `same_npu_dual_die` (`S1`, `die_count=2`) 策略下，必须在同一张物理卡上分配两个均空闲（无进程且 `HBM < 8192 MB`）的兄弟 Die `(2k, 2k+1)`。由于 Card 0 的 `Phy-ID 1` 已被占用 (`40000 MB`)，Card 0 的 `Phy-ID 0` 不可用于 `same_npu_dual_die`，应分配 Card 1 的 `"2,3"`（若 Card 1 的 `Phy-ID 2` 出现 `>= 8192 MB` 未归属显存占用，则应自动跳过 Card 1 分配 Card 2 的 `"4,5"`）；在 `disaggregated_pd_pair` (`die_count=4`, `prefill=2`, `decode=2`) 下，应优先整卡分配互不重叠的 `{"prefill": "2,3", "decode": "4,5"}`；
3. **多角色 `distinct_hosts=True` 放置冲突与容量不足时的半分配 (`placement_ok` 与 `placement_busy`)**：在 64 个 runtime 目录中为 `prefill` 与 `decode` 分配 `Ascend910C` 时，`distinct_hosts=True` 应选出不同主机上的 `["runtime-32", "runtime-36"]`（而非同主机的 `32, 33`）；当 `runtime-36..63` 均忙碌导致可用主机不足 2 台时，必须整单拒绝并返回 `runtime_ids = []` 及非空 `reason`；
4. **DCMI 降频掩码位定义颠倒 (`dcmi_decoded`)**：bit `1 << 2` (`4`) 对应 `thermal_limit=True`，bit `1 << 3` (`8`) 对应 `power_limit=True`，未知高位（如 bit `63`）应记录在 `unknown_bits = [63]`，当 `ret != 0` (`ret_neg1`) 时 `thermal_limit` 应返回 `None`。

请修复 `vllm_ascend_diag/multi_die_lease_allocator.py` 中的 `reconcile_multi_die_and_leases()`。
（注：当前为本地离线复现环境，请优先检索 MindIE 领域知识库或直接复用环境中已安装的 `npu_top` 与 `mindie_coordinator` 组件完成修复与验证；若当前环境未安装对应组件，请按上述规范直接修复模块代码并用 `python3` 自测，无需连接远端 NPU。）
