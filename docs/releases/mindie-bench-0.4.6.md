# mindie-bench 0.4.6：`mindie_debugger` 故障与精度评测对齐、口语化题面套件、策略采纳观测与行为判分加固

**中文** · [English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.6/docs/releases/mindie-bench-0.4.6.en.md)

`mindie-bench 0.4.6` 将昇腾 NPU 故障根因定界与算子/张量数值精度定位评测全面对齐至独立组件 `mindie_debugger`，重写了面向真实开发者的口语化模糊题面套件，补齐了策略延续执行与轨迹采纳观测闭环，并将全量基准任务判分器加固为独立子进程行为验证，同时提供不污染历史回执的候选代码重评工具。

## 一、核心特性与工程改进

- **NPU 故障定界与数值精度定位切换至 `mindie_debugger`（职责边界解耦）**：
  - 将组件价值评测中的确定性昇腾 PLOG 故障根因定界（`value_diagnostics.py`）切换至 `mindie_debugger.npu_fault_triage`，将 `diagnostics` 严格聚焦于组件自身运行异常记录、隐私脱敏与去重上报。
  - 新增 5 个贴合真实生产缺陷的数值精度定位基准场景（`value_precision.py`），直接调用 `mindie_debugger.precision` 原语进行验证：覆盖 Kimi K3 空闲 DP 支路 GDN 状态污染、SFA MTP 草稿长度元数据别名覆写、KDA 变长算子第 74 个 chunk 越界踩内存导致 NaN 锁死、GroupNorm BF16 累加截断与 SparseMLA `-1` 索引散射锁死，以及 `256 -> 16` 专家裁剪 MoE 治愈锁步对比与 W4A8 伪量化/采样器漂移分离；确保修复前缺陷 100% 检出、修复后零误报且双次运行字节级一致。
  - 在发布依赖门禁（`check_release_dependencies.py`）、组件运行时元数据（`component_runtime.py`、`product_native.py`、`native_product.py`）及组件价值视图中完整注册 `mindie-debugger` 及新增宿主插件标识。
- **口语化开发者题面套件与 V2 100% 真实 `vllm-ascend` 34 题基准**：
  - 将口语化基准任务（`*-zh-vague`、`*-rsi-zh`）彻底淘汰早期 `vllm_ascend_diag` 人造教学桩，全部 34 道题目 100% 基于真实 `vllm_ascend/...` 生产模块构建，并将 `instruction.md` 统一为贴近日常研发提问习惯的自然中文描述（只给一句话现象与原始报错堆栈/日志路径，零工具名剧透），全量生成并通过健全性回执（`soundness_receipt.json`，`34/34` `0.0 -> 1.0`）。
  - 修复多领域知识库在原生产品模式下的按域初始化与挂载、分级知识捕获透传、远端 SSH shim 边界处理、DSH OpenAI-completions 缓存 Token 总量对账（`trial_finalize.py`）、DSH 单轮 `turn_end` 轮次保底（`drivers/dsh.py`）以及 MCP SDK 非 JSON 诊断输出过滤（`runner/process.py`）。
  - 完整归档跨 Codex (`gpt-6.1-sol`, `xhigh`) 与 DSH (`deepseek-flash`, `max`) 的 **34 题 144 场 Trial 八大组件与 `G0–G5` 全矩阵实测包**（[`reports/g0-g5-vllm-ascend-v2-20261103/REPORT.md`](../../reports/g0-g5-vllm-ascend-v2-20261103/REPORT.md)、`summary.json`、`audit_manifest.json`、`reproduce.sh`）。
- **策略延续执行器（`StrategyContinuationExecutor`）与轨迹策略采纳观测**：
  - 新增 `StrategyContinuationExecutor`（`runner/strategy_continuation.py`），在隔离的 `WorktreeSlot` 与原生沙箱中执行选定策略卡的后续修复，并完全交由独立本地判分器（`local_verifier`）裁定通过与退化率（`selected_and_completed_rate` / `regressed_rate`）。
  - 新增轨迹级策略采纳观测（`evaluation/strategy_adoption.py`），严格区分“已加载策略卡”（`loaded_strategy_refs`，仅统计 `preflight` 之后返回非空正文与明确版本的 `mindie_knowledge_explain` 调用）与“声明采纳”（`declared_adoption`，对应 `event_type="applied"` 的 `mindie_knowledge_feedback`），不把口头反馈当成实际修复成功。
  - 拆分 `runner/dsh_catalog.py` 使 `profile_builder.py` 保持在 `< 800` 行以内，并在 `verifier.py`、`trial_finalize.py`、`worktree_slot.py` 中确保 `TimeoutError` 先于 `OSError` 原样向上抛出。
- **行为级判分器加固（Behavioral Verifier Hardening）与历史候选重评器（`reevaluate.py`）**：
  - 将 CPU、L2、L3 与 G4 开放探索任务的判分器统一升级为真实子进程动态行为与数值校验（`verify.py` / `verify_cpu.py` / `runtime_controls.py` / `reference_cpu.py`），淘汰静态 AST 字符串匹配或局部代理判分。
  - 新增零模型调用的历史候选重评工具（`evaluation/reevaluate.py`），支持从已归档的本地工作区压缩包或远端补丁与未跟踪文件快照（`remote_cleanup.json` + `remote_final.patch`）还原候选代码，并用最新独立判分器重新打分；重评过程绝不修改原始 `trial_receipt.json`，也绝不将历史 `unknown` 或 `infra_failed` 篡改为通过。

## 二、测试与验证

- 策略延续执行、轨迹策略采纳、行为判分契约、历史候选重评与发布基准门禁等核心回归测试全部通过；`ruff` 静态检查与 Unix `LF` 行尾门禁全部通过。
