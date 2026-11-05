# mindie-bench 0.4.7：Ascend 950DT 端到端 RSI 自动晋升与 C6 沉淀 A/B 基准套件、单一变量门禁与 DSH 收据加固

**中文** · [English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.7/docs/releases/mindie-bench-0.4.7.en.md)

`mindie-bench 0.4.7` 新增面向 **Ascend 950DT (A5)** 真实业务场景的端到端 RSI 自动晋升（`reuse -> explore`）与 Stop 阶段 `graded-c6` 沉淀 A/B 对照基准套件（`track-RSI-E2E-v4`），强化 `G2` 与 `G4` 单一变量控制与 `MINDIE_DISABLE_RSI_EXPLORE=1` 隔离门禁，并修复 macOS 进程组清理 `PermissionError` 与 DSH `max_turns` 边界下的原生完成回执判定。

## 核心更新

- **Ascend 950DT (A5) 端到端 RSI A/B 评测任务套件（`tasks/zh-vague/*-rsi-zh`）**：
  - 覆盖 `vllm-ascend-mla-prolog-rope-kvcache-fusion-rsi-zh`、`dsv4-sparse-mla-qli-fusion-rsi-zh`、`vllm-ascend-dspark-mtp-aclgraph-d2h-rsi-zh` 与 `triton-spatial-norm-dim-merge-rsi-zh` 4 个受控任务 Fixture，全部采用自然语言模糊提示词 + 服务拉起/压测命令 + 日志路径，公开测试文件采用 SHA-256 摘要校验内部契约，零明文私有函数/因子名泄露。
- **单一变量控制与 `MINDIE_DISABLE_RSI_EXPLORE=1` 隔离（`product_native.py` / `graded_capture.py`）**：
  - 确保同一任务的 `G2`（Control）与 `G4`（Treatment）共享完全相同的 `task_digest`、`corpus_commit` 与初始 `corpus_view_sha256`；在 `G2` 中自动注入 `MINDIE_DISABLE_RSI_EXPLORE=1` 物理剥离 `L4` 知识单元。
- **DSH `max_turns` 完成态与 macOS `killpg` 权限边界加固（`drivers/dsh.py` / `dsh_catalog.py` / `trial_preflight.py`）**：
  - 当 DSH 会话在完成代码修复并通过评测后于最后一轮触达 `max_turns` 时，允许 `graded-c6` Stop 捕获流程正常提炼并提交本地 Git 知识单元；修复 macOS 下 `os.killpg` 抛出 `PermissionError` 的边界异常。
- **16 组端到端 RSI A/B 实测结果（`track-RSI-E2E-v4`）**：
  - 在 Codex (`gpt-6.1-sol xhigh`) 与 DSH (`deepseek-flash max`) 双宿主上，`G2` 对照组通过率 `0/8 (0.0%)`，`G4` 实验组通过率 `8/8 (100.0%)`，`8/8` 触发零人工干预的 `reuse -> explore` 自动晋升（`caller_initiated_explore=False`），`8/8` 完成 Stop 阶段 `graded-c6` 本地裸 Git 仓库沉淀（`completed:applied`），并通过 Codex `gpt-6-astra (max)` 独立可信度审计。
