# mindie-bench 0.4.7：Ascend 950DT 端到端 RSI 自动晋升与 C6 沉淀 A/B 基准套件、单一变量门禁与 DSH 收据加固

**中文** · [English](https://github.com/mindie-agent/mindie-bench/blob/v0.4.7/docs/releases/mindie-bench-0.4.7.en.md)

`mindie-bench 0.4.7` 新增面向 **Ascend 950DT (A5)** 真实业务场景的端到端 RSI 自动晋升（`reuse -> explore`）与 Stop 阶段 `graded-c6` 沉淀 A/B 对照基准套件（`track-RSI-E2E-v4`），强化 `G2` 与 `G4` 单一变量控制与 `MINDIE_DISABLE_RSI_EXPLORE=1` 隔离门禁，并修复 macOS 进程组清理 `PermissionError` 与 DSH `max_turns` 边界下的原生完成回执判定。

## 发布与自动更新闭环

代码仓统一从受保护 `main` 启动发布，事务内固定同一个控制器提交。第一方依赖通过已发布旧客户端的完整校验；完整安装包、基准、校验和与真实整代升级证据先进入草稿并下载验证，随后才由专用环境密钥创建精确源码 tag，公开不可变 Release。公开后再次使用正常发布源验收升级。

每次发布实际升级完整八组件基线；局部报告不能授权正式发布。Codex 另从正式 `0.1.23` 经过组件独立升级、外部组件绑定和过期 CAS 拒绝，再由真实原生启动恢复并读取 MCP 工具，同时保留组件版本下限和禁用的 Stop Hook。候选源与公开源证据分别记录，默认追最新版不变。以下性能实验仍保留原始测量范围，不作为这次发布控制器改动的新实验。

`knowledge` 可选依赖锁定正式 `0.13.14` 的精确源码提交，修正原先版本号为 `0.13.13`、提交却与正式 tag 不一致的依赖。升级验收与发布校验使用已发布旧客户端声明的完整运行依赖，并在独立干净任务中重复公开源升级验收。

## 核心更新

- **Ascend 950DT (A5) 端到端 RSI A/B 评测任务套件（`tasks/zh-vague/*-rsi-zh`）**：
  - 覆盖 `vllm-ascend-mla-prolog-rope-kvcache-fusion-rsi-zh`、`dsv4-sparse-mla-qli-fusion-rsi-zh`、`vllm-ascend-dspark-mtp-aclgraph-d2h-rsi-zh` 与 `triton-spatial-norm-dim-merge-rsi-zh` 4 个受控任务 Fixture，全部采用自然语言模糊提示词 + 服务拉起/压测命令 + 日志路径，公开测试文件采用 SHA-256 摘要校验内部契约，零明文私有函数/因子名泄露。
- **单一变量控制与 `MINDIE_DISABLE_RSI_EXPLORE=1` 隔离（`product_native.py` / `graded_capture.py`）**：
  - 确保同一任务的 `G2`（Control）与 `G4`（Treatment）共享完全相同的 `task_digest`、`corpus_commit` 与初始 `corpus_view_sha256`；在 `G2` 中自动注入 `MINDIE_DISABLE_RSI_EXPLORE=1` 物理剥离 `L4` 知识单元。
- **DSH `max_turns` 完成态与 macOS `killpg` 权限边界加固（`drivers/dsh.py` / `dsh_catalog.py` / `trial_preflight.py`）**：
  - 当 DSH 会话在完成代码修复并通过评测后于最后一轮触达 `max_turns` 时，允许 `graded-c6` Stop 捕获流程正常提炼并提交本地 Git 知识单元；修复 macOS 下 `os.killpg` 抛出 `PermissionError` 的边界异常。
- **16 组端到端 RSI A/B 实测结果（`track-RSI-E2E-v4`）**：
  - 在 Codex (`gpt-6.1-sol xhigh`) 与 DSH (`deepseek-flash max`) 双宿主上，`G2` 对照组通过率 `0/8 (0.0%)`，`G4` 实验组通过率 `8/8 (100.0%)`，`8/8` 触发零人工干预的 `reuse -> explore` 自动晋升（`caller_initiated_explore=False`），`8/8` 完成 Stop 阶段 `graded-c6` 本地裸 Git 仓库沉淀（`completed:applied`），并通过 Codex `gpt-6-astra (max)` 独立可信度审计。
