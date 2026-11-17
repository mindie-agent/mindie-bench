# mindie-bench 0.4.11：核心评测运行器模块化瘦身、`mindie-harness` 首发策略支持与 `bench` 治理对齐

[English](mindie-bench-0.4.11.en.md)

`mindie-bench v0.4.11` 完成评测框架洁净架构重构与组织级发布控制面扩展：

- **核心大模块 `< 800` 行正交拆分与不可达孤岛清理**：将超长运行器模块（`gate.py`、`harness.py`、`release.py`、`value.py` 等）按职责拆分为 `< 800` 行的内聚子模块并补齐四段式契约头，清理历史未引用的死代码孤岛，保持 20 个行为级任务与 109 项三态/辅助变体对照契约 100% 不变。
- **四态诚实判分与超时异常透传加固**：严格区分 `completed_pass`、`completed_fail`、`infra_failed` 与 `unknown`，在全部 `OSError` 捕获边界前置 `except TimeoutError: raise`，杜绝基础设施异常或超时静默混入模型业务得分。
- **`mindie-harness` 首发发布策略与 `mindie-agent-oc` 消费契约接入**：在 `release-policy.json`、`check_release_dependencies.py`、`upgrade_acceptance.py` 与 `consumer_contract.py` 中接入 `mindie-agent/harness` 首发校验及 `mindie-agent/mindie-agent-oc` 原生插件发布元数据校验，并启用 `--profile bench` PR 设计原则门禁。
