# mindie-bench 0.4.12：对齐 Codex 原生旧端升级验收的默认开启知识 MCP 发现契约

[English](mindie-bench-0.4.12.en.md)

`mindie-bench v0.4.12` 将原生 Codex 旧客户端升级验收门禁（`.github/scripts/native_upgrade_acceptance.py`）与全七端静默自启（Silent Default-On，`P2.7` / `P2.8`）公开契约对齐：

- **默认开启知识 MCP 发现契约校验**：从历史 `v0.1.23` 升级至新版 `mindie-agent-codex`（`v0.1.37+`）并在全新会话执行 `mcpServerStatus/list` 时，`mindie-tools` 默认暴露全套 5 项 `L1–L4` 知识工具（`knowledge_query`、`knowledge_explain`、`knowledge_feedback`、`knowledge_distill`、`knowledge_sync`）与确定性组件工具（`diagnostics_status` 等），同时 `mindie-remote-dev` 保持纯远端工具边界（零 `knowledge_*` 混入），并继续验证 `Stop` 禁用配置继承、独立组件楼层保留与外部组件代际重绑。
