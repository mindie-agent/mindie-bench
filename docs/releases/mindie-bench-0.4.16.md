# mindie-bench 0.4.16：`host-accept` 域存储路径、7 端原生转录 Schema 与 macOS 并发预算校准

[English](mindie-bench-0.4.16.en.md)

`mindie-bench v0.4.16` 基于首次三平台 7 端真实候选包联动验收结果，校准 `host-accept` 验收夹具与执行预算：

- **知识域存储路径与 `mindie-sim` 只读探测对齐（`upgrade_cutover.py` · `hooks_capture.py` · `mcp_wire.py`）**：`Store(root, domain=...)` 内部自动通过 `prepare_layout(root, domain)` 拼接域子目录，移除 `upgrade_cutover.py` 与 `hooks_capture.py` 中重复拼接的 `/vllm-ascend`；将 `mindie-sim` 实时烟雾探测参数统一收敛为零副作用只读子命令 `["catalog"]`。
- **7 端原生钩子与转录 Schema 精确对齐（`manifests.py` · `hooks_capture.py`）**：解析 `kimi.plugin.json` 中 `uv run ... ./scripts/mindie_launch.py` 钩子命令时保留真实脚本入口参数；同步补齐 DSH `session` 头记录与 `time`/`data.content` 结构、WorkBuddy `sessionId` + `message` 包裹字段及 OpenCode v2 SQLite 表结构，并将合成转录时间戳前移 `30s` 消除采集上界抖动。
- **三平台冷构建并发预算优化（`cli.py` · `verify-host-plugin.yml`）**：将 `host-accept` 默认单场景预算提升至 `240s`（`verify-host-plugin.yml` 设为 `300s`），并将冷启动虚拟环境并发上限收敛至 `<= 2`，消除 macOS GitHub Runner 四路并发 `uv` 构建饥饿。
