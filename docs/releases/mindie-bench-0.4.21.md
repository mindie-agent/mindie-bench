# mindie-bench 0.4.21：原生 Codex 升级门禁兼容 `mindie_sim.cli` 规范化模块绑定

[English](mindie-bench-0.4.21.en.md)

`mindie-bench v0.4.21` 对齐 Codex 旧端原生升级门禁与 `mindie-harness` 规范化组件命令表：

- **识别迁移后的 `-m mindie_sim.cli` 绑定（`.github/scripts/native_upgrade_acceptance.py`）**：`mindie_harness.component_commands.rebind_component_commands` 在宿主迁移时会将旧版 `["...", "-m", "mindie_sim"]` 规范化升级为 `["...", "-m", "mindie_sim.cli"]`；`native_upgrade_acceptance.py:verify` 现同时认可规范化后的 `mindie_sim.cli` 入口并校验其已重绑至新代 `current["python"]`。
