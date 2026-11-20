# mindie-bench 0.4.17：Windows UTF-8 发布元数据解码加固（`build_wheels_bundle.py`）

[English](mindie-bench-0.4.17.en.md)

`mindie-bench v0.4.17` 修复了 Windows GitHub Runner 在组装候选与正式 wheelhouse 时解析含中文双语 Release 说明的 GitHub API 响应时的编码异常：

- **显式 UTF-8 子进程解码（`.github/scripts/build_wheels_bundle.py`）**：为 `build_wheels_bundle._run` 显式指定 `encoding="utf-8", errors="replace"`，消除 Windows 默认 `cp1252` 读取器线程在解析中文 Release JSON 字节流时抛出 `UnicodeDecodeError` 并导致 `CompletedProcess.stdout` 置为 `None` 的问题。
- **全量可复用发版与验收工作流强制 UTF-8 模式（`.github/workflows/{publish-release,publish-product,verify-host-plugin,fullstack-smoke}.yml`）**：在三平台工作流环境变量中统一注入 `PYTHONUTF8=1`，确保跨平台发版门禁与 7 端验收在 Windows、macOS 与 Linux 上行为一致。
