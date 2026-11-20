# mindie-bench 0.4.20：修复 Windows 无窗 `uv` 钩子守卫桩的句柄继承死锁与 MCP 进程树清理

[English](mindie-bench-0.4.20.en.md)

`mindie-bench v0.4.20` 根治 Windows 下 `codex`（PowerShell）、`cc`（Git Bash）与 `agy`（Git Bash）在 `hooks_and_capture` 场景中的无窗管道死锁：

- **显式传递 `uv-guard` 桩的 `stdin`/`stdout`/`stderr` 句柄（`host_acceptance/hooks_capture.py`）**：Windows PEP-405 `.exe` 守卫桩在 `CREATE_NO_WINDOW` 下显式传递有效标准句柄（对无效句柄回退 `DEVNULL`），修复子 `uv run` 因 `bInheritHandles=False` 丢失重定向管道导致的永久阻塞。
- **Windows MCP 会话超时树级清理（`host_acceptance/mcp_wire.py`）**：`MCPWireSession.close()` 超时回退时调用无窗 `taskkill /F /T /PID`，完整回收 `uv.exe -> python.exe -> .venv\Scripts\python.exe` 多层子进程树。
