# mindie-bench 0.4.20: Fix Windows Windowless `uv` Hook-Guard Stdio Inheritance Deadlock & MCP Tree Reap

[中文](mindie-bench-0.4.20.md)

`mindie-bench v0.4.20` resolves the Windows pipe-inheritance hang affecting `codex` (PowerShell), `cc` (Git Bash), and `agy` (Git Bash) during `hooks_and_capture`:

- **Explicit `stdin`/`stdout`/`stderr` Forwarding in `_install_uv_sync_guard` (`host_acceptance/hooks_capture.py`)**: Passes validated stdio handles (`DEVNULL` fallback) to `subprocess.run(..., creationflags=CREATE_NO_WINDOW)` inside the PEP-405 `.exe` guard shim so `bInheritHandles=True` preserves redirected hook pipes across PowerShell and Git Bash.
- **Windows Process-Tree Reaping in `MCPWireSession.close()` (`host_acceptance/mcp_wire.py`)**: Uses windowless `taskkill /F /T /PID` when `proc.wait` times out on Windows so `uv.exe -> python.exe -> .venv\Scripts\python.exe` child trees never outlive scenario teardown.
