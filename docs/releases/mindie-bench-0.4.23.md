# mindie-bench 0.4.23：Windows `host-accept` 文件重定向防管道句柄阻塞、同盘硬链接与多宿主共享只读钩子环境

[English](mindie-bench-0.4.23.en.md)

`mindie-bench v0.4.23` 修复 Windows 云端运行器在产品整包 `qualify-product` 中因继承管道句柄导致的 `Popen._communicate` 等待阻塞与跨盘重复拷贝开销：

- **文件日志重定向替代 `capture_output=True` 匿名管道（`.github/workflows/publish-product.yml` · `.github/workflows/publish-release.yml`）**：将 `host-accept` 的 `stdout`/`stderr` 重定向至磁盘日志文件并将 `TMP`/`TEMP`/`TMPDIR` 锚定至 `$RUNNER_TEMP`，消除 Windows 后台孙进程继承管道写端导致 `stdout_thread.join()` 阻塞至超时的现象，同时对齐 `D:` 盘免 Defender 实时扫描路径。
- **Windows 同盘自动启用 `UV_LINK_MODE=hardlink` 与多宿主共享只读钩子资产（`src/mindie_bench/host_acceptance/`）**：当 `UV_CACHE_DIR` 与沙箱位于同一盘符时默认启用 `hardlink`（跨盘自动回退 `copy`），并在 `--all-hosts` 运行期一次性预置只读 `hook-venv` 与缓存的 `gitleaks` 二进制供 7 个宿主共享复用。
