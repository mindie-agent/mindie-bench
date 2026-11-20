# mindie-bench 0.4.24：显式临时目录清理配对对齐 Hub `R4` `P8` 存储静态门禁

[English](mindie-bench-0.4.24.en.md)

`mindie-bench v0.4.24` 将 `run_host_acceptance` 中运行期共享目录的回收展开为直接变量绑定调用：

- **显式绑定 `rmtree` 清理（`src/mindie_bench/host_acceptance/cli.py`）**：对 `shared_hook_stage` 与 `shared_uv_cache` 分别执行显式 `rmtree_force` / `shutil.rmtree` 清理，满足顶层 Hub `R4`（`storage_temp_ast.py`）对每个 `mkdtemp` 变量的静态生命周期配对校验。
