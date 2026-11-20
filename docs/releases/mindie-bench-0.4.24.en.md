# mindie-bench 0.4.24: Explicit `mkdtemp` Cleanup Pairing for Hub `R4` `P8` Storage AST Gate

[中文](mindie-bench-0.4.24.md)

`mindie-bench v0.4.24` unrolls run-scoped temporary directory cleanup in `run_host_acceptance` for direct AST variable tracking:

- **Direct `rmtree` Variable Pairing (`src/mindie_bench/host_acceptance/cli.py`)**: Unrolls `finally` cleanup onto `shared_hook_stage` and `shared_uv_cache` directly so Hub `R4` (`storage_temp_ast.py`) statically verifies `P8` temporary-directory pairing.
