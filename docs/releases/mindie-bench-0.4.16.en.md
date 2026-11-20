# mindie-bench 0.4.16: `host-accept` Domain Store Layout, Native Transcript Schemas & macOS Concurrency Budget Calibration

[中文](mindie-bench-0.4.16.md)

`mindie-bench v0.4.16` calibrates the `host-accept` harness and execution budgets following the first live 3-OS × 7-host candidate qualification sweep:

- **Knowledge Domain Store Layout & Read-Only `mindie-sim` Probe Alignment (`upgrade_cutover.py` · `hooks_capture.py` · `mcp_wire.py`)**: Aligns `Store(root, domain=...)` invocations with `prepare_layout(root, domain)` by removing duplicate `/vllm-ascend` path suffixes in `upgrade_cutover.py` and `hooks_capture.py`, and standardizes `mindie-sim` live smoke calls on the read-only `["catalog"]` subcommand.
- **Exact Native Hook & Transcript Schema Parity (`manifests.py` · `hooks_capture.py`)**: Preserves the script entrypoint and trailing hook arguments when parsing `uv run ... ./scripts/mindie_launch.py` commands in `kimi.plugin.json`, aligns DSH `session` header + `time`/`data.content` envelopes, WorkBuddy `sessionId` + `message` records, and OpenCode v2 SQLite tables, and backdates synthetic fixture timestamps by `30s` to prevent scan-horizon boundary jitter.
- **3-OS Cold-Build Concurrency & Timeout Budget (`cli.py` · `verify-host-plugin.yml`)**: Raises the default per-scenario timeout to `240s` (`300s` in `verify-host-plugin.yml`) and caps parallel scenario workers at `<= 2` to prevent 4-way cold `uv` virtual-environment starvation on macOS runners.
