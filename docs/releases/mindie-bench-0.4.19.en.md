# mindie-bench 0.4.19: Run-Scoped Shared `uv` Cache, Windows I/O Concurrency Bounding & Timeout Arithmetic Calibration

[中文](mindie-bench-0.4.19.md)

`mindie-bench v0.4.19` eliminates redundant PyPI wheel downloads and Windows NTFS/Defender I/O thrashing across the 4-scenario `host-accept` acceptance suite:

- **Run-Scoped Isolated `uv` Cache Sharing (`host_acceptance/cli.py` · `__init__.py`)**: Allocates one disposable `mindie-ha-uvcache-*` directory per `run_host_acceptance` invocation (reaped in `finally`) shared across all 4 scenarios and all hosts in `--all-hosts`, eliminating repeated downloads and zip extractions of 180+ third-party wheels while preserving zero host pollution.
- **Windows Concurrency Bounding & Timeout Arithmetic (`cli.py` · `verify-host-plugin.yml` · `publish-release.yml` · `publish-product.yml`)**: Serializes per-host scenarios on Windows (`scen_workers=1`, `host_workers<=2`) so Scenario 1 warms the cache contention-free, and calibrates single-host budgets to `600s` inside `30m–55m` workflow envelopes (`outer_timeout >= sum(inner_budgets) + cleanup_grace`).
