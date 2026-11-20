# mindie-bench 0.4.22: Bounded 2-Host Concurrency in `host-accept --all-hosts` & Structured Workflow Diagnostics

[中文](mindie-bench-0.4.22.md)

`mindie-bench v0.4.22` eliminates CPU starvation when running `host-accept --all-hosts` on 3–4 vCPU GitHub Actions runners during 3-OS `qualify-product` verification and surfaces structured per-host failure summaries directly:

- **Bounded 2-Host Concurrency & Sequential Per-Host Scenarios (`src/mindie_bench/host_acceptance/cli.py`)**: Caps cross-host concurrency at `host_workers = min(2, len(clean_hosts))` across all platforms and runs the 4 scenarios sequentially per host while sharing one run-scoped `uv` wheel cache, preventing `7 hosts × 2 scenarios = 14`-way `uv`/MCP CPU thrashing on 3–4 vCPU runners from starving the `6s` `Stop` capture hook budget.
- **Structured Failure Summary Surfacing (`.github/workflows/publish-product.yml` · `.github/workflows/publish-release.yml`)**: Extracts per-host and per-scenario `error` payloads from `--output` JSON before tail-truncating `stdout`/`stderr` whenever `host-accept --all-hosts` exits non-zero.
