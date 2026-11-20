# mindie-bench 0.4.23: Windows File-Backed `host-accept` Log Redirection, Same-Volume Hardlinks & Shared Read-Only Hook Assets

[中文](mindie-bench-0.4.23.md)

`mindie-bench v0.4.23` eliminates Windows `Popen._communicate` grandchild pipe-handle blocking and cross-volume venv copy overhead during 3-OS `qualify-product` verification:

- **File-Backed `host-accept` Log Redirection & `$RUNNER_TEMP` Alignment (`.github/workflows/publish-product.yml` · `.github/workflows/publish-release.yml`)**: Redirects `host-accept` `stdout`/`stderr` to a file instead of `capture_output=True` pipes and binds `TMP`/`TEMP`/`TMPDIR` to `$RUNNER_TEMP`, preventing detached Windows grandchildren from holding pipe EOF open in `Popen._communicate` while placing sandboxes on Defender-excluded runner storage.
- **Same-Volume `UV_LINK_MODE=hardlink` & Shared Read-Only Multi-Host Hook Assets (`src/mindie_bench/host_acceptance/`)**: Uses `hardlink` on Windows whenever `UV_CACHE_DIR` and `sandbox_root` share a drive letter (falling back to `copy` across drives) and provisions one run-scoped read-only `hook-venv` + cached `gitleaks` binary across multi-host `host-accept --all-hosts` runs.
