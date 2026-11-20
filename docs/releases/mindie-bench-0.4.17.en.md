# mindie-bench 0.4.17: Windows UTF-8 Release Metadata Decoding Hardening (`build_wheels_bundle.py`)

[中文](mindie-bench-0.4.17.md)

`mindie-bench v0.4.17` fixes a Windows character-decoding failure when assembling wheelhouses against GitHub Release API payloads containing bilingual UTF-8 release notes:

- **Explicit UTF-8 Subprocess Decoding (`.github/scripts/build_wheels_bundle.py`)**: Sets `encoding="utf-8", errors="replace"` on `build_wheels_bundle._run` so Windows `cp1252` reader threads cannot fail with `UnicodeDecodeError` and leave `CompletedProcess.stdout` as `None` when `gh api` returns UTF-8 release bodies.
- **Workflow-Wide `PYTHONUTF8=1` Enforcement (`.github/workflows/{publish-release,publish-product,verify-host-plugin,fullstack-smoke}.yml`)**: Sets `PYTHONUTF8=1` across all 3-OS release and host-plugin qualification workflows for deterministic cross-platform behavior.
