# mindie-bench 0.4.5: Native Knowledge Evaluation Integration, Shell Temp Isolation & Missing-Metric Accounting

[中文](https://github.com/mindie-agent/mindie-bench/blob/v0.4.5/docs/releases/mindie-bench-0.4.5.md) · **English**

`mindie-bench 0.4.5` standardizes production knowledge evaluation wiring across the native Codex and DSH harnesses, fixes `zsh` heredoc temp-file sandbox denials, and tightens metric accounting when private CLI knowledge calls are only partially observable via named MCP events.

## 1. Key Fixes & Engineering Improvements

- **Production Native Knowledge Wiring for Codex & DSH**:
  - Native Codex evaluation runs through the plugin's production entry point, real `UserPromptSubmit` hook authorization, and private knowledge CLI while preserving isolated profiles and configured model/reasoning settings.
  - Native DSH configuration injects `mcpPluginUrl` and verifies the mounted host catalog during preflight rather than treating standalone Store or MCP startup as proof of host attachment.
  - Both harnesses reuse installed component CLIs and task dependency paths, recording loaded module source and wheel digests in artifact provenance.
- **Honest `null` Accounting for Partially Observed CLI Knowledge Calls**:
  - When named MCP events do not cover all private CLI knowledge invocations in Codex, total `query`/`explain` counts, knowledge latency, and residual non-knowledge time remain `null`, while observed named calls are reported separately as a lower bound.
  - Summary and component reports no longer coerce missing coverage to `0` or count an entire shell command's elapsed time as knowledge retrieval time.
- **Sandboxed `zsh` `TMPPREFIX` Binding**:
  - Native terminal setup binds `TMPPREFIX` to the profile's managed temporary directory so `zsh` heredocs and here-strings no longer attempt to write to unauthorized `/tmp/zsh*` paths.

## 2. Verification

- **`73 passed`** across focused local regression tests covering real terminal processes, native event parsing, and missing-value aggregation boundaries; `ruff`, Unix `LF` line-ending checks, and cross-platform CI all pass.
