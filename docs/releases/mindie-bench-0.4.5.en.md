# mindie-bench 0.4.5: Native Knowledge Evaluation and Missing Measurement Semantics

[中文](https://github.com/mindie-agent/mindie-bench/blob/v0.4.5/docs/releases/mindie-bench-0.4.5.md) · **English**

Native Codex evaluation uses the plugin's production entry, actual `UserPromptSubmit` authorization, and private knowledge CLI while preserving isolated profiles and the selected model and reasoning configuration. Native DSH configuration includes the required `mcpPluginUrl` and checks the actually mounted host catalog before paid execution. Starting a standalone Store or MCP process is not accepted as proof that the host loaded the plugin. Both hosts reuse installed component commands and task dependency paths; artifact identities now include actual module source and wheel digests.

When named MCP events cannot fully observe Codex's private knowledge calls, absolute query/explain counts, knowledge time, and the corresponding remainder remain `null`. Observed named calls are retained separately as lower bounds. Aggregates and component reports no longer convert missing or partial coverage to zero, or treat an entire shell command's duration as knowledge time. Actual model tokens, wall time, and business oracle results remain separate measurements.

Native terminal setup binds zsh's `TMPPREFIX` to the existing profile temporary directory, preventing heredoc and here-string operations from trying to write to unauthorized `/tmp/zsh` paths without broadening sandbox access. Historical receipts, failure logs, frozen artifacts, and paid-request usage are preserved.

The merged functional commit passed 73 focused local tests covering real terminal processes, native event parsing, and missing-value aggregation. Ruff, LF, and cross-platform CI also passed. The formal release workflow still installs this exact tag's locked dependencies, verifies formal first-party dependency Releases, runs local contracts and the retrieval-scoring microbenchmark, and publishes evidence and checksums.

This release corrects evaluation integration and measurement, not proof of an overall knowledge-policy benefit. Small single-bug CPU pairs showed a cost-reduction signal, but the complex-task non-regression threshold was not met. Codex temporary-file noise, unfinished attempts, and incomplete usage remain part of the evidence. Release preparation adds no paid model calls and does not rewrite historical results.
