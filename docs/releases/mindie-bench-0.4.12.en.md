# mindie-bench 0.4.12: Align Native Codex Upgrade Acceptance with Default-On Knowledge MCP Discovery

[中文](mindie-bench-0.4.12.md)

`mindie-bench v0.4.12` aligns the native Codex old-client upgrade acceptance gate (`.github/scripts/native_upgrade_acceptance.py`) with the Silent Default-On (`P2.7` / `P2.8`) public contract:

- **Default-on `knowledge_*` MCP discovery contract**: After upgrading from `v0.1.23` to `mindie-agent-codex v0.1.37+` and querying `mcpServerStatus/list` on a fresh native thread, `mindie-tools` must advertise all five `L1–L4` knowledge tools (`knowledge_query`, `knowledge_explain`, `knowledge_feedback`, `knowledge_distill`, `knowledge_sync`) alongside deterministic component tools (`diagnostics_status`), while `mindie-remote-dev` remains strictly scoped to remote tools (no `knowledge_*` leakage) and explicit `Stop` disablement, independent component floors, and external component rebindings remain preserved.
