# mindie-bench 0.4.18: Canonical `local-only` Consent Authority, Windows Extended-Path Scan & Cold `remote_auth` Status Parity

[中文](mindie-bench-0.4.18.md)

`mindie-bench v0.4.18` aligns `host-accept` wire and capture verification contracts across all 7 real host plugins:

- **Canonical `mindie-consent/1` & `mindie-community-config/1` Authority Fields (`hooks_capture.py`)**: Populates valid 32-hex `choice_revision` and `generation` fields plus `"local_only": true` for positive `Stop` capture verification, matching `mindie_knowledge.loop.settings` production validation.
- **Windows Extended-Length (`\\?\`) Material Traversal & Lock Skipping (`hooks_capture.py`)**: Walks `<MINDIE_DATA_HOME>` with Windows `\\?\` extended paths so `> 260`-character `materials/tasks/<sha256>/blocks/<sha256>.md` artifacts are verified cleanly on Windows while skipping active `.lock` files.
- **Cold-Sandbox `remote_auth(action="status")` Structured Response Check (`mcp_wire.py`)**: Accepts structured `error_code == "endpoint_incomplete"` responses when probing `remote_auth` in a cold sandbox with no configured default SSH host.
