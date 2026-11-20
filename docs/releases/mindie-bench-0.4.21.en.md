# mindie-bench 0.4.21: Accept Canonical `-m mindie_sim.cli` Rebinding in Native Codex Upgrade Gate

[中文](mindie-bench-0.4.21.md)

`mindie-bench v0.4.21` aligns the native Codex old-client upgrade verifier with `mindie-harness`'s canonical component command table:

- **Recognize Migrated `-m mindie_sim.cli` Bindings (`.github/scripts/native_upgrade_acceptance.py`)**: `mindie_harness.component_commands.rebind_component_commands` upgrades legacy `["...", "-m", "mindie_sim"]` entries to `["...", "-m", "mindie_sim.cli"]` during host migration; `native_upgrade_acceptance.py:verify` now accepts the canonical `mindie_sim.cli` module alongside `mindie_sim` while verifying interpreter rebinding to `current["python"]`.
