## 变更摘要 (Summary)

- Describe the root cause / trigger and the resulting behavioral change.

## 设计原则 Review (Pre-PR Checklist — `bench`)

- [ ] **BENCH-ORACLE** — Sealed oracle bundles, negative-control probes, and candidate workspaces remain strictly isolated, with `outer_timeout >= inner_budget + 20s` on remote private verifiers.
- [ ] **BENCH-VERDICT** — `completed_pass`, `completed_fail`, `infra_failed`, and `unknown` remain distinct in every denominator, and post-verifier cleanup/metric exceptions never overwrite a confirmed `PASS`/`FAIL` verdict (`cleanup_warning` / `metrics_incomplete`).
- [ ] **BENCH-PROVENANCE** — Cross-boundary SHA-256/Git provenance (candidate, runner, sealed oracle, release assets, `mindie-harness` birth-release gate) remains verified without redundant per-trial internal source re-walks.
- [ ] **BENCH-ISOLATION** — G0–G3 ablation groups, credential scopes, temporary worktrees/containers, and benchmark-only rules stay strictly isolated from production plugins, with Unix `LF` line endings across all files.

## 证据记录 (Evidence)

- **BENCH-ORACLE**: `<command / test and measured output, or not-applicable: docs-only>`
- **BENCH-VERDICT**: `<command / test and receipt verification, or not-applicable: docs-only>`
- **BENCH-PROVENANCE**: `<command / release or provenance check, or not-applicable: docs-only>`
- **BENCH-ISOLATION**: `<command / isolation & cleanup check, or not-applicable: docs-only>`

## 验证结果 (Verification)

- Executed test suites (`ruff`, `pytest`, `mindie-bench demo`), wheel smoke test, LF check, and any unexecuted live-NPU scope.
