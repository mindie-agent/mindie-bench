# MindIE Bench — Agent-Native Architecture & Invariants Guide (`v0.0.2`)

`mindie-bench` is the closed-loop evaluation, ablation, and NPU verification harness for the MindIE Agent ecosystem. This codebase is **100% Agent-maintained** and follows the `v0.0.2` Agent-Native architecture standards.

---

## 1. Module Panorama (`src/mindie_bench/`)

Every core module starts with a standardized **Module-Top Agent Invariant Header** (`[Role]`, `[Invariants]`, `[Timeouts & Locks]`, `[Side-Effect Policy]`) within its first 30 lines:

| Domain | Module | Responsibility & Key Entrypoints |
|---|---|---|
| **Models & Schemas** | `src/mindie_bench/models.py` | Pydantic task, trajectory, receipt, and ablation schemas (`TaskSpec`, `TrialReceipt`, `TerminalStatus`, `Group` `G0..G3`). |
| **Runner (`runner/`)** | `src/mindie_bench/runner/orchestrator.py` | End-to-end trial and matrix orchestrator (`run_trial`, `run_matrix`), enforcing pre-flight MCP probing, G0..G3 isolation, and post-trial experience capture. |
| | `src/mindie_bench/runner/verifier.py` | Fail-closed L1/L2/L3 task verifier (`verify_task`), enforcing pre/post workspace inventory, read-only path checks, and post-timeout external side-effect classification. |
| | `src/mindie_bench/runner/remote_slot.py` | Remote SSH + Docker Ascend NPU slot controller (`RemoteSlot`, `SlotTarget`), executing commands in isolated process groups with lease heartbeats and non-business fault tagging. |
| | `src/mindie_bench/runner/profile_builder.py` | Ablation profile builder (`build_profile`, `filtered_corpus`), constructing isolated `G0`–`G3` harness workspaces and enforcing temporal/oracle holdout filtering. |
| | `src/mindie_bench/runner/mcp_probe.py` | Pre-trial JSON-RPC over stdio MCP handshake probe (`probe_mcp_servers`, `probe_profile_mcp`), failing closed as `infra_failed` before trial start when required MCP servers fail. |
| | `src/mindie_bench/runner/gateway.py` | Local OpenAI/Anthropic HTTP proxy (`MeteringGateway`) that meters per-trial token usage, latency, and late-arriving callbacks within a bounded settlement window. |
| **Evaluation (`evaluation/`)** | `src/mindie_bench/evaluation/oracle_freezer.py` | Pre-trial oracle test sealing (`freeze_oracle`, `restore_oracle_tests`), moving `oracle_tests/` into a host-only git bundle + SHA-256 manifest before agent startup. |
| | `src/mindie_bench/evaluation/source_visibility.py` | Target source visibility auditor (`audit_source_visibility`, `source_evidence_metrics`), verifying whether target source paths exist in the agent's visible workspace or remote container. |
| | `src/mindie_bench/evaluation/coverage.py` | Matrix and metric readiness evaluator (`assess_coverage`), separating cell inventory completeness from metric values. |
| | `src/mindie_bench/evaluation/broker_knowledge.py` | Broker-recorded MCP knowledge call log verifier (`load_broker_knowledge`), aligning out-of-band MCP call intervals with the trial clock. |
| | `src/mindie_bench/evaluation/audit.py` | Contamination, holdout, and receipt integrity audits (`check_holdout`, `verify_oracle_manifest`). |
| **Scorer (`scorer/`)** | `src/mindie_bench/scorer/metrics.py` | Trial (`trajectory_metrics`) and matrix (`summarize`, `retrieval_metrics`, `routing_metrics`) statistical scorer with strict missing-data semantics. |
| | `src/mindie_bench/scorer/command_observations.py` | Static shell/AST recognizer for NPU test commands (`observe_npu_command`, `is_recognized_npu_command`). |
| | `src/mindie_bench/scorer/tool_observations.py` | Remote tool error classifier (`summarize_tool_observations`), separating infrastructure faults from normal test/command failures. |
| | `src/mindie_bench/scorer/statistics.py` | Paired bootstrap CI and exact permutation tests for ablation comparisons. |
| **Integrations (`integrations/`)** | `src/mindie_bench/integrations/component_runtime.py` | Production adapter (`ComponentRuntimeAdapter`) bridging `knowledge`, `remote-dev`, and `coordinator` into `mindie-bench` without shadow fallbacks. |
| | `src/mindie_bench/integrations/first_party.py` | Lockfile and release consistency verifier (`verify_ecosystem_checkouts`) across all 10 MindIE Agent repositories. |
| **Mining (`mining/`)** | `src/mindie_bench/mining/knowledge_to_task.py` | Automated knowledge-to-benchmark-task miner (`mine_knowledge`), turning validated knowledge packages with diff/verification blocks into draft L1/L2/L3 tasks. |

---

## 2. Core Evaluation State Machine & Engineering Invariants

### 2.1 Four-State Terminal Receipt Boundary (`TerminalStatus`)
Every trial produces a `TrialReceipt` with one of four strictly separated terminal states:
1. **`completed_pass` (`TerminalStatus.PASS`)**: Pre-flight probes succeeded, agent completed or stopped at a known execution budget with settled effects, read-only and environment guards held, and all sealed oracle tests passed (`exit 0`).
2. **`completed_fail` (`TerminalStatus.FAIL`)**: Infrastructure was healthy and workspace boundaries remained intact, but the agent's patch failed oracle tests, violated read-only paths, or left unauthorized residual processes.
3. **`infra_failed` (`TerminalStatus.INFRA`)**: Pre-trial MCP probe failure, SSH/container connectivity loss, NPU hardware/driver fault, or clean test timeout where post-execution inventory proves zero residual processes and zero unauthorized writes.
4. **`unknown` (`TerminalStatus.UNKNOWN`)**: Post-execution state is indeterminate (e.g., workspace inventory failed after timeout, oracle restoration failed, or residual processes survived termination).
   - **Scoring Invariant**: `pass@1`, `pass^k`, and valid wall-clock means are computed **strictly over valid terminal outcomes (`completed_pass` + `completed_fail`)**. `infra_failed` and `unknown` are reported in `statuses` and `matrix_coverage`—never silently folded into `completed_fail` or dropped from denominator reporting.
   - **Budget Invariant**: Agent-limit exhaustion and total delivery overruns are separate resource observations. They never overwrite a verified business verdict, erase a measured slowdown, or promote an `infra_failed`/`unknown` outcome. Reports retain actual time/cost and expose budget compliance separately; complete plan coverage does not imply budget compliance.

### 2.2 Timeout Arithmetic Invariants (`outer_timeout >= sum(inner_budgets) + cleanup_grace`)
All layered execution paths enforce strict timeout hierarchy so inner cleanup and inventory always complete before the outer caller gives up:
- **Local Verifier (`runner/verifier.py`)**:
  - `INVENTORY_GRACE_SEC = 10.0`
  - `outer_deadline = monotonic() + timeout_sec + INVENTORY_GRACE_SEC`
  - The inner test subprocess runs with `timeout=timeout_sec`. If it times out, `terminate_tree` (`SIGTERM` -> `2.0s` wait -> `SIGKILL`) and `finally: inventory()` run within the reserved `10.0s` grace budget so a clean test timeout is accurately classified as `infra_failed` (`TestTimeoutError`) rather than falsely escalating to `unknown` (`UnknownExternalEffects`).
- **Remote Slot Execution (`runner/remote_slot.py`)**:
  - Inner command timeout: `timeout` (`1..1200s`, default `240s`).
  - Remote container `WORKER` outer deadline: `outer_timeout = float(request["timeout"]) + 10.0s` (reserving budget for `SIGTERM` -> `2.0s` process-group drain -> `SIGKILL` and stdout/stderr pipe flush).
  - Host SSH transport pipe timeout: `limit + 15.0s` (5.0s above remote `WORKER` `outer_timeout` so the structured JSON payload from `WORKER` is never truncated by a premature local SSH kill).
- **Metering Gateway (`runner/gateway.py`)**:
  - `settlement_grace_sec = 30.0s` after trial completion to drain in-flight provider responses; if requests remain unsettled after grace, `tokens.complete` and `cost_complete` are marked `False` (`cost = None`).

### 2.3 Oracle Anti-Tamper & Ablation Isolation Contracts
- **Oracle Sealing (`evaluation/oracle_freezer.py`)**: Before agent launch, `oracle_tests/` is bundled into a read-only host-side `oracle.bundle` + `manifest.json` (`sha256` per file) and deleted from the agent workspace. During verification, `restore_oracle_tests` restores from the sealed bundle and verifies SHA-256 hashes before running tests.
- **Ablation Groups (`G0..G3`)**:
  - `G0`: Base harness only (no MindIE MCP servers or skills).
  - `G1`: `+ remote-dev` / `npu-top` / `diagnostics` MCP servers.
  - `G2`: `G1 + knowledge` read-only retrieval (filtered by `cutoff_date` and `holdout_ids` so oracle knowledge packages never leak into the corpus).
  - `G3`: `G2 +` post-trial experience capture (`capture_command` required; fails closed if omitted).

### 2.4 Agent-Native Coding & Testing Rules
- Keep modules concise (target `< 800` lines per file so a single `view_file` call captures the entire module) and maintain the Module-Top Agent Invariant Header (`[Role]`, `[Invariants]`, `[Timeouts & Locks]`, `[Side-Effect Policy]`).
- Write comments that explain **business invariants, protocol boundaries, and hardware quirks** (e.g., A3 dual-Die `Phy-ID` mapping or why `later_same_remote_job_terminal` uses confirmation turn for outcome evidence), never excuses for defensive imaginary-enemy checks.
- **No filler tests**: Never write tests that merely check Python types/defensive `ValueError` branches, `chmod`/`symlink` hypotheticals, or regex-grep Markdown documentation.

---

## 3. Standing Ascend NPU Clusters & Verification Commands

### 3.1 Standing Live NPU Environments
- **Ascend A2 (910B) Standing Cluster**:
  - Host: `root@173.125.1.2`
  - Default test container: `mindie-bench-a2-core-20261017`
  - Environment overrides: `MINDIE_BENCH_A2_HOST`, `MINDIE_BENCH_A2_CONTAINER`, `MINDIE_BENCH_A2_DEVICE` (default `0`), `MINDIE_BENCH_A2_TASK_ROOT` (isolated under `/tmp/mindie-bench-live-slot-*`).
- **Ascend A3 (910C Dual-Die) Standing Cluster**:
  - Hosts: `root@192.168.13.165` (`.156` / `.154`)
  - Dual-Die topology: 8 logical NPUs $\times$ 2 Dies (`Phy-ID` `0..15`). Always use isolated sandbox paths (`/tmp/mindie-bench-*`) and non-destructive probes (`npu-smi info`) so active training/inference jobs are never disturbed.

### 3.2 Standard Verification Commands
```bash
# 1. Lint & static check
uv run ruff check .

# 2. Tier 0 + Tier 1 fast unit & local contract suite
uv run pytest -m "not live_npu"

# 3. Tier 2 live Ascend NPU slot integration suite (connects to standing A2 cluster)
uv run pytest -m live_npu

# 4. Full test suite (Tier 0 + Tier 1 + Tier 2 live NPU)
uv run pytest
```
