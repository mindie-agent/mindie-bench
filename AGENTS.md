# MindIE Bench (`mindie-bench`)

Closed-loop evaluation, ablation (`G0..G3`), sealed-oracle NPU verification, and release qualification controller for the MindIE Agent ecosystem (`bench` governance profile).

## 1. Domain Boundary & Core Operating Rules

- **Sealed Oracle & Workspace Isolation (`BENCH-ORACLE`)**: Before agent admission, `freeze_oracle` moves protected oracle files out of the candidate workspace into a host-only SHA-256-bound bundle. Remote verifiers stage read-only private bundles with a per-trial 256-bit nonce and verify pre/post SHA-256 inventories before reporting rewards.
- **Verdict & Denominator Honesty (`BENCH-VERDICT`)**:
  - Every trial emits a four-state `TerminalStatus` (`completed_pass`, `completed_fail`, `infra_failed`, `unknown`). `pass@1`, `pass^k`, and valid wall-clock means include only valid outcomes (`completed_pass` + `completed_fail`), while `infra_failed`, `unknown`, and blocked cells remain explicit in every denominator.
  - Once the external verifier establishes a `completed_pass` or `completed_fail` business verdict, post-trial auxiliary exceptions (native service close, trajectory/component metric extraction, remote/local workspace teardown) **never** overwrite `terminal_status` to `infra_failed`; they record `cleanup_warning` and/or `metrics_incomplete=True` in `auxiliary_metrics` while preserving the verified business verdict.
- **Boundary-Only Provenance Hashing (`BENCH-PROVENANCE`)**: External trust boundaries (candidate & runner Git SHAs, downloaded release artifacts, `SHA256SUMS`, task digests, and sealed oracle manifests) are verified explicitly, while unchanged runner package digests and sandboxed runtime import probes are computed once per run boundary rather than re-walking `*.py` files on every trial.
- **Zero Production Reverse-Contamination (`BENCH-ISOLATION`)**: Evaluation rules, synthetic fixtures, and holdout metadata live exclusively inside `mindie-bench` and never leak into production host plugins or shared runtime components.
- **Timeout Arithmetic**:
  - Local verifier: `outer_deadline = monotonic() + timeout_sec + 10.0s` (`INVENTORY_GRACE_SEC`).
  - Remote private verifier: `outer_timeout = float(timeout_sec) + 20.0s`, giving inner probes and `verifier/test.sh` the full `timeout_sec` while reserving `20s` for staging, pre/post SHA-256 inventories, and `EXIT` trap cleanup (`RemoteSlot.verify` pipe wait `= outer_timeout + 25.0s`).

## 2. Module Map & Living Documentation

- `src/mindie_bench/runner/`: Trial & matrix orchestration (`orchestrator.py`, `matrix_runner.py`, `trial_preflight.py`, `trial_finalize.py`), L1/L2/L3 verifiers (`verifier.py`, `private_remote_judge.py`), `RemoteSlot` / `WorktreeSlot`, `G0..G3` `QualificationGateRunner`, and `evaluate_rsi_candidate_gates` (`__init__.py`).
- `src/mindie_bench/evaluation/`: Oracle freezing (`oracle_freezer.py`), contamination/holdout auditing (`audit.py`), strategy selection & mechanism replay (`strategy_selection.py`, `mechanism_replay.py`).
- `src/mindie_bench/scorer/` & `release*.py`: Trajectory/matrix statistical scoring, v1/v2 release benchmark contracts, and `.github/scripts/*` publication & old-client upgrade validators (including `mindie-agent/harness` birth-release qualification).
- **Living Specifications (`docs/`)**: [`docs/design.md`](docs/design.md), [`docs/usage.md`](docs/usage.md), [`docs/acceptance.md`](docs/acceptance.md), [`docs/release-benchmarks.md`](docs/release-benchmarks.md), [`docs/rsi-inference.md`](docs/rsi-inference.md). Historical pre-G / wave review notes reside under [`docs/archive/`](docs/archive/).

## 3. Verification & Pre-PR Gate (`bench`)

```bash
uv run ruff check .
uv run pytest -m "not live_npu"
uv run mindie-bench demo --output /tmp/mindie-bench-demo
python3 .github/scripts/check_lf_eol.py
```

For standing live Ascend A2 (`root@173.125.1.2`) and A3 (`root@192.168.13.165` / `.154`) NPU verification:

```bash
uv run pytest -m live_npu
```

Complete `.github/pull_request_template.md` (`BENCH-ORACLE`, `BENCH-VERDICT`, `BENCH-PROVENANCE`, `BENCH-ISOLATION`) and verify via `python3 .github/scripts/check_pr_checklist.py --profile bench`.
