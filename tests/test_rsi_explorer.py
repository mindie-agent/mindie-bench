from __future__ import annotations

import json
import uuid

import pytest

from mindie_bench.runner.rsi_explorer import (
    evaluate_hidden_input_generalization,
    evaluate_rsi_candidate_gates,
    execute_rsi_compile_and_timing,
    run_rsi_gate_campaign,
    verify_sequential_ablation,
)


class _ContractLease:
    """Test-only lease whose identity is still externally validated."""

    def __init__(self, execution_id, devices):
        self.execution_id = execution_id
        self.devices = list(devices)
        self.released = False

    def receipt(self):
        return {
            "execution_id": self.execution_id,
            "devices": list(self.devices),
            "state": "released" if self.released else "running",
            "authority": "test-contract-authority",
            "released": self.released,
        }

    def verify(self):
        return True


class _ContractAuthority:
    mode = "test_contract_authority"
    evidence_class = "test_contract_authority"

    def __init__(self):
        self.issued = []

    def acquire(self, devices, *, phase, candidate, timeout_seconds):
        execution_id = uuid.uuid4().hex + uuid.uuid4().hex
        self.issued.append(execution_id)
        return _ContractLease(execution_id, devices)

    def validate_result(self, lease, result, *, phase, devices):
        if result.get("run_id") != lease.execution_id:
            raise ValueError("result does not preserve the admitted execution ID")

    def release(self, lease):
        lease.released = True
        return lease.receipt()



def test_execute_rsi_compile_and_timing_runs_cpu_waves_then_serial_npu_timing(tmp_path):
    compile_calls: list[str] = []
    timing_calls: list[tuple[str, int, list[int]]] = []
    active = {"current": 0, "maximum": 0}

    def compile_runner(candidate):
        compile_calls.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "compile-fail":
            return {"compile_ok": False, "failure_reason": "bisheng compile error"}
        if candidate["candidate_id"] == "missing-proof":
            return {}
        return {
            "compile_ok": True,
            "compile_proof": {"status": "passed"},
            "run_id": f"compile-{candidate['candidate_id']}",
            "oracle_passed": True,
        }

    def timing_runner(candidate):
        active["current"] += 1
        active["maximum"] = max(active["maximum"], active["current"])
        timing_calls.append(
            (candidate["candidate_id"], candidate["npu_device"], candidate["assigned_devices"])
        )
        active["current"] -= 1
        return {
            "run_id": candidate["coordinator_lease"].execution_id,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.3 if candidate["candidate_id"] == "winner" else 1.08,
            "ablation_retained": True,
            "workspace_delta_bytes": 2048,
            "guard_workloads": [
                {"name": "uniform-guard", "regression_ratio": 1.01, "max_regression_ratio": 1.05}
            ],
        }

    candidates = [
        {"candidate_id": "pre-proved", "compile_ok": True, "mutation_summary": "ready"},
        {"candidate_id": "winner", "mutation_summary": "fuse epilogue"},
        {"candidate_id": "compile-fail", "mutation_summary": "invalid tile"},
        {"candidate_id": "missing-proof", "mutation_summary": "not compiled"},
    ]
    result = execute_rsi_compile_and_timing(
        candidates,
        compile_runner=compile_runner,
        timing_runner=timing_runner,
        lease_authority=_ContractAuthority(),
        state_dir=tmp_path / "rsi-executor",
        npu_device=3,
        max_compile_slots=2,
        ledger_path=tmp_path / "experiment-ledger.jsonl",
        progress_path=tmp_path / "PROGRESS.md",
        goal="Raise decode throughput",
    )

    # Waves are planned in input order, but a pre-proved candidate is not compiled again.
    assert compile_calls == ["winner", "compile-fail", "missing-proof"]
    assert result["compile_order"] == compile_calls
    assert result["timing_order"] == ["pre-proved", "winner"]
    assert timing_calls == [("pre-proved", 3, [3]), ("winner", 3, [3])]
    assert active["maximum"] == 1
    assert result["plan"]["phase2_timing"]["candidates"] == ["pre-proved", "winner"]
    assert [row["candidate_id"] for row in result["evaluations"]] == [
        "pre-proved", "winner", "compile-fail", "missing-proof"
    ]
    assert [row["verdict"] for row in result["evaluations"]] == [
        "supported", "supported", "inconclusive", "inconclusive"
    ]
    assert [row["gate_reached"] for row in result["evaluations"]] == [4, 4, 1, 1]
    assert result["best_candidate"]["candidate_id"] == "winner"
    assert result["best_candidate"]["speedup_ratio"] == pytest.approx(1.3)

    ledger = [
        json.loads(line)
        for line in (tmp_path / "experiment-ledger.jsonl").read_text().splitlines()
    ]
    assert [row["candidate_id"] for row in ledger] == [
        "pre-proved", "winner", "compile-fail", "missing-proof"
    ]
    progress = (tmp_path / "PROGRESS.md").read_text()
    assert "# RSI Progress: rsi-executor" in progress
    assert "Raise decode throughput" in progress
    assert "winner" in progress
    assert "## Experiments To Complete" in progress
    assert "compile_failed" in progress
    assert "compile-fail" in progress and "missing-proof" in progress


def test_execute_rsi_compile_and_timing_fails_closed_on_workspace_and_guard(tmp_path):
    def timing_runner(candidate):
        lease_id = candidate["coordinator_lease"].execution_id
        if candidate["candidate_id"] == "overflow":
            return {
                "run_id": lease_id, "oracle_passed": True,
                "measurement_passed": True, "speedup_ratio": 1.2,
                "ablation_retained": True,
                "workspace_delta_bytes": {"0": 1024, "1": 8192},
            }
        if candidate["candidate_id"] == "guard-regression":
            return {
                "run_id": lease_id, "oracle_passed": True,
                "measurement_passed": True, "speedup_ratio": 1.2,
                "ablation_retained": True, "workspace_delta_bytes": 1024,
                "guard_workloads": [
                    {"name": "uniform-guard", "regression_ratio": 1.2,
                     "max_regression_ratio": 1.05}
                ],
            }
        return {
            "run_id": lease_id, "oracle_passed": True,
            "measurement_passed": True, "speedup_ratio": 1.2,
            "ablation_retained": True, "workspace_delta_bytes": 1024,
            "guard_workloads": [
                {"name": "uniform-guard", "baseline_ms": 100.0, "candidate_ms": 101.0,
                 "max_regression_ratio": 1.05}
            ],
        }

    result = execute_rsi_compile_and_timing(
        [
            {"candidate_id": "overflow", "compile_ok": True},
            {"candidate_id": "guard-regression", "compile_ok": True},
            {"candidate_id": "clean", "compile_ok": True},
        ],
        timing_runner=timing_runner,
        lease_authority=_ContractAuthority(),
        state_dir=tmp_path / "rsi-guards",
        max_workspace_delta_bytes=4096,
    )
    entries = {row["candidate_id"]: row for row in result["evaluations"]}

    assert entries["overflow"]["verdict"] == "inconclusive"
    assert entries["overflow"]["gate_reached"] == 2
    assert entries["overflow"]["failure_reason"] == "workspace_delta_above_limit"
    assert entries["overflow"]["gates"]["gate2_measurement_guard"] == {
        "passed": False, "reasons": ["workspace_delta_above_limit"]
    }
    assert entries["guard-regression"]["verdict"] == "inconclusive"
    assert entries["guard-regression"]["gate_reached"] == 2
    assert "guard_workload_regressed:uniform-guard" in entries["guard-regression"]["failure_reason"]
    assert entries["clean"]["verdict"] == "supported"
    assert result["best_candidate"]["candidate_id"] == "clean"


def test_hidden_input_generalization_classifies_all_five_categories_cleanly():
    cases = [
        {"case_id": "dynamic-tail", "bucket": "dynamic-shape-257", "shape": [2, 257],
         "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": 100.0, "opt_latency_us": 50.0},
        {"case_id": "high-dynamic-range", "bucket": "hdr-fp16",
         "ref_ok": True, "opt_ok": True, "max_abs_err": 1e-4, "max_rel_err": 1e-4,
         "ref_latency_us": 1000.0, "opt_latency_us": 250.0},
        {"case_id": "nonfinite-latency", "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": float("nan"), "opt_latency_us": 10.0},
        {"case_id": "regressed-tail", "bucket": "tail-17", "ref_ok": True, "opt_ok": False,
         "parity_ok": True, "ref_latency_us": 100.0, "opt_latency_us": 100.0},
        {"case_id": "both-fail", "ref_ok": False, "opt_ok": False, "parity_ok": True},
        {"case_id": "fixed-baseline", "ref_ok": False, "opt_ok": True, "parity_ok": True},
        {"case_id": "timeout", "status": "timeout"},
        {"case_id": "nan-parity", "ref_ok": True, "opt_ok": True,
         "max_abs_err": float("inf")},
    ]
    summary = evaluate_hidden_input_generalization(cases)

    assert summary["schema_version"] == "mindie-bench-l4-generalization/1"
    assert summary["category_counts"] == {
        "both_pass": 3, "opt_regression": 2, "both_fail": 1,
        "opt_improvement": 1, "evaluation_error": 1,
    }
    assert summary["total_cases"] == len(cases)
    assert summary["generalization_ok"] is False
    assert summary["failed_case_ids"] == [
        "regressed-tail", "both-fail", "fixed-baseline", "timeout", "nan-parity",
    ]
    # Only the two finite both_pass ratios contribute: sqrt(2 * 4), never a
    # regression or failing case.
    assert summary["both_pass_speedup"] == pytest.approx(8 ** 0.5)
    classified = {row["case_id"]: row for row in summary["cases"]}
    assert classified["dynamic-tail"]["bucket"] == "dynamic-shape-257"
    assert classified["dynamic-tail"]["shape"] == [2, 257]
    assert classified["dynamic-tail"]["speedup_ratio"] == pytest.approx(2.0)
    assert classified["high-dynamic-range"]["category"] == "both_pass"
    assert classified["nonfinite-latency"]["category"] == "both_pass"
    assert classified["nonfinite-latency"]["speedup_ratio"] is None
    assert classified["nan-parity"]["category"] == "opt_regression"
    assert classified["nan-parity"]["parity_ok"] is False

    clean = evaluate_hidden_input_generalization([
        {"case_id": "pass", "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": 90.0, "opt_latency_us": 45.0},
        {"case_id": "improved", "ref_ok": False, "opt_ok": True, "parity_ok": True},
    ])
    assert clean["generalization_ok"] is True
    assert clean["both_pass_speedup"] == pytest.approx(2.0)
    assert clean["failed_case_ids"] == ["improved"]
    assert evaluate_hidden_input_generalization([
        {"case_id": "both-fail", "ref_ok": False, "opt_ok": False, "parity_ok": True},
    ])["generalization_ok"] is False


def test_sequential_ablation_contract_c4_retains_unverified_candidates():
    kwargs = {
        "baseline_run_id": "baseline",
        "candidate_run_id": "candidate",
        "baseline_latency_us": 1000.0,
        "combined_latency_us": 500.0,
        "factor_trials": [
            {"factor_name": "tiling", "run_id": "loo-tiling",
             "compile_ok": True, "correctness_ok": True, "latency_us": 800.0},
            {"factor_name": "launch", "run_id": "loo-launch",
             "compile_ok": True, "correctness_ok": True, "latency_us": 502.0},
            {"factor_name": "required", "run_id": "loo-required",
             "compile_ok": False, "correctness_ok": True, "latency_us": 500.0},
            {"factor_name": "oracle", "run_id": "loo-oracle",
             "compile_ok": True, "correctness_ok": False, "latency_us": 500.0},
        ],
    }
    retained = verify_sequential_ablation(**kwargs)

    assert retained["schema_version"] == "mindie-bench-ablation/1"
    assert retained["pruned_combination_verified"] is False
    assert retained["ablation_retained"] is True
    assert retained["pruned_factors"] == []
    assert retained["retained_factors"] == ["tiling", "launch", "required", "oracle"]
    decisions = {row["factor_name"]: row for row in retained["factor_decisions"]}
    assert decisions["tiling"]["status"] == "essential"
    assert decisions["launch"]["status"] == "candidate_for_sequential_pruning"
    assert all(row["decision"] == "RETAIN" for row in decisions.values())
    assert decisions["launch"]["contribution_us"] == pytest.approx(2.0)
    assert retained["net_speedup"] == pytest.approx(2.0)

    missing_identity = verify_sequential_ablation(
        **kwargs,
        pruned_combination_run={"run_id": "", "verified": True},
    )
    assert missing_identity["pruned_combination_verified"] is False
    assert missing_identity["ablation_retained"] is True
    assert all(row["decision"] == "RETAIN" for row in missing_identity["factor_decisions"])

    implied = verify_sequential_ablation(
        **kwargs,
        pruned_combination_run={
            "run_id": "pruned-implied", "compile_ok": True,
            "correctness_ok": True, "latency_us": 502.0,
        },
    )
    assert implied["pruned_combination_verified"] is True
    assert implied["verification_run_id"] == "pruned-implied"
    assert implied["pruned_factors"] == ["launch"]
    assert implied["retained_factors"] == ["tiling", "required", "oracle"]

    verified = verify_sequential_ablation(
        **kwargs,
        pruned_combination_run={
            "run_id": "pruned-verified", "verified": True,
            "compile_ok": True, "correctness_ok": True, "latency_us": 505.0,
        },
    )
    assert verified["pruned_combination_verified"] is True
    assert verified["ablation_retained"] is False
    verified_decisions = {row["factor_name"]: row for row in verified["factor_decisions"]}
    assert verified_decisions["launch"]["decision"] == "PRUNE"
    assert verified_decisions["tiling"]["decision"] == "RETAIN"


def test_candidate_gates_consume_generalization_and_ablation_receipts():
    values = {
        "run_id": "remote-evidence", "candidate_id": "evidence",
        "mutation_summary": "fuse epilogue", "conditions": {"soc": "A2"},
        "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
        "speedup_ratio": 1.3, "ablation_retained": True,
    }
    generalized = evaluate_rsi_candidate_gates(
        **values, hidden_input_summary={"generalization_ok": True},
    )
    assert generalized["verdict"] == "supported"
    assert generalized["generalization_ok"] is True

    regressed = evaluate_rsi_candidate_gates(
        **values, hidden_input_summary={"generalization_ok": False},
    )
    assert regressed["verdict"] == "refuted"
    assert regressed["gate_reached"] == 1
    assert regressed["gates"]["gate1_compile_oracle"]["passed"] is False
    assert "hidden_input_generalization_failed" in regressed["gates"]["gate1_compile_oracle"]["reasons"]
    assert regressed["failure_reason"] == "hidden_input_generalization_failed"

    unverified = evaluate_rsi_candidate_gates(
        **values, ablation_receipt={"ablation_retained": False},
    )
    assert unverified["verdict"] == "refuted"
    assert unverified["gate_reached"] == 3
    assert unverified["gates"]["gate3_ablation_gate"]["passed"] is False
    assert "ablation_receipt_not_retained" in unverified["gates"]["gate3_ablation_gate"]["reasons"]
    assert unverified["ablation_receipt"] == {"ablation_retained": False}

    retained = evaluate_rsi_candidate_gates(
        **values, ablation_receipt={"pruned_combination_verified": False},
    )
    assert retained["verdict"] == "supported"
    assert retained["gates"]["gate3_ablation_gate"]["passed"] is True


def test_run_rsi_gate_campaign_closes_loop_and_isolates_accounting_errors(tmp_path):
    stalled = evaluate_rsi_candidate_gates(
        run_id="stalled-1", candidate_id="old-tiling", mutation_summary="retry tile",
        conditions={"soc": "A2"}, compile_ok=True, oracle_passed=True,
        measurement_passed=True, speedup_ratio=1.0, ablation_retained=True,
    )
    ledger_entries = [stalled, {**stalled, "run_id": "stalled-2"}]
    units = [
        {"tier": "L1", "unit_id": "l1", "axioms": ["preserve-dtype"],
         "measurement_rules": ["synchronize-before-timing"]},
        {"tier": "L2", "unit_id": "l2", "observations": ["vector-bound"]},
        {"tier": "L3", "unit_id": "l3-bias", "workflow": "retry-the-stale-tiling"},
        {"tier": "L4", "unit_id": "l4", "operators": ["fuse-producer-consumer"]},
    ]
    hidden_cases = [
        {"case_id": "dynamic-tail", "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": 100.0, "opt_latency_us": 50.0},
    ]
    ablation_spec = {
        "baseline_run_id": "baseline", "candidate_run_id": "candidate",
        "baseline_latency_us": 1000.0, "combined_latency_us": 500.0,
        "factor_trials": [
            {"factor_name": "tiling", "run_id": "loo-tiling",
             "compile_ok": True, "correctness_ok": True, "latency_us": 800.0},
            {"factor_name": "launch", "run_id": "loo-launch",
             "compile_ok": True, "correctness_ok": True, "latency_us": 502.0},
        ],
    }

    class GateStub:
        gate_specs = {"G0": "g0", "G1": "g1", "G2": "g2", "G3": "g3"}

        def __init__(self):
            self.calls = []

        def run_gate(self, gate, spec):
            self.calls.append(gate)
            return {"gate": gate, "state": "completed", "stages": [], "reasons": []}

    gate_runner = GateStub()

    def runner(candidate):
        if candidate.get("phase") == "compile":
            return {"compile_ok": True, "compile_proof": {"status": "passed"},
                    "run_id": "compile-c1", "oracle_passed": True}
        return {"run_id": candidate["coordinator_lease"].execution_id,
                "oracle_passed": True, "measurement_passed": True,
                "speedup_ratio": 1.4, "ablation_retained": True}

    feedback_entries = []

    def broken_distill(entry):
        raise RuntimeError("distill accounting unavailable")

    def collect_feedback(entry):
        feedback_entries.append(entry)
        return {"accepted": True}

    campaign = run_rsi_gate_campaign(
        goal="Improve decode attention",
        units=units,
        ledger_entries=ledger_entries,
        bottleneck_summary={"observations": ["host-side bubble"]},
        conditions={"soc": "A2"},
        consecutive_limit=2,
        command_plan={"candidates": [{"candidate_id": "c1", "mutation_summary": "fuse epilogue"}]},
        runner=runner,
        lease_authority=_ContractAuthority(),
        gate_runner=gate_runner,
        hidden_input_cases=hidden_cases,
        ablation_spec=ablation_spec,
        ledger_path=tmp_path / "experiment-ledger.jsonl",
        progress_path=tmp_path / "PROGRESS.md",
        distill_callback=broken_distill,
        feedback_callback=collect_feedback,
    )

    assert campaign["should_escalate"] is True
    assert campaign["escalation"]["should_escalate"] is True
    assert campaign["escalation"]["consecutive_limit"] == 2
    assert campaign["context"]["stripped_l3_ids"] == ["l3-bias"]
    assert "retry-the-stale-tiling" not in json.dumps(campaign["context"])
    pack = campaign["context"]["subagent_fact_pack"]
    assert pack["schema_version"] == "mindie-rsi-subagent-pack/1"
    assert [row["run_id"] for row in pack["failed_attempts"]] == ["stalled-1", "stalled-2"]
    assert pack["foundation"]["axioms"] == ["preserve-dtype"]

    assert campaign["execution"]["compile_order"] == ["c1"]
    assert campaign["execution"]["timing_order"] == ["c1"]
    assert gate_runner.calls == ["G0", "G1", "G2", "G3"]
    assert campaign["qualification_ok"] is True
    assert all(result["state"] == "completed" for result in campaign["qualification_gates"].values())

    assert campaign["verdict"] == "supported"
    assert campaign["campaign_passed"] is True
    assert campaign["evaluation"]["verdict"] == "supported"
    assert campaign["evaluation"]["execution_evidence"]["production_adapter"] is False
    assert campaign["evaluation"]["execution_evidence"]["physical_acceptance"] is False
    assert campaign["hidden_input_summary"]["generalization_ok"] is True
    assert campaign["ablation_receipt"]["ablation_retained"] is True
    assert campaign["ablation_receipt"]["pruned_combination_verified"] is False

    assert campaign["ledger_appended"] is True
    ledger = [json.loads(line) for line in (tmp_path / "experiment-ledger.jsonl").read_text().splitlines()]
    assert [row["candidate_id"] for row in ledger] == ["c1"]
    assert ledger[0]["verdict"] == "supported"
    assert "c1" in (tmp_path / "PROGRESS.md").read_text()

    # Principle 14: accounting callbacks are isolated from the business verdict
    # and the gate receipts. Feedback still runs after distillation fails.
    assert campaign["accounting_warnings"] == [
        "distill_callback:RuntimeError:distill accounting unavailable"
    ]
    assert campaign["feedback_result"] == {"accepted": True}
    assert feedback_entries == [campaign["evaluation"]]
    assert campaign["verdict"] == "supported"
    assert campaign["gates"]["gate4_distill_or_refute"]["passed"] is True


def test_apple_rsi_and_geak_safeguards_category_gate_jit_isolation_speculative_and_harvest():
    from mindie_bench.runner.rsi_explorer import (
        build_isolated_jit_cache_env,
        classify_candidate_patch_shape,
        evaluate_speculative_acceptance,
        harvest_candidate_patch,
        run_rsi_gate_campaign,
    )

    # 1. Apple category_gate.py: pure_flag_flip, config_default_flip, artifact_only, source_edit
    flag_diff = "diff --git a/launch_server.sh b/launch_server.sh\n+++ b/launch_server.sh\n+--chunked-prefill-size 8192\n"
    flag_res = classify_candidate_patch_shape(flag_diff, reject_narrow_flips=True)
    assert flag_res["category"] == "pure_flag_flip"
    assert flag_res["rejected_by_narrow_gate"] is True

    cfg_diff = "diff --git a/model_config.py b/model_config.py\n+++ b/model_config.py\n+MAX_NUM_SEQS = 256\n"
    cfg_res = classify_candidate_patch_shape(cfg_diff, reject_narrow_flips=True)
    assert cfg_res["category"] == "config_default_flip"
    assert cfg_res["rejected_by_narrow_gate"] is True

    src_diff = (
        "diff --git a/kernels/rmsnorm_triton.py b/kernels/rmsnorm_triton.py\n"
        "+++ b/kernels/rmsnorm_triton.py\n"
        "+def fused_rmsnorm(x, w):\n"
        "+    return tl.rsqrt(x)\n"
    )
    src_res = classify_candidate_patch_shape(src_diff, reject_narrow_flips=True)
    assert src_res["category"] == "source_edit"
    assert src_res["rejected_by_narrow_gate"] is False

    # 2. Apple _paired_helper.py: isolated JIT cache env per side
    prist_env = build_isolated_jit_cache_env("/tmp/rsi-state", "pristine")
    cand_env = build_isolated_jit_cache_env("/tmp/rsi-state", "candidate-1")
    assert prist_env["TRITON_CACHE_DIR"] != cand_env["TRITON_CACHE_DIR"]
    assert prist_env["ASCEND_CACHE_PATH"] != cand_env["ASCEND_CACHE_PATH"]

    # 3. Apple acceptance.py: speculative decoding draft count & acceptance length guard
    no_drafts = evaluate_speculative_acceptance(
        {"num_drafts": 0, "mean_acceptance_length": 0.0},
        {"num_drafts": 64, "mean_acceptance_length": 3.2},
    )
    assert no_drafts["acceptance_ok"] is False and no_drafts["reason"] == "no_drafts"

    degraded = evaluate_speculative_acceptance(
        {"num_drafts": 64, "mean_acceptance_length": 2.4},
        {"num_drafts": 64, "mean_acceptance_length": 3.2},
        rtol=0.05,
    )
    assert degraded["acceptance_ok"] is False and degraded["reason"] == "acceptance_degraded"

    healthy = evaluate_speculative_acceptance(
        {"num_drafts": 64, "mean_acceptance_length": 3.15},
        {"num_drafts": 64, "mean_acceptance_length": 3.2},
        rtol=0.05,
    )
    assert healthy["acceptance_ok"] is True

    # 4. GEAK kernel_lane.js: measurement-anchored patch harvesting rescues timed-out disk patches
    rescued = harvest_candidate_patch({"timed_out": True}, patch_text=src_diff, baseline_floor=1.0)
    assert rescued["should_verify"] is True and rescued["harvest_mode"] == "disk_patch_rescued"

    skipped = harvest_candidate_patch(
        {"measurement_valid": True, "speedup_ratio": 0.94}, patch_text=src_diff, baseline_floor=1.0
    )
    assert skipped["should_verify"] is False and skipped["harvest_mode"] == "skip_verified_below_floor"

    # 5. Campaign integration: narrow flag flip rejected at G3; incumbent_speedup prevents false winner
    rejected_campaign = run_rsi_gate_campaign(
        candidate={
            "candidate_id": "flag-gamer",
            "compile_ok": True,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.35,
            "diff_text": flag_diff,
        },
        reject_narrow_flips=True,
    )
    assert rejected_campaign["verdict"] == "refuted"
    assert rejected_campaign["patch_classification"]["category"] == "pure_flag_flip"
    assert "narrow_patch_rejected:pure_flag_flip" in rejected_campaign["evaluation"]["failure_reason"]

    below_incumbent = run_rsi_gate_campaign(
        candidate={
            "candidate_id": "c-sub-incumbent",
            "compile_ok": True,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.5,
            "ablation_retained": True,
            "diff_text": src_diff,
        },
        incumbent_speedup=2.5,
        reject_narrow_flips=True,
    )
    assert below_incumbent["verdict"] == "supported"
    assert below_incumbent["incumbent_comparison"]["improved_over_incumbent"] is False
    assert below_incumbent["incumbent_comparison"]["became_winner"] is False
    assert below_incumbent["reprofile_required"] is False
