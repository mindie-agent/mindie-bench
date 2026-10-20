"""Tier 0 unit tests for `cann-bench` Roofline HAP scoring and `cannbot-sentry` trajectory auditing.

[Role]
Verify Hardware-Anchored Performance (`compute_hap_score`), composite operator
scoring (`compute_operator_composite_score`) with L1-L5 complexity taxonomy,
Sentry dual-verification trajectory auditing (`audit_sentry_trajectory`), and
RSI gate integration (`evaluate_rsi_candidate_gates`, `RSICandidateResult`).

[Invariants]
- Pure Tier 0 in-memory unit tests: zero subprocesses, zero network I/O, zero sleeps.
- Every test exercises real mathematical formulas, boundary guards, or state
  reconciliation invariants from `cann-bench` and `cannbot-sentry`.

[Timeouts & Locks]
- Whole file executes in `< 1s` with no locks or external resources.

[Side-Effect Policy]
- No filesystem, environment, or process mutations.
"""

from __future__ import annotations

import math

import pytest

from mindie_bench.runner import (
    CANONICAL_SWIMLANES,
    OPERATOR_COMPLEXITY_TAXONOMY,
    RSICandidateResult,
    audit_sentry_trajectory,
    classify_operator_complexity,
    compare_swimlane_attributions,
    compute_hap_score,
    compute_operator_composite_score,
    evaluate_rsi_candidate_gates,
)
from mindie_bench.runner.rsi_explorer import run_rsi_gate_campaign


def test_compute_hap_score_anchoring_properties():
    # T_cand == T_baseline -> HAP = 0.5
    assert compute_hap_score(100.0, 100.0, 40.0) == pytest.approx(0.5)
    # T_cand == T_HW -> HAP = 1.0
    assert compute_hap_score(40.0, 100.0, 40.0) == pytest.approx(1.0)
    # T_cand > T_baseline (slower than baseline) -> HAP in (0, 0.5)
    slower_hap = compute_hap_score(160.0, 100.0, 40.0)
    assert slower_hap is not None
    assert 0.0 < slower_hap < 0.5
    assert slower_hap == pytest.approx((100.0 - 40.0) / ((160.0 - 40.0) + (100.0 - 40.0)))
    # T_cand < T_baseline (faster than baseline) -> HAP > 0.5
    faster_hap = compute_hap_score(70.0, 100.0, 40.0)
    assert faster_hap is not None
    assert 0.5 < faster_hap < 1.0
    assert faster_hap == pytest.approx((100.0 - 40.0) / ((70.0 - 40.0) + (100.0 - 40.0)))
    # Super-Roofline candidate (0 < T_cand < T_HW with positive denominator) -> HAP > 1.0
    super_hap = compute_hap_score(30.0, 100.0, 40.0)
    assert super_hap is not None
    assert super_hap > 1.0
    assert super_hap == pytest.approx(60.0 / (-10.0 + 60.0))


@pytest.mark.parametrize(
    ("t_cand_us", "t_baseline_us", "t_hw_us"),
    [
        (0.0, 100.0, 40.0),
        (-5.0, 100.0, 40.0),
        (80.0, 0.0, 40.0),
        (80.0, -100.0, 40.0),
        (80.0, 100.0, 0.0),
        (80.0, 100.0, -40.0),
        (None, 100.0, 40.0),
        (80.0, None, 40.0),
        (80.0, 100.0, None),
        (True, 100.0, 40.0),
        (float("nan"), 100.0, 40.0),
        (80.0, float("inf"), 40.0),
        (80.0, 100.0, float("-inf")),
        # Numerator <= 0 (baseline at or below hardware floor)
        (80.0, 40.0, 40.0),
        (80.0, 35.0, 40.0),
        # Denominator <= 0: (T_cand - T_HW) + (T_baseline - T_HW) <= 0
        (10.0, 50.0, 40.0),
        (20.0, 60.0, 40.0),
    ],
)
def test_compute_hap_score_boundary_guards_return_none(t_cand_us, t_baseline_us, t_hw_us):
    assert compute_hap_score(t_cand_us, t_baseline_us, t_hw_us) is None


def test_compute_operator_composite_score_eq4_and_complexity_taxonomy():
    # Compile failure zeroes out all score components (delta_pass = 0)
    failed_compile = compute_operator_composite_score(
        False,
        [{"accuracy_passed": True, "t_cand_us": 50.0, "t_baseline_us": 100.0, "t_hw_us": 40.0}],
        operator="FlashAttention",
    )
    assert failed_compile["each_operator_score"] == 0.0
    assert failed_compile["compilation_score"] == 0.0
    assert failed_compile["accuracy_score"] == 0.0
    assert failed_compile["performance_score"] == 0.0
    assert failed_compile["mean_hap"] is None
    assert failed_compile["complexity_level"] == "L4"

    # Mixed 4-case operator evaluation:
    # Case 0: T_cand == T_HW -> HAP = 1.0
    # Case 1: T_cand == T_baseline -> HAP = 0.5
    # Case 2: accuracy failed -> excluded from HAP sum
    # Case 3: accuracy passed but missing T_HW -> HAP is None (contributes 0 to perf sum)
    cases = [
        {"accuracy_passed": True, "t_cand_us": 40.0, "t_baseline_us": 100.0, "t_hw_us": 40.0, "operator": "SwiGLU"},
        {"accuracy_passed": True, "t_cand_us": 100.0, "t_baseline_us": 100.0, "t_hw_us": 40.0, "operator": "Softmax"},
        {"accuracy_passed": False, "t_cand_us": 40.0, "t_baseline_us": 100.0, "t_hw_us": 40.0, "operator": "GroupedMatmul"},
        {"accuracy_passed": True, "t_cand_us": 60.0, "t_baseline_us": 100.0, "t_hw_us": None, "operator": "KimiDeltaAttention"},
    ]
    composite = compute_operator_composite_score(True, cases, operator="FlashAttention")
    # Compile = 20.0, Accuracy = (3/4)*30 = 22.5, Perf = ((1.0 + 0.5 + 0 + 0)/4)*50 = 18.75
    assert composite["compilation_score"] == pytest.approx(20.0)
    assert composite["accuracy_score"] == pytest.approx(22.5)
    assert composite["performance_score"] == pytest.approx(18.75)
    assert composite["each_operator_score"] == pytest.approx(61.25)
    assert composite["accuracy_pass_rate"] == pytest.approx(0.75)
    assert composite["mean_hap"] == pytest.approx(0.75)
    assert composite["missing_hap_cases"] == 1
    assert set(composite["complexity_summary"]) == {"L1", "L2", "L3", "L4", "L5"}
    assert composite["complexity_summary"]["L1"]["mean_hap"] == pytest.approx(1.0)
    assert composite["complexity_summary"]["L2"]["mean_hap"] == pytest.approx(0.5)
    assert composite["complexity_summary"]["L3"]["passed_cases"] == 0
    assert composite["complexity_summary"]["L5"]["passed_cases"] == 1

    # Super-Roofline operator score can exceed 100.0
    super_score = compute_operator_composite_score(
        True,
        [{"accuracy_passed": True, "t_cand_us": 30.0, "t_baseline_us": 100.0, "t_hw_us": 40.0}],
        operator="KimiDeltaAttention",
    )
    assert super_score["each_operator_score"] > 100.0
    assert super_score["complexity_level"] == "L5"

    # Anti-cheat zeroing (CPU fallback or missing NPU kernel)
    anti_cheat = compute_operator_composite_score(
        True,
        [{"accuracy_passed": True, "hap_score": 0.9, "cpu_fallback_detected": True}],
    )
    assert anti_cheat["zeroed_by_anti_cheat"] is True
    assert anti_cheat["each_operator_score"] == 0.0


def test_classify_operator_complexity_covers_all_five_levels():
    assert set(OPERATOR_COMPLEXITY_TAXONOMY) == {"L1", "L2", "L3", "L4", "L5"}
    assert classify_operator_complexity("RMSNorm")["complexity_level"] == "L1"
    assert classify_operator_complexity("L2/Softmax")["complexity_level"] == "L2"
    assert classify_operator_complexity("GroupedMatmul")["complexity_level"] == "L3"
    assert classify_operator_complexity("MLAProlog")["complexity_level"] == "L4"
    assert classify_operator_complexity("SparseFlashAttention")["complexity_level"] == "L5"


def test_audit_sentry_trajectory_compliant_and_swimlane_ab_comparison():
    without_knowledge_events = [
        {"step_index": 0, "event_id": "wk0", "lane": "explore", "tool": "read_file", "tokens": 1200, "duration_ms": 500},
        {"step_index": 1, "event_id": "wk1", "lane": "code", "tool": "edit_file", "tokens": 1800, "duration_ms": 900},
        {"step_index": 2, "event_id": "wk2", "lane": "compile", "tool": "compile_check", "tokens": 1000, "duration_ms": 1200},
    ]
    events = [
        {"step_index": 1, "event_id": "e-know", "stage": "knowledge_query", "tokens": 300, "duration_ms": 120},
        {"step_index": 2, "event_id": "e-plan", "lane": "design", "tool": "tiling_plan", "tokens": 250, "duration_ms": 100},
        {"step_index": 3, "event_id": "e-edit", "stage": "code_mutation", "tool": "edit_file", "tokens": 600, "duration_ms": 300},
        {"step_index": 4, "event_id": "e-comp", "stage": "compile_check", "succeeded": True, "tokens": 400, "duration_ms": 800},
        {"step_index": 5, "event_id": "e-time", "stage": "npu_timing", "speedup_ratio": 1.4, "tokens": 350, "duration_ms": 600},
        {"step_index": 6, "event_id": "e-gen", "stage": "hidden_input_generalization", "succeeded": True, "tokens": 200, "duration_ms": 250},
        {"step_index": 7, "event_id": "e-abl", "stage": "sequential_ablation", "succeeded": True, "tokens": 220, "duration_ms": 280},
        {"step_index": 8, "event_id": "e-dist", "lane": "distill", "tool": "append_ledger", "tokens": 180, "duration_ms": 90,
         "reported_claims": {"reported_speedup": 1.4, "reported_verdict": "supported", "reported_accuracy_passed": True}},
    ]
    contract = {
        "require_knowledge_before_mutation": True,
        "require_compile_before_timing": True,
        "require_generalization_gate": True,
        "require_ablation_gate": True,
        "factor_count": 2,
        "without_knowledge_events": without_knowledge_events,
    }
    receipt = {
        "verdict": "supported",
        "compile_ok": True,
        "oracle_passed": True,
        "generalization_ok": True,
        "measurement_passed": True,
        "ablation_retained": True,
        "speedup_ratio": 1.4,
        "t_cand_us": 70.0,
        "t_baseline_us": 98.0,
        "t_hw_us": 42.0,
    }
    audit = audit_sentry_trajectory(events, contract, receipt)
    assert audit["audit_passed"] is True
    assert audit["dual_verification_passed"] is True
    assert audit["calibration_verdict"] == "TRUSTED"

    proc = audit["process_compliance"]
    assert proc["compliant"] is True
    assert proc["status"] == "OK"
    assert proc["compliance_rate"] == pytest.approx(1.0)
    assert proc["checks"]["knowledge_before_mutation"]["evidence_backlinks"] == [
        {"step_index": 1, "event_id": "e-know"},
        {"step_index": 3, "event_id": "e-edit"},
    ]
    assert proc["checks"]["compile_before_timing"]["evidence_backlinks"] == [
        {"step_index": 4, "event_id": "e-comp"},
        {"step_index": 5, "event_id": "e-time"},
    ]

    rc = audit["result_consistency"]
    assert rc["consistent"] is True
    assert rc["reward_hacking_detected"] is False
    assert rc["verified_summary"]["hap_score"] == pytest.approx(compute_hap_score(70.0, 98.0, 42.0))

    swim = audit["swimlane_attribution"]
    assert tuple(swim["lane_order"]) == CANONICAL_SWIMLANES
    assert set(swim["active_lanes"]) == set(CANONICAL_SWIMLANES)
    assert swim["total_tokens"] == 2500
    ab = swim["ab_comparison"]
    assert ab["total_tokens_delta"] == 2500 - 4000
    assert ab["token_reduction_ratio"] == pytest.approx((4000 - 2500) / 4000)


def test_audit_sentry_trajectory_catches_process_and_reward_hacking_violations():
    # Mutation before knowledge_query, timing after uncompiled mutation, missing ablation,
    # and inflated speedup claim vs actual verification receipt.
    events = [
        {"step_index": 0, "event_id": "ev-mut-1", "stage": "code_mutation", "tokens": 500},
        {"step_index": 1, "event_id": "ev-know-late", "stage": "knowledge_query", "tokens": 200},
        {"step_index": 2, "event_id": "ev-comp-1", "stage": "compile_check", "succeeded": True, "tokens": 300},
        {"step_index": 3, "event_id": "ev-mut-2", "stage": "code_mutation", "tokens": 400},
        {"step_index": 4, "event_id": "ev-time-bad", "stage": "npu_timing", "tokens": 250,
         "reported_speedup": 2.40, "reported_verdict": "supported", "reported_accuracy_passed": True},
    ]
    contract = {
        "require_knowledge_before_mutation": True,
        "require_compile_before_timing": True,
        "require_generalization_gate": True,
        "factor_count": 3,
    }
    receipt = {
        "verdict": "refuted",
        "compile_ok": True,
        "oracle_passed": False,
        "generalization_ok": False,
        "measurement_passed": True,
        "ablation_retained": False,
        "speedup_ratio": 1.02,
    }
    audit = audit_sentry_trajectory(events, contract, receipt)
    assert audit["audit_passed"] is False
    assert audit["dual_verification_passed"] is False
    assert audit["calibration_verdict"] == "HIGH_SCORE_SUSPECT"

    proc = audit["process_compliance"]
    assert proc["compliant"] is False
    assert "knowledge_query_after_code_mutation" in proc["violation_reasons"]
    assert "uncompiled_code_mutation_before_npu_timing" in proc["violation_reasons"]
    assert "missing_hidden_input_generalization_check" in proc["violation_reasons"]
    assert "missing_multi_factor_ablation_check" in proc["violation_reasons"]
    assert proc["checks"]["knowledge_before_mutation"]["step_indices"] == [0, 1]
    assert proc["checks"]["compile_before_timing"]["event_ids"] == ["ev-comp-1", "ev-mut-2", "ev-time-bad"]

    rc = audit["result_consistency"]
    assert rc["consistent"] is False
    assert rc["reward_hacking_detected"] is True
    assert set(rc["discrepancy_codes"]) >= {
        "speedup_mismatch",
        "unverified_accuracy_claim",
        "verdict_mismatch",
        "unverified_ablation_claim",
    }


def test_evaluate_rsi_candidate_gates_and_candidate_result_integrate_hap():
    entry = evaluate_rsi_candidate_gates(
        run_id="run-hap-1",
        candidate_id="cand-uf4",
        mutation_summary="vectorize UB load/store",
        conditions={"shape": "4096x4096"},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=True,
        t_cand_us=70.0,
        t_baseline_us=100.0,
        roofline_floor_us=40.0,
        ablation_retained=True,
    )
    expected_hap = (100.0 - 40.0) / ((70.0 - 40.0) + (100.0 - 40.0))
    assert entry["verdict"] == "supported"
    assert entry["speedup_ratio"] == pytest.approx(100.0 / 70.0)
    assert entry["hap_score"] == pytest.approx(expected_hap)
    assert entry["t_hw_us"] == pytest.approx(40.0)
    assert entry["roofline_floor_us"] == pytest.approx(40.0)

    cand_obj = RSICandidateResult.from_gate_entry(entry)
    assert cand_obj.hap_score == pytest.approx(expected_hap)
    assert cand_obj["hap_score"] == pytest.approx(expected_hap)
    assert math.isclose(cand_obj.speedup_ratio or 0.0, 100.0 / 70.0)

    # Failed measurement suppresses hap_score just like speedup_ratio
    invalid_entry = evaluate_rsi_candidate_gates(
        run_id="run-hap-bad",
        candidate_id="cand-bad",
        mutation_summary="bad timing",
        conditions={"t_hw_us": 40.0, "t_baseline_us": 100.0, "t_cand_us": 70.0},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=False,
        speedup_ratio=1.4,
    )
    assert invalid_entry["speedup_ratio"] is None
    assert invalid_entry["hap_score"] is None

    # Campaign forwards latency/roofline anchors into receipt
    campaign = run_rsi_gate_campaign(
        campaign_id="camp-hap",
        candidate={
            "run_id": "run-camp-1",
            "candidate_id": "cand-camp",
            "compile_ok": True,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.25,
            "t_baseline_us": 100.0,
            "roofline_floor_us": 50.0,
            "ablation_retained": True,
        },
    )
    assert campaign["receipt"]["t_cand_us"] == pytest.approx(80.0)
    assert campaign["receipt"]["hap_score"] == pytest.approx(50.0 / (30.0 + 50.0))
    assert compare_swimlane_attributions(
        {"total_tokens": 1000, "lanes": {"explore": {"tokens": 400}}},
        {"total_tokens": 2000, "lanes": {"explore": {"tokens": 1100}}},
    )["token_reduction_ratio"] == pytest.approx(0.5)
