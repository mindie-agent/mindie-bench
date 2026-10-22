"""Section 4.7 & Section 16 conditional strategy selection pilot tests.

Exercises real domain corpora (`knowledge-vllm-ascend` + `knowledge-operators`),
strict B vs C context isolation, shared-budget accounting, primary-verdict
preservation under selector failure, and 36-cell A/B/C pilot comparisons.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from mindie_bench.cli import main as cli_main
from mindie_bench.evaluation.knowledge_tiered_support import (
    build_strategy_navigation_cards as _strategy_cards,
)
from mindie_bench.evaluation.mechanism_replay import run_host_mechanism_replay
from mindie_bench.evaluation.strategy_selection import (
    CANDIDATE_UNIT_IDS,
    DEV_CASES,
    PILOT_CASES,
    StrategySelectionBudget,
    evaluate_strategy_continuation,
    load_combined_routing_index,
    plan_strategy_selection_cells,
    prepare_strategy_selector_view,
    run_strategy_selection_cell,
    run_strategy_selection_pilot,
    verify_pilot_lineage_and_soundness,
    verify_selector_isolation,
)

if _strategy_cards is None:
    pytest.skip(
        "installed mindie_knowledge predates build_strategy_navigation_cards",
        allow_module_level=True,
    )


def _corpus_roots() -> list[Path]:
    worktree_root = Path(__file__).resolve().parents[2]
    vllm_root = worktree_root / "knowledge-vllm-ascend"
    ops_root = worktree_root / "knowledge-operators"
    if not vllm_root.is_dir():
        vllm_root = Path("/Volumes/data/code/knowledge-vllm-ascend")
    if not ops_root.is_dir():
        ops_root = Path("/Volumes/data/code/knowledge-operators")
    return [vllm_root, ops_root]


@pytest.fixture(scope="module")
def routing_index():
    return load_combined_routing_index(_corpus_roots())


@pytest.mark.tier0
def test_arm_b_and_arm_c_selector_views_enforce_strict_input_and_body_isolation(routing_index):
    for case in PILOT_CASES.values():
        view_b = prepare_strategy_selector_view(case, arm="B", routing_index=routing_index)
        view_c = prepare_strategy_selector_view(case, arm="C", routing_index=routing_index)

        assert view_b["tool_access"] == "none"
        assert view_c["tool_access"] == "none"
        assert "pre_decision_observations" not in view_b
        assert "observed_conditions" not in view_b
        assert len(view_c["pre_decision_observations"]) == len(case.pre_decision_observations)
        assert [c["id"] for c in view_b["candidate_navigation_cards"]] == [
            c["id"] for c in view_c["candidate_navigation_cards"]
        ]
        for card in view_b["candidate_navigation_cards"] + view_c["candidate_navigation_cards"]:
            assert "body" not in card and "content" not in card and "excerpt" not in card
            assert card["closure_token_estimate"] >= card["body_token_estimate"] > 0

        # Injecting an execution observation into Arm B or a leak marker into Arm C must fail.
        polluted_b = {**view_b, "leaked_note": case.pre_decision_observations[0].statement}
        iso_b = verify_selector_isolation(polluted_b, case, arm="B")
        assert iso_b["isolated"] is False

        polluted_c = {**view_c, "leaked_explanation": case.private_judge["leak_markers"][0]}
        iso_c = verify_selector_isolation(polluted_c, case, arm="C")
        assert iso_c["isolated"] is False


@pytest.mark.tier0
def test_dev_and_pilot_lineage_separation_and_private_judge_soundness():
    readiness = verify_pilot_lineage_and_soundness()
    assert readiness["ready"] is True, readiness
    assert readiness["independent_pilot_families"] == 3
    assert readiness["overlapping_families"] == []
    assert readiness["soundness_failures"] == []

    # Overlapping a Dev family into Pilot or providing < 3 independent families marks `not_ready`.
    leaked_case = dataclasses.replace(
        PILOT_CASES["S1-F1-trigger"],
        family_id=DEV_CASES["DEV-ACLGRAPH-01"].family_id,
    )
    not_ready = verify_pilot_lineage_and_soundness(
        [leaked_case, *list(PILOT_CASES.values())[1:]]
    )
    assert not_ready["ready"] is False
    assert not_ready["status"] == "not_ready"

    # Blind judge ignores arm labels (`A` / `B` / `C`).
    case = PILOT_CASES["S1-F1-trigger"]
    res_a = evaluate_strategy_continuation(case, {**case.reference_solution, "arm": "A"})
    res_c = evaluate_strategy_continuation(case, {**case.reference_solution, "arm": "C"})
    assert res_a == res_c


@pytest.mark.tier0
def test_shared_budget_timeout_arithmetic_and_selector_failure_preserves_primary_pass(routing_index):
    with pytest.raises(ValueError, match="outer_timeout_sec"):
        StrategySelectionBudget(
            business_window_sec=600.0,
            verifier_timeout_sec=120.0,
            cleanup_grace_sec=5.0,
            outer_timeout_sec=720.0,
        )

    case = PILOT_CASES["S1-F1-boundary"]
    cells, _ = plan_strategy_selection_cells([case], repetitions=1)
    cell_c = next(c for c in cells if c.arm == "C")
    budget = StrategySelectionBudget()
    seen_remaining_calls = []

    def timeout_selector(_view):
        raise TimeoutError("selector gateway timed out after 30s")

    def succeeding_default_executor(
        *,
        case,
        arm,
        selector_decision,
        routing_index,
        remaining_model_calls,
        remaining_window_sec,
    ):
        seen_remaining_calls.append((remaining_model_calls, remaining_window_sec, selector_decision))
        return {
            **case.reference_solution,
            "model_calls": 3,
            "prompt_tokens": None,
            "completion_tokens": None,
            "wall_clock_sec": 25.0,
            "npu_runs": 1,
        }

    outcome = run_strategy_selection_cell(
        cell_c,
        case,
        routing_index=routing_index,
        budget=budget,
        selector_fn=timeout_selector,
        executor_fn=succeeding_default_executor,
    )
    assert outcome["selector_status"] == "unknown"
    assert outcome["terminal_status"] == "pass"
    assert outcome["selector_model_calls"] == 1
    assert outcome["total_model_calls"] == 4
    assert seen_remaining_calls == [(11, 570.0, None)]
    assert outcome["usage_status"] == "unavailable"
    assert outcome["prompt_tokens"] is None
    assert outcome["completion_tokens"] is None

    # Verify static mechanism replay also preserves None usage rather than defaulting to 0.
    static_replay = run_host_mechanism_replay(
        "M1-01",
        decision_payload={"action": "phase0_fusion_triage", "hypothesis_id": "H1_fusion_boundary"},
    )
    assert static_replay["execution_metrics"]["usage_status"] == "unavailable"
    assert static_replay["execution_metrics"]["prompt_tokens"] is None
    assert static_replay["execution_metrics"]["completion_tokens"] is None


@pytest.mark.tier1
def test_full_36_cell_abc_pilot_and_cli_report_evidence_gain_and_cost_ledgers(tmp_path):
    roots = _corpus_roots()
    dev_report = run_strategy_selection_pilot(
        corpus_roots=roots,
        cases=tuple(DEV_CASES.values()),
        repetitions=1,
    )
    assert dev_report["arms"]["C"]["pass_rate"]["value"] == 1.0

    pilot_out = tmp_path / "strategy_selection_pilot.json"
    rc = cli_main(
        [
            "strategy-selection-pilot",
            "--corpus",
            str(roots[0]),
            "--corpus",
            str(roots[1]),
            "--repetitions",
            "2",
            "--output",
            str(pilot_out),
        ]
    )
    assert rc == 0
    report = json.loads(pilot_out.read_text(encoding="utf-8"))
    assert report["readiness"]["ready"] is True
    assert report["upper_bounds"]["total_cells"] == 36
    assert report["upper_bounds"]["selector_calls"] == 24
    assert report["upper_bounds"]["max_model_calls"] == 432
    assert report["upper_bounds"]["max_npu_runs"] == 72
    assert report["upper_bounds"]["max_outer_wall_clock_sec"] == 26100.0

    # Verify balanced triplet order reversal between repetition 1 and repetition 2.
    f1_rep1 = [
        c["cell"]["arm"]
        for c in report["cells"]
        if c["cell"]["case_id"] == "S1-F1-trigger" and c["cell"]["repetition"] == 1
    ]
    f1_rep2 = [
        c["cell"]["arm"]
        for c in report["cells"]
        if c["cell"]["case_id"] == "S1-F1-trigger" and c["cell"]["repetition"] == 2
    ]
    assert f1_rep1 == list(reversed(f1_rep2))

    # C (task + execution evidence) outperforms B (task-only) on selection reasonableness,
    # avoids false adoptions on boundary cases, and outperforms A (default) with 0 negative transfer.
    c_vs_b = report["comparisons"]["C_vs_B"]
    c_vs_a = report["comparisons"]["C_vs_A"]
    assert c_vs_b["selection_reasonable_delta"] > 0
    assert c_vs_b["false_adoption_delta"] < 0
    assert c_vs_b["pass_rate_delta"] > 0
    assert c_vs_a["pass_rate_delta"] > 0
    assert c_vs_a["negative_transfer_count"] == 0
    assert report["decision_gate"] == "continue_expanding_families"

    # Verify A3 Die-hours vs Card-hours separation (2 Dies per Card on A3 S1 cases).
    cont_a = report["cost_ledgers"]["continuation_cost_by_arm"]["A"]
    cont_c = report["cost_ledgers"]["continuation_cost_by_arm"]["C"]
    assert cont_c["npu_die_hours"] > cont_c["npu_card_hours"] > 0
    assert cont_c["cost_of_pass"]["wall_clock_sec_per_pass"] < cont_a["cost_of_pass"]["wall_clock_sec_per_pass"]
    assert cont_c["cost_of_pass"]["npu_die_hours_per_pass"] < cont_a["cost_of_pass"]["npu_die_hours_per_pass"]
    pareto = report["comparisons"]["pareto_and_cost_of_pass"]["pareto_selection"]
    assert pareto["frontier_ids"] == ["C"]
    assert "A" in pareto["dominated_ids"]
    assert pareto["representatives"]["lite"]["id"] == "C"
    assert pareto["representatives"]["medium"]["id"] == "C"
    assert pareto["representatives"]["heavy"]["id"] == "C"
    assert CANDIDATE_UNIT_IDS[0] in json.dumps(report)
