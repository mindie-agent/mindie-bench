"""Section 4.7 & Section 16 conditional strategy selection pilot tests.

Exercises committed public corpus units (`knowledge-vllm-ascend` + `knowledge-operators`),
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
    condition_aware_selector,
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
    root = Path(__file__).parent / "fixtures" / "strategy-corpora"
    return [root / "knowledge-vllm-ascend", root / "knowledge-operators"]


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


class StepClock:
    """perf_counter stand-in advanced only by the injected selector and executor."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.mark.tier0
def test_cell_accounting_uses_reported_spend_and_measured_time(routing_index):
    with pytest.raises(ValueError, match="outer_timeout_sec"):
        StrategySelectionBudget(
            business_window_sec=600.0,
            verifier_timeout_sec=120.0,
            cleanup_grace_sec=5.0,
            outer_timeout_sec=720.0,
        )

    case = PILOT_CASES["S1-F1-boundary"]
    cell_c = next(c for c in plan_strategy_selection_cells([case], repetitions=1)[0] if c.arm == "C")
    clock, granted = StepClock(), []
    # A continuation that independently reaches the reference outcome in 20 s.
    continuation_outcome = dict(case.reference_solution)

    def executor(*, case, arm, selector_decision, routing_index, remaining_model_calls, remaining_window_sec):
        granted.append((remaining_model_calls, remaining_window_sec))
        clock.now += 20
        return {
            **continuation_outcome,
            "model_calls": 3, "prompt_tokens": 100, "completion_tokens": 10, "npu_runs": 1,
        }

    def selector(seconds, *, usage=None, error=None):
        def run(view):
            clock.now += seconds
            if error is not None:
                raise error
            return {"selected_strategy_ref": "default", "applicability_basis": "mismatch", "usage": usage}
        return run

    def cell(selector_fn, executor_fn=executor):
        return run_strategy_selection_cell(
            cell_c, case, routing_index=routing_index, selector_fn=selector_fn,
            executor_fn=executor_fn, clock=clock,
        )

    # The rule selector makes no model call, so it spends and reserves none of the budget.
    rule = cell(condition_aware_selector)
    assert granted.pop() == (12, 600.0)
    assert rule["selector_model_calls"] == 0 and rule["total_model_calls"] == 3
    assert rule["usage_status"] == "native" and rule["prompt_tokens"] == 100
    assert rule["budget_compliant"] is True

    # A decision returned after its 30 s budget is kept; the overrun is stated separately.
    native = {"model_calls": 1, "prompt_tokens": 300, "completion_tokens": 40}
    late = cell(selector(45, usage=native))
    assert granted.pop() == (11, 555.0)
    assert late["selector_status"] == "completed" and late["selection"]["selection_acceptable"] is True
    assert late["selector_wall_clock_sec"] == 45.0 and late["selector_budget_exceeded"] is True
    assert late["terminal_status"] == "pass" and late["budget_compliant"] is False
    assert late["total_model_calls"] == 4 and late["prompt_tokens"] == 400

    # A genuine timeout returns nothing: spend stays unknown and time is the measured 31 s.
    timed_out = cell(selector(31, error=TimeoutError("selector gateway timed out")))
    assert granted.pop() == (11, 569.0)
    assert timed_out["selector_status"] == "unknown" and timed_out["selector_wall_clock_sec"] == 31.0
    assert timed_out["selector_model_calls"] is None and timed_out["total_model_calls"] is None
    assert timed_out["executor_model_calls"] == 3 and timed_out["executor_prompt_tokens"] == 100
    assert timed_out["usage_status"] == "unavailable" and timed_out["prompt_tokens"] is None
    assert timed_out["terminal_status"] == "pass" and timed_out["budget_compliant"] is False
    # One reported NPU run without measured occupancy is not converted into card time.
    assert timed_out["npu_runs"] == 1
    assert timed_out["npu_card_hours"] is None and timed_out["npu_die_hours"] is None

    # A dispatched continuation that times out leaves the cell total unknown.
    def lost_executor(**kwargs):
        clock.now += 50
        raise TimeoutError("continuation receipt lost")

    lost = cell(selector(2, usage=native), lost_executor)
    assert lost["terminal_status"] == "unknown" and lost["executor_error"] == "continuation receipt lost"
    assert lost["selector_model_calls"] == 1 and lost["total_model_calls"] is None
    assert lost["wall_clock_sec"] == 52.0 and lost["budget_compliant"] is None

    # Verify static mechanism replay also preserves None usage rather than defaulting to 0.
    static_replay = run_host_mechanism_replay(
        "M1-01",
        decision_payload={"action": "phase0_fusion_triage", "hypothesis_id": "H1_fusion_boundary"},
    )
    assert static_replay["execution_metrics"]["usage_status"] == "unavailable"
    assert static_replay["execution_metrics"]["prompt_tokens"] is None
    assert static_replay["execution_metrics"]["completion_tokens"] is None


@pytest.mark.tier0
def test_executor_never_receives_judge_answers_and_card_time_needs_measured_occupancy(routing_index):
    case = PILOT_CASES["S1-F1-trigger"]
    cells, _ = plan_strategy_selection_cells([case], repetitions=1)
    received = []

    def executor(*, case, arm, selector_decision, routing_index, remaining_model_calls, remaining_window_sec):
        received.append((case.private_judge, case.reference_solution, case.known_bad_solution))
        ref = (selector_decision or {}).get("selected_strategy_ref", "default")
        outcome = {
            "selected_strategy_ref": ref, "declared_adoption": ref != "default",
            "applied_strategy_ref": ref, "next_action": "measured_probe", "verifier_passed": False,
            "model_calls": 2, "prompt_tokens": 100, "completion_tokens": 10, "npu_runs": 1,
        }
        return outcome if arm == "A" else {**outcome, "npu_occupancy_sec": 1800.0}

    rows = {
        cell.arm: run_strategy_selection_cell(
            cell, case, routing_index=routing_index, executor_fn=executor,
        )
        for cell in cells
    }
    assert received == [({}, {}, {})] * 3
    assert case.private_judge["acceptable_strategies"]
    # The rule selector reports zero model calls, so only the continuation spends tokens.
    assert rows["B"]["selector_model_calls"] == 0 and rows["B"]["total_model_calls"] == 2
    assert rows["B"]["usage_status"] == "native" and rows["B"]["prompt_tokens"] == 100
    assert rows["A"]["usage_status"] == "native" and rows["A"]["prompt_tokens"] == 100
    assert rows["A"]["npu_runs"] == 1 and rows["A"]["npu_card_hours"] is None
    assert rows["C"]["npu_card_hours"] == 0.5 * case.npu_cards
    assert rows["C"]["npu_die_hours"] == 0.5 * case.npu_dies


@pytest.mark.tier1
def test_unreported_continuation_spend_stays_unknown_through_cost_of_pass():
    def executor(*, case, arm, selector_decision, routing_index, remaining_model_calls, remaining_window_sec):
        ref = (selector_decision or {}).get("selected_strategy_ref", "default")
        return {
            "selected_strategy_ref": ref, "declared_adoption": ref != "default", "applied_strategy_ref": ref,
            "next_action": "measured_probe", "verifier_passed": True, "npu_runs": 0,
        }

    report = run_strategy_selection_pilot(corpus_roots=_corpus_roots(), repetitions=1, executor_fn=executor)
    assert report["continuation"] == "executed"
    for ledger in report["cost_ledgers"]["continuation_cost_by_arm"].values():
        assert ledger["model_calls"] is None and ledger["npu_card_hours"] is None
        assert ledger["cost_of_pass"]["model_calls_per_pass"] is None
        assert ledger["cost_of_pass"]["npu_die_hours_per_pass"] is None
        assert ledger["cost_of_pass"]["wall_clock_sec_per_pass"] is not None


@pytest.mark.tier1
def test_cli_pilot_without_executor_reports_only_judged_selector_decisions(tmp_path):
    roots = _corpus_roots()
    dev_report = run_strategy_selection_pilot(
        corpus_roots=roots,
        cases=tuple(DEV_CASES.values()),
        repetitions=1,
    )
    assert dev_report["continuation"] == "not_executed"
    assert {row["terminal_status"] for row in dev_report["cells"]} == {"not_executed"}

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

    # Without a continuation executor nothing beyond the selector decision is reported.
    assert report["continuation"] == "not_executed"
    assert report["decision_gate"] == "continuation_not_executed"
    assert "comparisons" not in report and "cost_ledgers" not in report and "arms" not in report
    for row in report["cells"]:
        assert row["terminal_status"] == "not_executed" and row["judgment"] is None
        assert row["npu_runs"] is None and row["npu_card_hours"] is None
        assert row["executor_model_calls"] is None
        assert (row["selection"] is None) == (row["cell"]["arm"] == "A")

    # The private judge scores the real B/C selector decisions on isolated views.
    selection = report["selection"]
    assert selection["arms"]["B"]["judged_cells"] == selection["arms"]["C"]["judged_cells"] == 12
    assert selection["C_vs_B"]["selection_acceptable_delta"] > 0
    assert selection["C_vs_B"]["false_adoption_delta"] < 0
    assert CANDIDATE_UNIT_IDS[0] in json.dumps(report)
