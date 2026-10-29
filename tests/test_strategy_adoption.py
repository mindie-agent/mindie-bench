"""Native tool receipts establish loading/declarations, separately from verdicts."""

from dataclasses import replace

import pytest

from mindie_bench.evaluation.strategy_adoption import (
    MAX_REFS,
    delivered_explain_refs,
    judge_continuation,
    observe_strategy_adoption,
)
from mindie_bench.evaluation.strategy_selection import (
    PILOT_CASES,
    plan_strategy_selection_cells,
    run_strategy_selection_cell,
)
from mindie_bench.models import ToolCall, Trajectory
from mindie_bench.runner.drivers.codex import CodexDriver
from mindie_bench.runner.strategy_continuation import ContinuationExecution

CASE = PILOT_CASES["S1-F1-trigger"]
UNIT = CASE.candidate_ids[0]
REF = f"mindie://operators/units/{UNIT}@1"


def native_call(driver, name, arguments, response, *, error=False, identity="call"):
    driver.consume({
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call", "id": identity, "server": "knowledge",
            "tool": name, "arguments": arguments,
            "status": "failed" if error else "completed",
            "result": {"isError": error, "structured_content": response},
        },
    }, elapsed=1, generation=1)
    driver.consume({"type": "turn.completed"}, elapsed=2, generation=1)
    return driver.trajectory()


def explain(driver, *, response=None, error=False, identity="explain"):
    body = {
        "units": [{"id": UNIT, "revision": 1, "ref": REF, "content": "Use the verified path."}],
        "requires": [], "unresolved": [],
    }
    return native_call(
        driver, "knowledge_explain", {"refs": [REF]},
        body if response is None else response, error=error, identity=identity,
    )


def application(driver, *, ref=REF, error=False):
    return native_call(
        driver, "knowledge_feedback",
        {"ref": ref, "event_type": "applied", "run_id": "native-run"},
        {"unit_id": ref.split("/units/")[-1].split("@")[0], "revision": 1,
         "event_type": "applied", "run_id": "native-run", "evidence_source": "agent_declared"},
        error=error, identity="feedback",
    )


@pytest.mark.tier0
def test_actual_native_explain_result_proves_loading_but_not_application():
    driver = CodexDriver()
    trajectory = explain(driver)
    observed = observe_strategy_adoption(candidate_ids=CASE.candidate_ids, trajectory=trajectory)
    assert observed["loaded_strategy_refs"] == [f"{UNIT}@1"]
    assert observed["loading_observation_complete"] is True
    assert observed["applied_strategy_refs"] is None
    payload = {**CASE.reference_solution, "trajectory": trajectory.model_dump()}
    judged = judge_continuation(CASE, payload)
    assert judged["task_succeeded"] is True
    assert judged["declared_adoption"] is None
    assert judged["applied_strategy_ref"] is None
    assert judged["applied_in_actions"] is None


@pytest.mark.tier0
@pytest.mark.parametrize("observation", ["empty", "failed", "unresolved", "malformed"])
def test_present_native_observation_overrides_application_claims(observation):
    driver = CodexDriver()
    if observation == "empty":
        trajectory = Trajectory(native_completed=True)
    elif observation == "failed":
        trajectory = application(driver, error=True)
    elif observation == "malformed":
        trajectory = {"calls": [{"name": "knowledge_feedback"}]}
    else:
        trajectory = explain(driver, response={
            "units": [], "requires": [],
            "unresolved": [{"ref": REF, "status": "not_found"}],
        })
    judged = judge_continuation(CASE, {**CASE.reference_solution, "trajectory": trajectory})
    assert judged["task_succeeded"] is True
    assert judged["applied_strategy_ref"] is None
    assert judged["false_adoption"] is None
    if observation == "malformed":
        assert "ValidationError" in judged["adoption_observation"]["error"]
    elif observation == "failed":
        assert judged["adoption_observation"]["call_outcomes_by_source"]["native_trajectory"] == {
            "failed_calls": 1, "unsettled_calls": 0,
        }


@pytest.mark.tier0
def test_successful_feedback_overrides_contradictory_claim_without_erasing_verifier_pass():
    trajectory = application(CodexDriver())
    case = replace(CASE, private_judge={
        **CASE.private_judge, "forbidden_strategies": [UNIT], "acceptable_strategies": [],
    })
    execution = ContinuationExecution(
        payload={"verifier_passed": False, "applied_strategy_ref": "default"},
        terminal_status="pass", completed=True, passed=True, evidence={},
        trajectory=trajectory,
    )
    judged = judge_continuation(case, execution.payload, execution=execution)
    assert judged["applied_strategy_ref"] == UNIT
    assert judged["declared_adoption"] is True
    assert judged["false_adoption"] is True
    assert judged["task_succeeded"] is True
    assert judged["applied_in_actions"] is None
    assert judged["adoption_observation"]["application_events"][0]["requested_evidence_source"] == "agent_declared"


@pytest.mark.tier0
def test_legacy_claims_stay_compatible_only_without_execution_observations():
    judged = judge_continuation(CASE, CASE.reference_solution)
    assert judged["applied_in_actions"] is True
    assert judged["adoption_observation"] is None


@pytest.mark.tier0
def test_partial_application_evidence_does_not_certify_absence_of_false_adoption():
    trajectory = application(CodexDriver())
    forbidden = CASE.candidate_ids[1]
    trajectory.native_completed = False
    trajectory.calls.append(ToolCall(
        id="pending-feedback", name="knowledge_feedback", component="knowledge", turn=1,
        arguments={"ref": f"{forbidden}@1", "event_type": "applied", "run_id": "native-run"},
    ))
    case = replace(CASE, private_judge={
        **CASE.private_judge, "acceptable_strategies": [UNIT], "forbidden_strategies": [forbidden],
    })
    judged = judge_continuation(case, {**CASE.reference_solution, "trajectory": trajectory})
    assert judged["declared_adoption"] is True
    assert judged["false_adoption"] is None
    assert judged["task_succeeded"] is True
    case = replace(case, private_judge={**case.private_judge, "forbidden_strategies": [UNIT]})
    assert judge_continuation(case, {**CASE.reference_solution, "trajectory": trajectory})["false_adoption"] is True


@pytest.mark.tier0
def test_failed_tool_with_malformed_arguments_preserves_real_error_and_bounded_projection():
    driver = CodexDriver()
    trajectory = native_call(
        driver, "knowledge_explain", {"refs": [{}]},
        {"error": "refs must contain strings"}, error=True,
    )
    observed = observe_strategy_adoption(candidate_ids=CASE.candidate_ids, trajectory=trajectory)
    assert observed["loading_observation_complete"] is False


    assert observed["call_outcomes_by_source"]["native_trajectory"]["failed_calls"] == 1
    for revision in range(1, MAX_REFS + 3):
        ref = f"{UNIT}@{revision}"
        native_call(driver, "knowledge_explain", {"refs": [ref]}, {
            "units": [{"id": UNIT, "ref": ref, "revision": revision, "content": "body"}],
            "requires": [],
        }, identity=str(revision))
    observed = observe_strategy_adoption(candidate_ids=CASE.candidate_ids, trajectory=driver.trajectory())
    assert len(observed["loaded_strategy_refs"]) == MAX_REFS
    assert len(observed["requested_strategy_refs"]) == MAX_REFS
    assert observed["loading_observation_complete"] is False


@pytest.mark.tier0
def test_native_completed_mcp_error_remains_failed_and_cannot_establish_loading():
    driver = CodexDriver()
    driver.consume({
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call", "id": "mcp-error", "server": "knowledge",
            "tool": "knowledge_explain", "arguments": {"refs": [REF]}, "status": "completed",
            "result": {"isError": True, "content": [{"type": "text", "text": "unit unavailable"}]},
        },
    }, elapsed=1, generation=1)
    observed = observe_strategy_adoption(candidate_ids=CASE.candidate_ids, trajectory=driver.trajectory())
    assert observed["loaded_strategy_refs"] == []
    assert observed["call_outcomes_by_source"]["native_trajectory"] == {
        "failed_calls": 1, "unsettled_calls": 0,
    }


@pytest.mark.tier0
@pytest.mark.parametrize("body", [
    {"id": UNIT}, {"units": [{"id": UNIT, "revision": 1}]},
    {"units": [{"id": UNIT, "content": "unpinned body"}]},
])
def test_missing_delivered_body_or_revision_is_incomplete_evidence(body):
    assert delivered_explain_refs({"structuredContent": body}) == ([], False)


@pytest.mark.tier0
@pytest.mark.parametrize("passed", [True, None])
def test_cell_retains_native_observations_even_without_a_verifier_outcome(passed):
    cell = next(row for row in plan_strategy_selection_cells([CASE], repetitions=1)[0] if row.arm == "A")
    execution = ContinuationExecution(
        payload={}, terminal_status="pass" if passed else "infra", completed=True,
        passed=passed, evidence={},
        error=None if passed else "verifier unavailable", trajectory=explain(CodexDriver()),
    )
    row = run_strategy_selection_cell(
        cell, CASE, routing_index=None, executor_fn=lambda **_: execution,
    )
    assert row["continuation_passed"] is passed
    assert row["adoption_observation"]["loaded_strategy_refs"] == [f"{UNIT}@1"]
    assert row["terminal_status"] == ("pass" if passed else "infra")
