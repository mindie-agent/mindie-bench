import pytest

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory, TrialReceipt
from mindie_bench.scorer.comparator import CONTROL_KEYS, comparison, markdown
from mindie_bench.scorer.metrics import summarize, trajectory_metrics


def row(task="task", repetition=1, group="G0", status="completed_pass", **metrics):
    primary = dict(
        resolved=status == "completed_pass" if status.startswith("completed") else None,
        reward=float(status == "completed_pass") if status.startswith("completed") else None,
        total_wall_clock_sec=10,
        delivery_wall_clock_sec=20,
        localization_penalty_turns=40,
        turns_to_first_localization=2,
        tokens=dict(input_uncached=10, input_cached=5, output=2, complete=True),
        cost_complete=False,
    )
    primary.update(metrics)
    return TrialReceipt(
        trial_id=f"{task}-{group}-{repetition}",
        instance_id=task,
        repetition=repetition,
        tier="l3",
        ablation_group=group,
        knowledge_gradient="analogical",
        task_digest=f"{task}-digest",
        experiment_digest="experiment",
        environment_fingerprint={key: None for key in CONTROL_KEYS},
        terminal_status=status,
        failure_reason=None if status == "completed_pass" else "preserved original error",
        primary_metrics=primary,
        auxiliary_metrics=dict(
            knowledge_queries_count=0,
            knowledge_explains_count=0,
            remote_tool_non_business_errors=0,
        ),
        evidence="synthetic",
    )


def cell(task="task", repetition=1, group="G0"):
    return dict(
        instance_id=task,
        tier="l3",
        group=group,
        repetition=repetition,
        role="eval",
        task_digest=f"{task}-digest",
    )


def test_invalid_outcomes_still_have_recorded_elapsed_time_with_its_own_denominator():
    rows = [
        row(),
        row("wrong", status="completed_fail", total_wall_clock_sec=30, delivery_wall_clock_sec=40),
        row("uncertain", status="unknown", total_wall_clock_sec=600, delivery_wall_clock_sec=650),
        row("broken", status="infra_failed", total_wall_clock_sec=None, delivery_wall_clock_sec=None),
    ]
    result = summarize(rows)
    assert result["mean_wall_clock_sec"] == 20
    assert result["mean_delivery_wall_clock_sec"] == 30
    assert result["mean_delivery_wall_clock_count"] == 2
    assert result["all_attempt_recorded_wall_clock_sec_mean"] == pytest.approx(640 / 3)
    assert result["all_attempt_recorded_delivery_wall_clock_sec_mean"] == pytest.approx(710 / 3)
    assert result["all_attempt_recorded_delivery_wall_clock_sec_count"] == 3
    coverage = result["measurement_coverage"]
    assert coverage["delivery_valid"]["population_attempts"] == 2
    assert coverage["delivery_valid"]["observed_attempts"] == 2
    assert coverage["delivery_all_attempts"]["population_attempts"] == 4
    assert coverage["delivery_all_attempts"]["missing_attempts"] == 1
    assert coverage["delivery_all_attempts"]["state"] == "partial"
    assert rows[2].terminal_status.value == "unknown"


def test_empty_summary_has_no_measured_zero_or_matrix_reliability_claim():
    result = summarize([])
    assert result["all_attempt_recorded_delivery_wall_clock_sec_mean"] is None
    assert result["measurement_coverage"]["tokens"]["state"] == "empty"
    assert result["knowledge_observations"]["query_count"] is None
    assert result["knowledge_observations"]["state"] == "empty"
    assert result["reliability_coverage"]["state"] == "empty"
    assert result["matrix_coverage"]["complete"] is None
    assert result["matrix_coverage"]["reason"] == "plan_inventory_unavailable"


def test_knowledge_timing_comparison_rejects_different_measurement_scopes():
    baseline, candidate = row(), row(group="G1")
    baseline.primary_metrics.update(knowledge_query_sec=0, llm_and_exploration_sec=10)
    candidate.primary_metrics.update(
        knowledge_query_sec=3,
        llm_and_exploration_sec=7,
        knowledge_measurement_basis="broker_tool_start_end/1",
    )
    result = comparison([baseline], [candidate])
    assert result["funnel"]["knowledge_query_sec"] is None
    assert result["funnel"]["llm_and_exploration_sec"] is None
    assert result["funnel_measurement_bases"]["knowledge_query_sec"]["reason"] == (
        "knowledge_measurement_basis_mismatch"
    )


def test_zero_mcp_counts_are_distinct_from_missing_or_native_body_read_measurement():
    observed = summarize([row(group="G2")])
    knowledge = observed["knowledge_observations"]
    assert knowledge["state"] == "observed_zero_mcp_calls"
    assert knowledge["query_count"] == knowledge["explain_count"] == 0
    assert knowledge["query_count_coverage"]["observed_attempts"] == 1
    assert knowledge["native_body_access"] == "not_measured_by_mcp_counters"
    assert knowledge["benefit"]["value"] is None
    assert knowledge["benefit"]["reason"] == "no_observed_mcp_retrieval"

    missing = row("legacy", group="G2")
    missing.auxiliary_metrics = {}
    knowledge = summarize([row(group="G2"), missing])["knowledge_observations"]
    assert knowledge["state"] == "partial"
    assert knowledge["query_count"] is None
    assert knowledge["observed_query_count_lower_bound"] == 0
    assert knowledge["query_count_coverage"]["missing_attempts"] == 1
    assert knowledge["benefit"]["reason"] == "mcp_observation_incomplete"


def test_token_localization_and_remote_error_gaps_are_not_zero_observations():
    legacy = row("legacy", status="unknown", turns_to_first_localization=None)
    legacy.primary_metrics["tokens"]["complete"] = False
    legacy.auxiliary_metrics["remote_tool_non_business_errors"] = None
    result = summarize([row(), legacy])
    coverage = result["measurement_coverage"]
    for key in ["tokens", "localization", "remote_non_business_errors"]:
        assert coverage[key]["observed_attempts"] == 1
        assert coverage[key]["missing_attempts"] == 1
        assert coverage[key]["state"] == "partial"
    assert coverage["tokens"]["recorded_attempts"] == 2
    assert coverage["localization"]["reason"] == "source_access_proxy_unobserved"
    assert coverage["remote_non_business_errors"]["reason"] == "classification_missing"


def test_eligible_reliability_cannot_be_presented_as_the_full_task_population():
    rows = [row("easy", i) for i in (1, 2, 3)]
    rows += [
        row("partial"),
        row("partial", 2, status="unknown"),
        row("partial", 3, status="infra_failed"),
    ]
    result = summarize(rows)
    assert result["pass_pow_k"] == 1
    coverage = result["reliability_coverage"]
    assert coverage["eligible_tasks"] == 1
    assert coverage["observed_tasks"] == 2
    assert coverage["ineligible_tasks"] == ["partial"]
    assert coverage["complete_for_observed_tasks"] is False
    assert result["pass_pow_k_scope"] == "eligible_observed_tasks_only"
    assert result["whole_matrix_pass_pow_k"] is None


def test_duplicate_repetition_cannot_satisfy_three_independent_repeat_requirement():
    result = summarize([row(), row(), row()])
    assert result["pass_pow_k"] is None
    assert result["reliability_coverage"]["eligible_tasks"] == 0


def test_explicit_inventory_reports_blocked_and_missing_without_fabricating_receipts():
    expected = [cell(repetition=i, group=g) for g in ["G0", "G2"] for i in [1, 2, 3]]
    left = [row(repetition=i) for i in [1, 2]]
    right = [row(repetition=i, group="G2") for i in [1, 2]]
    blocked = [cell(repetition=3, group="G2")]
    result = comparison(left, right, expected_cells=expected, blocked_cells=blocked)
    coverage = result["comparison_coverage"]
    assert coverage["matched_observed_pairs"] == coverage["matched_valid_pairs"] == 2
    assert coverage["complete"] is False
    left_matrix = result["baseline"]["matrix_coverage"]
    right_matrix = result["candidate"]["matrix_coverage"]
    assert left_matrix["planned_cells"] == right_matrix["planned_cells"] == 3
    assert len(left_matrix["missing_terminal_cells"]) == 1
    assert left_matrix["blocked_cells"] == []
    assert len(left_matrix["unclassified_missing_cells"]) == 1
    assert right_matrix["blocked_cells"] == blocked
    assert right_matrix["unclassified_missing_cells"] == []
    assert right_matrix["all_planned_cells_accounted_for"] is True
    assert right_matrix["complete"] is False
    assert result["metric_qualifications"]["wall_clock_speedup_on_mutual_success"]["causal"] is False
    assert result["knowledge_effectiveness"]["value"] is None


def test_no_inventory_cannot_prove_missing_blocked_or_complete_matrix():
    result = comparison([row()], [row(group="G2")])
    coverage = result["comparison_coverage"]
    assert coverage["complete"] is None
    assert result["candidate"]["matrix_coverage"]["missing_terminal_cells"] is None
    assert result["candidate"]["matrix_coverage"]["blocked_cells"] is None
    assert result["knowledge_effectiveness"]["reason"] == "no_observed_mcp_retrieval"
    text = markdown(result)
    assert "eligible observed tasks only" in text
    assert "Recorded elapsed time across all attempts" in text
    assert "plan_inventory_unavailable" in text
    assert "not evidence of knowledge benefit" in text


def test_complete_inventory_can_qualify_reliability_but_not_causal_knowledge_benefit():
    rows = [row(repetition=i, group="G2") for i in [1, 2, 3]]
    for receipt in rows:
        receipt.auxiliary_metrics.update(knowledge_queries_count=1, knowledge_explains_count=1)
    result = summarize(
        rows, expected_cells=[cell(repetition=i, group="G2") for i in [1, 2, 3]], blocked_cells=[]
    )
    assert result["matrix_coverage"]["complete"] is True
    assert result["whole_matrix_pass_pow_k"] == 1
    assert result["knowledge_observations"]["state"] == "observed_mcp_calls"
    assert result["knowledge_observations"]["benefit"]["value"] is None
    assert result["knowledge_observations"]["benefit"]["reason"] == "causal_protocol_not_supplied"


def test_explicit_inventory_does_not_allow_intersection_only_comparison():
    with pytest.raises(ValueError, match="same tasks, tiers, and repetitions"):
        comparison(
            [row(repetition=1), row(repetition=2)],
            [row(repetition=1, group="G2")],
            expected_cells=[cell(repetition=i, group=g) for g in ["G0", "G2"] for i in [1, 2]],
            blocked_cells=[cell(repetition=2, group="G2")],
        )


def test_wrong_declared_task_digest_is_reported_and_cannot_qualify_a_comparison():
    wrong = cell(group="G2")
    wrong["task_digest"] = "different-task-revision"
    summary = summarize([row(group="G2")], expected_cells=[wrong], blocked_cells=[])
    assert summary["matrix_coverage"]["identity_mismatches"][0]["fields"] == ["task_digest"]
    assert summary["matrix_coverage"]["complete"] is False
    with pytest.raises(ValueError, match="declared inventory"):
        comparison(
            [row()],
            [row(group="G2")],
            expected_cells=[cell(), wrong],
            blocked_cells=[],
        )


def test_blocked_is_an_explicit_absent_cell_and_cannot_cover_an_actual_receipt():
    with pytest.raises(ValueError, match="blocked cell cannot also have a terminal receipt"):
        summarize([row()], expected_cells=[cell()], blocked_cells=[cell()])


def test_coverage_copies_only_public_cell_identity_not_private_operator_notes():
    blocked = dict(cell(repetition=2), private_command="private operator payload")
    result = summarize([row()], expected_cells=[cell(), blocked], blocked_cells=[blocked])[
        "matrix_coverage"
    ]
    assert result["blocked_cells"] == [cell(repetition=2)]
    assert result["missing_terminal_cells"] == [cell(repetition=2)]


def test_structured_remote_classification_is_separate_from_native_error_flag(case):
    business = ToolCall(
        id="business",
        name="remote_bash",
        turn=1,
        is_error=False,
        result_observation=ToolResultObservation(
            source="codex_mcp",
            classification="business_failure",
            operation_kind="execution",
            exit_code=1,
            classification_reason="explicit_remote_command_exit",
        ),
    )
    transport = ToolCall(
        id="transport",
        name="remote_bash",
        turn=2,
        is_error=False,
        result_observation=ToolResultObservation(
            source="codex_mcp",
            classification="transport_error",
            operation_kind="execution",
            error_domain="transport",
            classification_reason="explicit_error_details",
        ),
    )
    legacy = ToolCall(id="legacy", name="remote_bash", turn=3, is_error=False)
    _, complete = trajectory_metrics(
        Trajectory(calls=[business, transport]), case["spec"], case["config"].harness, 1, tier="l3"
    )
    assert complete["remote_tool_errors"] == 0
    assert complete["remote_tool_non_business_errors"] == 1
    assert complete["tool_observations"]["classifications"]["business_failure"] == 1
    _, partial = trajectory_metrics(
        Trajectory(calls=[business, transport, legacy]),
        case["spec"],
        case["config"].harness,
        1,
        tier="l3",
    )
    assert partial["remote_tool_errors"] == 0
    assert partial["remote_tool_non_business_errors"] is None
    assert partial["tool_observations"]["known_non_business_error_count"] == 1
    assert partial["tool_observations"]["missing_observations"] == 1
