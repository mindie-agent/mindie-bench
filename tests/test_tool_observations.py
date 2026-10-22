import pytest

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.scorer.tool_observations import (
    observe_codex_result,
    observe_dsh_result,
    summarize_tool_observations,
)


def call(identity="call", *, name="remote_job_stdin", classification="pending", **fields):
    return ToolCall(
        id=identity,
        name=name,
        component="remote-dev",
        arguments={"chars": ""},
        turn=1,
        is_error=False,
        result_observation=ToolResultObservation(
            source="codex_mcp",
            classification=classification,
            operation_kind="observation",
            classification_reason="test_explicit_facts",
            **fields,
        ),
    )


def test_old_trajectory_remains_readable_with_unknown_result_coverage():
    old = Trajectory.model_validate(
        {"calls": [{"id": "old", "name": "remote_bash", "turn": 1, "is_error": False}]}
    )
    assert old.calls[0].result_observation is None
    result = summarize_tool_observations(old.calls)
    assert result["missing_observations"] == 1
    assert result["observation_coverage"] == 0
    assert result["non_business_error_count"] is None
    assert result["known_non_business_error_count"] == 0
    assert result["latency_by_operation"][0]["wall_clock_sec"]["p95"] is None


def test_empty_observations_have_no_invented_rates_or_latencies():
    result = summarize_tool_observations([])
    assert result["calls"] == 0
    assert result["observation_coverage"] is None
    assert result["non_business_error_count"] is None
    assert result["latency_by_operation"] == []


def test_pending_job_and_repeated_polls_are_not_extra_command_completions():
    start = call("start", name="remote_bash", job_id="job-1", target_id="host-1")
    polls = [
        call("poll1", job_id="job-1", target_id="host-1"),
        call(
            "poll2",
            job_id="job-1",
            target_id="host-1",
            classification="business_failure",
            exit_code=1,
        ),
        call(
            "poll3",
            job_id="job-1",
            target_id="host-1",
            classification="business_failure",
            exit_code=1,
        ),
    ]
    result = summarize_tool_observations([start, *polls])
    assert result["calls"] == 4
    assert result["execution_tool_call_count"] == 1
    assert result["job_result_observation_count"] == 4
    assert result["distinct_jobs_observed"] == 1
    assert result["job_terminal_counts"] == {"business_failure": 1}
    assert result["non_business_error_count"] == 0


def test_job_identity_includes_the_reported_endpoint():
    result = summarize_tool_observations(
        [
            call("one", job_id="same-id", target_id="endpoint-a"),
            call("two", job_id="same-id", target_id="endpoint-b"),
        ]
    )
    assert result["distinct_jobs_observed"] == 2


def test_unknown_job_fact_is_not_cleared_by_another_exit_zero_observation():
    result = summarize_tool_observations(
        [
            call("unknown", classification="unknown", job_id="job-1", exit_code=0),
            call("known", classification="business_success", job_id="job-1", exit_code=0),
        ]
    )
    assert result["job_terminal_counts"] == {"unknown": 1}
    assert result["non_business_error_count"] is None
    assert result["non_business_attribution_coverage"] == 0.5


def test_transport_error_and_legacy_missing_preserve_the_known_lower_bound():
    known = call(
        classification="unknown",
        error_domain="transport",
        error_category="rpc_disconnected",
        submission_state="uncertain",
    )
    legacy = ToolCall(id="old", name="remote_bash", turn=1)
    result = summarize_tool_observations([known, legacy])
    assert result["known_non_business_error_count"] == 1
    assert result["non_business_error_count"] is None
    assert result["observation_coverage"] == 0.5


def test_comparable_latency_groups_keep_operation_wait_policy_and_sample_coverage():
    first = call("one")
    first.started_sec, first.ended_sec = 1, 2
    first.arguments["yield_time_ms"] = 1000
    second = call("two")
    second.started_sec, second.ended_sec = 2, 5
    second.arguments["yield_time_ms"] = 1000
    missing = call("missing")
    missing.arguments["yield_time_ms"] = 1000
    different = call("different")
    different.started_sec, different.ended_sec = 5, 15
    different.arguments["yield_time_ms"] = 100
    result = summarize_tool_observations([first, second, missing, different])
    assert len(result["latency_by_operation"]) == 2
    group = next(
        row
        for row in result["latency_by_operation"]
        if row["wait_parameters"] == {"yield_time_ms": 1000}
    )
    assert group["wall_clock_sec"]["sample_count"] == 2
    assert group["wall_clock_sec"]["missing_count"] == 1
    assert group["wall_clock_sec"]["coverage"] == pytest.approx(2 / 3)
    assert group["wall_clock_sec"]["p50"] == 1
    assert group["wall_clock_sec"]["p95"] == 3
    assert result["latency_quantile_method"] == "nearest_rank"
    assert "Not a service SLA" in result["latency_scope"]


def test_invalid_tool_timing_stays_visible():
    item = call()
    item.started_sec, item.ended_sec = 2, 1
    with pytest.raises(ValueError, match="end precedes start"):
        summarize_tool_observations([item])


@pytest.mark.parametrize(
    ("details", "classification", "domain"),
    [
        ({"category": "rpc_disconnected", "submission_state": "uncertain"}, "unknown", "transport"),
        (
            {"category": "connection_unavailable", "submission_state": "not_sent"},
            "transport_error",
            "transport",
        ),
        ({"category": "command_protocol"}, "protocol_error", "protocol"),
        ({"category": "validation", "submission_state": "not_sent"}, "component_error", "component"),
        ({"category": "new_category"}, "unclassified", None),
    ],
)
def test_only_explicit_error_details_establish_failure_domain(details, classification, domain):
    item = call()
    result = observe_codex_result(
        {
            "type": "mcp_tool_call",
            "status": "failed",
            "result": {
                "structured_content": {
                    "schema_version": "remote-dev.result.v1",
                    "outcome": "failed",
                    "status": "failed",
                    "error_details": details,
                }
            },
        },
        item,
    )
    assert result.classification == classification
    assert result.error_domain == domain


def test_completed_business_exit_survives_a_separate_local_recording_error():
    item = call()
    result = observe_codex_result(
        {
            "type": "mcp_tool_call",
            "status": "failed",
            "result": {
                "structured_content": {
                    "schema_version": "remote-dev.result.v1",
                    "state": "succeeded",
                    "exit_code": 0,
                    "outcome": "failed",
                    "status": "local_recording_failed",
                    "operation_completed": True,
                    "local_recording": {"status": "failed", "error": "private detail"},
                }
            },
        },
        item,
    )
    assert result.classification == "component_error"
    assert result.operation_completed is True
    assert result.exit_code == 0


def test_missing_or_boolean_exit_code_is_not_a_command_success():
    for code in [None, True, "0"]:
        item = call()
        result = observe_codex_result(
            {
                "type": "mcp_tool_call",
                "status": "completed",
                "result": {
                    "structured_content": {
                        "schema_version": "remote-dev.result.v1",
                        "state": "succeeded",
                        "exit_code": code,
                        "outcome": "success",
                        "job_id": "job-1",
                    }
                },
            },
            item,
        )
        assert result.classification == "unclassified"
        assert result.exit_code is None


def test_negative_tool_timing_is_not_a_latency_sample():
    item = call()
    item.started_sec, item.ended_sec = -1, 2
    with pytest.raises(ValueError, match="invalid tool timing"):
        summarize_tool_observations([item])


def test_recording_error_keeps_a_known_terminal_job_exit_in_deduplication():
    item = call(
        classification="component_error",
        job_id="job-1",
        target_id="endpoint-1",
        exit_code=0,
        remote_status="succeeded",
        error_domain="component",
        operation_completed=True,
    )
    result = summarize_tool_observations([item])
    assert result["job_terminal_counts"] == {"business_success": 1}
    assert result["non_business_error_count"] == 1


def test_conflicting_exit_facts_do_not_become_a_deduplicated_job_success():
    item = call(
        classification="protocol_error",
        job_id="job-1",
        remote_status="succeeded",
        exit_code=0,
        error_domain="protocol",
    )
    item.result_observation.classification_reason = "conflicting_exit_codes"
    result = summarize_tool_observations([item])
    assert result["job_terminal_counts"] == {"unknown": 1}
    assert result["non_business_error_count"] == 1


def test_terminal_state_without_exit_is_not_reported_as_still_pending():
    result = summarize_tool_observations(
        [
            call("start", job_id="job-1"),
            call(
                "tail",
                name="remote_job_tail",
                classification="success",
                job_id="job-1",
                remote_status="succeeded",
            ),
        ]
    )
    assert result["job_terminal_counts"] == {"unclassified": 1}


def test_dsh_native_remote_text_and_unknown_tool_errors_are_attributable():
    bash_call = ToolCall(
        id="call_01",
        name="remote_bash",
        component="remote-dev",
        arguments={"command": "python3 reproduce.py"},
        turn=1,
        is_error=False,
    )
    bash_call.result_observation = observe_dsh_result(
        {
            "type": "tool_result",
            "callId": "call_01",
            "status": "completed",
            "result": "Remote command succeeded. Exit code: 0.\nsession_id: job-20261022T053855Z-ed70b8ba\n",
        },
        bash_call,
    )
    unknown_tool_call = ToolCall(
        id="call_02",
        name="remote_job_output",
        component="remote-dev",
        arguments={"job_id": "job-20261022T053855Z-ed70b8ba"},
        turn=1,
        is_error=True,
    )
    unknown_tool_call.result_observation = observe_dsh_result(
        {
            "type": "tool_result",
            "callId": "call_02",
            "status": "error",
            "result": 'Error: unknown tool "mcp__remote-dev__remote_job_output"',
        },
        unknown_tool_call,
    )
    summary = summarize_tool_observations([bash_call, unknown_tool_call])
    assert bash_call.result_observation.classification == "business_success"
    assert bash_call.result_observation.job_id == "job-20261022T053855Z-ed70b8ba"
    assert unknown_tool_call.result_observation.classification == "component_error"
    assert unknown_tool_call.result_observation.error_domain == "component"
    assert summary["non_business_attribution_coverage"] == 1.0
    assert summary["non_business_error_count"] == 1

