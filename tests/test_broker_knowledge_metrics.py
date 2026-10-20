import json

import pytest

from mindie_bench.evaluation.broker_knowledge import summarize_broker_knowledge
from mindie_bench.models import ToolCall, Trajectory, TrialReceipt
from mindie_bench.scorer.metrics import summarize, trajectory_metrics


def event(kind, time, *, request=1, tool="knowledge_query", session="session", **values):
    return dict(
        kind=kind,
        monotonic_sec=time,
        server="knowledge",
        session_id=session,
        request_id=request,
        tool=tool,
        **values,
    )


def journal(tmp_path, events):
    path = tmp_path / "broker.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    return path


def paired(*, start=101, end=104, tool="knowledge_query", request=1, session="session", error=False):
    return [
        event("knowledge_tool_start", start, request=request, tool=tool, session=session),
        event(
            "knowledge_tool_end",
            end,
            request=request,
            tool=tool,
            session=session,
            elapsed_sec=end - start,
            is_error=error,
        ),
    ]


def test_broker_counts_manual_sdk_call_absent_from_native_trajectory(case, tmp_path):
    observed = summarize_broker_knowledge(journal(tmp_path, paired()))
    trajectory = Trajectory()
    primary, aux = trajectory_metrics(
        trajectory, case["spec"], case["config"].harness, 10, tier="l2", broker_knowledge=observed
    )
    assert aux["knowledge_queries_count"] == 0  # Native field keeps its original scope.
    assert aux["knowledge_measurement"]["query_count"] == 1
    assert aux["knowledge_measurement"]["basis"] == "broker_tool_start_end/1"
    assert primary["knowledge_query_sec"] == 3
    assert primary["foreground_knowledge_queries_count"] == 1
    assert primary["knowledge_measurement_basis"] == "broker_tool_start_end/1"
    assert primary["llm_and_exploration_sec"] is None
    assert aux["tool_timing"]["remainder_reason"] == "broker_native_clock_alignment_missing"
    row = TrialReceipt(
        trial_id="manual-query",
        instance_id="task",
        repetition=1,
        tier="l2",
        ablation_group="G2",
        knowledge_gradient="analogical",
        task_digest="task-digest",
        experiment_digest="experiment",
        environment_fingerprint={},
        terminal_status="infra_failed",
        failure_reason="original caller frame failure remains visible",
        primary_metrics=primary,
        auxiliary_metrics=aux,
        evidence="synthetic",
    )
    summary = summarize([row])["knowledge_observations"]
    assert summary["query_count"] == 1
    assert summary["state"] == "observed_mcp_calls"
    assert summary["measurement_bases"] == ["broker_tool_start_end/1"]
    assert summary["broker_count_coverage"]["state"] == "complete"


def test_union_uses_broker_clock_bridge_and_avoids_nested_native_double_count(case, tmp_path):
    events = paired(start=102, end=106) + paired(start=104, end=108, request=2, tool="knowledge_explain")
    observed = summarize_broker_knowledge(journal(tmp_path, events), process_origin_monotonic_sec=100)
    trajectory = Trajectory(
        calls=[
            ToolCall(
                id="remote",
                name="remote_bash",
                component="remote-dev",
                arguments={"command": "true"},
                turn=1,
                started_sec=1,
                ended_sec=4,
            ),
            ToolCall(
                id="native-wrapper",
                name="knowledge_query",
                turn=1,
                started_sec=1.5,
                ended_sec=6.5,
            ),
        ]
    )
    primary, aux = trajectory_metrics(
        trajectory, case["spec"], case["config"].harness, 10, tier="l2", broker_knowledge=observed
    )
    assert primary["knowledge_query_sec"] == 6
    assert primary["llm_and_exploration_sec"] == 3  # [1,8] union, not 3+4+4+5.
    assert aux["broker_knowledge"]["timing"]["raw_cumulative_sec"] == 8
    assert aux["broker_knowledge"]["timing"]["overlap_sec"] == 2
    assert aux["tool_timing"]["combined"]["observed_union_sec"] == 7
    assert aux["tool_timing"]["native_knowledge"]["observed_union_sec"] == 5


def test_incomplete_and_error_calls_keep_count_and_unknown_timing(case, tmp_path):
    events = paired(error=True) + [event("knowledge_tool_start", 106, request=2)]
    observed = summarize_broker_knowledge(journal(tmp_path, events))
    assert observed["query_count"] == 2
    assert observed["completed_calls"] == 1
    assert observed["failed_calls"] == 1
    assert observed["incomplete_calls"] == 1
    assert observed["timing"]["observed_union_sec"] == 3
    assert observed["timing"]["complete"] is False
    primary, aux = trajectory_metrics(
        Trajectory(), case["spec"], case["config"].harness, 10, tier="l2", broker_knowledge=observed
    )
    assert primary["knowledge_query_sec"] is None
    assert aux["knowledge_measurement"]["query_count"] == 2
    assert aux["knowledge_measurement"]["successful_calls"] == 0
    assert primary["llm_and_exploration_sec"] is None


def test_catalog_and_independent_consumer_do_not_count_as_foreground_knowledge(tmp_path):
    path = journal(
        tmp_path,
        [
            dict(kind="knowledge_catalog_projection", monotonic_sec=1),
            dict(kind="consumer_query", monotonic_sec=2, tool="knowledge_query"),
        ],
    )
    observed = summarize_broker_knowledge(path)
    assert observed["query_count"] == observed["explain_count"] == 0
    assert observed["timing"]["complete"] is True
    assert observed["timing"]["observed_union_sec"] == 0


def test_separate_sessions_can_reuse_jsonrpc_ids(tmp_path):
    observed = summarize_broker_knowledge(
        journal(tmp_path, paired() + paired(session="other", start=102, end=105))
    )
    assert observed["query_count"] == 2
    assert observed["timing"]["observed_union_sec"] == 4


def test_completed_request_id_can_be_reused_sequentially_without_collapsing_calls(tmp_path):
    observed = summarize_broker_knowledge(journal(tmp_path, paired() + paired(start=106, end=107)))
    assert observed["query_count"] == 2
    assert observed["completed_calls"] == 2


@pytest.mark.parametrize(
    "events",
    [
        paired() + paired(),
        [event("knowledge_tool_end", 2, elapsed_sec=1, is_error=False)],
        paired(start=104, end=101),
    ],
)
def test_corrupt_or_ambiguous_settlement_is_visible_not_zero(tmp_path, events):
    with pytest.raises(ValueError):
        summarize_broker_knowledge(journal(tmp_path, events))


def test_origin_mismatch_does_not_destroy_absolute_knowledge_measurement(tmp_path):
    observed = summarize_broker_knowledge(journal(tmp_path, paired()), process_origin_monotonic_sec=200)
    assert observed["timing"]["observed_union_sec"] == 3
    assert observed["relative_intervals"] is None
    assert observed["alignment_reason"] == "broker_call_precedes_native_process_origin"


def test_completed_preflight_query_does_not_inflate_native_agent_reuse(case, tmp_path):
    path = journal(
        tmp_path,
        paired(start=90, end=93, session="preflight")
        + paired(start=102, end=104, session="agent"),
    )
    observed = summarize_broker_knowledge(
        path, process_origin_monotonic_sec=100, preflight_finished_monotonic_sec=95
    )
    primary, _ = trajectory_metrics(
        Trajectory(), case["spec"], case["config"].harness, 10, tier="l2", broker_knowledge=observed
    )
    assert observed["preflight"]["completed_calls"] == 1
    assert observed["preflight"]["calls"][0]["session_id"] == "preflight"
    assert primary["foreground_knowledge_queries_count"] == 1
    assert primary["knowledge_query_sec"] == 2
    assert observed["relative_intervals"] == [dict(started_sec=2, ended_sec=4)]


def test_unsettled_read_is_not_hidden_by_a_preflight_boundary(tmp_path):
    path = journal(tmp_path, [event("knowledge_tool_start", 90)])
    observed = summarize_broker_knowledge(
        path, process_origin_monotonic_sec=100, preflight_finished_monotonic_sec=95
    )
    assert observed["preflight"]["completed_calls"] == 0
    assert observed["incomplete_calls"] == 1
    assert observed["alignment_reason"] == "broker_call_precedes_native_process_origin"
