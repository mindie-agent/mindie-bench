import shlex

import pytest

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.scorer.metrics import first_localization, trajectory_metrics

TARGET = "vllm_ascend/ops/layernorm.py"


def call(identity, command=None, *, turn=1, start=0, end=1, error=False, name="bash", **arguments):
    if command is not None:
        arguments["command"] = command
    return ToolCall(
        id=identity,
        name=name,
        arguments=arguments,
        turn=turn,
        started_sec=start,
        ended_sec=end,
        is_error=error,
    )


def metrics(case, calls, wall=20):
    task = case["spec"].model_copy(deep=True)
    task.mindie_knowledge.target_files = [TARGET]
    return trajectory_metrics(Trajectory(calls=calls), task, case["config"].harness, wall, tier="l3")


@pytest.mark.parametrize(
    "command",
    [
        f"cat {TARGET}",
        f"sed -n '34,55p' {TARGET}",
        f"rg -n 'quant_description|class AscendRMSNorm' {TARGET}",
        "/bin/zsh -c "
        + shlex.quote(
            "ssh -p 22 bench-slot " + shlex.quote(f"cd /bench/work && sed -n '34,55p' {TARGET}")
        ),
        f"python3 -c \"from pathlib import Path; p=Path('{TARGET}'); print(p.read_text())\"",
        f"python3 -c \"from pathlib import Path as P; p=P('{TARGET}'); print(p.read_text())\"",
        f"python3 - <<'PY'\nfrom pathlib import Path\np=Path('{TARGET}')\nprint(p.read_text())\nPY",
        f"bash -c 'true' && cat {TARGET}",
        "/bin/bash -c "
        + shlex.quote(
            f"python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('{TARGET}').read_text())\nPY"
        ),
        f"/bin/bash -c 'python3 -' <<'PY'\nfrom pathlib import Path\nprint(Path('{TARGET}').read_text())\nPY",
    ],
)
def test_shell_and_python_source_access_is_observed_without_execution(command):
    trajectory = Trajectory(calls=[call("read", command, turn=3)])
    assert first_localization(trajectory, [TARGET]) == 3


@pytest.mark.parametrize(
    "command",
    [
        f"echo 'cat {TARGET}'",
        f"# cat {TARGET}\necho harmless",
        f"grep '{TARGET}' other.txt",
        f"cat <<'PATCH'\n*** Update File: {TARGET}\nPATCH",
        f"python3 -c \"print('{TARGET}')\"",
        f"python3 -c \"from pathlib import Path; Path('patch.txt').write_text('cat {TARGET}')\"",
        f"python3 - <<'PY'\nfrom pathlib import Path\ndef unused():\n    return Path('{TARGET}').read_text()\nPY",
    ],
)
def test_filename_mentions_and_opaque_payloads_are_not_source_access(command):
    assert first_localization(Trajectory(calls=[call("mention", command)]), [TARGET]) is None


def test_edit_metrics_use_successful_edit_calls_never_a_later_read(case):
    calls = [
        call("read", f"cat {TARGET}", turn=2, start=2, end=3),
        call("failed-edit", f"sed -i 's/a/b/' {TARGET}", turn=3, start=4, end=5, error=True),
        call(
            "first-edit",
            f"python3 -c \"from pathlib import Path; Path('{TARGET}').write_text('new')\"",
            turn=4,
            start=6,
            end=7,
        ),
        call("test", "ssh bench-slot python3 test_rmsnorm_npu.py", turn=5, start=8, end=9),
        call("last-edit", name="Edit", paths=[TARGET], turn=6, start=10, end=11),
        call("last-read", f"sed -n '1,80p' {TARGET}", turn=8, start=13, end=14),
    ]
    primary, aux = metrics(case, calls)
    assert primary["turns_to_first_localization"] == 2
    assert primary["time_to_first_localization_sec"] == 2
    assert primary["turns_to_first_target_edit"] == 4
    assert primary["time_to_first_target_edit_sec"] == 6
    assert primary["turns_to_last_target_edit"] == 6
    assert primary["time_to_last_target_edit_sec"] == 10
    assert primary["npu_runs_before_last_target_edit"] == 1
    assert aux["target_edit_attempts_by_outcome"] == {"success": 2, "failed": 1, "unknown": 0}
    assert aux["target_edit_metric_reasons"]["turns_to_last_target_edit"] is None


@pytest.mark.parametrize(
    "attempt",
    [
        call("failed-cat", f"cat {TARGET}", turn=2, start=2, end=3, error=True),
        call("unconfirmed-read", name="remote_read", path=TARGET, turn=2, start=2, end=3),
    ],
)
def test_failed_or_unconfirmed_access_attempt_does_not_claim_localization(case, attempt):
    trajectory = Trajectory(calls=[attempt])
    assert first_localization(trajectory, [TARGET]) is None
    primary, aux = metrics(case, trajectory.calls)
    assert primary["turns_to_first_localization"] is None
    assert primary["time_to_first_localization_sec"] is None
    assert len(aux["target_access_observations"]) == 1
    assert aux["target_access_observations"][0]["call_id"] == attempt.id
    assert aux["first_target_access_attempt_turn"] == 2
    assert aux["first_target_access_attempt_sec"] == 2
    assert aux["first_target_access_attempt_outcome"] in {"failed", "unknown"}
    assert aux["target_access_attempts_by_outcome"] == {
        "success": 0,
        "failed": int(attempt.is_error is True),
        "unknown": int(attempt.is_error is not True),
    }
    assert (
        aux["target_access_metric_reasons"]["turns_to_first_localization"]
        == "only_failed_or_unconfirmed_access_attempts"
    )
    assert (
        aux["target_edit_metric_reasons"]["time_to_first_localization_sec"]
        == "only_failed_or_unconfirmed_access_attempts"
    )


def test_localization_uses_first_successful_access_after_failed_attempt(case):
    calls = [
        call("failed-cat", f"cat {TARGET}", turn=2, start=2, end=3, error=True),
        call("successful-cat", f"cat {TARGET}", turn=4, start=5, end=6),
    ]
    assert first_localization(Trajectory(calls=calls), [TARGET]) == 4
    primary, aux = metrics(case, calls)
    assert primary["turns_to_first_localization"] == 4
    assert primary["time_to_first_localization_sec"] == 5
    assert [row["call_outcome"] for row in aux["target_access_observations"]] == [
        "failed",
        "success",
    ]


def test_legacy_mcp_completion_does_not_prove_an_edit_command_succeeded(case):
    primary, aux = metrics(
        case, [call("untyped", f"sed -i 's/a/b/' {TARGET}", name="remote_bash", turn=3)]
    )
    assert primary["turns_to_first_target_edit"] is None
    assert primary["npu_runs_before_last_target_edit"] is None
    assert aux["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert (
        aux["target_edit_metric_reasons"]["turns_to_last_target_edit"]
        == "only_failed_or_unconfirmed_edit_attempts"
    )


def test_python_file_handle_and_path_aliases_but_not_last_target_mention(case):
    primary, _ = metrics(
        case,
        [
            call(
                "edit",
                f"python3 - <<'PY'\np='{TARGET}'\nwith open(p, 'w') as f:\n    f.write('new')\nPY",
                turn=3,
                start=2,
                end=3,
            ),
            call("mention", f"echo {TARGET}", turn=8, start=10, end=11),
        ],
    )
    assert primary["turns_to_last_target_edit"] == 3


def test_actual_typed_patch_target_is_an_edit_but_reading_patch_text_is_not(case):
    patch = f"*** Begin Patch\n*** Update File: {TARGET}\n@@\n-old\n+new\n*** End Patch"
    edit = call("patch", name="remote_apply_patch", patch=patch, turn=4, start=3, end=4)
    edit.result_observation = ToolResultObservation(
        source="codex_mcp",
        classification="success",
        operation_kind="file",
        classification_reason="explicit_component_success",
    )
    primary, _ = metrics(case, [edit])
    assert primary["turns_to_first_target_edit"] == 4
    assert (
        first_localization(
            Trajectory(
                calls=[
                    call("read-patch", name="remote_read", path="fix.patch", content=patch),
                ]
            ),
            [TARGET],
        )
        is None
    )


def test_union_timing_deduplicates_parent_child_and_cross_component_overlap(case):
    primary, aux = metrics(
        case,
        [
            call("parent", "pytest", name="remote_bash", start=0, end=10),
            call("child", "pytest", name="remote_bash", start=3, end=8),
            call("knowledge", name="knowledge_query", start=7, end=12),
        ],
        wall=15,
    )
    assert primary["npu_execution_wait_sec"] == 10
    assert primary["knowledge_query_sec"] == 5
    assert primary["llm_and_exploration_sec"] == 3
    assert aux["tool_timing"]["remote"]["raw_cumulative_sec"] == 15
    assert aux["tool_timing"]["remote"]["overlap_sec"] == 5
    assert aux["tool_timing"]["combined"]["observed_union_sec"] == 12
    assert aux["tool_timing"]["cross_component_overlap_sec"] == 3


def test_missing_timing_does_not_turn_observed_union_into_complete_coverage(case):
    primary, aux = metrics(
        case,
        [
            call("known", "pytest", name="remote_bash", start=0, end=4),
            call("missing", "pytest", name="remote_bash", start=None, end=6),
        ],
    )
    assert primary["npu_execution_wait_sec"] is None
    assert primary["llm_and_exploration_sec"] is None
    assert aux["tool_timing"]["remote"]["observed_union_sec"] == 4
    assert aux["tool_timing"]["remote"]["missing_intervals"] == 1


def test_poll_wait_is_removed_from_remainder_but_is_not_a_test_attempt(case):
    primary, aux = metrics(
        case,
        [
            call("poll", name="remote_job_stdin", start=2, end=10, job_id="job", chars=""),
            call("query", name="knowledge_query", start=7, end=12),
        ],
        wall=15,
    )
    assert primary["npu_test_runs"] == 0
    assert primary["npu_execution_wait_sec"] == 0
    assert primary["llm_and_exploration_sec"] == 5
    assert aux["tool_timing"]["cross_component_overlap_sec"] == 3
    assert aux["validation_tail"]["total_empty_job_polls"] == 1
    assert aux["validation_tail"]["empty_job_poll_timing"]["observed_union_sec"] == 8


def test_timestamp_outside_wall_clock_is_not_hidden_by_a_short_union(case):
    primary, aux = metrics(case, [call("late", "pytest", name="remote_bash", start=19, end=24)], wall=20)
    assert primary["npu_execution_wait_sec"] == 5
    assert primary["llm_and_exploration_sec"] is None
    assert aux["tool_timing"]["within_wall_clock"] is False
    assert aux["tool_timing"]["remainder_reason"] == "tool_interval_outside_wall_clock"


def test_unresolved_dynamic_program_is_diagnosed_without_inventing_target_access(case):
    primary, aux = metrics(
        case,
        [call("dynamic", "python3 -c \"from pathlib import Path; Path(get_path()).write_text('new')\"")],
    )
    assert primary["turns_to_first_target_edit"] is None
    assert aux["target_access_unresolved"]


def test_post_success_validation_and_empty_poll_are_diagnostics_not_execution_or_stop_rules(case):
    test = call("test", "python3 test_rmsnorm_npu.py", name="remote_bash")
    test.result_observation = ToolResultObservation(
        source="codex_mcp",
        classification="business_success",
        operation_kind="execution",
        exit_code=0,
        classification_reason="explicit_remote_command_exit",
    )
    primary, aux = metrics(
        case,
        [
            test,
            call("empty-poll", name="remote_job_stdin", job_id="job", chars="", turn=2),
            call("later-test", "python3 test_rmsnorm_npu.py", name="remote_bash", turn=3),
        ],
    )
    assert primary["npu_test_runs"] == 2
    assert aux["validation_tail"]["recognized_test_attempts_after_first_recorded_success"] == 1
    assert aux["validation_tail"]["empty_job_polls_after_first_recorded_success"] == 1
    assert aux["validation_tail"]["is_stop_rule"] is False
