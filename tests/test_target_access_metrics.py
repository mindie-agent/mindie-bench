
import pytest

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.scorer.metrics import (
    target_access_observations,
    trajectory_metrics,
)

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


def pending_edit_job(*, classification="pending"):
    result = call(
        "submitted-edit",
        f"sed -i 's/a/b/' {TARGET}",
        name="remote_bash",
        turn=4,
        start=6,
        end=7,
    )
    result.component = "remote-dev"
    result.result_observation = ToolResultObservation(
        source="codex_mcp",
        result_schema="remote-dev.result.v1",
        classification=classification,
        operation_kind="execution",
        job_id="owned-job",
        target_id="endpoint-one",
        remote_status="running" if classification == "pending" else "unknown",
        exit_code=0,
        classification_reason="explicit_remote_job_pending",
    )
    return result


def terminal_job(identity="confirmed", *, code=0, classification=None, **changes):
    result = call(identity, name="remote_job_status", job_id="owned-job", turn=8, start=15, end=16)
    result.component = "remote-dev"
    result.result_observation = ToolResultObservation(
        source="codex_mcp",
        result_schema="remote-dev.result.v1",
        classification=classification or ("business_failure" if code else "business_success"),
        operation_kind="observation",
        job_id="owned-job",
        target_id="endpoint-one",
        remote_status="completed",
        exit_code=code,
        classification_reason="explicit_remote_command_exit",
    ).model_copy(update=changes)
    return result


def test_later_same_job_terminal_success_confirms_edit_without_rewriting_original_call(case):
    submitted, confirmed = pending_edit_job(), terminal_job()
    primary, auxiliary = metrics(case, [submitted, confirmed])
    assert primary["turns_to_first_target_edit"] == 4
    assert primary["time_to_first_target_edit_sec"] == 6
    assert primary["turns_to_last_target_edit"] == 4
    assert primary["first_source_evidence_turn"] == 8
    assert primary["first_source_evidence_sec"] == 16
    evidence = auxiliary["target_access_observations"][0]["outcome_evidence"]
    assert evidence["basis"] == "later_same_remote_job_terminal"
    assert evidence["confirmation_call_ids"] == ["confirmed"]
    assert evidence["job_id"] == "owned-job"
    assert evidence["target_id"] == "endpoint-one"
    assert evidence["edit_time_basis"] == "original_command_started_sec_proxy"
    assert evidence["confirmation_turn"] == 8
    assert evidence["confirmation_ended_sec"] == 16
    assert auxiliary["target_access_observations"][0]["ended_sec"] == 7
    assert submitted.result_observation.classification == "pending"
    assert submitted.result_observation.remote_status == "running"


def test_confirmed_job_without_confirmation_time_does_not_reuse_submission_end(case):
    submitted, confirmed = pending_edit_job(), terminal_job()
    confirmed.started_sec = confirmed.ended_sec = None
    primary, auxiliary = metrics(case, [submitted, confirmed])
    assert primary["time_to_first_target_edit_sec"] == 6
    assert primary["first_source_evidence_turn"] == 8
    assert primary["first_source_evidence_sec"] is None
    assert auxiliary["source_evidence_basis"]["reasons"]["first_source_evidence_sec"] == (
        "source_evidence_output_or_end_time_missing"
    )


def test_later_same_job_failure_confirms_failure_not_initial_pending_exit_zero(case):
    submitted, confirmed = pending_edit_job(), terminal_job(code=2)
    primary, auxiliary = metrics(case, [submitted, confirmed])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["failed"] == 1
    evidence = auxiliary["target_access_observations"][0]["outcome_evidence"]
    assert evidence["confirmed_exit_code"] == 2
    assert evidence["original_exit_code"] == 0
    assert submitted.result_observation.classification == "pending"


def test_original_unknown_is_not_promoted_by_a_later_successful_poll(case):
    submitted = pending_edit_job(classification="unknown")
    primary, auxiliary = metrics(case, [submitted, terminal_job()])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert submitted.result_observation.classification == "unknown"


@pytest.mark.parametrize(
    "observations",
    [
        [terminal_job("success"), terminal_job("failure", code=2)],
        [terminal_job("unknown", classification="unknown"), terminal_job("success")],
        [
            terminal_job(
                "conflict",
                classification="protocol_error",
                classification_reason="conflicting_exit_codes",
            ),
            terminal_job("success"),
        ],
    ],
)
def test_uncertain_or_conflicting_same_job_results_do_not_confirm_success(case, observations):
    primary, auxiliary = metrics(case, [pending_edit_job(), *observations])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert auxiliary["target_access_observations"][0]["outcome_evidence"]["reason"] in {
        "conflicting_job_terminal_results",
        "uncertain_job_result",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"target_id": "different-endpoint"},
        {"target_id": None},
        {"job_id": "different-job"},
        {"classification": "success", "exit_code": None},
        {"remote_status": "running"},
        {"operation_kind": "control"},
    ],
)
def test_missing_or_nonterminal_or_different_job_evidence_does_not_join(case, changes):
    primary, auxiliary = metrics(case, [pending_edit_job(), terminal_job(**changes)])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1


def test_an_earlier_status_or_another_component_is_not_a_later_confirmation(case):
    earlier, submitted = terminal_job(), pending_edit_job()
    other = terminal_job("other")
    other.component = "other-component"
    primary, auxiliary = metrics(case, [earlier, submitted, other])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1


def test_a_reused_job_identity_for_multiple_execution_calls_is_ambiguous(case):
    first, second = pending_edit_job(), pending_edit_job()
    second.id = "second-submission"
    primary, auxiliary = metrics(case, [first, second, terminal_job()])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 2


@pytest.mark.parametrize("name", ["remote_job_stdin", "remote_job_status", "remote_job_tail"])
@pytest.mark.parametrize(
    "arguments",
    [
        {"session_id": "owned-job"},
        {
            "job_id": "owned-job",
            "session_id": "owned-job",
        },
    ],
)
@pytest.mark.parametrize("code", [0, 2])
def test_terminal_job_argument_aliases_join_only_the_same_result_job(case, name, arguments, code):
    submitted, confirmed = pending_edit_job(), terminal_job(code=code)
    confirmed.name, confirmed.arguments = name, dict(arguments)
    primary, auxiliary = metrics(case, [submitted, confirmed])
    observation = auxiliary["target_access_observations"][0]
    assert observation["call_outcome"] == ("success" if code == 0 else "failed")
    assert observation["outcome_evidence"]["confirmed_exit_code"] == code
    assert confirmed.arguments == arguments
    assert submitted.result_observation.classification == "pending"
    assert primary["turns_to_first_target_edit"] == (4 if code == 0 else None)


@pytest.mark.parametrize(
    "arguments",
    [
        {"job_id": "owned-job", "session_id": "different-job"},
        {"job_id": "different-job", "session_id": "owned-job"},
        {"session_id": "different-job"},
        {"job_id": "owned-job", "session_id": None},
        {"session_id": ""},
        {"session_id": ["owned-job"]},
    ],
)
def test_conflicting_or_wrong_alias_cannot_be_hidden_by_another_success(case, arguments):
    invalid = terminal_job("invalid")
    invalid.name, invalid.arguments = "remote_job_stdin", dict(arguments)
    primary, auxiliary = metrics(case, [pending_edit_job(), invalid, terminal_job("valid")])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_access_observations"][0]["call_outcome"] == "unknown"


def test_session_alias_cannot_promote_original_unknown(case):
    submitted, confirmed = pending_edit_job(classification="unknown"), terminal_job()
    confirmed.name, confirmed.arguments = "remote_job_stdin", {"session_id": "owned-job"}
    primary, _ = metrics(case, [submitted, confirmed])
    assert primary["turns_to_first_target_edit"] is None
    assert submitted.result_observation.classification == "unknown"


@pytest.mark.parametrize(
    ("command", "recognized", "programs", "unresolved"),
    [
        ("echo 'pytest test_npu.py'", False, [], []),
        ("python3 -m py_compile test_npu.py", False, [], []),
        (
            "cat <<'PATCH'\npytest -q tests/test_npu.py\nPATCH",
            False,
            [],
            [],
        ),
        (
            "python3 - <<'PY'\nimport subprocess\nsubprocess.run(['pytest'])\nPY",
            False,
            [],
            ["python_inline_or_stdin_not_inspected"],
        ),
        (
            "bash -lc 'python3 -m pytest tests/test_npu.py'",
            True,
            ["python3 -m pytest"],
            [],
        ),
        ("$RUNNER test_npu.py", False, [], ["dynamic_program_not_inspected"]),
    ],
)
def test_npu_count_uses_command_positions_and_retains_opaque_uncertainty(
    case, command, recognized, programs, unresolved
):
    primary, auxiliary = metrics(case, [call("submitted", command, name="remote_bash")])
    assert primary["npu_test_runs"] == int(recognized)
    assert primary["remote_test_invocations"] == int(recognized)
    observed = auxiliary["npu_command_observations"]
    assert observed["basis"] == "literal_command_position/1"
    assert "not complete hardware" in observed["scope"]
    assert observed["calls"] == [
        dict(
            call_id="submitted",
            recognized=recognized,
            matched_programs=programs,
            unresolved_reasons=unresolved,
        )
    ]
    assert "command" not in observed["calls"][0]


@pytest.mark.parametrize(
    ("command", "expected_kind"),
    [
        (f"python3 -c \"from pathlib import Path; Path('{TARGET}').open('r').read()\"", "read"),
        (f"python3 -c \"from pathlib import Path; Path('{TARGET}').open(mode='w').write('x')\"", "edit"),
        (
            f"python3 -c \"from pathlib import Path; Path('tmp.py').replace(Path('{TARGET}'))\"",
            "edit",
        ),
        (f"python3 -c \"import os; os.replace('tmp.py', '{TARGET}')\"", "edit"),
        (f"python3 -c \"import os; os.rename('{TARGET}', 'backup.py')\"", "edit"),
        (f"python3 -c \"from pathlib import Path; Path('{TARGET}').rename('backup.py')\"", "edit"),
        (f"python3 -c \"import shutil; shutil.copy2('tmp.py', '{TARGET}')\"", "edit"),
        (f"tee {TARGET} </dev/null", "edit"),
        (f"cp -f /tmp/staged.py {TARGET}", "edit"),
        (f"mv {TARGET} /tmp/backup.py", "edit"),
        (f"perl -pi -e 's/old/new/g' {TARGET}", "edit"),
    ],
)
def test_python_and_shell_atomic_replacement_and_path_open_access(command, expected_kind):
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("access", command, turn=2)]), [TARGET]
    )
    assert [row["kind"] for row in found] == [expected_kind]
    assert unresolved == []


def test_string_replacement_of_a_filename_is_not_file_access():
    command = f"python3 -c \"name='{TARGET}'; print(name.replace('ops', 'layers'))\""
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("mention", command)]), [TARGET]
    )
    assert found == []
    assert unresolved == []


@pytest.mark.parametrize(
    "command",
    [
        f"grep -E -n 'Ascend(RMS|Layer)Norm' {TARGET}",
        f"sed -E -n '1,5p' {TARGET}",
        f"perl -E 'say for <>' {TARGET}",
    ],
)
def test_extended_expression_flags_do_not_turn_patterns_into_paths(command):
    found, unresolved = target_access_observations(Trajectory(calls=[call("read", command)]), [TARGET])
    assert [row["kind"] for row in found] == ["read"]
    assert unresolved == []


def test_rsi_reuse_to_explore_observations_and_g4_group(case):
    from mindie_bench.models import Group

    assert Group.G4.value == "G4"
    reuse_call = call(
        "k1",
        name="knowledge_query",
        turn=1,
        start=1,
        end=2,
        query="SparseMLA QLI-v2",
        mode="reuse",
    )
    explore_call = call(
        "k2",
        name="knowledge_query",
        turn=2,
        start=3,
        end=4,
        query="SparseMLA QLI-v2 mutator",
        mode="explore",
        bottleneck_summary={"layer": "SparseMLA", "tensor_moves": 3},
        goal="reduce TPOT below 18 ms",
    )
    _primary, auxiliary = metrics(case, [reuse_call, explore_call])
    obs = auxiliary["rsi_exploration"]
    assert obs["transitioned_reuse_to_explore"] is True
    assert obs["reuse_calls_count"] == 1
    assert obs["explore_calls_count"] == 1
    assert obs["mode_sequence"] == ["reuse", "explore"]
    assert obs["explore_calls"][0]["has_bottleneck_summary"] is True
    assert obs["explore_calls"][0]["goal"] == "reduce TPOT below 18 ms"


