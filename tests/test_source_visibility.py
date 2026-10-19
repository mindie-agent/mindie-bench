import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from mindie_bench.evaluation.source_visibility import observe_source_visibility
from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.scorer.metrics import trajectory_metrics


def native_file(path, events):
    path.write_text(
        "".join(
            json.dumps({"observed_sec": observed, "native_line": json.dumps(event)}) + "\n"
            for observed, event in events
        )
    )
    return path


def pending_read():
    return ToolCall(
        id="read",
        name="remote_bash",
        component="remote-dev",
        arguments={"command": "cat module.py"},
        turn=3,
        started_sec=2,
        ended_sec=3,
        result_observation=ToolResultObservation(
            classification="pending",
            source="codex_mcp",
            operation_kind="execution",
            classification_reason="running",
        ),
    )


def result(text, *, identity="read"):
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "id": identity,
            "status": "completed",
            "result": {
                "structured_content": {
                    "status": "running",
                    "preview": {"stdout": text},
                    "exit_code": None,
                }
            },
        },
    }


def observe(case, path, trajectory):
    return observe_source_visibility(path, trajectory, case["spec"].environment["l2"], ["module.py"])


def test_pending_job_can_already_return_exact_initial_source_without_reclassifying(case, tmp_path):
    trajectory = Trajectory(calls=[pending_read()])
    native = native_file(tmp_path / "events.jsonl", [(3, result("prefix\nVALUE = 1\nsuffix"))])
    before = trajectory.model_dump()
    observed = observe(case, native, trajectory)
    primary, auxiliary = trajectory_metrics(
        trajectory,
        case["spec"],
        case["config"].harness,
        10,
        tier="l2",
        source_visibility=observed,
    )
    assert primary["turns_to_first_localization"] is None
    assert primary["first_verified_source_output_turn"] == 3
    assert primary["first_verified_source_output_sec"] == 3
    assert primary["first_source_evidence_turn"] == 3
    assert primary["first_source_evidence_sec"] == 3
    witness = auxiliary["source_visibility"]["witnesses"][0]
    assert witness["source_bytes"] == len(b"VALUE = 1\n")
    assert witness["source_sha256"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert witness["output_byte_offset"] == len(b"prefix\n")
    assert witness["call_classification"] == "pending"
    assert witness["native_event_line"] == 1
    assert trajectory.model_dump() == before


def test_only_initial_git_blob_and_actual_tool_output_are_evidence(case, tmp_path):
    repo = case["spec"].environment["l2"].repo_path
    # Even if the final workspace contains repaired bytes, they cannot become
    # evidence for the initial source visible to the model.
    (Path(repo) / "module.py").write_text("VALUE = 2\n")
    events = [
        (1, {"type": "item.completed", "item": {"type": "agent_message", "text": "VALUE = 1\n"}}),
        (
            2,
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "id": "read",
                    "command": "echo 'VALUE = 1\n'",
                    "aggregated_output": "VALUE = 2\n",
                },
            },
        ),
        (3, result("VALUE = 1")),  # Partial content is not a whole-blob match.
    ]
    observed = observe(
        case, native_file(tmp_path / "events.jsonl", events), Trajectory(calls=[pending_read()])
    )
    assert observed["witnesses"] == []
    assert observed["state"] == "no_exact_output_match"


def test_combined_evidence_uses_completion_time_and_does_not_replace_legacy_start(case, tmp_path):
    call = pending_read().model_copy(
        update={
            "id": "success",
            "turn": 8,
            "started_sec": 1,
            "ended_sec": 7,
            "result_observation": None,
            "is_error": False,
            "name": "bash",
            "component": None,
        }
    )
    trajectory = Trajectory(calls=[pending_read(), call])
    observed = observe(
        case, native_file(tmp_path / "events.jsonl", [(3, result("VALUE = 1\n"))]), trajectory
    )
    primary, aux = trajectory_metrics(
        trajectory, case["spec"], case["config"].harness, 10, tier="l2", source_visibility=observed
    )
    assert primary["time_to_first_localization_sec"] == 1
    assert primary["first_source_evidence_sec"] == 3
    assert primary["first_source_evidence_turn"] == 3
    assert aux["source_evidence_basis"]["schema"] == "mindie-bench-source-evidence/1"


def test_missing_output_clock_never_uses_call_start(case, tmp_path):
    call = pending_read().model_copy(update={"ended_sec": None})
    path = native_file(tmp_path / "events.jsonl", [(None, result("VALUE = 1\n"))])
    observed = observe(case, path, Trajectory(calls=[call]))
    assert observed["first_verified_source_output_turn"] == 3
    assert observed["first_verified_source_output_sec"] is None
    assert observed["reasons"]["first_verified_source_output_sec"] == "output_observation_time_missing"


def test_native_output_without_correlated_call_does_not_guess_turn(case, tmp_path):
    path = native_file(tmp_path / "events.jsonl", [(4, result("VALUE = 1\n", identity="unmapped"))])
    observed = observe(case, path, Trajectory())
    assert observed["first_verified_source_output_turn"] is None
    assert observed["first_verified_source_output_sec"] == 4
    assert observed["reasons"]["first_verified_source_output_turn"] == "output_call_not_in_trajectory"


def test_claude_tool_result_is_visible_even_when_result_is_an_error(case, tmp_path):
    event = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "read",
                    "is_error": True,
                    "content": [{"type": "text", "text": "VALUE = 1\n"}],
                }
            ]
        },
    }
    observed = observe(
        case, native_file(tmp_path / "events.jsonl", [(3, event)]), Trajectory(calls=[pending_read()])
    )
    assert observed["first_verified_source_output_turn"] == 3
    assert observed["witnesses"][0]["output_field"] == "message.content[0].content[0].text"


def test_successful_access_missing_end_does_not_use_start_for_combined_time(case, tmp_path):
    call = pending_read().model_copy(
        update={
            "name": "bash",
            "component": None,
            "is_error": False,
            "result_observation": None,
            "ended_sec": None,
        }
    )
    primary, aux = trajectory_metrics(
        Trajectory(calls=[call]),
        case["spec"],
        case["config"].harness,
        10,
        tier="l2",
    )
    assert primary["time_to_first_localization_sec"] == 2
    assert primary["first_source_evidence_sec"] is None
    assert (
        aux["source_evidence_basis"]["reasons"]["first_source_evidence_sec"]
        == "source_evidence_output_or_end_time_missing"
    )


def test_git_or_native_parse_failure_is_not_reported_as_no_output(case, tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("not json\n")
    with pytest.raises(ValueError, match="native event line 1"):
        observe(case, path, Trajectory())
    path = native_file(path, [])
    environment = case["spec"].environment["l2"].model_copy(update={"base_commit": "0" * 40})
    with pytest.raises(subprocess.CalledProcessError):
        observe_source_visibility(path, Trajectory(), environment, ["module.py"])


def test_declared_public_asset_uses_original_bytes_and_explicit_non_git_provenance(case, tmp_path):
    public = tmp_path / "task-public"
    public.mkdir()
    initial = b"OVERRIDE = 'legacy'\n"
    (public / "deployment.py").write_bytes(initial)
    # A final file in the source checkout is never the public asset's initial value.
    (Path(case["spec"].environment["l2"].repo_path) / "deployment.py").write_text("REPAIRED = True\n")
    native = native_file(
        tmp_path / "events.jsonl",
        [(2, result("REPAIRED = True\n")), (3, result(initial.decode()))],
    )
    observed = observe_source_visibility(
        native,
        Trajectory(calls=[pending_read()]),
        case["spec"].environment["l2"],
        ["deployment.py"],
        public_root=public,
    )
    assert len(observed["witnesses"]) == 1
    witness = observed["witnesses"][0]
    assert witness["native_event_line"] == 2
    assert witness["source_origin"] == "task_public_asset"
    assert witness["git_blob_id"] is None
    assert witness["source_sha256"] == hashlib.sha256(initial).hexdigest()
    assert witness["source_bytes"] == len(initial)


def test_public_asset_missing_never_becomes_initial_source(case, tmp_path):
    public = tmp_path / "task-public"
    public.mkdir()
    with pytest.raises(ValueError, match="public asset"):
        observe_source_visibility(
            native_file(tmp_path / "events.jsonl", []),
            Trajectory(),
            case["spec"].environment["l2"],
            ["deployment.py"],
            public_root=public,
        )


@pytest.mark.parametrize(
    "entry",
    [
        b"100644 blob abc\tdeployment.py\0" * 2,
        b"040000 tree abc\tdeployment.py\0",
        b"120000 blob abc\tdeployment.py\0",
    ],
)
def test_ambiguous_or_nonregular_git_entry_cannot_fall_back_to_public(
    case, tmp_path, monkeypatch, entry
):
    from types import SimpleNamespace

    public = tmp_path / "task-public"
    public.mkdir()
    (public / "deployment.py").write_text("NOT_A_GIT_FALLBACK\n")
    run = subprocess.run

    def selected_run(command, **kwargs):
        if command[3] == "ls-tree":
            return SimpleNamespace(stdout=entry)
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", selected_run)
    with pytest.raises(ValueError, match="Git blob"):
        observe_source_visibility(
            native_file(tmp_path / "events.jsonl", []),
            Trajectory(),
            case["spec"].environment["l2"],
            ["deployment.py"],
            public_root=public,
        )
