import json
from types import SimpleNamespace

import pytest

from mindie_bench.models import Trajectory
from mindie_bench.runner.orchestrator import native_capture_payload


def claude_profile(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "profile"
    projects = root / "claude/projects/owned"
    projects.mkdir(parents=True)
    transcript = projects / "session-1.jsonl"
    transcript.write_text("{}\n")
    event = dict(
        hook_event_name="Stop",
        session_id="session-1",
        prompt_id="native-prompt-1",
        transcript_path=str(transcript),
        cwd=str(workspace),
    )
    (root / "native-stop.json").write_text(json.dumps(event))
    profile = SimpleNamespace(root=root, env={"CLAUDE_CONFIG_DIR": str(root / "claude")})
    return workspace, profile, event


def test_claude_capture_binds_stop_to_actual_session_and_owned_transcript(tmp_path):
    workspace, profile, _ = claude_profile(tmp_path)
    payload = native_capture_payload(
        "claude-code", Trajectory(native_session_id="session-1"), profile, workspace
    )
    assert payload["harness"] == "claude-code"
    assert payload["transcript_path"].endswith("projects/owned/session-1.jsonl")
    assert "turn_id" not in payload


@pytest.mark.parametrize("field", ["session_id", "cwd", "transcript_path"])
def test_claude_capture_rejects_cross_session_or_external_material(tmp_path, field):
    workspace, profile, event = claude_profile(tmp_path)
    if field == "session_id":
        event[field] = "different-session"
    elif field == "cwd":
        event[field] = str(tmp_path)
    else:
        external = tmp_path / "external.jsonl"
        external.write_text("{}\n")
        event[field] = str(external)
    (profile.root / "native-stop.json").write_text(json.dumps(event))
    with pytest.raises(ValueError):
        native_capture_payload(
            "claude-code", Trajectory(native_session_id="session-1"), profile, workspace
        )
