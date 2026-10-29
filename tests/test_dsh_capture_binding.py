import json
from types import SimpleNamespace

import pytest

from mindie_bench.runner.drivers.dsh import DshDriver
from mindie_bench.runner.orchestrator import native_capture_payload


def completed_session(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "profile"
    transcript = root / "dsh-home/sessions/project/session-observed/session.v4.jsonl.zstd"
    transcript.parent.mkdir(parents=True)
    # This test checks location/identity binding only, not the compressed decoder.
    transcript.touch()
    profile = SimpleNamespace(root=root, env={"DSH_HOME": str(root / "dsh-home")})
    driver = DshDriver()
    driver.consume({"type": "session", "sessionId": "session-observed", "cwd": str(workspace)}, 0, 0)
    driver.consume(
        {"type": "status", "phase": "turn_end", "turn": 7, "reason": {"kind": "completed"}}, 1, 1
    )
    return profile, workspace, transcript, driver


def test_dsh_metadata_binds_real_session_turn_and_artifact_without_claiming_c6(tmp_path):
    profile, workspace, transcript, driver = completed_session(tmp_path)
    result = native_capture_payload(
        "dsh", driver.trajectory(), profile, workspace, native_descriptor=driver.native_capture_payload()
    )
    assert result["transcript_path"] == str(transcript)
    event = json.loads((profile.root / "native-stop.json").read_text())
    assert event["session_id"] == "session-observed"
    assert event["turn_id"] == "7"
    assert event["event_source"] == "observed_headless_turn_end"
    assert result["capture_backend_status"] == "not_executed"
    assert result["history_read"] is False


def test_long_turn_final_step_can_enter_capture_only_after_native_completion(tmp_path):
    profile, workspace, transcript, _ = completed_session(tmp_path)
    driver = DshDriver()
    driver.consume({"type": "session", "sessionId": "session-observed", "cwd": str(workspace)}, 0, 0)
    driver.consume({"type": "status", "phase": "turn_start", "turn": 7}, 0, 0)
    for step in range(1, 121):
        driver.consume({"type": "status", "phase": "step_end", "turn": 7, "step": step}, step, step)
    # A final text or a locally passing test is not the host's turn-end event.
    driver.consume({"type": "final", "text": "Parser repaired and local checks passed."}, 120, 120)
    assert not driver.native_session_capture().capture_ready
    driver.consume(
        {"type": "status", "phase": "turn_end", "turn": 7, "reason": {"kind": "completed"}},
        120, 120,
    )
    result = native_capture_payload(
        "dsh", driver.trajectory(), profile, workspace, native_descriptor=driver.native_capture_payload()
    )
    assert result["turn_id"] == "7"
    assert result["transcript_path"] == str(transcript)
    assert result["capture_backend_status"] == "not_executed"


def test_dsh_binding_rejects_mismatched_or_ambiguous_native_artifacts(tmp_path):
    profile, workspace, transcript, driver = completed_session(tmp_path)
    descriptor = driver.native_capture_payload()
    descriptor["session_id"] = "another-session"
    with pytest.raises(ValueError, match="this foreground"):
        native_capture_payload(
            "dsh", driver.trajectory(), profile, workspace, native_descriptor=descriptor
        )
    descriptor = driver.native_capture_payload()
    descriptor["cwd"] = str(tmp_path)
    with pytest.raises(ValueError, match="workspace differs"):
        native_capture_payload(
            "dsh", driver.trajectory(), profile, workspace, native_descriptor=descriptor
        )
    duplicate = transcript.parents[2] / "another-project/session-observed/session.v4.jsonl.zstd"
    duplicate.parent.mkdir(parents=True)
    duplicate.touch()
    with pytest.raises(ValueError, match="exactly one"):
        native_capture_payload(
            "dsh",
            driver.trajectory(),
            profile,
            workspace,
            native_descriptor=driver.native_capture_payload(),
        )
