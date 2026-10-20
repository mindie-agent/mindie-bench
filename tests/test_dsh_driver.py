import json
from types import SimpleNamespace

import pytest

from mindie_bench.runner.drivers.dsh import DshDriver


def test_dsh_stream_tracks_tool_result_usage_and_completion():
    driver = DshDriver()
    driver.server_components = {"remote": "remote-dev"}
    driver.consume({"type": "session", "sessionId": "session-native-1", "cwd": "/workspace"}, 0, 0)
    driver.consume({"type": "status", "phase": "turn_start", "turn": 1}, 0, 1)
    driver.consume({"type": "status", "phase": "step_start", "turn": 1, "step": 1}, 0, 1)
    driver.consume(
        {
            "type": "tool_call",
            "callId": "call-1",
            "tool": "mcp__remote__remote_bash",
            "input": {"command": "pytest -q"},
        },
        1,
        1,
    )
    driver.consume(
        {
            "type": "tool_result",
            "callId": "call-1",
            "status": "completed",
            "result": json.dumps(
                {
                    "schema_version": "remote-dev.result.v1",
                    "outcome": "success",
                    "status": "succeeded",
                    "operation_completed": True,
                    "exit_code": 0,
                }
            ),
        },
        3,
        1,
    )
    driver.consume(
        {
            "type": "status",
            "phase": "step_end",
            "turn": 1,
            "step": 1,
            "usage": {
                "inputTokens": 10,
                "cacheReadTokens": 4,
                "cacheWriteTokens": 3,
                "outputTokens": 2,
            },
        },
        4,
        1,
    )
    driver.consume({"type": "text", "text": "fixed"}, 4, 1)
    driver.consume(
        {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}}, 5, 1
    )
    driver.consume({"type": "final", "text": "fixed"}, 5, 1)
    trajectory = driver.trajectory()
    assert trajectory.native_session_id == "session-native-1"
    assert trajectory.native_completed
    assert trajectory.total_turns == 1
    assert len(trajectory.calls) == 1
    call = trajectory.calls[0]
    assert call.name == "remote_bash"
    assert call.component == "remote-dev"
    assert call.ended_sec == 3
    assert call.result_observation is not None
    assert call.result_observation.source == "dsh_tool_result"
    assert call.result_observation.classification == "business_success"
    assert trajectory.tokens.input_uncached == 13
    assert trajectory.tokens.input_cached == 4
    assert trajectory.tokens.cache_creation == 3
    assert trajectory.tokens.output == 2
    assert trajectory.tokens.complete
    assert [message["text"] for message in trajectory.public_messages] == ["fixed"]


def test_dsh_error_turn_is_not_reported_as_completed():
    driver = DshDriver()
    driver.consume({"type": "session", "sessionId": "s"}, 0, 0)
    driver.consume(
        {
            "type": "status",
            "phase": "turn_end",
            "turn": 1,
            "reason": {"kind": "error", "error": {"message": "budget exhausted"}},
        },
        1,
        1,
    )
    trajectory = driver.trajectory()
    assert not trajectory.native_completed
    assert trajectory.native_error


def test_dsh_capture_is_unavailable_without_a_real_session_event():
    driver = DshDriver()
    driver.command(
        SimpleNamespace(fingerprint={"server_components": {}}),
        SimpleNamespace(executable="/opt/homebrew/bin/dsh"),
    )
    driver.consume(
        {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
        1,
        1,
    )
    capture = driver.native_session_capture()
    assert capture.status == "unavailable"
    assert capture.session_id is None
    assert capture.capture_ready is False
    assert capture.continuation_command is None
    assert capture.source is None
    assert capture.history_read is False
    assert capture.external_messages_sent is False
    assert driver.trajectory().native_session_id is None


def test_dsh_session_event_drives_continuation_without_replacing_the_launcher():
    driver = DshDriver()
    base = driver.command(
        SimpleNamespace(fingerprint={"server_components": {}}),
        SimpleNamespace(executable="/opt/homebrew/bin/dsh"),
    )
    driver.consume({"type": "session", "sessionId": "session-native-1", "cwd": "/workspace"}, 0, 0)
    incomplete = driver.native_session_capture()
    assert incomplete.status == "incomplete"
    assert incomplete.capture_ready is False
    assert incomplete.continuation_command == (
        "/opt/homebrew/bin/dsh",
        "--profile",
        "bench",
        "--json",
        "--session-id",
        "session-native-1",
    )
    driver.consume(
        {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
        1,
        1,
    )
    capture = driver.native_session_capture()
    assert capture.status == "available"
    assert capture.capture_ready is True
    assert capture.source == "session_event"
    assert capture.session_id == "session-native-1"
    assert capture.continuation_command == (*base, "--session-id", "session-native-1")
    assert any("not a G3 task outcome" in item for item in capture.limitations)
    payload = driver.native_capture_payload()
    assert payload["harness"] == "dsh"
    assert payload["session_id"] == "session-native-1"
    assert payload["turn_id"] == "1"
    assert payload["completed"] is True
    assert payload["cwd"] == "/workspace"
    assert payload["continuation_command"] == [*base, "--session-id", "session-native-1"]
    assert payload["history_read"] is False
    assert payload["transcript_path"] is None
    assert payload["source"] == "headless_json_projection"
    with pytest.raises(ValueError, match="completed native session event"):
        DshDriver().native_capture_payload()
    with pytest.raises(ValueError, match="changed its native session identity"):
        driver.consume({"type": "session", "sessionId": "session-native-2"}, 2, 1)
    malformed = DshDriver()
    malformed.consume({"type": "session", "sessionId": ""}, 0, 0)
    malformed_capture = malformed.native_session_capture()
    assert malformed_capture.status == "unavailable"
    assert malformed_capture.session_id is None
    assert "no valid native session identity" in (malformed_capture.reason or "")


def test_dsh_error_events_and_turn_errors_are_not_success():
    driver = DshDriver()
    driver.consume({"type": "session", "sessionId": "s"}, 0, 0)
    driver.consume({"type": "error", "code": "usage", "message": "provider failed"}, 1, 1)
    assert not driver.trajectory().native_completed
    assert driver.trajectory().native_error == "provider failed"

    driver = DshDriver()
    driver.consume({"type": "session", "sessionId": "s"}, 0, 0)
    driver.consume(
        {
            "type": "status",
            "phase": "turn_end",
            "turn": 1,
            "reason": {
                "kind": "error",
                "error": {"code": "RATE_LIMIT", "message": "retry later"},
            },
        },
        1,
        1,
    )
    trajectory = driver.trajectory()
    assert not trajectory.native_completed
    assert trajectory.native_error == "retry later"


def test_dsh_capture_needs_observed_turn_identity_and_does_not_reuse_prior_completion():
    driver = DshDriver()
    driver.consume({"type": "session", "sessionId": "s"}, 0, 0)
    driver.consume({"type": "status", "phase": "turn_end", "reason": {"kind": "completed"}}, 1, 1)
    assert not driver.native_session_capture().capture_ready
    driver.consume(
        {"type": "status", "phase": "turn_end", "turn": 2, "reason": {"kind": "completed"}}, 2, 2
    )
    assert driver.native_session_capture().capture_ready
    driver.consume({"type": "status", "phase": "turn_start", "turn": 3}, 3, 3)
    assert not driver.native_session_capture().capture_ready
    driver.consume(
        {
            "type": "status",
            "phase": "turn_end",
            "turn": 3,
            "reason": {"kind": "error", "error": "original provider error"},
        },
        4,
        3,
    )
    assert driver.trajectory().native_error == "original provider error"
