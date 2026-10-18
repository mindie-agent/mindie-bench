import json

from mindie_bench.runner.drivers.dsh import DshDriver


def test_dsh_stream_tracks_tool_result_usage_and_completion():
    driver = DshDriver()
    driver.server_components = {"remote": "remote-dev"}
    driver.consume(
        {"type": "session", "sessionId": "session-native-1", "cwd": "/workspace"}, 0, 0
    )
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
        {"type": "tool_result", "callId": "call-1", "status": "completed", "result": json.dumps({"schema_version": "remote-dev.result.v1", "outcome": "success", "status": "succeeded", "operation_completed": True, "exit_code": 0})},
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
    driver.consume({"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}}, 5, 1)
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
