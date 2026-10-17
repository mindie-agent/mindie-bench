import json

import pytest

from mindie_bench.runner.drivers.claude_code import ClaudeCodeDriver
from mindie_bench.runner.drivers.codex import CodexDriver


def remote_result(**fields):
    return {
        "schema_version": "remote-dev.result.v1",
        "tool": "remote.job_stdin",
        "invocation_id": "invocation-1",
        "target": {"endpoint_id": "endpoint-1"},
        "outcome": "success",
        "status": "running",
        "job_id": "job-1",
        "state": "running",
        "exit_code": None,
        **fields,
    }


def codex_call(result, *, tool="remote_job_stdin", status="completed", error=None):
    driver = CodexDriver()
    driver.server_components = {"remote": "remote-dev"}
    item = {
        "id": "item_18",
        "type": "mcp_tool_call",
        "server": "remote",
        "tool": tool,
        "arguments": {"session_id": "job-1", "chars": ""},
    }
    driver.consume({"type": "item.started", "item": item}, 1, 1)
    driver.consume(
        {
            "type": "item.completed",
            "item": {**item, "status": status, "result": result, "error": error},
        },
        2,
        1,
    )
    return driver.trajectory().calls[0]


def test_codex_successful_status_request_preserves_failed_job():
    # remote.job_status itself succeeds while reporting a failed remote command.
    result = remote_result(
        tool="remote.job_status",
        state=None,
        exit_code=None,
        status="failed",
        job={
            "job_id": "job-1",
            "remote_status": {"state": "failed", "result": {"exit_code": 1}},
        },
    )
    call = codex_call({"structured_content": result}, tool="remote_job_status")
    assert call.is_error is False
    assert call.result_observation.classification == "business_failure"
    assert call.result_observation.exit_code == 1
    assert call.result_observation.operation_kind == "observation"


@pytest.mark.parametrize("tool", ["remote_job_start", "remote_bash", "remote_job_stdin"])
def test_successful_background_admission_is_pending_not_command_success(tool):
    call = codex_call({"structured_content": remote_result()}, tool=tool)
    assert call.result_observation.classification == "pending"
    assert call.result_observation.exit_code is None


def test_explicit_unknown_wins_over_known_exit_zero():
    call = codex_call(
        {
            "structured_content": remote_result(
                state="unknown",
                exit_code=0,
                error_details={"category": "rpc_disconnected", "submission_state": "uncertain"},
            )
        }
    )
    observed = call.result_observation
    assert observed.classification == "unknown"
    assert observed.exit_code == 0
    assert observed.error_domain == "transport"
    assert observed.submission_state == "uncertain"


def test_claude_remote_business_exit_from_real_mcp_text_envelope():
    driver = ClaudeCodeDriver()
    driver.server_components = {"remote": "remote-dev"}
    driver.consume(
        {
            "type": "assistant",
            "message": {
                "id": "assistant-1",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "mcp__remote__remote_job_stdin",
                        "input": {"session_id": "job-1", "chars": ""},
                    }
                ],
            },
        },
        1,
        1,
    )
    driver.consume(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "is_error": True,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "structuredContent": remote_result(
                                            outcome="failed",
                                            status="failed",
                                            state="failed",
                                            exit_code=1,
                                        ),
                                        "isError": True,
                                    }
                                ),
                            }
                        ],
                    }
                ]
            },
        },
        2,
        1,
    )
    call = driver.trajectory().calls[0]
    assert call.is_error is True
    assert call.result_observation.classification == "business_failure"
    assert call.result_observation.mcp_is_error is True


def test_unstructured_error_does_not_invent_component_attribution():
    call = codex_call(
        {"content": [{"type": "text", "text": "tool failed without structured details"}]},
        status="failed",
    )
    assert call.is_error is True
    assert call.result_observation.classification == "unclassified"
    assert call.result_observation.error_domain is None


def test_normalized_observation_does_not_copy_tool_or_command_content():
    result = remote_result(state="failed", status="failed", outcome="failed", exit_code=1)
    result["preview"] = {"stdout": "PRIVATE_TOOL_OUTPUT", "stderr": "PRIVATE_TOOL_ERROR"}
    call = codex_call({"structured_content": result})
    serialized = call.result_observation.model_dump_json()
    assert "PRIVATE_TOOL_OUTPUT" not in serialized
    assert "PRIVATE_TOOL_ERROR" not in serialized
    assert len(call.result_observation.result_sha256) == 64


def test_claude_attached_result_with_a_different_tool_id_is_not_borrowed():
    driver = ClaudeCodeDriver()
    driver.server_components = {"remote": "remote-dev"}
    driver.consume(
        {
            "type": "assistant",
            "message": {
                "id": "assistant-1",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "mcp__remote__remote_job_status",
                        "input": {"job_id": "job-1"},
                    }
                ],
            },
        },
        1,
        1,
    )
    driver.consume(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "unstructured response",
                    }
                ]
            },
            "tool_use_result": {
                "tool_use_id": "different-tool",
                **remote_result(state="succeeded", exit_code=0),
            },
        },
        2,
        1,
    )
    observation = driver.trajectory().calls[0].result_observation
    assert observation.classification == "unclassified"
    assert observation.exit_code is None


def test_nested_unknown_is_preserved_even_with_a_top_level_running_state():
    call = codex_call(
        {
            "structured_content": remote_result(
                state="running",
                exit_code=0,
                job={"remote_status": {"state": "unknown", "result": {"exit_code": 0}}},
            ),
        }
    )
    assert call.result_observation.classification == "unknown"
    assert call.result_observation.exit_code == 0


def test_codex_native_knowledge_completion_is_call_success_without_inventing_hits():
    driver = CodexDriver()
    driver.server_components = {"knowledge": "knowledge"}
    driver.consume(
        {
            "type": "item.completed",
            "item": {
                "id": "query-1",
                "type": "mcp_tool_call",
                "server": "knowledge",
                "tool": "knowledge_query",
                "arguments": {"query": "some query"},
                "status": "completed",
                "result": {"content": [{"type": "text", "text": '{"results": []}'}]},
                "error": None,
            },
        },
        1,
        1,
    )
    observation = driver.trajectory().calls[0].result_observation
    assert observation.classification == "success"
    assert observation.classification_reason == "explicit_native_mcp_completion"
    assert observation.mcp_is_error is None


@pytest.mark.parametrize(
    ("result", "error", "reason"),
    [
        ({"structured_content": []}, None, "invalid_structured_result"),
        (None, {"code": -32602, "message": "private parameters"}, "jsonrpc_protocol_error"),
        (
            {
                "structured_content": remote_result(
                    state="succeeded",
                    exit_code=0,
                    job={"remote_status": {"state": "failed", "result": {"exit_code": 1}}},
                )
            },
            None,
            "conflicting_exit_codes",
        ),
    ],
)
def test_explicit_malformed_or_conflicting_protocol_facts_remain_errors(result, error, reason):
    observed = codex_call(result, error=error).result_observation
    assert observed.classification == "protocol_error"
    assert observed.error_domain == "protocol"
    assert observed.classification_reason == reason
    assert "private parameters" not in observed.model_dump_json()


def test_unknown_native_error_is_not_overridden_by_a_false_mcp_error_flag():
    driver = CodexDriver()
    driver.server_components = {"knowledge": "knowledge"}
    driver.consume(
        {
            "type": "item.completed",
            "item": {
                "id": "query-1",
                "type": "mcp_tool_call",
                "server": "knowledge",
                "tool": "knowledge_query",
                "status": "failed",
                "error": {"message": "opaque native failure"},
                "result": {"isError": False},
            },
        },
        1,
        1,
    )
    call = driver.trajectory().calls[0]
    assert call.is_error is True
    assert call.result_observation.classification == "unclassified"
    assert call.result_observation.error_domain is None


@pytest.mark.parametrize(
    ("name", "arguments", "attached"),
    [
        (
            "Read",
            {"file_path": "target.py"},
            {"type": "text", "file": {"filePath": "target.py", "content": "actual source bytes"}},
        ),
        (
            "Bash",
            {"command": "cat target.py"},
            {"stdout": "actual source bytes", "stderr": "", "interrupted": False},
        ),
    ],
)
def test_successful_claude_native_source_access_keeps_successful_localization(name, arguments, attached):
    from mindie_bench.scorer.metrics import first_localization

    driver = ClaudeCodeDriver()
    driver.consume(
        {
            "type": "assistant",
            "message": {
                "id": "native-assistant",
                "content": [{"type": "tool_use", "id": "native-read", "name": name, "input": arguments}],
            },
        },
        1,
        2,
    )
    driver.consume(
        {
            "type": "user",
            "tool_use_result": attached,
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "native-read",
                        "is_error": False,
                        "content": "actual source bytes",
                    }
                ]
            },
        },
        2,
        2,
    )
    trajectory = driver.trajectory()
    observed = trajectory.calls[0].result_observation
    assert observed.classification == "success"
    assert observed.classification_reason == "explicit_native_tool_success"
    assert first_localization(trajectory, ["target.py"]) == 2


@pytest.mark.parametrize(
    ("attached", "classification"),
    [
        ({"exit_code": 1}, "business_failure"),
        ({"exit_code": 0, "status": "running"}, "pending"),
        ({"exit_code": 0, "status": "pending"}, "pending"),
        ({"exit_code": 0, "status": "unknown"}, "unknown"),
        ({"structuredContent": []}, "protocol_error"),
        ({"exit_code": False}, "protocol_error"),
    ],
)
def test_claude_native_false_error_bit_does_not_override_explicit_result_facts(attached, classification):
    from mindie_bench.models import ToolCall
    from mindie_bench.scorer.tool_observations import observe_claude_result

    call = ToolCall(id="native-shell", name="Bash", turn=1, is_error=False)
    observed = observe_claude_result(
        {"type": "tool_result", "tool_use_id": call.id, "is_error": False, "content": ""},
        call,
        attached,
    )
    assert observed.classification == classification


def test_stop_result_reads_actual_top_level_remote_status_command_exit():
    observed = codex_call(
        {
            "structured_content": remote_result(
                tool="remote.job_stop",
                state=None,
                exit_code=None,
                status="failed",
                remote_status={"state": "failed", "result": {"exit_code": -15}},
            )
        },
        tool="remote_job_stop",
    ).result_observation
    assert observed.classification == "business_failure"
    assert observed.exit_code == -15
    assert observed.operation_kind == "control"


def test_successful_tail_reports_call_success_without_inventing_a_command_exit():
    observed = codex_call(
        {
            "structured_content": remote_result(
                tool="remote.job_tail",
                state="failed",
                exit_code=None,
                status="ok",
            )
        },
        tool="remote_job_tail",
    ).result_observation
    assert observed.classification == "success"
    assert observed.remote_status == "failed"
    assert observed.exit_code is None
