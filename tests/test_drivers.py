from mindie_bench.runner.drivers import normalize_tool
from mindie_bench.runner.drivers.claude_code import ClaudeCodeDriver
from mindie_bench.runner.drivers.codex import CodexDriver


def test_codex_item_updates_are_not_duplicate_tools_or_tokens():
    driver = CodexDriver()
    item = dict(
        id="cmd1", type="command_execution", command="ssh bench-slot pytest", status="in_progress"
    )
    driver.consume(dict(type="item.started", item=item), 1, 2)
    driver.consume(dict(type="item.completed", item=dict(item, status="completed", exit_code=0)), 4, 2)
    driver.consume(
        dict(
            type="turn.completed", usage=dict(input_tokens=100, cached_input_tokens=80, output_tokens=10)
        ),
        5,
        2,
    )
    trajectory = driver.trajectory()
    assert len(trajectory.calls) == 1
    assert trajectory.calls[0].turn == 2
    assert trajectory.tokens.input_uncached == 20
    assert trajectory.tokens.input_cached == 80
    assert trajectory.tokens.complete


def test_codex_native_failure_is_visible():
    driver = CodexDriver()
    driver.consume({"type": "turn.failed", "error": {"message": "gateway timeout"}}, 1, 1)
    assert driver.trajectory().native_error
    assert not driver.trajectory().native_completed


def test_claude_repeated_message_usage_is_deduplicated():
    driver = ClaudeCodeDriver()
    message = dict(
        id="m1",
        content=[dict(type="tool_use", id="c1", name="Read", input={"file_path": "module.py"})],
        usage=dict(input_tokens=10, cache_read_input_tokens=20, output_tokens=2),
    )
    driver.consume(dict(type="assistant", message=message), 0, 1)
    driver.consume(dict(type="assistant", message=message), 0.1, 1)
    driver.consume(
        dict(
            type="user",
            message=dict(content=[dict(type="tool_result", tool_use_id="c1", is_error=False)]),
        ),
        2,
        1,
    )
    trajectory = driver.trajectory()
    assert len(trajectory.calls) == 1
    assert trajectory.tokens.input_uncached == 10
    assert trajectory.calls[0].ended_sec == 2
    assert not trajectory.tokens.complete


def test_claude_result_usage_is_authoritative():
    driver = ClaudeCodeDriver()
    driver.consume(
        dict(
            type="result",
            subtype="success",
            usage=dict(
                input_tokens=11,
                cache_read_input_tokens=30,
                cache_creation_input_tokens=7,
                output_tokens=4,
            ),
        ),
        5,
        3,
    )
    trajectory = driver.trajectory()
    assert trajectory.native_completed
    assert trajectory.tokens.input_uncached == 18
    assert trajectory.tokens.input_cached == 30
    assert trajectory.tokens.output == 4
    assert trajectory.tokens.complete


def test_native_mcp_names_preserve_component_attribution():
    components = {"remote": "remote-dev", "knowledge": "knowledge"}
    assert normalize_tool("mcp__remote__bash", None, components) == ("remote_bash", "remote-dev")
    assert normalize_tool("query", "knowledge", components) == ("knowledge_query", "knowledge")
    assert normalize_tool("query", "unrelated", components) == ("query", None)


def test_claude_preserves_distinct_public_blocks_with_shared_message_id():
    driver = ClaudeCodeDriver()
    for identity, text in [("block1", "first"), ("block2", "second"), ("block2", "second")]:
        driver.consume(
            {
                "type": "assistant",
                "uuid": identity,
                "message": {"id": "shared", "content": [{"type": "text", "text": text}]},
            },
            1,
            1,
        )
    assert [m["text"] for m in driver.trajectory().public_messages] == ["first", "second"]
