import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from test_knowledge_broker import request, setup_profile, start_front, stop_front

from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.knowledge_broker import MAX_FRAME, KnowledgeSession
from mindie_bench.runner.orchestrator import runtime_paths
from mindie_bench.runner.profile_builder import bind_mcp_transport

FASTMCP_SERVER = """
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

server = FastMCP("actual-catalog-regression")

@server.tool()
def knowledge_query(query: str) -> dict:
    return {"query": query, "fixture": "actual-query"}

@server.tool()
def knowledge_explain(ref: str) -> str:
    return "actual-explanation:" + ref

@server.tool()
def private_admin() -> str:
    Path(sys.argv[1]).write_text("UNEXPECTED_BACKEND_EXECUTION")
    return "not part of the knowledge interface"

@server.resource("knowledge://catalog-note")
def catalog_note() -> str:
    Path(sys.argv[1]).write_text("UNEXPECTED_RESOURCE_READ")
    return "resource-body"

@server.prompt()
def catalog_prompt() -> str:
    return "prompt-body"

server.run(transport="stdio")
"""


def actual_mcp_python():
    explicit = os.environ.get("MINDIE_BENCH_TEST_MCP_PYTHON")
    if explicit:
        executable = Path(explicit).absolute()
        subprocess.run(
            [str(executable), "-I", "-c", "from mcp.server.fastmcp import FastMCP"],
            check=True,
            capture_output=True,
        )
        return executable
    pytest.importorskip("mcp.server.fastmcp", reason="requires installed MCP component runtime")
    return Path(sys.executable).absolute()


def test_actual_fastmcp_catalog_and_scoped_denials_keep_knowledge_session_usable(case, corpus):
    executable = actual_mcp_python()
    config, trial, profile = setup_profile(case, corpus)
    backend = Path(config.runtime_read_paths[0])
    backend.write_text(FASTMCP_SERVER)
    sentinel = profile.private_root / "unexpected-backend-operation"
    profile.knowledge_servers["knowledge"].update(
        command=str(executable), args=["-I", str(backend), str(sentinel)]
    )
    config.runtime_read_paths.append(str(executable.parent.parent))
    paths = runtime_paths(config, profile)
    with Gateway(
        config.harness,
        case["spec"],
        case["task"],
        case["repo"],
        trial / "broker.jsonl",
        use_unix_socket=platform.system() == "Linux",
    ) as gateway:
        gateway.attach_knowledge(profile, paths)
        profile.env["MINDIE_BENCH_GATEWAY"] = gateway.url
        bind_mcp_transport(profile)
        front = start_front(profile, paths, trial, gateway)
        try:
            initialized = request(
                front,
                1,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "catalog-regression", "version": "1"},
                },
            )
            front.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            front.stdin.flush()
            resources = request(front, 3, "resources/list")
            assert resources["result"]["resources"][0]["uri"] == "knowledge://catalog-note"
            assert request(front, 4, "resources/templates/list")["result"]["resourceTemplates"] == []
            assert request(front, 5, "prompts/list")["result"]["prompts"][0]["name"] == "catalog_prompt"
            refused = request(front, 6, "resources/read", {"uri": "knowledge://catalog-note"})
            assert refused["error"]["code"] == -32601
            assert refused["error"]["data"]["method"] == "resources/read"
            assert request(front, 7, "private/erase")["error"]["code"] == -32601
            assert (
                request(front, 10, "prompts/get", {"name": "catalog_prompt"})["error"]["code"] == -32601
            )
            blocked_tool = request(front, 8, "tools/call", {"name": "private_admin", "arguments": {}})
            assert blocked_tool["error"]["code"] == -32602
            query = request(
                front,
                9,
                "tools/call",
                {"name": "knowledge_query", "arguments": {"query": "current-evidence"}},
            )
            assert query["result"]["isError"] is False
            assert "actual-query" in json.dumps(query)
            assert set(initialized["result"]["capabilities"]) == {"tools"}
            tools = request(front, 2, "tools/list")
            assert {tool["name"] for tool in tools["result"]["tools"]} == {
                "knowledge_query",
                "knowledge_explain",
            }
            assert not sentinel.exists()
            assert gateway.infrastructure_error is None
            assert gateway.model_calls == 0
        finally:
            stop_front(front)
    rows = [json.loads(line) for line in (trial / "broker.jsonl").read_text().splitlines()]
    projections = [row for row in rows if row["kind"] == "knowledge_catalog_projection"]
    assert {row["method"] for row in projections} == {"initialize", "tools/list"}
    assert all(row["original_sha256"] != row["projected_sha256"] for row in projections)
    denials = [row for row in rows if row["kind"] == "knowledge_request_rejected"]
    assert {row["method"] for row in denials} == {
        "resources/read",
        "private/erase",
        "prompts/get",
        "tools/call",
    }
    assert all(len(row["request_sha256"]) == 64 for row in denials)
    session = next(iter(gateway.knowledge.sessions.values()))
    protocol = [json.loads(line) for line in (session.root / "protocol.jsonl").read_text().splitlines()]
    original_query = next(
        row["frame"]
        for row in protocol
        if row["direction"] == "response" and row["frame"].get("id") == 9
    )
    assert original_query == query
    assert all(
        session.closed and session.process.poll() is not None
        for session in gateway.knowledge.sessions.values()
    )


def observation_session():
    session = object.__new__(KnowledgeSession)
    session.output_buffer = b""
    session.requests = {}
    session.raw = io.BytesIO()
    session.name = "knowledge"
    session.identity = "observation-regression"
    session.record = lambda *args, **kwargs: None
    return session


def test_partial_backend_frames_are_withheld_and_notifications_are_preserved():
    session = observation_session()
    partial = b'{"jsonrpc":"2.0","method":"notifications/tools/'
    assert session._observe(partial) == b""
    tail = b'list_changed","params":{}}\n'
    assert session._observe(tail) == partial + tail
    assert session.output_buffer == b""


def test_partial_backend_frame_limit_stays_fail_closed():
    session = observation_session()
    assert session._observe(b"x" * MAX_FRAME) == b""
    with pytest.raises(ValueError, match="output frame exceeds"):
        session._observe(b"x")


def test_locally_generated_denials_remain_bounded_without_backend_admission(monkeypatch):
    from mindie_bench.runner import knowledge_broker

    session = observation_session()
    session.input_buffer = session.pending = session.local_output = b""
    session.output_bytes = 0
    session.eof = False
    monkeypatch.setattr(knowledge_broker, "MAX_SESSION_BYTES", 1)
    frame = b'{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"private"}}\n'
    with pytest.raises(ValueError, match="session output exceeds"):
        session._submit(frame, False)
    assert not session.requests
    assert session.pending == b""
    records = [json.loads(line) for line in session.raw.getvalue().splitlines()]
    assert records[0] == {
        "direction": "rejected_request",
        "frame": json.loads(frame),
    }
