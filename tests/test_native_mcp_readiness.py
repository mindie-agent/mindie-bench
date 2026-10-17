"""Actual native startup with scripted local Responses output, never paid inference."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_knowledge_broker import setup_profile

from mindie_bench.models import Group, MCPServer, TerminalStatus
from mindie_bench.runner.orchestrator import run_trial

SERVER = """
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        time.sleep(1.5)
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "slow-synthetic-startup", "version": "1"},
        }
    else:
        result = {"tools": [
            {"name": name, "description": "Synthetic startup contract fixture",
             "inputSchema": {"type": "object", "properties": {}}}
            for name in ["remote_bash", "knowledge_query", "knowledge_explain"]
        ]}
    print(json.dumps({"jsonrpc":"2.0", "id":request["id"], "result":result}), flush=True)
"""


def test_declared_mcp_tools_are_visible_in_the_first_native_model_turn(case, corpus, monkeypatch):
    executable = os.environ.get("MINDIE_BENCH_TEST_CODEX_BIN")
    catalog = os.environ.get("MINDIE_BENCH_TEST_CODEX_CATALOG")
    if not executable or not catalog:
        pytest.skip("select the native Codex binary and its exact code-mode model catalog")
    config, _, _ = setup_profile(case, corpus)
    backend = Path(config.runtime_read_paths[0])
    backend.write_text(SERVER)
    config.groups[Group.G2].mcp_servers["remote"] = MCPServer(
        command=sys.executable,
        args=["-I", str(backend)],
        component="remote-dev",
        required_tools=["remote_bash"],
    )
    config.groups[Group.G2].mcp_servers["knowledge"].args = ["-I", str(backend)]
    config.harness.provider_url = "http://127.0.0.1:1/v1"
    config.harness.api_key_env = "SCRIPTED_NATIVE_KEY"
    config.harness.name = "codex"
    config.harness.executable = executable
    config.harness.expected_version = subprocess.check_output(
        [executable, "--version"], text=True, timeout=10
    ).strip()
    config.harness.model_catalog_path = catalog
    models = json.loads(Path(catalog).read_text())["models"]
    config.harness.model = models[0]["slug"]
    config.component_revisions = {"knowledge": "synthetic-fixture", "remote-dev": "synthetic-fixture"}
    config.backend = "native"
    task_file = case["task"] / "task.toml"
    task_file.write_text(task_file.read_text().replace("timeout_sec = 3", "timeout_sec = 30"))
    monkeypatch.setenv("SCRIPTED_NATIVE_KEY", "zero-paid-fixture")
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if len(requests) == 1:
                item = dict(
                    id="ctc_fixture",
                    type="custom_tool_call",
                    call_id="call_fixture",
                    name="exec",
                    namespace="functions",
                    input="text(ALL_TOOLS.filter(x=>x.name.startsWith('mcp__knowledge__')).map(x=>x.name))",
                    status="completed",
                )
            else:
                item = dict(
                    id="msg_fixture",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        dict(type="output_text", text="Protocol fixture complete.", annotations=[])
                    ],
                )
            response = dict(
                id=f"resp_fixture_{len(requests)}",
                object="response",
                created_at=1792200000,
                status="completed",
                model=request["model"],
                output=[item],
                usage=dict(
                    input_tokens=12,
                    output_tokens=3,
                    total_tokens=15,
                    input_tokens_details=dict(cached_tokens=0),
                ),
            )
            events = [
                dict(type="response.created", response=dict(response, status="in_progress", usage=None)),
                dict(type="response.output_item.done", output_index=0, item=item),
                dict(type="response.completed", response=response),
            ]
            raw = b"".join(("data: " + json.dumps(row) + "\n\n").encode() for row in events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config.harness.provider_url = f"http://127.0.0.1:{server.server_port}/v1"
    output = case["root"] / "native-startup-run"
    output.mkdir()
    try:
        receipt = run_trial(
            case["task"], config, tier="l2", group=Group.G2, repetition=1, run_root=output
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert receipt.terminal_status == TerminalStatus.FAIL, receipt
    assert len(requests) == 2
    tool_outputs = [
        item
        for item in requests[1]["input"]
        if item.get("type") == "custom_tool_call_output" and item.get("call_id") == "call_fixture"
    ]
    assert len(tool_outputs) == 1
    actual = json.dumps(tool_outputs[0])
    assert "mcp__knowledge__knowledge_query" in actual
    assert "mcp__knowledge__knowledge_explain" in actual
    trial = output / receipt.trial_id
    events = [json.loads(line) for line in (trial / "broker.jsonl").read_text().splitlines()]
    session = [row for row in events if row["kind"] == "knowledge_session_open"][-1]["session_id"]
    catalog = next(
        row
        for row in events
        if row["kind"] == "knowledge_catalog_projection"
        and row["session_id"] == session
        and row["method"] == "tools/list"
    )
    first_model = next(row for row in events if row["kind"] == "model_start")
    assert catalog["monotonic_sec"] < first_model["monotonic_sec"]
