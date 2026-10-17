import base64
import json
import urllib.error
import urllib.request

import pytest
from test_knowledge_broker import setup_profile

from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.orchestrator import runtime_paths


def post(gateway, payload):
    request = urllib.request.Request(
        gateway.url + "/bench/knowledge-stdio",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


@pytest.mark.parametrize(
    "malformed",
    [
        b"Content-Length: 123\r\n\r\n",
        b"\xff\n",
        b"[]\n",
        b'{"jsonrpc":"2.0","method":"initialize","params":[]}\n',
    ],
)
def test_bad_client_frames_are_visible_without_poisoning_a_valid_session(case, corpus, malformed):
    config, trial, profile = setup_profile(case, corpus)
    with Gateway(
        config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
    ) as gateway:
        gateway.attach_knowledge(profile, runtime_paths(config, profile))
        identity = post(gateway, {"action": "open", "server": "knowledge"})["session_id"]
        with pytest.raises(urllib.error.HTTPError) as rejected:
            post(
                gateway,
                {
                    "action": "exchange",
                    "session_id": identity,
                    "stdin_b64": base64.b64encode(malformed).decode(),
                },
            )
        assert rejected.value.code == 400
        assert "invalid" in rejected.value.read().decode().lower()
        assert gateway.infrastructure_error is None
        session = gateway.knowledge.sessions[identity]
        assert session.raw.tell() == 0
        frame = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "knowledge_query", "arguments": {"query": "valid after rejection"}},
        }
        result = post(
            gateway,
            {
                "action": "exchange",
                "session_id": identity,
                "stdin_b64": base64.b64encode((json.dumps(frame) + "\n").encode()).decode(),
            },
        )
        output = base64.b64decode(result["stdout_b64"])
        for _ in range(20):
            if b"\n" in output:
                break
            result = post(gateway, {"action": "exchange", "session_id": identity})
            output += base64.b64decode(result["stdout_b64"])
        response = json.loads(output)
        assert response["id"] == 1
        assert response["result"]["isError"] is False
        assert gateway.infrastructure_error is None
        assert post(gateway, {"action": "close", "session_id": identity}) == {"closed": True}
        assert session.process.poll() == 0
    events = [json.loads(line) for line in (trial / "broker.jsonl").read_text().splitlines()]
    assert len([row for row in events if row["kind"] == "knowledge_client_rejected"]) == 1
    assert not any(row["kind"] == "knowledge_error" for row in events)


def test_invalid_http_request_shape_is_a_client_error_before_backend_admission(case, corpus):
    config, trial, profile = setup_profile(case, corpus)
    with Gateway(
        config.harness, case["spec"], case["task"], case["repo"], trial / "broker.jsonl"
    ) as gateway:
        gateway.attach_knowledge(profile, runtime_paths(config, profile))
        with pytest.raises(urllib.error.HTTPError) as rejected:
            post(gateway, ["not", "a", "request"])
        assert rejected.value.code == 400
        assert gateway.infrastructure_error is None
        assert not gateway.knowledge.sessions
