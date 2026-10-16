import base64
import json
import urllib.error
import urllib.request

import pytest

from mindie_bench.models import ToolCall, Trajectory
from mindie_bench.runner.gateway import Gateway
from mindie_bench.scorer.metrics import npu_test_runs


class Remote:
    def __init__(self):
        self.executed = []
        self.opened = []
        self.exchanges = []

    def execute(self, command, stdin):
        self.executed.append((command, stdin))
        return {"stdout_b64": "", "stderr_b64": "", "exit_code": 1}

    def open_stream(self, command):
        self.opened.append(command)
        return {"stream_id": f"owned-{len(self.opened)}"}

    def exchange_stream(self, identity, stdin, *, eof):
        self.exchanges.append((identity, stdin, eof))
        return {"stdout_b64": "", "stderr_b64": "", "exit_code": None}


def post(gateway, path, payload):
    request = urllib.request.Request(
        gateway.url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def open_rpc(gateway):
    return post(gateway, "/bench/ssh-stream", {"action": "open", "command": "python3 -u -"})["stream_id"]


def submit(gateway, transport, command, stdin=b"", *, identity=None):
    if transport == "ssh":
        return post(
            gateway,
            "/bench/ssh",
            {"command": command, "stdin_b64": base64.b64encode(stdin).decode()},
        )
    if transport == "stream-open":
        return post(gateway, "/bench/ssh-stream", {"action": "open", "command": command})
    frame = (
        json.dumps(
            {
                "id": 1,
                "kind": "python",
                "code": "uploaded worker source mentioning pytest",
                "payload": {
                    "spec": {"command": command},
                    "stdin_b64": base64.b64encode(stdin).decode(),
                },
            }
        ).encode()
        + b"\n"
    )
    return post(
        gateway,
        "/bench/ssh-stream",
        {
            "action": "exchange",
            "stream_id": identity,
            "stdin_b64": base64.b64encode(frame).decode(),
            "eof": False,
        },
    )


@pytest.mark.parametrize("transport", ["ssh", "stream-rpc"])
def test_patch_stdin_and_uploaded_source_do_not_consume_test_budget(case, transport):
    case["spec"].budget.max_npu_runs = 1
    remote = Remote()
    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=remote,
    ) as gateway:
        identity = open_rpc(gateway) if transport == "stream-rpc" else None
        patch = b"@@ -1 +1 @@\n+import pytest\n+# python3 -I test_example.py\n"
        submit(gateway, transport, "patch --dry-run -p1", patch, identity=identity)
        assert gateway.remote_test_calls == 0
        submit(gateway, transport, "pytest -q", identity=identity)
        assert gateway.remote_test_calls == 1
        with pytest.raises(urllib.error.HTTPError) as caught:
            submit(gateway, transport, "pytest -q other", identity=identity)
        assert caught.value.code == 429
        assert gateway.stop_reason == "max_npu_runs"
        assert not gateway.unknown
        assert len(remote.executed if transport == "ssh" else remote.exchanges) == 2


@pytest.mark.parametrize("transport", ["ssh", "stream-open", "stream-rpc"])
def test_all_remote_paths_count_recognized_attempts_before_submission(case, transport):
    case["spec"].budget.max_npu_runs = 1
    remote = Remote()
    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=remote,
    ) as gateway:
        identity = open_rpc(gateway)
        command = "python3 -m ruff check tests/test_layer.py && python3 -I test_layer.py"
        submit(gateway, transport, command, identity=identity)
        assert gateway.remote_test_calls == 1
        admitted = (len(remote.executed), len(remote.opened), len(remote.exchanges))
        for next_transport in ("ssh", "stream-open", "stream-rpc"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                submit(gateway, next_transport, "pytest -q", identity=identity)
            assert caught.value.code == 429
        assert (len(remote.executed), len(remote.opened), len(remote.exchanges)) == admitted
        assert gateway.remote_test_calls == 1
        assert not gateway.unknown


def test_reported_test_invocations_include_unfinished_tool_attempts():
    trajectory = Trajectory(
        calls=[
            ToolCall(
                id="unfinished",
                name="Bash",
                arguments={"command": "ssh bench-slot 'python3 -I test_layer.py'"},
                turn=1,
                started_sec=0.1,
            ),
            ToolCall(
                id="failed",
                name="remote_bash",
                arguments={"command": "pytest -q"},
                turn=2,
                started_sec=0.2,
                ended_sec=0.3,
                is_error=True,
            ),
        ]
    )
    assert npu_test_runs(trajectory) == 2
