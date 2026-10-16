import json
import urllib.error
import urllib.request

import pytest

from mindie_bench.runner.gateway import Gateway


def test_direct_stream_commands_cannot_bypass_recognized_test_budget(case):
    case["spec"].budget.max_npu_runs = 1
    commands = []

    class Remote:
        def open_stream(self, command):
            commands.append(command)
            return {"stream_id": f"owned-{len(commands)}"}

    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=Remote(),
    ) as gateway:

        def start(command):
            request = urllib.request.Request(
                gateway.url + "/bench/ssh-stream",
                data=json.dumps(dict(action="open", command=command)).encode(),
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(request, timeout=5)

        with start("python3 -I test_rmsnorm.py"):
            pass
        assert gateway.remote_test_calls == 1
        with pytest.raises(urllib.error.HTTPError) as caught:
            start("pytest -q")
        assert caught.value.code == 429
        assert gateway.stop_reason == "max_npu_runs"
        assert commands == ["python3 -I test_rmsnorm.py"]
