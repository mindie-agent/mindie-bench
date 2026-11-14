import http.client
import json
import os
import socket

import pytest

from mindie_bench.runner.gateway import Gateway
from mindie_bench.synthetic import demo


def test_long_report_path_preserves_the_full_synthetic_trial(tmp_path):
    output = tmp_path / ("release-evidence-" * 8) / "demo"
    if os.name == "nt":
        import subprocess
        import sys

        from mindie_bench.runner.host_platform import rmtree_force, subprocess_window_kwargs

        probe = output / "run" / ("p" * 80)
        try:
            probe.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-I", "-c", "pass"],
                cwd=probe,
                check=True,
                capture_output=True,
                timeout=10,
                **subprocess_window_kwargs(),
            )
        except OSError:
            pytest.skip("Windows LongPathsEnabled=0 forbids Win32 CreateProcessW cwd > MAX_PATH")
        finally:
            rmtree_force(tmp_path / ("release-evidence-" * 8))
    result = demo(output)
    assert result["model_calls"] == result["npu_runs"] == 0
    receipts = list((output / "run").glob("*/trial_receipt.json"))
    assert len(receipts) == 3
    for path in receipts:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        assert receipt["terminal_status"] == "completed_pass"
        assert not receipt["errors"]


@pytest.mark.skipif(os.name == "nt", reason="UnixStreamServer requires POSIX AF_UNIX")
def test_unix_gateway_removes_its_socket_after_trial_failure(case):
    with pytest.raises(RuntimeError, match="trial interrupted"):
        with Gateway(
            case["config"].harness, case["spec"], case["task"], case["repo"],
            case["root"] / "broker.jsonl", use_unix_socket=True,
        ) as gateway:
            directory = gateway.unix_socket.parent
            connection = http.client.HTTPConnection("localhost", timeout=5)
            connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.sock.settimeout(5)
            connection.sock.connect(str(gateway.unix_socket))
            try:
                connection.request(
                    "POST", "/bench/replay", body=json.dumps({"argv": ["pytest", "-q"]}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                assert response.status == 200
                assert json.load(response)["exit_code"] == 1
            finally:
                connection.close()
            raise RuntimeError("trial interrupted")
    assert not directory.exists()
