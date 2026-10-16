import os
import signal
import subprocess
import sys
import threading
import time

from mindie_bench.models import RemoteConfig
from mindie_bench.runner.remote_slot import RemoteSlot


def test_submission_to_peer_that_stops_reading_has_a_deadline_and_becomes_unknown(case):
    slot = RemoteSlot(
        RemoteConfig(
            host="fixture",
            container="fixture",
            repo_path="/fixture",
            devices=[0],
            probe_command=["probe"],
        ),
        case["spec"].environment["l2"],
        "blocked-write",
        [0],
        1,
    )
    slot.process = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    failures = []

    def send():
        try:
            slot._request({"command": "x" * (1024 * 1024)}, 0.1)
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=send, daemon=True)
    started = time.monotonic()
    try:
        thread.start()
        thread.join(timeout=0.5)
        assert not thread.is_alive(), "request deadline did not cover pipe submission"
        assert time.monotonic() - started < 0.5
        assert len(failures) == 1 and isinstance(failures[0], TimeoutError)
        assert slot.operation_unknown
    finally:
        os.killpg(slot.process.pid, signal.SIGKILL)
        slot.process.wait(timeout=5)
        thread.join(timeout=5)
        for stream in [slot.process.stdin, slot.process.stdout, slot.process.stderr]:
            stream.close()
