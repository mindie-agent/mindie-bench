import base64
import io
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mindie_bench.models import RemoteConfig, SourceEnvironment
from mindie_bench.runner.remote_slot import WORKER, RemoteSlot

STREAM_BOUND = 8 * 1024 * 1024


def new_slot(tmp_path):
    return RemoteSlot(
        RemoteConfig(
            host="fixture",
            container="fixture",
            repo_path="/fixture",
            devices=[0],
            probe_command=["probe"],
        ),
        SourceEnvironment(
            repo_path="/fixture",
            base_commit="0" * 40,
            npu_model="Ascend910",
            npu_count=1,
        ),
        "stream-fixture",
        [0],
        1,
        artifact_dir=tmp_path,
    )


@pytest.fixture
def worker_slot(tmp_path):
    owned = []

    def start(cleanup=None):
        slot = new_slot(tmp_path)
        command = [sys.executable, "-I", "-u", "-c", WORKER]
        if cleanup is not None:
            postflight = f"import json; print(json.dumps({cleanup!r}),flush=True)"
            command = [
                "/bin/bash",
                "-c",
                shlex.join(command) + "\n" + shlex.join([sys.executable, "-I", "-u", "-c", postflight]),
            ]
        slot.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )
        owned.append(slot)
        assert slot._read(5) == {"ready": True}
        return slot

    yield start
    for slot in owned:
        # These workers run without a PID namespace. Never send real quiesce
        # here; reclaim only their newly created test process groups.
        try:
            os.killpg(slot.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        slot.process.wait(timeout=5)
        for pipe in [slot.process.stdin, slot.process.stdout, slot.process.stderr]:
            pipe.close()


def collect_until(slot, stream_id, condition, *, initial=None):
    stdout, stderr = bytearray(), bytearray()
    deadline = time.monotonic() + 10
    frame = initial
    while time.monotonic() < deadline:
        if frame is None:
            frame = slot.exchange_stream(stream_id)
        out = base64.b64decode(frame["stdout_b64"])
        err = base64.b64decode(frame["stderr_b64"])
        assert len(out) + len(err) <= STREAM_BOUND
        stdout.extend(out)
        stderr.extend(err)
        if condition(stdout, stderr, frame["exit_code"]):
            return bytes(stdout), bytes(stderr), frame["exit_code"]
        assert frame["exit_code"] is None, (stdout, stderr, frame)
        frame = None
    pytest.fail("remote stream did not produce its expected protocol result")


def assert_unknown_blocks_every_write(slot):
    original = slot.process.stdin
    blocked_wire = io.BytesIO()
    slot.process.stdin = blocked_wire
    try:
        for operation in [
            lambda: slot.execute("must not execute"),
            lambda: slot.open_stream("must not open"),
            lambda: slot.exchange_stream("owned", "eA=="),
            slot.quiesce,
        ]:
            with pytest.raises(RuntimeError, match="prior remote operation is unknown"):
                operation()
        assert blocked_wire.getvalue() == b""
    finally:
        slot.process.stdin = original


def test_stream_ready_before_stdin_multiple_rpc_requests_stderr_and_eof(worker_slot):
    slot = worker_slot()
    script = """
import json,os,sys
print(json.dumps({"ready":os.getpid()}),flush=True)
print("ready-stderr",file=sys.stderr,flush=True)
for line in sys.stdin:
    request=json.loads(line)
    print(json.dumps({"pid":os.getpid(),"sequence":request["sequence"]}),flush=True)
print("EOF",flush=True)
print("closed-stderr",file=sys.stderr,flush=True)
"""
    stream_id = slot.open_stream(shlex.join([sys.executable, "-I", "-u", "-c", script]))["stream_id"]
    stdout, stderr, code = collect_until(
        slot,
        stream_id,
        lambda out, err, code: b"\n" in out and b"ready-stderr\n" in err,
    )
    identity = json.loads(stdout)["ready"]
    assert stderr == b"ready-stderr\n"
    assert code is None
    for sequence in [1, 2]:
        request = base64.b64encode(json.dumps({"sequence": sequence}).encode() + b"\n").decode()
        stdout, stderr, code = collect_until(
            slot,
            stream_id,
            lambda out, err, code: b"\n" in out,
            initial=slot.exchange_stream(stream_id, request),
        )
        assert json.loads(stdout) == {"pid": identity, "sequence": sequence}
        assert stderr == b""
        assert code is None
    stdout, stderr, code = collect_until(
        slot,
        stream_id,
        lambda out, err, code: code is not None,
        initial=slot.exchange_stream(stream_id, eof=True),
    )
    assert stdout == b"EOF\n"
    assert stderr == b"closed-stderr\n"
    assert code == 0


def test_large_stream_input_does_not_block_output_drain(worker_slot):
    slot = worker_slot()
    script = f"""
import sys
sys.stdout.buffer.write(b"x"*{STREAM_BOUND})
sys.stdout.buffer.flush()
data=sys.stdin.buffer.read()
print("\\nREAD="+str(len(data)),flush=True)
"""
    stream_id = slot.open_stream(shlex.join([sys.executable, "-I", "-u", "-c", script]))["stream_id"]
    stdout, stderr, code = collect_until(
        slot,
        stream_id,
        lambda out, err, code: code is not None,
        initial=slot.exchange_stream(
            stream_id, base64.b64encode(b"y" * STREAM_BOUND).decode(), eof=True
        ),
    )
    assert stdout == b"x" * STREAM_BOUND + f"\nREAD={STREAM_BOUND}\n".encode()
    assert stderr == b""
    assert code == 0


def test_worker_stream_error_blocks_execute_quiesce_and_stream_writes(worker_slot):
    slot = worker_slot()
    with pytest.raises(RuntimeError, match="remote operation uncertain: KeyError"):
        slot.exchange_stream("unowned-stream")
    assert slot.operation_unknown
    assert_unknown_blocks_every_write(slot)


def test_oversized_stream_input_is_visible_and_never_replayed(worker_slot):
    slot = worker_slot()
    stream_id = slot.open_stream("cat")["stream_id"]
    with pytest.raises(RuntimeError, match="oversized remote stream input"):
        slot.exchange_stream(stream_id, base64.b64encode(b"x" * (STREAM_BOUND + 1)).decode())
    assert slot.operation_unknown
    assert_unknown_blocks_every_write(slot)


@pytest.mark.parametrize("operation", ["execute", "open_stream", "exchange_stream", "quiesce"])
def test_lost_control_acknowledgement_blocks_all_later_writes(tmp_path, monkeypatch, operation):
    slot = new_slot(tmp_path)
    slot.process = SimpleNamespace(stdin=io.BytesIO())

    def lost(timeout):
        raise TimeoutError("lost acknowledgement")

    monkeypatch.setattr(slot, "_send_frame", lambda frame, deadline: slot.process.stdin.write(frame))
    monkeypatch.setattr(slot, "_read", lost)
    with pytest.raises(TimeoutError):
        if operation == "execute":
            slot.execute("first command")
        elif operation == "open_stream":
            slot.open_stream("first command")
        elif operation == "exchange_stream":
            slot.exchange_stream("owned-stream", "eA==")
        else:
            slot.quiesce()
    assert slot.operation_unknown
    assert len(slot.process.stdin.getvalue().splitlines()) == 1
    assert_unknown_blocks_every_write(slot)


def test_late_stream_acknowledgement_is_retained_before_real_cleanup(worker_slot, tmp_path, monkeypatch):
    probe = {
        "0": {"used_mb": 0, "pids": [], "model": "Ascend910"},
        "_environment": dict.fromkeys(
            ["image_digest", "cann", "torch_npu", "driver", "python", "kernel"], "fixture"
        ),
    }
    cleanup = {"cleanup": {"probe": probe, "errors": []}, "remote_artifact": "/fixture.final.json"}
    slot = worker_slot(cleanup)
    read = slot._read

    def lost_after_delivery(timeout):
        readable, _, _ = select.select([slot.process.stdout], [], [], 5)
        assert readable
        raise TimeoutError("stream acknowledgement was not delivered to caller")

    monkeypatch.setattr(slot, "_read", lost_after_delivery)
    with pytest.raises(TimeoutError):
        slot.open_stream("printf late")
    monkeypatch.setattr(slot, "_read", read)
    slot.close()
    late = [
        json.loads(line) for line in (tmp_path / "remote_late_results.jsonl").read_text().splitlines()
    ]
    assert len(late) == 1
    assert isinstance(late[0]["stream_id"], str)
    assert slot.cleanup_receipt == cleanup
    assert slot.postflight["0"]["pids"] == []
