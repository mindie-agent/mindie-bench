import errno
import json
import os
import pathlib
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mindie_bench.runner.remote_slot import POSTFLIGHT


def postflight_with_process_race(tmp_path, monkeypatch, *, stage, error):
    real_path = pathlib.Path
    slot = tmp_path / "owned-slot"
    killed = []
    descriptors_closed = []
    probe_calls = []

    class Proc:
        def __init__(self, pid, *, disappearing=False):
            self.name = str(pid)
            self.disappearing = disappearing

        def stat(self):
            if self.disappearing and stage == "stat":
                raise error
            return SimpleNamespace(st_uid=os.getuid())

        def __truediv__(self, name):
            assert name == "environ"
            return SimpleNamespace(read_bytes=lambda: b"MINDIE_BENCH_SLOT_ID=owned-slot\0")

    procs = [Proc(41001, disappearing=True), Proc(41002)]

    def fake_path(value):
        if value == "/proc":
            return SimpleNamespace(iterdir=lambda: iter(procs))
        return real_path(value)

    def pidfd_open(pid):
        if pid == 41001 and stage == "pidfd":
            raise error
        return pid + 100

    def run_probe(command, **kwargs):
        assert command == ["owned-probe"]
        assert kwargs["check"] is True
        probe_calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))

    monkeypatch.setattr(pathlib, "Path", fake_path)
    monkeypatch.setattr(os, "pidfd_open", pidfd_open, raising=False)
    monkeypatch.setattr(os, "close", descriptors_closed.append)
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: killed.append((fd, sig)), raising=False
    )
    monkeypatch.setattr(subprocess, "run", run_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postflight",
            json.dumps(
                {
                    "path": str(slot),
                    "source_projection": [],
                    "probe_command": ["owned-probe"],
                    "devices": [0],
                    "memory_limit_mb": 100,
                    "repo_path": str(tmp_path / "source"),
                }
            ),
        ],
    )
    exec(compile(POSTFLIGHT, "remote-postflight", "exec"), {})
    artifact = real_path(str(slot) + ".final.json")
    return json.loads(artifact.read_text()), killed, descriptors_closed, probe_calls


@pytest.mark.parametrize("stage", ["stat", "pidfd"])
@pytest.mark.parametrize("error_type", [FileNotFoundError, ProcessLookupError])
def test_postflight_process_disappearance_preserves_cleanup_receipt(
    tmp_path, monkeypatch, capsys, stage, error_type
):
    final, killed, closed, probe_calls = postflight_with_process_race(
        tmp_path, monkeypatch, stage=stage, error=error_type("process disappeared")
    )
    assert final == {"probe": {"0": {"used_mb": 1, "pids": []}}, "errors": []}
    assert killed == [(41102, signal.SIGKILL)]
    assert closed == [41102]
    assert probe_calls == [["owned-probe"]]
    assert json.loads(capsys.readouterr().out)["cleanup"] == final


@pytest.mark.parametrize("stage", ["stat", "pidfd"])
@pytest.mark.parametrize("error", [PermissionError("denied"), OSError(errno.EIO, "proc read failed")])
def test_postflight_does_not_hide_other_process_inventory_errors(tmp_path, monkeypatch, stage, error):
    with pytest.raises(type(error)) as raised:
        postflight_with_process_race(tmp_path, monkeypatch, stage=stage, error=error)
    assert str(raised.value) == str(error)
    assert not (tmp_path / "owned-slot.final.json").exists()
