"""Cleanup timeouts retain their uncertain-effects classification."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mindie_bench.models import UnknownExternalEffects
from mindie_bench.runner import verifier


@pytest.mark.tier1
def test_kill_timeout_is_not_erased_by_a_later_pipe_drain(tmp_path, monkeypatch):
    actual_popen = subprocess.Popen
    processes = []

    class KillTimeout:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def communicate(self, *, timeout):
            return self.process.communicate(timeout=min(timeout, 0.01))

        def kill(self):
            raise TimeoutError("process termination acknowledgement timed out")

    def start(command, **kwargs):
        process = actual_popen(command, **kwargs)
        processes.append(process)
        return KillTimeout(process)

    # Exercise the process-handle cleanup contract without changing Python's
    # global platform identity. The child and pipe timeout are real.
    monkeypatch.setattr(verifier, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(verifier.subprocess, "Popen", start)
    try:
        with pytest.raises(
            UnknownExternalEffects, match="process termination acknowledgement timed out",
        ) as caught:
            verifier._run_local_judge(
                [sys.executable, "-I", "-c", "import threading; threading.Event().wait(60)"],
                cwd=tmp_path, env=dict(os.environ), timeout=0.01,
            )
        assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=1)
