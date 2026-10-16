import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

from mindie_bench.runner import process as process_module


def gateway():
    return SimpleNamespace(unknown=False, infrastructure_error=None, stop_reason=None, model_calls=0)


def test_foreground_timeout_also_covers_a_peer_that_does_not_read_prompt(tmp_path, monkeypatch):
    owned = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        result = original(*args, **kwargs)
        owned.append(result)
        return result

    monkeypatch.setattr(process_module.subprocess, "Popen", start)
    results = []

    def run():
        results.append(
            process_module.stream_process(
                [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                env=dict(os.environ),
                cwd=tmp_path,
                prompt="x" * (1024 * 1024),
                output=tmp_path / "events.jsonl",
                stderr=tmp_path / "stderr",
                timeout=0.1,
                max_bytes=1024,
                consume=lambda *args: None,
                gateway=gateway(),
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    try:
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive(), "foreground timeout did not cover stdin submission"
        assert results[0].stop_reason == "timeout"
        assert results[0].elapsed < 1
    finally:
        for process in owned:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        thread.join(timeout=5)


def test_large_prompt_reaches_native_process_completely_before_eof(tmp_path):
    observed = []
    result = process_module.stream_process(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,sys; data=sys.stdin.buffer.read(); print(json.dumps({'bytes':len(data)}))",
        ],
        env=dict(os.environ),
        cwd=tmp_path,
        prompt="x" * (256 * 1024),
        output=tmp_path / "events.jsonl",
        stderr=tmp_path / "stderr",
        timeout=3,
        max_bytes=1024,
        consume=lambda value, *_: observed.append(value),
        gateway=gateway(),
    )
    assert result.returncode == 0
    assert result.stop_reason is None
    assert not result.errors
    assert observed == [{"bytes": 256 * 1024}]


def test_zero_exit_before_receiving_prompt_is_a_visible_submission_failure(tmp_path):
    result = process_module.stream_process(
        [sys.executable, "-I", "-c", "pass"],
        env=dict(os.environ),
        cwd=tmp_path,
        prompt="x" * (1024 * 1024),
        output=tmp_path / "events.jsonl",
        stderr=tmp_path / "stderr",
        timeout=3,
        max_bytes=1024,
        consume=lambda *args: None,
        gateway=gateway(),
    )
    assert any(
        error["stage"] == "agent" and error["type"] == "BrokenPipeError" for error in result.errors
    )
