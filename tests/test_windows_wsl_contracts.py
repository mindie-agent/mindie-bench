"""Cross-platform Windows (Win32 / PS 5.1 / CP936) and WSL2 boundary contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mindie_bench.models import ToolCall, Trajectory
from mindie_bench.runner.host_platform import (
    PipeReader,
    PipeWriter,
    is_windows_store_stub,
    is_wsl_interop_path,
    resolve_git_bash,
    subprocess_window_kwargs,
    which_native,
    write_command_alias,
    write_executable_shim,
)
from mindie_bench.runner.termination import spawn_owned_popen, terminate_process_group
from mindie_bench.scorer.command_observations import observe_npu_command
from mindie_bench.scorer.target_access import target_access_observations


def test_wsl_interop_and_windows_store_stub_classifiers(tmp_path):
    assert is_wsl_interop_path("/mnt/c/Windows/System32/bash.exe")
    assert is_wsl_interop_path("/mnt/d/Program Files (x86)/Git/cmd/git.exe")
    assert not is_wsl_interop_path("/usr/bin/git")
    assert not is_wsl_interop_path("/mnt/wslg/runtime")

    fake_apps = tmp_path / "Microsoft" / "WindowsApps"
    fake_apps.mkdir(parents=True)
    stub = fake_apps / "python.exe"
    stub.write_bytes(b"")
    assert is_windows_store_stub(stub)

    real_dir = tmp_path / "Python313"
    real_dir.mkdir()
    real_py = real_dir / "python.exe"
    real_py.write_bytes(b"MZ-real-binary")
    assert not is_windows_store_stub(real_py)


def test_which_native_rejects_wsl_drvfs_and_windows_stubs(tmp_path, monkeypatch):
    if os.name != "nt":
        wsl_dir = tmp_path / "mnt" / "c" / "Program Files (x86)" / "Tool"
        wsl_dir.mkdir(parents=True)
        win_exe = wsl_dir / "mytool.exe"
        win_exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        win_exe.chmod(0o755)
        win_extensionless = wsl_dir / "mytool"
        win_extensionless.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        win_extensionless.chmod(0o755)

        linux_dir = tmp_path / "usr_bin"
        linux_dir.mkdir()
        linux_bin = linux_dir / "mytool"
        linux_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        linux_bin.chmod(0o755)

        monkeypatch.setattr(
            "mindie_bench.runner.host_platform.is_wsl_interop_path",
            lambda p: "/mnt/c/" in Path(p).as_posix(),
        )
        mixed_path = f"{wsl_dir}{os.pathsep}{linux_dir}"
        assert which_native("mytool", path=mixed_path) == str(linux_bin)
        assert which_native("mytool.exe", path=str(wsl_dir)) is None
    else:
        apps_dir = tmp_path / "Microsoft" / "WindowsApps"
        apps_dir.mkdir(parents=True)
        (apps_dir / "mytool.exe").write_bytes(b"")
        real_dir = tmp_path / "real"
        real_bin = write_executable_shim(real_dir, "mytool", "print('ok')\n")
        mixed_path = f"{apps_dir}{os.pathsep}{real_dir}"
        resolved = which_native("mytool", path=mixed_path)
        assert resolved is not None
        assert Path(resolved).resolve() == real_bin.resolve()


def test_resolve_git_bash_rejects_system32_wsl_stub(tmp_path, monkeypatch):
    bash_path = resolve_git_bash()
    norm = bash_path.replace("\\", "/").lower()
    assert "/system32/bash.exe" not in norm
    assert "/sysnative/bash.exe" not in norm
    assert "/windowsapps/bash.exe" not in norm


def test_extended_windows_prefix_and_exe_normalization_in_scorer():
    calls = [
        ToolCall(
            id="c1",
            name="Read",
            arguments={
                "file_path": r"\\?\C:\Users\runneradmin\AppData\Local\Temp\trial\workspace\vllm_ascend\worker\model_runner.py"
            },
            turn=1,
            started_sec=0.1,
            ended_sec=0.2,
            is_error=False,
        ),
        ToolCall(
            id="c2",
            name="Bash",
            arguments={
                "command": r"C:\Tools\python.exe -c pass && cat '\\?\C:\Users\runneradmin\AppData\Local\Temp\trial\workspace\vllm_ascend\worker\worker.py'"
            },
            turn=2,
            started_sec=0.3,
            ended_sec=0.5,
            is_error=False,
        ),
    ]
    observations, unresolved = target_access_observations(
        Trajectory(calls=calls),
        ["vllm_ascend/worker/model_runner.py", "vllm_ascend/worker/worker.py"],
    )
    assert [row["target"] for row in observations] == [
        "vllm_ascend/worker/model_runner.py",
        "vllm_ascend/worker/worker.py",
    ]
    assert not [u for u in unresolved if u["reason"] == "dynamic_or_glob_path_unresolved"]

    obs = observe_npu_command(r"C:\Tools\python.exe -m pytest tests\test_runner.py && C:\Tools\pytest.cmd -q")
    assert obs["recognized"] is True
    assert "python -m pytest" in obs["matched_programs"]
    assert "pytest" in obs["matched_programs"]


def test_pipe_reader_and_writer_enforce_deadlines():
    sleeper = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        **subprocess_window_kwargs(),
    )
    try:
        reader = PipeReader(sleeper.stdout)
        timed_out, chunk = reader.read_chunk(0.1)
        assert timed_out is True and chunk == b""

        writer = PipeWriter(sleeper.stdin)
        with pytest.raises(TimeoutError, match="pipe deadline"):
            writer.write_frame(
                b"x" * (2 * 1024 * 1024),
                time.monotonic() + 0.15,
                timeout_message="pipe deadline",
            )
    finally:
        sleeper.kill()
        sleeper.wait(timeout=5)
        for s in (sleeper.stdin, sleeper.stdout, sleeper.stderr):
            if s:
                s.close()


@pytest.mark.tier1
def test_spawn_owned_popen_and_terminate_kills_grandchild_tree(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    parent_code = (
        "import pathlib, subprocess, sys, time\n"
        f"win_kw = {{'creationflags': subprocess.CREATE_NO_WINDOW}} if sys.platform == 'win32' else {{}}\n"
        "child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(60)'], **win_kw)\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    proc, close_owner = spawn_owned_popen(
        [sys.executable, "-I", "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.is_file(), "parent process did not spawn grandchild"
        grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
        terminate_process_group(proc, term_wait=0.2, kill_wait=2.0)
        assert proc.poll() is not None
        # Verify grandchild is also gone.
        deadline = time.monotonic() + 2.0
        alive = True
        while time.monotonic() < deadline:
            if sys.platform == "win32":
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {grandchild_pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **subprocess_window_kwargs(),
                ).stdout
                if str(grandchild_pid) not in out:
                    alive = False
                    break
            else:
                try:
                    os.kill(grandchild_pid, 0)
                except ProcessLookupError:
                    alive = False
                    break
            time.sleep(0.05)
        assert not alive, f"grandchild PID {grandchild_pid} survived group termination"
    finally:
        close_owner()
        for s in (proc.stdout, proc.stderr):
            if s:
                s.close()


@pytest.mark.tier1
def test_executable_shim_and_alias_preserve_utf8_and_multiline_args(tmp_path):
    echo_shim = write_executable_shim(
        tmp_path,
        "utf8-echo",
        "import json, sys\nsys.stdout.buffer.write((json.dumps(sys.argv[1:], ensure_ascii=False) + '\\n').encode('utf-8'))\n",
    )
    alias = write_command_alias(tmp_path, "utf8-alias", [str(echo_shim), "--prefix=中文 空格"])
    payload_arg = "第一行 line 1\n第二行 \"quoted\" %PATH% !BANG!"
    completed = subprocess.run(
        [str(alias), payload_arg],
        capture_output=True,
        timeout=15,
        check=True,
        **subprocess_window_kwargs(),
    )
    decoded = json.loads(completed.stdout.decode("utf-8"))
    assert decoded == ["--prefix=中文 空格", payload_arg]
    if os.name == "nt":
        assert echo_shim.read_bytes() == alias.read_bytes()


def test_file_uri_wheel_sha256_handles_windows_drive_paths(tmp_path):
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    wheel = tmp_path / "mindie_test-0.1.0-py3-none-any.whl"
    data = b"synthetic-wheel-bytes"
    wheel.write_bytes(data)
    uri = wheel.resolve().as_uri()
    parsed = urlparse(uri)
    raw_path = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
    resolved = Path(url2pathname(unquote(raw_path)))
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest()


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell 5.1 headless UTF-8 smoke test")
def test_headless_windows_powershell_51_utf8_under_create_no_window():
    ps_script = (
        "$utf8 = New-Object System.Text.UTF8Encoding($false); "
        "$OutputEncoding = $utf8; "
        "try { [Console]::InputEncoding = $utf8; [Console]::OutputEncoding = $utf8 } catch [System.IO.IOException] {} ; "
        "$raw = [Console]::In.ReadToEnd(); "
        "$obj = $raw | ConvertFrom-Json; "
        "$out = @{ ok = $true; echo = $obj.msg } | ConvertTo-Json -Compress; "
        "[Console]::Out.WriteLine($out)"
    )
    res = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_script,
        ],
        input=json.dumps({"msg": "昇腾MindIE中文测试"}, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=15,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    parsed = json.loads(res.stdout.decode("utf-8"))
    assert parsed == {"ok": True, "echo": "昇腾MindIE中文测试"}
