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
        try:
            with pytest.raises(TimeoutError, match="pipe deadline"):
                writer.write_frame(
                    b"x" * (2 * 1024 * 1024),
                    time.monotonic() + 0.15,
                    timeout_message="pipe deadline",
                )
        finally:
            writer.close()
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


def test_pipe_writer_close_and_gc_do_not_leak_worker_threads():
    import gc
    import threading

    baseline = sum(1 for t in threading.enumerate() if t.name == "mindie-pipe-writer")
    fds: list[tuple[int, int]] = []
    try:
        for idx in range(5):
            r_fd, w_fd = os.pipe()
            fds.append((r_fd, w_fd))
            with os.fdopen(w_fd, "wb", closefd=False) as stream:
                writer = PipeWriter(stream)
                writer.write_frame(b"ping\n", time.monotonic() + 2.0, timeout_message="ping timeout")
                if idx % 2 == 0:
                    writer.close()
                del writer
        gc.collect()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            alive = sum(1 for t in threading.enumerate() if t.name == "mindie-pipe-writer")
            if alive <= baseline:
                break
            time.sleep(0.02)
        assert sum(1 for t in threading.enumerate() if t.name == "mindie-pipe-writer") <= baseline
    finally:
        for r_fd, w_fd in fds:
            for fd in (r_fd, w_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass


@pytest.mark.tier1
def test_ssh_shim_preserves_delayed_chunked_stdin_until_eof(tmp_path):
    import base64
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            captured.append(body)
            reply = json.dumps({"stdout_b64": "", "stderr_b64": "", "exit_code": 7}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        env = dict(os.environ, MINDIE_BENCH_GATEWAY=f"http://127.0.0.1:{server.server_port}")
        env.pop("MINDIE_BENCH_SSH_STREAM", None)
        env.pop("REMOTE_DEV_HARNESS_PROFILE", None)
        proc = subprocess.Popen(
            [sys.executable, "-m", "mindie_bench.runner.ssh_shim", "bench-slot", "bash -s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            **subprocess_window_kwargs(),
        )
        assert proc.stdin is not None
        proc.stdin.write(b"echo first\n")
        proc.stdin.flush()
        time.sleep(0.35)
        proc.stdin.write(b"exit 7\n")
        proc.stdin.close()
        rc = proc.wait(timeout=10)
        for s in (proc.stdout, proc.stderr):
            if s:
                s.close()
        assert rc == 7
        assert len(captured) == 1
        assert base64.b64decode(captured[0]["stdin_b64"]) == b"echo first\nexit 7\n"
    finally:
        server.shutdown()
        server.server_close()


def test_claude_driver_stop_hook_command_preserves_windows_paths_in_posix_shlex(tmp_path, monkeypatch):
    import shlex
    from types import SimpleNamespace

    import mindie_bench.runner.drivers.claude_code as claude_mod

    root = tmp_path / "profile dir"
    cfg_dir = tmp_path / "claude cfg"
    root.mkdir()
    cfg_dir.mkdir()
    (root / "settings.json").write_text("{}\n", encoding="utf-8")
    (root / "mcp.json").write_text('{"mcpServers":{}}\n', encoding="utf-8")
    monkeypatch.setattr(claude_mod, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(claude_mod, "sys", SimpleNamespace(executable=r"C:\Python313\python.exe"))
    profile = SimpleNamespace(
        root=root,
        env={"CLAUDE_CONFIG_DIR": str(cfg_dir)},
        fingerprint={"server_components": {}, "group": "G3"},
    )
    config = SimpleNamespace(executable="claude", model="sonnet", reasoning_effort=None)
    claude_mod.ClaudeCodeDriver().command(profile, config)
    saved = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    hook_cmd = saved["hooks"]["Stop"][0]["hooks"][0]["command"]
    tokens = shlex.split(hook_cmd, posix=True)
    assert tokens[0] == "C:/Python313/python.exe"
    assert tokens[-1] == str(root / "native-stop.json").replace("\\", "/")
    assert "\\" not in hook_cmd


def test_pipe_reader_propagates_read_oserror_instead_of_truncating_as_eof(monkeypatch):
    import mindie_bench.runner.host_platform as hp
    import mindie_bench.runner.ssh_shim as ssh_shim

    calls = {"n": 0}

    def flaky_read(_fd: int, _size: int) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            return b"echo prefix\n"
        raise OSError("injected pipe read failure")

    monkeypatch.setattr(hp.os, "read", flaky_read)
    r_fd, w_fd = os.pipe()
    try:
        with os.fdopen(r_fd, "rb", closefd=False) as stream:
            from types import SimpleNamespace

            monkeypatch.setattr(ssh_shim, "sys", SimpleNamespace(stdin=stream))
            with pytest.raises(OSError, match="injected pipe read failure"):
                ssh_shim._read_oneshot_stdin(wait_for_start=True)
    finally:
        for fd in (r_fd, w_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_knowledge_session_write_timeout_taints_stream_and_never_resubmits_on_close(monkeypatch):
    import io
    import threading
    from types import SimpleNamespace

    import mindie_bench.runner.host_platform as hp
    import mindie_bench.runner.knowledge_broker as kb

    session = kb.KnowledgeSession.__new__(kb.KnowledgeSession)
    session.identity = session.name = "pipe-timeout-repro"
    session.record = lambda *_a, **_k: None
    session.lock = threading.RLock()
    session.pending = session.input_buffer = session.output_buffer = session.local_output = b""
    session.requests = {}
    session.eof = session.closed = session._stdout_eof = session._stderr_eof = False
    session._stdin_tainted = None
    session.output_bytes = 0
    session.raw = io.BytesIO()
    session.errors = io.BytesIO()
    session._close_owner = lambda: None

    r_fd, w_fd = os.pipe()
    stream = os.fdopen(w_fd, "wb", buffering=0)
    session.process = SimpleNamespace(
        stdin=stream,
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
        returncode=0,
        poll=lambda: 0,
    )
    replies = iter([json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode() + b"\n", b""])
    session._stdout_reader = SimpleNamespace(
        read_chunk=lambda _t: (False, next(replies, b"")),
        wait_readable=lambda _t: True,
    )
    session._stderr_reader = SimpleNamespace(
        read_chunk=lambda _t: (False, b""),
        wait_readable=lambda _t: True,
    )
    received = bytearray()

    def read_peer() -> None:
        try:
            while chunk := os.read(r_fd, 65536):
                received.extend(chunk)
        finally:
            os.close(r_fd)

    peer = threading.Thread(target=read_peer, daemon=True)
    monkeypatch.setattr(hp, "os", SimpleNamespace(name="nt", write=os.write))
    writer = hp.PipeWriter(stream)
    calls: list[bytes] = []

    def send(frame: bytes, *_a, **_k) -> None:
        calls.append(frame)
        writer.write_frame(
            frame,
            time.monotonic() + (0.05 if len(calls) == 1 else 2.0),
            timeout_message="blocked write",
        )

    session._stdin_writer = SimpleNamespace(write_frame=send, close=writer.close)
    frame = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "knowledge_query", "arguments": {"query": "x" * 200000}},
            }
        ).encode()
        + b"\n"
    )
    with pytest.raises(TimeoutError, match="blocked write"):
        session.exchange(frame)
    peer.start()
    monkeypatch.setattr(kb, "terminate_process_group", lambda *_a, **_k: 0)
    with pytest.raises(RuntimeError, match="failed stdin submission"):
        session.close()
    peer.join(2.0)
    writer._thread.join(2.0)
    assert len(calls) == 1
    assert len(received) <= len(frame)
    assert bytes(received) == frame[: len(received)]
    assert not writer._thread.is_alive()


def test_reevaluate_run_verifier_withholds_completed_verdict_when_cleanup_fails(tmp_path, monkeypatch):
    import mindie_bench.evaluation.reevaluate as reeval_mod

    task = tmp_path / "task"
    (task / "verifier").mkdir(parents=True)
    (task / "verifier/test.sh").write_text(
        '#!/bin/sh\necho \'{"reward": 1.0}\'\nexit 0\n', encoding="utf-8", newline="\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    def fail_cleanup(*_args, **_kwargs):
        raise TimeoutError("owned process group survived")

    monkeypatch.setattr(reeval_mod, "terminate_process_group", fail_cleanup)
    with pytest.raises(TimeoutError, match="owned process group survived"):
        reeval_mod.run_verifier(task, workspace, output, Path(sys.executable), 10.0)
