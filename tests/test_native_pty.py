import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.models import Group
from mindie_bench.runner.profile_builder import build_profile
from mindie_bench.runner.sandbox import native_command

PROBE = r"""
import ctypes
import json
import os
import select
import socket
import sys
import termios
import traceback
from pathlib import Path

observed = {}
master = slave = None
try:
    master, slave = os.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~(termios.ECHO | termios.ICANON)
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    libc = ctypes.CDLL(None)
    libc.ptsname.restype = ctypes.c_char_p
    slave_name = libc.ptsname(master)
    assert slave_name is not None
    os.write(slave, b"owned-pty-canary\n")
    assert select.select([master], [], [], 2)[0], "owned PTY output did not become readable"
    payload = os.read(master, 1024)
    observed["pty"] = {
        "ok": True,
        "isatty": os.isatty(slave),
        "slave": slave_name.decode(),
        "payload": payload.decode(),
    }
except Exception as exc:
    observed["pty"] = {
        "ok": False,
        "type": type(exc).__name__,
        "errno": getattr(exc, "errno", None),
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
finally:
    for fd in (master, slave):
        if fd is not None:
            os.close(fd)

for name, operation in (
    ("outside_secret", lambda: Path(sys.argv[1]).read_bytes()),
    (
        "preexisting_owned_pty",
        lambda: os.close(os.open(sys.argv[2], os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)),
    ),
    (
        "loopback",
        lambda: socket.create_connection(("127.0.0.1", int(sys.argv[3])), timeout=1).close(),
    ),
):
    try:
        operation()
        observed[name] = {"denied": False}
    except Exception as exc:
        observed[name] = {
            "denied": isinstance(exc, PermissionError),
            "type": type(exc).__name__,
            "errno": getattr(exc, "errno", None),
            "message": str(exc),
        }

print(json.dumps(observed, sort_keys=True))
sys.exit(0 if observed["pty"]["ok"] and all(
    observed[name]["denied"]
    for name in ("outside_secret", "preexisting_owned_pty", "loopback")
) else 1)
"""


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires real macOS Seatbelt")
def test_native_pty_creation_keeps_existing_ttys_secrets_and_network_private(tmp_path):
    work = tmp_path / "allowed"
    work.mkdir()
    home = work / "home"
    home.mkdir()
    temporary = work / "tmp"
    temporary.mkdir()
    secret = tmp_path / "outside-owned-secret"
    secret.write_text("owned secret outside sandbox read roots\n")
    script = work / "probe.py"
    script.write_text(PROBE)
    policy = tmp_path / "native.sb"
    command = native_command(
        [sys.executable, "-I", str(script), str(secret)],
        read=[Path(sys.prefix), Path(sys.base_prefix)],
        write=[work],
        cwd=work,
        policy_path=policy,
    )
    master, slave = os.openpty()
    try:
        # This PTY is owned by the test but predates the sandbox's PTY extension.
        existing_slave = os.ttyname(slave)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            result = subprocess.run(
                [*command, existing_slave, str(listener.getsockname()[1])],
                cwd=work,
                env={
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "HOME": str(home),
                    "TMPDIR": str(temporary),
                    "PYTHONNOUSERSITE": "1",
                    "LANG": "en_US.UTF-8",
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
    finally:
        os.close(master)
        os.close(slave)

    (tmp_path / "probe.stdout").write_text(result.stdout)
    (tmp_path / "probe.stderr").write_text(result.stderr)
    (tmp_path / "probe.returncode").write_text(f"{result.returncode}\n")
    observed = json.loads(result.stdout)
    (tmp_path / "observed.json").write_text(json.dumps(observed, indent=2) + "\n")
    for name in ("outside_secret", "preexisting_owned_pty", "loopback"):
        assert observed[name]["denied"], observed
    assert observed["pty"]["ok"], observed
    assert observed["pty"]["isatty"], observed
    assert observed["pty"]["payload"] in ("owned-pty-canary\n", "owned-pty-canary\r\n"), observed
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not shutil.which("zsh") or not shutil.which("sandbox-exec"),
    reason="requires real zsh and macOS Seatbelt",
)
def test_profile_shell_redirections_use_owned_temp_directory(case, tmp_path):
    trial = tmp_path / "shell-trial"
    trial.mkdir()
    profile = build_profile(trial, case["spec"], Group.G0, case["config"])
    workspace = trial / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-profile"
    outside.write_text("private")
    probe = """set -e
/bin/cat <<'EOF'
heredoc-ok
EOF
/bin/cat <<< 'here-string-ok'
print -r -- =(print -r -- temporary-content)
if print -r -- forbidden > "$1"; then
    exit 9
fi
"""
    command = native_command(
        [shutil.which("zsh"), "-c", probe, "probe", str(outside)],
        read=[],
        write=[workspace, profile.root],
        cwd=workspace,
        policy_path=trial / "shell.sb",
    )
    result = subprocess.run(
        command, cwd=workspace, env=profile.env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    heredoc, here_string, temporary = result.stdout.splitlines()
    assert heredoc == "heredoc-ok"
    assert here_string == "here-string-ok"
    assert Path(temporary).parent == Path(profile.env["TMPDIR"])
    assert outside.read_text() == "private"
