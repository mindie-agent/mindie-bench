import json
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.runner import sandbox


@pytest.mark.skipif(
    platform.system() != "Linux" or not shutil.which("bwrap"),
    reason="requires real Linux bubblewrap",
)
def test_aliased_python_runtime_keeps_virtualenv_and_isolation(tmp_path):
    # uv's version-independent interpreter alias differs from sys.base_prefix.
    base = Path(sys.base_prefix).resolve()
    alias = tmp_path / "python-current"
    alias.symlink_to(base, target_is_directory=True)
    executable = alias / Path(sys._base_executable).resolve().relative_to(base)
    runtime = tmp_path / "runtime"
    subprocess.run(
        [str(executable), "-I", "-m", "venv", "--without-pip", str(runtime)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    work = tmp_path / "workspace"
    work.mkdir()
    secret = tmp_path / "private-secret"
    secret.write_text("outside the declared runtime and workspace")
    owner = r"""
import json, runpy, subprocess, sys
from pathlib import Path
native_command = runpy.run_path(sys.argv[1])["native_command"]
work = Path(sys.argv[2])
probe = '''
import json, socket, sys
from pathlib import Path
assert not Path(sys.argv[1]).exists(), "ungranted file became readable"
with socket.socket() as connection:
    connection.settimeout(1)
    assert connection.connect_ex(("127.0.0.1", int(sys.argv[2]))) != 0
Path("result.txt").write_text("workspace write succeeded")
print(json.dumps({"prefix": sys.prefix}))
'''
command = native_command(
    [sys.executable, "-I", "-c", probe, sys.argv[3], sys.argv[4]],
    read=[Path(sys.prefix), Path(sys.base_prefix)],
    write=[work],
    cwd=work,
    policy_path=work / "policy",
)
result = subprocess.run(command, capture_output=True, text=True, timeout=5)
assert result.returncode == 0, result.stderr
print(result.stdout, end="")
"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = subprocess.run(
            [
                str(runtime / "bin/python"),
                "-I",
                "-c",
                owner,
                str(Path(sandbox.__file__).resolve()),
                str(work),
                str(secret),
                str(listener.getsockname()[1]),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert json.loads(result.stdout)["prefix"] == str(runtime)
    assert (work / "result.txt").read_text() == "workspace write succeeded"
    assert secret.read_text() == "outside the declared runtime and workspace"
