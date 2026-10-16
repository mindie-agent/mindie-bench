"""OS-enforced filesystem and network boundaries; no unsandboxed fallback."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path


def native_command(
    command: list[str],
    *,
    read: list[Path],
    write: list[Path],
    cwd: Path,
    policy_path: Path,
    gateway_port: int | None = None,
    gateway_socket: Path | None = None,
) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        executable = shutil.which("sandbox-exec")
        if not executable:
            raise RuntimeError("macOS sandbox-exec is unavailable; refusing an unisolated trial")
        # File metadata is observable, but neither personal data nor oracle
        # bytes are readable. Only the model broker's loopback port is reachable.
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process-exec)",
            "(allow process-fork)",
            "(allow process-info* (target same-sandbox))",
            "(allow sysctl*)",
            "(allow signal (target same-sandbox))",
            "(allow ipc-posix*)",
            "(allow file-map-executable)",
            "(allow file-read-metadata)",
            '(allow file-read* (subpath "/System") (subpath "/usr") '
            '(subpath "/bin") (subpath "/sbin") (subpath "/Library/Apple") '
            '(subpath "/opt/homebrew") (subpath "/private/etc") '
            '(literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))',
            '(allow file-write* (literal "/dev/null"))',
        ]
        for path in sorted(set(p.resolve() for p in read + write)):
            predicate = "subpath" if path.is_dir() else "literal"
            lines.append(f"(allow file-read* ({predicate} {json.dumps(str(path))}))")
        for path in sorted(set(p.resolve() for p in write)):
            lines.append(f"(allow file-write* (subpath {json.dumps(str(path))}))")
        if gateway_port:
            lines.append(f'(allow network-outbound (remote ip "localhost:{gateway_port}"))')
        policy_path.write_text("\n".join(lines) + "\n")
        return [executable, "-f", str(policy_path), *command]
    if system == "Linux":
        executable = shutil.which("bwrap")
        if not executable:
            raise RuntimeError("Linux bubblewrap is unavailable; refusing an unisolated trial")
        args = [
            executable,
            "--die-with-parent",
            "--unshare-all",
            "--new-session",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for directory in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"]:
            if Path(directory).exists():
                args += ["--ro-bind", directory, directory]
        for path in sorted(set(p.resolve() for p in read)):
            args += ["--ro-bind", str(path), str(path)]
        for path in sorted(set(p.resolve() for p in write)):
            args += ["--bind", str(path), str(path)]
        if gateway_socket:
            args += ["--ro-bind", str(gateway_socket), str(gateway_socket)]
            command = [
                sys.executable,
                "-I",
                "-m",
                "mindie_bench.runner.net_bridge",
                str(gateway_socket),
                str(gateway_port),
                *command,
            ]
        return [*args, "--chdir", str(cwd), "--", *command]
    raise RuntimeError(
        f"no verified sandbox backend for {system}; native Windows requires an isolated worker"
    )
