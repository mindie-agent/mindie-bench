"""[Role] Build native OS filesystem and network boundaries.

[Invariants] Only declared roots are readable; Python runtime aliases preserve
the same granted files, never a broader parent or an unsandboxed fallback.
[Timeouts & Locks] No waits or locks; process budgets belong to the caller.
[Side-Effect Policy] Write the macOS policy file; return Linux mount arguments.
"""

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
            "(allow pseudo-tty)",
            '(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))',
            # Native PTY extensions grant only newly allocated slave terminals.
            '(allow file-read* file-write* file-ioctl (require-all (regex #"^/dev/ttys[0-9]+$") '
            '(extension "com.apple.sandbox.pty")))',
            "(allow file-map-executable)",
            "(allow file-read-metadata)",
            '(allow file-read* (subpath "/System") (subpath "/usr") '
            '(subpath "/bin") (subpath "/sbin") (subpath "/Library/Apple") '
            '(subpath "/opt/homebrew") (subpath "/private/etc") '
            '(subpath "/private/var/db/timezone") '
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
        read_roots = set(p.resolve() for p in read)
        for path in sorted(read_roots):
            args += ["--ro-bind", str(path), str(path)]
        # uv virtualenv executables point through a version-independent runtime
        # symlink, while sys.base_prefix names the versioned directory. The
        # alias must exist inside the namespace as well as its granted target.
        base = Path(sys.base_prefix).resolve()
        if any(base.is_relative_to(path) for path in read_roots):
            for alias in Path(sys._base_executable).parents:
                if alias != base and alias.resolve() == base:
                    args += ["--ro-bind", str(base), str(alias)]
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
