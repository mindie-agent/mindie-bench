"""Cross-platform host discovery, non-blocking pipe I/O, and native Win32 shims.

[Role]
Provides platform-honest binary resolution (`which_native`, `resolve_git_bash`),
thread-backed non-blocking pipe readers/writers (`PipeReader`, `write_pipe_bounded`),
and native Windows `.exe` script shims (`write_executable_shim`).

[Invariants]
- On POSIX/WSL2, `which_native` rejects `/mnt/<drive>` Windows interop binaries and
  `.exe`/`.cmd`/`.bat`/`.ps1` shims unless explicitly requested by suffix.
- On Windows, `which_native` rejects 0-byte `WindowsApps` Store stubs, and
  `resolve_git_bash` rejects `System32\\bash.exe` / `Sysnative\\bash.exe` WSL stubs.
- `PipeReader` distinguishes timeout `(True, b"")` from genuine EOF `(False, b"")`.

[Timeouts & Locks]
- `PipeReader` and `PipeWriter` honor caller-supplied monotonic deadlines via bounded
  queues (`maxsize=64` / `maxsize=1`); `_SHIM_LOCK` serializes one-time PEP-405 stub load.

[Side-Effect Policy]
- Writes shims only inside caller-owned directories with zero global temp files.
"""

from __future__ import annotations

import io
import json
import os
import platform
import posixpath
import queue
import re
import select
import shutil
import stat
import subprocess
import sys
import threading
import time
import weakref
import zipfile
from collections.abc import Mapping
from pathlib import Path

_WSL_MOUNT_RE = re.compile(r"^/mnt/[a-zA-Z](?:/|$)")
_WIN_EXTS = (".exe", ".cmd", ".bat", ".ps1")
_SHIM_LOCK = threading.Lock()
_CACHED_SHIM_BYTES: bytes | None = None


def subprocess_window_kwargs() -> dict[str, int]:
    """Hide the Windows console for background subprocesses (P1.5)."""
    if os.name != "nt":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def rmtree_force(path: Path) -> None:
    """Remove a directory tree, clearing Windows read-only bits on `.git/objects`."""
    target_str = str(path.absolute())
    if os.name == "nt" and not target_str.startswith("\\\\?\\"):
        ext = "\\\\?\\UNC\\" + target_str[2:] if target_str.startswith("\\\\") else "\\\\?\\" + target_str
    else:
        ext = target_str
    if not os.path.exists(ext):
        return

    def _clear_readonly(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
        except TimeoutError:
            raise
        except OSError:
            return
        func(target)

    for attempt in range(5 if os.name == "nt" else 1):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(ext, onexc=_clear_readonly)
            else:
                shutil.rmtree(ext, onerror=_clear_readonly)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            time.sleep(0.05)


def is_wsl_interop_path(candidate: str | Path) -> bool:
    raw = os.fsdecode(candidate).replace("\\", "/")
    if _WSL_MOUNT_RE.match(posixpath.normpath(raw)):
        return True
    if os.name == "nt":
        return False
    try:
        resolved = os.path.realpath(raw)
    except TimeoutError:
        raise
    except OSError:
        return False
    return bool(_WSL_MOUNT_RE.match(resolved))


def is_windows_store_stub(candidate: str | Path) -> bool:
    norm = os.fsdecode(candidate).replace("/", "\\").lower()
    if "\\microsoft\\windowsapps\\" in norm:
        return True
    if os.name == "nt":
        try:
            return Path(candidate).stat().st_size == 0
        except TimeoutError:
            raise
        except OSError:
            return True
    return False


def _windows_extra_tool_dirs(cmd_lower: str) -> list[str]:
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
    local = os.environ.get("LOCALAPPDATA") or (str(Path(home) / "AppData/Local") if home else "")
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    extra: list[str] = []
    if cmd_lower in {"uv", "uv.exe"}:
        if home:
            extra.append(str(Path(home) / ".local/bin"))
        if local:
            extra.append(str(Path(local) / "Microsoft/WinGet/Links"))
            pkgs = Path(local) / "Microsoft/WinGet/Packages"
            if pkgs.is_dir():
                extra.extend(str(p) for p in sorted(pkgs.glob("astral-sh.uv_*")))
    elif cmd_lower in {"git", "git.exe"}:
        extra.extend([str(Path(pf) / "Git/cmd"), str(Path(pfx86) / "Git/cmd")])
        if local:
            extra.append(str(Path(local) / "Programs/Git/cmd"))
    return extra


def which_native(cmd: str, path: str | None = None) -> str | None:
    """Locate a host-native executable without WSL `/mnt/<drive>` or `WindowsApps` traps."""
    if not cmd:
        return None
    search_path = os.environ.get("PATH", "") if path is None else path
    explicit_ext = cmd.lower().endswith(_WIN_EXTS)
    if os.name != "nt":
        if "/" in cmd:
            if is_wsl_interop_path(cmd) or (not explicit_ext and cmd.lower().endswith(_WIN_EXTS)):
                return None
            return cmd if os.path.isfile(cmd) and os.access(cmd, os.X_OK) else None
        entries = [entry for entry in search_path.split(os.pathsep) if entry and not is_wsl_interop_path(entry)]
        found = shutil.which(cmd, path=os.pathsep.join(entries))
        if not found or is_wsl_interop_path(found):
            return None
        if not explicit_ext and found.lower().endswith(_WIN_EXTS):
            return None
        return found
    if "/" in cmd or "\\" in cmd:
        found = shutil.which(cmd)
        return None if not found or is_windows_store_stub(found) else found
    dirs = [d for d in search_path.split(os.pathsep) if d]
    if path is None:
        dirs.extend(_windows_extra_tool_dirs(cmd.lower()))
    for entry in dirs:
        if "\\microsoft\\windowsapps" in entry.replace("/", "\\").lower():
            continue
        found = shutil.which(cmd, path=entry)
        if found and not is_windows_store_stub(found):
            return found
    return None


def _is_wsl_bash_stub(candidate: str | Path) -> bool:
    norm = os.fsdecode(candidate).replace("/", "\\").lower()
    return any(
        marker in norm
        for marker in (
            "\\windows\\system32\\bash.exe",
            "\\windows\\sysnative\\bash.exe",
            "\\windows\\syswow64\\bash.exe",
            "\\microsoft\\windowsapps\\bash.exe",
        )
    )


def resolve_git_bash(env: Mapping[str, str] | None = None) -> str:
    """Resolve POSIX `bash` or Git for Windows `bash.exe`, never `System32\\bash.exe`."""
    merged = env if env is not None else os.environ
    if os.name != "nt":
        if Path("/bin/bash").is_file():
            return "/bin/bash"
        found = which_native("bash", path=merged.get("PATH")) or which_native("bash")
        if found:
            return found
        raise FileNotFoundError("POSIX bash executable not found")

    candidates: list[Path] = []
    for env_key in ("MINDIE_GIT_BASH", "CLAUDE_CODE_GIT_BASH_PATH"):
        override = merged.get(env_key) or os.environ.get(env_key)
        if override:
            candidates.append(Path(override))
    for path_arg in (merged.get("PATH"), None):
        bash_hit = which_native("bash", path=path_arg)
        if bash_hit and not _is_wsl_bash_stub(bash_hit):
            return bash_hit
    for path_arg in (merged.get("PATH"), None):
        git_hit = which_native("git", path=path_arg)
        if git_hit:
            for parent in Path(git_hit).resolve().parents[:3]:
                candidates.extend([parent / "bin/bash.exe", parent / "usr/bin/bash.exe"])
    home = os.environ.get("USERPROFILE") or ""
    local = os.environ.get("LOCALAPPDATA") or (str(Path(home) / "AppData/Local") if home else "")
    for root in (
        os.environ.get("ProgramFiles", r"C:\Program Files") + r"\Git",
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)") + r"\Git",
        str(Path(local) / "Programs/Git") if local else "",
        str(Path(home) / "scoop/apps/git/current") if home else "",
    ):
        if root:
            candidates.extend([Path(root) / "bin/bash.exe", Path(root) / "usr/bin/bash.exe"])
    for candidate in candidates:
        if not _is_wsl_bash_stub(candidate) and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Git for Windows bash.exe not found; refusing C:\\Windows\\System32\\bash.exe WSL stub"
    )


_WIN32_DISPATCHER_MAIN = """import json, os, runpy, subprocess, sys
if len(sys.argv) == 2 and sys.argv[1] == "--__mindie_shim_warmup__":
    raise SystemExit(0)
_exe = os.path.abspath(sys.argv[0])
_stem = _exe[:-4] if _exe.lower().endswith(".exe") else _exe
with open(_stem + "-shim.json", "r", encoding="utf-8") as _fh:
    _spec = json.load(_fh)
if _spec["mode"] == "inproc":
    _extra = list(_spec.get("sys_path", []))
    if not _spec.get("isolated", True):
        _extra = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p] + _extra
    for _p in reversed(_extra):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
    sys.argv[0] = _exe
    runpy.run_path(_stem + "-shim.py", run_name="__main__")
else:
    import ctypes
    def _std(s):
        if s is None:
            return subprocess.DEVNULL
        try:
            return s if s.fileno() >= 0 else subprocess.DEVNULL
        except OSError:
            return subprocess.DEVNULL
    _flags = subprocess.CREATE_NO_WINDOW if not ctypes.windll.kernel32.GetConsoleWindow() else 0
    raise SystemExit(subprocess.call(
        [*_spec["argv"], *sys.argv[1:]],
        stdin=_std(sys.stdin),
        stdout=_std(sys.stdout),
        stderr=_std(sys.stderr),
        creationflags=_flags,
    ))
"""
_SHIM_WARMED = False


def _win32_dispatcher_exe() -> bytes:
    """Build a single deterministic PEP-405 `.exe` so Smart App Control caches one SHA256."""
    global _CACHED_SHIM_BYTES
    with _SHIM_LOCK:
        if _CACHED_SHIM_BYTES is not None:
            return _CACHED_SHIM_BYTES
        import ensurepip

        bundled = Path(ensurepip.__file__).resolve().parent / "_bundled"
        wheel = next(bundled.glob("pip-*.whl"))
        arch = platform.machine().lower()
        stub_name = "t64-arm.exe" if arch in {"arm64", "aarch64"} else "t64.exe"
        with zipfile.ZipFile(wheel) as zf:
            stub = zf.read(f"pip/_vendor/distlib/{stub_name}")
        base_py = Path(sys.base_prefix).resolve() / "python.exe"
        if not base_py.is_file():
            base_py = Path(sys.executable).resolve()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            info = zipfile.ZipInfo("__main__.py", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, _WIN32_DISPATCHER_MAIN.encode("utf-8"))
        _CACHED_SHIM_BYTES = stub + f'#!"{base_py}" -I -X utf8\r\n'.encode("utf-8") + buf.getvalue()
        return _CACHED_SHIM_BYTES


def _write_win32_dispatcher(exe_path: Path, spec: dict) -> Path:
    global _SHIM_WARMED
    stem = exe_path.name[:-4] if exe_path.name.lower().endswith(".exe") else exe_path.name
    (exe_path.parent / f"{stem}-shim.json").write_text(
        json.dumps(spec), encoding="utf-8", newline="\n"
    )
    exe_path.write_bytes(_win32_dispatcher_exe())
    if not _SHIM_WARMED:
        with _SHIM_LOCK:
            if not _SHIM_WARMED:
                for attempt in range(4):
                    try:
                        subprocess.run(
                            [str(exe_path), "--__mindie_shim_warmup__"],
                            capture_output=True,
                            timeout=10,
                            check=False,
                            **subprocess_window_kwargs(),
                        )
                        _SHIM_WARMED = True
                        break
                    except TimeoutError:
                        raise
                    except OSError as exc:
                        if getattr(exc, "winerror", None) != 4551 or attempt == 3:
                            raise
                        time.sleep(0.3 * (attempt + 1))
    return exe_path


def write_command_alias(directory: Path, name: str, target_argv: list[str]) -> Path:
    """Create a directly spawnable command forwarding all CLI flags (`python3` -> `python.exe`)."""
    if not target_argv:
        raise ValueError("target_argv must be non-empty")
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        import shlex

        target = directory / name
        quoted = " ".join(shlex.quote(part) for part in target_argv)
        target.write_text(f"#!/bin/sh\nexec {quoted} \"$@\"\n", encoding="utf-8")
        target.chmod(0o700)
        return target
    stem = name[:-4] if name.lower().endswith(".exe") else name
    return _write_win32_dispatcher(directory / f"{stem}.exe", {"mode": "exec", "argv": list(target_argv)})


def write_executable_shim(
    directory: Path,
    name: str,
    script_body: str,
    *,
    python_exe: str | None = None,
    isolated: bool = True,
) -> Path:
    """Create a directly spawnable command (`CreateProcessW` `.exe` on Windows, `#!` on POSIX)."""
    directory.mkdir(parents=True, exist_ok=True)
    interpreter = python_exe or sys.executable
    if os.name != "nt":
        if " " not in interpreter and len(interpreter) < 240:
            flag = " -I" if isolated else ""
            target = directory / name
            target.write_text(f"#!{interpreter}{flag}\n{script_body}", encoding="utf-8")
            target.chmod(0o700)
            return target
        script_path = directory / f"{name}-script.py"
        script_path.write_text(script_body, encoding="utf-8")
        py_args = [interpreter, *(["-I"] if isolated else []), str(script_path)]
        return write_command_alias(directory, name, py_args)
    stem = name[:-4] if name.lower().endswith(".exe") else name
    script_path = directory / f"{stem}-shim.py"
    script_path.write_text(script_body, encoding="utf-8", newline="\n")
    same_py = os.path.normcase(str(Path(interpreter).resolve())) == os.path.normcase(
        str(Path(sys.executable).resolve())
    )
    if same_py:
        cwd_norm = os.path.normcase(os.getcwd())
        extra_paths = [
            p for p in sys.path[1:] if p and os.path.normcase(os.path.abspath(p)) != cwd_norm
        ]
        spec = {"mode": "inproc", "isolated": isolated, "sys_path": extra_paths}
    else:
        py_args = [interpreter, *(["-I"] if isolated else []), "-X", "utf8", str(script_path)]
        spec = {"mode": "exec", "argv": py_args}
    return _write_win32_dispatcher(directory / f"{stem}.exe", spec)


class PipeReader:
    """Bounded daemon-thread pipe reader safe on both POSIX and Windows anonymous pipes."""

    def __init__(self, stream, *, chunk_size: int = 65536) -> None:
        self.stream = stream
        self._fd = stream if isinstance(stream, int) else stream.fileno()
        self._chunk_size = chunk_size
        self._queue: queue.Queue[tuple[bytes, BaseException | None]] = queue.Queue(maxsize=64)
        self._peeked: tuple[bytes, BaseException | None] | None = None
        self._eof = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                chunk = os.read(self._fd, self._chunk_size)
            except TimeoutError:
                raise
            except (OSError, ValueError) as exc:
                self._queue.put((b"", exc))
                return
            self._queue.put((chunk, None))
            if not chunk:
                return

    def wait_readable(self, timeout: float) -> bool:
        if self._eof:
            return True
        if self._peeked is not None:
            return True
        try:
            self._peeked = self._queue.get(timeout=max(0.0, timeout))
            return True
        except queue.Empty:
            return False

    def read_chunk(self, timeout: float) -> tuple[bool, bytes]:
        if self._eof:
            return False, b""
        if self._peeked is not None:
            item = self._peeked
            self._peeked = None
        else:
            try:
                item = self._queue.get(timeout=max(0.0, timeout))
            except queue.Empty:
                return True, b""
        chunk, err = item
        if err is not None:
            self._eof = True
            raise err
        if not chunk:
            self._eof = True
        return False, chunk


_PipeWriteJob = tuple[bytes, threading.Event, list[BaseException]]


def _stop_pipe_writer_queue(jobs: queue.Queue[_PipeWriteJob | None]) -> None:
    while True:
        try:
            jobs.put_nowait(None)
            return
        except queue.Full:
            try:
                jobs.get_nowait()
            except queue.Empty:
                pass


def _pipe_writer_loop(jobs: queue.Queue[_PipeWriteJob | None], stream) -> None:
    while True:
        item = jobs.get()
        if item is None:
            return
        payload, done, errors = item
        try:
            view = memoryview(payload)
            while view:
                written = os.write(stream.fileno(), view[:65536])
                if not written:
                    raise BrokenPipeError("pipe accepted no bytes")
                view = view[written:]
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()


class PipeWriter:
    """Deadline-bounded pipe frame writer supporting Windows blocking anonymous pipes."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self._closed = False
        self._failed: BaseException | None = None
        self._jobs: queue.Queue[_PipeWriteJob | None] | None = None
        self._finalizer: weakref.finalize | None = None
        if os.name == "nt":
            jobs: queue.Queue[_PipeWriteJob | None] = queue.Queue(maxsize=1)
            self._jobs = jobs
            self._thread = threading.Thread(
                target=_pipe_writer_loop, args=(jobs, stream), name="mindie-pipe-writer", daemon=True
            )
            self._thread.start()
            self._finalizer = weakref.finalize(self, _stop_pipe_writer_queue, jobs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._finalizer is not None:
            self._finalizer()

    def write_frame(self, frame: bytes, deadline: float, *, timeout_message: str) -> None:
        if self._closed:
            raise ValueError("PipeWriter is closed")
        if self._failed is not None:
            raise RuntimeError("PipeWriter stream is tainted by an earlier failed write") from self._failed
        try:
            if os.name != "nt":
                fd = self.stream.fileno()
                os.set_blocking(fd, False)
                remaining = memoryview(frame)
                while remaining:
                    wait = deadline - time.monotonic()
                    if wait <= 0 or not select.select([], [fd], [], wait)[1]:
                        raise TimeoutError(timeout_message)
                    try:
                        written = os.write(fd, remaining[:65536])
                    except BlockingIOError:
                        continue
                    if not written:
                        raise BrokenPipeError("pipe accepted no bytes")
                    remaining = remaining[written:]
                return
            assert self._jobs is not None
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise TimeoutError(timeout_message)
            done = threading.Event()
            errors: list[BaseException] = []
            try:
                self._jobs.put((frame, done, errors), timeout=wait)
            except queue.Full as exc:
                raise TimeoutError(timeout_message) from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not done.wait(remaining):
                raise TimeoutError(timeout_message)
            if errors:
                raise errors[0]
        except BaseException as exc:
            self._failed = exc
            raise
