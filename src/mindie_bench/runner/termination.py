"""Bounded, two-stage owned-process termination and cross-platform group ownership.

[Role]
Spawns owned subprocess groups (`spawn_owned_popen` via `owner.py` pipe on POSIX or
Win32 `KILL_ON_JOB_CLOSE` Job Object on Windows) and terminates them in two bounded
stages: `SIGTERM` -> bounded grace wait -> `SIGKILL`/`TerminateJobObject` -> reap wait.

[Invariants]
- Cleanup never blocks forever: both graceful and forceful waits are strictly bounded.
- A group that survives forceful termination raises `TimeoutError`, never silent success.
- On Windows, Job Object `ActiveProcesses` tracks grandchildren even after leader exit.

[Timeouts & Locks]
- Graceful wait: `term_wait` (default `2.0s`); forceful wait: `kill_wait` (default `5.0s`).
- `taskkill /F /T` fallback uses a bounded `5.0s` timeout. No locks held here.

[Side-Effect Policy]
- Signals/terminates only the supplied owned process group or Win32 Job Object.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable

_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_STDLIB_POPEN = subprocess.Popen

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as _wt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    _CREATE_SUSPENDED = 0x00000004
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_op", "w_op", "o_op", "r_tr", "w_tr", "o_tr")]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", _wt.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", _wt.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", _wt.DWORD),
            ("SchedulingClass", _wt.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", _wt.DWORD),
            ("TotalProcesses", _wt.DWORD),
            ("ActiveProcesses", _wt.DWORD),
            ("TotalTerminatedProcesses", _wt.DWORD),
        ]

    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, _wt.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = _wt.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [_wt.HANDLE, ctypes.c_int, ctypes.c_void_p, _wt.DWORD]
    _kernel32.SetInformationJobObject.restype = _wt.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [_wt.HANDLE, _wt.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = _wt.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        _wt.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        _wt.DWORD,
        ctypes.c_void_p,
    ]
    _kernel32.QueryInformationJobObject.restype = _wt.BOOL
    _kernel32.TerminateJobObject.argtypes = [_wt.HANDLE, _wt.UINT]
    _kernel32.TerminateJobObject.restype = _wt.BOOL
    _kernel32.CloseHandle.argtypes = [_wt.HANDLE]
    _kernel32.CloseHandle.restype = _wt.BOOL
    _ntdll.NtResumeProcess.argtypes = [_wt.HANDLE]
    _ntdll.NtResumeProcess.restype = ctypes.c_long


def _close_win_job(process) -> None:
    if sys.platform != "win32":
        return
    job = getattr(process, "_win_job_handle", None)
    if job:
        process._win_job_handle = None
        _kernel32.CloseHandle(_wt.HANDLE(job))


def _win_job_active_count(process) -> int | None:
    if sys.platform != "win32":
        return None
    job = getattr(process, "_win_job_handle", None)
    if not job:
        return None
    acct = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
    ok = _kernel32.QueryInformationJobObject(
        _wt.HANDLE(job), 1, ctypes.byref(acct), ctypes.sizeof(acct), None
    )
    return int(acct.ActiveProcesses) if ok else None


def _win_taskkill_tree(pid: int | None) -> None:
    if sys.platform != "win32" or not isinstance(pid, int) or pid <= 0:
        return
    try:
        with _STDLIB_POPEN(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ) as proc:
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate(timeout=1)
    except TimeoutError:
        raise
    except (OSError, subprocess.SubprocessError):
        pass


def spawn_owned_popen(command: list[str], **popen_kwargs) -> tuple[subprocess.Popen, Callable[[], None]]:
    """Spawn a process group owned by an `owner.py` pipe (POSIX) or Job Object (Win32)."""
    if sys.platform != "win32":
        read_fd, owner_fd = os.pipe()
        closed = False

        def _close_owner() -> None:
            nonlocal closed
            if not closed:
                closed = True
                os.close(owner_fd)

        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-m", "mindie_bench.runner.owner", str(read_fd), *command],
                start_new_session=True,
                pass_fds=(read_fd,),
                **popen_kwargs,
            )
        except BaseException:
            _close_owner()
            raise
        finally:
            os.close(read_fd)
        return proc, _close_owner

    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        err = ctypes.get_last_error()
        _kernel32.CloseHandle(job)
        raise OSError(err, "SetInformationJobObject failed")
    flags = int(popen_kwargs.pop("creationflags", 0)) | subprocess.CREATE_NO_WINDOW | _CREATE_SUSPENDED
    popen_kwargs.pop("start_new_session", None)
    child_env = popen_kwargs.get("env")
    if command and "/" not in command[0] and "\\" not in command[0] and isinstance(child_env, dict):
        from mindie_bench.runner.host_platform import which_native

        resolved = which_native(command[0], path=child_env.get("PATH"))
        if resolved:
            command = [resolved, *command[1:]]
    try:
        proc = subprocess.Popen(command, creationflags=flags, **popen_kwargs)
    except BaseException:
        _kernel32.CloseHandle(job)
        raise
    if type(proc) is not _STDLIB_POPEN:
        handle = getattr(proc, "_handle", None)
        if handle is not None:
            _ntdll.NtResumeProcess(_wt.HANDLE(int(handle)))
        _kernel32.CloseHandle(job)
        return proc, lambda: None
    try:
        if not _kernel32.AssignProcessToJobObject(job, _wt.HANDLE(int(proc._handle))):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        status = _ntdll.NtResumeProcess(_wt.HANDLE(int(proc._handle)))
        if status != 0:
            raise OSError(f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}")
    except BaseException:
        proc.kill()
        proc.wait(timeout=2)
        _kernel32.CloseHandle(job)
        raise
    proc._win_job_handle = int(job)

    def _close_job() -> None:
        _close_win_job(proc)

    return proc, _close_job


def _wait_bounded(process, timeout: float):
    """Wait for either a Popen handle or a multiprocessing.Process handle."""
    if hasattr(process, "wait"):
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    process.join(timeout)
    return None if process.is_alive() else process.exitcode


def _exited(process) -> bool:
    return process.poll() is not None if hasattr(process, "poll") else not process.is_alive()


def _group_alive(process) -> bool:
    if os.name != "posix":
        active = _win_job_active_count(process)
        if active is not None:
            return active > 0 or not _exited(process)
        return not _exited(process)
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin reports EPERM for a group containing only zombies.
        return not _exited(process)
    return True


def _signal_group(process, signum, *, force: bool = False) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    if sys.platform == "win32":
        job = getattr(process, "_win_job_handle", None)
        if job:
            if force:
                _kernel32.TerminateJobObject(_wt.HANDLE(job), 1)
        elif isinstance(process, _STDLIB_POPEN):
            # Plain Win32 `Popen.terminate()` calls `TerminateProcess` on the leader only;
            # without a Job Object we must kill the process tree before the leader PID exits.
            _win_taskkill_tree(getattr(process, "pid", None))
        elif force:
            _win_taskkill_tree(getattr(process, "pid", None))
    method = process.kill if force else process.terminate
    try:
        method()
    except TimeoutError:
        raise
    except (ProcessLookupError, OSError):
        pass


def terminate_process_group(
    process,
    *,
    term_wait: float = 2.0,
    kill_wait: float = 5.0,
):
    """Terminate the owned process group in two bounded stages."""
    for name, value in (("term_wait", term_wait), ("kill_wait", kill_wait)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    try:
        if _exited(process) and not _group_alive(process):
            _close_win_job(process)
            return getattr(process, "returncode", getattr(process, "exitcode", None))

        _signal_group(process, signal.SIGTERM, force=False)
        deadline = time.monotonic() + float(term_wait)
        while time.monotonic() < deadline:
            _wait_bounded(process, min(0.05, max(0.001, deadline - time.monotonic())))
            if not _group_alive(process):
                _wait_bounded(process, 0.05)
                _close_win_job(process)
                return getattr(process, "returncode", getattr(process, "exitcode", None))
            time.sleep(0.01)

        if _group_alive(process):
            _signal_group(process, _SIGKILL, force=True)
        deadline = time.monotonic() + float(kill_wait)
        while time.monotonic() < deadline:
            _wait_bounded(process, min(0.05, max(0.001, deadline - time.monotonic())))
            if not _group_alive(process):
                _wait_bounded(process, 0.05)
                _close_win_job(process)
                return getattr(process, "returncode", getattr(process, "exitcode", None))
            time.sleep(0.01)
    except BaseException:
        _signal_group(process, _SIGKILL, force=True)
        _wait_bounded(process, 0.2)
        _close_win_job(process)
        raise
    _close_win_job(process)
    raise TimeoutError("owned process group survived SIGTERM and SIGKILL")
