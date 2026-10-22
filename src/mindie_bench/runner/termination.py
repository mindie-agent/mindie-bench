"""Bounded, two-stage owned-process termination shared by runner components.

[Role]
Terminates subprocess groups owned by a runner component after a timeout or a failed
handshake: `SIGTERM` -> bounded grace wait -> `SIGKILL` -> bounded reap wait.

[Invariants]
- The caller owns the POSIX process group (`start_new_session=True` or a child `setsid()`).
- Cleanup never blocks forever: both the graceful and forceful waits are bounded.
- A group that survives `SIGKILL` is reported as `TimeoutError`, never silently ignored.

[Timeouts & Locks]
- Graceful wait: `term_wait` (default `2.0s`).
- Forceful wait: `kill_wait` (default `5.0s`).
- No locks are held by this helper.

[Side-Effect Policy]
- Sends signals only to the supplied owned process group; on non-POSIX platforms it falls
  back to the process handle's `terminate()`/`kill()` methods.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time


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
        return not _exited(process)
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin reports EPERM for a group containing only zombies.
        return not _exited(process)
    return True


def _signal_group(process, signum) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    method = process.terminate if signum == signal.SIGTERM else process.kill
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
    """Terminate the owned process group in two bounded stages.

    Returns the final exit code when available. Raises `TimeoutError` only when
    the group still exists after the full `SIGKILL` reap window.
    """
    for name, value in (("term_wait", term_wait), ("kill_wait", kill_wait)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if _exited(process) and not _group_alive(process):
        return getattr(process, "returncode", getattr(process, "exitcode", None))

    _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + float(term_wait)
    while time.monotonic() < deadline:
        _wait_bounded(process, min(0.05, max(0.001, deadline - time.monotonic())))
        if not _group_alive(process):
            _wait_bounded(process, 0.05)
            return getattr(process, "returncode", getattr(process, "exitcode", None))
        time.sleep(0.01)

    if _group_alive(process):
        _signal_group(process, signal.SIGKILL)
    deadline = time.monotonic() + float(kill_wait)
    while time.monotonic() < deadline:
        _wait_bounded(process, min(0.05, max(0.001, deadline - time.monotonic())))
        if not _group_alive(process):
            _wait_bounded(process, 0.05)
            return getattr(process, "returncode", getattr(process, "exitcode", None))
        time.sleep(0.01)
    raise TimeoutError("owned process group survived SIGTERM and SIGKILL")
