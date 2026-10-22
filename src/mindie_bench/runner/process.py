"""Bounded foreground ownership, streaming journals, and explicit cleanup errors.

[Role]
Runs one native agent/capture command under the `owner.py` lifetime guard
(`stream_process`), forwards the prompt over a non-blocking pipe, streams
stdout/stderr into the trial journal, and reports an explicit stop reason.

[Invariants]
- `timeout`, cancellation, `gateway.unknown`, and infrastructure errors are the
  only stop reasons; a completed business result is never overwritten by later
  cleanup or receipt trouble.
- Reader/JSONL failures land in `ProcessResult.errors`; a truncated prompt is a
  `BrokenPipeError`, never a silently successful turn.

[Timeouts & Locks]
- The `timeout` loop is the outer budget; teardown closes the owner pipe, then
  sends `SIGTERM` to the owned group, waits 2s, sends `SIGKILL`, waits 5s, and
  joins readers 1s each. The event queue is bounded (`maxsize=32`) so a stalled
  consumer cannot grow memory without bound.

[Side-Effect Policy]
- Creates `output`/`stderr` journals exclusively (`open("x")`) and owns the
  child process group; it never mutates the workspace or remote state.
"""

from __future__ import annotations

import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from mindie_bench.runner.termination import terminate_process_group


@dataclass
class ProcessResult:
    returncode: int | None = None
    elapsed: float = 0
    stop_reason: str | None = None
    errors: list[dict[str, str]] = field(default_factory=list)
    started: bool = False
    started_monotonic_sec: float | None = None


def stream_process(
    command: list[str],
    *,
    env: dict,
    cwd: Path,
    prompt: str,
    output: Path,
    stderr: Path,
    timeout: float,
    max_bytes: int,
    consume,
    gateway,
    cancelled=None,
) -> ProcessResult:
    result = ProcessResult()
    read_fd, owner_fd = os.pipe()
    process = None
    readers = []
    events = queue.Queue(maxsize=32)
    readers_stopped = threading.Event()
    count = 0
    started = time.monotonic()
    result.started_monotonic_sec = started
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "mindie_bench.runner.owner", str(read_fd), *command],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(read_fd,),
        )
        result.started = True
        os.close(read_fd)
        read_fd = -1

        def enqueue(value):
            while not readers_stopped.is_set():
                try:
                    events.put(value, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def reader(stream, kind):
            try:
                while chunk := stream.readline(1024 * 1024 + 1):
                    enqueue((kind, chunk, time.monotonic()))
            except Exception as exc:
                enqueue(("error", f"{type(exc).__name__}: {exc}".encode(), time.monotonic()))
            finally:
                enqueue((kind, None, time.monotonic()))

        for stream, kind in [(process.stdout, "stdout"), (process.stderr, "stderr")]:
            thread = threading.Thread(target=reader, args=(stream, kind), daemon=True)
            thread.start()
            readers.append(thread)
        pending_prompt = memoryview(prompt.encode())
        os.set_blocking(process.stdin.fileno(), False)
        ended = 0
        with output.open("x", encoding="utf-8") as raw, stderr.open("xb") as err:
            while ended < 2:
                elapsed = time.monotonic() - started
                if cancelled is not None and cancelled.is_set():
                    result.stop_reason = "cancelled"
                    break
                if gateway.unknown:
                    result.stop_reason = "uncertain_external_effects"
                    break
                if gateway.infrastructure_error:
                    result.stop_reason = "infrastructure_error"
                    break
                if gateway.stop_reason:
                    result.stop_reason = gateway.stop_reason
                    break
                if elapsed >= timeout:
                    result.stop_reason = "timeout"
                    break
                # Submit incrementally while still draining output and checking
                # cancellation/deadlines. A native CLI may stop reading stdin.
                for _ in range(16):
                    if not pending_prompt or not select.select([], [process.stdin], [], 0)[1]:
                        break
                    try:
                        written = os.write(process.stdin.fileno(), pending_prompt[:65536])
                    except BlockingIOError:
                        break
                    if not written:
                        raise BrokenPipeError("native process accepted no prompt bytes")
                    pending_prompt = pending_prompt[written:]
                    if time.monotonic() - started >= timeout:
                        break
                if not pending_prompt and not process.stdin.closed:
                    process.stdin.close()
                try:
                    kind, chunk, observed = events.get(
                        timeout=min(0.01 if pending_prompt else 0.1, timeout - elapsed)
                    )
                except queue.Empty:
                    if process.poll() is not None and all(not t.is_alive() for t in readers):
                        break
                    continue
                if chunk is None:
                    ended += 1
                    continue
                count += len(chunk)
                if count > max_bytes or len(chunk) > 1024 * 1024:
                    result.stop_reason = "output_limit"
                    break
                if kind == "error":
                    raise RuntimeError(chunk.decode())
                if kind == "stderr":
                    err.write(chunk)
                    err.flush()
                    continue
                # Flush before interpretation: malformed or unknown terminal
                # events cannot erase already-observed native evidence.
                text = chunk.decode("utf-8")
                raw.write(
                    json.dumps(dict(observed_sec=observed - started, native_line=text.rstrip("\n")))
                    + "\n"
                )
                raw.flush()
                event = json.loads(text)
                if not isinstance(event, dict):
                    raise ValueError("native JSONL event is not an object")
                consume(event, observed - started, gateway.model_calls)
        if result.stop_reason is None:
            if pending_prompt:
                raise BrokenPipeError("native process ended before receiving the complete prompt")
            result.returncode = process.wait(timeout=2)
    except Exception as exc:
        result.errors.append(dict(stage="agent", type=type(exc).__name__, message=str(exc)))
    finally:
        readers_stopped.set()
        result.elapsed = time.monotonic() - started
        # Closing the only owner writer kills the entire owned POSIX group,
        # even when its leader has already exited but descendants remain.
        if read_fd >= 0:
            os.close(read_fd)
        if process:
            try:
                code = terminate_process_group(process, term_wait=2, kill_wait=5)
                if result.returncode is None and result.stop_reason is None:
                    result.returncode = code
            except (TimeoutError, OSError, subprocess.SubprocessError) as exc:
                result.errors.append(
                    dict(stage="agent_cleanup", type=type(exc).__name__, message=str(exc))
                )
            for thread in readers:
                thread.join(timeout=1)
            for stream in [process.stdin, process.stdout, process.stderr]:
                stream.close()
        os.close(owner_fd)
    return result
