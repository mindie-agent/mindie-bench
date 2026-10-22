# [Role] Bridges a private knowledge MCP stdio session for one trial.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Bounded native MCP stdio relay; corpus paths never enter its configuration."""

from __future__ import annotations

import base64
import os
import select
import sys
import time

from mindie_bench.runner.ssh_shim import _post


def main():
    gateway = os.environ["MINDIE_BENCH_GATEWAY"]
    if len(sys.argv) != 2:
        raise ValueError("knowledge relay requires its registered server name")
    opened = _post(gateway, "/bench/knowledge-stdio", {"action": "open", "server": sys.argv[1]})
    identity = opened["session_id"]
    eof = False
    primary = None
    try:
        while True:
            data = b""
            close_input = False
            if not eof:
                if select.select([sys.stdin.fileno()], [], [], 0.05)[0]:
                    data = os.read(sys.stdin.fileno(), 65536)
                    eof = not data
                    close_input = eof
            else:
                time.sleep(0.05)
            result = _post(
                gateway,
                "/bench/knowledge-stdio",
                {
                    "action": "exchange",
                    "session_id": identity,
                    "stdin_b64": base64.b64encode(data).decode(),
                    "eof": close_input,
                },
            )
            sys.stdout.buffer.write(base64.b64decode(result["stdout_b64"], validate=True))
            sys.stdout.buffer.flush()
            if result["exit_code"] is not None:
                if type(result["exit_code"]) is not int:
                    raise ValueError("knowledge relay returned an invalid exit code")
                return result["exit_code"]
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            _post(gateway, "/bench/knowledge-stdio", {"action": "close", "session_id": identity})
        except Exception as exc:
            if primary is None:
                raise
            primary.add_note(f"knowledge relay close also failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
