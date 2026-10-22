# [Role] Reads the Codex Stop-hook artifact used by native capture.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Retain an actual native Stop event for the isolated post-turn consumer."""

import json
import os
import sys
from pathlib import Path


def main():
    destination = Path(sys.argv[1])
    data = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("native Stop event exceeds its evidence bound")
    event = json.loads(data)
    if event.get("hook_event_name") != "Stop" or event.get("stop_hook_active", False):
        raise ValueError("expected a nonrecursive native Stop event")
    for key in ["session_id", "turn_id", "transcript_path", "cwd"]:
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError(f"native Stop event lacks {key}")
    with destination.open("x") as stream:
        json.dump(event, stream)
        stream.flush()
        os.fsync(stream.fileno())
    print("{}")


if __name__ == "__main__":
    main()
