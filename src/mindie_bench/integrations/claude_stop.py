"""Retain the real Claude Stop identity without manufacturing a transcript or turn."""

import json
import os
import sys
from pathlib import Path

MAX_EVENT_BYTES = 1024 * 1024


def validate_stop(event):
    if not isinstance(event, dict):
        raise ValueError("native Claude Stop event must be an object")
    if event.get("hook_event_name") != "Stop" or event.get("stop_hook_active", False) is not False:
        raise ValueError("expected a nonrecursive native Claude Stop event")
    if event.get("agent_id") or event.get("agent_type"):
        raise ValueError("subagent Stop cannot stand in for the foreground Claude task")
    for key in ["session_id", "prompt_id", "transcript_path", "cwd"]:
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError(f"native Claude Stop event lacks {key}")
    if len(event["session_id"]) > 256 or len(event["prompt_id"]) > 256:
        raise ValueError("native Claude Stop identity exceeds its evidence bound")
    if not Path(event["transcript_path"]).is_absolute() or not Path(event["cwd"]).is_absolute():
        raise ValueError("native Claude Stop transcript and cwd must be absolute")
    return event


def read_stop(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_EVENT_BYTES + 1)
    if len(data) > MAX_EVENT_BYTES:
        raise ValueError("native Claude Stop event exceeds its evidence bound")
    return validate_stop(json.loads(data))


def main():
    data = sys.stdin.buffer.read(MAX_EVENT_BYTES + 1)
    if len(data) > MAX_EVENT_BYTES:
        raise ValueError("native Claude Stop event exceeds its evidence bound")
    event = validate_stop(json.loads(data))
    with Path(sys.argv[1]).open("x") as stream:
        json.dump(event, stream)
        stream.flush()
        os.fsync(stream.fileno())
    print("{}")


if __name__ == "__main__":
    main()
