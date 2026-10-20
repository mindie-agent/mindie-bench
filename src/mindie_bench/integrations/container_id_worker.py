"""Container-ID worker sent over `run_remote_python` stdin.

[Role] Resolve one container name to its immutable Docker ID.
[Invariants] A failed inspect aborts (`check=True`); the caller never treats a
missing ID as an empty container.
[Timeouts & Locks] `docker inspect` is capped at 15s; no locks.
[Side-Effect Policy] Read-only `docker inspect`; prints JSON to stdout.
"""

import json
import subprocess
import sys

request = json.load(sys.stdin)
identity = subprocess.run(
    ["docker", "inspect", "--format", "{{.Id}}", request["container"]],
    capture_output=True,
    text=True,
    check=True,
    timeout=15,
).stdout.strip()
print(json.dumps(dict(container_id=identity)))
