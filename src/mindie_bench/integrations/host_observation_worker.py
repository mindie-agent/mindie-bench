"""Remote host-observation worker sent over `run_remote_python` stdin.

[Role] Read one JSON request, attest the PID-namespace owner, run bounded
`npu-smi info`, and print owned/live PIDs as JSON.
[Invariants] Only processes in the attested PID namespace count as owned; a
recycled PID or changed owner aborts instead of silently widening ownership.
[Timeouts & Locks] `npu-smi` is capped at 20s; no local locks (one request).
[Side-Effect Policy] Read-only `/proc` and `npu-smi` observation; prints JSON
to stdout only, never writes host state.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

request = json.load(sys.stdin)
boot_id = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
owner = request.get("owner")


def snapshot():
    result = {}
    for process in pathlib.Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
            result[int(process.name)] = dict(
                start_ticks=fields[19],
                pid_namespace=os.readlink(process / "ns/pid"),
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


before = snapshot()
if owner and request["require_owner"]:
    if boot_id != owner["boot_id"] or before.get(owner["pid"]) != dict(
        start_ticks=owner["start_ticks"], pid_namespace=owner["pid_namespace"]
    ):
        raise RuntimeError("the attested namespace owner is no longer present")
result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
if result.returncode:
    raise RuntimeError("npu-smi info failed: " + result.stderr[-2000:])
after = snapshot()
if owner and request["require_owner"] and after.get(owner["pid"]) != before[owner["pid"]]:
    raise RuntimeError("the attested namespace owner changed during observation")
owned = set()
if owner:
    for pid in before.keys() | after.keys():
        prior, current = before.get(pid), after.get(pid)
        # The namespace owns descendants even when remote-dev replaces its
        # per-job environment token. A recycled PID never inherits ownership.
        if prior and current and prior != current:
            continue
        identity = current or prior
        if identity["pid_namespace"] == owner["pid_namespace"]:
            owned.add(pid)
print(
    json.dumps(
        dict(
            npu_smi=result.stdout,
            owned_pids=sorted(owned),
            live_pids=sorted(after),
            observed_at=time.time(),
            boot_id=boot_id,
            ownership_basis="attested PID namespace and start ticks; before/after sample",
        )
    )
)
