"""Persisted postflight-receipt reconciliation and stale-artifact sweep probes.

[Role]
Read the durable `<slot>.final.json` receipt after postflight stdout was lost, and
remove only this slot's artifact plus safely aged sibling artifacts.

[Invariants]
- Reconciliation is read-only and accepts a receipt only when both `trial_id` and
  the canonical slot path match this exact slot. Canonicalization treats `/` and
  backslash as equivalent so a Windows-local unit test and the remote POSIX probe
  agree.
- The sweep removes the owned artifact unconditionally; any other candidate must
  belong to this user, be older than the age bound, and have no live worktree or
  `MINDIE_BENCH_SLOT_ID` process.

[Timeouts & Locks]
Both scripts run under `RemoteSlot`'s bounded SSH waits (`10.0s` reconciliation,
`15.0s` artifact sweep). Neither script starts a child process or waits on a lock.

[Side-Effect Policy]
Reconciliation never writes or deletes. The sweep deletes only `.final.json`
files selected by the ownership, age, and liveness rules above.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath


def canonical_slot_path(value: object) -> str | None:
    """Normalize mixed POSIX/Windows separators to one comparable identity."""
    if not isinstance(value, str) or not value:
        return None
    return os.path.normcase(PurePosixPath(value.replace("\\", "/")).as_posix())


RECONCILE_FINAL_ARTIFACT = r"""
import json, os, pathlib, sys
def canonical(value):
    if not isinstance(value, str) or not value:
        return None
    return os.path.normcase(pathlib.PurePosixPath(value.replace("\\", "/")).as_posix())
slot = pathlib.Path(sys.argv[1]); trial_id = slot.name
try:
    payload = json.loads(slot.with_name(trial_id + ".final.json").read_text(encoding="utf-8"))
except TimeoutError:
    raise
except (OSError, ValueError):
    raise SystemExit(3)
if (not isinstance(payload, dict) or payload.get("trial_id") != trial_id
        or canonical(payload.get("slot_path")) != canonical(sys.argv[1])):
    raise SystemExit(4)
print(json.dumps(payload))
"""

# Runs inside the remote container. The owned artifact is removed unconditionally;
# other candidates must be this user's, older than the age bound, and not backed by
# a live slot worktree or `MINDIE_BENCH_SLOT_ID` process.
REMOTE_FINAL_ARTIFACT_SWEEP = r"""
import json, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
owned = sys.argv[2]
cutoff = time.time() - float(sys.argv[3])
uid = getattr(os, "getuid", lambda: None)()
proc_root = pathlib.Path("/proc")
def active(trial):
    if (root / trial).exists():
        return True
    if not proc_root.is_dir():
        return False
    token = b"MINDIE_BENCH_SLOT_ID=" + os.fsencode(trial)
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if uid is not None and proc.stat().st_uid != uid:
                continue
            if token in (proc / "environ").read_bytes().split(b"\0"):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False
removed, warnings = [], []
for artifact in sorted(root.glob("*.final.json")):
    try:
        stat = artifact.stat()
    except TimeoutError:
        raise
    except OSError:
        continue
    if uid is not None and stat.st_uid != uid:
        continue
    is_owned = artifact.name == owned
    if not is_owned and (stat.st_mtime > cutoff or active(artifact.name[:-len(".final.json")])):
        continue
    try:
        artifact.unlink()
        removed.append(artifact.name)
    except TimeoutError:
        raise
    except OSError as exc:
        warnings.append(artifact.name + ": " + type(exc).__name__ + ": " + str(exc))
print(json.dumps({"removed": removed, "warnings": warnings}))
"""
