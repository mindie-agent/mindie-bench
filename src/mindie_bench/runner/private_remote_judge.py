# ruff: noqa: E701
# [Role] Execute the private Oracle probes and judge inside the remote slot.
# [Invariants] Files are staged read-only, SHA-256 inventory is checked before and
#   after, negative controls must match exact exit/stdout, and the judge outcome is
#   preserved even if post-judge cleanup fails.
# [Timeouts & Locks] Overall deadline is the caller timeout; probes/judge reserve
#   `inventory_grace = 5.0` seconds for the final inventory instead of resetting it.
# [Side-Effect Policy] Writes only inside `MINDIE_BENCH_ORACLE_ROOT`; the caller's
#   EXIT trap removes that owned directory.

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

payload=json.loads(base64.b64decode(sys.argv[1], validate=True))
root=pathlib.Path(os.environ["MINDIE_BENCH_ORACLE_ROOT"])
overall_deadline=time.monotonic()+max(.001, float(payload["timeout"]))
# The post-run inventory must always run. Reserve a fixed slice of the outer
# worker budget for it instead of granting probes and judge a fresh deadline.
inventory_grace=min(5.0, max(.001, float(payload["timeout"]) / 2.0))
deadline=overall_deadline
receipt=dict(schema_version="mindie-bench-private-oracle/1",
    nonce=os.environ["MINDIE_BENCH_PERFORMANCE_NONCE"],
    manifest_sha256=os.environ["MINDIE_BENCH_ORACLE_SHA256"],
    probes=[], judge=None, before=None, after=None, tampered=None, original_error=None,
    timed_out=False)
class IntegrityError(Exception): pass
def inventory():
    result={}
    for path in root.rglob("*"):
        if path.is_symlink(): raise IntegrityError("private oracle symlink: "+str(path.relative_to(root)))
        if path.is_file():
            result[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    if result!=payload["expected"]:
        changed=sorted(k for k in result.keys()|payload["expected"].keys()
                       if result.get(k)!=payload["expected"].get(k))
        raise IntegrityError("private oracle SHA-256 changed: "+", ".join(changed))
    return dict(status="intact", file_count=len(result))
def execute(command):
    result=subprocess.run(command, capture_output=True,
                          timeout=max(.001, deadline-time.monotonic()),
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return dict(returncode=result.returncode,
                stdout_b64=base64.b64encode(result.stdout).decode(),
                stderr_b64=base64.b64encode(result.stderr).decode(),
                stdout_sha256=hashlib.sha256(result.stdout).hexdigest())
try:
    for relative, encoded in payload["files"].items():
        rel=pathlib.PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts: raise IntegrityError("invalid private oracle path")
        path=root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded, validate=True))
        path.chmod(0o444)
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(0o555)
    root.chmod(0o555)
    receipt["before"]=inventory()
    deadline=max(time.monotonic()+.001, overall_deadline-inventory_grace)
    for probe in payload["probes"]:
        command=[part.replace("{oracle}", str(root)) for part in probe["command"]]
        observed=execute(command)
        observed["name"]=probe["name"]
        receipt["probes"].append(observed)
        if (observed["returncode"]!=probe["expected_returncode"]
            or observed["stdout_sha256"]!=probe["stdout_sha256"]):
            raise IntegrityError("oracle negative control "+repr(probe["name"])+" failed: exit="
                +str(observed["returncode"])+", stdout_sha256="+observed["stdout_sha256"])
    receipt["judge"]=execute(["/bin/bash", str(root/"verifier/test.sh")])
except IntegrityError as error:
    receipt["tampered"]=str(error)
except subprocess.TimeoutExpired as error:
    receipt["timed_out"]=True
    receipt["original_error"]=dict(type=type(error).__name__, message=str(error))
except Exception as error:
    receipt["original_error"]=dict(type=type(error).__name__, message=str(error))
finally:
    try: receipt["after"]=inventory()
    except IntegrityError as error: receipt["tampered"]=str(error)
    except Exception as error:
        receipt["original_error"]=receipt["original_error"] or dict(type=type(error).__name__,message=str(error))
print(json.dumps(receipt, allow_nan=False), flush=True)
sys.exit(2 if receipt["original_error"] else 0)
