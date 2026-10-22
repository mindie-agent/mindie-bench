# [Role] Tear down one allocated remote slot and extract its final patch/inventory.
# [Invariants] Only processes carrying this slot's exact environment token are killed;
#   the final artifact is written even when diff extraction fails.
# [Timeouts & Locks] Probe and Git subprocesses are bounded at 10s/60s; NPU drain
#   polling is bounded to 20 attempts at 0.5s.
# [Side-Effect Policy] Kills only this slot's owned processes, removes only this
#   slot's worktree metadata, and writes `<slot>.final.json`.

import base64
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

cfg=json.loads(sys.argv[1])
path=pathlib.Path(cfg["path"])
errors=[]
diff_paths=["--",".",*[":(exclude)"+name for name in cfg["source_projection"]]]
token=("MINDIE_BENCH_SLOT_ID="+path.name).encode()
# Only this allocated slot's exact environment token is eligible for cleanup.
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():
        continue
    try:
        if proc.stat().st_uid!=os.getuid():
            continue
        fd=os.pidfd_open(int(proc.name))
    except (FileNotFoundError, ProcessLookupError):
        continue
    try:
        if token in (proc/"environ").read_bytes().split(b"\0"):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass
    except Exception as exc:
        errors.append(type(exc).__name__+": "+str(exc))
    finally:
        os.close(fd)
probe=None
probe_errors=[]
for attempt in range(20):
    try:
        probe=json.loads(subprocess.run(cfg["probe_command"],check=True,capture_output=True,
            text=True,timeout=10).stdout)
        if all(probe[str(d)]["used_mb"]<cfg["memory_limit_mb"] and not probe[str(d)]["pids"]
               for d in cfg["devices"]):
            break
    except Exception as exc:
        probe_errors.append(type(exc).__name__+": "+str(exc))
        break
    time.sleep(.5)
final={"probe":probe,"errors":errors}
if probe_errors:
    final["probe_errors"]=probe_errors
if path.is_dir():
    try:
        common=subprocess.run(["git","-C",cfg["repo_path"],"rev-parse","--git-common-dir"],
            check=True,capture_output=True,text=True,timeout=60).stdout.strip()
        common=pathlib.Path(common)
        if not common.is_absolute():
            common=pathlib.Path(cfg["repo_path"])/common
        matches=[p for p in (common/"worktrees").iterdir()
                 if (p/"gitdir").read_text().strip()==str(path/".git")]
        if len(matches)!=1:
            raise RuntimeError("cannot identify this slot's worktree metadata")
        gitfile=path/".git"
        if gitfile.is_symlink() or gitfile.is_file():
            gitfile.unlink()
        if gitfile.exists():
            raise RuntimeError("agent created an unexpected .git directory")
        gitfile.write_text("gitdir: "+str(matches[0])+"\n")
        patch=subprocess.run(["git","-C",str(path),"diff","--binary","--no-ext-diff","--no-textconv",
            "HEAD",*diff_paths],check=True,capture_output=True,timeout=60).stdout
        if len(patch)>32*1024*1024:
            raise RuntimeError("remote final patch exceeds artifact bound")
        final["patch_b64"]=base64.b64encode(patch).decode()
        changed=subprocess.run(["git","-C",str(path),"diff","--name-only","-z",
            "--no-ext-diff","--no-textconv","HEAD",*diff_paths],
            check=True,capture_output=True,timeout=60).stdout.split(b"\0")
        untracked=subprocess.run(["git","-C",str(path),"ls-files","--others","--exclude-standard","-z",
            *diff_paths],
            check=True,capture_output=True,timeout=60).stdout.split(b"\0")
        final["untracked"]={}
        size=0
        ignored_parts={"__pycache__",".pytest_cache",".ruff_cache",".mypy_cache"}
        ignored_suffixes={".pyc",".pyo"}
        for name in untracked:
            if not name or name.startswith(b".bench-home/"):
                continue
            rel_str=os.fsdecode(name)
            rel_path=pathlib.PurePosixPath(rel_str)
            if ignored_parts.intersection(rel_path.parts) or rel_path.suffix.lower() in ignored_suffixes:
                continue
            item=path/rel_str
            if item.is_symlink() or not item.resolve().is_relative_to(path.resolve()):
                raise RuntimeError("untracked artifact escapes slot")
            data=item.read_bytes()
            size+=len(data)
            if size>32*1024*1024:
                raise RuntimeError("remote untracked artifact bound exceeded")
            final["untracked"][rel_str]=base64.b64encode(data).decode()
        final["files_modified"]=sorted(set(
            [os.fsdecode(name) for name in changed if name]+list(final["untracked"])))
        subprocess.run(["git","-C",cfg["repo_path"],"worktree","remove","--force",str(path)],
            check=True,capture_output=True,timeout=60)
    except Exception as exc:
        final["errors"].append(type(exc).__name__+": "+str(exc))
artifact=str(path)+".final.json"
pathlib.Path(artifact).write_text(json.dumps(final))
print(json.dumps({"cleanup":final,"remote_artifact":artifact}),flush=True)
