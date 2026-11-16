# [Role] Tear down one allocated remote slot and extract its final patch/inventory.
# [Invariants] Only processes carrying this slot's exact environment token are killed;
#   the final artifact is written before teardown and remains valid even when teardown
#   times out. Only this slot's worktree and its exact `.git/worktrees/<slot>` admin
#   directory are removed; shared metadata is never swept globally.
# [Timeouts & Locks] One monotonic postflight deadline splits into diff/status,
#   a bounded 10s teardown, and a 5s artifact reserve. Git/probe subprocesses are
#   capped by the current phase deadline; artifact writes are atomic.
# [Side-Effect Policy] Kills only this slot's owned processes, always writes
#   `<slot>.final.json`, removes this slot's worktree plus exact admin entry and
#   `.bench-home` cache, and never mutates a sibling slot's metadata.

import base64
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

cfg=json.loads(sys.argv[1])
path=pathlib.Path(cfg["path"])
artifact=pathlib.Path(str(path)+".final.json")
errors=[]
diff_paths=["--",".",*[":(exclude)"+name for name in cfg["source_projection"]]]
token=("MINDIE_BENCH_SLOT_ID="+path.name).encode()
win_kw={"creationflags":subprocess.CREATE_NO_WINDOW} if os.name=="nt" else {}
started=time.monotonic()
budget=max(.1,float(cfg.get("postflight_budget_s",30.0)))
artifact_reserve=max(0.0,float(cfg.get("postflight_artifact_reserve_s",5.0)))
teardown_budget=max(0.0,float(cfg.get("postflight_teardown_s",10.0)))
artifact_deadline=started+max(.001,budget-artifact_reserve)
teardown_budget=min(teardown_budget,max(0.0,artifact_deadline-started))
extract_deadline=max(started,artifact_deadline-teardown_budget)
if extract_deadline<=started:
    extract_deadline=started+(artifact_deadline-started)/2


def phase_timeout(limit,until):
    remaining=until-time.monotonic()
    if remaining<=0:
        raise TimeoutError("postflight phase deadline exhausted")
    return min(limit,remaining)


def write_artifact(payload):
    """Publish the receipt atomically; the direct write is a last-resort fallback."""
    data=json.dumps(payload)
    temp=artifact.with_name(artifact.name+".tmp-"+str(os.getpid()))
    try:
        with temp.open("w",encoding="utf-8",newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,artifact)
    except Exception:
        try:
            temp.unlink()
        except Exception:
            pass
        artifact.write_text(data,encoding="utf-8",newline="\n")


def worktree_metadata(until):
    common=subprocess.run(["git","-C",cfg["repo_path"],"rev-parse","--git-common-dir"],
        check=True,capture_output=True,text=True,encoding="utf-8",
        timeout=phase_timeout(10,until),**win_kw).stdout.strip()
    common=pathlib.Path(common)
    if not common.is_absolute():
        common=pathlib.Path(cfg["repo_path"])/common
    common=common.resolve()
    worktrees=common/"worktrees"
    if not worktrees.is_dir():
        return common,[]
    expected=os.path.normcase(os.path.normpath(str(path/".git")))
    matches=[]
    for entry in worktrees.iterdir():
        try:
            target=(entry/"gitdir").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            # Another slot's Git process does not participate in our lock.
            continue
        candidate=pathlib.Path(target)
        if not candidate.is_absolute():
            candidate=entry/candidate
        if os.path.normcase(os.path.normpath(str(candidate)))==expected:
            matches.append(entry)
    return common,matches


def remove_worktree(until,admin=None):
    """Remove this slot's tree and exact admin entry; never sweep shared metadata."""
    problems=[]
    try:
        if path.is_symlink():
            path.unlink()
    except Exception as exc:
        problems.append("slot symlink removal: "+type(exc).__name__+": "+str(exc))
    gitfile=path/".git"
    try:
        if gitfile.is_symlink() or gitfile.is_file():
            gitfile.unlink()
        elif gitfile.is_dir():
            # The agent may have turned the hidden pointer into a real repository.
            shutil.rmtree(gitfile)
    except Exception as exc:
        problems.append("agent-created .git removal: "+type(exc).__name__+": "+str(exc))
    if admin is not None and admin.parent.name!="worktrees":
        problems.append("refusing to remove metadata outside this slot's admin entry")
        admin=None
    if admin is None:
        try:
            _,matches=worktree_metadata(until)
            if len(matches)==1:
                admin=matches[0].resolve()
            elif matches:
                raise RuntimeError("cannot identify this slot's exact worktree metadata")
        except Exception as exc:
            problems.append("worktree metadata removal: "+type(exc).__name__+": "+str(exc))
    if admin is not None and path.is_dir():
        try:
            if gitfile.exists() or gitfile.is_symlink():
                raise RuntimeError("unexpected .git after reattach preparation")
            gitfile.write_text("gitdir: "+str(admin)+"\n",encoding="utf-8",newline="\n")
            subprocess.run(["git","-C",cfg["repo_path"],"worktree","remove","--force",str(path)],
                check=True,capture_output=True,timeout=phase_timeout(10,until),**win_kw)
        except Exception as exc:
            problems.append("worktree metadata removal: "+type(exc).__name__+": "+str(exc))
    if path.exists():
        try:
            shutil.rmtree(path)
        except Exception as exc:
            problems.append("slot tree removal: "+type(exc).__name__+": "+str(exc))
    if admin is not None:
        try:
            if admin.exists():
                shutil.rmtree(admin)
        except Exception as exc:
            problems.append("slot worktree admin removal: "+type(exc).__name__+": "+str(exc))
    return problems


# Record the exact admin entry while metadata is still cheap to resolve. This
# keeps teardown working even if patch extraction consumes the teardown reserve.
slot_admin=None
try:
    _,initial_matches=worktree_metadata(min(artifact_deadline,started+5.0))
    if len(initial_matches)==1:
        slot_admin=initial_matches[0].resolve()
except Exception:
    slot_admin=None

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
    if time.monotonic()>=extract_deadline:
        break
    try:
        probe=json.loads(subprocess.run(cfg["probe_command"],check=True,capture_output=True,
            text=True,timeout=phase_timeout(10,extract_deadline)).stdout)
        if all(probe[str(d)]["used_mb"]<cfg["memory_limit_mb"] and not probe[str(d)]["pids"]
               for d in cfg["devices"]):
            break
    except Exception as exc:
        probe_errors.append(type(exc).__name__+": "+str(exc))
        break
    if time.monotonic()+.5>=extract_deadline:
        break
    time.sleep(.5)
final={"probe":probe,"errors":errors}
if probe_errors:
    final["probe_errors"]=probe_errors
if path.is_dir():
    try:
        _,matches=worktree_metadata(extract_deadline)
        if len(matches)!=1:
            raise RuntimeError("cannot identify this slot's worktree metadata")
        gitfile=path/".git"
        if gitfile.is_symlink() or gitfile.is_file():
            gitfile.unlink()
        if gitfile.exists():
            raise RuntimeError("agent created an unexpected .git directory")
        gitfile.write_text("gitdir: "+str(matches[0].resolve())+"\n",encoding="utf-8",newline="\n")
        patch=subprocess.run(["git","-C",str(path),"diff","--binary","--no-ext-diff","--no-textconv",
            "HEAD",*diff_paths],check=True,capture_output=True,
            timeout=phase_timeout(60,extract_deadline),**win_kw).stdout
        if len(patch)>32*1024*1024:
            raise RuntimeError("remote final patch exceeds artifact bound")
        final["patch_b64"]=base64.b64encode(patch).decode()
        changed=subprocess.run(["git","-C",str(path),"diff","--name-only","-z",
            "--no-ext-diff","--no-textconv","HEAD",*diff_paths],check=True,capture_output=True,
            timeout=phase_timeout(60,extract_deadline),**win_kw).stdout.split(b"\0")
        untracked=subprocess.run(["git","-C",str(path),"ls-files","--others","--exclude-standard","-z",
            *diff_paths],check=True,capture_output=True,
            timeout=phase_timeout(60,extract_deadline),**win_kw).stdout.split(b"\0")
        final["untracked"]={}
        size=0
        ignored_parts={"__pycache__",".pytest_cache",".ruff_cache",".mypy_cache"}
        ignored_suffixes={".pyc",".pyo"}
        for name in untracked:
            if time.monotonic()>=extract_deadline:
                raise TimeoutError("postflight artifact extraction deadline exhausted")
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
    except Exception as exc:
        final["errors"].append(type(exc).__name__+": "+str(exc))
# Publish before teardown: a hanging filesystem unlink or Git removal can then
# never prevent the outer SSH caller from receiving a complete receipt.
write_artifact(final)
teardown_deadline=min(artifact_deadline,time.monotonic()+teardown_budget)
if path.exists() or path.is_symlink() or pathlib.Path(cfg["repo_path"]).exists():
    try:
        final["errors"].extend(remove_worktree(teardown_deadline,slot_admin))
    except Exception as exc:
        final["errors"].append("worktree teardown: "+type(exc).__name__+": "+str(exc))
    if time.monotonic()<artifact_deadline:
        write_artifact(final)
print(json.dumps({"cleanup":final,"remote_artifact":str(artifact)}),flush=True)
