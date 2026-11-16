# [Role] Tear down one allocated remote slot and extract its final patch/inventory.
# [Invariants] Only processes carrying this slot's exact environment token are killed;
#   the final artifact is written before teardown and remains valid even when teardown
#   times out. A failed worktree removal first renames the slot to an owned `.trash-*`
#   sibling, so the slot path is vacated before any unbounded filesystem work begins.
#   Only this slot's worktree and its exact `.git/worktrees/<slot>` admin directory
#   are removed; shared metadata is never swept globally.
# [Timeouts & Locks] One monotonic receipt deadline splits into diff/status, a
#   bounded teardown, and a dedicated final-receipt reserve. After the slot path
#   is atomically vacated, physical `.trash-*`/admin deletion runs in a daemon
#   thread joined only until the teardown deadline: a blocked unlink/rmdir is
#   abandoned there instead of delaying the receipt. The script emits its stdout
#   receipt before the caller's SSH wait can expire; artifact writes are atomic.
# [Side-Effect Policy] Kills only this slot's owned processes, always writes
#   `<slot>.final.json`, removes this slot's worktree plus exact admin entry,
#   atomically parks leftover tree bytes under `.trash-*`, and never mutates a
#   sibling slot's metadata.

import base64
import json
import os
import pathlib
import signal
import stat
import subprocess
import sys
import threading
import time

cfg=json.loads(sys.argv[1])
path=pathlib.Path(cfg["path"])
artifact=pathlib.Path(str(path)+".final.json")
errors=[]
diff_paths=["--",".",*[":(exclude)"+name for name in cfg["source_projection"]]]
token=("MINDIE_BENCH_SLOT_ID="+path.name).encode()
win_kw={"creationflags":subprocess.CREATE_NO_WINDOW} if os.name=="nt" else {}
started=time.monotonic()
# The caller waits POSTFLIGHT_OUTER_WAIT_SEC for this process. Keep teardown and
# receipt emission inside that outer budget even if a caller raises the plain
# postflight budget: the SSH waiter must never outlive our stdout by construction.
budget=max(.1,min(float(cfg.get("postflight_budget_s",30.0)),40.0))
outer_wait=max(.1,float(cfg.get("postflight_outer_wait_s",budget+15.0)))
artifact_reserve=min(
    max(0.0,float(cfg.get("postflight_artifact_reserve_s",5.0))),
    max(0.0,budget-.1),
)
finish_span=max(.1,min(budget,outer_wait-5.0))
receipt_deadline=started+finish_span
teardown_deadline=started+min(finish_span,max(1.0,finish_span-artifact_reserve))
teardown_budget=max(0.0,float(cfg.get("postflight_teardown_s",10.0)))
teardown_budget=min(teardown_budget,max(0.0,teardown_deadline-started))
extract_deadline=max(started,teardown_deadline-teardown_budget)
if extract_deadline<=started:
    extract_deadline=started+(teardown_deadline-started)/2


def phase_timeout(limit,until):
    remaining=until-time.monotonic()
    if remaining<=0:
        raise TimeoutError("postflight phase deadline exhausted")
    return min(limit,remaining)


def check_deadline(until):
    if time.monotonic()>=until:
        raise TimeoutError("postflight teardown deadline exhausted")


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


def _make_writable(target,mode):
    """Clear read-only bits without turning a failed chmod into a timeout."""
    try:
        os.chmod(target,mode)
    except TimeoutError:
        raise
    except OSError:
        pass


def _bounded_unlink(target,until):
    check_deadline(until)
    try:
        os.unlink(target)
        return
    except FileNotFoundError:
        return
    except TimeoutError:
        raise
    except OSError:
        pass
    _make_writable(target,stat.S_IRUSR|stat.S_IWUSR)
    _make_writable(os.path.dirname(os.fspath(target)),0o700)
    check_deadline(until)
    try:
        os.unlink(target)
    except FileNotFoundError:
        pass


def _bounded_rmdir(target,until):
    check_deadline(until)
    try:
        os.rmdir(target)
        return
    except FileNotFoundError:
        return
    except TimeoutError:
        raise
    except OSError:
        pass
    _make_writable(target,0o700)
    _make_writable(os.path.dirname(os.fspath(target)),0o700)
    check_deadline(until)
    try:
        os.rmdir(target)
    except FileNotFoundError:
        pass


def _bounded_remove_tree(root,until):
    """Delete `root` iteratively, checking the monotonic deadline before every I/O.

    A deadline hit raises TimeoutError and deliberately leaves the remaining bytes
    parked under `.trash-*`; callers record that state instead of pretending the
    filesystem is clean. Read-only files/directories get one retry after chmod.
    """
    pending=[(pathlib.Path(root),None)]
    while pending:
        check_deadline(until)
        current,iterator=pending[-1]
        if not os.path.lexists(current):
            pending.pop()
            continue
        if os.path.islink(current) or not os.path.isdir(current):
            pending.pop()
            _bounded_unlink(current,until)
            continue
        if iterator is None:
            try:
                iterator=os.scandir(current)
            except FileNotFoundError:
                pending.pop()
                continue
            except TimeoutError:
                raise
            except OSError:
                _make_writable(current,0o700)
                check_deadline(until)
                try:
                    iterator=os.scandir(current)
                except FileNotFoundError:
                    pending.pop()
                    continue
            pending[-1]=(current,iterator)
        try:
            entry=next(iterator)
        except StopIteration:
            iterator.close()
            pending.pop()
            _bounded_rmdir(current,until)
        else:
            pending.append((pathlib.Path(entry.path),None))


def _try_remove_tree(target,until,label,problems):
    try:
        _bounded_remove_tree(target,until)
    except TimeoutError:
        raise
    except OSError as exc:
        problems.append(label+": "+type(exc).__name__+": "+str(exc))


def _try_remove_path(target,until,label,problems):
    try:
        if target.is_symlink() or not target.is_dir():
            _bounded_unlink(target,until)
        else:
            _bounded_remove_tree(target,until)
    except TimeoutError:
        raise
    except FileNotFoundError:
        return
    except OSError as exc:
        problems.append(label+": "+type(exc).__name__+": "+str(exc))


def _trash_path():
    candidate=path.parent/(".trash-"+path.name+"-"+str(os.getpid()))
    if os.path.lexists(candidate):
        candidate=path.parent/(candidate.name+"-"+str(time.time_ns()))
    return candidate


def _sweep_old_trash(until,keep):
    """Reclaim older retired trees only while this postflight still has budget."""
    problems=[]
    if time.monotonic()>=until:
        return problems
    try:
        entries=list(os.scandir(path.parent))
    except FileNotFoundError:
        return problems
    except TimeoutError:
        raise
    except OSError as exc:
        problems.append("stale trash sweep: "+type(exc).__name__+": "+str(exc))
        return problems
    for entry in entries:
        if time.monotonic()>=until:
            break
        if not entry.name.startswith(".trash-"):
            continue
        candidate=pathlib.Path(entry.path)
        if keep is not None and os.path.abspath(candidate)==os.path.abspath(keep):
            continue
        try:
            _try_remove_path(candidate,until,"stale trash sweep",problems)
        except TimeoutError:
            break
    return problems


def _slot_status(trash):
    slot_gone=not os.path.lexists(path)
    trash_present=trash is not None and os.path.lexists(trash)
    return {
        "slot_vacated":slot_gone,
        "slot_removed":slot_gone and not trash_present,
        "trash_deferred":slot_gone and trash_present,
    }


def _retire_trees(trash,admin,until,result):
    """Delete already-renamed trees off the receipt path; never raise."""
    problems=result["problems"]
    try:
        if trash is not None and os.path.lexists(trash):
            try:
                _try_remove_tree(trash,until,"slot tree removal",problems)
            except TimeoutError as exc:
                problems.append("slot tree removal deferred: TimeoutError: "+str(exc))
        if admin is not None and os.path.lexists(admin):
            try:
                _try_remove_tree(admin,until,"slot worktree admin removal",problems)
            except TimeoutError as exc:
                problems.append("slot worktree admin removal: TimeoutError: "+str(exc))
        try:
            result["sweep"].extend(_sweep_old_trash(until,trash))
        except TimeoutError as exc:
            result["sweep"].append("stale trash sweep: TimeoutError: "+str(exc))
    except Exception as exc:
        problems.append("slot tree retirement: "+type(exc).__name__+": "+str(exc))


def _join_retire(worker,until):
    """Wait for retirement only inside the teardown window, never past it."""
    while worker.is_alive():
        remaining=until-time.monotonic()
        if remaining<=0:
            return False
        worker.join(min(.25,remaining))
    return True


def remove_worktree(until,admin=None):
    """Remove this slot's tree and exact admin entry; never sweep shared metadata."""
    problems=[]
    trash=None
    try:
        if path.is_symlink():
            path.unlink()
    except TimeoutError:
        raise
    except OSError as exc:
        problems.append("slot symlink removal: "+type(exc).__name__+": "+str(exc))
    gitfile=path/".git"
    if os.path.lexists(gitfile):
        try:
            _try_remove_path(gitfile,until,"agent-created .git removal",problems)
        except TimeoutError as exc:
            problems.append("agent-created .git removal: TimeoutError: "+str(exc))
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
        except TimeoutError as exc:
            problems.append("worktree metadata removal: TimeoutError: "+str(exc))
        except Exception as exc:
            problems.append("worktree metadata removal: "+type(exc).__name__+": "+str(exc))
    if admin is not None and path.is_dir():
        try:
            if gitfile.exists() or gitfile.is_symlink():
                raise RuntimeError("unexpected .git after reattach preparation")
            gitfile.write_text("gitdir: "+str(admin)+"\n",encoding="utf-8",newline="\n")
            subprocess.run(["git","-C",cfg["repo_path"],"worktree","remove","--force",str(path)],
                check=True,capture_output=True,timeout=phase_timeout(10,until),**win_kw)
        except TimeoutError as exc:
            problems.append("worktree metadata removal: TimeoutError: "+str(exc))
        except Exception as exc:
            problems.append("worktree metadata removal: "+type(exc).__name__+": "+str(exc))
    # Once Git has failed, O(1) rename the slot out of its public path before any
    # recursive deletion. The slot path is then safe to reuse or inspect even if
    # the trash tree is too large to finish within this postflight budget.
    if os.path.lexists(path):
        candidate=_trash_path()
        try:
            os.replace(path,candidate)
            trash=candidate
        except FileNotFoundError:
            pass
        except TimeoutError:
            raise
        except OSError as exc:
            problems.append("slot tree retirement: "+type(exc).__name__+": "+str(exc))
    # A stalled unlink/rmdir inside this daemon thread must never own the
    # receipt window. The main thread joins until the teardown deadline, then
    # records the parked tree as deferred and enters `finally` immediately.
    needs_retire=(
        (trash is not None and os.path.lexists(trash))
        or (admin is not None and os.path.lexists(admin))
    )
    if needs_retire:
        retire_result={"problems":[],"sweep":[]}
        retire_deferred=False
        retire_joined=False
        worker=threading.Thread(target=_retire_trees,args=(trash,admin,until,retire_result),
            daemon=True,name="postflight-trash-retirement")
        try:
            worker.start()
        except RuntimeError as exc:
            retire_deferred=True
            problems.append("slot tree retirement worker: "+type(exc).__name__+": "+str(exc))
        else:
            retire_joined=_join_retire(worker,until)
            if retire_joined:
                problems.extend(retire_result["problems"])
                problems.extend(retire_result["sweep"])
            else:
                retire_deferred=True
                problems.append(
                    "slot tree removal deferred: TimeoutError: teardown deadline exhausted")
                if admin is not None:
                    problems.append(
                        "slot worktree admin removal: TimeoutError: teardown deadline exhausted")
    else:
        retire_result={"problems":[],"sweep":[]}
        retire_deferred=False
        retire_joined=False
    if retire_deferred and trash is not None:
        # The rename above already vacated the slot; avoid another filesystem
        # stat on a stalled mount before the receipt is written.
        status={"slot_vacated":True,"slot_removed":False,"trash_deferred":True}
    else:
        status=_slot_status(trash)
    status["trash_sweep_errors"]=list(retire_result["sweep"]) if retire_joined else []
    return problems,status


# Record the exact admin entry while metadata is still cheap to resolve. This
# keeps teardown working even if patch extraction consumes the teardown reserve.
slot_admin=None
try:
    _,initial_matches=worktree_metadata(min(teardown_deadline,started+5.0))
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
final={"probe":probe,"errors":errors,"trial_id":path.name,"slot_path":str(path),
    "slot_vacated":not os.path.lexists(path),"slot_removed":False,
    "trash_deferred":False,"teardown_complete":False}
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
try:
    write_artifact(final)
except Exception as exc:
    final["errors"].append("artifact publish: "+type(exc).__name__+": "+str(exc))
teardown_limit=min(teardown_deadline,time.monotonic()+teardown_budget)
teardown_complete=False
try:
    if os.path.lexists(path) or pathlib.Path(cfg["repo_path"]).exists():
        problems,status=remove_worktree(teardown_limit,slot_admin)
        final["errors"].extend(problems)
        final.update(status)
    else:
        final.update({"slot_vacated":True,"slot_removed":True,"trash_deferred":False})
    teardown_complete=True
except TimeoutError as exc:
    final["errors"].append("worktree teardown: TimeoutError: "+str(exc))
    final.update(_slot_status(None))
except Exception as exc:
    final["errors"].append("worktree teardown: "+type(exc).__name__+": "+str(exc))
    final.update(_slot_status(None))
finally:
    final["teardown_complete"]=teardown_complete
    # The receipt reserve belongs to this rewrite even when teardown exhausted
    # its own earlier deadline; stdout still reports a reserve miss honestly.
    if time.monotonic()<receipt_deadline:
        try:
            write_artifact(final)
        except Exception as exc:
            final["errors"].append("artifact rewrite: "+type(exc).__name__+": "+str(exc))
    else:
        final["errors"].append("artifact rewrite: TimeoutError: receipt deadline exhausted")
    print(json.dumps({"cleanup":final,"remote_artifact":str(artifact)}),flush=True)
