# [Role] Reuse a verified KDA build package keyed by exact source/command identity.
# [Invariants] A cached package is copied only when its manifest SHA-256 matches;
#   generated build directories are excluded from the source identity. The shared
#   cache keeps at most 16 completed keyed entries, evicting the least recently
#   used (a cache hit refreshes mtime). In-flight `.tmp-<pid>` directories never
#   count as eviction candidates. Windows dead-PID probing uses
#   `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` and `GetExitCodeProcess`,
#   never `os.kill(pid, 0)` (signal 0 is a console control event on Windows).
# [Timeouts & Locks] The wrapped build subprocess is bounded at 60s; cache writes
#   use a PID-suffixed temporary directory and atomic rename, and every failed
#   write removes its temporary directory. A dead-PID `.tmp-<pid>` directory is
#   reclaimed only after it is older than 1h and its PID exited, via atomic
#   `.trash-*` rename.
# [Side-Effect Policy] Writes only under `$HOME/cache/kda_build_out` and the slot's
#   own `csrc/build_out`; no network or host-state mutation.

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

GENERATED = {"build", "build_out", "output"}
MAX_CACHE_ENTRIES = 16
STALE_TMP_AGE_SEC = 60 * 60
TRASH_PREFIX = ".trash-"
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
ERROR_ACCESS_DENIED = 5
ERROR_INVALID_PARAMETER = 87


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def current_sources(root):
    return {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.relative_to(root).parts[0] not in GENERATED and not p.is_symlink() and p.is_file()
    }


def _windows_kernel32():
    """Load and type the process-query API without ever sending a signal."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    return kernel32, ctypes.get_last_error


def _windows_pid_exited(pid, kernel32, get_last_error):
    """Return True only when Windows positively proves the PID has exited.

    `OpenProcess` failing with ERROR_INVALID_PARAMETER is the only proof that no
    process currently owns the PID. Access-denied and every other failure are
    conservative "alive/unknown" answers, so a live build is never evicted.
    """
    import ctypes
    from ctypes import wintypes

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return get_last_error() == ERROR_INVALID_PARAMETER
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value != STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_exited(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        kernel32, get_last_error = _windows_kernel32()
        return _windows_pid_exited(pid, kernel32, get_last_error)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except TimeoutError:
        raise
    except OSError:
        return False
    return False


def _temporary_pid(name):
    marker = ".tmp-"
    if marker not in name:
        return None
    try:
        return int(name.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _is_temporary_name(name):
    return name.startswith(".tmp") or ".tmp-" in name or name.endswith(".tmp")


def _discard_stale_temporary(path):
    """Atomically rename a dead temporary directory before deleting it."""
    trash = path.with_name(
        TRASH_PREFIX + path.name + "-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    try:
        path.rename(trash)
    except TimeoutError:
        raise
    except OSError:
        return
    shutil.rmtree(trash, ignore_errors=True)


def prune_cache(root, current):
    entries = []
    stale_temporaries = []
    now = time.time()
    for path in root.iterdir():
        if path == current or not path.is_dir() or path.is_symlink():
            continue
        if path.name.startswith(TRASH_PREFIX):
            continue
        if _is_temporary_name(path.name):
            pid = _temporary_pid(path.name)
            if pid is None:
                continue
            try:
                aged = now - path.stat().st_mtime >= STALE_TMP_AGE_SEC
            except TimeoutError:
                raise
            except OSError:
                continue
            if aged and _pid_exited(pid):
                stale_temporaries.append(path)
            continue
        try:
            entries.append((path.stat().st_mtime, path))
        except TimeoutError:
            raise
        except OSError:
            continue
    entries.sort(reverse=True)
    for _, path in entries[MAX_CACHE_ENTRIES - 1:]:
        shutil.rmtree(path, ignore_errors=True)
    for path in stale_temporaries:
        _discard_stale_temporary(path)


def main():
    receipt_path, csrc, cache_dir = pathlib.Path("../build-receipt.json"), pathlib.Path("csrc"), None
    if receipt_path.is_file() and csrc.is_dir():
        try:
            rec = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (rec.get("schema") == "mindie-bench-kda-source-build/1"
                    and rec.get("status") == "prepared"
                    and current_sources(csrc) == rec.get("source_files_sha256")):
                identity = {"source_files_sha256": rec["source_files_sha256"],
                            "input_archives_sha256": rec.get("input_archives_sha256"),
                            "generic_dependency": rec.get("generic_dependency"),
                            "command": rec.get("command"), "args": sys.argv[1:]}
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                home = (os.environ.get("HOME") or os.environ.get("USERPROFILE")
                        or str(pathlib.Path.home()))
                cache_dir = pathlib.Path(home) / "cache" / "kda_build_out" / key
                manifest_path = cache_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    cached_pkg = cache_dir / manifest["name"]
                    if cached_pkg.is_file() and sha256(cached_pkg) == manifest["sha256"]:
                        build_out = csrc / "build_out"
                        build_out.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cached_pkg, build_out / manifest["name"])
                        try:
                            os.utime(cache_dir, None)
                        except TimeoutError:
                            raise
                        except OSError:
                            pass
                        return 0
        except TimeoutError:
            raise
        except Exception:
            cache_dir = None
    bash_bin = ("/bin/bash" if pathlib.Path("/bin/bash").is_file()
                else (shutil.which("bash") or "bash"))
    win_kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    res = subprocess.run([bash_bin, *sys.argv[1:]], timeout=60, **win_kw)
    if res.returncode == 0 and cache_dir is not None:
        tmp_dir = None
        try:
            pkgs = list((csrc / "build_out").glob("*.run"))
            if len(pkgs) == 1:
                pkg = pkgs[0]
                tmp_dir = cache_dir.with_suffix(".tmp-" + str(os.getpid()))
                tmp_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pkg, tmp_dir / pkg.name)
                (tmp_dir / "manifest.json").write_text(
                    json.dumps({"name": pkg.name, "sha256": sha256(pkg)}),
                    encoding="utf-8",
                )
                if not cache_dir.exists():
                    tmp_dir.rename(cache_dir)
                else:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                prune_cache(cache_dir.parent, cache_dir)
        except TimeoutError:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        except Exception:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
