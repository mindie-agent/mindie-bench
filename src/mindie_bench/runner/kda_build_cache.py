# [Role] Reuse a verified KDA build package keyed by exact source/command identity.
# [Invariants] A cached package is copied only when its manifest SHA-256 matches;
#   generated build directories are excluded from the source identity.
# [Timeouts & Locks] The wrapped build subprocess is bounded at 60s; cache writes
#   use a PID-suffixed temporary directory and atomic rename.
# [Side-Effect Policy] Writes only under `$HOME/cache/kda_build_out` and the slot's
#   own `csrc/build_out`; no network or host-state mutation.

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

GENERATED = {"build", "build_out", "output"}
def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
def current_sources(root):
    return {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.relative_to(root).parts[0] not in GENERATED and not p.is_symlink() and p.is_file()
    }
receipt_path, csrc, cache_dir = pathlib.Path("../build-receipt.json"), pathlib.Path("csrc"), None
if receipt_path.is_file() and csrc.is_dir():
    try:
        rec = json.loads(receipt_path.read_text())
        if (rec.get("schema") == "mindie-bench-kda-source-build/1" and rec.get("status") == "prepared"
                and current_sources(csrc) == rec.get("source_files_sha256")):
            identity = {"source_files_sha256": rec["source_files_sha256"],
                        "input_archives_sha256": rec.get("input_archives_sha256"),
                        "generic_dependency": rec.get("generic_dependency"),
                        "command": rec.get("command"), "args": sys.argv[1:]}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cache_dir = pathlib.Path(os.environ["HOME"]) / "cache" / "kda_build_out" / key
            manifest_path = cache_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                cached_pkg = cache_dir / manifest["name"]
                if cached_pkg.is_file() and sha256(cached_pkg) == manifest["sha256"]:
                    build_out = csrc / "build_out"
                    build_out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached_pkg, build_out / manifest["name"])
                    raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        cache_dir = None
res = subprocess.run(["/bin/bash", *sys.argv[1:]], timeout=60)
if res.returncode == 0 and cache_dir is not None:
    try:
        pkgs = list((csrc / "build_out").glob("*.run"))
        if len(pkgs) == 1:
            pkg = pkgs[0]
            tmp_dir = cache_dir.with_suffix(".tmp-" + str(os.getpid()))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pkg, tmp_dir / pkg.name)
            (tmp_dir / "manifest.json").write_text(json.dumps({"name": pkg.name, "sha256": sha256(pkg)}))
            if not cache_dir.exists():
                tmp_dir.rename(cache_dir)
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
raise SystemExit(res.returncode)
