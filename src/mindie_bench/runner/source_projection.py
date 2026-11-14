# [Role] Strips harness/Git metadata from task source snapshots before exposure.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Remove native client configuration from owned evaluation snapshots only."""

from pathlib import Path

from mindie_bench.runner.host_platform import rmtree_force

HARNESS_CONFIG_PATHS = (".claude", ".codex", ".dsh", ".kimi", ".gemini", ".agents", ".mcp.json")

PROJECTOR = r"""
import os,pathlib,shutil,stat,sys
def _ro(fn,p,_e):
    os.chmod(p,stat.S_IWRITE|stat.S_IREAD); fn(p)
root=pathlib.Path(sys.argv[1])
for name in %r:
    path=root/name
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path,onerror=_ro)
""" % (HARNESS_CONFIG_PATHS,)


def diff_paths() -> list[str]:
    return ["--", ".", *[f":(exclude){name}" for name in HARNESS_CONFIG_PATHS]]


def project_source(root: Path):
    for name in HARNESS_CONFIG_PATHS:
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree_force(path)
