# [Role] Strips harness/Git metadata from task source snapshots before exposure.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Remove native client configuration from owned evaluation snapshots only."""

from pathlib import Path

HARNESS_CONFIG_PATHS = (".claude", ".codex", ".dsh", ".kimi", ".gemini", ".agents", ".mcp.json")

PROJECTOR = r"""
import pathlib,shutil,sys
root=pathlib.Path(sys.argv[1])
for name in %r:
    path=root/name
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
""" % (HARNESS_CONFIG_PATHS,)


def diff_paths() -> list[str]:
    return ["--", ".", *[f":(exclude){name}" for name in HARNESS_CONFIG_PATHS]]


def project_source(root: Path):
    import shutil

    for name in HARNESS_CONFIG_PATHS:
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
