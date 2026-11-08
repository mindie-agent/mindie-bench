# [Role] Provides strict deterministic hashing, JSON, and safe-path artifact helpers.
# [Invariants] Fail closed: missing, unreadable, or uncertain evidence is never
#   hashed as a partial tree or reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Strict, deterministic local artifact operations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

IGNORED_CACHE_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})
IGNORED_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})


def is_ignored_cache_path(rel_path: Path) -> bool:
    return (
        bool(IGNORED_CACHE_PARTS.intersection(rel_path.parts))
        or rel_path.suffix.lower() in IGNORED_CACHE_SUFFIXES
    )


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading its full contents into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hash(root: Path, *, exclude: set[str] | None = None) -> str:
    """Hash regular files under root. Missing, unreadable, or uncertain trees raise.

    Do not use `os.walk`: it treats `DirEntry.is_dir()` `OSError` as a non-directory
    and silently omits that subtree. Classify every entry with `lstat` and fail closed.
    """
    files = {}
    root = Path(root)
    mode = root.lstat().st_mode
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(root)

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            found = sorted(entries, key=lambda entry: entry.name)
        for entry in found:
            rel = Path(entry.path).relative_to(root)
            if entry.name == ".git" or is_ignored_cache_path(rel):
                continue
            st = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                continue
            if stat.S_ISDIR(st.st_mode):
                visit(Path(entry.path))
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            posix = rel.as_posix()
            if exclude and posix in exclude:
                continue
            files[posix] = sha256_file(Path(entry.path))

    visit(root)
    return digest(files)


def task_hash(root: Path) -> str:
    return tree_hash(root, exclude={"soundness_receipt.json"})


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"expected a confined relative path: {value!r}")
    return value
