"""Strict, deterministic local artifact operations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tree_hash(root: Path, *, exclude: set[str] | None = None) -> str:
    files = {}
    for path in sorted(root.rglob("*")):
        if exclude and path.relative_to(root).as_posix() in exclude:
            continue
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in benchmark inputs: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
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
