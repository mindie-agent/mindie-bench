"""Inventory exact known KDA fixed source across recorded foreground grants."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import zipfile
from pathlib import Path

FIXED_SHA = "23c6e095b97293088e136ea55b1d3b2701189f3652e7f24ab4c86bb87b68963f"
FIXED_BYTES = 16725
FIXED_EXPRESSION = b"postWuScratchOffset+hChunkCount*shape.vHeads*chunkSize*shape.kDim*sizeof(float)"
TEXT = {".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".txt", ".md", ".cmake", ".patch", ".diff"}
ARCHIVES = (".zip", ".whl", ".egg", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")
MAX_SOURCE = 8 * 1024 * 1024
MAX_ARCHIVE = 128 * 1024 * 1024


def inventory(roots: list[Path]) -> dict:
    result = {
        "schema": "mindie-bench-kda-known-source-inventory/1",
        "roots": [str(path) for path in roots],
        "files": 0,
        "symlinks": [],
        "archives": 0,
        "known_fixed": [],
        "git_metadata": [],
        "review": [],
        "errors": [],
        "compiled_objects_count": 0,
        "universal_no_leakage_claim": False,
    }
    seen = set()

    def inspect(data: bytes, label: str, name: str, depth=0):
        if name.lower().endswith(ARCHIVES):
            result["archives"] += 1
            if depth > 2 or len(data) > MAX_ARCHIVE:
                result["review"].append({"path": label, "reason": "archive exceeds inspection bound"})
                return
            stream = io.BytesIO(data)
            try:
                if zipfile.is_zipfile(stream):
                    with zipfile.ZipFile(stream) as archive:
                        members = archive.infolist()
                        if sum(item.file_size for item in members) > MAX_ARCHIVE:
                            raise ValueError("expanded ZIP exceeds inspection bound")
                        for item in members:
                            if not item.is_dir():
                                inspect(
                                    archive.read(item),
                                    label + "!" + item.filename,
                                    item.filename,
                                    depth + 1,
                                )
                else:
                    stream.seek(0)
                    with tarfile.open(fileobj=stream) as archive:
                        total = 0
                        for item in archive:
                            total += item.size
                            if total > MAX_ARCHIVE:
                                raise ValueError("expanded TAR exceeds inspection bound")
                            if item.isfile():
                                inspect(
                                    archive.extractfile(item).read(),
                                    label + "!" + item.name,
                                    item.name,
                                    depth + 1,
                                )
                            elif item.issym() or item.islnk():
                                result["review"].append(
                                    {
                                        "path": label + "!" + item.name,
                                        "reason": "archive link",
                                        "target": item.linkname,
                                    }
                                )
            except Exception as error:
                result["review"].append({"path": label, "reason": repr(error)})
            return
        if len(data) <= MAX_SOURCE and (Path(name).suffix.lower() in TEXT or len(data) == FIXED_BYTES):
            exact = hashlib.sha256(data).hexdigest() == FIXED_SHA
            expression = FIXED_EXPRESSION in re.sub(rb"\s+", b"", data)
            if exact or expression:
                result["known_fixed"].append(
                    {
                        "path": label,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "exact_file": exact,
                        "expression": expression,
                    }
                )

    def walk(path: Path):
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                result["symlinks"].append({"path": str(path), "target": os.readlink(path)})
                try:
                    path = path.resolve(strict=True)
                    info = path.stat()
                except FileNotFoundError:
                    result["review"].append({"path": str(path), "reason": "dangling symlink"})
                    return
            identity = info.st_dev, info.st_ino
            if identity in seen:
                return
            seen.add(identity)
            if path.name == ".git":
                result["git_metadata"].append(str(path))
            if stat.S_ISDIR(info.st_mode):
                for child in sorted(path.iterdir()):
                    walk(child)
            elif stat.S_ISREG(info.st_mode):
                result["files"] += 1
                lower = path.name.lower()
                if path.suffix in {".so", ".a", ".o"} or ".so." in lower:
                    result["compiled_objects_count"] += 1
                if (
                    lower.endswith(ARCHIVES)
                    or path.suffix.lower() in TEXT
                    or info.st_size == FIXED_BYTES
                ):
                    if info.st_size > MAX_ARCHIVE:
                        result["review"].append(
                            {"path": str(path), "reason": "file exceeds inspection bound"}
                        )
                    else:
                        inspect(path.read_bytes(), str(path), path.name)
        except Exception as error:
            result["errors"].append({"path": str(path), "error": repr(error)})

    for root in roots:
        walk(root)
    result["known_source_status"] = (
        "blocked_known_fixed_or_git"
        if result["known_fixed"] or result["git_metadata"]
        else "requires_bounded_review"
        if result["review"] or result["errors"]
        else "no_known_fixed_source_found_within_recorded_scope"
    )
    result["compiled_boundary"] = (
        "No claim about arbitrary compiled, bytecode or opaque vendor payload equivalents."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), sort_keys=True))
