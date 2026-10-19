"""Shared evidence writing for bounded component scenarios.

[Role] Record source identity and task outcomes without inventing performance scores.
[Invariants] Raw observations and comparison limitations remain separate.
[Timeouts & Locks] Metadata Git reads have a five-second deadline.
[Side-Effect Policy] Writes only the caller's output directory.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def source_revisions():
    packages = {}
    for name in ("mindie-coordinator", "npu-top", "mindie-diagnostics", "mindie-knowledge"):
        distribution = importlib.metadata.distribution(name)
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        packages[name] = {
            "version": distribution.version,
            "revision": origin.get("vcs_info", {}).get("commit_id"),
            "editable": origin.get("dir_info", {}).get("editable", False),
        }
    return packages


def environment():
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def git_source(root: Path):
    def read(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, timeout=5
        ).strip()

    return {"revision": read("rev-parse", "HEAD"), "dirty": bool(read("status", "--porcelain"))}
