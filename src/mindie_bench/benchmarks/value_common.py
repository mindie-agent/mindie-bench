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

from mindie_bench.runner.host_platform import subprocess_window_kwargs, which_native


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def source_revisions(packages: tuple[str, ...] | None = None):
    packages = packages or (
        "mindie-coordinator", "npu-top", "mindie-diagnostics", "mindie-knowledge",
    )
    revisions = {}
    for name in packages:
        distribution = importlib.metadata.distribution(name)
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        revisions[name] = {
            "version": distribution.version,
            "revision": origin.get("vcs_info", {}).get("commit_id"),
            "editable": origin.get("dir_info", {}).get("editable", False),
        }
    return revisions


def environment():
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def git_source(root: Path):
    git_exe = which_native("git") or "git"

    def read(*args):
        return subprocess.check_output(
            [git_exe, "-C", str(root), *args],
            text=True,
            encoding="utf-8",
            timeout=5,
            **subprocess_window_kwargs(),
        ).strip()

    return {"revision": read("rev-parse", "HEAD"), "dirty": bool(read("status", "--porcelain"))}
