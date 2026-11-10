"""Verify installed first-party components have matching formal releases.

[Role] Gate publication on installed component version and immutable Git source.
[Invariants] Every installed MindIE dependency maps to a formal tag at its commit.
[Timeouts & Locks] Each read-only GitHub request has a 30 second timeout.
[Side-Effect Policy] Read installed metadata and public GitHub release/tag data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path

COMPONENTS = {
    "mindie-diagnostics": "mindie-agent/diagnostics",
    "mindie-debugger": "mindie-agent/debugger",
    "mindie-knowledge": "mindie-agent/knowledge",
    "remote-dev": "mindie-agent/remote-dev",
    "mindie-coordinator": "mindie-agent/coordinator",
    "npu-top": "mindie-agent/npu-top",
    "mindie-profiler": "mindie-agent/profiler",
    "mindie-sim": "mindie-agent/sim",
    "mindie-bench": "mindie-agent/mindie-bench",
}


def installed_dependencies(candidate):
    """The local publishing source is checked by source CI, not its own release."""
    installed = {}
    for name, repository in COMPONENTS.items():
        if repository == candidate:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        revision = (direct.get("vcs_info") or {}).get("commit_id", "")
        if (direct.get("url", "").removesuffix(".git") != "https://github.com/" + repository
                or not re.fullmatch(r"[0-9a-f]{40}", revision)):
            raise ValueError(f"{name} dependency has no official immutable installation receipt")
        installed[name] = dict(version=distribution.version, revision=revision)
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python")
    parser.add_argument("--consumer")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(installed_dependencies(args.repository), sort_keys=True))
        return
    if not args.python or not args.consumer:
        parser.error("dependency validation requires --python and --consumer")
    from consumer_contract import verify_dependency

    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run([
        args.python, "-I", str(Path(__file__).resolve()),
        "--repository", args.repository, "--inventory",
    ], capture_output=True, check=False, timeout=20, **flags)
    if result.returncode:
        raise RuntimeError("dependency inventory failed:\n"
                           + result.stderr.decode("utf-8", errors="replace")[-4000:])
    installed = json.loads(result.stdout)
    deadline = time.monotonic() + 300
    for name, repository in COMPONENTS.items():
        if repository == args.repository or name not in installed:
            continue
        row = installed[name]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("first-party dependency verification budget exhausted")
        verify_dependency(args.consumer, repository, row["version"], row["revision"],
                          budget=remaining)
        print(f"{repository} v{row['version']} {row['revision']}: published consumer verified")


if __name__ == "__main__":
    main()
