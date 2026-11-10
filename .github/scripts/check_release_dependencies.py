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
import subprocess
import time

from consumer_contract import verify_dependency

COMPONENTS = {
    "mindie-diagnostics": "mindie-agent/diagnostics",
    "mindie-debugger": "mindie-agent/debugger",
    "mindie-knowledge": "mindie-agent/knowledge",
    "remote-dev": "mindie-agent/remote-dev",
    "mindie-coordinator": "mindie-agent/coordinator",
    "npu-top": "mindie-agent/npu-top",
    "mindie-profiler": "mindie-agent/profiler",
    "mindie-sim": "mindie-agent/sim",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--consumer", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    script = args.consumer + "/mindie_knowledge/runtime_inventory.py"
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run([args.python, "-I", script], capture_output=True,
                            check=True, timeout=20, **flags)
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
