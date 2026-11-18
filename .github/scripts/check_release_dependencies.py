"""Verify installed first-party components have matching formal releases.

[Role] Gate publication on installed component version and immutable Git source.
[Invariants] Every installed MindIE dependency maps to a formal tag at its commit.
[Timeouts & Locks] Each read-only GitHub request has a 30 second timeout.
[Side-Effect Policy] Read installed metadata and public GitHub release/tag data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path

COMPONENTS = {
    "mindie-harness": "mindie-agent/harness",
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
        url = str(direct.get("url") or "")
        revision = (direct.get("vcs_info") or {}).get("commit_id", "")
        if (url.removesuffix(".git") == "https://github.com/" + repository
                and re.fullmatch(r"[0-9a-f]{40}", revision)):
            installed[name] = dict(version=distribution.version, revision=revision)
            continue
        wheel_match = re.fullmatch(
            rf"https://github\.com/{re.escape(repository)}/releases/download/v(\d+\.\d+\.\d+)/([^/]+-py3-none-any\.whl)",
            url,
        )
        sha256 = ((direct.get("archive_info") or {}).get("hashes") or {}).get("sha256", "")
        record_raw = distribution.read_text("RECORD") or ""
        record_sha = hashlib.sha256(record_raw.encode("utf-8")).hexdigest() if record_raw else ""
        if (wheel_match and wheel_match.group(1) == distribution.version
                and (re.fullmatch(r"[0-9a-f]{64}", sha256) or record_sha)):
            installed[name] = dict(
                version=distribution.version, wheel_name=wheel_match.group(2),
                archive_sha256=sha256 or None, record_sha256=record_sha or None,
            )
            continue
        raise ValueError(f"{name} dependency has no official immutable installation receipt")
    return installed


def _resolve_wheel_revision(repository: str, row: dict, *, budget: float) -> str:
    import io
    import zipfile

    from consumer_contract import parse_sums
    from release_transport import GitHub

    api = GitHub(repository, budget=budget)
    version = row["version"]
    release = api.release("v" + version)
    if release is None:
        raise ValueError(f"dependency release is not published: {repository} v{version}")
    assets = {item["name"]: item for item in release["assets"]}
    if "SHA256SUMS" not in assets or row["wheel_name"] not in assets:
        raise ValueError(f"{repository} v{version}: missing wheel or SHA256SUMS")
    sums = parse_sums(api.asset(assets["SHA256SUMS"]))
    expected_sha = sums.get(row["wheel_name"])
    if not expected_sha:
        raise ValueError(f"{repository} v{version}: wheel absent from SHA256SUMS")
    if row.get("archive_sha256"):
        if row["archive_sha256"] != expected_sha:
            raise ValueError(f"{repository} v{version}: wheel SHA-256 differs from release SHA256SUMS")
    else:
        raw_wheel = api.asset(assets[row["wheel_name"]])
        if hashlib.sha256(raw_wheel).hexdigest() != expected_sha:
            raise ValueError(f"{repository} v{version}: release wheel failed SHA256SUMS")
        with zipfile.ZipFile(io.BytesIO(raw_wheel)) as zf:
            rec_names = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
            if len(rec_names) != 1 or hashlib.sha256(zf.read(rec_names[0])).hexdigest() != row.get("record_sha256"):
                raise ValueError(f"{repository} v{version}: installed RECORD differs from release wheel")
    obj = api.api(f"repos/{repository}/git/ref/tags/v{version}")["object"]
    for _ in range(8):
        if obj["type"] == "commit":
            return obj["sha"]
        if obj["type"] != "tag":
            break
        obj = api.api(f"repos/{repository}/git/tags/{obj['sha']}")["object"]
    raise ValueError(f"{repository} v{version}: tag does not resolve to a commit")


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
    from consumer_contract import load_consumer, verify_dependency

    _, consumer_components = load_consumer(args.consumer)
    from mindie_knowledge.component_runtime import read_versions

    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    candidate = next((name for name, repo in COMPONENTS.items() if repo == args.repository), None)
    if candidate is not None:
        if candidate in consumer_components:
            read_versions(args.python, [candidate])
        else:
            probe = subprocess.run(
                [args.python, "-I", "-c", "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))", candidate],
                capture_output=True, check=False, timeout=15, **flags,
            )
            if probe.returncode:
                raise RuntimeError(
                    f"candidate distribution {candidate} is not importable:\n"
                    + probe.stderr.decode("utf-8", errors="replace")[-2000:]
                )
    result = subprocess.run([
        args.python, "-I", str(Path(__file__).resolve()),
        "--repository", args.repository, "--inventory",
    ], capture_output=True, check=False, timeout=20, **flags)
    if result.returncode:
        raise RuntimeError("dependency inventory failed:\n"
                           + result.stderr.decode("utf-8", errors="replace")[-4000:])
    installed = json.loads(result.stdout)
    if candidate is None and not installed:
        raise RuntimeError(f"host plugin {args.repository} has no installed first-party MindIE substrate or component")
    deadline = time.monotonic() + 300
    for name, repository in COMPONENTS.items():
        if repository == args.repository or name not in installed:
            continue
        row = installed[name]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("first-party dependency verification budget exhausted")
        revision = row.get("revision") or _resolve_wheel_revision(repository, row, budget=remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("first-party dependency verification budget exhausted")
        verify_dependency(args.consumer, repository, row["version"], revision,
                          budget=remaining)
        print(f"{repository} v{row['version']} {revision}: published consumer verified")


if __name__ == "__main__":
    main()
