"""Verify installed first-party components have matching formal releases.

[Role] Gate publication on installed component version and immutable Git source.
[Invariants] Every installed MindIE dependency maps to a formal tag at its commit.
[Timeouts & Locks] Each read-only GitHub request has a 30 second timeout.
[Side-Effect Policy] Read installed metadata and public GitHub release/tag data.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

COMPONENTS = {
    "mindie-diagnostics": "mindie-agent/diagnostics",
    "mindie-knowledge": "mindie-agent/knowledge",
    "remote-dev": "mindie-agent/remote-dev",
    "mindie-coordinator": "mindie-agent/coordinator",
    "npu-top": "mindie-agent/npu-top",
    "mindie-profiler": "mindie-agent/profiler",
    "mindie-sim": "mindie-agent/sim",
}


def github(path: str) -> dict:
    request = Request(
        "https://api.github.com/" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def verify(repository: str, version: str, direct_url: dict, fetch=github) -> str:
    source = urlparse(direct_url["url"].removeprefix("git+"))
    assert source.netloc == "github.com"
    assert source.path.strip("/").removesuffix(".git") == repository
    commit = direct_url["vcs_info"]["commit_id"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "component must have an exact Git source"
    tag = "v" + version
    release = fetch(f"repos/{repository}/releases/tags/{tag}")
    assert release["tag_name"] == tag
    assert not release["draft"] and not release["prerelease"], f"{repository} {tag} is not formal"
    obj = fetch(f"repos/{repository}/git/ref/tags/{tag}")["object"]
    seen = set()
    while obj["type"] == "tag":
        assert obj["sha"] not in seen, "cyclic annotated tag"
        seen.add(obj["sha"])
        obj = fetch(f"repos/{repository}/git/tags/{obj['sha']}")["object"]
    assert obj["type"] == "commit" and obj["sha"] == commit, (
        f"{repository} {tag} does not match installed source {commit}"
    )
    return commit


def main() -> None:
    for name, repository in COMPONENTS.items():
        if repository == os.environ["GITHUB_REPOSITORY"]:
            continue
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        direct = distribution.read_text("direct_url.json")
        assert direct, f"{name} has no installed Git provenance"
        commit = verify(repository, distribution.version, json.loads(direct))
        print(f"{repository} v{distribution.version} {commit}: formal release verified")


if __name__ == "__main__":
    main()
