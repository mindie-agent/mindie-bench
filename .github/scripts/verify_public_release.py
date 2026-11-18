"""[Role] Verify anonymous public CDN availability and SHA256SUMS of a published release.
[Invariants] Release must be non-draft, non-prerelease, tag-dereferenced to the exact
source SHA, and every asset must download anonymously over HTTPS and match SHA256SUMS.
[Timeouts & Locks] Single 25s enclosing deadline (< 30s budget); no locks.
[Side-Effect Policy] Downloads into disposable work-dir and writes public-upgrade.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from consumer_contract import parse_sums

_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
PUBLIC_BUDGET_SECONDS = 25.0


def _run(argv, *, timeout=10.0):
    return subprocess.run(
        [str(v) for v in argv], check=True, capture_output=True, text=True,
        encoding="utf-8", timeout=max(0.5, float(timeout)), **_FLAGS,
    ).stdout


def _download_public_url(url: str, dest: Path, *, timeout: float) -> None:
    with urlopen(url, timeout=max(0.5, float(timeout))) as resp, dest.open("wb") as out:
        while chunk := resp.read(65536):
            out.write(chunk)


def resolve_tag_commit(repository: str, version: str, *, deadline: float, run_fn=_run) -> str:
    obj = json.loads(run_fn(
        ["gh", "api", f"repos/{repository}/git/ref/tags/v{version}"],
        timeout=deadline - time.monotonic(),
    ))["object"]
    for _ in range(8):
        if obj["type"] == "commit":
            return obj["sha"]
        if obj["type"] != "tag":
            break
        obj = json.loads(run_fn(
            ["gh", "api", f"repos/{repository}/git/tags/{obj['sha']}"],
            timeout=deadline - time.monotonic(),
        ))["object"]
    raise ValueError(f"{repository}: v{version} does not resolve to a commit")


def verify_public_release(*, repository: str, version: str, revision: str, work_dir: Path,
                          output: Path, run_fn=_run, download_url_fn=_download_public_url,
                          budget: float = PUBLIC_BUDGET_SECONDS) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a 40-hex commit SHA")
    deadline = time.monotonic() + float(budget)
    tag = f"v{version}"
    view = json.loads(run_fn([
        "gh", "release", "view", tag, "--repo", repository,
        "--json", "tagName,isDraft,isPrerelease,url,assets",
    ], timeout=deadline - time.monotonic()))
    if view.get("tagName") != tag or view.get("isDraft") or view.get("isPrerelease"):
        raise ValueError(f"{repository} {tag} is not a formal public release")
    if resolve_tag_commit(repository, version, deadline=deadline, run_fn=run_fn) != revision:
        raise ValueError(f"{repository} {tag} does not point to {revision}")
    asset_names = [a["name"] for a in (view.get("assets") or []) if a.get("name")]
    if "SHA256SUMS" not in asset_names:
        raise ValueError("public release is missing SHA256SUMS")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://github.com/{repository}/releases/download/{tag}"
    for name in sorted(set(asset_names)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("public release verification exceeded budget")
        download_url_fn(f"{base_url}/{name}", work_dir / Path(name).name, timeout=remaining)
    sums = parse_sums((work_dir / "SHA256SUMS").read_bytes())
    downloaded = {p.name: p for p in work_dir.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.name != "SHA256SUMS"}
    if set(downloaded) != set(sums) or (set(asset_names) - {"SHA256SUMS"}) != set(sums):
        raise ValueError("public release assets differ from SHA256SUMS")
    for name, expected in sorted(sums.items()):
        with downloaded[name].open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"public asset SHA-256 mismatch: {name}")
    required = {"benchmark-results.json", "benchmark-results.md", "benchmark-SHA256SUMS", "upgrade-acceptance.json"}
    if repository == "mindie-agent/mindie-agent-codex":
        required.add("native-upgrade-acceptance.json")
    if not required <= set(downloaded):
        raise ValueError(f"public release is missing required gate evidence: {sorted(required - set(downloaded))}")
    bench = json.loads(downloaded["benchmark-results.json"].read_text(encoding="utf-8"))
    upgrade = json.loads(downloaded["upgrade-acceptance.json"].read_text(encoding="utf-8"))
    if bench.get("release_gate") is not True or upgrade.get("status") != "passed":
        raise ValueError("public release gate evidence is not passing")
    if repository == "mindie-agent/mindie-agent-codex":
        native = json.loads(downloaded["native-upgrade-acceptance.json"].read_text(encoding="utf-8"))
        if native.get("status") != "passed":
            raise ValueError("public native upgrade evidence is not passing")
    report = dict(
        schema="mindie-public-release-verification/1", status="passed",
        repository=repository, version=version, revision=revision, tag=tag,
        url=view.get("url"), assets=sorted(sums), sha256sums_verified=True,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "version", "revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    verify_public_release(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
