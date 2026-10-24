"""Collect benchmark evidence from formal component releases.

[Role] Download source-bound public reports and reuse the release collection gate.
[Invariants] Release tags, manifest SHAs, report versions and asset hashes agree.
[Timeouts & Locks] Each GitHub read is bounded to 60 seconds; no local locks.
[Side-Effect Policy] Reads GitHub and writes public output files; never publishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from mindie_bench.release import collect_release
from mindie_bench.release_contract import ReleaseManifest

ASSETS = ("benchmark-results.json", "benchmark-results.md", "benchmark-SHA256SUMS")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def github(path, *, binary=False):
    command = ["gh", "api", path]
    if binary:
        command += ["-H", "Accept: application/octet-stream"]
    result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    return result.stdout if binary else json.loads(result.stdout)


def tag_revision(repository, version, fetch=github):
    if not VERSION.fullmatch(version):
        raise ValueError(f"{repository}: expected a formal MAJOR.MINOR.PATCH version")
    obj = fetch(f"repos/{repository}/git/ref/tags/v{version}")["object"]
    for _ in range(8):
        if obj["type"] == "commit":
            return obj["sha"]
        if obj["type"] != "tag":
            break
        obj = fetch(f"repos/{repository}/git/tags/{obj['sha']}")["object"]
    raise ValueError(f"{repository}: tag does not resolve to a commit")


def verify_checksums(directory):
    entries = {}
    for line in (directory / "benchmark-SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (benchmark-results\.(?:json|md))", line)
        if match is None or match[2] in entries:
            raise ValueError("invalid or repeated benchmark checksum entry")
        entries[match[2]] = match[1]
    if set(entries) != set(ASSETS[:2]):
        raise ValueError("benchmark checksum file does not cover both reports")
    for name, expected in entries.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"benchmark checksum mismatch: {name}")


def release_identity(release):
    # Downloads change download_count; only publication identity must stay fixed.
    return (
        release["id"], release["tag_name"], release["draft"], release["prerelease"],
        sorted(
            (asset["id"], asset["name"], asset.get("size"), asset.get("digest"),
             asset.get("updated_at"))
            for asset in release["assets"] if asset["name"] in ASSETS
        ),
    )


def download_release(target, directory, fetch=github):
    repository, version = target.repository, target.version
    path = f"repos/{repository}/releases/tags/v{version}"
    release = fetch(path)
    identity = release_identity(release)
    if (release["tag_name"] != f"v{version}"
            or release["draft"] or release["prerelease"]):
        raise ValueError(f"{repository}: benchmark must belong to a formal release")
    if tag_revision(repository, version, fetch) != target.revision:
        raise ValueError(f"{repository}: release tag differs from manifest revision")
    selected = [asset for asset in release["assets"] if asset["name"] in ASSETS]
    if len(selected) != len(ASSETS) or {asset["name"] for asset in selected} != set(ASSETS):
        raise ValueError(f"{repository}: missing or repeated public benchmark assets")
    directory.mkdir(parents=True)
    assets = []
    for asset in sorted(selected, key=lambda value: value["name"]):
        raw = fetch(f"repos/{repository}/releases/assets/{asset['id']}", binary=True)
        (directory / asset["name"]).write_bytes(raw)
        assets.append(dict(name=asset["name"], id=asset["id"],
                           sha256=hashlib.sha256(raw).hexdigest()))
    verify_checksums(directory)
    if (release_identity(fetch(path)) != identity
            or tag_revision(repository, version, fetch) != target.revision):
        raise ValueError(f"{repository}: release changed while downloading evidence")
    return dict(repository=repository, version=version, revision=target.revision,
                release_id=release["id"], tag=release["tag_name"],
                release_url=release["html_url"], assets=assets)


def validate_channel(channel, contract):
    """The update channel and public release must name the same component set."""
    raw = channel.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "mindie-release-channel/1":
        raise ValueError("unsupported product channel schema")
    if document.get("current_patch_version") != contract.product_version:
        raise ValueError("channel patch version differs from product release")
    components = list(document["components"].values())
    by_repository = {item["repository"]: item for item in components}
    if (len(by_repository) != len(components)
            or set(by_repository) != {target.repository for target in contract.repositories}):
        raise ValueError("channel and benchmark manifest component sets differ")
    for target in contract.repositories:
        item = by_repository[target.repository]
        if (item["version"] != target.version or item["tag"] != f"v{target.version}"
                or item["commit"] != target.revision
                or item.get("revision", target.revision) != target.revision):
            raise ValueError(f"{target.repository}: channel identity differs from release manifest")
    return raw


def collect_product(*, repository, version, revision, manifest, output, fetch=github, channel=None):
    contract = ReleaseManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    if contract.product_version != version:
        raise ValueError("product tag and benchmark manifest versions differ")
    channel_raw = validate_channel(channel, contract) if channel is not None else None
    if tag_revision(repository, version, fetch) != revision:
        raise ValueError("product tag differs from the checked-out source")
    output.mkdir(parents=True, exist_ok=False)
    sources, inputs = [], []
    for target in contract.repositories:
        directory = output / "components" / target.repository
        sources.append(download_release(target, directory, fetch))
        inputs.append(directory)
    collection = collect_release(manifest=manifest, inputs=inputs, output=output)
    if not collection["release_gate"]:
        raise ValueError("product benchmark gate blocked: " + ", ".join(collection["blocking_reasons"]))
    if tag_revision(repository, version, fetch) != revision:
        raise ValueError("product tag changed while collecting evidence")
    shutil.copyfile(manifest, output / "benchmark-manifest.json")
    provenance = dict(
        schema_version="mindie-product-release-provenance/1",
        repository=repository, version=version, revision=revision, tag=f"v{version}",
        manifest_sha256=collection["manifest_sha256"], releases=sources,
    )
    if channel_raw is not None:
        (output / "release-channel.json").write_bytes(channel_raw)
        provenance["channel_sha256"] = hashlib.sha256(channel_raw).hexdigest()
    (output / "product-release-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(output.iterdir()) if path.is_file()
        ), encoding="utf-8",
    )
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=Path)
    args = parser.parse_args()
    collect_product(**vars(args))


if __name__ == "__main__":
    main()
