"""[Role] Publish a fully tested release from a verified private draft.

[Invariants] Public releases are never edited. A new tag is created only by the
final publish operation, after old-client upgrade and downloaded-asset validation.
An uncertain write is read back before any further action, never blindly replayed.
[Timeouts & Locks] Each network call is bounded; the reusable workflow serializes releases.
[Side-Effect Policy] Creates a draft, uploads artifacts, then publishes once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from consumer_contract import validate_evidence, validate_release
from release_transport import GitHub


def marker(revision):
    return f"<!-- mindie-release-source:{revision} -->"


def verify_remote(api, release, *, consumer, version, revision):
    if release["tag_name"] != "v" + version or release.get("prerelease"):
        raise ValueError("release identity differs from the candidate")
    rows = release["assets"]
    assets = {row["name"]: row for row in rows}
    if len(assets) != len(rows) or any(row.get("state") != "uploaded" for row in rows):
        raise ValueError("release contains duplicate or unfinished assets")
    cached = {}

    def read(filename):
        if filename not in cached:
            cached[filename] = api.asset(assets[filename])
        return cached[filename]

    sums = validate_evidence(api.repository, version, revision, rows, read)
    for filename, digest in sums.items():
        if api.asset_digest(assets[filename]) != digest:
            raise ValueError(f"uploaded artifact digest mismatch: {filename}")
    # Draft URLs and uncreated tag refs are unavailable to an ordinary client.
    # Adapt those reads, preserving the client's exact validation implementation.
    advertised = dict(release, draft=False)

    def fetch(path):
        if path == f"{api.prefix}/git/ref/tags/v{version}" and release["draft"]:
            return {"object": {"type": "commit", "sha": revision}}
        return api.api(path)

    validate_release(consumer, api.repository, version, revision, advertised, read, fetch)


def publish(api, directory, *, consumer, version, revision):
    directory = Path(directory)
    files = sorted(path for path in directory.iterdir()
                   if path.is_file() and not path.name.startswith("."))
    local = {path.name: path for path in files}
    rows = [dict(name=path.name, size=path.stat().st_size) for path in files]
    validate_evidence(api.repository, version, revision, rows,
                      lambda filename: local[filename].read_bytes())
    tag = "v" + version
    release = api.release(tag)
    if release is not None and not release["draft"]:
        verify_remote(api, release, consumer=consumer, version=version, revision=revision)
        return dict(status="already_published", tag=tag, revision=revision)
    expected_marker = marker(revision)
    if release is None:
        body_file = local.get("release-notes.md", local["benchmark-results.md"])
        try:
            release = api.create(tag, revision, body_file.read_text(encoding="utf-8")
                                 + "\n\n" + expected_marker)
        except Exception:
            release = api.release(tag)
            if release is None:
                raise
    if (not release["draft"] or release.get("target_commitish") != revision
            or expected_marker not in release.get("body", "")):
        raise ValueError("existing draft is not owned by this source transaction")
    for path in files:
        release = api.release(tag)
        existing = next((row for row in release["assets"] if row["name"] == path.name), None)
        if existing is not None:
            with path.open("rb") as stream:
                expected = hashlib.file_digest(stream, "sha256").hexdigest()
            if api.asset_digest(existing) != expected:
                raise ValueError(f"existing draft artifact differs: {path.name}")
            continue
        try:
            api.upload(tag, path)
        except Exception:
            observed = api.release(tag)
            existing = next((row for row in observed["assets"] if row["name"] == path.name), None)
            if existing is None:
                raise
            with path.open("rb") as stream:
                if api.asset_digest(existing) != hashlib.file_digest(stream, "sha256").hexdigest():
                    raise
    release = api.release(tag)
    verify_remote(api, release, consumer=consumer, version=version, revision=revision)
    try:
        api.publish(release["id"])
    except Exception:
        observed = api.release(tag)
        if observed is None or observed["draft"]:
            raise
    observed = api.release(tag)
    if observed is None or observed["draft"]:
        raise ValueError("formal publication was not observed")
    verify_remote(api, observed, consumer=consumer, version=version, revision=revision)
    if observed.get("immutable") is not True:
        raise ValueError("publication succeeded but immutable-release enforcement is missing")
    return dict(status="published", tag=tag, revision=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "version", "revision", "assets", "consumer"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must identify the tested commit")
    print(json.dumps(publish(GitHub(args.repository), args.assets, consumer=args.consumer,
                             version=args.version, revision=args.revision), sort_keys=True))


if __name__ == "__main__":
    main()
