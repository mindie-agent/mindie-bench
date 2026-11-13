"""[Role] Prepare native plugin metadata for every shared release publication.

[Invariants] Repository identities, source layouts and compatibility validation
come from load_consumer against the released consumer. Metadata names the checked-out
commit and its unchanged declaration; existing incorrect assets fail visibly.
[Timeouts & Locks] Each local Git read is bounded to 10 seconds; no locks.
[Side-Effect Policy] Writes only a missing asset in the external staging directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from consumer_contract import load_consumer


def git(source, *arguments):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], timeout=10, **flags,
    )


def prepare(*, consumer, repository, version, revision, source, assets):
    """Generate an omitted asset or validate a repository's existing output."""
    _, components = load_consumer(consumer)
    if repository in components.values():
        return None
    from mindie_knowledge.releases import RELEASE_SCHEMA, REPOSITORIES, validate_release_metadata
    from mindie_knowledge.releases_io import _COMPATIBILITY_PATHS, ASSET_NAME

    harness = next((name for name, repo in REPOSITORIES.items() if repo == repository), None)
    if harness is None:
        raise ValueError("repository is not a released component or native plugin")
    declaration = _COMPATIBILITY_PATHS[harness]
    if git(source, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("plugin source does not match the tested commit")
    committed = git(source, "show", f"{revision}:{declaration}")
    if (Path(source) / declaration).read_bytes() != committed:
        raise ValueError("plugin compatibility declaration differs from the tested commit")
    metadata = validate_release_metadata(dict(
        schema=RELEASE_SCHEMA, version=version,
        source=dict(repository=repository, sha=revision),
        compatibility=json.loads(committed),
    ), repository=repository, harness_name=harness)
    output = Path(assets) / ASSET_NAME
    if output.exists():
        actual = validate_release_metadata(
            json.loads(output.read_bytes()), repository=repository, harness_name=harness,
        )
        if actual != metadata:
            raise ValueError("existing plugin metadata differs from the tested release")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("consumer", "repository", "version", "revision", "source", "assets"):
        parser.add_argument("--" + name, required=True)
    output = prepare(**vars(parser.parse_args()))
    print("native plugin metadata: " + ("not required" if output is None else "verified"))


if __name__ == "__main__":
    main()
