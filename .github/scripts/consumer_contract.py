"""[Role] Run the published updater's actual release validator in publication.

[Invariants] The consumer checkout is immutable and unmodified. Exact-release and
staged-candidate reads adapt transport only; validation remains the old client code.
[Timeouts & Locks] The caller bounds network reads and the enclosing workflow.
[Side-Effect Policy] Read-only except for the consumer's bounded public HTTP cache.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from release_transport import GitHub


def load_consumer(source):
    sys.path.insert(0, str(Path(source).resolve()))
    from mindie_knowledge.component_releases import Catalog
    from mindie_knowledge.runtime_inventory import COMPONENTS
    return Catalog, COMPONENTS


def validate_release(consumer, repository, version, revision, release, read, fetch):
    Catalog, components = load_consumer(consumer)
    name = next((key for key, repo in components.items() if repo == repository), None)
    if name is None:
        from mindie_knowledge.releases import REPOSITORIES
        from mindie_knowledge.releases_io import verify_release_source
        harness = next(key for key, repo in REPOSITORIES.items() if repo == repository)
        metadata = verify_release_source(json.loads(read("mindie-plugin-release.json")),
                                         repository=repository, harness_name=harness)
        if metadata["version"] != version or metadata["source"]["sha"] != revision:
            raise ValueError("plugin metadata does not name the tested source")
        return metadata

    class ExactRelease(Catalog):
        def read(self, path, *, force=False):
            if path.endswith("/releases/latest"):
                return release
            return fetch(path.removeprefix("/"))

        def asset(self, component, tag, filename):
            return read(filename)

    actual = ExactRelease().latest(name)
    if actual != dict(repository=repository, version=version, revision=revision):
        raise ValueError("published consumer selected a different source")
    return actual


def parse_sums(raw):
    result = {}
    for line in raw.decode("utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.removeprefix("*")
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or filename in result or Path(filename).name != filename):
            raise ValueError("invalid or repeated release checksum")
        result[filename] = digest
    return result


def validate_evidence(repository, version, revision, assets, read):
    """The transaction verifies every package byte, in addition to old-client checks."""
    names = {row["name"] for row in assets}
    required = {"SHA256SUMS", "benchmark-results.json", "benchmark-results.md",
                "benchmark-SHA256SUMS", "upgrade-acceptance.json"}
    if repository == "mindie-agent/mindie-agent-codex":
        required.add("native-upgrade-acceptance.json")
    if not required <= names:
        raise ValueError("release is missing benchmark or real upgrade evidence")
    sums = parse_sums(read("SHA256SUMS"))
    if set(sums) != names - {"SHA256SUMS"}:
        raise ValueError("release checksums do not cover the complete asset set")
    for filename in required - {"SHA256SUMS"}:
        if hashlib.sha256(read(filename)).hexdigest() != sums[filename]:
            raise ValueError(f"release evidence digest mismatch: {filename}")
    report = json.loads(read("benchmark-results.json"))
    if (report.get("schema_version") != "mindie-release-benchmark/2"
            or report.get("repository") != repository or report.get("version") != version
            or report.get("release_gate") is not True or report.get("blocking_reasons") != []):
        raise ValueError("benchmark evidence does not authorize this publication")
    for key in ("source", "source_after"):
        if report.get(key) != {"revision": revision, "dirty": False}:
            raise ValueError("benchmark evidence names a different or dirty source")
    upgrade = json.loads(read("upgrade-acceptance.json"))
    if (upgrade.get("schema") != "mindie-upgrade-acceptance/1"
            or upgrade.get("status") != "passed" or upgrade.get("repository") != repository
            or upgrade.get("version") != version or upgrade.get("revision") != revision
            or not upgrade.get("consumer_revision")
            or upgrade.get("installed_revision") != revision):
        raise ValueError("old-client upgrade evidence does not authorize this publication")
    if repository == "mindie-agent/mindie-agent-codex":
        native = json.loads(read("native-upgrade-acceptance.json"))
        if (native.get("schema") != "mindie-native-upgrade-acceptance/1"
                or native.get("status") != "passed" or native.get("repository") != repository
                or native.get("version") != version or native.get("revision") != revision
                or native.get("scope") != "native_codex_old_client"
                or native.get("baseline_plugin") != "0.1.23"
                or native.get("external_components_rebound") is not True
                or native.get("disabled_stop_preserved") is not True):
            raise ValueError("real native old-client upgrade evidence is required")
    return sums


def verify_dependency(consumer, repository, version, revision, *, budget=120):
    api = GitHub(repository, budget=budget)
    release = api.release("v" + version)
    if release is None:
        raise ValueError(f"dependency is not published: {repository} v{version}")
    assets = {row["name"]: row for row in release["assets"]}
    return validate_release(consumer, repository, version, revision, release,
                            lambda filename: api.asset(assets[filename]), api.api)
