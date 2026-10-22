# [Role] Reads native local packages and the published knowledge feed contract.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Explicit readers for native local packages and the published feed contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        name = loader.construct_object(key, deep=deep)
        if name in result:
            raise ValueError(f"duplicate YAML frontmatter key: {name}")
        result[name] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def frontmatter(path: Path) -> dict:
    text = path.read_text()
    parts = text.split("---\n", 2)
    if len(parts) != 3 or parts[0]:
        raise ValueError(f"missing explicit frontmatter: {path}")
    header = yaml.load(parts[1], Loader=UniqueLoader)
    if not isinstance(header, dict):
        raise ValueError(f"invalid frontmatter object: {path}")
    return header


@dataclass(frozen=True)
class Package:
    layout: str
    navigation: Path
    blocks: list[Path]
    title: str
    source_status: str


def read_package(root: Path) -> Package:
    local, public = root / "PACKAGE.md", root / "index.md"
    if local.is_file() == public.is_file():
        raise ValueError(f"package must have exactly one recognized navigation format: {root.name}")
    if local.is_file():
        blocks = sorted((root / "materials").glob("*.md"))
        if not blocks:
            raise ValueError(f"local package has no material blocks: {root.name}")
        title = next(
            (
                line.lstrip("# ").strip()
                for line in local.read_text().splitlines()
                if line.startswith("#")
            ),
            root.name,
        )
        return Package("local-package-materials", local, blocks, title, "unspecified")
    header = frontmatter(public)
    if (
        header.get("schema") != "mindie-material-task/1"
        or header.get("task_id") != root.name
        or header.get("status") not in {"pending", "incomplete", "complete", "failed", "uncertain"}
    ):
        raise ValueError(f"invalid published task contract: {root.name}")
    descriptors = header.get("blocks")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError(f"published package has no block descriptors: {root.name}")
    identities = [row["block_id"] for row in descriptors]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate published block IDs")
    declared = {str(identity) + ".md" for identity in identities}
    actual = {path.name for path in (root / "blocks").iterdir()}
    if declared != actual:
        raise ValueError(f"published block inventory mismatch: {root.name}")
    blocks = []
    for row in descriptors:
        path = root / "blocks" / (row["block_id"] + ".md")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("published block path escapes package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"published block hash mismatch: {root.name}/{path.name}")
        block = frontmatter(path)
        if block.get("schema") != "mindie-material-block/1" or block.get("block_id") != row["block_id"]:
            raise ValueError("published block identity mismatch")
        blocks.append(path)
    entry = header.get("entry", {})
    if entry.get("schema") != "mindie-entry/3" or entry.get("entry_id") != root.name:
        raise ValueError("published entry identity mismatch")
    return Package("published-index-blocks", public, blocks, entry["title"], header["status"])
