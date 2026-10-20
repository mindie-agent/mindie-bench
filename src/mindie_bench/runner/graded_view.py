"""[Role] Physically project authoritative graded Markdown into a trial.

[Invariants] The explicit publication contract selects the format. Excluded
units and descendants cannot reach the agent through derived files or docs.
The selected knowledge runtime validates complete C5 units before retrieval.
[Timeouts & Locks] Local bounded file reads; no subprocess or locks.
[Side-Effect Policy] Copies selected Markdown and the format contract only.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from mindie_bench.io import tree_hash


def is_graded(source: Path) -> bool:
    contract = source / "publication-contract.json"
    if not contract.is_file():
        return False
    value = json.loads(contract.read_text())
    schema = value.get("schema")
    if schema not in {"mindie-publication-contract/1", "mindie-publication-contract/2"}:
        raise ValueError("corpus publication contract has an unsupported format")
    return schema == "mindie-publication-contract/2"


def units(source: Path) -> dict[str, tuple[Path, dict]]:
    selected = {}
    if (source / "tasks").exists():
        raise ValueError("graded corpus cannot contain the legacy tasks layout")
    for path in sorted(source.rglob("unit-*.md")):
        relative = path.relative_to(source)
        if relative.parts[0] in {"docs", ".github"}:
            continue
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(source.resolve()):
            raise ValueError("graded unit escapes the selected corpus")
        raw = path.read_text()
        if not raw.startswith("---\n") or "\n---\n" not in raw:
            raise ValueError("graded unit requires its authoritative frontmatter")
        metadata = yaml.safe_load(raw.split("\n---\n", 1)[0][4:])
        identity = metadata["id"]
        if metadata.get("schema") != "mindie-graded-unit/1" or (
            relative.as_posix() != f"{metadata['domain_path']}/{identity}.md"
        ):
            raise ValueError("graded unit path and identity disagree")
        if identity in selected:
            raise ValueError("graded corpus repeats a stable unit identity")
        selected[identity] = (path, metadata)
    return selected


def identifiers(source: Path) -> set[str]:
    if is_graded(source):
        return set(units(source))
    return {path.name for path in (source / "tasks").iterdir() if path.is_dir()}


def project(source: Path, destination: Path, excluded: set[str]) -> str:
    selected = units(source)
    removed = set(excluded)
    while True:
        inherited = {
            identity
            for identity, (_, meta) in selected.items()
            if set(meta.get("inherits_from", [])) & removed
        }
        if inherited <= removed:
            break
        removed.update(inherited)
    destination.mkdir()
    (destination / "publication-contract.json").write_bytes(
        (source / "publication-contract.json").read_bytes()
    )
    for identity, (path, _) in selected.items():
        if identity in removed:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return tree_hash(destination)
