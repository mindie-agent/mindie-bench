"""Metadata-only lineage grouping for cross-split corpus preparation.

[Role]
`build_lineage_groups` partitions whitelist-projected inventory metadata into
conservative same-problem families before any distillation or split assignment.

[Invariants]
- Only exact session ids, complete-transcript hashes, exact source identities,
  and public source-repository families create strong unions.
- Hash-prefixed source ids never bypass scope or nonempty checks. Shared step
  outputs, excerpts and dependency files stay duplicate-content candidates;
  equality of these bytes does not prove the sessions share a problem.
- Resolution is source-repository scope only; session and branch coupling never
  proves independent problem lineage, and timestamps are observation metadata.

[Timeouts & Locks]
- Pure in-memory single pass; no locks, clocks, IO, subprocess, or sleeps.

[Side-Effect Policy]
- No filesystem, network, log, or global-state mutation; returns a new dict.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SOURCE_RECORD_TYPES = frozenset({"source_file", "source_repository"})
ASSIGNED_SPLITS = frozenset({"seed", "dev", "heldout"})

SESSION_EDGE = "exact_session_id"
HASH_EDGE = "positive_sha256"
HASH_CANDIDATE_EDGE = "duplicate_content_not_resolved_problem_lineage"
SOURCE_ID_EDGE = "exact_source_identity"
PUBLIC_REPO_EDGE = "public_source_repository"
RELATED_EDGE = "related_source_reference"
TOPIC_EDGE = "same_repo_declared_family_topic"


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _positive_size(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_content_key(row: dict) -> tuple[str, int] | None:
    sha = _text(row.get("sha256"))
    if not sha or sha.lower() == EMPTY_SHA256:
        return None
    size = row.get("size")
    if not _positive_size(size):
        return None
    return sha, size


def _source_identity(row: dict) -> tuple[str, str] | None:
    namespace = _text(row.get("source_namespace"))
    source_id = _text(row.get("source_id"))
    if not namespace or not source_id:
        return None
    if source_id.lower().startswith("sha256:"):
        # A hash-shaped source id is not proof of content equality; only rule 2 may union it.
        return None
    return namespace, source_id


def _public_repo_identity(row: dict) -> str | None:
    repo = _text(row.get("repo_identity"))
    if not repo or row.get("public_catalog") is not True:
        return None
    if _text(row.get("record_type")) not in SOURCE_RECORD_TYPES:
        return None
    return repo


def _keyed(rows: list[dict], key: Callable[[dict], Any]) -> dict[Any, list[dict]]:
    indexed: dict[Any, list[dict]] = {}
    for row in rows:
        value = key(row)
        if value:
            indexed.setdefault(value, []).append(row)
    return indexed


def _same_source_identity(left: dict, right: dict) -> bool:
    return _source_identity(left) is not None and _source_identity(left) == _source_identity(right)


def _record_edge(
    edges: dict[tuple[str, str, str], dict],
    left: str,
    right: str,
    reason: str,
    evidence: dict,
) -> None:
    pair = (left, right) if left < right else (right, left)
    edges[(pair[0], pair[1], reason)] = evidence


def _join_star(
    disjoint: "_DisjointSet",
    edges: dict[tuple[str, str, str], dict],
    rows: list[dict],
    reason: str,
    evidence: dict,
) -> None:
    """Union one star per equivalence key, keeping strong edges linear in record count."""
    ordered = sorted(rows, key=lambda row: row["asset_id"])
    if len(ordered) < 2:
        return
    left = ordered[0]["asset_id"]
    for row in ordered[1:]:
        right = row["asset_id"]
        disjoint.union(left, right)
        _record_edge(edges, left, right, reason, dict(evidence))


def _render_edges(edges: dict[tuple[str, str, str], dict]) -> list[dict]:
    return [
        {"left": left, "right": right, "reason": reason, "evidence": dict(evidence)}
        for (left, right, reason), evidence in sorted(edges.items())
    ]


def _unique_text(rows: list[dict], field: str) -> list[str]:
    return sorted({_text(row.get(field)) for row in rows if _text(row.get(field))})


def _assigned_splits(rows: list[dict]) -> list[str]:
    return sorted({_text(row.get("split")) for row in rows} & ASSIGNED_SPLITS)


def _absolute_timestamp(value: Any) -> bool:
    try:
        return datetime.fromisoformat(_text(value)).tzinfo is not None
    except ValueError:
        return False


class _DisjointSet:
    def __init__(self, identities: list[str]):
        self._parent = {identity: identity for identity in identities}

    def find(self, identity: str) -> str:
        parent = self._parent
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def _group_id(asset_ids: list[str]) -> str:
    payload = json.dumps(asset_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_lineage_groups(records: list[dict]) -> dict:
    """Group projected metadata without reading content or assigning splits.

    Strong unions are intentionally narrow: shared sessions, whole-transcript
    hashes, exact source identities, and public source-repository families. A
    source-repository resolution is only a statement that the public repository
    family can move as one unit; it is not evidence of independent problem lineage.
    """
    by_id: dict[str, dict] = {}
    for row in records:
        asset_id = _text(row.get("asset_id"))
        if not asset_id:
            raise ValueError("lineage metadata requires a nonempty asset_id")
        if asset_id in by_id:
            raise ValueError(f"duplicate lineage asset_id: {asset_id}")
        projected = dict(row)
        projected["asset_id"] = asset_id
        by_id[asset_id] = projected

    ordered_ids = sorted(by_id)
    rows = [by_id[asset_id] for asset_id in ordered_ids]
    disjoint = _DisjointSet(ordered_ids)
    strong_edges: dict[tuple[str, str, str], dict] = {}

    for session_id, session_rows in _keyed(rows, lambda row: _text(row.get("session_id"))).items():
        _join_star(
            disjoint,
            strong_edges,
            session_rows,
            SESSION_EDGE,
            {"session_id": session_id},
        )

    complete_transcripts = [
        row for row in rows if row.get("hash_lineage_scope") == "complete_session_transcript"
    ]
    for (sha, size), hash_rows in _keyed(complete_transcripts, _positive_content_key).items():
        _join_star(
            disjoint,
            strong_edges,
            hash_rows,
            HASH_EDGE,
            {"sha256": sha, "size": size},
        )

    for (namespace, source_id), identity_rows in _keyed(rows, _source_identity).items():
        _join_star(
            disjoint,
            strong_edges,
            identity_rows,
            SOURCE_ID_EDGE,
            {"source_namespace": namespace, "source_id": source_id},
        )

    for repo_identity, repo_rows in _keyed(rows, _public_repo_identity).items():
        _join_star(
            disjoint,
            strong_edges,
            repo_rows,
            PUBLIC_REPO_EDGE,
            {
                "repo_identity": repo_identity,
                "public_catalog": True,
                "record_types": sorted({_text(row.get("record_type")) for row in repo_rows}),
                "revisions": _unique_text(repo_rows, "revision"),
            },
        )

    reference_index: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        namespace = _text(row.get("source_namespace"))
        source_id = _text(row.get("source_id"))
        if namespace and source_id:
            reference_index.setdefault((namespace, source_id), []).append(row["asset_id"])

    candidate_edges: dict[tuple[str, str, str], dict] = {}
    for (sha, size), hash_rows in _keyed(rows, _positive_content_key).items():
        representative = min(hash_rows, key=lambda row: row["asset_id"])["asset_id"]
        for row in hash_rows:
            if disjoint.find(representative) != disjoint.find(row["asset_id"]):
                _record_edge(
                    candidate_edges,
                    representative,
                    row["asset_id"],
                    HASH_CANDIDATE_EDGE,
                    {
                        "sha256": sha,
                        "size": size,
                        "claim": "equal bytes; whole-problem equivalence not established",
                    },
                )
    unresolved_references: list[dict] = []
    unresolved_by_asset: dict[str, list[dict]] = {}
    seen_unresolved: set[tuple[str, str, str]] = set()

    for row in rows:
        asset_id = row["asset_id"]
        namespace = _text(row.get("related_source_namespace")) or _text(row.get("source_namespace"))
        related = row.get("related_source_ids") or []
        for raw_reference in related:
            source_id = _text(raw_reference)
            if not source_id:
                continue
            targets = sorted(
                target
                for target in reference_index.get((namespace, source_id), [])
                if target != asset_id
            )
            if not targets:
                unresolved_key = (asset_id, namespace, source_id)
                if unresolved_key not in seen_unresolved:
                    seen_unresolved.add(unresolved_key)
                    reference = {
                        "asset_id": asset_id,
                        "related_source_namespace": namespace,
                        "related_source_id": source_id,
                    }
                    unresolved_references.append(reference)
                    unresolved_by_asset.setdefault(asset_id, []).append(reference)
                continue
            target = next(
                (
                    candidate
                    for candidate in targets
                    if disjoint.find(asset_id) != disjoint.find(candidate)
                ),
                None,
            )
            if target is None:
                continue
            _record_edge(
                candidate_edges,
                asset_id,
                target,
                RELATED_EDGE,
                {
                    "related_source_namespace": namespace,
                    "related_source_id": source_id,
                    "claim": "explicit map relation; not resolved problem lineage",
                },
            )

    for (repo_identity, declared_family), topic_rows in _keyed(
        rows,
        lambda row: (
            (_text(row.get("repo_identity")), _text(row.get("declared_family")))
            if _text(row.get("repo_identity")) and _text(row.get("declared_family"))
            else None
        ),
    ).items():
        ordered = sorted(topic_rows, key=lambda row: row["asset_id"])
        representative = ordered[0]
        for other in ordered[1:]:
            if _same_source_identity(representative, other):
                continue
            if disjoint.find(representative["asset_id"]) == disjoint.find(other["asset_id"]):
                continue
            _record_edge(
                candidate_edges,
                representative["asset_id"],
                other["asset_id"],
                TOPIC_EDGE,
                {
                    "repo_identity": repo_identity,
                    "declared_family": declared_family,
                    "claim": "repo/topic candidate; not same-problem evidence",
                },
            )

    roots: dict[str, list[str]] = {}
    for asset_id in ordered_ids:
        roots.setdefault(disjoint.find(asset_id), []).append(asset_id)

    basis_by_root: dict[str, set[str]] = {}
    for left, _right, reason in strong_edges:
        basis_by_root.setdefault(disjoint.find(left), set()).add(reason)

    groups: list[dict] = []
    conflicts: list[dict] = []
    for root, member_ids in roots.items():
        members = sorted(member_ids)
        group_rows = [by_id[asset_id] for asset_id in members]
        group_id = _group_id(members)
        basis = sorted(basis_by_root.get(root, set()))
        record_types = {_text(row.get("record_type")) for row in group_rows}
        repos = _unique_text(group_rows, "repo_identity")
        revisions = _unique_text(group_rows, "revision")
        splits = _assigned_splits(group_rows)

        split_conflict = len(splits) > 1
        if split_conflict:
            by_split: dict[str, list[str]] = {}
            for row in group_rows:
                split = _text(row.get("split"))
                if split in ASSIGNED_SPLITS:
                    by_split.setdefault(split, []).append(row["asset_id"])
            conflicts.append(
                {
                    "kind": "cross_split_conflict",
                    "group_id": group_id,
                    "details": {
                        "splits": splits,
                        "asset_ids_by_split": {
                            split: sorted(identities) for split, identities in sorted(by_split.items())
                        },
                    },
                }
            )

        source_only = record_types <= SOURCE_RECORD_TYPES
        if source_only and len(repos) == 1 and len(revisions) > 1:
            conflicts.append(
                {
                    "kind": "revision_drift",
                    "group_id": group_id,
                    "details": {
                        "repo_identity": repos[0],
                        "revisions": revisions,
                        "asset_ids": members,
                    },
                }
            )

        unknown_references = [
            reference for asset_id in members for reference in unresolved_by_asset.get(asset_id, [])
        ]

        public_scope = all(row.get("public_catalog") is True for row in group_rows)
        repo_scope = len(repos) == 1 and all(_text(row.get("repo_identity")) for row in group_rows)
        revision_scope = all(
            re.fullmatch(r"[0-9a-f]{40,64}", _text(row.get("revision"))) for row in group_rows
        )
        timestamp_scope = all(_absolute_timestamp(row.get("timestamp")) for row in group_rows)
        has_repo_basis = PUBLIC_REPO_EDGE in basis
        has_session_basis = SESSION_EDGE in basis
        resolved = (
            source_only
            and public_scope
            and repo_scope
            and revision_scope
            and timestamp_scope
            and (len(members) == 1 or has_repo_basis)
            and not has_session_basis
            and not split_conflict
            and not unknown_references
        )

        reasons: set[str] = set()
        if not source_only:
            reasons.add("unsupported_record_type")
        if not public_scope:
            reasons.add("missing_public_catalog")
        if any(not _text(row.get("repo_identity")) for row in group_rows):
            reasons.add("missing_repo_identity")
        if len(repos) > 1:
            reasons.add("mixed_repo_identity")
        if not revision_scope:
            reasons.add("missing_revision")
        if not timestamp_scope:
            reasons.add("missing_timestamp")
        if has_session_basis:
            reasons.add("session_coupling_not_problem_lineage")
        if split_conflict:
            reasons.add("cross_split_conflict")
        if unknown_references:
            reasons.add("unknown_related_reference")
        if len(members) > 1 and not has_repo_basis:
            reasons.add("insufficient_source_repository_evidence")
        if not resolved and not reasons:
            reasons.add("problem_lineage_not_established")

        groups.append(
            {
                "group_id": group_id,
                "asset_ids": members,
                "source_collections": _unique_text(group_rows, "source_collection"),
                "source_namespaces": _unique_text(group_rows, "source_namespace"),
                "source_ids": _unique_text(group_rows, "source_id"),
                "session_ids": _unique_text(group_rows, "session_id"),
                "repo_identities": repos,
                "revisions": revisions,
                "declared_family_candidates": _unique_text(group_rows, "declared_family"),
                "inherited_splits": splits,
                "grouping_basis": basis,
                "lineage_resolved": resolved,
                "resolution_scope": (
                    "public_source_repository" if resolved else "unresolved_problem_lineage"
                ),
                "unresolved_reasons": [] if resolved else sorted(reasons),
            }
        )

    groups.sort(key=lambda group: group["group_id"])
    conflicts.sort(key=lambda conflict: (conflict["kind"], conflict["group_id"]))
    unresolved_references.sort(
        key=lambda reference: (
            reference["asset_id"],
            reference["related_source_namespace"],
            reference["related_source_id"],
        )
    )
    return {
        "groups": groups,
        "strong_edges": _render_edges(strong_edges),
        "candidate_edges": _render_edges(candidate_edges),
        "conflicts": conflicts,
        "unresolved_references": unresolved_references,
    }
