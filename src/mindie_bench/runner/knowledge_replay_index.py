"""Read-only graded knowledge index used by offline replay.

[Role]
Delegate graded corpus loading, condition matching, navigation and dependency
closure to the production `mindie_knowledge.graded` package, then expose the
small compatibility projection consumed by `knowledge_replay.py`.

[Invariants]
- Production `GradedUnit` and `RoutingIndex` own parsing, BM25 navigation and
  closure traversal; this module only projects their results for replay.
- Navigation cards never inline unit bodies; `explain_closure` returns complete
  bodies only and reports `budget_excluded` instead of truncating a procedure.
- `known_refs` is exact `id@revision`. A bare `unit-id` resolves to the current
  revision in this replay adapter, while an outdated revision is never visible.
- A shared parent appears once in a closure; `known_refs` is the exact set of
  complete unit versions already visible in the caller's current context.
- Token estimates use the production UTF-8 body bytes / 4 ceiling estimator.

[Timeouts & Locks]
- Bounded read-only file reads during construction; pure in-memory query and
  closure operations afterwards. No waits, subprocesses or locks.

[Side-Effect Policy]
- Read-only corpus access. This module never writes corpus or workspace files.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_KNOWLEDGE_ROOT = Path("/Users/maoxx241/code/mindie-release-20261014/knowledge")


def _import_graded():
    """Import the production graded index, bootstrapping its checkout if needed."""
    try:
        from mindie_knowledge.graded.conditions import match_conditions
        from mindie_knowledge.graded.index import (
            RoutingIndex as ProductionRoutingIndex,
        )
        from mindie_knowledge.graded.index import token_estimate as body_token_estimate
    except ImportError:
        if str(_KNOWLEDGE_ROOT) not in sys.path and _KNOWLEDGE_ROOT.is_dir():
            sys.path.insert(0, str(_KNOWLEDGE_ROOT))
        from mindie_knowledge.graded.conditions import match_conditions
        from mindie_knowledge.graded.index import (
            RoutingIndex as ProductionRoutingIndex,
        )
        from mindie_knowledge.graded.index import token_estimate as body_token_estimate
    return ProductionRoutingIndex, body_token_estimate, match_conditions


ProductionRoutingIndex, _body_tokens, _match_conditions = _import_graded()

RECORD_OVERHEAD = 24
TIER_RANK = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}


def seq(value: Any) -> list[Any]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence) else [value]


def estimate_tokens(value: Any) -> int:
    return -(-len(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()) // 4)


def as_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def unit_id(ref: Any) -> str:
    return str(ref).strip().split("@", 1)[0]


def condition_status(
    required: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[str, list[str], list[str]]:
    """Project the production applicability decision onto the legacy tuple."""
    observed = dict(observed or {})
    result = _match_conditions(dict(required), observed)
    status = {"mismatch": "condition_mismatch", "unknown": "condition_unknown"}.get(
        result.status, "matched")
    if status == "matched":
        return status, [], []
    missing = [key for key in required
               if key not in observed or observed[key] is None or observed[key] == "unknown"]
    unmatched = [key for key in required if key not in missing]
    return status, missing, unmatched


def _match_status(value: str) -> str:
    return {"mismatched": "condition_mismatch"}.get(str(value), str(value))


class RoutingIndex:
    """Graded corpus adapter whose matching and closure come from production."""

    def __init__(self, corpus_root: str | Path):
        self.root = Path(corpus_root)
        self._index = ProductionRoutingIndex.from_root(self.root)
        # Compatibility projection: replay code addresses units by these rows
        # and never mutates the production grading metadata.
        self.units: dict[str, dict[str, Any]] = {}
        for identity, unit in self._index.units.items():
            metadata = unit.metadata
            self.units[identity] = {
                "id": identity,
                "revision": unit.revision,
                "tier": str(metadata.get("tier", "")).upper(),
                "category": str(metadata.get("category", "")),
                "title": str(metadata.get("title", "")),
                "conditions": dict(metadata.get("conditions") or {}),
                "inherits_from": [str(parent) for parent in metadata.get("inherits_from", [])],
                "keywords": [str(word) for word in (metadata.get("triggers") or {}).get("keywords", [])],
                "token_estimate": _body_tokens(unit.body),
                "body": unit.body,
            }

    def unit(self, ref: Any) -> dict[str, Any]:
        return self.units[unit_id(ref)]

    def closure(self, refs: Sequence[Any]) -> list[str]:
        """Production topological ancestors plus each requested unit, deduplicated."""
        order: list[str] = []
        for ref in seq(refs):
            identity = unit_id(ref)
            if identity not in self._index.units:
                continue
            for member in [*self._index.closure[identity], identity]:
                if member not in order:
                    order.append(member)
        return order

    def resolve_known_refs(self, refs: Sequence[Any]) -> list[str]:
        """Return exact visible refs, upgrading bare IDs but never old revisions."""
        exact: set[str] = set()
        for raw in seq(refs):
            text = str(raw).strip()
            identity = unit_id(text)
            unit = self.units.get(identity)
            if unit is None:
                continue
            if "@" in text:
                revision = text.rsplit("@", 1)[1]
                if not revision.isdigit() or int(revision) != unit["revision"]:
                    continue
            exact.add(f"{identity}@{unit['revision']}")
        return sorted(exact)

    def _card(self, card: Mapping[str, Any]) -> dict[str, Any]:
        identity = card["id"]
        return {
            "ref": f"{identity}@{card['revision']}",
            "unit_id": identity,
            "id": identity,
            "revision": card["revision"],
            "tier": card["tier"],
            "category": card["category"],
            "title": card["title"],
            "match_status": _match_status(card["match_status"]),
            "ready_to_apply": card["ready_to_apply"],
            "missing_conditions": list(card.get("missing_conditions") or []),
            "body_token_estimate": card["body_token_estimate"],
            "closure_token_estimate": card["closure_token_estimate"],
            "token_estimate": card["body_token_estimate"],
            "score": card.get("score", 0.0),
        }

    def query(
        self, query: str, conditions: Mapping[str, Any] | None = None, *,
        token_budget: int = 1024, limit: int = 8, include_unknown: bool = True,
    ) -> list[dict[str, Any]]:
        """Return bounded navigation cards; never inlines a body."""
        response = self._index.query(
            str(query), conditions=None if conditions is None else dict(conditions),
            token_budget=token_budget, limit=limit, include_unknown=include_unknown,
        )
        return [self._card(card) for card in response["cards"]]

    def legacy_query(
        self, query: str, conditions: Mapping[str, Any] | None = None, *,
        token_budget: int = 1024, limit: int = 8, include_unknown: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the production result view (excerpts and expanded parents)."""
        response = self._index.query(
            str(query), conditions=None if conditions is None else dict(conditions),
            token_budget=token_budget, limit=limit, include_unknown=include_unknown,
        )
        rows = []
        for row in response["results"]:
            identity = row["id"]
            projection = self.units[identity]
            rows.append({
                **row,
                "ref": f"{identity}@{row['revision']}",
                "unit_id": identity,
                "match_status": _match_status(row["match_status"]),
                "token_estimate": projection["token_estimate"],
                "body_token_estimate": projection["token_estimate"],
                "body": row.get("excerpt", ""),
            })
        return rows

    def _closure_unit(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        identity = entry["id"]
        body = str(entry.get("content", ""))
        return {
            "ref": f"{identity}@{entry['revision']}",
            "unit_id": identity,
            "id": identity,
            "revision": entry["revision"],
            "tier": entry["tier"],
            "title": entry["title"],
            "token_estimate": _body_tokens(body),
            "body_token_estimate": _body_tokens(body),
            "body": body,
            "ready_to_apply": bool(entry.get("ready_to_apply")),
            "missing_conditions": list(entry.get("missing_conditions") or []),
        }

    def _unresolved(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        identity = str(entry.get("id", ""))
        projection = self.units.get(identity)
        mapped: dict[str, Any] = {
            "ref": (f"{identity}@{projection['revision']}" if projection else identity),
            "reason": str(entry.get("status", "unresolved")),
        }
        for key in ("required_tokens", "current_revision", "superseded_by", "missing_conditions"):
            if key in entry:
                mapped[key] = entry[key]
        return mapped

    def explain_closure(
        self, refs: Sequence[Any], *, known_refs: Sequence[Any] = (),
        conditions: Mapping[str, Any] | None = None, token_budget: int = 4096,
    ) -> dict[str, Any]:
        """Return complete closure units only: no truncated procedural body ever leaves."""
        requested_refs = list(seq(refs))
        response = self._index.explain_closure(
            requested_refs, conditions=None if conditions is None else dict(conditions),
            known_refs=self.resolve_known_refs(known_refs), token_budget=token_budget,
        )
        delivered = {entry["id"]: entry
                     for entry in [*response["requires"], *response["units"]]}
        unresolved_ids = {str(entry.get("id")) for entry in response["unresolved"]}
        requested = {unit_id(ref) for ref in requested_refs}
        units = [self._closure_unit(delivered[identity])
                 for identity in self.closure(requested_refs)
                 if identity in delivered and identity not in unresolved_ids]
        requires = [unit for unit in units if unit["unit_id"] not in requested]
        unresolved = [self._unresolved(entry) for entry in response["unresolved"]]
        usage = dict(response["usage"])
        usage["budget"] = token_budget
        usage["complete"] = not unresolved
        return {"units": units, "requires": requires, "unresolved": unresolved,
                "usage": usage, "estimator": "utf8_bytes_div_4_ceiling"}


__all__ = [
    "RECORD_OVERHEAD", "TIER_RANK", "RoutingIndex", "as_int", "condition_status",
    "estimate_tokens", "seq", "unit_id",
]
