"""Read-only graded knowledge index used by offline replay.

[Role]
Load a graded corpus routing manifest (or bare `unit-*.md` files), expose
navigation-card queries and complete dependency-closure reads for replay.

[Invariants]
- Navigation cards never inline unit bodies; `explain_closure` returns complete
  bodies only and reports `budget_excluded` instead of truncating a procedure.
- Conditions are conjunctive: missing facts are `condition_unknown`, explicit
  contradictions are `condition_mismatch`; unknown never counts as matched.
- A shared parent appears once in a closure; `known_refs` is the set of complete
  unit IDs already visible in the caller's current context.
- Token estimates use UTF-8 body bytes / 4, matching the routing manifest.

[Timeouts & Locks]
- Bounded read-only file reads during construction; pure in-memory query and
  closure operations afterwards. No waits, subprocesses or locks.

[Side-Effect Policy]
- Read-only corpus access. This module never writes corpus or workspace files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

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


def _compare(rule: str, value: Any) -> bool | None:
    """Evaluate the simple numeric comparisons used by declared conditions."""
    for operator in (">=", "<=", "==", "!=", ">", "<"):
        if rule.startswith(operator):
            try:
                left, right = float(value), float(rule[len(operator):])
            except (TypeError, ValueError):
                return None
            return {">=": left >= right, "<=": left <= right, "==": left == right,
                    "!=": left != right, ">": left > right, "<": left < right}[operator]
    return None


def _match_value(rule: Any, value: Any) -> bool:
    if isinstance(rule, Mapping):
        exact = rule.get("exact_set")
        if exact is not None:
            return set(seq(value)) == set(seq(exact))
        return isinstance(value, Mapping) and condition_status(rule, value)[0] == "matched"
    if isinstance(rule, str) and (comparison := _compare(rule.strip(), value)) is not None:
        return comparison
    if isinstance(rule, Sequence) and not isinstance(rule, str):
        return set(value) <= set(rule) if isinstance(value, Sequence) and not isinstance(value, str) else value in rule
    return value == rule


def condition_status(
    required: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[str, list[str], list[str]]:
    """Return matched/condition_unknown/condition_mismatch plus missing/unmatched keys."""
    missing, unmatched = [], []
    for key, rule in required.items():
        if key not in observed or observed[key] is None or observed[key] == "unknown":
            missing.append(key)
        elif not _match_value(rule, observed[key]):
            unmatched.append(key)
    if unmatched:
        return "condition_mismatch", missing, unmatched
    return ("condition_unknown", missing, []) if missing else ("matched", [], [])


class RoutingIndex:
    """Graded corpus index with navigation cards and complete closure reads."""

    def __init__(self, corpus_root: str | Path):
        self.root = Path(corpus_root)
        rows: list[dict[str, Any]] = []
        manifests = (self.root / "index" / "routing-manifest.json",
                     self.root / "routing-manifest.json")
        manifest = next((path for path in manifests if path.is_file()), None)
        if manifest is not None:
            rows = list(json.loads(manifest.read_text(encoding="utf-8")).get("units", []))
        else:
            for path in sorted(self.root.rglob("unit-*.md")):
                raw = path.read_text(encoding="utf-8")
                if raw.startswith("---\n") and "\n---\n" in raw:
                    metadata = yaml.safe_load(raw.split("\n---\n", 1)[0][4:])
                    metadata["path"] = path.relative_to(self.root).as_posix()
                    rows.append(metadata)
        self.units: dict[str, dict[str, Any]] = {}
        for row in rows:
            raw = (self.root / str(row["path"])).read_text(encoding="utf-8")
            body = raw.split("\n---\n", 1)[1] if "\n---\n" in raw else raw
            identity = str(row["id"])
            self.units[identity] = {
                "id": identity, "revision": as_int(row.get("revision"), 1),
                "tier": str(row.get("tier", "")).upper(), "category": str(row.get("category", "")),
                "title": str(row.get("title", "")), "conditions": dict(row.get("conditions") or {}),
                "inherits_from": [str(parent) for parent in row.get("inherits_from", [])],
                "keywords": [str(word) for word in (row.get("triggers") or {}).get("keywords", [])],
                "token_estimate": as_int(row.get("token_estimate")) or -(-len(body.encode()) // 4),
                "body": body,
            }
        if not self.units:
            raise ValueError("corpus_root contains no graded knowledge units")

    def unit(self, ref: Any) -> dict[str, Any]:
        return self.units[unit_id(ref)]

    def closure(self, refs: Sequence[Any]) -> list[str]:
        seen: set[str] = set()
        order: list[str] = []

        def visit(identity: str) -> None:
            if identity in seen or identity not in self.units:
                return
            seen.add(identity)
            for parent in self.units[identity]["inherits_from"]:
                visit(parent)
            order.append(identity)

        for ref in refs:
            visit(unit_id(ref))
        return order

    def query(
        self, query: str, conditions: Mapping[str, Any] | None = None, *,
        token_budget: int = 1024, limit: int = 8, include_unknown: bool = True,
    ) -> list[dict[str, Any]]:
        """Return bounded navigation cards; never inlines a body."""
        terms = [term for term in str(query).lower().replace("/", " ").split() if term]
        candidates: list[dict[str, Any]] = []
        for unit in self.units.values():
            status, missing, unmatched = condition_status(unit["conditions"], conditions or {})
            if status == "condition_mismatch" or (status == "condition_unknown" and not include_unknown):
                continue
            text = " ".join([unit["id"], unit["title"], unit["category"], " ".join(unit["keywords"])]).lower()
            score = sum(term in text for term in terms) + (2 if status == "matched" else 0)
            if not terms or score:
                candidates.append({
                    "ref": f"{unit['id']}@{unit['revision']}", "unit_id": unit["id"],
                    "revision": unit["revision"], "tier": unit["tier"],
                    "category": unit["category"], "title": unit["title"],
                    "match_status": status, "ready_to_apply": status == "matched",
                    "missing_conditions": missing, "unmatched_conditions": unmatched,
                    "token_estimate": unit["token_estimate"], "score": score,
                })
        candidates.sort(key=lambda card: (-card["score"], TIER_RANK.get(card["tier"], 9),
                                         card["unit_id"]))
        cards: list[dict[str, Any]] = []
        for card in candidates:
            if estimate_tokens(cards + [card]) <= token_budget:
                cards.append(card)
                if len(cards) >= limit:
                    break
        return cards

    def explain_closure(
        self, refs: Sequence[Any], *, known_refs: Sequence[Any] = (),
        conditions: Mapping[str, Any] | None = None, token_budget: int = 4096,
    ) -> dict[str, Any]:
        """Return complete closure units only: no truncated procedural body ever leaves."""
        known = {unit_id(ref) for ref in known_refs}
        required = {unit_id(ref) for ref in refs}
        units: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        used = 0
        for identity in self.closure(refs):
            unit = self.units[identity]
            ref = f"{unit['id']}@{unit['revision']}"
            if conditions is not None and condition_status(unit["conditions"], conditions)[0] == "condition_mismatch":
                unresolved.append({"ref": ref, "reason": "condition_mismatch"})
            elif identity not in known:
                cost = unit["token_estimate"] + RECORD_OVERHEAD
                if used + cost > token_budget:
                    unresolved.append({"ref": ref, "reason": "budget_excluded"})
                else:
                    units.append({"ref": ref, "unit_id": identity, "revision": unit["revision"],
                                  "tier": unit["tier"], "title": unit["title"],
                                  "token_estimate": unit["token_estimate"], "body": unit["body"]})
                    used += cost
        return {"units": units, "requires": [unit for unit in units if unit["unit_id"] not in required],
                "unresolved": unresolved,
                "usage": {"budget": token_budget, "estimated_tokens": used, "complete": not unresolved}}
