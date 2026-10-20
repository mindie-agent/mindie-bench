"""Offline knowledge replay for paired loading configs and long-horizon growth.

[Role]
Replay the three Section 13.4 loading configurations over a task suite and the
four Section 13.5 growth milestones, reporting cost, readiness and trajectory
metrics without calling a model or touching a live corpus.

[Invariants]
- Config A loads no domain knowledge. Config B preserves legacy repeated parent
  expansion and truncated non-L1 excerpts. Config C returns only complete
  closure units, deduplicates `known_refs` across turns and re-reads after a
  native compaction only what is required.
- Token estimates are UTF-8 body bytes / 4, matching the routing manifest; a
  response is charged for navigation cards plus complete unit bodies.
- Loading-config comparisons are simulations over real corpus reads: they are
  marked `simulation_only` and name the `utf8_bytes_div_4_ceiling` estimator.
- Growth ratios are `None` when no verified milestone exists. Merges and
  supersedes never count as new active units; a branch retains a new condition.
- Rare conditions stay recallable after consolidation unless explicitly dropped.

[Timeouts & Locks]
- Pure in-memory computation with bounded read-only corpus file reads; no waits,
  subprocesses or locks.

[Side-Effect Policy]
- Read-only corpus access. This module never writes corpus, workspace or reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mindie_bench.runner import knowledge_replay_index as _index

RoutingIndex = _index.RoutingIndex
_int = _index.as_int
_condition_status = _index.condition_status
_tokens = _index.estimate_tokens
_seq = _index.seq
_unit_id = _index.unit_id
_RECORD_OVERHEAD = _index.RECORD_OVERHEAD

SCHEMA_VERSION = "mindie-bench-knowledge-replay/1"
CATEGORIES = ("0day", "diagnose", "operator_perf", "model_perf")
LEGACY_EXCERPT_TOKENS = 512


def _category(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    normalized = {"zero_day": "0day", "operator": "operator_perf", "operator_performance":
                  "operator_perf", "model": "model_perf", "model_performance": "model_perf"}.get(
        normalized, normalized)
    if normalized not in CATEGORIES:
        raise ValueError(f"unknown task category: {value!r}")
    return normalized


def _turns(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = task.get("turns") or [{}]
    if isinstance(raw, Mapping):
        raw = [raw]
    return [{**task, **(turn if isinstance(turn, Mapping) else {"query": str(turn)})}
            for turn in raw]


def _required(turn: Mapping[str, Any]) -> list[str]:
    return [_unit_id(ref) for ref in _seq(turn.get("required_refs"))]


def _empty_metrics() -> dict[str, int]:
    return {key: 0 for key in (
        "task_count", "turn_count", "ready_turns", "knowledge_turns", "total_loaded_tokens",
        "duplicate_parent_tokens", "truncated_procedural_units", "compaction_reread_tokens",
        "steps", "attempts")}


def _accumulate(total: dict[str, int], row: Mapping[str, int]) -> None:
    for key in total:
        total[key] += int(row.get(key, 0))


def _finalize(metrics: Mapping[str, int]) -> dict[str, Any]:
    tasks, knowledge = metrics["task_count"], metrics["knowledge_turns"]
    return {
        "task_count": tasks, "turn_count": metrics["turn_count"],
        "total_loaded_tokens": metrics["total_loaded_tokens"],
        "duplicate_parent_tokens": metrics["duplicate_parent_tokens"],
        "truncated_procedural_units": metrics["truncated_procedural_units"],
        "ready_to_apply_rate": round(metrics["ready_turns"] / knowledge, 6) if knowledge else 0.0,
        "compaction_reread_tokens": metrics["compaction_reread_tokens"],
        "mean_steps": round(metrics["steps"] / tasks, 6) if tasks else 0.0,
        "mean_attempts": round(metrics["attempts"] / tasks, 6) if tasks else 0.0,
    }


def _simulate(index: RoutingIndex, task: Mapping[str, Any], config: str,
              nav_budget: int, body_budget: int) -> dict[str, int]:
    turns = _turns(task)
    row = _empty_metrics()
    row["task_count"], row["turn_count"] = 1, len(turns)
    baseline_steps = _int(task.get("baseline_steps"), len(_required(task)) + 2)
    baseline_attempts = _int(task.get("baseline_attempts"), 2)
    if config == "A":
        row["steps"], row["attempts"] = baseline_steps, baseline_attempts
        row["knowledge_turns"] = sum(bool(_required(turn)) for turn in turns)
        return row
    extra_steps = extra_attempts = 0
    visible: set[str] = set()
    for turn in turns:
        required = _required(turn)
        conditions = dict(turn.get("conditions") or {})
        cards = index.query(str(turn.get("query", task.get("task_id", ""))), conditions,
                            token_budget=nav_budget, include_unknown=True)
        row["knowledge_turns"] += int(bool(required))
        available: set[str] = set()
        if config == "B":
            loaded, seen_parents = _tokens(cards), set()
            for card in cards:
                unit, complete = index.unit(card["unit_id"]), True
                body = unit["token_estimate"]
                if unit["tier"] != "L1" and body > LEGACY_EXCERPT_TOKENS:
                    body, complete = LEGACY_EXCERPT_TOKENS, False
                    row["truncated_procedural_units"] += 1
                loaded += body
                if complete:
                    available.add(unit["id"])
                for parent_id in unit["inherits_from"]:
                    parent = index.unit(parent_id)
                    if parent_id in seen_parents:
                        row["duplicate_parent_tokens"] += parent["token_estimate"]
                    seen_parents.add(parent_id)
                    loaded += parent["token_estimate"]
                    if parent["tier"] == "L1" or parent["token_estimate"] <= LEGACY_EXCERPT_TOKENS:
                        available.add(parent_id)
            row["total_loaded_tokens"] += loaded
            if turn.get("compacted"):
                row["compaction_reread_tokens"] += loaded
        else:
            if turn.get("compacted"):
                # Exact id@revision visibility only; a bare retained ID resolves
                # to the current revision, while an outdated version stays unseen.
                visible = set(index.resolve_known_refs(_seq(turn.get("retained_refs"))))
            explanation = index.explain_closure(required, known_refs=sorted(visible),
                                                conditions=conditions, token_budget=body_budget)
            body_loaded = sum(unit["token_estimate"] + _RECORD_OVERHEAD
                              for unit in explanation["units"])
            visible.update(index.resolve_known_refs(
                [f"{unit['unit_id']}@{unit['revision']}" for unit in explanation["units"]]))
            available = {_unit_id(ref) for ref in visible}
            row["total_loaded_tokens"] += _tokens(cards) + body_loaded
            if turn.get("compacted"):
                row["compaction_reread_tokens"] += body_loaded
        closure = index.closure(required)
        ready = bool(required) and set(required) <= set(closure) and all(
            identity in available and _condition_status(
                index.units[identity]["conditions"], conditions)[0] == "matched"
            for identity in closure)
        row["ready_turns"] += int(ready)
        if not ready:
            extra_steps += 1
            extra_attempts += 1
    row["steps"], row["attempts"] = baseline_steps + extra_steps, baseline_attempts + extra_attempts
    return row


def compare_loading_configs(
    tasks: Sequence[Mapping[str, Any]], *, corpus_root: str | Path,
    nav_budget: int = 1024, body_budget: int = 4096,
) -> dict[str, Any]:
    """Compare A baseline, B legacy dump and C progressive closure over paired tasks."""
    if nav_budget < 1 or body_budget < 1:
        raise ValueError("nav_budget and body_budget must be positive")
    index = RoutingIndex(corpus_root)
    suite = [{**dict(task), "category": _category(
        task.get("category", task.get("task_type", task.get("kind"))))} for task in tasks]
    configs: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, Any]] = {category: {} for category in CATEGORIES}
    for name in ("A", "B", "C"):
        total = _empty_metrics()
        per_category = {category: _empty_metrics() for category in CATEGORIES}
        for task in suite:
            row = _simulate(index, task, name, nav_budget, body_budget)
            _accumulate(total, row)
            _accumulate(per_category[task["category"]], row)
        configs[name] = _finalize(total)
        configs[name]["name"] = {"A": "no_domain_knowledge", "B": "legacy_query_dump",
                                 "C": "progressive_navigation_and_closure"}[name]
        configs[name]["config_name"] = configs[name]["name"]
        configs[name]["categories"] = {category: _finalize(per_category[category])
                                       for category in CATEGORIES}
        for category in CATEGORIES:
            categories[category][name] = configs[name]["categories"][category]
    savings = configs["B"]["total_loaded_tokens"] - configs["C"]["total_loaded_tokens"]
    configs["C"]["net_token_savings_c_vs_b"] = savings
    configs["A"]["net_token_savings_c_vs_b"] = configs["B"]["net_token_savings_c_vs_b"] = 0
    for category in CATEGORIES:
        category_savings = (categories[category]["B"]["total_loaded_tokens"]
                            - categories[category]["C"]["total_loaded_tokens"])
        for name in ("A", "B", "C"):
            categories[category][name]["net_token_savings_c_vs_b"] = (
                category_savings if name == "C" else 0)
    return {
        "schema_version": SCHEMA_VERSION, "simulation_only": True,
        "estimator": "utf8_bytes_div_4_ceiling", "corpus_root": str(corpus_root),
        "nav_budget": nav_budget, "body_budget": body_budget, "task_count": len(suite),
        "configs": configs, "config_a": configs["A"], "config_b": configs["B"],
        "config_c": configs["C"], "categories": categories,
        "net_token_savings_c_vs_b": savings,
    }


_ACTION_ALIASES = {
    "merge": "merge_into_existing", "merge_into_existing": "merge_into_existing",
    "update": "merge_into_existing", "revise": "merge_into_existing",
    "noop": "noop", "no_op": "noop", "create": "create", "new": "create", "branch": "branch",
    "split": "branch", "supersede": "supersede", "replace": "supersede",
    "consolidate": "consolidate", "merge_units": "consolidate", "archive": "archive",
    "retire": "archive", "remove": "archive",
}
_GROWTH_CATEGORIES = {
    "repeat": "repeated_success", "repeated_success": "repeated_success",
    "same_topic_success": "repeated_success", "repeated_same_topic": "repeated_success",
    "repeated_same_topic_success": "repeated_success", "same_topic": "repeated_success",
    "new_mechanism": "new_mechanism", "new_mechanism_discovery": "new_mechanism",
    "new_domain": "new_mechanism", "discovery": "new_mechanism",
    "version_contradiction": "version_contradiction", "contradiction": "version_contradiction",
    "software_version_contradiction": "version_contradiction",
    "version_migration": "version_contradiction", "rare_condition": "rare_condition",
    "rare": "rare_condition", "low_frequency": "rare_condition",
    "rare_unique_condition": "rare_condition", "rare_condition_retention": "rare_condition",
    "consolidation": "rare_condition",
}


def _action(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized not in _ACTION_ALIASES:
        raise ValueError(f"unknown growth action: {value!r}")
    return _ACTION_ALIASES[normalized]


def _growth_category(value: Any, action: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if not normalized:
        return {"merge_into_existing": "repeated_success", "noop": "repeated_success",
                "create": "new_mechanism", "branch": "version_contradiction",
                "supersede": "version_contradiction", "consolidate": "rare_condition",
                "archive": "rare_condition"}[action]
    return _GROWTH_CATEGORIES.get(normalized, normalized)


def _body_delta(action: str, event: Mapping[str, Any]) -> int:
    if "body_delta" in event:
        return _int(event["body_delta"])
    body = _int(event.get("body_tokens", event.get("body_size")))
    reclaimed = event.get("reclaimed_body_tokens", event.get("reclaimed_tokens"))
    return -abs(_int(reclaimed, body)) if action in {"consolidate", "archive"} else body


def _ratios(delta: int, milestones: int, body: int, reads: int,
            revisions: int, creations: int, recall: float) -> dict[str, Any]:
    return {
        "g_units": delta / milestones if milestones else None,
        "g_body": body / milestones if milestones else None,
        "g_read": reads / milestones if milestones else None,
        "revision_to_creation_ratio": revisions / creations if creations else None,
        "rare_condition_recall_rate": recall,
    }


def run_growth_replay(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay repeated successes, new mechanisms, version conflicts and rare conditions."""
    events = list(events)
    active = creations = revisions = archived = consolidations = milestones = 0
    body = reads = 0
    known_rare: list[str] = []
    retained_rare: set[str] = set()
    trajectory: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, int]] = {}
    for number, event in enumerate(events, start=1):
        action = _action(event.get("action", event.get("event_type", event.get("type"))))
        category = _growth_category(event.get("category", event.get("kind")), action)
        verified = event.get("verified", True) is not False
        before = active
        milestones += int(verified)
        if action == "create":
            active += 1
            creations += 1
        elif action == "merge_into_existing":
            revisions += 1
        elif action == "branch":
            active, creations, revisions = active + 1, creations + 1, revisions + 1
        elif action == "supersede":
            revisions += 1
        elif action == "consolidate":
            active = max(0, active - _int(
                event.get("merged_units", event.get("merge_count")), 1))
            consolidations += 1
            revisions += 1
        elif action == "archive":
            active = max(0, active - 1)
            archived += 1
        delta = active - before
        event_body = _body_delta(action, event)
        body += event_body
        reads += _int(event.get("read_tokens", event.get("read_cost")), 0)
        rare = _seq(event.get("rare_condition")) + _seq(event.get("rare_conditions"))
        for condition in rare:
            if condition not in known_rare:
                known_rare.append(condition)
            retained_rare.add(condition)
        dropped = (_seq(event.get("dropped_rare_conditions")) + _seq(event.get("drops"))
                   + _seq(event.get("archived_rare_conditions")))
        if event.get("rare_condition_preserved") is False:
            dropped.extend(rare)
        retained_rare.difference_update(dropped)
        for condition in (_seq(event.get("retained_rare_conditions"))
                          + _seq(event.get("rare_conditions_retained"))):
            if condition not in known_rare:
                known_rare.append(condition)
            retained_rare.add(condition)
        recall = len(retained_rare) / len(known_rare) if known_rare else 1.0
        stats = category_stats.setdefault(category, {
            "events": 0, "milestones": 0, "delta_u_active": 0, "creations": 0, "revisions": 0})
        stats["events"] += 1
        stats["milestones"] += int(verified)
        stats["delta_u_active"] += delta
        stats["creations"] += int(action in {"create", "branch"})
        stats["revisions"] += int(action in {"merge_into_existing", "branch", "supersede", "consolidate"})
        trajectory.append({
            "step": number, "event_id": str(event.get("event_id", number)),
            "action": action, "category": category, "verified": verified,
            "active_units": active, "delta_u_active": delta,
            "creations": creations, "revisions": revisions,
            "body_tokens": body, "body_delta": event_body, "read_tokens": reads,
            "milestones": milestones,
            **_ratios(delta, milestones, body, reads, revisions, creations, recall),
        })
    recall = len(retained_rare) / len(known_rare) if known_rare else 1.0
    final = {
        "event_count": len(events), "milestones": milestones, "active_units": active,
        "delta_u_active": active, "creations": creations, "revisions": revisions,
        "archived": archived, "consolidations": consolidations,
        "body_tokens": body, "read_tokens": reads,
        **_ratios(active, milestones, body, reads, revisions, creations, recall),
    }
    for stats in category_stats.values():
        stats["g_units"] = stats["delta_u_active"] / stats["milestones"] if stats["milestones"] else None
    return {"schema_version": SCHEMA_VERSION, "trajectory": trajectory, "final": final,
            **final, "categories": category_stats, "per_category": category_stats,
            "rare_conditions": {"known": known_rare, "retained": sorted(retained_rare)}}


compare_configurations = compare_loading_configs
evaluate_growth_curve = run_growth_replay
