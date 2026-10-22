"""Mechanism replay evaluation and host runner.

[Role]
Evaluates structured decisions against replay criteria and executes a case
through a host runner or static payload while capturing usage/latency.
[Invariants]
- Prompt isolation is verified before evaluation; post-decision material never
  enters the learner view.
- Missing usage stays `unavailable`, never zero; failed and unknown criteria are
  counted separately.
[Timeouts & Locks]
- Host runner wall time is measured with `perf_counter`; no timeout is imposed
  by this module and no locks are held.
[Side-Effect Policy]
- In-memory evaluation only; reads a supplied repository and never mutates it.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Mapping

from mindie_bench.evaluation.mechanism_replay_cases import MechanismCase

MAX_MERGED_BODY_BYTES = 8 * 1024
SCHEMA = "mindie-bench-mechanism-replay/1"

def _m6_repository(decision, case, context):
    actual = context.get("m6_repository")
    if not actual:
        return None, "no real GradedRepository revision evidence was supplied"
    revision = decision.get("revision") or {}
    previous_revision = revision.get("previous_revision", 1)
    adds_valid = revision.get("adds_valid_branch") is True
    expected_sha = revision.get("body_sha256", actual.get("current_body_sha256"))
    checks = {
        "revision_advanced": isinstance(actual.get("current_revision"), int)
        and actual["current_revision"] == int(previous_revision) + 1,
        "old_ref_removed": actual.get("old_ref_status") == "removed_or_superseded",
        "old_conditions_retained": actual.get("old_conditions_retained") is True,
        "new_branch_recalled": actual.get("new_branch_recalled") is True,
        "old_conditions_recalled": actual.get("old_conditions_recalled") is True,
        "active_unit_count_unchanged": isinstance(actual.get("active_units"), int)
        and actual.get("active_units") == actual.get("before_active_units"),
        "body_within_budget": isinstance(actual.get("current_body_bytes"), int)
        and 0 < actual["current_body_bytes"] <= MAX_MERGED_BODY_BYTES,
        "body_shrank": isinstance(actual.get("current_body_bytes"), int)
        and isinstance(actual.get("before_body_bytes"), int)
        and (actual["current_body_bytes"] < actual["before_body_bytes"] or adds_valid),
        "body_sha_matches": bool(expected_sha) and actual.get("current_body_sha256") == expected_sha,
        "complete_body_hash": actual.get("body_complete_hash_matches") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return not failed, f"failed_repository_checks={failed}"




def collect_m6_repository_evidence(
    repository,
    *,
    unit_id: str,
    old_conditions: Mapping,
    new_query: str,
    old_query: str,
    previous_revision: int = 1,
    before_body: str | None = None,
    before_active_units: int | None = None,
) -> dict[str, Any]:
    """Read a real GradedRepository after U@1 -> U@2; never mutate it."""
    from mindie_knowledge.materials.references import ReadReferenceError

    current_ref = repository.ref(unit_id)
    view = repository.explain(current_ref)
    previous_ref = repository.ref(unit_id, previous_revision)
    try:
        repository.explain(previous_ref)
        old_ref_status = "readable"
    except ReadReferenceError:
        old_ref_status = "removed_or_superseded"
    current_body = view["content"]
    new_result = repository.query(new_query, conditions=dict(old_conditions), limit=5,
                                  token_budget=4096, include_unknown=True)
    old_result = repository.query(old_query, conditions=dict(old_conditions), limit=5,
                                  token_budget=4096, include_unknown=True)
    def ids(payload):
        return {
            str(row["id"])
            for bucket in ("results", "cards", "foundation_cards")
            for row in payload.get(bucket, ())
        }

    current_bytes = len(current_body.encode("utf-8"))
    before_bytes = len(before_body.encode("utf-8")) if before_body is not None else None
    before_sha = hashlib.sha256(before_body.encode("utf-8")).hexdigest() if before_body else None
    return {
        "unit_id": unit_id,
        "current_ref": current_ref,
        "current_revision": view.get("revision"),
        "previous_revision": previous_revision,
        "current_body_bytes": current_bytes,
        "current_body_sha256": hashlib.sha256(current_body.encode("utf-8")).hexdigest(),
        "before_body_bytes": before_bytes,
        "before_body_sha256": before_sha,
        "before_active_units": before_active_units,
        "active_units": len(repository._load().units),
        "old_ref_status": old_ref_status,
        "old_conditions_retained": view.get("frontmatter", {}).get("conditions")
        == dict(old_conditions),
        "new_branch_recalled": unit_id in ids(new_result),
        "old_conditions_recalled": unit_id in ids(old_result),
        "body_complete_hash_matches": before_sha is not None
        and before_sha == hashlib.sha256(before_body.encode("utf-8")).hexdigest(),
        "frontmatter_conditions": view.get("frontmatter", {}).get("conditions"),
    }


# --------------------------------------------------------------- public evaluator


def evaluate_mechanism_replay(
    case: str | MechanismCase,
    decision: Mapping,
    *,
    repository=None,
    revision_context: Mapping | None = None,
) -> dict[str, Any]:
    """Evaluate a structured decision without exposing post-decision material."""
    from mindie_bench.evaluation.mechanism_replay import (
        _CRITERIA,
        _evidence,
        _hypothesis_entry_grounded,
        prepare_learner_prompt_view,
        resolve_case,
        verify_prompt_isolation,
    )

    selected = resolve_case(case) if isinstance(case, str) else case
    prompt = prepare_learner_prompt_view(selected)
    leak_report = verify_prompt_isolation(prompt, selected)
    context: dict[str, Any] = {}
    if selected.case_id == "M6-01" and repository is not None:
        revision = decision.get("revision") or {}
        required = ("target_unit_id", "old_conditions", "new_branch_query", "old_query")
        missing = [key for key in required if key not in revision]
        if not missing:
            before = revision_context or {}
            context["m6_repository"] = collect_m6_repository_evidence(
                repository,
                unit_id=revision["target_unit_id"],
                old_conditions=before.get("before_conditions") or revision["old_conditions"],
                new_query=revision["new_branch_query"],
                old_query=revision["old_query"],
                previous_revision=int(revision.get("previous_revision", 1)),
                before_body=before.get("before_body"),
                before_active_units=before.get("before_active_units"),
            )
    criteria = []
    allowed = selected.private_judge.get("allowed_actions", ())
    action = decision.get("action")
    action_ok = action in allowed
    criteria.append({
        "id": "action_allowed",
        "passed": action_ok,
        "detail": f"action={action} allowed={list(allowed)}",
    })
    hypothesis = decision.get("hypothesis_id")
    hypothesis_ids = {item.hypothesis_id for item in selected.hypotheses}
    grounded_hypothesis = _hypothesis_entry_grounded(hypothesis, selected, decision)
    hypothesis_ok = grounded_hypothesis is not None
    criteria.append({
        "id": "hypothesis_grounded",
        "passed": hypothesis_ok,
        "detail": f"hypothesis={hypothesis} grounded_as={grounded_hypothesis} competing={sorted(hypothesis_ids)}",
    })
    evidence_ok, evidence_detail = _evidence(decision, selected)
    criteria.append({"id": "public_evidence", "passed": evidence_ok, "detail": evidence_detail})
    for criterion_id in selected.criteria:
        function = _CRITERIA[criterion_id]
        passed, detail = function(decision, selected, context)
        criteria.append({"id": criterion_id, "passed": passed, "detail": detail})
    if any(item["passed"] is False for item in criteria):
        state = "failed"
    elif any(item["passed"] is None for item in criteria):
        state = "unknown"
    else:
        state = "completed"
    return {
        "schema_version": SCHEMA,
        "case_id": selected.case_id,
        "state": state,
        "accepted": state == "completed",
        "criteria": criteria,
        "passed": sum(item["passed"] is True for item in criteria),
        "failed": sum(item["passed"] is False for item in criteria),
        "unknown": sum(item["passed"] is None for item in criteria),
        "total": len(criteria),
        "prompt_isolation": leak_report,
        "sources": list(selected.sources),
    }


def run_host_mechanism_replay(
    case: str | MechanismCase,
    *,
    runner: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    decision_payload: Mapping[str, Any] | None = None,
    repository=None,
    revision_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a mechanism replay case via a host runner or decision payload and capture metrics."""
    from mindie_bench.evaluation.mechanism_replay import (
        prepare_learner_prompt_view,
        resolve_case,
    )

    selected = resolve_case(case) if isinstance(case, str) else case
    prompt_view = prepare_learner_prompt_view(selected)
    started = time.perf_counter()
    if runner is not None:
        raw_out = dict(runner(prompt_view))
    elif decision_payload is not None:
        raw_out = dict(decision_payload)
    else:
        raise ValueError("either runner or decision_payload is required")
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    decision = raw_out.get("decision") if isinstance(raw_out.get("decision"), Mapping) else raw_out
    metrics = raw_out.get("execution_metrics") if isinstance(raw_out.get("execution_metrics"), Mapping) else {}
    report = evaluate_mechanism_replay(
        selected,
        decision,
        repository=repository,
        revision_context=revision_context,
    )
    raw_prompt_tokens = metrics.get("prompt_tokens")
    raw_completion_tokens = metrics.get("completion_tokens")
    has_native_usage = (
        runner is not None
        and raw_prompt_tokens is not None
        and raw_completion_tokens is not None
    )
    report["execution_metrics"] = {
        "execution_mode": "host_runner" if runner is not None else "static_payload",
        "usage_status": "native" if has_native_usage else "unavailable",
        "wall_clock_ms": float(metrics.get("wall_clock_ms", elapsed_ms)),
        "prompt_tokens": int(raw_prompt_tokens) if raw_prompt_tokens is not None else None,
        "completion_tokens": int(raw_completion_tokens) if raw_completion_tokens is not None else None,
        "tool_calls": list(metrics.get("tool_calls", ())),
        "commands_executed": list(metrics.get("commands_executed", ())),
    }
    return report
