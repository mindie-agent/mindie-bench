"""[Role] Project strategy reads and application events from execution records.
[Invariants] Explain delivery proves loading only. A successful applied feedback
call records a declaration, never behavioral proof. Observed records override
executor claims; missing or failed records cannot establish application.
[Timeouts & Locks] Pure bounded reference projection; no I/O, locks or waits.
[Side-Effect Policy] Returns observations without changing trajectories or verdicts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from itertools import islice

from mindie_bench.models import Trajectory

MAX_REFS = 64
MAX_REF_LENGTH = 512
_UNIT_REF = re.compile(r"unit-[a-z0-9]+(?:-[a-z0-9]+)*(?:@[1-9][0-9]*)?\Z")
EXPLAIN_TOOLS = {"knowledge_explain", "explain"}


def tool_name(name):
    return name.rsplit("__", 1)[-1].rsplit(".", 1)[-1]


def unit_ref(value):
    if not isinstance(value, str) or len(value) > MAX_REF_LENGTH:
        return None
    if value.startswith("mindie://") and "/units/" in value:
        value = value.split("/units/", 1)[1]
    return value if _UNIT_REF.fullmatch(value) else None


def requested_explain_refs(arguments):
    """Only explicit graded references are candidates; query text is not a ref."""
    if not isinstance(arguments, Mapping):
        return [], False
    values = [arguments["ref"]] if arguments.get("ref") is not None else []
    many = arguments.get("refs", [])
    if isinstance(many, str):
        many = [many]
    if not isinstance(many, list):
        return [], False
    complete = len(values) + len(many) <= MAX_REFS
    values.extend(many[:MAX_REFS - len(values)])
    refs = []
    for value in values:
        ref = unit_ref(value)
        if ref is None:
            complete = False
        elif ref not in refs:
            refs.append(ref)
    return refs, complete


def delivered_explain_refs(payload):
    """Read C5 explain bodies from real MCP structured/text result envelopes."""
    if not isinstance(payload, Mapping):
        return None, False
    structured = payload.get("structuredContent", payload.get("structured_content"))
    if structured is None:
        content = payload.get("content")
        texts = [content] if isinstance(content, str) else [
            part.get("text") for part in (content[:MAX_REFS] if isinstance(content, list) else [])
            if isinstance(part, Mapping) and part.get("type") == "text"
        ]
        parsed = []
        for text in texts:
            if not isinstance(text, str) or len(text.encode()) > 262144:
                continue
            try:
                value = json.loads(text)
            except ValueError:
                continue
            if isinstance(value, Mapping):
                parsed.append(value)
        if len(parsed) == 1 and (not isinstance(content, list) or len(content) <= MAX_REFS):
            structured = parsed[0]
    if not isinstance(structured, Mapping):
        return None, False
    if not any(key in structured for key in ("units", "requires", "ref", "id")):
        return None, False
    rows = []
    complete = True
    for key in ("units", "requires"):
        values = structured.get(key, [])
        if not isinstance(values, list):
            complete = False
            continue
        complete = complete and len(values) <= MAX_REFS
        rows.extend(values[:MAX_REFS])
    if "ref" in structured or "id" in structured:
        rows.append(structured)
    refs = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("content"), str):
            complete = False
            continue
        if not row["content"].strip():
            complete = False
            continue
        ref = unit_ref(row.get("ref"))
        if ref is None:
            ref = unit_ref(row.get("id"))
            revision = row.get("revision")
            if ref is not None and "@" not in ref and type(revision) is int and revision > 0:
                ref = unit_ref(f"{ref}@{revision}")
        if ref is None or "@" not in ref:
            complete = False
        elif ref not in refs:
            refs.append(ref)
    return refs[:MAX_REFS], complete and len(refs) <= MAX_REFS


def _successful(call):
    result = call.result_observation
    return (
        call.is_error is False
        and (result is None or result.mcp_is_error is not True
             and result.native_status != "failed" and result.classification in {
            "success", "business_success", "unclassified",
        })
    )


def observe_strategy_adoption(*, candidate_ids, trajectory=None, broker_knowledge=None):
    """Consume native Trajectory and the production broker summary, not agent prose."""
    if trajectory is None and broker_knowledge is None:
        return None
    candidates = {ref.split("@", 1)[0] for ref in candidate_ids}
    requested, loaded, applied, application_events = set(), set(), set(), []
    load_complete, application_complete = True, False
    outcomes = {}
    sources = []

    def retain(target, refs):
        complete = True
        for value in islice(refs or [], MAX_REFS + 1):
            ref = unit_ref(value)
            if ref and ref.split("@", 1)[0] in candidates:
                if len(target) < MAX_REFS or ref in target:
                    target.add(ref)
                else:
                    complete = False
        return complete

    if trajectory is not None:
        native = trajectory if isinstance(trajectory, Trajectory) else Trajectory.model_validate(trajectory)
        sources.append("native_trajectory")
        outcomes["native_trajectory"] = {"failed_calls": 0, "unsettled_calls": 0}
        load_complete = native.native_completed
        application_complete = native.native_completed
        for call in native.calls:
            name = tool_name(call.name)
            if name not in EXPLAIN_TOOLS | {"knowledge_feedback", "feedback"}:
                continue
            if name in {"explain", "feedback"} and call.component != "knowledge":
                continue
            if name in EXPLAIN_TOOLS:
                refs, complete = requested_explain_refs(call.arguments)
                load_complete = retain(requested, refs) and complete and load_complete
            if not _successful(call):
                result = call.result_observation
                failed = call.is_error is True or bool(result and (
                    result.mcp_is_error is True or result.native_status == "failed"
                    or result.classification in {
                        "business_failure", "transport_error", "protocol_error", "component_error",
                    }
                ))
                outcomes["native_trajectory"]["failed_calls"] += failed
                outcomes["native_trajectory"]["unsettled_calls"] += not failed
                if name in EXPLAIN_TOOLS:
                    load_complete = False
                else:
                    application_complete = False
                continue
            if name in EXPLAIN_TOOLS:
                result = call.result_observation
                complete = retain(loaded, result.knowledge_loaded_refs if result else None)
                load_complete = load_complete and complete and bool(result and result.knowledge_refs_complete)
            elif call.arguments.get("event_type") == "applied":
                ref = unit_ref(call.arguments.get("ref"))
                if ref and ref.split("@", 1)[0] in candidates:
                    application_complete = retain(applied, [ref]) and application_complete
                    if len(application_events) < MAX_REFS:
                        requested_source = call.arguments.get("evidence_source", "agent_declared")
                        application_events.append({
                            "call_id": call.id[:MAX_REF_LENGTH], "ref": ref, "event_type": "applied",
                            "requested_evidence_source": (
                                requested_source[:MAX_REF_LENGTH]
                                if isinstance(requested_source, str) else None
                            ),
                        })
                    else:
                        application_complete = False
    if broker_knowledge is not None:
        from .broker_knowledge import BASIS, SCHEMA

        if (
            not isinstance(broker_knowledge, Mapping)
            or broker_knowledge.get("schema") != SCHEMA or broker_knowledge.get("basis") != BASIS
        ):
            raise ValueError("strategy observations require the production broker knowledge summary")
        sources.append("broker_explain")
        outcomes["broker_explain"] = {"failed_calls": 0, "unsettled_calls": 0}
        for call in broker_knowledge["calls"]:
            if call["operation"] != "explain":
                continue
            load_complete = retain(requested, call.get("requested_refs")) and load_complete
            load_complete = load_complete and call.get("requested_refs_complete") is True
            if call["is_error"] is False and call["ended_monotonic_sec"] is not None:
                complete = retain(loaded, call.get("loaded_refs"))
                load_complete = load_complete and complete and call.get("refs_complete") is True
            else:
                outcomes["broker_explain"]["failed_calls"] += call["is_error"] is True
                outcomes["broker_explain"]["unsettled_calls"] += call["ended_monotonic_sec"] is None
                load_complete = False
    return {
        "basis": sources, "requested_strategy_refs": sorted(requested),
        "loaded_strategy_refs": sorted(loaded), "loading_observation_complete": load_complete,
        "applied_strategy_refs": sorted(applied) if applied else None,
        "application_events": application_events,
        "application_events_complete": application_complete,
        "application_basis": "successful_applied_feedback_declaration" if applied else None,
        "call_outcomes_by_source": outcomes,
        "behavioral_application_observed": None,
    }


def judge_continuation(case, payload, *, execution=None):
    """Keep outcome ownership separate from observed loading/application declarations."""
    from .strategy_selection_cases import evaluate_strategy_continuation

    try:
        observed = observe_strategy_adoption(
            candidate_ids=case.candidate_ids,
            trajectory=execution.trajectory if execution else payload.get("trajectory"),
            broker_knowledge=execution.broker_knowledge if execution else payload.get("broker_knowledge"),
        )
    except (ValueError, TypeError, KeyError) as exc:
        observed = {"error": f"{type(exc).__name__}: {exc}"[:1024]}
    judged = evaluate_strategy_continuation(case, payload)
    if execution is not None or observed is not None:
        judged["task_succeeded"] = execution.passed if execution else payload.get("verifier_passed") is True
        for key in ("declared_adoption", "applied_strategy_ref", "applied_in_actions",
                    "switched_after_observation", "next_action", "selection_reasonable",
                    "false_adoption"):
            judged[key] = None
    if observed is not None:
        refs = observed.get("applied_strategy_refs")
        if refs:
            identities = {ref.split("@", 1)[0] for ref in refs}
            judged["declared_adoption"] = True
            judged["applied_strategy_ref"] = next(iter(identities)) if len(identities) == 1 else None
            forbidden = {ref.split("@", 1)[0] for ref in case.private_judge.get("forbidden_strategies", ())}
            acceptable = {ref.split("@", 1)[0] for ref in case.private_judge.get("acceptable_strategies", ())}
            false_adoption = bool(identities & forbidden) or (
                case.case_kind == "boundary_or_mismatch" and not identities <= acceptable
            )
            judged["false_adoption"] = (
                false_adoption if false_adoption or observed["application_events_complete"] else None
            )
    judged["adoption_observation"] = observed
    return judged
