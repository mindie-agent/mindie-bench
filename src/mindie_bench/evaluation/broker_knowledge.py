"""[Role] Measure foreground knowledge calls in the parent-owned MCP broker.

[Invariants] Native tool lists can miss shell-launched MCP clients. Completed
preflight reads are separated only by the runner's explicit preflight boundary;
incomplete calls and clock mismatches remain visible.
[Timeouts & Locks] Local bounded journal parsing; no waits or locks.
[Side-Effect Policy] Reads original request/response receipts without mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from mindie_bench.evaluation.strategy_adoption import MAX_REFS, unit_ref

SCHEMA = "mindie-bench-broker-knowledge/1"
BASIS = "broker_tool_start_end/1"
TOOLS = {
    "knowledge_query": "query",
    "query": "query",
    "knowledge_explain": "explain",
    "explain": "explain",
}


def _number(value):
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("broker time must be finite and nonnegative")
    return value


def _identity(event):
    server, session, request = (event.get(key) for key in ("server", "session_id", "request_id"))
    if not isinstance(server, str) or not server or not isinstance(session, str) or not session:
        raise ValueError("knowledge observation has no server/session identity")
    if type(request) not in {int, str}:
        raise ValueError("knowledge observation has no JSON-RPC request identity")
    return server, session, type(request).__name__, request


def _references(event, field):
    refs = event.get(field)
    if refs is None:
        return None
    if (
        not isinstance(refs, list) or len(refs) > MAX_REFS
        or any(not isinstance(value, str) or unit_ref(value) != value for value in refs)
    ):
        raise ValueError(f"invalid bounded broker {field}")
    return refs


def summarize_broker_knowledge(
    path: Path,
    *,
    process_origin_monotonic_sec: float | None = None,
    preflight_finished_monotonic_sec: float | None = None,
) -> dict:
    """Read one foreground broker journal; malformed settlement is never zero."""
    if process_origin_monotonic_sec is not None:
        _number(process_origin_monotonic_sec)
    if preflight_finished_monotonic_sec is not None:
        _number(preflight_finished_monotonic_sec)
    calls, active, last_finished, digest = [], {}, {}, hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("broker event must be an object")
                kind = event.get("kind")
                if kind not in {"knowledge_tool_start", "knowledge_tool_end"}:
                    continue
                key, observed = _identity(event), _number(event.get("monotonic_sec"))
                tool = event.get("tool")
                if tool not in TOOLS:
                    raise ValueError("unsupported broker knowledge tool")
                if kind == "knowledge_tool_start":
                    if key in active or observed < last_finished.get(key, 0):
                        raise ValueError("duplicate broker knowledge request identity")
                    call = dict(
                        server=event["server"],
                        session_id=event["session_id"],
                        request_id=event["request_id"],
                        tool=tool,
                        operation=TOOLS[tool],
                        start_line=line_number,
                        started_monotonic_sec=observed,
                        ended_monotonic_sec=None,
                        elapsed_sec=None,
                        is_error=None,
                        request_sha256=event.get("request_sha256"),
                        response_sha256=None,
                        requested_refs=_references(event, "requested_refs"),
                        requested_refs_complete=event.get("requested_refs_complete") is True,
                        loaded_refs=None,
                        refs_complete=False,
                    )
                    active[key] = call
                    calls.append(call)
                else:
                    call = active.pop(key, None)
                    if call is None:
                        raise ValueError("orphan or duplicate broker knowledge completion")
                    if call["tool"] != tool or observed < call["started_monotonic_sec"]:
                        raise ValueError("broker knowledge completion contradicts its request")
                    elapsed = _number(event.get("elapsed_sec"))
                    if type(event.get("is_error")) is not bool:
                        raise ValueError("broker knowledge completion has no explicit error flag")
                    call.update(
                        end_line=line_number,
                        ended_monotonic_sec=observed,
                        elapsed_sec=elapsed,
                        is_error=event["is_error"],
                        response_sha256=event.get("response_sha256"),
                        loaded_refs=_references(event, "loaded_refs"),
                        refs_complete=event.get("refs_complete") is True,
                    )
                    last_finished[key] = observed
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid broker knowledge event line {line_number}: {exc}") from exc
    preflight = [
        row
        for row in calls
        if preflight_finished_monotonic_sec is not None
        and row["ended_monotonic_sec"] is not None
        and row["ended_monotonic_sec"] <= preflight_finished_monotonic_sec
    ]
    rows = [row for row in calls if row not in preflight]
    completed = [row for row in rows if row["ended_monotonic_sec"] is not None]
    intervals = sorted((row["started_monotonic_sec"], row["ended_monotonic_sec"]) for row in completed)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    raw = sum(end - start for start, end in intervals)
    union = sum(end - start for start, end in merged)
    relative, alignment_reason = None, None
    if not rows:
        relative = []
    elif process_origin_monotonic_sec is None:
        alignment_reason = "broker_native_clock_alignment_missing"
    elif any(row["started_monotonic_sec"] < process_origin_monotonic_sec for row in rows):
        alignment_reason = "broker_call_precedes_native_process_origin"
    else:
        relative = [
            dict(
                started_sec=row["started_monotonic_sec"] - process_origin_monotonic_sec,
                ended_sec=(
                    row["ended_monotonic_sec"] - process_origin_monotonic_sec
                    if row["ended_monotonic_sec"] is not None
                    else None
                ),
            )
            for row in rows
        ]
    operations = Counter(row["operation"] for row in rows)
    return dict(
        schema=SCHEMA,
        basis=BASIS,
        state="observed_calls" if rows else "observed_zero_calls",
        journal_sha256=digest.hexdigest(),
        query_count=operations["query"],
        explain_count=operations["explain"],
        admitted_calls=len(rows),
        completed_calls=len(completed),
        successful_calls=sum(row["is_error"] is False for row in completed),
        failed_calls=sum(row["is_error"] is True for row in completed),
        incomplete_calls=len(rows) - len(completed),
        calls=rows,
        preflight=dict(
            finished_monotonic_sec=preflight_finished_monotonic_sec,
            completed_calls=len(preflight),
            calls=preflight,
            scope="completed reads before the runner's explicit MCP preflight boundary",
        ),
        timing=dict(
            calls=len(rows),
            observed_intervals=len(completed),
            missing_intervals=len(rows) - len(completed),
            complete=len(rows) == len(completed),
            coverage=len(completed) / len(rows) if rows else None,
            raw_cumulative_sec=raw,
            reported_elapsed_cumulative_sec=sum(row["elapsed_sec"] for row in completed),
            observed_union_sec=union,
            overlap_sec=raw - union,
            first_started_sec=min((start for start, _ in intervals), default=None),
            last_ended_sec=max((end for _, end in intervals), default=None),
            clock_domain="absolute_monotonic_sec",
        ),
        latency_by_operation={
            operation: dict(
                sample_count=len(durations),
                admitted_count=operations[operation],
                mean_sec=mean(durations) if durations else None,
                minimum_sec=min(durations, default=None),
                maximum_sec=max(durations, default=None),
                scope="empirical completed broker request latency including startup; not an SLA",
            )
            for operation in ("query", "explain")
            for durations in [[row["elapsed_sec"] for row in completed if row["operation"] == operation]]
        },
        process_origin_monotonic_sec=process_origin_monotonic_sec,
        relative_intervals=relative,
        alignment_reason=alignment_reason,
        scope="foreground broker tools; completed preflight reads, catalogs and G3 consumers excluded",
    )
