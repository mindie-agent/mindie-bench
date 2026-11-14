"""RSI exploration mode sequence observations extracted from knowledge audits or trajectories.

[Role]
Derives `reuse -> diagnose -> explore` mode transitions and mounted L4 ledger counts
from authoritative knowledge server audit records or fallback tool-call trajectories.

[Invariants]
- Malformed or missing mode fields in server audit logs return explicit
  `server_audit_log_unreadable/1` observations rather than fabricating zero counts.

[Timeouts & Locks]
- Pure in-memory analysis; no subprocesses, file I/O, or locks.

[Side-Effect Policy]
- Pure functions returning JSON-serializable observation dicts.
"""

from __future__ import annotations

import json

from mindie_bench.models import ToolCall


def _tool_leaf(name: str) -> str:
    return name.rsplit("__", 1)[-1].rsplit(".", 1)[-1]


def _extract_shell_knowledge_query_args(command: str) -> list[dict]:
    if "knowledge_query" not in command:
        return []
    out = []
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        brace = command.find("{", idx)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(command, brace)
            if isinstance(obj, dict) and "query" in obj:
                out.append(obj)
                idx = end
                continue
        except ValueError:
            pass
        idx = brace + 1
    return out


def _rsi_exploration_observations(
    knowledge_calls: list[ToolCall],
    broker_knowledge: dict | None = None,
    knowledge_audit_records: list[dict] | None = None,
    knowledge_audit_error: str | None = None,
) -> dict:
    del broker_knowledge
    modes: list[str] = []
    explore_calls: list[dict] = []
    seen_reuse = False
    transitioned = False

    unmeasured = dict.fromkeys(
        (
            "reuse_calls_count",
            "diagnose_calls_count",
            "explore_calls_count",
            "mode_sequence",
            "transitioned_reuse_to_explore",
            "explore_calls",
        )
    )
    if knowledge_audit_error:
        return dict(basis="server_audit_log_unreadable/1", error=knowledge_audit_error, **unmeasured)

    if isinstance(knowledge_audit_records, list):
        invalid = [
            f"{idx}:{type(rec).__name__}"
            for idx, rec in enumerate(knowledge_audit_records)
            if not isinstance(rec, dict)
        ]
        if invalid:
            return dict(
                basis="server_audit_log_unreadable/1",
                error="knowledge audit records are not objects: " + ",".join(invalid),
                **unmeasured,
            )
        for idx, rec in enumerate(knowledge_audit_records):
            if rec.get("operation") not in (None, "query"):
                continue
            declared = []
            if "effective_mode" in rec:
                declared.append(rec.get("effective_mode"))
            if "mode" in rec:
                declared.append(rec.get("mode"))
            if not declared:
                return dict(
                    basis="server_audit_log_unreadable/1",
                    error=f"knowledge audit record {idx} has unusable mode None",
                    **unmeasured,
                )
            usable = []
            for value in declared:
                if not isinstance(value, str) or value not in {"reuse", "diagnose", "explore"}:
                    return dict(
                        basis="server_audit_log_unreadable/1",
                        error=f"knowledge audit record {idx} has unusable mode {value!r}",
                        **unmeasured,
                    )
                usable.append(value)
            eff_mode = usable[0]
            modes.append(eff_mode)
            if eff_mode == "reuse":
                seen_reuse = True
            elif eff_mode == "explore":
                if seen_reuse:
                    transitioned = True
                try:
                    ledger_ids = rec.get("ledger_run_ids")
                    if ledger_ids is None:
                        ledger_entries_count = int(rec.get("observed_failed_attempts", 0))
                    elif isinstance(ledger_ids, list):
                        ledger_entries_count = len(ledger_ids)
                    else:
                        raise TypeError("ledger_run_ids")
                    mounted = rec.get("mounted_l4_ids") or []
                    if not isinstance(mounted, list):
                        raise TypeError("mounted_l4_ids")
                except (TypeError, ValueError) as exc:
                    return dict(
                        basis="server_audit_log_unreadable/1",
                        error=f"knowledge audit record {idx}: {type(exc).__name__}: {exc}",
                        **unmeasured,
                    )
                explore_calls.append(
                    dict(
                        call_id=f"audit-query-{idx + 1}",
                        tool="knowledge_query",
                        turn=idx + 1,
                        requested_mode=rec.get("requested_mode"),
                        effective_mode=eff_mode,
                        auto_switched_from_reuse=bool(rec.get("auto_switched_from_reuse", False)),
                        has_bottleneck_summary=bool(rec.get("has_bottleneck_summary", False)),
                        ledger_entries_count=ledger_entries_count,
                        mounted_l4_ids=list(mounted),
                        native_subcontext=bool(rec.get("native_subcontext", False)),
                    )
                )
        return dict(
            basis="server_audit_log/1",
            reuse_calls_count=sum(m == "reuse" for m in modes),
            diagnose_calls_count=sum(m == "diagnose" for m in modes),
            explore_calls_count=sum(m == "explore" for m in modes),
            mode_sequence=modes,
            transitioned_reuse_to_explore=transitioned,
            explore_calls=explore_calls,
        )

    for call in knowledge_calls:
        if call.is_error is True:
            continue
        tname = _tool_leaf(call.name)
        if tname not in {"knowledge_query", "query"}:
            continue
        args = call.arguments if isinstance(call.arguments, dict) else {}
        req_mode = str(args.get("mode") or "reuse")
        eff_mode = req_mode
        out_text = getattr(call, "output", None)
        if isinstance(out_text, str) and out_text.strip().startswith("{"):
            try:
                resp = json.loads(out_text)
                if isinstance(resp, dict) and resp.get("mode") in {"reuse", "diagnose", "explore"}:
                    eff_mode = str(resp["mode"])
            except ValueError:
                pass
        modes.append(eff_mode)
        if eff_mode == "reuse":
            seen_reuse = True
        elif eff_mode == "explore":
            if seen_reuse:
                transitioned = True
            ledger = args.get("ledger_entries")
            bottleneck = args.get("bottleneck_summary")
            explore_calls.append(
                dict(
                    call_id=call.id,
                    tool=tname,
                    turn=call.turn,
                    requested_mode=req_mode,
                    effective_mode=eff_mode,
                    goal=args.get("goal") or None,
                    has_bottleneck_summary=isinstance(bottleneck, dict) and bool(bottleneck),
                    ledger_entries_count=len(ledger) if isinstance(ledger, list) else 0,
                    native_subcontext=bool(args.get("native_subcontext", False)),
                )
            )
    return dict(
        basis="tool_trajectory/1",
        reuse_calls_count=sum(m == "reuse" for m in modes),
        diagnose_calls_count=sum(m == "diagnose" for m in modes),
        explore_calls_count=sum(m == "explore" for m in modes),
        mode_sequence=modes,
        transitioned_reuse_to_explore=transitioned,
        explore_calls=explore_calls,
    )
