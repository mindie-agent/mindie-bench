"""Normalize explicit tool receipts without interpreting command output as a verdict."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean

from mindie_bench.models import ToolCall, ToolResultObservation

SCHEMA = "mindie-bench-tool-observation/1"
REMOTE_SCHEMA = "remote-dev.result.v1"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_PENDING = {"created", "starting", "queued", "prepared", "running", "stopping", "cancelling"}
_TERMINAL = {"succeeded", "failed", "timeout", "cancelled", "completed"}
_UNKNOWN = {"unknown", "uncertain", "lost", "lost_outcome", "absent"}
_EXECUTION = {"remote_bash", "remote_job_start", "remote_exec", "remote_command"}
_DOMAINS = {
    "connection_unavailable": "transport",
    "rpc_disconnected": "transport",
    "rpc_send": "transport",
    "transport": "transport",
    "command_protocol": "protocol",
    "protocol": "protocol",
    "configuration": "component",
    "connection_capacity": "component",
    "process_start": "component",
    "worker_capacity": "component",
    "remote_worker": "component",
    "internal": "component",
    "internal_exception": "component",
    "validation": "component",
    "caller": "component",
    "permission": "component",
    "local_recording": "component",
    "cleanup": "component",
}


def _label(value, limit=96):
    return value if isinstance(value, str) and len(value) <= limit and _LABEL.fullmatch(value) else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def operation_kind(call: ToolCall) -> str:
    if call.name in _EXECUTION or call.name.lower() in {"bash", "exec_command", "shell"}:
        return "execution"
    if call.name == "remote_job_stdin":
        return "control" if call.arguments.get("chars") or call.arguments.get("eof") else "observation"
    if call.name in {"remote_job_status", "remote_job_tail"}:
        return "observation"
    if call.name in {"remote_job_stop", "remote_job_cancel"}:
        return "control"
    if call.component == "knowledge":
        return "knowledge"
    if call.name in {
        "Read",
        "Edit",
        "Grep",
        "remote_read",
        "remote_edit",
        "remote_grep",
        "remote_write",
        "remote_multi_edit",
        "remote_apply_patch",
    }:
        return "file"
    return "other"


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _number(value):
    return value if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None


def _structured(payload):
    """Accept explicit MCP envelopes or a versioned result, never prose heuristics."""
    if not isinstance(payload, dict):
        return None, False
    for key in ("structured_content", "structuredContent"):
        if payload.get(key) is not None:
            return _dict(payload[key]) or None, not isinstance(payload[key], dict)
    if payload.get("schema_version") == REMOTE_SCHEMA:
        return payload, False
    return None, False


def _observation(call, payload, *, source, native_status=None, error=None):
    observed = ToolResultObservation(
        source=source,
        classification="unclassified",
        operation_kind=operation_kind(call),
        native_status=_label(native_status),
        result_sha256=_sha({"result": payload, "error": error}),
        classification_reason="no_structured_outcome",
    )
    envelope = _dict(payload)
    bit = envelope.get("isError", envelope.get("is_error"))
    observed.mcp_is_error = bit if type(bit) is bool else None
    result, malformed = _structured(payload)
    if malformed:
        observed.classification = "protocol_error"
        observed.error_domain = "protocol"
        observed.classification_reason = "invalid_structured_result"
        return observed
    structured = result or {}
    observed.result_schema = _label(structured.get("schema_version"))
    details = _dict(structured.get("error_details"))
    rpc_error = _dict(error) or _dict(envelope.get("error"))
    details = details or _dict(rpc_error.get("data"))
    observed.error_category = _label(details.get("category"))
    observed.error_domain = _DOMAINS.get(observed.error_category)
    observed.submission_state = _label(details.get("submission_state"))
    completed = structured.get("operation_completed", details.get("operation_completed"))
    observed.operation_completed = completed if type(completed) is bool else None
    if structured.get("schema_version") == REMOTE_SCHEMA:
        job = _dict(structured.get("job"))
        remote_rows = [
            row
            for row in (job.get("remote_status"), structured.get("remote_status"))
            if isinstance(row, dict)
        ]
        observed.job_id = _label(structured.get("job_id") or job.get("job_id"), 256)
        observed.invocation_id = _label(structured.get("invocation_id"), 256)
        observed.target_id = _label(_dict(structured.get("target")).get("endpoint_id"), 256)
        observed.remote_outcome = _label(structured.get("outcome"))
        state = (
            structured.get("state")
            or next((row["state"] for row in remote_rows if row.get("state")), None)
            or structured.get("status")
        )
        observed.remote_status = _label(state)
        observed.component_duration_ms = _number(structured.get("duration_ms"))
        codes = [
            code
            for code in (
                structured.get("exit_code"),
                *(_dict(row.get("result")).get("exit_code") for row in remote_rows),
            )
            if type(code) is int
        ]
        observed.exit_code = codes[0] if codes else None
        # Certainty is independent of a previously observed command exit.
        if (
            observed.submission_state == "uncertain"
            or any(
                value in _UNKNOWN
                for value in (
                    observed.remote_outcome,
                    state,
                    structured.get("status"),
                    *(row.get("state") for row in remote_rows),
                )
                if isinstance(value, str)
            )
            or structured.get("unknown")
            or any(row.get("unknown") for row in remote_rows)
        ):
            observed.classification = "unknown"
            observed.classification_reason = "explicit_remote_uncertainty"
            return observed
        if len(set(codes)) > 1:
            observed.classification = "protocol_error"
            observed.error_domain = "protocol"
            observed.classification_reason = "conflicting_exit_codes"
            return observed
        if structured.get("local_recording") or structured.get("supervisor_error"):
            observed.error_domain = "component"
        if observed.error_domain:
            observed.classification = observed.error_domain + "_error"
            observed.classification_reason = "explicit_error_details"
        elif details:
            observed.classification_reason = "unrecognized_error_details"
        elif observed.remote_status in _TERMINAL and observed.exit_code is not None:
            observed.classification = "business_failure" if observed.exit_code else "business_success"
            observed.classification_reason = "explicit_remote_command_exit"
        elif observed.remote_status in _PENDING:
            observed.classification = "pending"
            observed.classification_reason = "explicit_remote_job_pending"
        elif (
            observed.remote_outcome in {"success", "cancelled"}
            and observed.operation_kind != "execution"
            and call.name != "remote_job_stdin"
        ):
            observed.classification = "success"
            observed.classification_reason = "explicit_component_success"
        return observed
    if observed.submission_state == "uncertain":
        observed.classification = "unknown"
        observed.classification_reason = "explicit_submission_uncertainty"
    elif observed.error_domain:
        observed.classification = observed.error_domain + "_error"
        observed.classification_reason = "explicit_error_details"
    elif type(rpc_error.get("code")) is int and rpc_error["code"] in {
        -32700,
        -32600,
        -32601,
        -32602,
        -32603,
    }:
        observed.classification = "protocol_error"
        observed.error_domain = "protocol"
        observed.classification_reason = "jsonrpc_protocol_error"
    elif (
        call.component != "remote-dev"
        and observed.mcp_is_error is False
        and native_status != "failed"
        and not error
        and not rpc_error
        and not details
    ):
        observed.classification = "success"
        observed.classification_reason = "explicit_mcp_success"
    elif (
        call.component != "remote-dev"
        and native_status == "completed"
        and observed.mcp_is_error is not True
        and not error
        and not rpc_error
        and not details
    ):
        observed.classification = "success"
        observed.classification_reason = "explicit_native_mcp_completion"
    return observed


def observe_codex_result(item: dict, call: ToolCall) -> ToolResultObservation:
    if item["type"] in {"command_execution", "file_change"}:
        code = item.get("exit_code")
        known_code = code if type(code) is int else None
        status = item.get("status")
        classification = "unclassified"
        if item["type"] == "command_execution" and known_code is not None:
            classification = "business_failure" if known_code else "business_success"
        elif item["type"] == "file_change" and status == "completed":
            classification = "success"
        return ToolResultObservation(
            source="native_command" if item["type"] == "command_execution" else "native_file",
            classification=classification,
            operation_kind=operation_kind(call),
            native_status=_label(status),
            exit_code=known_code,
            result_sha256=_sha(item),
            classification_reason="explicit_native_exit" if known_code is not None else "native_status",
        )
    return _observation(
        call,
        item.get("result"),
        source="codex_mcp",
        native_status=item.get("status"),
        error=item.get("error"),
    )


def observe_claude_result(block: dict, call: ToolCall, attached=None) -> ToolResultObservation:
    payload = attached if isinstance(attached, dict) else block
    if payload is block and not any(k in block for k in ("structuredContent", "structured_content")):
        content = block.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [
                part.get("text")
                for part in content or []
                if isinstance(part, dict) and part.get("type") == "text"
            ]
        )
        candidates = []
        for text in texts:
            if not isinstance(text, str):
                continue
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and (
                candidate.get("schema_version") == REMOTE_SCHEMA
                or any(k in candidate for k in ("structuredContent", "structured_content", "error"))
            ):
                candidates.append(candidate)
        if len(candidates) == 1:
            payload = candidates[0]
    observed = _observation(call, payload, source="claude_tool_result")
    if (
        call.component is not None
        or block.get("is_error") is True
        or observed.mcp_is_error is True
        or call.name not in {"Bash", "Read", "Edit", "Write", "Grep", "Glob", "LS", "NotebookEdit"}
        or observed.classification not in {"unclassified", "success"}
        or observed.result_schema is not None
        or any(key in _dict(payload) for key in ("structuredContent", "structured_content"))
        or _dict(payload).get("error")
    ):
        return observed
    native = _dict(payload)
    state = native.get("state", native.get("status"))
    if isinstance(state, str) and state in _UNKNOWN or native.get("interrupted") is True:
        observed.classification = "unknown"
        observed.classification_reason = "explicit_native_uncertainty"
    elif isinstance(state, str) and state in _PENDING | {"pending"}:
        observed.classification = "pending"
        observed.classification_reason = "explicit_native_pending"
    else:
        codes = [native[key] for key in ("exit_code", "exitCode") if key in native]
        if codes and (any(type(code) is not int for code in codes) or len(set(codes)) > 1):
            observed.classification = "protocol_error"
            observed.error_domain = "protocol"
            observed.classification_reason = "invalid_native_exit"
        elif call.name == "Bash" and codes:
            observed.exit_code = codes[0]
            observed.classification = "business_failure" if codes[0] else "business_success"
            observed.classification_reason = "explicit_native_exit"
        elif block.get("is_error") is False:
            # Attached native output may omit the outer tool_result error bit.
            observed.classification = "success"
            observed.classification_reason = "explicit_native_tool_success"
    return observed


def _distribution(values, total):
    ordered = sorted(values)
    return {
        "sample_count": len(values),
        "missing_count": total - len(values),
        "coverage": len(values) / total if total else None,
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1] if ordered else None,
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
    }


def summarize_tool_observations(calls: list[ToolCall]) -> dict:
    """Descriptive per-operation observations, never a service SLA or kernel benchmark."""
    observed = [call.result_observation for call in calls if call.result_observation is not None]
    attributable = [
        result
        for result in observed
        if result.error_domain or result.classification not in {"unknown", "unclassified"}
    ]
    known_non_business = sum(result.error_domain is not None for result in observed)
    grouped = defaultdict(list)
    jobs = defaultdict(list)
    for call in calls:
        wait = tuple(
            (key, call.arguments[key])
            for key in ("yield_time_ms", "timeout_ms")
            if key in call.arguments and _number(call.arguments[key]) is not None
        )
        grouped[(call.component, call.name, operation_kind(call), wait)].append(call)
        result = call.result_observation
        if result is not None and result.job_id is not None:
            jobs[(call.component, result.target_id, result.job_id)].append(result)
    job_counts = Counter()
    for results in jobs.values():
        exits = {
            result.exit_code
            for result in results
            if result.exit_code is not None
            and (
                result.classification in {"business_success", "business_failure"}
                or result.remote_status in _TERMINAL
                or result.operation_completed is True
            )
        }
        if (
            any(
                result.classification == "unknown"
                or result.classification_reason == "conflicting_exit_codes"
                for result in results
            )
            or len(exits) > 1
        ):
            job_counts["unknown"] += 1
        elif exits:
            job_counts["business_failure" if next(iter(exits)) else "business_success"] += 1
        elif any(result.remote_status in _TERMINAL for result in results):
            job_counts["unclassified"] += 1
        elif any(result.classification == "pending" for result in results):
            job_counts["pending"] += 1
        else:
            job_counts["unclassified"] += 1
    latency = []
    for (component, name, kind, wait), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = []
        component_values = []
        for call in rows:
            if call.started_sec is not None and call.ended_sec is not None:
                if _number(call.started_sec) is None or _number(call.ended_sec) is None:
                    raise ValueError("invalid tool timing")
                if call.ended_sec < call.started_sec:
                    raise ValueError("tool end precedes start")
                values.append(call.ended_sec - call.started_sec)
            if call.result_observation and call.result_observation.component_duration_ms is not None:
                component_values.append(call.result_observation.component_duration_ms)
        latency.append(
            {
                "component": component,
                "tool": name,
                "operation_kind": kind,
                "wait_parameters": dict(wait),
                "calls": len(rows),
                "wall_clock_sec": _distribution(values, len(rows)),
                "component_duration_ms": _distribution(component_values, len(rows)),
            }
        )
    return {
        "schema_version": SCHEMA,
        "calls": len(calls),
        "observation_count": len(observed),
        "observation_coverage": len(observed) / len(calls) if calls else None,
        "missing_observations": len(calls) - len(observed),
        "classifications": dict(Counter(result.classification for result in observed)),
        "raw_native_error_count": sum(call.is_error is True for call in calls),
        "raw_native_error_flag_coverage": (
            sum(call.is_error is not None for call in calls) / len(calls) if calls else None
        ),
        "known_non_business_error_count": known_non_business,
        "non_business_error_count": (
            known_non_business if calls and len(attributable) == len(calls) else None
        ),
        "non_business_attribution_coverage": (len(attributable) / len(calls) if calls else None),
        "execution_tool_call_count": sum(operation_kind(call) == "execution" for call in calls),
        "job_result_observation_count": sum(len(rows) for rows in jobs.values()),
        "distinct_jobs_observed": len(jobs),
        "job_terminal_counts": dict(job_counts),
        "latency_by_operation": latency,
        "latency_quantile_method": "nearest_rank",
        "latency_scope": (
            "Empirical tool-call wall time, grouped by tool, operation kind and explicit wait "
            "parameters; includes requested waiting and output delivery. Not a service SLA, "
            "controlled-load latency test or NPU kernel timing. Quantiles retain sample counts."
        ),
        "execution_scope": (
            "Execution tool calls are requests, not confirmed command completions or NPU "
            "tensor runs. Job terminal counts deduplicate repeated observations of the same "
            "component/target/job within this trajectory; absent result facts remain missing. "
            "Pending describes an observed reply, not current remote-process liveness. Nonzero "
            "business exits include requested cancellation and must be read with outcome/status."
        ),
    }
