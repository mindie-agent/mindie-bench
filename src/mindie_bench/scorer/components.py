"""Summarize the controller's real component calls in the trial receipt."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean


def _latency(values):
    if not values:
        return {"observations": 0, "mean_sec": None, "p95_sec": None}
    ordered = sorted(values)
    return {
        "observations": len(values),
        "mean_sec": mean(values),
        "p95_sec": ordered[math.ceil(0.95 * len(values)) - 1],
    }


def summarize_component_runtime(path: Path) -> dict:
    pending = defaultdict(deque)
    operations = defaultdict(
        lambda: {"attempts": 0, "acknowledged": 0, "statuses": Counter(), "durations": []}
    )
    provenance, peaks, aicore, phases = {}, {}, {}, Counter()
    observer_times, diagnostic_times = [], []
    samples = rejections = diagnostic_records = diagnostic_failures = dropped_records = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        event = row["event"]
        timestamp = row["monotonic_sec"]
        if type(timestamp) not in {int, float} or not math.isfinite(timestamp):
            raise ValueError("component journal requires finite monotonic timestamps")
        if event == "component_provenance":
            provenance[row["component"]] = row["actual"]
        elif event == "coordinator_request":
            action = row["request"]["action"]
            pending[(row["trace_id"], action)].append(timestamp)
            operations[action]["attempts"] += 1
        elif event in {"coordinator_response", "coordinator_error"}:
            action = row["action"]
            queue = pending[(row["trace_id"], action)]
            if not queue:
                raise ValueError("coordinator outcome has no corresponding request")
            elapsed = timestamp - queue.popleft()
            if elapsed < 0:
                raise ValueError("component outcome precedes its request")
            status = row["response"]["status"] if event == "coordinator_response" else "error"
            operations[action]["acknowledged"] += event == "coordinator_response"
            operations[action]["statuses"][status] += 1
            operations[action]["durations"].append(elapsed)
        elif event == "npu_observation":
            samples += 1
            phases[row["phase"]] += 1
            for device, values in row["selected_devices"].items():
                peaks[device] = max(peaks.get(device, 0), values["used_mb"])
                if values.get("aicore_percent") is not None:
                    aicore[device] = max(aicore.get(device, 0), values["aicore_percent"])
            if "elapsed_sec" in row:
                observer_times.append(row["elapsed_sec"])
        elif event == "npu_observation_rejected":
            rejections += 1
        elif event == "diagnostics":
            result = row["result"]
            diagnostic_records += bool(result.get("record_ref"))
            diagnostic_failures += bool(result["logging_failed"])
            dropped_records += result["dropped_records"]
            if "elapsed_sec" in row:
                diagnostic_times.append(row["elapsed_sec"])
    return {
        "scope": "controller leases and live monitoring; foreground tools are measured separately",
        "provenance": provenance,
        "coordinator": {
            action: {
                "attempts": values["attempts"],
                "acknowledged": values["acknowledged"],
                "unacknowledged": values["attempts"] - values["acknowledged"],
                "statuses": dict(values["statuses"]),
                "latency": _latency(values["durations"]),
            }
            for action, values in operations.items()
        },
        "npu_top": {
            "accepted_samples": samples,
            "rejected_samples": rejections,
            "phases": dict(phases),
            "peak_observed_hbm_mb": peaks,
            "peak_observed_aicore_percent": aicore,
            "latency": _latency(observer_times),
        },
        "diagnostics": {
            "recorded_events": diagnostic_records,
            "logging_failures": diagnostic_failures,
            "dropped_records": dropped_records,
            "latency": _latency(diagnostic_times),
        },
    }


def summarize_five_components(receipts: list) -> dict:
    """Aggregate five-component attribution evidence across trial receipts."""
    remote_calls = [
        r.auxiliary_metrics.get("remote_tool_calls_count")
        for r in receipts
        if type(r.auxiliary_metrics.get("remote_tool_calls_count")) is int
    ]
    remote_non_business_errors = [
        r.auxiliary_metrics.get("remote_tool_non_business_errors")
        for r in receipts
        if type(r.auxiliary_metrics.get("remote_tool_non_business_errors")) is int
    ]
    empty_polls = [
        r.auxiliary_metrics.get("validation_tail", {}).get("total_empty_job_polls")
        for r in receipts
        if type(r.auxiliary_metrics.get("validation_tail", {}).get("total_empty_job_polls")) is int
    ]
    first_loc_turns = [
        r.primary_metrics.get("turns_to_first_localization")
        for r in receipts
        if type(r.primary_metrics.get("turns_to_first_localization")) in {int, float}
    ]
    first_edit_turns = [
        r.primary_metrics.get("turns_to_first_target_edit")
        for r in receipts
        if type(r.primary_metrics.get("turns_to_first_target_edit")) in {int, float}
    ]
    knowledge_queries = [
        r.primary_metrics.get(
            "foreground_knowledge_queries_count",
            r.auxiliary_metrics.get("knowledge_queries_count"),
        )
        for r in receipts
    ]
    knowledge_queries = [v for v in knowledge_queries if type(v) is int]
    knowledge_explains = [
        r.primary_metrics.get(
            "foreground_knowledge_explains_count",
            r.auxiliary_metrics.get("knowledge_explains_count"),
        )
        for r in receipts
    ]
    knowledge_explains = [v for v in knowledge_explains if type(v) is int]
    knowledge_secs = [
        r.primary_metrics.get("knowledge_query_sec")
        for r in receipts
        if type(r.primary_metrics.get("knowledge_query_sec")) in {int, float}
    ]
    captures_completed = sum(
        r.auxiliary_metrics.get("capture", {}).get("status") == "completed" for r in receipts
    )
    corpus_lint_passed = sum(
        r.auxiliary_metrics.get("capture", {}).get("corpus_lint", {}).get("status") == "passed"
        for r in receipts
    )
    coord_trials = 0
    coord_ack = 0
    coord_unack = 0
    npu_accepted = 0
    npu_rejected = 0
    diag_recorded = 0
    diag_failures = 0
    diag_dropped = 0
    for r in receipts:
        comp = r.auxiliary_metrics.get("component_runtime_metrics")
        if not isinstance(comp, dict):
            continue
        if isinstance(comp.get("coordinator"), dict):
            coord_trials += 1
            for action_stats in comp["coordinator"].values():
                if isinstance(action_stats, dict):
                    coord_ack += int(action_stats.get("acknowledged", 0))
                    coord_unack += int(action_stats.get("unacknowledged", 0))
        if isinstance(comp.get("npu_top"), dict):
            npu_accepted += int(comp["npu_top"].get("accepted_samples", 0))
            npu_rejected += int(comp["npu_top"].get("rejected_samples", 0))
        if isinstance(comp.get("diagnostics"), dict):
            diag_recorded += int(comp["diagnostics"].get("recorded_events", 0))
            diag_failures += int(comp["diagnostics"].get("logging_failures", 0))
            diag_dropped += int(comp["diagnostics"].get("dropped_records", 0))
    return {
        "remote-dev": {
            "remote_tool_calls_total": sum(remote_calls) if remote_calls else 0,
            "remote_non_business_errors_total": (
                sum(remote_non_business_errors) if remote_non_business_errors else 0
            ),
            "empty_job_polls_total": sum(empty_polls) if empty_polls else 0,
            "mean_first_localization_turn": mean(first_loc_turns) if first_loc_turns else None,
            "mean_first_target_edit_turn": mean(first_edit_turns) if first_edit_turns else None,
        },
        "knowledge": {
            "queries_total": sum(knowledge_queries) if knowledge_queries else 0,
            "explains_total": sum(knowledge_explains) if knowledge_explains else 0,
            "mean_knowledge_query_sec": mean(knowledge_secs) if knowledge_secs else None,
            "captures_completed": captures_completed,
            "corpus_lint_passed": corpus_lint_passed,
        },
        "coordinator": {
            "trials_observed": coord_trials,
            "acknowledged_operations": coord_ack,
            "unacknowledged_operations": coord_unack,
        },
        "npu-top": {
            "accepted_samples_total": npu_accepted,
            "rejected_samples_total": npu_rejected,
            "post_flight_clean_trials": sum(
                r.auxiliary_metrics.get("post_flight_npu_clean") is True for r in receipts
            ),
        },
        "diagnostics": {
            "recorded_events_total": diag_recorded,
            "logging_failures_total": diag_failures,
            "dropped_records_total": diag_dropped,
        },
    }
