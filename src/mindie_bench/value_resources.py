"""Resource accounting and public component observations for value releases.

[Role] Total explicit trial selections and project known component scenarios.
[Invariants] Historical failures retain measured cost; missing observations are
null, not zero. Delivery already includes capture and verifier time.
[Timeouts & Locks] Pure in-memory transforms; no waits, subprocesses or locks.
[Side-Effect Policy] Never copy raw reports, local paths or arbitrary fields.
"""

from __future__ import annotations

import math
from collections import Counter

from mindie_bench.models import TrialReceipt

TOKEN_FIELDS = (
    "input_uncached",
    "input_cached",
    "cache_creation",
    "output",
    "indexer_small_model_total",
)
COMPONENT_METRICS = {
    "coordinator-contention": {
        "successful_listeners",
        "duplicate_device_selections",
        "bind_conflicts",
        "elapsed_sec",
        "release_to_next_success_sec",
    },
    "npu-top-physical-owner-query": {
        "human_answer_accuracy",
        "human_completion_time_sec",
        "golden_physical_devices_correct",
        "golden_process_assignments_correct",
        "golden_process_assignments_expected",
        "live_physical_devices",
        "live_container_owner_correct",
        "live_ssh_operations",
    },
    "diagnostics-fault-delivery": {
        "failure_occurrences",
        "raw_tracebacks",
        "contains_synthetic_secret",
        "automatic_delivery",
        "queue_items",
        "http_posts",
        "delivered_reports",
        "reconciled_without_duplicate",
        "private_fixture_removed",
        "useful_fault_fields_retained",
        "withdrawal_prevented_requests",
        "production_retry_wait_sec",
        "failure_to_delivery_sec",
    },
}


def _observations(values):
    observed = [value for value in values if value is not None]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in observed
    ):
        raise ValueError("resource observations must be finite nonnegative numbers")
    total = sum(observed) if observed else None
    return total, len(observed)


def resource_totals(rows: list[TrialReceipt]) -> dict:
    """Count each selected or historical attempt once, regardless of outcome."""
    if len({row.trial_id for row in rows}) != len(rows):
        raise ValueError("a trial cannot contribute resource costs more than once")
    metrics = [row.primary_metrics for row in rows]
    tokens, token_counts = {}, {}
    for field in TOKEN_FIELDS:
        tokens[field], token_counts[field] = _observations(
            [metric.get("tokens", {}).get(field) for metric in metrics]
        )
    delivery, delivery_count = _observations(
        [metric.get("delivery_wall_clock_sec") for metric in metrics]
    )
    foreground, foreground_count = _observations(
        [metric.get("total_wall_clock_sec") for metric in metrics]
    )
    cost, priced_count = _observations(
        [metric.get("estimated_cost_usd") for metric in metrics]
    )
    return {
        "attempts": len(rows),
        "statuses": dict(Counter(row.terminal_status.value for row in rows)),
        "task_roles": dict(Counter(row.task_role for row in rows)),
        "observed_tokens_lower_bound": tokens,
        "token_observed_attempts": token_counts,
        "token_metering_complete": bool(rows)
        and all(count == len(rows) for count in token_counts.values())
        and all(metric.get("tokens", {}).get("complete") is True for metric in metrics),
        "observed_delivery_wall_clock_sec_lower_bound": delivery,
        "delivery_observed_attempts": delivery_count,
        "delivery_timing_complete": bool(rows) and delivery_count == len(rows),
        "observed_foreground_wall_clock_sec_lower_bound": foreground,
        "foreground_observed_attempts": foreground_count,
        "foreground_timing_complete": bool(rows) and foreground_count == len(rows),
        "observed_cost_usd_lower_bound": cost,
        "priced_attempts": priced_count,
        "cost_complete": bool(rows)
        and priced_count == len(rows)
        and all(metric.get("cost_complete") is True for metric in metrics),
    }


def project_component_case(case: dict) -> dict:
    """Expose scenario measurements and source identity without private artifacts."""
    allowed = COMPONENT_METRICS[case["id"]]
    measurements = []
    for sample in case["measurements"]:
        public = {key: sample[key] for key in ("repetition", "status")}
        for side in ("baseline", "candidate"):
            observations = {}
            for key, value in sample[side].items():
                if key not in allowed:
                    continue
                if value is not None and type(value) not in (int, float, bool):
                    raise ValueError(f"component observation must be numeric or boolean: {key}")
                if type(value) is float and not math.isfinite(value):
                    raise ValueError(f"component observation must be finite: {key}")
                observations[key] = value
            public[side] = observations
        measurements.append(public)
    return {
        **{
            key: case[key]
            for key in (
                "id",
                "component",
                "status",
                "scope",
                "baseline",
                "candidate",
                "repetitions",
                "limitations",
            )
        },
        "measurements": measurements,
        "source_revisions": {
            name: {key: source[key] for key in ("version", "revision", "editable")}
            for name, source in case["source_revisions"].items()
        },
    }
