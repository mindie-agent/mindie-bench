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
        "fault_log_redacted",
        "fault_triage_cascading_rank",
        "fault_triage_cascading_timeout",
        "fault_triage_deterministic",
        "fault_triage_domain",
        "fault_triage_root_cause_rank",
    },
    "profiler-rank-link-memory-attribution": {
        "raw_arithmetic_matches_report",
        "profile_steps_aligned",
        "slow_link_dominant_block",
        "slow_rank_dominant_block",
        "comparison_status",
        "comm_delta_ms",
        "attention_delta_ms",
        "known_sum_bytes",
        "residual_bytes",
        "residual_status",
        "memory_target_met",
        "msprof_reference_not_added",
        "db_mutation",
        "analyze_latency_ms",
        "compare_latency_ms",
        "memory_latency_ms",
    },
    "sim-a3-graph-and-boundary-screening": {
        "prompt_tokens",
        "output_tokens",
        "graph_ttft_ms",
        "graph_tpot_ms",
        "qwen_case_deterministic",
        "a3_graph_ttft_ms",
        "a3_graph_tpot_ms",
        "ttft_absolute_error_ms",
        "ttft_error_pct",
        "tpot_absolute_error_ms",
        "tpot_error_pct",
        "operator_plan_elapsed_ms",
        "subgraph_removable_gm_bytes",
        "subgraph_critical_path_saving_us",
        "subgraph_overflow_rejected",
        "qwen_prediction_latency_ms",
        "operator_plan_latency_ms",
        "subgraph_screen_latency_ms",
        "subgraph_overflow_latency_ms",
    },
}
COMPONENT_STRING_METRICS = {
    "fault_triage_cascading_rank",
    "fault_triage_domain",
    "fault_triage_root_cause_rank",
    "slow_link_dominant_block",
    "slow_rank_dominant_block",
    "comparison_status",
    "residual_status",
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


def setup_totals(reports: list[dict]) -> dict:
    """Keep measured authentication/debug probes outside task gain comparisons."""
    rows = [(report["plugin"], row) for report in reports for row in report["operations"]]
    if len({(plugin, row["id"]) for plugin, row in rows}) != len(rows):
        raise ValueError("a setup operation cannot contribute resource costs more than once")
    operations = [row for _, row in rows]
    elapsed, elapsed_count = _observations([row["elapsed_sec"] for row in operations])
    cost, cost_count = _observations([row["cost_usd"] for row in operations])
    tokens, counts = {}, {}
    for field in TOKEN_FIELDS:
        tokens[field], counts[field] = _observations([row["tokens"].get(field) for row in operations])
    return {
        "operations": len(operations),
        "observed_elapsed_sec_lower_bound": elapsed,
        "elapsed_observed_operations": elapsed_count,
        "elapsed_complete": bool(operations) and elapsed_count == len(operations),
        "observed_tokens_lower_bound": tokens,
        "token_metering_complete": bool(operations)
        and all(count == len(operations) for count in counts.values())
        and all(row["token_metering_complete"] for row in operations),
        "observed_cost_usd_lower_bound": cost,
        "cost_complete": bool(operations) and cost_count == len(operations),
    }


def project_setup_report(report: dict) -> dict:
    """Recompute totals from allowlisted operation measurements, not supplied totals."""
    if report["schema_version"] != "mindie-setup-costs/1":
        raise ValueError("unsupported setup cost report")
    public = {
        key: report[key] for key in ("schema_version", "plugin", "scope", "limitations")
    }
    public["operations"] = [
        {key: row[key] for key in (
            "id", "operation", "model", "reasoning_effort", "status", "elapsed_sec",
            "token_metering_complete", "cost_usd", "measurement_basis",
        )} | {"tokens": {key: row["tokens"].get(key) for key in TOKEN_FIELDS}}
        for row in report["operations"]
    ]
    public["totals"] = setup_totals([public])
    return public


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
                if key in COMPONENT_STRING_METRICS and type(value) is str:
                    if any(token in value for token in ("/", "\\", ":")):
                        raise ValueError(f"component string observation cannot contain paths or URLs: {key}")
                    observations[key] = value
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
