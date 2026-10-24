"""Outcome, effort, and cost metrics with explicit missing-data semantics.

[Role]
Computes trial-level (`trajectory_metrics`) and matrix-level (`summarize`) outcome, effort,
cost, target-file access, and retrieval/routing metrics from trajectories and trial receipts.

[Invariants]
- Missing or incomplete observations (unmetered tokens, incomplete tool intervals, unaligned
  broker clocks, missing matrix cells) must produce `None` or explicit `complete=False`
  coverage states—never zero-filled fake numbers.
- `pass@1`, `pass^k`, and valid wall-clock means are computed strictly over valid terminal
  outcomes (`completed_pass` / `completed_fail`), while `all_attempt_*` metrics explicitly
  include invalid/infrastructure outcomes.
- Target file access and edit observations distinguish direct file RPCs, statically resolved
  shell/Python commands, and unresolved dynamic commands without executing target strings.

[Timeouts & Locks]
- Pure in-memory AST/shlex analysis and statistical aggregation; no subprocesses, network I/O,
  or locks (`_python_accesses` bounds AST input to 256 KiB and 64-element literal sequences).

[Side-Effect Policy]
- Pure side-effect-free functions returning serializable dicts; `canonical_receipt` enforces
  `allow_nan=False` strict JSON serialization.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from statistics import mean

from mindie_bench.evaluation.broker_knowledge import BASIS as BROKER_KNOWLEDGE_BASIS
from mindie_bench.evaluation.broker_knowledge import SCHEMA as BROKER_KNOWLEDGE_SCHEMA  # noqa: F401
from mindie_bench.evaluation.budget import summarize_budget
from mindie_bench.evaluation.source_visibility import (
    source_evidence_metrics as source_evidence_metrics,  # noqa: F401
)
from mindie_bench.models import HarnessConfig as HarnessConfig  # noqa: F401
from mindie_bench.models import TaskSpec as TaskSpec  # noqa: F401
from mindie_bench.models import (
    TerminalStatus,
    TrialReceipt,
)
from mindie_bench.models import ToolCall as ToolCall  # noqa: F401
from mindie_bench.models import Trajectory as Trajectory  # noqa: F401
from mindie_bench.scorer.command_observations import (
    is_recognized_npu_command as is_recognized_npu_command,  # noqa: F401
)
from mindie_bench.scorer.command_observations import (
    observe_npu_command as observe_npu_command,  # noqa: F401
)
from mindie_bench.scorer.inference import summarize_inference
from mindie_bench.scorer.target_access import (  # noqa: F401
    READ_EDIT,
    REMOTE,
    _call_outcome,
    _duration_profile,
    _edit_metrics,
    _job_argument_identity,
    _npu_test_call,
    _pending_job_outcomes,
    _validation_tail,
    first_localization,
    npu_test_runs,
    remote_execution,
    target_access_observations,
    tool_name,
    trajectory_metrics,
)
from mindie_bench.scorer.target_access_shell import (  # noqa: F401
    _python_accesses,
    _shell_accesses,
)
from mindie_bench.scorer.tool_observations import (
    summarize_tool_observations as summarize_tool_observations,  # noqa: F401
)


def _measurement_coverage(known: list[bool], *, reason: str) -> dict:
    total, observed = len(known), sum(known)
    state = (
        "empty"
        if not total
        else "complete"
        if observed == total
        else "not_measured"
        if not observed
        else "partial"
    )
    return dict(
        population_attempts=total,
        observed_attempts=observed,
        missing_attempts=total - observed,
        coverage=observed / total if total else None,
        state=state,
        reason="empty_input" if not total else None if observed == total else reason,
    )


def _known_count(value) -> bool:
    return type(value) is int and value >= 0


def _matrix_coverage(receipts, expected_cells, blocked_cells) -> dict:
    result = dict(
        planned_cells=None,
        observed_terminal_cells=len(receipts),
        valid_outcome_cells=sum(
            row.terminal_status in {TerminalStatus.PASS, TerminalStatus.FAIL} for row in receipts
        ),
        missing_terminal_cells=None,
        blocked_cells=None,
        unclassified_missing_cells=None,
        unexpected_terminal_cells=None,
        identity_mismatches=None,
        terminal_receipts_complete=None,
        all_planned_cells_accounted_for=None,
        complete=None,
        reason="plan_inventory_unavailable",
    )
    if expected_cells is None:
        if blocked_cells is not None:
            raise ValueError("blocked cells require an explicit expected cell inventory")
        return result

    def key(cell):
        return cell["instance_id"], cell["tier"], cell["group"], cell["repetition"]

    def indexed(cells):
        indexed_cells = {}
        for cell in cells:
            if (
                not isinstance(cell, dict)
                or any(
                    not isinstance(cell.get(field), str) or not cell[field]
                    for field in ("instance_id", "tier", "group", "task_digest")
                )
                or cell.get("role") not in {"eval", "seed"}
                or not _known_count(cell.get("repetition"))
                or cell["repetition"] < 1
            ):
                raise ValueError(
                    "expected/blocked cells require explicit task, tier, group, repetition, role and digest"
                )
            identity = key(cell)
            if identity in indexed_cells:
                raise ValueError("duplicate declared matrix cell")
            indexed_cells[identity] = {
                field: cell[field]
                for field in ("instance_id", "tier", "group", "repetition", "role", "task_digest", "evidence")
                if field in cell
            }
        return indexed_cells

    expected = indexed(expected_cells)
    blocked = indexed(blocked_cells) if blocked_cells is not None else None
    actual = {
        (row.instance_id, row.tier, row.ablation_group.value, row.repetition): row for row in receipts
    }
    if blocked is not None:
        if blocked.keys() - expected.keys():
            raise ValueError("blocked cells are outside the expected cell inventory")
        if blocked.keys() & actual.keys():
            raise ValueError("a blocked cell cannot also have a terminal receipt")
        if any(
            any(
                field in blocked[identity]
                and blocked[identity][field] != expected[identity].get(field)
                for field in ("role", "task_digest", "evidence")
            )
            for identity in blocked
        ):
            raise ValueError("blocked cell identity differs from the expected inventory")
    missing = expected.keys() - actual.keys()
    unexpected = actual.keys() - expected.keys()
    mismatches = [
        dict(cell=expected[identity], fields=fields)
        for identity in expected.keys() & actual.keys()
        if (
            fields := [
                field
                for field, value in (
                    ("role", actual[identity].task_role),
                    ("task_digest", actual[identity].task_digest),
                    ("evidence", actual[identity].evidence),
                )
                if field in expected[identity] and expected[identity][field] != value
            ]
        )
    ]
    duplicate_actual = len(actual) != len(receipts)
    terminal_complete = bool(expected) and not (missing or unexpected or mismatches or duplicate_actual)
    complete = terminal_complete and result["valid_outcome_cells"] == len(expected)
    result.update(
        planned_cells=len(expected),
        missing_terminal_cells=[expected[identity] for identity in sorted(missing)],
        blocked_cells=list(blocked.values()) if blocked is not None else None,
        unclassified_missing_cells=(
            [expected[identity] for identity in sorted(missing - blocked.keys())]
            if blocked is not None
            else None
        ),
        unexpected_terminal_cells=[
            dict(instance_id=identity[0], tier=identity[1], group=identity[2], repetition=identity[3])
            for identity in sorted(unexpected)
        ],
        identity_mismatches=mismatches,
        duplicate_terminal_cells=duplicate_actual,
        terminal_receipts_complete=terminal_complete,
        all_planned_cells_accounted_for=(
            not (missing - blocked.keys() or unexpected or mismatches or duplicate_actual)
            if blocked is not None
            else True
            if terminal_complete
            else None
        ),
        complete=complete,
        reason=(
            "empty_plan"
            if not expected
            else "cell_identity_mismatch"
            if mismatches or unexpected or duplicate_actual
            else "missing_terminal_cells"
            if missing
            else "invalid_terminal_outcomes"
            if not complete
            else None
        ),
    )
    return result


def summarize(
    receipts: list[TrialReceipt],
    *,
    k: int = 3,
    fast_p: float = 1.15,
    expected_cells: list[dict] | None = None,
    blocked_cells: list[dict] | None = None,
) -> dict:
    if k < 1 or not math.isfinite(fast_p) or fast_p <= 0:
        raise ValueError("k and performance threshold must be positive")
    statuses = Counter(r.terminal_status.value for r in receipts)
    valid = [r for r in receipts if r.terminal_status in {TerminalStatus.PASS, TerminalStatus.FAIL}]
    passed = sum(r.terminal_status == TerminalStatus.PASS for r in valid)
    by_task = defaultdict(list)
    for receipt in valid:
        by_task[receipt.instance_id].append(receipt)
    # Do not cube an overall mixture of easy/hard tasks. Report the requested
    # plug-in estimate per task, and only where k valid repetitions exist.
    reliable = [
        rows
        for rows in by_task.values()
        if len({r.repetition for r in rows}) == len(rows) and len(rows) >= k
    ]
    pass_pow_k = (
        mean(
            (sum(r.terminal_status == TerminalStatus.PASS for r in rows) / len(rows)) ** k
            for rows in reliable
        )
        if reliable
        else None
    )
    unbiased = (
        mean(
            math.comb(sum(r.terminal_status == TerminalStatus.PASS for r in rows), k)
            / math.comb(len(rows), k)
            for rows in reliable
        )
        if reliable
        else None
    )
    # Absent prices are not a zero-dollar observation; a partial sum stays a lower bound.
    costs = [
        value
        for r in receipts
        if (value := r.primary_metrics.get("estimated_cost_usd")) is not None
    ]
    cost_complete = bool(receipts) and all(r.primary_metrics.get("cost_complete") for r in receipts)
    total_cost = sum(costs) if costs else None
    known_times = [
        r.primary_metrics["total_wall_clock_sec"]
        for r in valid
        if r.primary_metrics.get("total_wall_clock_sec") is not None
    ]
    delivery_times = [
        r.primary_metrics["delivery_wall_clock_sec"]
        for r in valid
        if r.primary_metrics.get("delivery_wall_clock_sec") is not None
    ]
    interventions = [r.primary_metrics.get("zero_human_intervention") for r in valid]
    performance = [r.primary_metrics.get("performance_metrics", {}).get("speedup_ratio") for r in valid]
    speedup_known = all(v is not None for v in performance) and bool(valid)
    token_fields = ["input_uncached", "input_cached", "output", "indexer_small_model_total"]
    observed_tokens = {
        field: sum(r.primary_metrics.get("tokens", {}).get(field, 0) for r in receipts)
        for field in token_fields
    }
    delivery_all = [r.primary_metrics.get("delivery_wall_clock_sec") for r in receipts]
    wall_all = [r.primary_metrics.get("total_wall_clock_sec") for r in receipts]
    remote_errors = [r.auxiliary_metrics.get("remote_tool_non_business_errors") for r in receipts]
    query_counts = [
        r.auxiliary_metrics.get("knowledge_measurement", {}).get(
            "query_count",
            r.primary_metrics.get(
                "foreground_knowledge_queries_count", r.auxiliary_metrics.get("knowledge_queries_count")
            ),
        )
        for r in receipts
    ]
    explain_counts = [
        r.auxiliary_metrics.get("knowledge_measurement", {}).get(
            "explain_count",
            r.primary_metrics.get(
                "foreground_knowledge_explains_count",
                r.auxiliary_metrics.get("knowledge_explains_count"),
            ),
        )
        for r in receipts
    ]
    query_coverage = _measurement_coverage(
        [_known_count(value) for value in query_counts], reason="mcp_count_missing"
    )
    explain_coverage = _measurement_coverage(
        [_known_count(value) for value in explain_counts], reason="mcp_count_missing"
    )
    query_total = (
        sum(value for value in query_counts if _known_count(value))
        if query_coverage["observed_attempts"]
        else None
    )
    explain_total = (
        sum(value for value in explain_counts if _known_count(value))
        if explain_coverage["observed_attempts"]
        else None
    )
    mcp_complete = query_coverage["state"] == explain_coverage["state"] == "complete"
    knowledge_state = (
        "empty"
        if not receipts
        else "observed_zero_mcp_calls"
        if mcp_complete and query_total == explain_total == 0
        else "observed_mcp_calls"
        if mcp_complete
        else "not_measured"
        if query_total is None and explain_total is None
        else "partial"
    )
    knowledge = dict(
        state=knowledge_state,
        measurement_bases=sorted(
            {
                row.auxiliary_metrics.get("knowledge_measurement", {}).get(
                    "basis",
                    row.primary_metrics.get(
                        "knowledge_measurement_basis", "legacy_native_tool_trajectory_only"
                    ),
                )
                for row in receipts
            }
        ),
        query_count=query_total if query_coverage["state"] == "complete" else None,
        explain_count=explain_total if explain_coverage["state"] == "complete" else None,
        broker_count_coverage=_measurement_coverage(
            [
                row.primary_metrics.get("knowledge_measurement_basis") == BROKER_KNOWLEDGE_BASIS
                or row.auxiliary_metrics.get("knowledge_measurement", {}).get("basis")
                == BROKER_KNOWLEDGE_BASIS
                for row in receipts
            ],
            reason="shell_launched_mcp_clients_not_covered_by_native_tool_trajectory",
        ),
        observed_query_count_lower_bound=query_total,
        observed_explain_count_lower_bound=explain_total,
        attempts_with_observed_queries=sum(_known_count(value) and value > 0 for value in query_counts),
        attempts_with_observed_explains=sum(
            _known_count(value) and value > 0 for value in explain_counts
        ),
        query_count_coverage=query_coverage,
        explain_count_coverage=explain_coverage,
        native_body_access="not_measured_by_mcp_counters",
        benefit=dict(
            value=None,
            state="not_measured",
            reason=(
                "empty_input"
                if not receipts
                else "mcp_observation_incomplete"
                if not mcp_complete
                else "no_observed_mcp_retrieval"
                if query_total == explain_total == 0
                else "causal_protocol_not_supplied"
            ),
        ),
    )
    measurement_coverage = dict(
        wall_clock_valid=_measurement_coverage(
            [r.primary_metrics.get("total_wall_clock_sec") is not None for r in valid],
            reason="timing_missing",
        ),
        delivery_valid=_measurement_coverage(
            [r.primary_metrics.get("delivery_wall_clock_sec") is not None for r in valid],
            reason="timing_missing",
        ),
        wall_clock_all_attempts=_measurement_coverage(
            [value is not None for value in wall_all], reason="timing_missing"
        ),
        delivery_all_attempts=_measurement_coverage(
            [value is not None for value in delivery_all], reason="timing_missing"
        ),
        tokens=_measurement_coverage(
            [r.primary_metrics.get("tokens", {}).get("complete") is True for r in receipts],
            reason="usage_incomplete_or_missing",
        ),
        localization=_measurement_coverage(
            [r.primary_metrics.get("turns_to_first_localization") is not None for r in receipts],
            reason="source_access_proxy_unobserved",
        ),
        remote_non_business_errors=_measurement_coverage(
            [_known_count(value) for value in remote_errors], reason="classification_missing"
        ),
        knowledge_queries=query_coverage,
        knowledge_explains=explain_coverage,
    )
    measurement_coverage["tokens"]["recorded_attempts"] = sum(
        isinstance(r.primary_metrics.get("tokens"), dict) for r in receipts
    )
    progress_metrics = {}
    for field in (
        "time_to_first_localization_sec",
        "turns_to_first_target_edit",
        "time_to_first_target_edit_sec",
        "turns_to_last_target_edit",
        "time_to_last_target_edit_sec",
        "npu_runs_before_last_target_edit",
    ):
        values = [row.primary_metrics.get(field) for row in receipts]
        coverage = _measurement_coverage(
            [value is not None for value in values], reason="target_progress_observation_missing"
        )
        progress_metrics[field] = dict(
            **coverage,
            mean_recorded_value=mean(value for value in values if value is not None)
            if coverage["observed_attempts"]
            else None,
            population="all_attempts_with_recorded_proxy",
        )
    observed_tasks = sorted({r.instance_id for r in receipts})
    eligible_tasks = {rows[0].instance_id for rows in reliable}
    reliability_complete = bool(receipts) and (
        len(eligible_tasks) == len(observed_tasks) and len(valid) == len(receipts)
    )
    matrix = _matrix_coverage(receipts, expected_cells, blocked_cells)
    return dict(
        inference=summarize_inference(receipts, expected_cells=expected_cells),
        budget=summarize_budget(receipts),
        attempts=len(receipts),
        statuses={s.value: statuses[s.value] for s in TerminalStatus},
        valid_attempts=len(valid),
        completed_pass=passed,
        coverage=len(valid) / len(receipts) if receipts else None,
        pass_at_1=passed / len(valid) if valid else None,
        all_attempt_success_rate=passed / len(receipts) if receipts else None,
        pass_pow_k=pass_pow_k,
        pass_pow_k_unbiased=unbiased,
        reliability_k=k,
        reliability_eligible_tasks=len(reliable),
        reliability_total_tasks=len({r.instance_id for r in receipts}),
        reliability_estimator="mean_per_task_plugin_power; minimum k valid independent repeats",
        pass_pow_k_scope="eligible_observed_tasks_only",
        whole_matrix_pass_pow_k=(
            pass_pow_k if matrix["complete"] is True and reliability_complete else None
        ),
        reliability_coverage=dict(
            state=(
                "empty"
                if not receipts
                else "complete_for_observed_tasks"
                if reliability_complete
                else "partial"
                if reliable
                else "not_measured"
            ),
            required_valid_independent_repetitions=k,
            observed_tasks=len(observed_tasks),
            eligible_tasks=len(eligible_tasks),
            ineligible_tasks=[task for task in observed_tasks if task not in eligible_tasks],
            planned_tasks=(
                len({cell["instance_id"] for cell in expected_cells})
                if expected_cells is not None
                else None
            ),
            unobserved_planned_tasks=(
                sorted({cell["instance_id"] for cell in expected_cells} - set(observed_tasks))
                if expected_cells is not None
                else None
            ),
            per_task=[
                dict(
                    instance_id=task,
                    observed_attempts=sum(row.instance_id == task for row in receipts),
                    valid_independent_repetitions=len({row.repetition for row in by_task[task]}),
                    eligible=task in eligible_tasks,
                )
                for task in observed_tasks
            ],
            eligible_task_fraction=len(eligible_tasks) / len(observed_tasks) if observed_tasks else None,
            complete_for_observed_tasks=reliability_complete,
            complete_for_declared_cells=(
                matrix["complete"] and reliability_complete if expected_cells is not None else None
            ),
            reason=(
                "empty_input"
                if not receipts
                else "insufficient_valid_independent_repetitions"
                if not reliable
                else "partial_eligible_task_coverage"
                if not reliability_complete
                else "plan_inventory_unavailable"
                if expected_cells is None
                else matrix["reason"]
            ),
        ),
        matrix_coverage=matrix,
        mean_wall_clock_sec=mean(known_times) if known_times else None,
        mean_delivery_wall_clock_sec=mean(delivery_times) if delivery_times else None,
        mean_wall_clock_coverage=len(known_times),
        mean_delivery_wall_clock_count=len(delivery_times),
        valid_mean_population="completed_pass_or_completed_fail_with_recorded_timing",
        all_attempt_recorded_wall_clock_sec_mean=(
            mean(value for value in wall_all if value is not None)
            if any(value is not None for value in wall_all)
            else None
        ),
        all_attempt_recorded_wall_clock_sec_count=sum(value is not None for value in wall_all),
        all_attempt_recorded_delivery_wall_clock_sec_mean=(
            mean(value for value in delivery_all if value is not None)
            if any(value is not None for value in delivery_all)
            else None
        ),
        all_attempt_recorded_delivery_wall_clock_sec_count=sum(
            value is not None for value in delivery_all
        ),
        all_attempt_recorded_time_scope="recorded_elapsed_including_invalid_outcomes; not_successful_delivery",
        measurement_coverage=measurement_coverage,
        knowledge_observations=knowledge,
        target_progress_metrics=progress_metrics,
        observed_remote_non_business_errors_lower_bound=(
            sum(value for value in remote_errors if _known_count(value))
            if any(_known_count(value) for value in remote_errors)
            else None
        ),
        zero_intervention_rate=(
            sum(v is True for v in interventions) / len(valid)
            if valid and all(v is not None for v in interventions)
            else None
        ),
        observed_cost_usd_lower_bound=total_cost,
        cost_complete=cost_complete,
        cost_per_resolved_task=(
            total_cost / passed if passed and cost_complete and total_cost is not None else None
        ),
        observed_tokens_lower_bound=observed_tokens,
        token_metering_complete=bool(receipts)
        and all(r.primary_metrics.get("tokens", {}).get("complete") is True for r in receipts),
        observed_delivery_wall_clock_sec_lower_bound=sum(t for t in delivery_all if t is not None),
        delivery_timing_complete=bool(receipts) and all(t is not None for t in delivery_all),
        fast_p_threshold=fast_p,
        fast_p=(
            sum(
                r.terminal_status == TerminalStatus.PASS and speed > fast_p
                for r, speed in zip(valid, performance, strict=True)
            )
            / len(valid)
            if speedup_known
            else None
        ),
    )


def retrieval_metrics(ranked: list[str], relevant: set[str], *, k: int = 3) -> dict:
    if k < 1 or not relevant:
        raise ValueError("positive k and nonempty relevance labels required")
    if len(ranked) != len(set(ranked)):
        raise ValueError("duplicate retrieval IDs distort rank metrics")
    top = ranked[:k]
    hits = [index + 1 for index, item in enumerate(top) if item in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dict(
        recall_at_k=len(hits) / len(relevant),
        reciprocal_rank=1 / hits[0] if hits else 0.0,
        ndcg_at_k=dcg / ideal,
        hit_at_k=bool(hits),
    )


def routing_metrics(labels: list[tuple[bool, bool]]) -> dict:
    if not labels:
        raise ValueError("routing evaluation is empty")
    tp = sum(gold and predicted for gold, predicted in labels)
    fp = sum(not gold and predicted for gold, predicted in labels)
    fn = sum(gold and not predicted for gold, predicted in labels)
    negatives = sum(not gold for gold, _ in labels)
    return dict(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        zero_false_trigger_rate=1 - fp / negatives if negatives else None,
    )


def canonical_receipt(receipt: TrialReceipt) -> str:
    return json.dumps(receipt.model_dump(mode="json"), sort_keys=True, allow_nan=False)
