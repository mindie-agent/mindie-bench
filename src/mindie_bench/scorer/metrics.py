"""Outcome, effort, and cost metrics with explicit missing-data semantics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean

from mindie_bench.models import HarnessConfig, TaskSpec, TerminalStatus, Trajectory, TrialReceipt

NPU_TEST = re.compile(
    r"(?<![\w-])(pytest\b|vllm\s+serve\b|python3?\s+-m\s+unittest\b|"
    r"python3?\s+.*test_|msprof\b|torchrun\b)"
)
REMOTE = {"remote_bash", "remote_job_start", "remote_exec", "remote_command"}
READ_EDIT = {
    "remote_read",
    "Read",
    "remote_edit",
    "Edit",
    "remote_grep",
    "Grep",
    "read_file",
    "edit_file",
}


def tool_name(name: str) -> str:
    return name.rsplit("__", 1)[-1].rsplit(".", 1)[-1]


def npu_test_runs(trajectory: Trajectory) -> int:
    count = 0
    for call in trajectory.calls:
        name = tool_name(call.name)
        command = call.arguments.get("command", call.arguments.get("cmd", ""))
        if not isinstance(command, str):
            continue
        remote = name in REMOTE or (
            name.lower() in {"bash", "exec_command", "shell"}
            and re.search(r"(?<!\w)ssh(?:\s|$)", command)
        )
        if remote and NPU_TEST.search(command):
            count += 1
    return count


def first_localization(trajectory: Trajectory, targets: list[str]) -> int | None:
    for call in sorted(trajectory.calls, key=lambda c: c.turn):
        if tool_name(call.name) not in READ_EDIT:
            continue
        paths = [call.arguments.get(k) for k in ("path", "file_path", "filename")]
        paths += call.arguments.get("paths", []) + call.arguments.get("matched_files", [])
        for path in paths:
            if isinstance(path, str) and any(path == t or path.endswith("/" + t) for t in targets):
                return call.turn
    return None


def trajectory_metrics(
    trajectory: Trajectory, task: TaskSpec, harness: HarnessConfig, wall_clock: float, *, tier: str
) -> tuple[dict, dict]:
    remote = [c for c in trajectory.calls if tool_name(c.name) in REMOTE]
    all_remote = [
        c
        for c in trajectory.calls
        if c.component == "remote-dev" or tool_name(c.name).startswith("remote_")
    ]
    knowledge = [
        c for c in trajectory.calls if tool_name(c.name) in {"knowledge_query", "knowledge_explain"}
    ]

    def duration(calls):
        if any(c.started_sec is None or c.ended_sec is None for c in calls):
            return None
        values = [c.ended_sec - c.started_sec for c in calls]
        if any(v < 0 for v in values):
            raise ValueError("tool end precedes start")
        return sum(values)

    remote_sec, knowledge_sec = duration(remote), duration(knowledge)
    remainder = None
    if remote_sec is not None and knowledge_sec is not None and remote_sec + knowledge_sec <= wall_clock:
        remainder = wall_clock - remote_sec - knowledge_sec
    rates = [
        harness.input_usd_per_million,
        harness.cached_usd_per_million,
        harness.output_usd_per_million,
    ]
    cost = None
    if all(rate is not None for rate in rates):
        cost = (
            sum(
                v * rate
                for v, rate in zip(
                    [
                        trajectory.tokens.input_uncached,
                        trajectory.tokens.input_cached,
                        trajectory.tokens.output,
                    ],
                    rates,
                    strict=True,
                )
            )
            / 1_000_000
        )
    if cost is not None and trajectory.tokens.cache_creation:
        if harness.cache_creation_usd_per_million is None:
            cost = None
        else:
            cost += (
                trajectory.tokens.cache_creation
                * (harness.cache_creation_usd_per_million - harness.input_usd_per_million)
                / 1_000_000
            )
    primary = dict(
        total_wall_clock_sec=wall_clock,
        npu_execution_wait_sec=remote_sec,
        knowledge_query_sec=knowledge_sec,
        llm_and_exploration_sec=remainder,
        total_turns=trajectory.total_turns,
        npu_test_runs=npu_test_runs(trajectory) if tier == "l3" else None,
        remote_test_invocations=npu_test_runs(trajectory),
        turns_to_first_localization=first_localization(trajectory, task.mindie_knowledge.target_files),
        localization_penalty_turns=task.budget.max_turns,
        zero_human_intervention=(
            trajectory.human_interventions == 0 if trajectory.human_interventions is not None else None
        ),
        tokens=trajectory.tokens.model_dump(),
        estimated_cost_usd=cost,
        cost_complete=trajectory.tokens.complete and cost is not None,
    )
    auxiliary = dict(
        knowledge_queries_count=sum(
            tool_name(c.name) in {"knowledge_query", "query"} for c in knowledge
        ),
        knowledge_explains_count=sum(
            tool_name(c.name) in {"knowledge_explain", "explain"} for c in knowledge
        ),
        remote_tool_calls_count=len(all_remote),
        remote_tool_errors=sum(c.is_error is True for c in all_remote),
        remote_tool_non_business_errors=None,
        tool_timing_complete=remote_sec is not None and knowledge_sec is not None,
        timing_overlaps_or_missing=remainder is None,
        hit_oracle_task_in_top3=None,
        hit_oracle_block=None,
    )
    return primary, auxiliary


def summarize(receipts: list[TrialReceipt], *, k: int = 3, fast_p: float = 1.15) -> dict:
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
    reliable = [rows for rows in by_task.values() if len(rows) >= k]
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
    costs = [r.primary_metrics.get("estimated_cost_usd") for r in receipts]
    cost_complete = bool(receipts) and all(r.primary_metrics.get("cost_complete") for r in receipts)
    total_cost = sum(c for c in costs if c is not None)
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
    return dict(
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
        mean_wall_clock_sec=mean(known_times) if known_times else None,
        mean_delivery_wall_clock_sec=mean(delivery_times) if delivery_times else None,
        mean_wall_clock_coverage=len(known_times),
        zero_intervention_rate=(
            sum(v is True for v in interventions) / len(valid)
            if valid and all(v is not None for v in interventions)
            else None
        ),
        observed_cost_usd_lower_bound=total_cost,
        cost_complete=cost_complete,
        cost_per_resolved_task=total_cost / passed if passed and cost_complete else None,
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
