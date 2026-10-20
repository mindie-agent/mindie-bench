"""Separate verified task outcomes from execution and delivery budget observations.

[Role] Read native budget evidence and summarize it without changing task scores.
[Invariants] Reaching an agent limit does not invalidate its existing artifact.
An overrun is a cost observation; missing budget evidence stays unmeasured.
[Timeouts & Locks] Pure receipt transforms; no I/O or waits.
[Side-Effect Policy] Never rewrite historical receipts or promote unknown outcomes.
"""

from mindie_bench.models import TerminalStatus

BUDGET_STOPS = {"timeout", "max_turns", "max_npu_runs"}


def is_budget_overrun(error: dict) -> bool:
    return (error.get("stage"), error.get("type")) == ("budget", "BudgetOverrun")


def has_non_budget_errors(receipt) -> bool:
    """Budget observations are not incomplete execution; all other errors remain."""
    return any(not is_budget_overrun(error) for error in receipt.errors)


def receipt_budget(receipt) -> dict:
    auxiliary = receipt.auxiliary_metrics
    process = auxiliary.get("native_process", {})
    stop = process.get(
        "stop_reason", receipt.environment_fingerprint.get("inference_native_stop_reason")
    )
    exhausted = auxiliary.get("agent_budget_exhausted")
    if exhausted is None and (process or "inference_native_stop_reason" in receipt.environment_fingerprint):
        exhausted = stop in BUDGET_STOPS
    lifecycle = auxiliary.get("lifecycle_budget", {})
    return dict(
        agent_budget_exhausted=exhausted,
        delivery_budget_exceeded=lifecycle.get("overrun"),
        declared_delivery_sec=lifecycle.get("declared_total_sec"),
        measured_delivery_sec=lifecycle.get("measured_total_sec"),
        delivery_overrun_sec=lifecycle.get("overrun_sec"),
    )


def summarize_budget(receipts) -> dict:
    observed = [(row, receipt_budget(row)) for row in receipts]
    valid = [
        (row, budget) for row, budget in observed
        if row.terminal_status in {TerminalStatus.PASS, TerminalStatus.FAIL}
    ]
    delivery_known = [
        (row, budget) for row, budget in valid
        if budget["delivery_budget_exceeded"] is not None
    ]
    passed_within = sum(
        row.terminal_status == TerminalStatus.PASS and not budget["delivery_budget_exceeded"]
        for row, budget in delivery_known
    )
    return dict(
        agent_limit_observed_attempts=sum(b["agent_budget_exhausted"] is not None for _, b in observed),
        agent_budget_exhausted_attempts=sum(b["agent_budget_exhausted"] is True for _, b in observed),
        completed_pass_at_agent_limit=sum(
            row.terminal_status == TerminalStatus.PASS and b["agent_budget_exhausted"] is True
            for row, b in observed
        ),
        delivery_budget_observed_attempts=sum(b["delivery_budget_exceeded"] is not None for _, b in observed),
        delivery_budget_exceeded_attempts=sum(b["delivery_budget_exceeded"] is True for _, b in observed),
        completed_pass_over_delivery_budget=sum(
            row.terminal_status == TerminalStatus.PASS and b["delivery_budget_exceeded"] is True
            for row, b in observed
        ),
        delivery_budget_valid_attempts=len(delivery_known),
        completed_pass_within_delivery_budget=passed_within,
        pass_at_1_within_delivery_budget=(
            passed_within / len(valid) if valid and len(delivery_known) == len(valid) else None
        ),
        scope="agent_limit_and_declared_delivery_time; business_outcomes_preserved",
    )
