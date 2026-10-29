"""[Role] Summarize independently verified strategy continuations.
[Invariants] No executor means missing metrics; regression requires an actual
paired default-path verdict. Selection correctness never changes task truth.
[Timeouts & Locks] Pure bounded work over the caller's finite cell list.
[Side-Effect Policy] Adds paired regression observations to the supplied rows.
"""

from __future__ import annotations

from mindie_bench.evaluation.knowledge_tiered_support import _ratio


def _total(rows, key, digits=None):
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    total = sum(values)
    return round(total, digits) if digits is not None else total


def _reported(usage, key):
    value = usage.get(key)
    return None if value is None else int(value)


def judgment_rate(rows, key, *, inverted=False):
    observed = [
        row["judgment"][key] for row in rows if row["judgment"].get(key) is not None
    ]
    return _ratio(
        sum((not value) if inverted else value for value in observed),
        len(observed), missing=len(rows) - len(observed),
    )


def rate_delta(left, right):
    return None if left is None or right is None else round(left - right, 6)


def continuation_summary(rows):
    by_cell = {
        (row["cell"]["case_id"], row["cell"]["repetition"], row["cell"]["arm"]): row
        for row in rows
    }
    for row in rows:
        cell = row["cell"]
        control = by_cell.get((cell["case_id"], cell["repetition"], "A"))
        if cell["arm"] == "A" or control is None:
            continue
        before, after = control["continuation_passed"], row["continuation_passed"]
        if before is not None and after is not None:
            row["continuation_regressed"] = before and not after
    summaries = {}
    for arm in ("A", "B", "C"):
        arm_rows = [row for row in rows if row["cell"]["arm"] == arm]
        selected = [
            row for row in arm_rows
            if (row["selector_decision"] or {}).get("selected_strategy_ref", "default") != "default"
        ]
        observed = [row for row in selected if row["continuation_passed"] is not None]
        paired = [row for row in selected if row["continuation_regressed"] is not None]
        summaries[arm] = {
            "selected_and_completed_rate": _ratio(
                sum(row["continuation_completed"] is True and row["continuation_passed"] is True
                    for row in observed),
                len(observed), missing=len(selected) - len(observed),
            ),
            "regressed_rate": _ratio(
                sum(row["continuation_regressed"] is True for row in paired),
                len(paired), missing=len(selected) - len(paired),
            ),
        }
    return summaries
