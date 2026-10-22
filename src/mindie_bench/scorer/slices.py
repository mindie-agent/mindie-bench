# [Role] Slices matrix receipts into task, group, and repetition cohorts.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Score observed routing/retrieval labels, never substitute fixtures for observations."""

import json
from pathlib import Path
from statistics import mean

from mindie_bench.scorer.metrics import retrieval_metrics, routing_metrics


def score_slices(path: Path, *, kind: str, k: int = 3) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("slice input is empty")
    identities = [row["sample_id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("slice IDs must be unique")
    if len({row["evidence"] for row in rows}) != 1:
        raise ValueError("do not mix synthetic labels with observed component results")
    if kind == "routing":
        for row in rows:
            if type(row["expected_trigger"]) is not bool or type(row["actual_trigger"]) is not bool:
                raise ValueError("routing labels must be explicit booleans")
        result = routing_metrics([(row["expected_trigger"], row["actual_trigger"]) for row in rows])
    elif kind == "retrieval":
        measured = [retrieval_metrics(row["ranked_ids"], set(row["relevant_ids"]), k=k) for row in rows]
        result = {key: mean(row[key] for row in measured) for key in measured[0]}
    else:
        raise ValueError("unknown slice kind")
    return dict(
        kind=kind,
        k=k,
        samples=len(rows),
        evidence=rows[0]["evidence"],
        role="secondary_diagnostic_only",
        metrics=result,
    )
