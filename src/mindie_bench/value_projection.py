"""Public projection for legacy component-value receipts.

[Role] Keep the v1 measurement contract while removing commands, endpoints and
paths from republished evidence.
[Invariants] Numeric observations and control identities remain; private
artifacts, error messages and provider endpoints do not.
[Timeouts & Locks] Pure in-memory transform; no I/O or locks.
[Side-Effect Policy] Returns a dictionary only.
"""

from __future__ import annotations

from mindie_bench.evaluation.budget import has_non_budget_errors, is_budget_overrun, receipt_budget
from mindie_bench.io import digest
from mindie_bench.models import TrialReceipt
from mindie_bench.scorer.comparator import CONTROL_KEYS

METRICS = {
    "reward",
    "resolved",
    "tokens",
    "total_turns",
    "total_wall_clock_sec",
    "delivery_wall_clock_sec",
    "estimated_cost_usd",
    "cost_complete",
    "npu_test_runs",
    "npu_execution_wait_sec",
    "knowledge_query_sec",
    "llm_and_exploration_sec",
    "knowledge_measurement_basis",
    "turns_to_first_localization",
    "localization_penalty_turns",
    "time_to_first_localization_sec",
    "time_to_first_target_edit_sec",
    "zero_human_intervention",
    "foreground_knowledge_queries_count",
    "foreground_knowledge_explains_count",
}


def public_receipt(row: TrialReceipt) -> dict:
    """Keep the measurement contract while removing commands, endpoints and paths."""
    source = row.environment_fingerprint
    fingerprint = {key: source[key] for key in CONTROL_KEYS if key in source}
    for key in ("provider_url", "remote_environment"):
        if key in fingerprint:
            fingerprint[key] = {"private_control_identity": digest(fingerprint[key])}
    for key in (
        "components", "component_revisions", "knowledge_mode", "skills_sha256", "corpus_view_sha256"
    ):
        if key in source:
            fingerprint[key] = source[key]
    capture = row.auxiliary_metrics.get("capture") or {}
    budget = receipt_budget(row)
    auxiliary = {
        "agent_budget_exhausted": budget["agent_budget_exhausted"],
        "lifecycle_budget": {
            "overrun": budget["delivery_budget_exceeded"],
            "declared_total_sec": budget["declared_delivery_sec"],
            "measured_total_sec": budget["measured_delivery_sec"],
            "overrun_sec": budget["delivery_overrun_sec"],
        },
        # Measured segments of delivery. They may overlap and do not cover cleanup,
        # so they qualify delivery_wall_clock_sec rather than partition it.
        "preflight_wall_clock_sec": row.auxiliary_metrics.get("preflight_wall_clock_sec"),
        "verifier_wall_clock_sec": row.auxiliary_metrics.get("verifier_wall_clock_sec"),
        "capture_wall_clock_sec": row.auxiliary_metrics.get("capture_wall_clock_sec"),
        "captured_corpus_sha256": row.auxiliary_metrics.get("captured_corpus_sha256"),
        "capture": {
            "status": capture.get("status"),
            "corpus_lint": {"status": capture.get("corpus_lint", {}).get("status")},
            "publication": {"revision": capture.get("publication", {}).get("revision")},
            "consumer": {"revision": capture.get("consumer", {}).get("revision")},
        },
        "knowledge_measurement": {
            key: value for key, value in row.auxiliary_metrics.get("knowledge_measurement", {}).items()
            if key in {"query_count", "explain_count", "successful_calls", "failed_calls",
                       "timing_complete", "basis"}
        },
    }
    errors = []
    if any(is_budget_overrun(error) for error in row.errors):
        errors.append(dict(stage="budget", type="BudgetOverrun", message="declared delivery budget exceeded"))
    if has_non_budget_errors(row):
        errors.append(dict(stage="trial", message="details retained in private evidence"))
    return row.model_copy(
        update={
            "environment_fingerprint": fingerprint,
            "primary_metrics": {
                key: value for key, value in row.primary_metrics.items() if key in METRICS
            },
            "auxiliary_metrics": auxiliary,
            "failure_reason": "details_retained_in_private_evidence" if row.failure_reason else None,
            "errors": errors,
            "artifacts": {},
        }
    ).model_dump(mode="json")
