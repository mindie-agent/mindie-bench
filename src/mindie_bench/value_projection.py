"""Public projection for legacy component-value receipts.

[Role] Keep the v1 measurement contract while removing commands, endpoints and
paths from republished evidence.
[Invariants] Numeric observations and control identities remain; private
artifacts, error messages and provider endpoints do not.
[Timeouts & Locks] Pure in-memory transform; no I/O or locks.
[Side-Effect Policy] Returns a dictionary only.
"""

from __future__ import annotations

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
    auxiliary = {
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
    return row.model_copy(
        update={
            "environment_fingerprint": fingerprint,
            "primary_metrics": {
                key: value for key, value in row.primary_metrics.items() if key in METRICS
            },
            "auxiliary_metrics": auxiliary,
            "failure_reason": "details_retained_in_private_evidence" if row.failure_reason else None,
            "errors": [{"stage": "trial", "message": "details retained in private evidence"}]
            if row.errors
            else [],
            "artifacts": {},
        }
    ).model_dump(mode="json")


