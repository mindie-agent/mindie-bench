"""Matched comparison, functional-claim and historical-attempt helpers.

[Role] Compute per-cohort paired comparisons, predeclared functional claims and
separately retained historical attempts for product publication.
[Invariants] Speedup is descriptive only and never a gate. Valid FAIL is a
result; missing/infra/unknown stay visible. Historical attempts keep original
runtime/runner identities and costs and are not forced into current pins.
[Timeouts & Locks] Pure in-memory transforms; no I/O or locks.
[Side-Effect Policy] Returns dictionaries only.
"""

from __future__ import annotations

from mindie_bench.io import digest
from mindie_bench.models import TrialReceipt
from mindie_bench.runner.matrix import ProductGroup, pair_matched_observations
from mindie_bench.runner.matrix_contracts import MatchedPair
from mindie_bench.value_product_public import public_product_receipt

VALID = {"completed_pass", "completed_fail"}


def _pair_summary(pairs, profiles: dict, baseline, candidate) -> dict:
    observed = [pair for pair in pairs if pair.both_observed]
    valid = [pair for pair in observed if pair.both_valid]
    solved = [
        pair for pair in valid
        if pair.left.status.value == "pass" and pair.right.status.value == "pass"
    ]
    ratios = [
        (pair.left.total_wall_clock_sec, pair.right.total_wall_clock_sec)
        for pair in solved
        if pair.left.total_wall_clock_sec is not None
        and pair.right.total_wall_clock_sec is not None
    ]
    speedup = (
        sum(left for left, _ in ratios) / sum(right for _, right in ratios) if ratios else None
    )
    denominator = {
        "planned": len(pairs),
        "observed": len(observed),
        "missing": len(pairs) - len(observed),
        "valid": len(valid),
        "unsettled": len(observed) - len(valid),
        "mutually_solved": len(solved),
    }
    return {
        "baseline_group": baseline.value,
        "candidate_group": candidate.value,
        "denominator": denominator,
        "wall_clock_speedup_on_mutual_success": speedup,
        "pairs_with_recorded_delivery_times": len(ratios),
        "measurement_is_delivery_including_preparation_verifier_cleanup": True,
        "baseline_budget": profiles[baseline].budget.comparison_dict(),
        "candidate_budget": profiles[candidate].budget.comparison_dict(),
        "conditional_reuse_only": True,
    }


def _blocked_comparison(spec, reason: str) -> dict:
    required = bool(spec.get("required"))
    return {
        "id": spec.get("id"),
        "cohort": spec.get("cohort"),
        "baseline_group": spec.get("baseline"),
        "candidate_group": spec.get("candidate"),
        "required": required,
        "budget_scope": spec.get("budget_scope", "downstream"),
        "claim_ready": False,
        "measurement_status": "scoped_limitation" if not required else "blocked",
        "scope": spec.get("scope"),
        "reason": reason,
        "denominator": {
            "planned": 1,
            "observed": 0,
            "missing": 1,
            "valid": 0,
            "unsettled": 0,
            "mutually_solved": 0,
        },
        "wall_clock_speedup_on_mutual_success": None,
        "pairs_with_recorded_delivery_times": 0,
        "measurement_is_delivery_including_preparation_verifier_cleanup": True,
    }


def _comparison(plan, observations, profiles, spec, p3=None) -> dict:
    baseline = ProductGroup(spec["baseline"])
    candidate = ProductGroup(spec["candidate"])
    required = bool(spec.get("required"))
    budget_scope = spec.get("budget_scope", "downstream")
    if candidate == ProductGroup.P3_V003_FLYWHEEL:
        if p3 is None or p3.get("status") != "admitted":
            return _blocked_comparison(spec, "p3_chain_not_admitted")
        candidate_profile = p3["profile"]
        cell = p3["cell"]
        left = next(
            (row for row in observations if row.group == baseline
             and row.task_id == cell.task_id and row.repetition == cell.repetition),
            None,
        )
        right = next(
            (row for row in observations if row.group == candidate
             and row.task_id == cell.task_id and row.repetition == cell.repetition),
            None,
        )
        pairs = [MatchedPair(cell.task_id, cell.task_family, cell.hardware_tier,
                             cell.repetition, left, right)]
    else:
        candidate_profile = profiles[candidate]
        pairs = pair_matched_observations(plan, observations, baseline, candidate)
    summary = _pair_summary(pairs, {**profiles, candidate: candidate_profile}, baseline, candidate)
    baseline_profile = profiles[baseline]
    if budget_scope == "lifecycle":
        if (
            baseline_profile.budget.lifecycle_model_calls is None
            or candidate_profile.budget.lifecycle_model_calls is None
            or baseline_profile.budget.lifecycle_model_calls
            != candidate_profile.budget.lifecycle_model_calls
            or baseline_profile.budget.lifecycle_cost_usd
            != candidate_profile.budget.lifecycle_cost_usd
        ):
            raise ValueError("lifecycle comparison needs equal predeclared total allocations")
    complete = (
        summary["denominator"]["missing"] == 0
        and summary["denominator"]["unsettled"] == 0
    )
    status = "measured" if complete else (
        "missing_cells" if summary["denominator"]["missing"] else "unsettled_outcomes"
    )
    if not required and not complete:
        status = "scoped_limitation"
    return {
        "id": spec.get("id"),
        "cohort": spec.get("cohort"),
        "baseline_group": baseline.value,
        "candidate_group": candidate.value,
        "required": required,
        "budget_scope": budget_scope,
        "claim_ready": complete,
        "measurement_status": status,
        "scope": spec.get("scope"),
        **summary,
    }


def _historical_record(item: dict, selected_by_trial: dict, problems: list) -> dict:
    row = TrialReceipt.model_validate(item["receipt"])
    if not item.get("reason"):
        raise ValueError("historical attempts need a reason")
    context = row.environment_fingerprint.get("experiment_context")
    context = context if isinstance(context, dict) else {}
    record = {
        "reason": item["reason"],
        "replacement_trial_id": item.get("replacement_trial_id"),
        "source_receipt_sha256": item.get("source_receipt_sha256") or digest(item["receipt"]),
        "canonical_receipt_sha256": item.get("canonical_receipt_sha256") or digest(item["receipt"]),
        "benchmark_runner": {
            "version": row.environment_fingerprint.get("benchmark_version"),
            "source_sha256": row.environment_fingerprint.get("benchmark_source_sha256"),
            "revision": row.environment_fingerprint.get("component_revisions", {}).get("mindie-bench"),
        },
        "component_revisions": row.environment_fingerprint.get("component_revisions", {}),
        "receipt": public_product_receipt(row),
    }
    if item.get("replacement_trial_id"):
        if row.terminal_status.value in VALID:
            raise ValueError("valid task outcomes cannot be replaced by a favorable retry")
        replacement = selected_by_trial.get(item["replacement_trial_id"])
        if (
            replacement is None
            or replacement[0].instance_id != row.instance_id
            or replacement[0].repetition != row.repetition
        ):
            problems.append(f"retest_chain_unsettled:{row.trial_id}")
    return record


def _functional_claim_status(cohort: dict, selected: dict, cohort_id: str) -> dict:
    claim = cohort.get("functional_claim")
    if not cohort["definition"].get("required"):
        return {"required": False, "claim_ready": None, "passes": None}
    group = ProductGroup(claim["group"])
    passes = 0
    for key, record in selected.items():
        if key[0] != cohort_id:
            continue
        _, cell_key = key
        cell_group, task_id, repetition = cell_key
        if cell_group != group:
            continue
        if (group, task_id, repetition) in cohort["required_cells"]:
            passes += record[1].status.value == "pass"
    minimum = claim["minimum_passes"]
    return {
        "required": True,
        "group": group.value,
        "minimum_passes": minimum,
        "passes": passes,
        "claim_ready": passes >= minimum,
    }
