"""Tiered L1/L2 applicability, L4 held-out generalization, and feedback learning.

[Role]
Evaluate graded knowledge against a real `mindie_knowledge.graded.GradedRepository`:
condition applicability and distractor adoption, frozen-candidate held-out L4
classification, run-receipt usage feedback, and sequential cross-session learning.

[Invariants]
- `GradedRepository`, `RoutingIndex`, `record_usage_event`, `summarize_usage_events`,
  `evaluate_exploration_candidate` and `Engine.distill_selected` remain the only
  owners of retrieval, usage accounting, distillation and candidate gates. This
  module neither reimplements condition matching nor maintains parallel state.
- A missing/unknown premise is never executable; `no_knowledge` and natural-trigger
  cases with zero query calls are never failures (Principles 6 and 7).
- Infra failure or invalid measurement maps to `evaluated/inconclusive`, never
  `refuted`. A secondary feedback/distillation failure never overwrites a primary
  completed/pass result, and never turns `is_error` true for that verified result.
- L4 performance ratios include only inputs where both frozen implementations
  pass and both timings are present; every other classification stays visible.

[Timeouts & Locks]
- Pure evaluation is in-memory or bounded local SQLite/file work; no sleeps.
- Caller budgets bound distillation. A deadline or `TimeoutError` is caught as an
  accounting diagnostic while the primary task verdict is preserved.

[Side-Effect Policy]
- Never mutates the selected source corpus. Distractor injection copies into a
  caller-owned temporary workspace. Writes are isolated SQLite stores and test
  workspaces; no remote publication, model call, or corpus rewrite.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindie_bench.evaluation import knowledge_tiered_support as _support
from mindie_bench.evaluation.knowledge_tiered_support import (
    BUCKETS,
    CONDITIONS,
    EVAL_OUTCOMES,
    SCHEMA_VERSION,
    Engine,
    RoutingIndex,
    freeze_l4_candidate,
    growth_metrics,
    match_conditions,
    record_usage_event,
    summarize_usage_events,
)


# ------------------------------------------------------- L1/L2 applicability
def evaluate_applicability_and_distractors(
    cases: Sequence[Mapping],
    *,
    corpus_root: str | Path,
    workspace: str | Path | None = None,
    conditions: Sequence[str] = CONDITIONS,
    distractors: Sequence[Mapping] = (),
    k: int = 5,
    nav_budget: int = 2048,
    body_budget: int = 4096,
) -> dict:
    """Run L1/L2 condition, retrieval, distractor and adoption checks."""
    source = Path(corpus_root).resolve(strict=True)
    selected = list(conditions)
    if any(item not in CONDITIONS for item in selected):
        raise ValueError(f"conditions must be drawn from {CONDITIONS}")
    if not cases:
        raise ValueError("at least one applicability case is required")
    root = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="mindie-tiered-"))
    root.mkdir(parents=True, exist_ok=True)
    source_before = _support._source_hash(source)
    source_index = RoutingIndex.from_root(source)
    base_corpus = _support._copy_corpus(source, root / "normal-corpus")
    distractor_corpus = _support._copy_corpus(source, root / "distractor-corpus")
    injected = _support._inject_distractors(distractor_corpus, distractors)
    stores = {
        "oracle": _support._repository(root / "oracle-store", base_corpus),
        "normal_retrieval": _support._repository(root / "normal-store", base_corpus),
        "with_distractors": _support._repository(root / "distractor-store", distractor_corpus),
    }
    report = {}
    try:
        for condition in selected:
            report[condition] = _evaluate_condition(
                condition,
                cases,
                source_index,
                stores.get(condition),
                injected,
                k=k,
                nav_budget=nav_budget,
                body_budget=body_budget,
            )
    finally:
        for store, _repository_object in stores.values():
            store.close()
    source_after = _support._source_hash(source)
    annotation_errors = []
    for item in injected:
        if not item["observed"]:
            continue
        actual = _support._status(
            match_conditions(item["unit_conditions"], item["observed"]).status
        )
        if actual != item["expected_status"]:
            annotation_errors.append(
                {
                    "unit_id": item["unit_id"],
                    "expected_status": item["expected_status"],
                    "actual_status": actual,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator": "applicability_and_distractors",
        "corpus_root": str(source),
        "corpus_sha256_before": source_before,
        "corpus_sha256_after": source_after,
        "source_corpus_mutated": source_before != source_after,
        "unit_count": len(source_index.units),
        "k": k,
        "distractors": injected,
        "annotation_errors": annotation_errors,
        "conditions": report,
    }


def _evaluate_condition(
    condition: str,
    cases: Sequence[Mapping],
    source_index: RoutingIndex,
    store_pair,
    injected: Sequence[Mapping],
    *,
    k: int,
    nav_budget: int,
    body_budget: int,
) -> dict:
    repository = store_pair[1] if store_pair else None
    rows = []
    query_calls = 0
    retrieved_total = expected_total = 0
    precision_returned = precision_relevant = 0
    applicable_labeled = applicable_correct = applicable_missing = 0
    adoption_total = adoption_complete = adoption_missing = 0
    wrong_cases = inapplicable_cases = 0
    for case in cases:
        if case.get("applies_to") and condition not in case["applies_to"]:
            continue
        conditions = dict(case.get("conditions") or case.get("observed") or {})
        expected = [
            _support._unit_id(ref)
            for ref in _support._seq(case.get("expected_refs") or case.get("relevant_refs"))
        ]
        probe = case.get("probe_ref") or case.get("probe") or (expected[0] if expected else None)
        adopted = [
            _support._unit_id(ref)
            for ref in _support._seq(case.get("adopted_refs") or case.get("adopted"))
        ]
        query = case.get("query") or case.get("question")
        retrieval_enabled = bool(query) and case.get("retrieval_expected", True)
        retrieved, loaded = set(), set()
        if condition == "oracle":
            view = _support._loaded_view(
                repository,
                case.get("oracle_refs") or expected,
                conditions,
                token_budget=body_budget,
            )
            retrieved, loaded = _support._view_ids(view), _support._view_ids(view)
        elif condition != "no_knowledge":
            if retrieval_enabled:
                query_calls += 1
                result = repository.query(
                    str(query),
                    conditions=conditions,
                    limit=k,
                    token_budget=nav_budget,
                    include_unknown=True,
                    navigation_only=True,
                )
                retrieved = {
                    str(item["id"])
                    for bucket in ("results", "cards", "foundation_cards")
                    for item in _support._seq(result.get(bucket))
                }
            if case.get("loaded_refs"):
                view = _support._loaded_view(
                    repository, case["loaded_refs"], conditions, token_budget=body_budget
                )
                loaded = _support._view_ids(view)
        actual = (
            _condition_status(condition, source_index, repository, probe, conditions)
            if probe else "not_found"
        )
        declared_status = case.get("expected_status", case.get("expected_applicability"))
        expected_status = _support._status(declared_status if declared_status is not None else actual)
        case_errors = []
        if probe and declared_status is not None:
            applicable_labeled += 1
            if actual == expected_status:
                applicable_correct += 1
            else:
                case_errors.append(f"applicability:{expected_status}->{actual}")
        else:
            applicable_missing += int(bool(probe))
        wrong = [
            ident for ident in adopted
            if _condition_status(condition, source_index, repository, ident, conditions) != "matched"
        ]
        if wrong:
            wrong_cases += 1
            case_errors.append("wrong_adoption:" + ",".join(sorted(wrong)))
        if case.get("must_not_adopt") and adopted:
            case_errors.append("inapplicable_adoption")
        if condition == "with_distractors" and injected:
            inapplicable_cases += 1
        if (
            expected
            and expected_status == "matched"
            and condition in {"oracle", "normal_retrieval", "with_distractors"}
        ):
            expected_total += len(expected)
            retrieved_total += len(set(expected) & retrieved)
            precision_returned += len(retrieved)
            precision_relevant += len(set(expected) & retrieved)
        for ident in adopted:
            evidence = case.get("loaded_refs") or []
            if condition == "oracle":
                evidence = evidence or case.get("oracle_refs") or []
            if not evidence or condition == "no_knowledge":
                adoption_missing += 1
                continue
            complete, reason = _support._premise_complete(
                source_index, ident, conditions, set(map(_support._unit_id, evidence))
            )
            adoption_total += 1
            adoption_complete += int(complete)
            if not complete:
                case_errors.append(f"premise:{ident}:{reason}")
        rows.append(
            {
                "case_id": case.get("case_id"),
                "level": case.get("level"),
                "query_calls": 1 if condition not in {"no_knowledge", "oracle"} and retrieval_enabled else 0,
                "retrieved_refs": sorted(retrieved),
                "loaded_refs": sorted(loaded),
                "adopted_refs": adopted,
                "applicability": actual,
                "expected_applicability": expected_status if declared_status is not None else None,
                "natural_trigger": _support._natural_trigger(case),
                "errors": case_errors,
                "passed": not case_errors,
            }
        )
    return {
        "condition": condition,
        "queries": query_calls,
        "cases": rows,
        "passed_cases": sum(row["passed"] for row in rows),
        "failed_cases": [row for row in rows if not row["passed"]],
        "natural_trigger_zero_query_is_not_failure": all(
            row["passed"] for row in rows if row["query_calls"] == 0 and row["natural_trigger"]
        ),
        "metrics": {
            "recall_at_k": _support._ratio(retrieved_total, expected_total)["value"],
            "precision_at_k": _support._ratio(precision_relevant, precision_returned)["value"],
            "applicability_accuracy": _support._ratio(applicable_correct, applicable_labeled)["value"],
            "wrong_adoption_rate": _support._ratio(wrong_cases, inapplicable_cases)["value"],
            "premise_completeness_rate": _support._ratio(adoption_complete, adoption_total)["value"],
        },
        "metric_counts": {
            "recall_at_k": _support._ratio(retrieved_total, expected_total),
            "precision_at_k": _support._ratio(precision_relevant, precision_returned),
            "applicability_accuracy": _support._ratio(
                applicable_correct, applicable_labeled, missing=applicable_missing
            ),
            "wrong_adoption_rate": _support._ratio(wrong_cases, inapplicable_cases),
            "premise_completeness_rate": _support._ratio(
                adoption_complete, adoption_total, missing=adoption_missing
            ),
        },
        "traceable": True,
    }


def _condition_status(condition, source_index, repository, ref, conditions) -> str:
    if condition == "with_distractors" and repository is not None:
        unit = repository._load().units.get(_support._unit_id(ref))
        if unit is not None:
            return _support._status(match_conditions(unit.metadata["conditions"], conditions).status)
    return _support._case_status(source_index, ref, conditions)


# ------------------------------------------------------- L4 frozen held-out
def evaluate_l4_heldout_generalization(
    *,
    candidate_id: str,
    baseline_results: Mapping[str, Mapping],
    candidate_results: Mapping[str, Mapping],
    conditions: Mapping | None = None,
    guard_workloads: Sequence[Any] = (),
    guard_tolerance: float = 0.0,
    out_of_scope: Sequence[str] | Mapping[str, Mapping] = (),
    min_speedup_ratio: float = 1.05,
    noise_margin: float = 0.01,
    ablation: Mapping | None = None,
    candidate_artifact: Any = None,
    run_id: str | None = None,
) -> dict:
    """Classify every frozen held-out input and reject system guard regressions."""
    if guard_tolerance < 0:
        raise ValueError("guard_tolerance must be nonnegative")
    freeze = freeze_l4_candidate(candidate_id, candidate_artifact)
    identifiers = sorted({*baseline_results, *candidate_results})
    freeze["heldout_input_ids"] = identifiers
    classifications = {}
    buckets = {name: [] for name in BUCKETS}
    for ident in identifiers:
        bucket, ratio, reason = _support._classify(
            baseline_results.get(ident), candidate_results.get(ident)
        )
        buckets[bucket].append(ident)
        classifications[ident] = {
            "input_id": ident,
            "bucket": bucket,
            "speedup_ratio": round(ratio, 6) if ratio is not None else None,
            "reason": reason,
            "candidate_result": candidate_results.get(ident),
        }
    both_pass = buckets["both_pass"]
    retention_denominator = len(both_pass) + len(buckets["opt_regression"])
    baseline_broken = retention_denominator == 0
    retention = len(both_pass) / retention_denominator if retention_denominator else None
    ratios = [
        classifications[ident]["speedup_ratio"]
        for ident in both_pass
        if classifications[ident]["speedup_ratio"] is not None
    ]
    speedup = round(math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios)), 6) if ratios else None
    comparable = [ident for ident in both_pass if classifications[ident]["speedup_ratio"] is not None]
    guards, guard_failures = _support._guard_results(
        guard_workloads, classifications, guard_tolerance
    )
    scope = _support._scope_results(out_of_scope, classifications)
    ablation_report = (
        _support._ablation_report(
            ablation, conditions or {}, candidate_id, run_id or candidate_id,
            min_speedup_ratio, noise_margin,
        )
        if ablation
        else None
    )
    verdict = "accepted"
    if baseline_broken:
        verdict = "baseline_broken"
    elif buckets["evaluation_error"]:
        verdict = "evaluation_incomplete"
    elif scope["violations"]:
        verdict = "rejected_out_of_scope_application"
    elif guard_failures:
        verdict = "rejected_guard_regression"
    elif ablation_report and not ablation_report["accepted"]:
        verdict = "rejected_ablation"
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator": "l4_heldout_generalization",
        "candidate_freeze": freeze,
        "conditions": dict(conditions or {}),
        "buckets": buckets,
        "bucket_rows": {
            name: [classifications[ident] for ident in ids] for name, ids in buckets.items()
        },
        "counts": {name: len(ids) for name, ids in buckets.items()},
        "classifications": classifications,
        "correctness_retention_rate": round(retention, 6) if retention is not None else None,
        "correctness_retention": _support._ratio(len(both_pass), retention_denominator),
        "baseline_broken": baseline_broken,
        "speedup": {
            "value": speedup,
            "estimator": "geometric_mean_of_comparable_both_pass",
            "comparable_inputs": comparable,
            "excluded_inputs": sorted(set(both_pass) - set(comparable)),
            "missing_speedup": len(set(both_pass) - set(comparable)),
        },
        "guard_workloads": guards,
        "guard_failures": guard_failures,
        "l4_01_applicability": scope,
        "ablation": ablation_report,
        "verdict": verdict,
        "accepted": verdict == "accepted",
    }


# ------------------------------------------------------- receipt -> feedback
@dataclass(frozen=True)
class ConsultedKnowledgeRef:
    ref: str
    discovered: bool = True
    loaded: bool = True
    applied: bool = False
    evaluation_outcome: str | None = None
    reason: str = ""
    step_id: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "ConsultedKnowledgeRef":
        if isinstance(value, cls):
            return value
        row = _support._mapping(value)
        ref = row.get("ref") or row.get("unit_ref") or row.get("feedback_ref")
        if not ref:
            unit_id, revision = row.get("unit_id"), row.get("revision")
            if not unit_id:
                raise ValueError("a consulted knowledge ref requires a unit ref or unit_id")
            ref = str(unit_id) + (f"@{revision}" if revision is not None else "")
        outcome = row.get("evaluation_outcome", row.get("outcome"))
        return cls(
            ref=str(ref),
            discovered=bool(row.get("discovered", True)),
            loaded=bool(row.get("loaded", True)),
            applied=bool(row.get("applied", False)),
            evaluation_outcome=str(outcome) if outcome else None,
            reason=str(row.get("reason") or ""),
            step_id=str(row.get("step_id") or ""),
        )


def _event(store, *, root_hash, ref, event_type, run_id, outcome=None,
           step_id="", reason="", evidence_source="agent_declared"):
    return record_usage_event(
        store,
        root_hash=root_hash,
        ref=ref,
        event_type=event_type,
        run_id=run_id,
        outcome=outcome,
        step_id=step_id,
        reason=reason,
        evidence_source=evidence_source,
    )


def bridge_run_receipt_to_feedback(
    receipt: Mapping, *, store, root_hash: str, timeout_sec: float | None = None
) -> dict:
    """Bridge real run facts into the four usage stages without shadow state."""
    row = receipt.model_dump(mode="json") if hasattr(receipt, "model_dump") else _support._mapping(receipt)
    primary = _support._primary_outcome(row)
    verified = _support._primary_verified(primary)
    run_id = str(row.get("run_id") or row.get("trial_id") or "") or (
        "run-" + _support._sha(row)[:16]
    )
    metrics = {
        **_support._mapping(row.get("primary_metrics")),
        **_support._mapping(row.get("metrics")),
        **_support._mapping(row.get("auxiliary_metrics")),
    }
    harness = str(
        row.get("harness")
        or _support._mapping(row.get("environment_fingerprint")).get("harness")
        or _support._mapping(row.get("environment_fingerprint")).get("harness_name")
        or ""
    )
    commit_sha = str(
        row.get("commit_sha")
        or _support._mapping(row.get("environment_fingerprint")).get("commit")
        or _support._mapping(row.get("environment_fingerprint")).get("base_commit")
        or ""
    )
    evidence_source = "run_manifest_verified" if (
        row.get("run_manifest_verified") is True
        or row.get("evidence") == "verified"
        or _support._mapping(row.get("verification")).get("verified") is True
    ) else "agent_declared"
    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": str(row.get("task_id") or row.get("instance_id") or ""),
        "harness": harness,
        "commit_sha": commit_sha,
        "primary": {
            "outcome": primary,
            "verified": verified,
            "pass": primary == "completed_pass",
        },
        "primary_outcome": primary,
        "completed": verified,
        "pass": primary == "completed_pass",
        "metrics": metrics,
        "events": [],
        "diagnostics": [],
        "accounting_error": None,
        "is_error": not verified,
    }
    try:
        if not hasattr(store, "graded") or not store.graded.available():
            raise ValueError("no graded corpus is installed")
        for value in _support._seq(
            row.get("consulted_knowledge") or row.get("consulted") or row.get("knowledge")
        ):
            consulted = ConsultedKnowledgeRef.from_value(value)
            if consulted.evaluation_outcome not in (None, *EVAL_OUTCOMES):
                raise ValueError(
                    f"evaluation_outcome must be one of {EVAL_OUTCOMES}"
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("feedback accounting exceeded its budget")
            ref = _support._pin(store.graded, consulted.ref)
            store.graded.explain(ref, run_id=run_id)
            for event_type, enabled in (
                ("discovered", consulted.discovered),
                ("loaded", consulted.loaded),
            ):
                if enabled:
                    result["events"].append(
                        _event(
                            store, root_hash=root_hash, ref=ref, event_type=event_type,
                            run_id=run_id, step_id=consulted.step_id,
                            reason=consulted.reason, evidence_source=evidence_source,
                        )
                    )
            if consulted.applied:
                result["events"].append(
                    _event(
                        store, root_hash=root_hash, ref=ref, event_type="applied",
                        run_id=run_id, step_id=consulted.step_id,
                        reason=consulted.reason, evidence_source=evidence_source,
                    )
                )
                result["events"].append(
                    _event(
                        store, root_hash=root_hash, ref=ref, event_type="evaluated",
                        run_id=run_id,
                        outcome=_support._evaluation_outcome(
                            primary, consulted.evaluation_outcome
                        ),
                        step_id=consulted.step_id, reason=consulted.reason,
                        evidence_source=evidence_source,
                    )
                )
    except Exception as error:
        result["accounting_error"] = {
            "stage": "feedback",
            "type": type(error).__name__,
            "message": str(error)[:512],
            "timeout": isinstance(error, TimeoutError),
        }
        result["diagnostics"].append(
            "Feedback accounting failed after the primary task verdict was recorded; "
            "the primary outcome remains authoritative."
        )
        result["is_error"] = not verified
        return result
    return result


def distill_selected(
    *,
    store,
    inputs,
    session_digest: str,
    task_state: str,
    proposal: Mapping,
    redactor_executable: str | None = None,
    timeout_sec: float | None = None,
) -> dict:
    """Call the production organizer; never synthesize a decision locally."""
    executable = redactor_executable or _support._find_redactor()
    engine = Engine(store, redactor_executable=executable)
    started = time.monotonic()
    value = engine.distill_selected(
        list(inputs),
        session_digest=session_digest,
        task_state=task_state,
        proposal=dict(proposal),
    )
    if timeout_sec is not None and time.monotonic() - started >= timeout_sec:
        raise TimeoutError("distillation exceeded its declared budget")
    return value


# ------------------------------------------------------- sequential learning
def run_sequential_learning_episodes(
    episodes: Sequence[Mapping],
    *,
    store,
    root_hash: str,
    rare_condition_probes: Sequence[Mapping] = (),
    redactor_executable: str | None = None,
    distill_timeout_sec: float = 120.0,
    distiller=None,
) -> dict:
    """Run A-E new-session episodes, idempotent feedback, growth and rare recall."""
    if not episodes:
        raise ValueError("at least one sequential episode is required")
    repository = store.graded
    if not repository.available():
        raise ValueError("seed the graded repository before sequential learning")
    distiller = distiller or distill_selected
    initial = _support._repository_units(repository)
    previous = initial
    milestones = 0
    seen_runs: set[str] = set()
    duplicate_runs = 0
    accounting_runs: set[str] = set()
    rows, errors, version_chain = [], [], []
    for episode in episodes:
        row = _support._mapping(episode)
        run_id = str(row.get("run_id") or "")
        if not run_id:
            raise ValueError("every sequential episode requires a run_id")
        receipt = dict(row.get("receipt") or row)
        receipt.setdefault("run_id", run_id)
        receipt.setdefault("primary_outcome", _support._primary_outcome(row))
        bridged = bridge_run_receipt_to_feedback(
            receipt, store=store, root_hash=root_hash, timeout_sec=distill_timeout_sec
        )
        if bridged["accounting_error"]:
            errors.append({"run_id": run_id, **bridged["accounting_error"]})
            accounting_runs.add(run_id)
        replayed = run_id in seen_runs
        episode_row = {
            "session": row.get("session"),
            "run_id": run_id,
            "primary": bridged["primary"],
            "feedback": bridged,
            "replayed": replayed,
            "distillation": None,
            "growth": None,
        }
        duplicate_runs += int(replayed)
        if replayed or not row.get("proposal") or not row.get("learn", True):
            seen_runs.add(run_id)
            rows.append(episode_row)
            continue
        primary = bridged["primary"]["outcome"]
        if primary not in {"completed_pass", "completed_fail"}:
            episode_row["distillation"] = {"status": "skipped_primary_incomplete"}
            seen_runs.add(run_id)
            rows.append(episode_row)
            continue
        proposal = _support._mapping(row["proposal"])
        inputs = list(row.get("evidence_inputs") or [])
        digest = str(
            row.get("session_digest")
            or _support._sha({"run_id": run_id, "task": row.get("task_id")})
        )
        task_state = str(
            row.get("task_state")
            or ("verified_milestone" if primary == "completed_pass" else "completed")
        )
        try:
            distillation = distiller(
                store=store,
                inputs=inputs,
                session_digest=digest,
                task_state=task_state,
                proposal=proposal,
                redactor_executable=redactor_executable,
                timeout_sec=distill_timeout_sec,
            )
        except Exception as error:
            episode_row["distillation"] = {
                "stage": "distillation",
                "status": "accounting_error",
                "type": type(error).__name__,
                "message": str(error)[:512],
                "timeout": isinstance(error, TimeoutError),
            }
            accounting_runs.add(run_id)
            errors.append(
                {
                    "run_id": run_id,
                    "stage": "distillation",
                    "type": type(error).__name__,
                    "message": str(error)[:512],
                }
            )
            episode_row["accounting_error"] = episode_row["distillation"]
            episode_row["diagnostics"] = [
                "Distillation accounting failed after the primary verdict; the primary outcome remains authoritative."
            ]
            episode_row["is_error"] = not bridged["primary"]["pass"]
            seen_runs.add(run_id)
            rows.append(episode_row)
            continue
        episode_row["distillation"] = distillation
        status = str(distillation.get("status"))
        if status in {
            "failed",
            "local_application_failed",
            "local_receipt_failed",
            "outcome_unknown",
        }:
            accounting_runs.add(run_id)
            errors.append(
                {
                    "run_id": run_id,
                    "stage": "distillation",
                    "type": status,
                    "message": str(distillation.get("reason") or status),
                }
            )
            episode_row["accounting_error"] = {
                "stage": "distillation", "type": status,
                "message": str(distillation.get("reason") or status), "timeout": False,
            }
            episode_row["diagnostics"] = [
                "Distillation application failed after the primary verdict; the primary outcome remains authoritative."
            ]
            episode_row["is_error"] = not bridged["primary"]["pass"]
        else:
            milestones += int(status == "applied")
        current = _support._repository_units(repository)
        old = {unit.id: unit for unit in previous}
        changed = [
            unit for unit in current
            if unit.id not in old or unit.content_sha256 != old[unit.id].content_sha256
        ]
        for unit in changed:
            version_chain.append(
                {
                    "run_id": run_id,
                    "unit_id": unit.id,
                    "revision": unit.revision,
                    "content_sha256": unit.content_sha256,
                    "source": "GradedRepository current index",
                }
            )
        episode_row["growth"] = growth_metrics(
            current, completed_milestones=int(status == "applied"), previous_units=previous
        )
        previous = current
        seen_runs.add(run_id)
        rows.append(episode_row)
    final = _support._repository_units(repository)
    growth = growth_metrics(final, completed_milestones=milestones, previous_units=initial)
    usage = {}
    for unit in final:
        try:
            summary = summarize_usage_events(
                store, unit.id, unit.revision, include_delivery=True
            )
        except Exception as error:
            errors.append(
                {
                    "run_id": None,
                    "stage": "usage_summary",
                    "type": type(error).__name__,
                    "message": str(error)[:512],
                }
            )
            continue
        if summary["discovered"] or summary["loaded"] or summary["applied"] or summary["evaluated"]:
            usage[unit.id] = summary
    try:
        recall = _support._rare_recall(repository, rare_condition_probes)
    except Exception as error:
        errors.append(
            {
                "run_id": None,
                "stage": "rare_condition_recall",
                "type": type(error).__name__,
                "message": str(error)[:512],
            }
        )
        recall = {"probes": [], "recall_rate": None, "recalled": 0, "total": 0}
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator": "sequential_cross_session_learning",
        "episodes": rows,
        "duplicate_runs": duplicate_runs,
        "accounting_errors": errors,
        "version_chain": version_chain,
        "growth": {"initial_units": len(initial), "final_units": len(final), **growth},
        "rare_condition_recall": recall,
        "usage_summaries": usage,
        "gate": {
            "episodes_completed": len(rows),
            "duplicate_events_do_not_add_evidence": duplicate_runs == sum(
                row["replayed"] for row in rows
            ),
            "primary_preserved_on_accounting_error": all(
                not row.get("is_error", False)
                for row in rows
                if row["run_id"] in accounting_runs and row["primary"]["pass"]
            ),
        },
    }
