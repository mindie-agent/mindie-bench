"""L4-02 generalization, L4-04/C4 ablation, and end-to-end RSI gate campaign.

[Role]
Close the L4 RSI loop: classify post-freeze hidden-input generalization, verify
Leave-One-Out factor contributions under Contract C4, and orchestrate one
campaign from escalation/context through gated execution, ledger capture and
best-effort accounting callbacks.

[Invariants]
- Hidden-input speedup is computed only from `both_pass` cases; `both_fail`,
  `opt_regression`, `opt_improvement` and `evaluation_error` never pollute it.
- Any optimization regression, evaluation error, or an empty `both_pass` set
  makes generalization fail.
- A factor is `essential` when its removal breaks compile/correctness or
  removes a meaningful contribution. A `candidate_for_sequential_pruning`
  factor stays `RETAIN` unless an independently supplied pruned combination is
  verified with a non-empty run identity and correctness/performance evidence.
- Distillation/feedback accounting is non-destructive: callback failures are
  warnings and never overwrite the primary verdict or gate receipts.

[Timeouts & Locks]
- Evaluation and receipt assembly are pure in-memory work with no waits/locks.
- Callers own compile, NPU timing, qualification-gate and ledger timeouts. This
  module never launches a subprocess or device workload itself.

[Side-Effect Policy]
- Only the campaign's optional ledger/progress append writes files, delegated
  to `append_ledger_and_render_progress`. Callbacks are caller-owned effects.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mindie_bench.runner.rsi_campaign_support import (
    _GATE_FAILED,
    _GATE_OK,
    _PASS_WORDS,
    _candidate_rows,
    _compile_ok,
    _flag,
    _gate_state,
    _mapping,
    _number,
    _run_qualification_gates,
    _runner_callbacks,
    _source_speedup,
    build_isolated_jit_cache_env,
    build_subagent_fact_pack,
    classify_candidate_patch_shape,
    evaluate_speculative_acceptance,
    harvest_candidate_patch,
)

SCHEMA_GENERALIZATION = "mindie-bench-l4-generalization/1"
SCHEMA_ABLATION = "mindie-bench-ablation/1"
SCHEMA_CAMPAIGN = "mindie-bench-rsi-campaign/1"
CATEGORIES = ("both_pass", "opt_regression", "both_fail", "opt_improvement", "evaluation_error")


def _positive(value: Any, label: str) -> float:
    number = _number(value)
    if number is None or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _case_flag(case: Mapping[str, Any], prefix: str) -> bool:
    for key in (f"{prefix}_ok", f"{prefix}_passed", f"{prefix}_correct"):
        if key in case and case[key] is not None:
            return _flag(case[key])
    status = case.get(f"{prefix}_status")
    return str(status).strip().lower() in _PASS_WORDS if status is not None else False


def _parity_ok(case: Mapping[str, Any], rtol: float, atol: float) -> bool | None:
    if case.get("parity_ok") is not None:
        return _flag(case["parity_ok"])
    checks = []
    if case.get("max_abs_err") is not None:
        absolute = _number(case.get("max_abs_err"))
        checks.append(absolute is not None and absolute <= atol + 1e-12)
    if case.get("max_rel_err") is not None:
        relative = _number(case.get("max_rel_err"))
        checks.append(relative is not None and relative <= rtol + 1e-12)
    return all(checks) if checks else None


def _classify_case(case: Mapping[str, Any], rtol: float, atol: float):
    if case.get("evaluation_error") or str(case.get("status", "")).strip().lower() in {"error", "timeout"}:
        return "evaluation_error", False, False, None
    ref_ok, opt_ok = _case_flag(case, "ref"), _case_flag(case, "opt")
    parity = _parity_ok(case, rtol, atol)
    if parity is None:
        parity = ref_ok and opt_ok  # no parity evidence means no observed mismatch
    if ref_ok and (not opt_ok or not parity):
        return "opt_regression", ref_ok, opt_ok, parity
    if not ref_ok and opt_ok and parity:
        return "opt_improvement", ref_ok, opt_ok, parity
    if not ref_ok and not (opt_ok and parity):
        return "both_fail", ref_ok, opt_ok, parity
    return "both_pass", ref_ok, opt_ok, parity


def evaluate_hidden_input_generalization(
    cases: Sequence[Mapping[str, Any]], *, rtol: float = 1e-3, atol: float = 1e-3
) -> dict[str, Any]:
    """Classify the post-freeze 5-category matrix without hiding regressions."""
    if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("rtol and atol must be finite and non-negative")
    buckets: dict[str, list[str]] = {name: [] for name in CATEGORIES}
    classified: list[dict[str, Any]] = []
    ratios: list[float] = []
    for index, raw in enumerate(cases or ()):
        case = _mapping(raw)
        case_id = str(case.get("case_id") or case.get("input_id") or f"case-{index}")
        category, ref_ok, opt_ok, parity = _classify_case(case, rtol, atol)
        ref_latency, opt_latency = _number(case.get("ref_latency_us")), _number(case.get("opt_latency_us"))
        speedup = (
            ref_latency / opt_latency
            if ref_latency is not None and opt_latency is not None and ref_latency > 0 and opt_latency > 0
            else None
        )
        if category == "both_pass" and speedup is not None:
            ratios.append(speedup)
        buckets[category].append(case_id)
        classified.append({
            **case, "case_id": case_id, "bucket": case.get("bucket", case.get("bucket_name")),
            "category": category, "classification": category, "ref_ok": ref_ok,
            "opt_ok": opt_ok, "parity_ok": parity, "speedup_ratio": speedup,
        })
    both_pass_speedup = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios)) if ratios else None
    failed = [row["case_id"] for row in classified if row["category"] != "both_pass"]
    return {
        "schema_version": SCHEMA_GENERALIZATION,
        "generalization_ok": not buckets["opt_regression"] and not buckets["evaluation_error"] and bool(classified and buckets["both_pass"]),
        "total_cases": len(classified),
        "category_counts": {name: len(buckets[name]) for name in CATEGORIES},
        "both_pass_speedup": both_pass_speedup,
        "both_pass_speedup_estimator": "geometric_mean_of_both_pass",
        "failed_case_ids": failed,
        "cases": classified,
    }


def _factor_trial(raw: Mapping[str, Any], index: int, combined: float, ratio_floor: float):
    trial = _mapping(raw)
    name = str(trial.get("factor_name") or trial.get("factor_id") or f"factor-{index}")
    compile_ok = _flag(trial.get("compile_ok", trial.get("compile_passed")), default=True)
    correctness_ok = _flag(
        trial.get("correctness_ok", trial.get("oracle_passed", trial.get("correctness_passed"))), default=True
    )
    latency = _number(trial.get("latency_us", trial.get("duration_us")))
    contribution = latency - combined if latency is not None and latency > 0 else None
    if latency is None or latency <= 0:
        status, reason = "essential", "latency_evidence_missing"
    elif not compile_ok:
        status, reason = "essential", "compile_requires_factor"
    elif not correctness_ok:
        status, reason = "essential", "correctness_requires_factor"
    elif contribution is not None and contribution >= max(ratio_floor, 0.0) * combined - 1e-9:
        status, reason = "essential", "meaningful_contribution"
    else:
        status, reason = "candidate_for_sequential_pruning", "contribution_below_threshold"
    return {
        "factor_name": name, "factor_id": name, "run_id": str(trial.get("run_id") or ""),
        "compile_ok": compile_ok, "correctness_ok": correctness_ok, "latency_us": latency,
        "contribution_us": contribution,
        "contribution_ratio": contribution / combined if contribution is not None else None,
        "status": status, "reason": reason,
    }


def verify_sequential_ablation(
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_latency_us: float,
    combined_latency_us: float,
    factor_trials: Sequence[Mapping[str, Any]],
    pruned_combination_run: Mapping[str, Any] | None = None,
    min_factor_contribution_ratio: float = 0.01,
) -> dict[str, Any]:
    """Verify LOO contributions and enforce Contract C4 before pruning anything."""
    baseline_latency = _positive(baseline_latency_us, "baseline_latency_us")
    combined = _positive(combined_latency_us, "combined_latency_us")
    ratio_floor = _number(min_factor_contribution_ratio)
    if ratio_floor is None or ratio_floor < 0:
        raise ValueError("min_factor_contribution_ratio must be finite and non-negative")
    pruned = _mapping(pruned_combination_run)
    pruned_run_id = str(pruned.get("run_id") or "").strip()
    pruned_latency = _number(pruned.get("latency_us", pruned.get("combined_latency_us")))
    pruned_implied = (
        _flag(pruned.get("compile_ok")) and _flag(pruned.get("correctness_ok", pruned.get("oracle_passed")))
        and pruned_latency is not None and 0 < pruned_latency <= combined * 1.005 + 1e-9
    )
    pruned_verified = bool(pruned_run_id) and bool(_flag(pruned.get("verified")) or pruned_implied)
    decisions = [_factor_trial(trial, index, combined, ratio_floor)
                 for index, trial in enumerate(factor_trials or ())]
    candidates = [row for row in decisions if row["status"] == "candidate_for_sequential_pruning"]
    verified_prune = bool(candidates) and pruned_verified
    retained = [row["factor_name"] for row in decisions
                if not (verified_prune and row["status"] == "candidate_for_sequential_pruning")]
    pruned_factors = [row["factor_name"] for row in decisions
                      if verified_prune and row["status"] == "candidate_for_sequential_pruning"]
    for row in decisions:
        row["decision"] = "PRUNE" if verified_prune and row["status"] == "candidate_for_sequential_pruning" else "RETAIN"
    return {
        "schema_version": SCHEMA_ABLATION,
        "baseline_run_id": str(baseline_run_id),
        "candidate_run_id": str(candidate_run_id),
        "verification_run_id": pruned_run_id or str(candidate_run_id),
        "baseline_latency_us": baseline_latency,
        "combined_latency_us": combined,
        "min_factor_contribution_ratio": ratio_floor,
        "ablation_retained": not verified_prune,
        "pruned_combination_verified": pruned_verified,
        "retained_factors": retained,
        "pruned_factors": pruned_factors,
        "factor_decisions": decisions,
        "net_speedup": baseline_latency / combined,
    }


def _ablation_from_spec(spec: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(spec)
    factors = row.get("factor_trials", row.get("factors", row.get("leave_one_out")))
    if isinstance(factors, Mapping):
        factors = [{"factor_name": name, **_mapping(value)} for name, value in factors.items()]
    return verify_sequential_ablation(
        baseline_run_id=str(row.get("baseline_run_id") or source.get("baseline_run_id") or "baseline"),
        candidate_run_id=str(row.get("candidate_run_id") or source.get("run_id")
                              or source.get("candidate_id") or "candidate"),
        baseline_latency_us=row.get("baseline_latency_us", source.get("baseline_latency_us")),
        combined_latency_us=row.get("combined_latency_us", row.get("candidate_latency_us", source.get("latency_us"))),
        factor_trials=factors or (),
        pruned_combination_run=row.get("pruned_combination_run", row.get("pruned_combination")),
        min_factor_contribution_ratio=row.get("min_factor_contribution_ratio", row.get("contribution_ratio", 0.01)),
    )


def run_rsi_gate_campaign(
    *,
    goal: str = "Optimize operator throughput",
    units: Sequence[Mapping[str, Any]] = (),
    ledger_entries: Sequence[Mapping[str, Any]] = (),
    bottleneck_summary: Mapping[str, Any] | None = None,
    conditions: Mapping[str, Any] | None = None,
    native_subcontext: bool = False,
    consecutive_limit: int = 2,
    candidate: Mapping[str, Any] | None = None,
    candidate_payload: Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    command_plan: Any = None,
    command_Plan: Any = None,
    runner: Any = None,
    execution_backend: Any = None,
    lease_authority: Any = None,
    gate_runner: Any = None,
    gate_specs: Any = None,
    hidden_input_cases: Sequence[Mapping[str, Any]] | None = None,
    hidden_cases: Sequence[Mapping[str, Any]] | None = None,
    ablation_spec: Mapping[str, Any] | None = None,
    ledger_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    campaign_id: str | None = None,
    state_dir: str | Path | None = None,
    npu_device: int = 0,
    max_compile_slots: int = 3,
    min_speedup_ratio: float = 1.05,
    min_factor_contribution_ratio: float = 0.01,
    incumbent_speedup: float = 1.0,
    min_improve_delta: float = 0.01,
    reject_narrow_flips: bool = False,
    speculative_metrics: Mapping[str, Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    distill_callback: Callable[..., Any] | None = None,
    feedback_callback: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one complete RSI campaign and return the primary gate receipt."""
    from mindie_bench.runner.rsi_explorer import (
        append_ledger_and_render_progress,
        build_debiased_rsi_context,
        evaluate_rsi_candidate_gates,
        execute_rsi_compile_and_timing,
        should_escalate_to_rsi,
    )

    entries = [dict(entry) for entry in ledger_entries or () if isinstance(entry, Mapping)]
    payload = dict(candidate or candidate_payload or {})
    plan = command_plan if command_plan is not None else command_Plan
    rows = _candidate_rows(plan, candidates)
    if payload:
        rows = _candidate_rows(None, [*rows, payload])
    if not rows:
        raise ValueError("campaign requires a candidate payload or command plan")
    identity = str(campaign_id or payload.get("candidate_id") or rows[0].get("candidate_id") or "rsi-campaign")
    summary, context_conditions = dict(bottleneck_summary or {}), dict(conditions or {})
    escalate, reason = should_escalate_to_rsi(entries, max_stalled_attempts=consecutive_limit)
    context = build_debiased_rsi_context(
        list(units), goal=goal, bottleneck_summary=summary,
        failed_attempts=entries, conditions=context_conditions, native_subcontext=native_subcontext,
    )

    execution = None
    source = dict(payload or rows[0])
    if plan is not None or runner is not None or execution_backend is not None or candidates is not None:
        if (execution_backend is None and callable(getattr(runner, "acquire", None))
                and callable(getattr(runner, "release", None))):
            # A production TaskClient adapter already owns both callbacks and
            # the lease authority; do not split it into unleased callables.
            execution_backend = runner
        if execution_backend is not None:
            compile_runner, timing_runner = None, None
        else:
            compile_runner, timing_runner = _runner_callbacks(runner)
        execution = execute_rsi_compile_and_timing(
            rows, compile_runner=compile_runner, timing_runner=timing_runner,
            execution_backend=execution_backend, lease_authority=lease_authority,
            state_dir=state_dir or f"/tmp/mindie-rsi-{identity}", npu_device=npu_device,
            max_compile_slots=max_compile_slots, min_speedup_ratio=min_speedup_ratio,
        )
        records = [_mapping(record) for record in execution.get("candidates", rows)]
        target = str(payload.get("candidate_id") or "")
        index = next((i for i, row in enumerate(records) if target and str(row.get("candidate_id")) == target), 0)
        source = records[index]

    warnings: list[str] = []
    qualification_gates = _run_qualification_gates(gate_runner, gate_specs, source, context)
    qualification_failed = [gate for gate, result in qualification_gates.items()
                            if _gate_state(result) in _GATE_FAILED]
    qualification_unknown = [gate for gate, result in qualification_gates.items()
                             if _gate_state(result) not in _GATE_OK and _gate_state(result) not in _GATE_FAILED]
    compile_ok = _compile_ok(source)
    oracle_passed = _flag(source.get("oracle_passed", source.get("correctness_ok")))
    measurement_passed = _flag(
        source.get("measurement_passed", source.get("measurement_valid", source.get("timing_valid")))
    )
    failure_reason = str(source.get("failure_reason") or "")
    if qualification_failed:
        if any(gate in {"G0", "G1"} for gate in qualification_failed):
            compile_ok = False
        elif any(gate in {"G2", "G3"} for gate in qualification_failed):
            oracle_passed = False
        else:
            measurement_passed = False
        failure_reason = failure_reason or "qualification_gate_failed:" + ",".join(sorted(qualification_failed))
    elif qualification_unknown:
        measurement_passed = False
        failure_reason = failure_reason or "qualification_gate_unknown:" + ",".join(sorted(qualification_unknown))

    cases = hidden_input_cases if hidden_input_cases is not None else hidden_cases
    hidden_summary = (
        evaluate_hidden_input_generalization(cases or (), rtol=rtol, atol=atol)
        if cases is not None else _mapping(source.get("hidden_input_summary")) or None
    )
    spec = _mapping(ablation_spec) if ablation_spec is not None else _mapping(source.get("ablation_spec"))
    ablation_receipt = (
        _ablation_from_spec({**spec, "min_factor_contribution_ratio": spec.get(
            "min_factor_contribution_ratio", min_factor_contribution_ratio)}, source)
        if spec else _mapping(source.get("ablation_receipt")) or None
    )
    patch_input = source.get("diff_text", source.get("patch_diff", source.get("changed_files")))
    patch_classification = (
        classify_candidate_patch_shape(patch_input, reject_narrow_flips=reject_narrow_flips)
        if patch_input is not None else None
    )
    ablation_retained = source.get("ablation_retained")
    if patch_classification and patch_classification["rejected_by_narrow_gate"]:
        ablation_retained = False
        failure_reason = failure_reason or f"narrow_patch_rejected:{patch_classification['category']}"
    spec_payload = _mapping(speculative_metrics) or _mapping(source.get("speculative_metrics"))
    spec_acceptance = (
        evaluate_speculative_acceptance(
            spec_payload.get("candidate", spec_payload),
            spec_payload.get("pristine", spec_payload.get("baseline")),
        )
        if spec_payload else None
    )
    if spec_acceptance and not spec_acceptance["acceptance_ok"]:
        oracle_passed = False
        failure_reason = failure_reason or f"speculative_acceptance_failed:{spec_acceptance['reason']}"
    speedup = _source_speedup(source)
    entry = evaluate_rsi_candidate_gates(
        run_id=str(source.get("run_id") or f"{identity}:candidate"),
        candidate_id=str(source.get("candidate_id") or identity),
        mutation_summary=str(source.get("mutation_summary") or ""),
        conditions=dict(source.get("conditions") or context_conditions),
        compile_ok=compile_ok, oracle_passed=oracle_passed, measurement_passed=measurement_passed,
        speedup_ratio=speedup, min_speedup_ratio=min_speedup_ratio,
        ablation_retained=ablation_retained, failure_reason=failure_reason,
        hidden_input_summary=hidden_summary, ablation_receipt=ablation_receipt,
        t_cand_us=_number(source.get("t_cand_us", source.get("cand_latency_us", source.get("latency_us")))),
        t_baseline_us=_number(source.get("t_baseline_us", source.get("baseline_perf_us", source.get("baseline_latency_us")))),
        t_hw_us=_number(source.get("t_hw_us")),
        roofline_floor_us=_number(source.get("roofline_floor_us")),
    )
    incumbent = max(1.0, float(_number(incumbent_speedup) or 1.0))
    delta = max(0.0, float(_number(min_improve_delta) or 0.0))
    evidence_valid = bool(compile_ok and oracle_passed and measurement_passed)
    improved_over_incumbent = bool(
        evidence_valid and speedup is not None and speedup >= incumbent * (1.0 + delta) - 1e-9
    )
    became_winner = bool(entry["verdict"] == "supported" and improved_over_incumbent)
    entry["incumbent_comparison"] = {
        "incumbent_speedup": incumbent,
        "improved_over_incumbent": improved_over_incumbent,
        "became_winner": became_winner,
        "reprofile_required": became_winner,
        "cited_then_verified": None if not evidence_valid else improved_over_incumbent,
    }
    if patch_classification is not None:
        entry["patch_classification"] = patch_classification
    if spec_acceptance is not None:
        entry["speculative_acceptance"] = spec_acceptance
    if qualification_gates:
        entry["qualification_gates"] = qualification_gates
        entry["qualification_ok"] = not qualification_failed and not qualification_unknown
    if execution:
        authority = _mapping(execution.get("execution_authority"))
        entry["execution_evidence"] = {
            "evidence_class": authority.get("evidence_class", "unavailable"),
            "production_adapter": bool(authority.get("production_adapter", False)),
            "physical_acceptance": False,
            "cleanup_ok": authority.get("cleanup_ok") is not False,
            "collection_mode": source.get("measurement_collection", "not_collected"),
        }
        if source.get("cleanup_failures"):
            entry["cleanup_failures"] = list(source["cleanup_failures"])
            entry["cleanup_ok"] = False

    ledger_appended = False
    if ledger_path is not None and progress_path is not None:
        try:
            append_ledger_and_render_progress(
                Path(ledger_path), Path(progress_path), entry, campaign_id=identity, goal=goal
            )
            ledger_appended = True
        except Exception as exc:
            warnings.append(f"ledger_append:{type(exc).__name__}:{exc}")
    elif ledger_path is not None or progress_path is not None:
        warnings.append("ledger_append:ledger_path_and_progress_path_are_both_required")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_CAMPAIGN, "campaign_id": identity, "goal": goal,
        "should_escalate": escalate, "escalation_reason": reason,
        "escalation": {"should_escalate": escalate, "reason": reason, "consecutive_limit": consecutive_limit},
        "rsi_context": context, "context": context, "execution": execution,
        "evaluations": execution.get("evaluations", []) if execution else [entry],
        "evaluation": entry, "receipt": entry,
        "verdict": entry["verdict"], "campaign_verdict": entry["verdict"], "outcome": entry["outcome"],
        "gate_reached": entry["gate_reached"], "gates": entry["gates"],
        "gate_results": qualification_gates, "qualification_gates": qualification_gates,
        "qualification_ok": not qualification_failed and not qualification_unknown,
        "hidden_input_summary": hidden_summary, "ablation_receipt": ablation_receipt,
        "patch_classification": patch_classification, "speculative_acceptance": spec_acceptance,
        "incumbent_comparison": entry["incumbent_comparison"],
        "reprofile_required": became_winner,
        "ledger_appended": ledger_appended, "distill_result": None, "feedback_result": None,
        "accounting_warnings": warnings, "campaign_passed": entry["verdict"] == "supported",
    }
    for name, callback, key in (
        ("distill_callback", distill_callback, "distill_result"),
        ("feedback_callback", feedback_callback, "feedback_result"),
    ):
        if callback is None:
            continue
        try:
            result[key] = callback(entry)
        except Exception as exc:
            warnings.append(f"{name}:{type(exc).__name__}:{exc}")
    return result


__all__ = [
    "build_isolated_jit_cache_env",
    "build_subagent_fact_pack",
    "classify_candidate_patch_shape",
    "evaluate_hidden_input_generalization",
    "evaluate_speculative_acceptance",
    "harvest_candidate_patch",
    "run_rsi_gate_campaign",
    "verify_sequential_ablation",
]
