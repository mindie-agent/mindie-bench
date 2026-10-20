"""Conditional ablation evidence for a declared optimization combination.

[Role] Evaluate C4 receipts without editing candidates or running experiments.
[Invariants] Leave-one-out is conditional evidence; every pruning step needs a
fresh correctness and performance receipt for the remaining combination.
[Timeouts & Locks] Pure in-memory evaluation; no waits, subprocesses or locks.
[Side-Effect Policy] Returns evidence and recommendations, never deletes code.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal

from pydantic import Field, model_validator

from mindie_bench.evaluation.ablation_statistics import (
    SampleGroup,
    TimingComparison,
    compare_timings,
    observed_lower_bounds_pass,
    timing_identity,
    validate_groups,
)
from mindie_bench.evaluation.performance import TimingRun
from mindie_bench.models import StrictModel

FactorRole = Literal["performance", "correctness_required", "dependency", "inseparable_group"]
RunStatus = Literal["passed", "failed", "infra_failed", "inconclusive", "cancelled", "unknown"]


class AblationFactor(StrictModel):
    factor_id: str = Field(min_length=1)
    role: FactorRole = "performance"
    depends_on: list[str] = Field(default_factory=list)
    # An inseparable group is one experimental unit, never several independent votes.
    members: list[str] = Field(default_factory=list)


class AblationRun(StrictModel):
    run_id: str = Field(min_length=1)
    active_factors: list[str]
    conditions: dict[str, Any] = Field(min_length=1)
    run_status: RunStatus
    correctness_status: Literal["passed", "failed", "not_measured"]
    measurement_valid: bool
    samples_ms: list[float] = Field(default_factory=list)
    timing: TimingRun | None = None
    sample_groups: list[SampleGroup] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    measurement_limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def measurement_samples(self):
        if any(not math.isfinite(value) or value <= 0 for value in self.samples_ms):
            raise ValueError("timing samples must be finite and positive")
        if len(set(self.active_factors)) != len(self.active_factors):
            raise ValueError("active factor identities must be unique")
        if self.measurement_valid and len(self.samples_ms) < 2:
            raise ValueError("valid measurements require repeated timing samples")
        if self.sample_groups:
            if self.timing is None:
                raise ValueError("case/block groups require bound raw timing evidence")
            validate_groups(self.timing, self.sample_groups, self.samples_ms)
        return self


class FactorReceipt(StrictModel):
    factor_id: str
    role: FactorRole
    depends_on: list[str]
    ablation_run_id: str | None = None
    correctness_status: str
    revert_speedup_p50: float | None = None
    conditional_delta_p50: float | None = None
    sample_iqr_ms: float | None = None
    revert_speedup_geomean: float | None = None
    conditional_delta_geomean: float | None = None
    contribution_comparison: TimingComparison | None = None
    disposition: str


class CombinationReceipt(StrictModel):
    retained_factors: list[str]
    verification_run_id: str
    correctness_verified: bool
    performance_verified: bool
    final_speedup_p50: float | None
    final_speedup_geomean: float | None = None
    baseline_comparison: TimingComparison | None = None
    retention_comparison: TimingComparison | None = None
    final_timing_sha256: str | None = None
    uncertainty_verified: bool = False


class AblationReceipt(StrictModel):
    schema_version: Literal["mindie-bench-ablation/1", "mindie-bench-ablation/2"] = (
        "mindie-bench-ablation/2"
    )
    task_id: str
    baseline_run_id: str
    full_candidate_run_id: str
    full_candidate_speedup: float | None
    noise_floor_rel_iqr: float | None
    full_candidate_comparison: TimingComparison | None = None
    max_relative_loss: float = 0.015
    min_speedup: float = 1.0
    noise_scope: str = (
        "maximum within-case within-block relative IQR; descriptive, not a confidence interval"
    )
    factors: list[FactorReceipt]
    pruned_combination: CombinationReceipt
    pruned_combination_verified: bool
    final_combination: CombinationReceipt | None = None
    final_combination_verified: bool = False
    status: Literal["passed", "failed", "inconclusive"]
    reasons: list[str]
    runs: list[AblationRun]
    pruning_verifications: list[CombinationReceipt]
    claim_scope: str = "conditional contributions under the recorded conditions and samples"


class AblationRequest(StrictModel):
    task_id: str
    factors: list[AblationFactor]
    baseline: AblationRun
    full_candidate: AblationRun
    ablations: list[AblationRun] = Field(default_factory=list)
    pruning_runs: list[AblationRun] = Field(default_factory=list)
    final_candidate: AblationRun | None = None
    min_speedup: float = Field(default=1.0, gt=0)
    max_relative_loss: float = Field(default=0.015, ge=0, lt=1)

    def evaluate(self) -> AblationReceipt:
        return AblationGate(
            min_speedup=self.min_speedup, max_relative_loss=self.max_relative_loss
        ).evaluate(
            task_id=self.task_id,
            factors=self.factors,
            baseline=self.baseline,
            full_candidate=self.full_candidate,
            ablations=self.ablations,
            pruning_runs=self.pruning_runs,
            final_candidate=self.final_candidate,
        )


def _iqr(values: list[float]) -> float:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _usable(run: AblationRun) -> bool:
    return (
        run.run_status == "passed"
        and run.correctness_status == "passed"
        and run.measurement_valid
        and not run.measurement_limitations
    )


class AblationGate:
    """Validate a measured proposal; a caller retains ownership of experiment execution.

    ``max_relative_loss`` declares the allowed final loss against the full
    candidate. The observed relative IQR is also retained as uncertainty, not
    treated as a confidence interval or evidence of independent causality.
    """

    def __init__(self, *, min_speedup: float = 1.0, max_relative_loss: float = 0.015):
        if not math.isfinite(min_speedup) or min_speedup <= 0 or not 0 <= max_relative_loss < 1:
            raise ValueError("invalid declared performance requirements")
        self.min_speedup = min_speedup
        self.max_relative_loss = max_relative_loss

    def evaluate(
        self,
        *,
        task_id: str,
        factors: list[AblationFactor],
        baseline: AblationRun,
        full_candidate: AblationRun,
        ablations: list[AblationRun] = (),
        pruning_runs: list[AblationRun] = (),
        final_candidate: AblationRun | None = None,
    ) -> AblationReceipt:
        factor_map = {factor.factor_id: factor for factor in factors}
        factor_ids = set(factor_map)
        if len(factor_map) != len(factors) or not factor_map:
            raise ValueError("declare nonempty, unique experimental factors")
        if baseline.active_factors or set(full_candidate.active_factors) != factor_ids:
            raise ValueError("baseline and full candidate do not match the declared combination")
        for factor in factors:
            if not set(factor.depends_on) <= factor_ids - {factor.factor_id}:
                raise ValueError("factor dependencies must name other declared factors")
            if factor.role == "inseparable_group" and len(set(factor.members)) < 2:
                raise ValueError("an inseparable group must identify its constituent modifications")
        self._check_dependency_cycles(factor_map)
        runs = [baseline, full_candidate, *ablations, *pruning_runs]
        if final_candidate is not None:
            runs.append(final_candidate)
        if len({run.run_id for run in runs}) != len(runs):
            raise ValueError("every experimental measurement needs a distinct run_id")
        if any(run.conditions != baseline.conditions for run in runs):
            raise ValueError("conditional comparisons require matched environment and input conditions")
        if any(not set(run.active_factors) <= factor_ids for run in runs):
            raise ValueError("measurement contains an undeclared factor")
        by_removed = {}
        for run in ablations:
            removed = factor_ids - set(run.active_factors)
            if len(removed) != 1:
                raise ValueError("leave-one-out removes one factor or one inseparable group")
            identity = next(iter(removed))
            if identity in by_removed:
                raise ValueError("combine repeated samples in one ablation run receipt")
            by_removed[identity] = run

        baseline_median = statistics.median(baseline.samples_ms) if _usable(baseline) else None

        def pooled_speedup(run):
            return (
                baseline_median / statistics.median(run.samples_ms)
                if (baseline_median is not None and _usable(run))
                else None
            )

        def comparison(left, right):
            return compare_timings(left, right) if _usable(left) and _usable(right) else None

        full_comparison = comparison(baseline, full_candidate)
        full_speedup = full_comparison.speedup_geomean if full_comparison else None
        full_p50 = pooled_speedup(full_candidate)
        noise = (
            max(
                (
                    value
                    for case in full_comparison.cases
                    for value in (
                        case.reference_within_block_max_rel_iqr,
                        case.candidate_within_block_max_rel_iqr,
                    )
                ),
                default=None,
            )
            if full_comparison
            else None
        )
        receipts = []
        reasons = []
        for factor in factors:
            run = by_removed.get(factor.factor_id)
            revert_comparison = comparison(baseline, run) if run else None
            contribution = comparison(run, full_candidate) if run else None
            revert = revert_comparison.speedup_geomean if revert_comparison else None
            revert_p50 = pooled_speedup(run) if run else None
            delta = full_speedup - revert if full_speedup is not None and revert is not None else None
            required = factor.role in {"correctness_required", "dependency"}
            depended_on = any(factor.factor_id in other.depends_on for other in factors)
            if required or depended_on:
                disposition = "retain_required_dependency"
            elif run is not None and run.run_status == "failed" and run.correctness_status == "failed":
                disposition = "retain_required_by_correctness_evidence"
            elif run is None or delta is None:
                disposition = "retain_pending_evidence"
                reasons.append(f"ablation_unresolved:{factor.factor_id}")
            elif observed_lower_bounds_pass(contribution, 1 + self.max_relative_loss):
                disposition = "retain_verified_contributor"
            else:
                disposition = "candidate_for_sequential_pruning"
            receipts.append(
                FactorReceipt(
                    factor_id=factor.factor_id,
                    role=factor.role,
                    depends_on=factor.depends_on,
                    ablation_run_id=run.run_id if run else None,
                    correctness_status=(
                        run.correctness_status
                        if run
                        else "required_for_safety"
                        if required
                        else "not_measured"
                    ),
                    revert_speedup_p50=revert_p50,
                    conditional_delta_p50=(
                        full_p50 - revert_p50
                        if full_p50 is not None and revert_p50 is not None
                        else None
                    ),
                    revert_speedup_geomean=revert,
                    conditional_delta_geomean=delta,
                    contribution_comparison=contribution,
                    sample_iqr_ms=_iqr(run.samples_ms) if run and _usable(run) else None,
                    disposition=disposition,
                )
            )
        receipt_map = {receipt.factor_id: receipt for receipt in receipts}

        def combination(run):
            against_baseline = comparison(baseline, run)
            retention = comparison(full_candidate, run)
            measured = against_baseline.speedup_geomean if against_baseline else None
            uncertainty_verified = observed_lower_bounds_pass(
                against_baseline, self.min_speedup
            ) and observed_lower_bounds_pass(retention, 1 - self.max_relative_loss)
            return CombinationReceipt(
                retained_factors=run.active_factors,
                verification_run_id=run.run_id,
                correctness_verified=run.run_status == "passed" and run.correctness_status == "passed",
                performance_verified=(
                    measured is not None
                    and full_speedup is not None
                    and measured >= self.min_speedup
                    and measured >= full_speedup * (1 - self.max_relative_loss)
                    and uncertainty_verified
                ),
                final_speedup_p50=pooled_speedup(run),
                final_speedup_geomean=measured,
                baseline_comparison=against_baseline,
                retention_comparison=retention,
                final_timing_sha256=timing_identity(run.timing) if run.timing else None,
                uncertainty_verified=uncertainty_verified,
            )

        current = factor_ids
        verified_steps = []
        failed = full_candidate.run_status == "failed" and full_candidate.correctness_status == "failed"
        unresolved = baseline_median is None or full_speedup is None
        if unresolved:
            reasons.append("baseline_or_full_candidate_unresolved")
        elif full_speedup < self.min_speedup:
            failed = True
            reasons.append("full_candidate_below_performance_requirement")
        for run in pruning_runs:
            remaining = set(run.active_factors)
            removed = current - remaining
            if not remaining <= current or len(removed) != 1:
                raise ValueError("prune one factor at a time and reverify the remaining combination")
            factor_id = next(iter(removed))
            factor = factor_map[factor_id]
            if factor.role in {"correctness_required", "dependency"} or any(
                factor_id in factor_map[retained].depends_on for retained in remaining
            ):
                raise ValueError(
                    "cannot prune a correctness requirement or a retained factor's dependency"
                )
            checked = combination(run)
            verified_steps.append(checked)
            if not checked.correctness_verified or not checked.performance_verified:
                step_failed = run.correctness_status == "failed" or (
                    checked.final_speedup_geomean is not None
                    and (
                        checked.final_speedup_geomean < self.min_speedup
                        or (
                            full_speedup is not None
                            and checked.final_speedup_geomean
                            < full_speedup * (1 - self.max_relative_loss)
                        )
                    )
                )
                failed = failed or step_failed
                unresolved = unresolved or not step_failed
                reasons.append(f"pruned_combination_not_verified:{run.run_id}")
                receipt_map[factor_id].disposition = "retain_pruning_not_verified"
                if run is not pruning_runs[-1]:
                    raise ValueError(
                        "a failed or unresolved pruning step cannot justify further pruning"
                    )
                break
            receipt_map[factor_id].disposition = "prune_verified_by_combination"
            current = remaining

        final = verified_steps[-1] if verified_steps else combination(full_candidate)
        pruned_verified = bool(verified_steps) and all(
            step.correctness_verified and step.performance_verified for step in verified_steps
        )
        final_verified = pruned_verified
        if final_candidate is not None:
            if set(final_candidate.active_factors) != current:
                raise ValueError("final candidate does not match the retained combination")
            final = combination(final_candidate)
            final_verified = (
                final.correctness_verified
                and final.performance_verified
                and (not verified_steps or pruned_verified)
            )
            if not final_verified:
                final_failed = final_candidate.correctness_status == "failed" or (
                    final.final_speedup_geomean is not None
                    and (
                        final.final_speedup_geomean < self.min_speedup
                        or (
                            full_speedup is not None
                            and final.final_speedup_geomean < full_speedup * (1 - self.max_relative_loss)
                        )
                    )
                )
                failed = failed or final_failed
                unresolved = unresolved or not final_failed
                reasons.append(f"final_combination_not_verified:{final_candidate.run_id}")
        unresolved = unresolved or bool(
            reasons
            and any(
                reason.startswith("ablation_unresolved:")
                and receipt_map[reason.split(":", 1)[1]].disposition != "prune_verified_by_combination"
                for reason in reasons
            )
        )
        return AblationReceipt(
            task_id=task_id,
            baseline_run_id=baseline.run_id,
            full_candidate_run_id=full_candidate.run_id,
            full_candidate_speedup=full_speedup,
            noise_floor_rel_iqr=noise,
            full_candidate_comparison=full_comparison,
            max_relative_loss=self.max_relative_loss,
            min_speedup=self.min_speedup,
            factors=receipts,
            pruned_combination=verified_steps[-1] if verified_steps else combination(full_candidate),
            pruned_combination_verified=pruned_verified,
            final_combination=final,
            final_combination_verified=final_verified,
            status="failed" if failed else "inconclusive" if unresolved else "passed",
            reasons=reasons,
            runs=runs,
            pruning_verifications=verified_steps,
        )

    @staticmethod
    def _check_dependency_cycles(factors: dict[str, AblationFactor]):
        visited, active = set(), set()

        def visit(identity):
            if identity in active:
                raise ValueError("factor dependencies contain a cycle; declare an inseparable group")
            if identity in visited:
                return
            active.add(identity)
            for dependency in factors[identity].depends_on:
                visit(dependency)
            active.remove(identity)
            visited.add(identity)

        for identity in factors:
            visit(identity)
