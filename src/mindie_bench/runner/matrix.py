"""Product-level comparison matrix, distinct from the legacy component groups.

[Role]
Plan, interleave, execute and account P0-P3 product comparisons across the bare host,
v0.0.2, v0.0.3, seed-capture-evaluation and independently reported external
product profiles.

[Invariants]
- `ProductGroup` is a separate protocol (`mindie-bench-product-matrix/1`); it
  never maps or reinterprets `models.Group` G0..G3 component-increment results.
- Matched profiles in one cohort share model, inference, task start, oracle,
  input, knowledge cutoff, tool/persistence permissions and matched downstream
  budget. Separately accounted prerequisite flywheel cost is not free Seed and
  is never added to P2's downstream total; equal downstream budget proves
  conditional reuse only.
- Faithful external profiles must declare a different execution protocol and
  are reported in their own columns; they are never silently paired or pooled.
- Every planned cell remains in the denominator. Pass, fail, infra, unknown and
  missing are separate counts, and completion is computed only over valid
  terminal outcomes.
- Only an explicitly supplied executor can run a cell. This module never
  starts a model, NPU or paid experiment on its own.

[Timeouts & Locks]
- Planning, validation and accounting are pure. The P3 execution entrance
  performs bounded read-only receipt/corpus verification before any caller-owned
  executor; the caller still owns native execution timeouts and locks.

[Side-Effect Policy]
- Planning performs no I/O. P3 matrix execution reads only bound receipts and
  corpus bytes before invoking the caller executor; it never starts a model, NPU
  or paid experiment itself.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mindie_bench.models import TrialReceipt
from mindie_bench.runner.matrix_contracts import (
    _OUTCOME_ALIASES as _OUTCOME_ALIASES,
)
from mindie_bench.runner.matrix_contracts import (
    _TERMINAL_OUTCOMES as _TERMINAL_OUTCOMES,
)
from mindie_bench.runner.matrix_contracts import (
    ALL_PRODUCT_GROUPS as ALL_PRODUCT_GROUPS,
)
from mindie_bench.runner.matrix_contracts import (
    EXTERNAL_PRODUCT_GROUPS as EXTERNAL_PRODUCT_GROUPS,
)
from mindie_bench.runner.matrix_contracts import (
    MATCHED_PRODUCT_GROUPS as MATCHED_PRODUCT_GROUPS,
)
from mindie_bench.runner.matrix_contracts import (
    PRODUCT_GROUP_DESCRIPTIONS as PRODUCT_GROUP_DESCRIPTIONS,
)
from mindie_bench.runner.matrix_contracts import (
    PRODUCT_MATRIX_PROTOCOL as PRODUCT_MATRIX_PROTOCOL,
)
from mindie_bench.runner.matrix_contracts import (
    BudgetEnvelope as BudgetEnvelope,
)
from mindie_bench.runner.matrix_contracts import (
    CellDenominator as CellDenominator,
)
from mindie_bench.runner.matrix_contracts import (
    CellOutcome as CellOutcome,
)
from mindie_bench.runner.matrix_contracts import (
    MatchedPair as MatchedPair,
)
from mindie_bench.runner.matrix_contracts import (
    MatchingConditions as MatchingConditions,
)
from mindie_bench.runner.matrix_contracts import (
    MatrixTask as MatrixTask,
)
from mindie_bench.runner.matrix_contracts import (
    MeasurementTotal as MeasurementTotal,
)
from mindie_bench.runner.matrix_contracts import (
    ProductCell as ProductCell,
)
from mindie_bench.runner.matrix_contracts import (
    ProductGroup as ProductGroup,
)
from mindie_bench.runner.matrix_contracts import (
    ProductGroupReport as ProductGroupReport,
)
from mindie_bench.runner.matrix_contracts import (
    ProductMatrixPlan as ProductMatrixPlan,
)
from mindie_bench.runner.matrix_contracts import (
    ProductMatrixReport as ProductMatrixReport,
)
from mindie_bench.runner.matrix_contracts import (
    ProductObservation as ProductObservation,
)
from mindie_bench.runner.matrix_contracts import (
    ProductProfile as ProductProfile,
)
from mindie_bench.runner.matrix_contracts import (
    _as_product_group as _as_product_group,
)
from mindie_bench.runner.matrix_contracts import (
    normalize_outcome as normalize_outcome,
)
from mindie_bench.runner.p3_lineage import P3LineageRejected


def _admit_p3_profile(
    profile: ProductProfile,
    task_id: str,
    *,
    verify_artifacts: bool,
    task_family: str | None = None,
    hardware_tier: str | None = None,
) -> None:
    """Use the same entrance as ProductExecutor.

    Planning may pass verify_artifacts=False. Matrix execution must pass True so
    a caller-owned executor cannot skip fresh receipt/corpus verification.
    """
    binding = profile.p3_lineage
    if binding is None:
        raise ValueError("P3 product profiles require a closed-family lineage admission")
    target = binding.manifest.target
    if target.matrix_task_family is not None and task_family is not None:
        if target.matrix_task_family != task_family:
            raise ValueError(
                "P3 lineage matrix task_family label conflicts with the planned cell "
                f"({target.matrix_task_family!r} != {task_family!r})"
            )
    if target.hardware_tier is not None and hardware_tier is not None:
        if target.hardware_tier != hardware_tier:
            raise ValueError(
                "P3 lineage hardware tier conflicts with the planned cell "
                f"({target.hardware_tier!r} != {hardware_tier!r})"
            )
    try:
        binding.admit(
            profile_id=profile.profile_id,
            product_group=profile.group.value,
            cohort=profile.cohort,
            protocol=profile.protocol,
            task_id=task_id,
            verify_artifacts=verify_artifacts,
        )
    except P3LineageRejected as error:
        raise ValueError(f"P3 lineage admission failed for {task_id}: {error}") from error


def _validate_profiles(profiles: Sequence[ProductProfile]) -> None:
    if not profiles:
        raise ValueError("at least one product profile is required")
    seen_ids = set()
    seen_groups = set()
    for profile in profiles:
        if not isinstance(profile, ProductProfile):
            raise TypeError("profiles must be ProductProfile values")
        if profile.profile_id in seen_ids:
            raise ValueError(f"duplicate product profile_id: {profile.profile_id}")
        if profile.group in seen_groups:
            raise ValueError(f"duplicate product group column: {profile.group.value}")
        seen_ids.add(profile.profile_id)
        seen_groups.add(profile.group)
    for cohort, members in _by_cohort(profiles).items():
        matched = [profile for profile in members if profile.fidelity == "matched"]
        external = [profile for profile in members if profile.fidelity == "faithful_external"]
        if not matched:
            continue
        fingerprints = {profile.comparison_fingerprint() for profile in matched}
        protocols = {profile.protocol for profile in matched}
        if len(protocols) != 1:
            raise ValueError(f"matched cohort {cohort!r} mixes execution protocols: {sorted(protocols)}")
        if len(fingerprints) != 1:
            first = matched[0]
            for other in matched[1:]:
                differences = _condition_differences(first, other)
                if differences:
                    raise ValueError(
                        f"matched cohort {cohort!r} is not condition-matched: " + "; ".join(differences)
                    )
        for profile in external:
            if profile.protocol in protocols:
                raise ValueError(
                    f"faithful external group {profile.group.value!r} must declare a different "
                    f"protocol than matched cohort {cohort!r} ({profile.protocol!r}); "
                    "it cannot be silently paired"
                )


def _by_cohort(profiles: Sequence[ProductProfile]) -> dict[str, list[ProductProfile]]:
    result = defaultdict(list)
    for profile in profiles:
        result[profile.cohort].append(profile)
    return result


def _condition_differences(left: ProductProfile, right: ProductProfile) -> list[str]:
    differences = []
    left_conditions = left.conditions.to_dict()
    right_conditions = right.conditions.to_dict()
    for name in left_conditions:
        if left_conditions[name] != right_conditions[name]:
            differences.append(f"{left.group.value}.{name} != {right.group.value}.{name}")
    left_budget = left.budget.comparison_dict()
    right_budget = right.budget.comparison_dict()
    for name in left_budget:
        if left_budget[name] != right_budget[name]:
            differences.append(f"{left.group.value}.budget.{name} != {right.group.value}.budget.{name}")
    return differences


def plan_product_matrix(
    tasks: Sequence[MatrixTask],
    profiles: Sequence[ProductProfile],
    *,
    repetitions: int = 1,
    require_all_groups: bool = False,
) -> ProductMatrixPlan:
    """Plan an interleaved matrix without executing any cell."""
    if not tasks:
        raise ValueError("at least one matrix task is required")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("matrix task IDs must be unique")
    _validate_profiles(profiles)
    p3_profiles = [profile for profile in profiles if profile.group == ProductGroup.P3_V003_FLYWHEEL]
    if p3_profiles:
        planned_tasks = {task.task_id for task in tasks}
        for profile in p3_profiles:
            target = profile.p3_lineage.manifest.target
            if planned_tasks != {target.task_id}:
                raise ValueError(
                    "P3 lineage must bind exactly the planned downstream Heldout task; "
                    f"lineage={target.task_id!r} plan={sorted(planned_tasks)}"
                )
            _admit_p3_profile(
                profile,
                target.task_id,
                verify_artifacts=False,
                task_family=next(
                    task.task_family for task in tasks if task.task_id == target.task_id
                ),
                hardware_tier=next(
                    task.hardware_tier for task in tasks if task.task_id == target.task_id
                ),
            )
    if require_all_groups and {profile.group for profile in profiles} != set(ALL_PRODUCT_GROUPS):
        missing = sorted(group.value for group in set(ALL_PRODUCT_GROUPS) - {p.group for p in profiles})
        raise ValueError("product matrix is missing groups: " + ", ".join(missing))
    ordered_profiles = sorted(profiles, key=lambda profile: ALL_PRODUCT_GROUPS.index(profile.group))
    ordered_tasks = sorted(tasks, key=lambda task: task.task_id)
    cells = []
    order = 0
    for repetition in range(1, repetitions + 1):
        for task_index, task in enumerate(ordered_tasks):
            rotation = (repetition - 1 + task_index) % len(ordered_profiles)
            for offset in range(len(ordered_profiles)):
                profile = ordered_profiles[(rotation + offset) % len(ordered_profiles)]
                cells.append(
                    ProductCell(
                        profile.group,
                        profile.profile_id,
                        profile.cohort,
                        profile.protocol,
                        task.task_id,
                        task.task_family,
                        task.hardware_tier,
                        repetition,
                        order,
                    )
                )
                order += 1
    return ProductMatrixPlan(
        protocol=PRODUCT_MATRIX_PROTOCOL,
        cells=tuple(cells),
        profiles=tuple(ordered_profiles),
        tasks=tuple(ordered_tasks),
        repetitions=repetitions,
    )


_TOKEN_TOTAL_FIELDS = ("input_uncached", "input_cached", "output")


def _receipt_tokens_total(tokens: object) -> int | None:
    """Read the real Tokens structure; a complete flag never invents missing keys."""
    if not isinstance(tokens, Mapping) or tokens.get("complete") is not True:
        return None
    if not all(type(tokens.get(name)) is int and tokens[name] >= 0 for name in _TOKEN_TOTAL_FIELDS):
        return None
    return sum(tokens[name] for name in _TOKEN_TOTAL_FIELDS)


def observation_from_receipt(receipt: TrialReceipt, *, cell: ProductCell) -> ProductObservation:
    """Bind a trial receipt to an explicit product cell, never to legacy Group."""
    if receipt.instance_id != cell.task_id:
        raise ValueError("trial receipt task differs from the product cell")
    if receipt.repetition != cell.repetition:
        raise ValueError("trial receipt repetition differs from the product cell")
    primary = receipt.primary_metrics
    auxiliary = receipt.auxiliary_metrics
    cost = primary.get("estimated_cost_usd") if primary.get("cost_complete") is True else None
    npu_runs = primary.get("npu_test_runs")
    if npu_runs is None:
        npu_runs = primary.get("remote_test_invocations")
    manual_interventions = primary.get("manual_interventions")
    if manual_interventions is None:
        manual_interventions = auxiliary.get("manual_interventions")
    if manual_interventions is None:
        manual_interventions = auxiliary.get("human_interventions")
    if manual_interventions is None and primary.get("zero_human_intervention") is True:
        manual_interventions = 0
    notes = tuple(value for value in (receipt.failure_reason,) if isinstance(value, str) and value)
    return ProductObservation(
        group=cell.group,
        task_id=cell.task_id,
        repetition=cell.repetition,
        status=receipt.terminal_status,
        receipt_id=receipt.trial_id,
        model_calls=primary.get("model_calls"),
        tokens_total=_receipt_tokens_total(primary.get("tokens")),
        estimated_cost_usd=cost,
        # Product budgets cover delivery, including preparation, verification and cleanup.
        total_wall_clock_sec=primary.get("delivery_wall_clock_sec"),
        npu_runs=npu_runs,
        manual_interventions=manual_interventions,
        notes=notes,
    )


def _measurement(values: Sequence[int | float | None], planned_count: int) -> MeasurementTotal:
    observed = [value for value in values if value is not None]
    return MeasurementTotal(
        planned_count=planned_count,
        available_count=len(observed),
        sum_observed=sum(observed) if observed else None,
        complete=planned_count > 0 and len(observed) == planned_count,
    )


def _denominator(statuses: Sequence[CellOutcome | None]) -> CellDenominator:
    counts = {
        CellOutcome.PASS: 0,
        CellOutcome.FAIL: 0,
        CellOutcome.INFRA: 0,
        CellOutcome.UNKNOWN: 0,
    }
    observed = 0
    missing = 0
    for status in statuses:
        if status is None:
            missing += 1
            continue
        observed += 1
        counts[status] += 1
    return CellDenominator(
        planned=len(statuses),
        observed=observed,
        missing=missing,
        passed=counts[CellOutcome.PASS],
        failed=counts[CellOutcome.FAIL],
        infra=counts[CellOutcome.INFRA],
        unknown=counts[CellOutcome.UNKNOWN],
    )


def _bucket_denominators(
    cells: Sequence[ProductCell],
    observations: Mapping[tuple[ProductGroup, str, int], ProductObservation],
    attribute: str,
) -> dict[str, CellDenominator]:
    buckets = defaultdict(list)
    for cell in cells:
        buckets[getattr(cell, attribute)].append(_status(cell, observations))
    return {name: _denominator(statuses) for name, statuses in sorted(buckets.items())}


def _status(
    cell: ProductCell,
    observations: Mapping[tuple[ProductGroup, str, int], ProductObservation],
) -> CellOutcome | None:
    observation = observations.get(cell.key())
    return observation.status if observation else None


def _present(
    cells: Sequence[ProductCell],
    observations: Mapping[tuple[ProductGroup, str, int], ProductObservation],
) -> tuple[ProductObservation, ...]:
    return tuple(observations[cell.key()] for cell in cells if cell.key() in observations)


_MEASUREMENT_FIELDS = {
    "tokens_total": "tokens_total",
    "estimated_cost_usd": "cost_usd_total",
    "model_calls": "model_calls_total",
    "total_wall_clock_sec": "wall_clock_sec_total",
    "npu_runs": "npu_runs_total",
    "manual_interventions": "manual_interventions_total",
}


def _attempt_totals(
    observations: Sequence[ProductObservation], planned_count: int
) -> dict[str, MeasurementTotal]:
    return {
        result_name: _measurement([getattr(item, attribute) for item in observations], planned_count)
        for attribute, result_name in _MEASUREMENT_FIELDS.items()
    }


def _budget_contract(profile: ProductProfile) -> dict:
    """Make downstream reuse vs full-lifecycle allocation explicit."""
    budget = profile.budget
    return {
        "matched_downstream": {
            "total_model_calls": budget.total_model_calls,
            "npu_runs": budget.npu_runs,
            "max_wall_clock_sec": budget.max_wall_clock_sec,
            "max_cost_usd": budget.max_cost_usd,
        },
        "conditional_reuse_only": True,
        "lifecycle_predeclared": budget.lifecycle_model_calls is not None
        and budget.lifecycle_cost_usd is not None,
        "lifecycle_model_calls": budget.lifecycle_model_calls,
        "lifecycle_cost_usd": budget.lifecycle_cost_usd,
        "seed_organizer_overhead": {
            "seed_model_calls": budget.seed_model_calls,
            "capture_model_calls": budget.capture_model_calls,
            "capture_organizer_model_calls": budget.capture_organizer_model_calls,
            "independent_review_model_calls": budget.independent_review_model_calls,
            "seed_npu_runs": budget.seed_npu_runs,
            "seed_cost_usd": budget.seed_cost_usd,
            "capture_cost_usd": budget.capture_cost_usd,
            "capture_organizer_cost_usd": budget.capture_organizer_cost_usd,
            "independent_review_cost_usd": budget.independent_review_cost_usd,
            "downstream_organizer_allowance": budget.organizer_model_calls,
            "downstream_review_allowance": budget.review_model_calls,
            "downstream_organizer_cost_allowance": budget.organizer_cost_usd,
            "downstream_review_cost_allowance": budget.review_cost_usd,
            "unknown_cost_roles": list(budget.unknown_cost_roles),
        },
        "same_lifecycle_budget_requires_predeclared_allocation": True,
        "matched_downstream_proves": "conditional_reuse_only",
        "prerequisite_costs_separately_accounted": True,
        "downstream_allowances_are_separate": True,
        "free_seed": False,
        "p2_waste_padding": False,
        "note": (
            "matched downstream budget is conditional reuse only, not a full lifecycle budget; "
            "same-lifecycle comparison requires the separately predeclared lifecycle allocation "
            "including Seed and Organizer overhead"
        ),
    }


def observation_budget(observation: ProductObservation, profile: ProductProfile) -> dict:
    """Report actual resource compliance independently of the business result."""
    declared = {
        "model_calls": profile.budget.total_model_calls,
        "npu_runs": profile.budget.npu_runs,
        "total_wall_clock_sec": profile.budget.max_wall_clock_sec,
        "manual_interventions": profile.budget.human_interventions,
    }
    if profile.budget.max_cost_usd is not None:
        declared["estimated_cost_usd"] = profile.budget.max_cost_usd
    missing = [field for field in declared if getattr(observation, field) is None]
    overruns = [
        dict(task_id=observation.task_id, repetition=observation.repetition,
             field=field, observed=getattr(observation, field), declared=limit)
        for field, limit in declared.items()
        if getattr(observation, field) is not None and getattr(observation, field) > limit
    ]
    return dict(
        exceeded=True if overruns else None if missing else False,
        unmeasured_fields=missing, overruns=overruns,
    )


def build_product_report(
    plan: ProductMatrixPlan,
    observations: Sequence[ProductObservation],
) -> ProductMatrixReport:
    """Account every planned cell, including unobserved and failed attempts."""
    planned = {cell.key(): cell for cell in plan.cells}
    if len(planned) != len(plan.cells):
        raise ValueError("product plan contains duplicate cells")
    by_key = {}
    for observation in observations:
        key = observation.key()
        if key in by_key:
            raise ValueError(f"duplicate observation for product cell: {key}")
        if key not in planned:
            raise ValueError(f"observation is outside the product plan: {key}")
        by_key[key] = observation
    profiles = plan.profile_by_group()
    groups = {}
    for group in ALL_PRODUCT_GROUPS:
        profile = profiles.get(group)
        if profile is None:
            continue
        cells = [cell for cell in plan.cells if cell.group == group]
        present = _present(cells, by_key)
        groups[group.value] = ProductGroupReport(
            group,
            profile.profile_id,
            profile.experiment_profile,
            profile.protocol,
            profile.fidelity,
            profile.cohort,
            profile.conditions,
            profile.budget,
            _denominator([_status(cell, by_key) for cell in cells]),
            _bucket_denominators(cells, by_key, "task_family"),
            _bucket_denominators(cells, by_key, "hardware_tier"),
            tuple(cell.to_dict() for cell in cells if cell.key() not in by_key),
            present,
            p3_lineage=profile.p3_lineage.to_dict() if profile.p3_lineage else None,
            budget_overruns=tuple(
                item for observation in present
                for item in observation_budget(observation, profile)["overruns"]
            ),
            budget_contract=_budget_contract(profile),
            **_attempt_totals(present, len(cells)),
        )
    cohort_buckets = defaultdict(list)
    for cell in plan.cells:
        cohort_buckets[cell.cohort].append(_status(cell, by_key))
    matched_cohorts = defaultdict(list)
    for profile in plan.profiles:
        if profile.fidelity == "matched":
            matched_cohorts[profile.cohort].append(profile.group.value)
    all_observations = _present(plan.cells, by_key)
    return ProductMatrixReport(
        protocol=plan.protocol,
        plan_digest=plan.digest(),
        observations=all_observations,
        groups=groups,
        overall=_denominator([_status(cell, by_key) for cell in plan.cells]),
        matched_overall=_denominator(
            [_status(cell, by_key) for cell in plan.cells if profiles[cell.group].fidelity == "matched"]
        ),
        by_cohort={name: _denominator(statuses) for name, statuses in sorted(cohort_buckets.items())},
        matched_cohorts={
            name: tuple(sorted(groups)) for name, groups in sorted(matched_cohorts.items())
        },
        faithful_external_groups=tuple(
            sorted(
                profile.group.value
                for profile in plan.profiles
                if profile.fidelity == "faithful_external"
            )
        ),
    )


Executor = Callable[[ProductCell], ProductObservation | TrialReceipt | Mapping[str, Any] | None]


def execute_product_matrix(plan: ProductMatrixPlan, execute: Executor) -> ProductMatrixReport:
    """Run planned cells in interleaved order through a caller-owned executor.
    Returning `None` records a missing cell rather than inventing an outcome.
    """
    observations = []
    profiles = plan.profile_by_group()
    for cell in plan.cells:
        if cell.group == ProductGroup.P3_V003_FLYWHEEL:
            _admit_p3_profile(
                profiles[cell.group],
                cell.task_id,
                verify_artifacts=True,
                task_family=cell.task_family,
                hardware_tier=cell.hardware_tier,
            )
        result = execute(cell)
        if result is None:
            continue
        if isinstance(result, ProductObservation):
            observation = result
        elif isinstance(result, TrialReceipt):
            observation = observation_from_receipt(result, cell=cell)
        elif isinstance(result, Mapping):
            observation = ProductObservation(**result)
        else:
            raise TypeError("executor must return ProductObservation, TrialReceipt, mapping or None")
        if (
            observation.group != cell.group
            or observation.task_id != cell.task_id
            or observation.repetition != cell.repetition
        ):
            raise ValueError("executor returned an observation for a different planned cell")
        observations.append(observation)
    return build_product_report(plan, observations)


def pair_matched_observations(
    plan: ProductMatrixPlan,
    observations: Sequence[ProductObservation],
    left: ProductGroup | str,
    right: ProductGroup | str,
) -> tuple[MatchedPair, ...]:
    """Pair two matched groups only when their explicit protocols agree."""
    left_group = _as_product_group(left)
    right_group = _as_product_group(right)
    if left_group == right_group:
        raise ValueError("matched pairing requires two different product groups")
    profiles = plan.profile_by_group()
    if left_group not in profiles or right_group not in profiles:
        raise ValueError("both groups must be present in the product plan")
    left_profile = profiles[left_group]
    right_profile = profiles[right_group]
    if left_profile.fidelity != "matched" or right_profile.fidelity != "matched":
        raise ValueError(
            "faithful external groups are reported separately and cannot be silently paired"
        )
    if left_profile.cohort != right_profile.cohort:
        raise ValueError("matched pairing requires the same declared cohort")
    if left_profile.protocol != right_profile.protocol:
        raise ValueError("matched pairing requires the same execution protocol")
    if left_profile.comparison_fingerprint() != right_profile.comparison_fingerprint():
        raise ValueError("matched pairing requires identical conditions and total budget")
    planned_keys = {cell.key() for cell in plan.cells}
    by_key = {}
    for observation in observations:
        if observation.key() in by_key:
            raise ValueError(f"duplicate observation for product cell: {observation.key()}")
        if observation.key() not in planned_keys:
            raise ValueError(f"observation is outside the product plan: {observation.key()}")
        by_key[observation.key()] = observation
    result = []
    for cell in sorted(
        (cell for cell in plan.cells if cell.group == left_group),
        key=lambda cell: (cell.task_id, cell.repetition),
    ):
        right_key = (right_group, cell.task_id, cell.repetition)
        result.append(
            MatchedPair(
                task_id=cell.task_id,
                task_family=cell.task_family,
                hardware_tier=cell.hardware_tier,
                repetition=cell.repetition,
                left=by_key.get(cell.key()),
                right=by_key.get(right_key),
            )
        )
    return tuple(result)
