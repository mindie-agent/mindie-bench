"""Product-level comparison matrix, distinct from the legacy component groups.

[Role]
Plan, interleave, execute and account P0-P3 product comparisons across the bare host,
v0.0.2, v0.0.3, seed-capture-evaluation and independently reported external
product profiles.

[Invariants]
- `ProductGroup` is a separate protocol (`mindie-bench-product-matrix/1`); it
  never maps or reinterprets `models.Group` G0..G3 component-increment results.
- Matched profiles in one cohort share model, inference, task start, oracle,
  input, knowledge cutoff, tool/persistence permissions and total budget.
- Faithful external profiles must declare a different execution protocol and
  are reported in their own columns; they are never silently paired or pooled.
- Every planned cell remains in the denominator. Pass, fail, infra, unknown and
  missing are separate counts, and completion is computed only over valid
  terminal outcomes.
- Only an explicitly supplied executor can run a cell. This module never
  starts a model, NPU or paid experiment on its own.

[Timeouts & Locks]
- Pure planning, validation and accounting; the caller owns execution timeouts.
  No locks or background workers are created here.

[Side-Effect Policy]
- No I/O, subprocess, network or filesystem mutation. The optional executor is
  a caller-provided callable invoked in the planned order.
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
        total_wall_clock_sec=primary.get("total_wall_clock_sec"),
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
    for cell in plan.cells:
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
