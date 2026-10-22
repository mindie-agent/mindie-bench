"""Four controlled mechanism ablations with explicit one-factor pairing.

[Role]
Plan and account matched pairs for roles, ledger, retrieval and parallel
exploration while holding raw knowledge, evidence, tools, model, task and budget.

[Invariants]
- Each pair changes exactly one mechanism field; every other setting and fixed input matches both arms.
- Raw knowledge, evidence, tools, model identity and budget are fixed inputs and
  are bound by a per-task `controls_digest`. A differing digest is rejected; a
  missing digest makes attribution insufficient, never silently matched.
- Delivered knowledge tokens, output tokens and wall clock are observed outcomes
  of a retrieval/exploration strategy, not inputs that must be equal. They are
  reported with their sample counts and completeness, not labelled confounded.
- Total wall clock is end-to-end; it is never decomposed into a single component.
- One-factor pairs do not estimate interactions. Pair assessments retain small
  sample limits and leave factor interactions explicitly unestimated.
- Independent task-family count is distinct from task/repetition count. Family
  and hardware tier (including A5 when present) are reported in separate buckets.
- Missing, infra and unknown cells remain in every denominator. Only valid
  terminal outcomes enter pass-rate comparisons.

[Timeouts & Locks]
- Pure in-memory planning/accounting. The caller-owned executor owns timeouts;
  this module creates no worker, lock or background task.

[Side-Effect Policy]
- No I/O, subprocess, network, model request or NPU run. Optional execution is a
  caller-provided callable in the planned order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from mindie_bench.runner.mechanism_contracts import (
    _OUTCOME_ALIASES as _OUTCOME_ALIASES,
)
from mindie_bench.runner.mechanism_contracts import (
    CONFIG_FIELDS as CONFIG_FIELDS,
)
from mindie_bench.runner.mechanism_contracts import (
    MECHANISM_ABLATION_PROTOCOL as MECHANISM_ABLATION_PROTOCOL,
)
from mindie_bench.runner.mechanism_contracts import (
    MECHANISM_FIELDS as MECHANISM_FIELDS,
)
from mindie_bench.runner.mechanism_contracts import (
    ALL_MECHANISMS as ALL_MECHANISMS,
)
from mindie_bench.runner.mechanism_contracts import (
    MECHANISMS as MECHANISMS,
)
from mindie_bench.runner.mechanism_contracts import (
    AblationCell as AblationCell,
)
from mindie_bench.runner.mechanism_contracts import (
    AblationDenominator as AblationDenominator,
)
from mindie_bench.runner.mechanism_contracts import (
    AblationObservation as AblationObservation,
)
from mindie_bench.runner.mechanism_contracts import (
    AblationOutcome as AblationOutcome,
)
from mindie_bench.runner.mechanism_contracts import (
    AblationTask as AblationTask,
)
from mindie_bench.runner.mechanism_contracts import (
    ArmConfig as ArmConfig,
)
from mindie_bench.runner.mechanism_contracts import (
    ArmRole as ArmRole,
)
from mindie_bench.runner.mechanism_contracts import (
    FixedControls as FixedControls,
)
from mindie_bench.runner.mechanism_contracts import (
    Mechanism as Mechanism,
)
from mindie_bench.runner.mechanism_contracts import (
    MechanismAblationPlan as MechanismAblationPlan,
)
from mindie_bench.runner.mechanism_contracts import (
    MechanismAblationReport as MechanismAblationReport,
)
from mindie_bench.runner.mechanism_contracts import (
    MechanismPair as MechanismPair,
)
from mindie_bench.runner.mechanism_contracts import (
    PairAssessment as PairAssessment,
)
from mindie_bench.runner.mechanism_contracts import (
    PairSpec as PairSpec,
)
from mindie_bench.runner.mechanism_contracts import (
    TaskControls as TaskControls,
)
from mindie_bench.runner.mechanism_contracts import (
    _normalize_outcome as _normalize_outcome,
)


def build_mechanism_pairs(
    specs: Sequence[PairSpec],
    *,
    require_all: bool = True,
) -> tuple[MechanismPair, ...]:
    """Validate the one-factor design and return pairs in canonical order."""
    if not specs:
        raise ValueError("at least one mechanism pair is required")
    pair_ids = [spec.pair_id for spec in specs]
    mechanisms = [spec.mechanism for spec in specs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("mechanism pair IDs must be unique")
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError("each mechanism can have only one controlled pair")
    if require_all and set(mechanisms) != set(MECHANISMS):
        missing = sorted(mechanism.value for mechanism in set(MECHANISMS) - set(mechanisms))
        raise ValueError("mechanism ablation is missing pairs: " + ", ".join(missing))
    pairs = [
        MechanismPair(
            pair_id=spec.pair_id,
            mechanism=spec.mechanism,
            control=spec.control,
            treatment=spec.treatment,
            control_id=spec.control_id,
            treatment_id=spec.treatment_id,
        )
        for spec in specs
    ]
    return tuple(sorted(pairs, key=lambda pair: ALL_MECHANISMS.index(pair.mechanism)))


def plan_mechanism_ablation(
    pairs: Sequence[MechanismPair],
    tasks: Sequence[AblationTask],
    *,
    controls_by_task: Mapping[str, FixedControls],
    repetitions: int = 1,
    require_all_mechanisms: bool = True,
) -> MechanismAblationPlan:
    """Plan an interleaved one-factor ablation with a real controls binding per task."""
    if not pairs:
        raise ValueError("at least one mechanism pair is required")
    if not tasks:
        raise ValueError("at least one ablation task is required")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    mechanisms = {pair.mechanism for pair in pairs}
    if require_all_mechanisms and not set(MECHANISMS).issubset(mechanisms):
        raise ValueError("ablation plan must contain all four mechanism pairs")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("ablation task IDs must be unique")
    if set(controls_by_task) != set(task_ids):
        raise ValueError("controls_by_task must bind exactly the planned task IDs")
    if any(not isinstance(controls, FixedControls) for controls in controls_by_task.values()):
        raise TypeError("every task must bind FixedControls")
    ordered_pairs = sorted(pairs, key=lambda pair: (ALL_MECHANISMS.index(pair.mechanism), pair.pair_id))
    ordered_tasks = sorted(tasks, key=lambda task: task.task_id)
    task_controls = tuple(
        TaskControls(task_id=task.task_id, controls=controls_by_task[task.task_id])
        for task in ordered_tasks
    )
    controls_digests = {item.task_id: item.digest() for item in task_controls}
    cells = []
    order = 0
    for repetition in range(1, repetitions + 1):
        for task_index, task in enumerate(ordered_tasks):
            for pair_index, pair in enumerate(ordered_pairs):
                arms = [
                    (ArmRole.CONTROL, pair.control_id),
                    (ArmRole.TREATMENT, pair.treatment_id),
                ]
                # Alternate arm order so neither arm systematically runs first.
                if (repetition + task_index + pair_index) % 2:
                    arms.reverse()
                for arm_role, arm_id in arms:
                    cells.append(
                        AblationCell(
                            pair_id=pair.pair_id,
                            mechanism=pair.mechanism,
                            arm_role=arm_role,
                            arm_id=arm_id,
                            task_id=task.task_id,
                            task_family=task.task_family,
                            hardware_tier=task.hardware_tier,
                            repetition=repetition,
                            order=order,
                            controls_digest=controls_digests[task.task_id],
                        )
                    )
                    order += 1
    return MechanismAblationPlan(
        protocol=MECHANISM_ABLATION_PROTOCOL,
        pairs=tuple(ordered_pairs),
        tasks=tuple(ordered_tasks),
        repetitions=repetitions,
        controls_by_task=task_controls,
        cells=tuple(cells),
    )


def _denominator(statuses: Sequence[AblationOutcome | None]) -> AblationDenominator:
    counts = {
        AblationOutcome.PASS: 0,
        AblationOutcome.FAIL: 0,
        AblationOutcome.INFRA: 0,
        AblationOutcome.UNKNOWN: 0,
    }
    observed = 0
    missing = 0
    for status in statuses:
        if status is None:
            missing += 1
            continue
        observed += 1
        counts[status] += 1
    return AblationDenominator(
        planned=len(statuses),
        observed=observed,
        missing=missing,
        passed=counts[AblationOutcome.PASS],
        failed=counts[AblationOutcome.FAIL],
        infra=counts[AblationOutcome.INFRA],
        unknown=counts[AblationOutcome.UNKNOWN],
    )


def _bucket_denominators(
    cells: Sequence[AblationCell],
    observations: Mapping[tuple[str, ArmRole, str, int], AblationObservation],
    attribute: str,
) -> dict[str, AblationDenominator]:
    buckets = defaultdict(list)
    for cell in cells:
        status = observations[cell.key()].status if cell.key() in observations else None
        buckets[getattr(cell, attribute)].append(status)
    return {name: _denominator(statuses) for name, statuses in sorted(buckets.items())}


def _pair_assessment(
    plan: MechanismAblationPlan,
    pair: MechanismPair,
    observations: Mapping[tuple[str, ArmRole, str, int], AblationObservation],
    *,
    min_independent_families: int,
) -> PairAssessment:
    cells = [cell for cell in plan.cells if cell.pair_id == pair.pair_id]
    control_cells = [cell for cell in cells if cell.arm_role == ArmRole.CONTROL]
    treatment_cells = [cell for cell in cells if cell.arm_role == ArmRole.TREATMENT]

    def status(cell: AblationCell) -> AblationOutcome | None:
        observation = observations.get(cell.key())
        return observation.status if observation else None

    control_denominator = _denominator([status(cell) for cell in control_cells])
    treatment_denominator = _denominator([status(cell) for cell in treatment_cells])
    combined = _denominator([status(cell) for cell in cells])
    control_by_task = {
        (cell.task_id, cell.repetition): observations[cell.key()]
        for cell in control_cells
        if cell.key() in observations
    }
    treatment_by_task = {
        (cell.task_id, cell.repetition): observations[cell.key()]
        for cell in treatment_cells
        if cell.key() in observations
    }
    paired = [
        (control_by_task[key], treatment_by_task[key])
        for key in sorted(set(control_by_task) & set(treatment_by_task))
    ]
    valid_pairs = [
        (control, treatment)
        for control, treatment in paired
        if control.status in {AblationOutcome.PASS, AblationOutcome.FAIL}
        and treatment.status in {AblationOutcome.PASS, AblationOutcome.FAIL}
    ]
    pass_delta = (
        sum(treatment.status == AblationOutcome.PASS for _, treatment in valid_pairs)
        - sum(control.status == AblationOutcome.PASS for control, _ in valid_pairs)
        if valid_pairs
        else None
    )
    token_pairs = [
        (control.completion_tokens, treatment.completion_tokens)
        for control, treatment in paired
        if control.completion_tokens is not None and treatment.completion_tokens is not None
    ]
    control_tokens = sum(control for control, _ in token_pairs) if token_pairs else None
    treatment_tokens = sum(treatment for _, treatment in token_pairs) if token_pairs else None
    wall_pairs = [
        (control.wall_clock_sec, treatment.wall_clock_sec)
        for control, treatment in paired
        if control.wall_clock_sec is not None and treatment.wall_clock_sec is not None
    ]
    control_wall = sum(control for control, _ in wall_pairs) if wall_pairs else None
    treatment_wall = sum(treatment for _, treatment in wall_pairs) if wall_pairs else None
    knowledge_pairs = [
        (control.knowledge_tokens, treatment.knowledge_tokens)
        for control, treatment in paired
        if control.knowledge_tokens is not None and treatment.knowledge_tokens is not None
    ]
    knowledge_control = sum(control for control, _ in knowledge_pairs) if knowledge_pairs else None
    knowledge_treatment = sum(treatment for _, treatment in knowledge_pairs) if knowledge_pairs else None
    pair_cells = [cell for cell in cells if cell.key() in observations]
    controls_binding_complete = bool(pair_cells) and all(
        observations[cell.key()].controls_digest == cell.controls_digest for cell in pair_cells
    )
    family_by_task = {task.task_id: task.task_family for task in plan.tasks}
    valid_families = {family_by_task[control.task_id] for control, _ in valid_pairs}
    small_sample = len(valid_families) < min_independent_families
    limitations = []
    if small_sample:
        limitations.append(
            f"small sample: {len(valid_families)} independent paired task families "
            f"< {min_independent_families}; {len(valid_pairs)} paired valid attempts"
        )
    incomplete = combined.missing + combined.infra + combined.unknown
    if incomplete:
        limitations.append(
            f"incomplete denominator: {combined.missing} missing, {combined.infra} infra, "
            f"{combined.unknown} unknown remain in the comparison"
        )
    if not controls_binding_complete:
        limitations.append(
            "controls_digest is unbound for at least one observed arm; fixed-input match is unverified"
        )
    if len(token_pairs) < len(paired):
        limitations.append(f"token totals cover {len(token_pairs)} of {len(paired)} paired cells")
    if len(wall_pairs) < len(paired):
        limitations.append(f"wall-clock totals cover {len(wall_pairs)} of {len(paired)} paired cells")
    if len(knowledge_pairs) < len(paired):
        limitations.append(
            f"delivered-knowledge totals cover {len(knowledge_pairs)} of {len(paired)} paired cells"
        )
    limitations.append(
        "token and wall-clock deltas are whole-pair totals, not single-component attribution"
    )
    if pair.mechanism == Mechanism.RETRIEVAL:
        limitations.append(
            "delivered knowledge tokens are a retrieval outcome; raw knowledge digest and "
            "knowledge-token budget are fixed inputs and unequal output is not a confound"
        )
    if pair.mechanism == Mechanism.PARALLEL_EXPLORATION:
        limitations.append("wall clock is end-to-end; the difference is not attributed to one component")
    if small_sample or incomplete or not controls_binding_complete:
        attribution: Literal["descriptive_only", "insufficient"] = "insufficient"
    else:
        attribution = "descriptive_only"
    interactions = tuple(
        f"{pair.changed_field} interaction with {name} is not estimated by this pair"
        for name in pair.held_constant
    )
    return PairAssessment(
        pair_id=pair.pair_id,
        mechanism=pair.mechanism,
        changed_field=pair.changed_field,
        held_constant=pair.held_constant,
        controls_digest_by_task={cell.task_id: cell.controls_digest for cell in cells},
        controls_binding_complete=controls_binding_complete,
        control=control_denominator,
        treatment=treatment_denominator,
        combined=combined,
        planned_pairs=len(control_cells),
        paired_tasks=len(paired),
        independent_paired_families=len(valid_families),
        paired_valid_outcomes=len(valid_pairs),
        paired_pass_delta=pass_delta,
        control_tokens=control_tokens,
        treatment_tokens=treatment_tokens,
        token_delta=(
            treatment_tokens - control_tokens
            if control_tokens is not None and treatment_tokens is not None
            else None
        ),
        token_sample_count=len(token_pairs),
        token_complete=len(control_cells) > 0 and len(token_pairs) == len(control_cells),
        control_wall_clock_sec=control_wall,
        treatment_wall_clock_sec=treatment_wall,
        wall_clock_delta_sec=(
            treatment_wall - control_wall
            if control_wall is not None and treatment_wall is not None
            else None
        ),
        wall_clock_sample_count=len(wall_pairs),
        wall_clock_complete=len(control_cells) > 0 and len(wall_pairs) == len(control_cells),
        knowledge_tokens_control=knowledge_control,
        knowledge_tokens_treatment=knowledge_treatment,
        knowledge_tokens_delta=(
            knowledge_treatment - knowledge_control
            if knowledge_control is not None and knowledge_treatment is not None
            else None
        ),
        knowledge_token_sample_count=len(knowledge_pairs),
        knowledge_token_complete=len(control_cells) > 0 and len(knowledge_pairs) == len(control_cells),
        observations=tuple(observations[cell.key()] for cell in cells if cell.key() in observations),
        small_sample=small_sample,
        attribution=attribution,
        limitations=tuple(limitations),
        interactions=interactions,
    )


def analyze_mechanism_ablations(
    plan: MechanismAblationPlan,
    observations: Sequence[AblationObservation],
    *,
    min_independent_families: int = 3,
) -> MechanismAblationReport:
    """Account paired outcomes without dropping failures, unknowns or missing cells."""
    if type(min_independent_families) is not int or min_independent_families < 1:
        raise ValueError("min_independent_families must be a positive integer")
    planned = {cell.key(): cell for cell in plan.cells}
    if len(planned) != len(plan.cells):
        raise ValueError("mechanism plan contains duplicate cells")
    by_key = {}
    for observation in observations:
        if observation.key() in by_key:
            raise ValueError(f"duplicate observation for ablation cell: {observation.key()}")
        if observation.key() not in planned:
            raise ValueError(f"observation is outside the ablation plan: {observation.key()}")
        expected_digest = planned[observation.key()].controls_digest
        if observation.controls_digest is not None and observation.controls_digest != expected_digest:
            raise ValueError(
                f"observation controls_digest for {observation.key()} does not match the "
                "fixed controls bound to that task"
            )
        by_key[observation.key()] = observation
    assessments = []
    for pair in plan.pairs:
        assessments.append(
            _pair_assessment(
                plan,
                pair,
                by_key,
                min_independent_families=min_independent_families,
            )
        )
    overall = _denominator(
        [by_key[cell.key()].status if cell.key() in by_key else None for cell in plan.cells]
    )
    families = sorted({task.task_family for task in plan.tasks})
    interactions = (
        "one-factor pairs do not identify interactions among roles, ledger, retrieval and exploration",
        "each effect is conditional on fixed raw knowledge, evidence, tools, model/inference, task start and budget",
    )
    limitations = []
    if any(assessment.small_sample for assessment in assessments):
        limitations.append(
            f"at least one pair has fewer than {min_independent_families} independent paired task families"
        )
    if any(
        assessment.combined.missing + assessment.combined.infra + assessment.combined.unknown
        for assessment in assessments
    ):
        limitations.append("missing, infra or unknown cells remain in at least one denominator")
    if any(assessment.controls_binding_complete is False for assessment in assessments):
        limitations.append("at least one observed arm is missing its controls_digest binding")
    return MechanismAblationReport(
        protocol=plan.protocol,
        plan_digest=plan.digest(),
        controls_digest_by_task=plan.controls_digests(),
        pairs=tuple(assessments),
        overall=overall,
        observations=tuple(by_key[cell.key()] for cell in plan.cells if cell.key() in by_key),
        task_families=tuple(families),
        independent_task_families=len(families),
        by_task_family=_bucket_denominators(plan.cells, by_key, "task_family"),
        by_hardware_tier=_bucket_denominators(plan.cells, by_key, "hardware_tier"),
        interactions=interactions,
        limitations=tuple(limitations),
    )


AblationExecutor = Callable[[AblationCell], AblationObservation | Mapping[str, Any] | None]


def execute_mechanism_ablation(
    plan: MechanismAblationPlan,
    execute: AblationExecutor,
    *,
    min_independent_families: int = 3,
) -> MechanismAblationReport:
    """Run planned ablation cells through a caller-owned executor."""
    observations = []
    for cell in plan.cells:
        result = execute(cell)
        if result is None:
            continue
        if isinstance(result, AblationObservation):
            observation = result
        elif isinstance(result, Mapping):
            observation = AblationObservation(**result)
        else:
            raise TypeError("executor must return AblationObservation, mapping or None")
        if observation.key() != cell.key():
            raise ValueError("executor returned an observation for a different planned cell")
        observations.append(observation)
    return analyze_mechanism_ablations(
        plan, observations, min_independent_families=min_independent_families
    )
