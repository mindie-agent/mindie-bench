from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_bench.runner.compaction_injector import SemanticEvent
from mindie_bench.runner.mechanism_ablation import (
    MECHANISM_ABLATION_PROTOCOL,
    MECHANISM_FIELDS,
    AblationObservation,
    AblationOutcome,
    AblationTask,
    ArmConfig,
    ArmRole,
    FixedControls,
    Mechanism,
    PairSpec,
    analyze_mechanism_ablations,
    build_mechanism_pairs,
    execute_mechanism_ablation,
    plan_mechanism_ablation,
)


def controls(task_id: str = "task-a", **overrides) -> FixedControls:
    values = dict(
        model_id="model-a",
        inference_profile="temperature=0,max_tokens=4096",
        tool_digest="tools-sha",
        raw_knowledge_digest="raw-knowledge-sha",
        evidence_digest="evidence-sha",
        task_start_digest=f"start-{task_id}",
        semantic_event=SemanticEvent.NEGATIVE_RESULT,
        model_call_budget=120,
        npu_run_budget=4,
        max_wall_clock_sec=3600.0,
        knowledge_token_budget=800,
    )
    values.update(overrides)
    return FixedControls(**values)


def base_config() -> ArmConfig:
    return ArmConfig(
        roles="single_agent",
        ledger="absent",
        retrieval="raw",
        exploration="sequential",
    )


def specs() -> list[PairSpec]:
    base = base_config()
    return [
        PairSpec(
            "pair-roles",
            Mechanism.SINGLE_AGENT_VS_ROLES,
            base,
            replace(base, roles="role_split"),
        ),
        PairSpec(
            "pair-ledger",
            Mechanism.LEDGER,
            base,
            replace(base, ledger="two_layer"),
        ),
        PairSpec(
            "pair-retrieval",
            Mechanism.RETRIEVAL,
            base,
            replace(base, retrieval="graded_conditions"),
        ),
        PairSpec(
            "pair-parallel",
            Mechanism.PARALLEL_EXPLORATION,
            base,
            replace(base, exploration="parallel"),
        ),
    ]


def tasks() -> list[AblationTask]:
    return [
        AblationTask("task-a", "family-alpha", "A2"),
        AblationTask("task-b", "family-beta", "A3"),
        AblationTask("task-c", "family-alpha", "A3"),
    ]


def build_plan(task_list=None, *, repetitions: int = 1):
    task_list = task_list or tasks()
    pairs = build_mechanism_pairs(specs())
    return plan_mechanism_ablation(
        pairs,
        task_list,
        controls_by_task={task.task_id: controls(task.task_id) for task in task_list},
        repetitions=repetitions,
    )


def observation(cell, status: str = "pass", **overrides) -> AblationObservation:
    values = dict(
        pair_id=cell.pair_id,
        arm_role=cell.arm_role,
        task_id=cell.task_id,
        repetition=cell.repetition,
        status=status,
        completion_tokens=10,
        wall_clock_sec=100.0,
        npu_runs=1,
        knowledge_tokens=100,
        first_effective_action="inspect-ledger",
        controls_digest=cell.controls_digest,
    )
    values.update(overrides)
    return AblationObservation(**values)


def all_observations(plan) -> list[AblationObservation]:
    return [observation(cell) for cell in plan.cells]


def test_four_mechanisms_change_exactly_one_field_and_bind_controls_per_task():
    pairs = build_mechanism_pairs(specs())
    assert [pair.mechanism for pair in pairs] == [
        Mechanism.SINGLE_AGENT_VS_ROLES,
        Mechanism.LEDGER,
        Mechanism.RETRIEVAL,
        Mechanism.PARALLEL_EXPLORATION,
    ]
    assert {pair.mechanism: pair.control.changed_fields(pair.treatment) for pair in pairs} == {
        Mechanism.SINGLE_AGENT_VS_ROLES: ("roles",),
        Mechanism.LEDGER: ("ledger",),
        Mechanism.RETRIEVAL: ("retrieval",),
        Mechanism.PARALLEL_EXPLORATION: ("exploration",),
    }
    for pair in pairs:
        assert pair.changed_field == MECHANISM_FIELDS[pair.mechanism]
        assert pair.changed_field not in pair.held_constant

    base = base_config()
    invalid = PairSpec(
        "pair-two-changes",
        Mechanism.LEDGER,
        base,
        replace(base, roles="role_split", ledger="two_layer"),
    )
    with pytest.raises(ValueError, match="must change only"):
        build_mechanism_pairs([invalid], require_all=False)

    with pytest.raises(ValueError, match="missing pairs"):
        build_mechanism_pairs(specs()[:3])


def test_plan_binds_real_per_task_controls_instead_of_one_shared_template():
    task_list = tasks()
    plan = build_plan(task_list, repetitions=2)
    assert plan.protocol == MECHANISM_ABLATION_PROTOCOL
    assert len(plan.cells) == 48
    assert {cell.arm_role for cell in plan.cells} == {ArmRole.CONTROL, ArmRole.TREATMENT}
    assert len({cell.pair_id for cell in plan.cells}) == 4
    first_pair_cells = [cell for cell in plan.cells if cell.pair_id == "pair-roles"]
    assert [cell.arm_role for cell in first_pair_cells[:2]] == [
        ArmRole.TREATMENT,
        ArmRole.CONTROL,
    ]
    assert [cell.arm_role for cell in first_pair_cells[2:4]] == [
        ArmRole.CONTROL,
        ArmRole.TREATMENT,
    ]
    assert set(plan.controls_digests()) == {"task-a", "task-b", "task-c"}
    for cell in plan.cells:
        assert cell.controls_digest == plan.controls_digests()[cell.task_id]
        assert plan.controls_for(cell.task_id).task_start_digest == f"start-{cell.task_id}"
    assert plan.controls_for("task-a").model_id == "model-a"
    assert plan.controls_for("task-a").inference_profile == "temperature=0,max_tokens=4096"
    assert plan.controls_for("task-a").raw_knowledge_digest == "raw-knowledge-sha"
    assert plan.controls_for("task-a").semantic_event == SemanticEvent.NEGATIVE_RESULT

    with pytest.raises(ValueError, match="exactly the planned task IDs"):
        plan_mechanism_ablation(
            plan.pairs,
            task_list,
            controls_by_task={"task-a": controls("task-a")},
        )


def test_mismatched_controls_digest_is_rejected_and_absent_binding_is_insufficient():
    plan = build_plan(tasks()[:1])
    observations = all_observations(plan)
    mismatched = replace(
        observations[0],
        controls_digest=plan.controls_for("task-a").digest() + "-different",
    )
    with pytest.raises(ValueError, match="does not match the fixed controls"):
        analyze_mechanism_ablations(plan, [mismatched, *observations[1:]], min_independent_families=1)

    unbound = [replace(item, controls_digest=None) for item in observations]
    report = analyze_mechanism_ablations(plan, unbound, min_independent_families=1)
    assert all(pair.controls_binding_complete is False for pair in report.pairs)
    assert all(pair.attribution == "insufficient" for pair in report.pairs)
    assert any(
        "controls_digest is unbound" in item for pair in report.pairs for item in pair.limitations
    )


def test_analysis_keeps_missing_fail_infra_unknown_and_small_sample_limits():
    plan = build_plan(tasks()[:2])
    observations = []
    for cell in plan.cells:
        status = "pass"
        if cell.arm_role == ArmRole.TREATMENT:
            if cell.pair_id == "pair-roles" and cell.task_id == "task-b":
                status = "fail"
            elif cell.pair_id == "pair-ledger" and cell.task_id == "task-a":
                status = "infra"
            elif cell.pair_id == "pair-ledger" and cell.task_id == "task-b":
                status = "unknown"
            elif cell.pair_id == "pair-retrieval" and cell.task_id == "task-b":
                continue
        observations.append(observation(cell, status))

    report = analyze_mechanism_ablations(plan, observations, min_independent_families=3)
    overall = report.overall.to_dict()
    assert overall["planned"] == 16
    assert overall["observed"] == 15
    assert overall["missing"] == 1
    assert overall["pass"] == 12
    assert overall["fail"] == 1
    assert overall["infra"] == 1
    assert overall["unknown"] == 1
    assert overall["valid"] == 13
    assert len(report.observations) == 15

    by_mechanism = {pair.mechanism: pair for pair in report.pairs}
    roles = by_mechanism[Mechanism.SINGLE_AGENT_VS_ROLES]
    assert roles.paired_tasks == 2
    assert roles.paired_valid_outcomes == 2
    assert roles.paired_pass_delta == -1
    assert roles.small_sample is True
    assert roles.attribution == "insufficient"
    assert any("small sample" in limitation for limitation in roles.limitations)

    ledger = by_mechanism[Mechanism.LEDGER]
    assert ledger.combined.infra == 1
    assert ledger.combined.unknown == 1
    assert any("incomplete denominator" in item for item in ledger.limitations)

    retrieval = by_mechanism[Mechanism.RETRIEVAL]
    assert retrieval.combined.missing == 1
    assert retrieval.paired_tasks == 1
    assert any("missing, infra or unknown" in item for item in report.limitations)
    assert len(report.interactions) == 2
    assert report.independent_task_families == 2
    assert report.task_families == ("family-alpha", "family-beta")


def test_retrieval_output_volume_difference_is_observational_not_confounded():
    plan = build_plan(tasks())
    baseline = all_observations(plan)
    unequal = [
        replace(item, knowledge_tokens=400)
        if item.pair_id == "pair-retrieval" and item.arm_role == ArmRole.TREATMENT
        else item
        for item in baseline
    ]
    report = analyze_mechanism_ablations(plan, unequal, min_independent_families=2)
    retrieval = next(pair for pair in report.pairs if pair.mechanism == Mechanism.RETRIEVAL)
    assert retrieval.knowledge_tokens_control == 300
    assert retrieval.knowledge_tokens_treatment == 1200
    assert retrieval.knowledge_tokens_delta == 900
    assert retrieval.knowledge_token_sample_count == 3
    assert retrieval.knowledge_token_complete is True
    assert retrieval.attribution == "descriptive_only"
    assert not any("confounded" in item for item in retrieval.limitations)
    assert any("retrieval outcome" in item for item in retrieval.limitations)

    matched = analyze_mechanism_ablations(plan, baseline, min_independent_families=2)
    retrieval = next(pair for pair in matched.pairs if pair.mechanism == Mechanism.RETRIEVAL)
    assert retrieval.knowledge_tokens_delta == 0
    assert retrieval.attribution == "descriptive_only"


def test_token_and_wall_clock_totals_report_sample_counts_and_completeness():
    task_list = tasks()[:2]
    plan = build_plan(task_list)
    observations = all_observations(plan)
    first_treatment = next(
        cell
        for cell in plan.cells
        if cell.pair_id == "pair-ledger"
        and cell.arm_role == ArmRole.TREATMENT
        and cell.task_id == "task-a"
    )
    observations = [
        replace(item, completion_tokens=None, wall_clock_sec=None)
        if item.key() == first_treatment.key()
        else item
        for item in observations
    ]
    report = analyze_mechanism_ablations(plan, observations, min_independent_families=1)
    ledger = next(pair for pair in report.pairs if pair.mechanism == Mechanism.LEDGER)
    assert ledger.paired_tasks == 2
    assert ledger.token_sample_count == 1
    assert ledger.token_complete is False
    assert ledger.wall_clock_sample_count == 1
    assert ledger.wall_clock_complete is False
    assert ledger.control_tokens == 10
    assert ledger.treatment_tokens == 10
    assert any("token totals cover 1 of 2" in item for item in ledger.limitations)


def test_parallel_wall_clock_remains_end_to_end_and_is_not_a_component_metric():
    plan = build_plan(tasks())
    observations = []
    for item in all_observations(plan):
        if item.pair_id == "pair-parallel":
            item = replace(
                item,
                wall_clock_sec=50.0 if item.arm_role == ArmRole.TREATMENT else 100.0,
            )
        observations.append(item)
    report = analyze_mechanism_ablations(plan, observations, min_independent_families=2)
    parallel = next(pair for pair in report.pairs if pair.mechanism == Mechanism.PARALLEL_EXPLORATION)
    assert parallel.control_wall_clock_sec == 300
    assert parallel.treatment_wall_clock_sec == 150
    assert parallel.wall_clock_delta_sec == -150
    assert parallel.wall_clock_sample_count == 3
    assert parallel.wall_clock_complete is True
    assert parallel.attribution == "descriptive_only"
    assert any("not attributed to one component" in limitation for limitation in parallel.limitations)


def test_independent_family_count_and_a5_tier_are_reported_separately():
    task_list = [
        *tasks(),
        AblationTask("task-d", "family-gamma", "A5"),
    ]
    plan = build_plan(task_list)
    observations = all_observations(plan)
    report = analyze_mechanism_ablations(plan, observations, min_independent_families=2)
    assert report.independent_task_families == 3
    assert report.task_families == ("family-alpha", "family-beta", "family-gamma")
    assert set(report.by_task_family) == {"family-alpha", "family-beta", "family-gamma"}
    assert report.by_task_family["family-alpha"].planned == 16
    assert report.by_task_family["family-alpha"].observed == 16
    assert set(report.by_hardware_tier) == {"A2", "A3", "A5"}
    assert report.by_hardware_tier["A5"].planned == 8
    assert report.by_hardware_tier["A5"].passed == 8


def test_executor_runs_planned_order_and_none_stays_missing():
    plan = build_plan(tasks()[:1])
    seen = []
    skipped_order = plan.cells[3].order

    def execute(cell):
        seen.append(cell.order)
        if cell.order == skipped_order:
            return None
        return observation(cell)

    report = execute_mechanism_ablation(plan, execute, min_independent_families=2)
    assert seen == [cell.order for cell in plan.cells]
    assert report.overall.planned == 8
    assert report.overall.observed == 7
    assert report.overall.missing == 1
    assert report.overall.passed == 7
    assert report.overall.to_dict()["completion_rate"] == 1.0


def test_observation_and_plan_reject_out_of_plan_cells():
    plan = build_plan(tasks()[:1])
    bad = AblationObservation("pair-roles", ArmRole.CONTROL, "task-a", 2, AblationOutcome.PASS)
    with pytest.raises(ValueError, match="outside the ablation plan"):
        analyze_mechanism_ablations(plan, [bad], min_independent_families=1)


def test_fixed_model_budget_includes_child_organizer_and_review_calls():
    with pytest.raises(ValueError, match="must fit the total model-call budget"):
        replace(
            controls(),
            child_model_call_budget=120,
            organizer_model_call_budget=1,
        )


def test_repeated_attempts_do_not_inflate_independent_task_family_evidence():
    plan = build_plan(tasks()[:1], repetitions=8)
    report = analyze_mechanism_ablations(plan, all_observations(plan))
    for pair in report.pairs:
        assert pair.paired_valid_outcomes == 8
        assert pair.independent_paired_families == 1
        assert pair.small_sample is True
        assert pair.attribution == "insufficient"
    exported = report.to_dict()
    assert exported["overall"]["valid"] == 64
    assert exported["pairs"][0]["control"]["pass"] == 8


def test_executor_cannot_certify_unobserved_controls_or_partial_metric_coverage():
    plan = build_plan(tasks()[:2])

    def execute(cell):
        if cell.task_id == "task-b":
            return None
        return observation(cell, controls_digest=None)

    report = execute_mechanism_ablation(plan, execute, min_independent_families=1)
    for pair in report.pairs:
        assert pair.planned_pairs == 2
        assert pair.token_sample_count == 1
        assert pair.token_complete is False
        assert pair.wall_clock_complete is False
        assert pair.knowledge_token_complete is False
        assert pair.controls_binding_complete is False
        assert pair.attribution == "insufficient"
        assert all(item.controls_digest is None for item in pair.observations)
