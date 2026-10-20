from __future__ import annotations

import pytest

from mindie_bench.models import Group
from mindie_bench.runner.matrix import (
    PRODUCT_MATRIX_PROTOCOL,
    BudgetEnvelope,
    MatchingConditions,
    MatrixTask,
    ProductGroup,
    ProductObservation,
    ProductProfile,
    build_product_report,
    execute_product_matrix,
    pair_matched_observations,
    plan_product_matrix,
)


def conditions(**overrides) -> MatchingConditions:
    values = dict(
        model_id="model-a",
        inference_profile="temperature=0,max_tokens=4096",
        task_start_digest="start-sha",
        oracle_digest="oracle-sha",
        input_digest="input-sha",
        knowledge_cutoff="2026-10-01",
        tool_permission_digest="tools-sha",
        persistence_permission_digest="persist-sha",
    )
    values.update(overrides)
    return MatchingConditions(**values)


def budget(**overrides) -> BudgetEnvelope:
    values = dict(
        total_model_calls=100,
        foreground_model_calls=40,
        child_model_calls=20,
        organizer_model_calls=20,
        review_model_calls=20,
        npu_runs=4,
        human_interventions=2,
        max_wall_clock_sec=3600.0,
    )
    values.update(overrides)
    return BudgetEnvelope(**values)


def profile(
    group: ProductGroup,
    *,
    fidelity: str = "matched",
    protocol: str = "matched-product/1",
    cohort: str = "main",
    experiment_profile: str | None = None,
    matching: MatchingConditions | None = None,
    envelope: BudgetEnvelope | None = None,
    capabilities: tuple[str, ...] = (),
) -> ProductProfile:
    return ProductProfile(
        profile_id=f"profile-{group.value}",
        group=group,
        experiment_profile=experiment_profile or f"experiments/{group.value.lower()}.toml",
        protocol=protocol,
        fidelity=fidelity,
        cohort=cohort,
        conditions=matching or conditions(),
        budget=envelope or budget(),
        preserved_capabilities=capabilities,
    )


def tasks() -> list[MatrixTask]:
    return [
        MatrixTask("task-a", "family-alpha", "A2"),
        MatrixTask("task-b", "family-beta", "A3"),
        MatrixTask("task-c", "family-alpha", "A3"),
    ]


def test_seven_product_columns_do_not_reinterpret_legacy_component_groups():
    with pytest.raises(TypeError, match="component-increment"):
        ProductProfile(
            profile_id="legacy",
            group=Group.G0,
            experiment_profile="experiments/legacy.toml",
            protocol="matched-product/1",
            fidelity="matched",
            cohort="main",
            conditions=conditions(),
            budget=budget(),
        )
    with pytest.raises(ValueError, match="independent profile"):
        profile(ProductGroup.P2_V003_FULL, experiment_profile="G2")


def test_total_model_budget_includes_child_organizer_and_review_calls():
    with pytest.raises(ValueError, match="total model budget"):
        budget(
            total_model_calls=50,
            foreground_model_calls=20,
            child_model_calls=20,
            organizer_model_calls=10,
            review_model_calls=10,
        )


def test_plan_interleaves_groups_and_keeps_every_interleaved_cell():
    plan = plan_product_matrix(
        tasks()[:2],
        [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL)],
        repetitions=2,
    )
    assert plan.protocol == PRODUCT_MATRIX_PROTOCOL
    assert len(plan.cells) == 8
    group_order = [cell.group for cell in plan.cells]
    assert group_order[:4] == [
        ProductGroup.P0_BARE_HOST,
        ProductGroup.P2_V003_FULL,
        ProductGroup.P2_V003_FULL,
        ProductGroup.P0_BARE_HOST,
    ]
    assert group_order.count(ProductGroup.P0_BARE_HOST) == 4
    assert group_order.count(ProductGroup.P2_V003_FULL) == 4
    for repetition in (1, 2):
        for task_id in ("task-a", "task-b"):
            selected = {
                cell.group
                for cell in plan.cells
                if cell.task_id == task_id and cell.repetition == repetition
            }
            assert selected == {ProductGroup.P0_BARE_HOST, ProductGroup.P2_V003_FULL}


def test_matched_cohort_rejects_different_inputs_and_common_total_budget():
    mismatched_cutoff = profile(
        ProductGroup.P2_V003_FULL, matching=conditions(knowledge_cutoff="2026-10-02")
    )
    with pytest.raises(ValueError, match="knowledge_cutoff"):
        plan_product_matrix(tasks(), [profile(ProductGroup.P0_BARE_HOST), mismatched_cutoff])

    # Foreground/organizer shares are mechanism choices; only the common total is matched.
    role_shift = profile(
        ProductGroup.P2_V003_FULL,
        envelope=budget(child_model_calls=11, organizer_model_calls=29),
    )
    plan_product_matrix(tasks(), [profile(ProductGroup.P0_BARE_HOST), role_shift])

    different_total = profile(ProductGroup.P2_V003_FULL, envelope=budget(total_model_calls=101))
    with pytest.raises(ValueError, match="budget.total_model_calls"):
        plan_product_matrix(tasks(), [profile(ProductGroup.P0_BARE_HOST), different_total])


def test_matched_cohort_requires_the_same_downstream_cost_cap():
    with pytest.raises(ValueError, match="max_cost_usd"):
        plan_product_matrix(
            tasks(),
            [
                profile(ProductGroup.P0_BARE_HOST, envelope=budget(max_cost_usd=2.0)),
                profile(ProductGroup.P2_V003_FULL, envelope=budget(max_cost_usd=3.0)),
            ],
        )


def test_faithful_external_group_requires_its_own_protocol_and_is_reported_separately():
    matched = [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL)]
    same_protocol_external = profile(
        ProductGroup.EXT_MODEL_AGENT,
        fidelity="faithful_external",
        protocol="matched-product/1",
        capabilities=("profiling", "native-memory"),
    )
    with pytest.raises(ValueError, match="different protocol"):
        plan_product_matrix(tasks(), [*matched, same_protocol_external])

    external = profile(
        ProductGroup.EXT_MODEL_AGENT,
        fidelity="faithful_external",
        protocol="model-agent-native/1",
        matching=conditions(model_id="external-original-model"),
        capabilities=("profiling", "native-memory"),
    )
    plan = plan_product_matrix(tasks(), [*matched, external])
    report = build_product_report(plan, [])
    assert report.faithful_external_groups == ("P_ext_model_agent",)
    assert report.matched_cohorts == {"main": ("P0_bare_host", "P2_v003_full")}
    assert "P_ext_model_agent" in report.groups
    assert report.groups["P_ext_model_agent"].protocol == "model-agent-native/1"
    with pytest.raises(ValueError, match="cannot be silently paired"):
        pair_matched_observations(plan, [], ProductGroup.P0_BARE_HOST, ProductGroup.EXT_MODEL_AGENT)


def test_report_keeps_pass_fail_infra_unknown_and_missing_in_every_denominator():
    plan = plan_product_matrix(
        tasks(), [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL)]
    )
    observations = [
        ProductObservation(
            ProductGroup.P0_BARE_HOST,
            "task-a",
            1,
            "pass",
            estimated_cost_usd=0.10,
            tokens_total=10,
            total_wall_clock_sec=1.0,
            npu_runs=1,
            manual_interventions=0,
        ),
        ProductObservation(
            ProductGroup.P0_BARE_HOST,
            "task-b",
            1,
            "fail",
            estimated_cost_usd=0.20,
            tokens_total=20,
            total_wall_clock_sec=2.0,
        ),
        ProductObservation(
            ProductGroup.P0_BARE_HOST, "task-c", 1, "infra_failed", estimated_cost_usd=0.30
        ),
        ProductObservation(
            ProductGroup.P2_V003_FULL,
            "task-a",
            1,
            "unknown",
            estimated_cost_usd=None,
            tokens_total=5,
        ),
        ProductObservation(
            ProductGroup.P2_V003_FULL, "task-b", 1, "completed_pass", estimated_cost_usd=0.40
        ),
    ]
    report = build_product_report(plan, observations)
    overall = report.overall.to_dict()
    assert overall == {
        "planned": 6,
        "observed": 5,
        "missing": 1,
        "pass": 2,
        "fail": 1,
        "infra": 1,
        "unknown": 1,
        "valid": 3,
        "completion_rate": pytest.approx(2 / 3),
    }
    g0 = report.groups["P0_bare_host"]
    assert g0.denominator.to_dict()["pass"] == 1
    assert g0.denominator.to_dict()["fail"] == 1
    assert g0.denominator.to_dict()["infra"] == 1
    assert g0.denominator.to_dict()["missing"] == 0
    assert g0.by_task_family["family-alpha"].to_dict()["pass"] == 1
    assert g0.by_task_family["family-beta"].to_dict()["fail"] == 1
    assert g0.by_hardware_tier["A2"].to_dict()["pass"] == 1
    g2 = report.groups["P2_v003_full"]
    assert g2.denominator.to_dict()["unknown"] == 1
    assert g2.denominator.to_dict()["missing"] == 1
    assert g2.missing_cells[0]["task_id"] == "task-c"
    assert report.by_cohort["main"].to_dict()["missing"] == 1
    assert report.to_dict()["overall"]["planned"] == 6
    exported = report.to_dict()
    assert exported["overall"] == overall
    assert exported["groups"]["P0_bare_host"]["denominator"] == g0.denominator.to_dict()
    assert exported["groups"]["P2_v003_full"]["by_task_family"]["family-alpha"]["valid"] == 0
    assert exported["groups"]["P0_bare_host"]["conditions_fingerprint"] == g0.conditions.fingerprint()

    # Every attempt with a measured value counts, including fail/infra/unknown.
    assert g0.cost_usd_total.sum_observed == pytest.approx(0.60)
    assert g0.cost_usd_total.available_count == 3
    assert g0.cost_usd_total.planned_count == 3
    assert g0.cost_usd_total.complete is True
    # A missing token measurement is absent, never imputed as zero.
    assert g0.tokens_total.sum_observed == 30
    assert g0.tokens_total.available_count == 2
    assert g0.tokens_total.planned_count == 3
    assert g0.tokens_total.complete is False
    assert g0.wall_clock_sec_total.sum_observed == pytest.approx(3.0)
    assert g0.npu_runs_total.sum_observed == 1
    assert g0.manual_interventions_total.sum_observed == 0
    assert g2.cost_usd_total.sum_observed == pytest.approx(0.40)
    assert g2.cost_usd_total.available_count == 1
    assert g2.cost_usd_total.planned_count == 3
    assert g2.cost_usd_total.complete is False
    assert [item.task_id for item in report.observations] == [
        "task-a",
        "task-a",
        "task-b",
        "task-b",
        "task-c",
    ]
    assert [item.task_id for item in g0.observations] == ["task-a", "task-b", "task-c"]


def test_executor_runs_interleaved_cells_and_none_remains_missing():
    plan = plan_product_matrix(
        tasks()[:2], [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL)]
    )
    seen = []

    def execute(cell):
        seen.append(cell.order)
        if cell.group == ProductGroup.P2_V003_FULL and cell.task_id == "task-b":
            return None
        return {
            "group": cell.group,
            "task_id": cell.task_id,
            "repetition": cell.repetition,
            "status": "completed_pass",
        }

    report = execute_product_matrix(plan, execute)
    assert seen == [cell.order for cell in plan.cells]
    assert report.overall.to_dict()["planned"] == 4
    assert report.overall.to_dict()["observed"] == 3
    assert report.overall.to_dict()["missing"] == 1
    assert report.groups["P2_v003_full"].missing_cells[0]["task_id"] == "task-b"


def test_matched_pairing_keeps_missing_sides_and_rejects_different_conditions():
    plan = plan_product_matrix(
        tasks()[:2], [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL)]
    )
    observations = [
        ProductObservation(ProductGroup.P0_BARE_HOST, "task-a", 1, "pass"),
        ProductObservation(ProductGroup.P2_V003_FULL, "task-a", 1, "fail"),
    ]
    pairs = pair_matched_observations(
        plan, observations, ProductGroup.P0_BARE_HOST, ProductGroup.P2_V003_FULL
    )
    assert len(pairs) == 2
    assert pairs[0].both_valid
    assert pairs[0].left.status.value == "pass"
    assert pairs[0].right.status.value == "fail"
    assert not pairs[1].both_observed

    other_cohort = profile(ProductGroup.P1_V002_BASELINE, cohort="other")
    two_cohorts = plan_product_matrix(
        tasks()[:1],
        [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL), other_cohort],
    )
    with pytest.raises(ValueError, match="same declared cohort"):
        pair_matched_observations(
            two_cohorts, [], ProductGroup.P0_BARE_HOST, ProductGroup.P1_V002_BASELINE
        )


def test_observation_rejects_legacy_group_instead_of_relabelling_it():
    with pytest.raises(TypeError, match="component-increment"):
        ProductObservation(Group.G2, "task-a", 1, "pass")


def test_receipt_binding_requires_an_explicit_product_cell_not_legacy_group():
    from mindie_bench.models import TerminalStatus, TrialReceipt

    plan = plan_product_matrix(
        tasks()[:1], [profile(ProductGroup.P0_BARE_HOST), profile(ProductGroup.P2_V003_FULL)]
    )
    cell = plan.cells[0]
    receipt = TrialReceipt(
        trial_id="trial-explicit",
        instance_id="task-a",
        repetition=1,
        tier="l2",
        ablation_group=Group.G0,
        knowledge_gradient="no_hit",
        task_digest="task-sha",
        experiment_digest="experiment-sha",
        environment_fingerprint={},
        terminal_status=TerminalStatus.PASS,
        primary_metrics={
            "reward": 1.0,
            "resolved": True,
            "tokens": {
                "input_uncached": 4,
                "input_cached": 2,
                "cache_creation": 1,
                "output": 3,
                "complete": True,
            },
            "cost_complete": True,
            "estimated_cost_usd": 0.25,
            "model_calls": 7,
            "total_wall_clock_sec": 12.5,
            "npu_test_runs": 2,
            "remote_test_invocations": 3,
            "zero_human_intervention": True,
        },
        evidence="candidate",
    )
    from mindie_bench.runner.matrix import observation_from_receipt

    observation = observation_from_receipt(receipt, cell=cell)
    assert observation.group == ProductGroup.P0_BARE_HOST
    assert observation.status.value == "pass"
    assert observation.tokens_total == 9
    assert observation.estimated_cost_usd == 0.25
    assert observation.total_wall_clock_sec == 12.5
    assert observation.npu_runs == 2
    assert observation.manual_interventions == 0

    # complete=True with a missing real token key is unobserved, not zero-filled.
    incomplete = receipt.model_copy(
        update={
            "primary_metrics": {
                **receipt.primary_metrics,
                "tokens": {"input_uncached": 4, "output": 3, "complete": True},
                "cost_complete": False,
                "total_wall_clock_sec": None,
            }
        }
    )
    incomplete_observation = observation_from_receipt(incomplete, cell=cell)
    assert incomplete_observation.tokens_total is None
    assert incomplete_observation.estimated_cost_usd is None
    assert incomplete_observation.total_wall_clock_sec is None

    unmeasured = receipt.model_copy(
        update={
            "primary_metrics": {
                key: value
                for key, value in receipt.primary_metrics.items()
                if key not in {"npu_test_runs", "remote_test_invocations", "zero_human_intervention"}
            }
        }
    )
    unmeasured_observation = observation_from_receipt(unmeasured, cell=cell)
    assert unmeasured_observation.npu_runs is None
    assert unmeasured_observation.manual_interventions is None
    # The cell, not receipt.ablation_group, selects the product column.
    g2_observation = observation_from_receipt(receipt, cell=plan.cells[1])
    assert g2_observation.group == ProductGroup.P2_V003_FULL
    different_task = receipt.model_copy(update={"instance_id": "task-b"})
    with pytest.raises(ValueError, match="task differs"):
        observation_from_receipt(different_task, cell=cell)
