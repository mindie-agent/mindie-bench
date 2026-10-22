import json
import platform
import shutil

import pytest

from mindie_bench.io import digest, write_json
from mindie_bench.models import CorpusConfig, Group, TaskSpec
from mindie_bench.runner import product_executor as product_executor_module
from mindie_bench.runner.matrix import (
    BudgetEnvelope,
    MatchingConditions,
    MatrixTask,
    ProductGroup,
    ProductProfile,
    execute_product_matrix,
    plan_product_matrix,
)
from mindie_bench.runner.product_executor import ProductExecutionBinding, ProductExecutor
from mindie_bench.runner.product_inputs import derive_inputs, runtime_snapshot


def setup_product(case, tmp_path):
    # This is a local identity/process contract test, not a 3s preparation SLA.
    # The shared product deadline starts before native preflight, so the generic
    # 3s synthetic timeout can leave no positive agent window on this host.
    task_file = case["task"] / "task.toml"
    task_file.write_text(
        task_file.read_text().replace("\ntimeout_sec = 3\n", "\ntimeout_sec = 30\n")
    )
    task = TaskSpec.load(case["task"])
    derived = derive_inputs(
        {task.task.instance_id: case["task"]}, case["config"], "l2", knowledge_cutoff="2026-10-01"
    )
    manifest = tmp_path / "product-runtime.json"
    write_json(
        manifest,
        {
            "profile_id": "fresh-bare-host",
            "product_group": ProductGroup.P0_BARE_HOST,
            "config_sha256": digest(case["config"].model_dump(mode="json")),
            "scope": "Synthetic local process-contract test only",
            "derived_inputs": derived,
            "runtime_snapshot": runtime_snapshot(case["config"], Group.G0),
        },
    )
    profile = ProductProfile(
        profile_id="fresh-bare-host",
        group=ProductGroup.P0_BARE_HOST,
        experiment_profile=str(manifest),
        protocol="native-product-contract-test/1",
        fidelity="matched",
        cohort="synthetic",
        conditions=MatchingConditions(**derived["conditions"]),
        budget=BudgetEnvelope(**derived["budget"]),
    )
    binding = ProductExecutionBinding(
        profile,
        case["config"],
        Group.G0,
        manifest,
        "l2",
    )
    plan = plan_product_matrix([MatrixTask(task.task.instance_id, "fixture", "cpu")], [profile])
    root = tmp_path / "runs"
    root.mkdir()
    executor = ProductExecutor(
        {profile.profile_id: binding}, {task.task.instance_id: case["task"]}, root
    )
    return executor, plan, binding


@pytest.mark.skipif(
    platform.system() not in {"Darwin", "Linux"}
    or (platform.system() == "Linux" and not shutil.which("bwrap")),
    reason="native sandbox required",
)
def test_product_identity_is_bound_before_real_process_and_survives_receipt(case, tmp_path):
    executor, plan, binding = setup_product(case, tmp_path)
    report = execute_product_matrix(plan, executor)
    assert report.overall.passed == 1
    (trial_dir,) = executor.run_root.iterdir()
    started = json.loads((trial_dir / "attempt_started.json").read_text())
    receipt = json.loads((trial_dir / "trial_receipt.json").read_text())
    observed = json.loads((trial_dir / "product_observation.json").read_text())
    context = started["experiment_context"]
    assert context["cell"]["group"] == "P0_bare_host"
    assert receipt["environment_fingerprint"]["experiment_context"] == context
    assert observed["experiment_context"] == context
    assert observed["observation"]["receipt_id"] == receipt["trial_id"]
    assert (
        observed["observation"]["total_wall_clock_sec"]
        == receipt["primary_metrics"]["delivery_wall_clock_sec"]
    )
    assert receipt["ablation_group"] == "G0"


def test_changed_runtime_configuration_is_rejected_before_starting_an_attempt(case, tmp_path):
    executor, plan, _ = setup_product(case, tmp_path)
    case["config"].harness.model = "different-model"
    with pytest.raises(ValueError, match="model differs"):
        executor(plan.cells[0])
    assert list(executor.run_root.iterdir()) == []


def test_changed_selected_task_is_rejected_before_starting_a_product_attempt(case, tmp_path):
    executor, plan, _ = setup_product(case, tmp_path)
    with (case["task"] / "verifier/test.sh").open("a") as output:
        output.write("\nexit 0\n")
    with pytest.raises(ValueError, match="selected task changed"):
        executor(plan.cells[0])
    assert list(executor.run_root.iterdir()) == []


def test_shared_phase_budget_uses_real_owner_bounds_and_shared_allowance():
    from types import SimpleNamespace

    import pytest

    from mindie_bench.runner.orchestrator import (
        ProductDeadlineExhausted,
        product_execution_window,
    )
    from mindie_bench.runner.phase_budget import lifecycle_overrun, product_phase_budget

    task = SimpleNamespace(budget=SimpleNamespace(timeout_sec=100.0, verifier_timeout_sec=20.0))
    budget = product_phase_budget(
        100.0,
        20.0,
        remote=True,
        native_product=True,
        knowledge_sessions=8,
        capture_timeout_sec=None,
    )
    terms = budget["terms"]
    # True maximum branches: one monitor stop, 3 verifier-before observes,
    # 3 finish observes + 2 releases + 1 diagnostic, 8 knowledge closes.
    assert terms["agent_termination"] == 2 + 2 + 5 + 1 + 1
    assert terms["gateway_model_settlement"] == 30 + 0.25
    assert terms["verifier_before_observe"] == 3 * 70
    assert terms["component_monitor_stop"] == 75
    assert terms["component_finish_observe"] == 3 * 70
    assert terms["component_finish_release"] == 2 * 70
    assert terms["component_finish_diagnostic"] == 1 * 70
    assert terms["knowledge_close"] == 8 * 10.05
    assert terms["native_service_stop"] == 10
    assert budget["post_agent_reserve_sec"] == 993.65
    assert budget["total_delivery_sec"] == 1093.65
    assert "capture" not in terms

    context = {"max_wall_clock_sec": budget["total_delivery_sec"]}
    whole, used = product_execution_window(context, task, 0.0)
    after_preparation, _ = product_execution_window(context, task, 25.0)
    assert whole == 100.0
    assert after_preparation == 75.0
    # The reserve is the predeclared difference, not an arm-specific recompute.
    assert used["post_agent_reserve_sec"] == context["max_wall_clock_sec"] - 100.0
    with pytest.raises(ProductDeadlineExhausted):
        product_execution_window(context, task, 100.0)

    overtime = lifecycle_overrun(budget["total_delivery_sec"], budget["total_delivery_sec"] + 7.5)
    assert overtime["overrun"] is True and overtime["overrun_sec"] == 7.5
    inside = lifecycle_overrun(budget["total_delivery_sec"], budget["total_delivery_sec"] - 1.0)
    assert inside["overrun"] is False and inside["overrun_sec"] == 0.0

    with_capture = product_phase_budget(
        100.0, 20.0, remote=True, native_product=True, knowledge_sessions=8, capture_timeout_sec=120.0
    )
    assert with_capture["total_delivery_sec"] - budget["total_delivery_sec"] == 120.0 + 60.25


def test_derived_product_budget_is_the_shared_phase_total(case, tmp_path):
    executor, _, _ = setup_product(case, tmp_path)
    manifest = json.loads((executor.run_root.parent / "product-runtime.json").read_text())
    basis = manifest["derived_inputs"]["budget_basis"]
    declared = manifest["derived_inputs"]["budget"]
    assert basis["contract"] == "phase-budget/2"
    assert declared["max_wall_clock_sec"] == basis["total_delivery_sec"]
    # Local fixture: model/remote settlement, agent termination, gateway join, verifier.
    assert basis["post_agent_reserve_sec"] == 30.25 + 30 + 11 + 2 + 3 + 10
    assert basis["total_delivery_sec"] == 30 + basis["post_agent_reserve_sec"]


def test_preparation_deadline_failure_is_infra_not_agent_unknown():
    from mindie_bench.models import TerminalStatus
    from mindie_bench.runner.orchestrator import ProductDeadlineExhausted, failure_status

    # Fixture-F1: the deadline check runs in preparation, not after the agent
    # stage marker. Only an actual agent-stage failure may become UNKNOWN.
    assert failure_status(ProductDeadlineExhausted("deadline"), "preparation") is TerminalStatus.INFRA
    assert failure_status(RuntimeError("profile"), "preparation") is TerminalStatus.INFRA
    assert failure_status(RuntimeError("native"), "agent") is TerminalStatus.UNKNOWN


def test_binding_prepares_static_runtime_once_and_two_cells_reuse_it(
    case, tmp_path, monkeypatch
):
    calls = []
    real_runtime = product_executor_module.runtime_snapshot

    def runtime(*args, **kwargs):
        calls.append(1)
        return real_runtime(*args, **kwargs)

    monkeypatch.setattr(product_executor_module, "runtime_snapshot", runtime)
    executor, plan, binding = setup_product(case, tmp_path)
    assert len(calls) == 1
    task_id = plan.cells[0].task_id
    two_cells = plan_product_matrix(
        [MatrixTask(task_id, "fixture", "cpu")], [binding.profile], repetitions=2
    ).cells
    for cell in two_cells:
        binding.context(cell, case["task"])
    assert len(calls) == 1


def test_p1_p2_different_corpora_same_common_cutoff_form_matched_cells(case):
    p1_config = case["config"].model_copy(deep=True)
    p2_config = case["config"].model_copy(deep=True)
    p1_config.corpora = {"p1-old": CorpusConfig(path="/corpus/p1", commit="a" * 40)}
    p2_config.corpora = {"p2-new": CorpusConfig(path="/corpus/p2", commit="b" * 40)}
    task_id = TaskSpec.load(case["task"]).task.instance_id
    p1 = derive_inputs(
        {task_id: case["task"]}, p1_config, "l2", knowledge_cutoff="2026-10-01"
    )
    p2 = derive_inputs(
        {task_id: case["task"]}, p2_config, "l2", knowledge_cutoff="2026-10-01"
    )
    assert p1["corpora"] != p2["corpora"]
    assert p1["conditions"] == p2["conditions"]
    assert p1["conditions"]["knowledge_cutoff"] == "2026-10-01"
    profiles = [
        ProductProfile(
            profile_id=f"corpus-{group.value}",
            group=group,
            experiment_profile=f"profiles/{group.value}.json",
            protocol="matched-product/1",
            fidelity="matched",
            cohort="corpus-revisions",
            conditions=MatchingConditions(**derived["conditions"]),
            budget=BudgetEnvelope(**derived["budget"]),
        )
        for group, derived in ((ProductGroup.P1_V002_BASELINE, p1), (ProductGroup.P2_V003_FULL, p2))
    ]
    plan = plan_product_matrix([MatrixTask(task_id, "fixture", "cpu")], profiles)
    assert {cell.group for cell in plan.cells} == {
        ProductGroup.P1_V002_BASELINE,
        ProductGroup.P2_V003_FULL,
    }
    changed_budget = p2["budget"] | {"total_model_calls": p2["budget"]["total_model_calls"] + 1}
    with pytest.raises(ValueError, match="budget"):
        plan_product_matrix(
            [MatrixTask(task_id, "fixture", "cpu")],
            [
                profiles[0],
                ProductProfile(
                    profile_id="corpus-P2_v003_full",
                    group=ProductGroup.P2_V003_FULL,
                    experiment_profile="profiles/P2_v003_full.json",
                    protocol="matched-product/1",
                    fidelity="matched",
                    cohort="corpus-revisions",
                    conditions=MatchingConditions(**p1["conditions"]),
                    budget=BudgetEnvelope(**changed_budget),
                ),
            ],
        )
