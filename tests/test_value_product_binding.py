from __future__ import annotations

from test_value_product import (
    BENCHMARK,
    plugin_identity,
    preparation_entries,
    release_identities,
    runtime_modules,
    source,
)

from mindie_bench.io import digest, write_json
from mindie_bench.models import Group, TaskSpec, TrialReceipt
from mindie_bench.runner.matrix import ProductProfile
from mindie_bench.runner.matrix_contracts import BudgetEnvelope, MatchingConditions
from mindie_bench.runner.product_executor import ProductExecutionBinding
from mindie_bench.runner.product_inputs import derive_inputs, runtime_snapshot
from mindie_bench.value_product import evaluate_product, public_suite
from mindie_bench.value_product_public import (
    public_catalog,
    public_experiment_context,
    public_runtime_manifest,
)


def test_real_product_execution_binding_context_projects_and_binds(tmp_path, case):
    # Make the shared product window positive for this local identity-only test.
    task_file = case["task"] / "task.toml"
    task_file.write_text(task_file.read_text().replace("\ntimeout_sec = 3\n", "\ntimeout_sec = 30\n"))
    task = TaskSpec.load(case["task"])
    task_id = task.task.instance_id
    derived = derive_inputs(
        {task_id: case["task"]}, case["config"], "l2", knowledge_cutoff="2026-10-01"
    )
    manifest_path = tmp_path / "product-runtime.json"
    benchmark = dict(BENCHMARK)
    profile = ProductProfile(
        profile_id="real-binding-p0",
        group="P0_bare_host",
        experiment_profile=str(manifest_path),
        protocol="matched-product/1",
        fidelity="matched",
        cohort="real-binding-cohort",
        conditions=MatchingConditions(**derived["conditions"]),
        budget=BudgetEnvelope(**derived["budget"]),
    )
    raw_manifest = {
        "profile_id": profile.profile_id,
        "product_group": "P0_bare_host",
        "config_sha256": digest(case["config"].model_dump(mode="json")),
        "scope": "Real ProductExecutionBinding context contract test",
        "derived_inputs": derived,
        "runtime_snapshot": runtime_snapshot(case["config"], Group.G0),
        "native_plugin": None,
        "catalog": {"schema": "mindie-catalog/1", "units": 0, "sha256": "0" * 64},
    }
    write_json(manifest_path, raw_manifest)
    binding = ProductExecutionBinding(
        profile, case["config"], Group.G0, manifest_path, "l2"
    )
    from mindie_bench.runner.matrix import MatrixTask, plan_product_matrix

    cell = plan_product_matrix(
        [MatrixTask(task_id, "fixture", "cpu")], [profile]
    ).cells[0]
    context = binding.context(cell, case["task"])
    public_context = public_experiment_context(context)
    assert public_context["cell"]["profile_id"] == profile.profile_id
    assert public_context["profile_id"] == profile.profile_id
    assert public_context["cohort"] == profile.cohort
    assert public_context["protocol"] == profile.protocol
    assert "runtime_manifest" not in public_context
    assert "profile" not in public_context
    projected_manifest = public_runtime_manifest(raw_manifest, benchmark)
    assert projected_manifest["runtime_modules"] == {}
    assert projected_manifest["native_plugin"] is None

    releases = release_identities()
    preparation = preparation_entries(releases)
    p0_source = source(
        group="P0_bare_host", modules={}, plugin=None, catalog_units=0,
        preparation_id="current-p0", releases=releases,
    )
    p0_source["catalog"] = raw_manifest["catalog"]
    p0_source["catalog_sha256"] = digest(public_catalog(p0_source["catalog"]))
    p1_source = source(
        group="P1_v002_baseline", modules=runtime_modules(releases["formal"]),
        plugin=plugin_identity("0.0.2", releases["formal"]["dsh"][1], "p1"),
        catalog_units=3, preparation_id="current-p1", releases=releases,
    )
    p2_source = source(
        group="P2_v003_full", modules=runtime_modules(releases["final"]),
        plugin=plugin_identity("0.0.3", releases["final"]["dsh"][1], "p2"),
        catalog_units=3, preparation_id="current-p2", releases=releases,
    )
    task_row = {
        "task_id": task_id,
        "task_family": "fixture",
        "hardware_tier": "cpu",
        "task_digest": derived["tasks"][task_id]["task_sha256"],
        "oracle_digest": derived["tasks"][task_id]["oracle_sha256"],
        "execution_tier": "l2",
        "role": "eval",
        "source_budget": {
            "max_turns": derived["budget"]["total_model_calls"],
            "max_npu_runs": derived["budget"]["npu_runs"],
            "timeout_sec": task.budget.timeout_sec,
            "verifier_timeout_sec": task.budget.verifier_timeout_sec,
            "max_wall_clock_sec": derived["budget"]["max_wall_clock_sec"],
            "human_interventions": derived["budget"].get("human_interventions", 0),
        },
    }
    profiles = []
    for group, profile_id, identity in (
        ("P0_bare_host", "real-binding-p0", p0_source),
        ("P1_v002_baseline", "real-binding-p1", p1_source),
        ("P2_v003_full", "real-binding-p2", p2_source),
    ):
        profiles.append({
            "profile_id": profile_id,
            "group": group,
            "experiment_profile": str(manifest_path) if group == "P0_bare_host" else f"profiles/{profile_id}.json",
            "protocol": "matched-product/1",
            "fidelity": "matched",
            "cohort": "real-binding-cohort",
            "conditions": derived["conditions"],
            "budget": derived["budget"],
            "source": identity,
        })
    suite = {
        "schema_version": "mindie-value-suite/2",
        "id": "real-binding-context",
        "product_version": "0.0.3",
        "protocol": "matched-product/1",
        "benchmark": benchmark,
        "tasks": [task_row],
        "preparation": preparation,
        "formal_baseline": releases["formal_baseline"],
        "final_release": releases["final_release"],
        "cohorts": [{
            "id": "real-binding-cohort",
            "scenario": "real-binding-contract",
            "task_ids": [task_id],
            "repetitions": 1,
            "required": True,
            "functional_claim": {"group": "P2_v003_full", "minimum_passes": 1},
            "profiles": profiles,
        }],
        "comparisons": [
            {"id": "p0-p1", "cohort": "real-binding-cohort", "baseline": "P0_bare_host", "candidate": "P1_v002_baseline", "required": True},
            {"id": "p1-p2", "cohort": "real-binding-cohort", "baseline": "P1_v002_baseline", "candidate": "P2_v003_full", "required": True},
        ],
        "component_cases": [],
        "native_plugins": {},
        "limitations": [],
    }
    from mindie_bench.value_product_protocol import build_product_plan

    bundle = build_product_plan(suite)
    receipt = TrialReceipt(
        trial_id="real-binding-p0-trial",
        instance_id=task_id,
        repetition=1,
        tier="l2",
        ablation_group=Group.G0,
        knowledge_gradient="no_hit",
        task_digest=derived["tasks"][task_id]["task_sha256"],
        experiment_digest=raw_manifest["config_sha256"],
        environment_fingerprint={
            "host_os": "test",
            "host_arch": "arm64",
            "harness_name": "fixture",
            "harness_version": "fixture-agent 1.0",
            "harness_binary_sha256": "h" * 64,
            "base_model": derived["conditions"]["model_id"],
            "reasoning_effort": None,
            "provider_url": None,
            "backend": "fixture",
            "budget": task.budget.model_dump(),
            "base_commit": task.environment["l2"].base_commit,
            "npu_model": None,
            "npu_count": 0,
            "remote_environment": None,
            "benchmark_version": benchmark["version"],
            "benchmark_source_sha256": benchmark["source_sha256"],
            "component_revisions": {"mindie-bench": benchmark["revision"]},
            "experiment_context": context,
        },
        terminal_status="completed_pass",
        primary_metrics={
            "reward": 1.0,
            "resolved": True,
            "tokens": {
                "input_uncached": 1, "input_cached": 0, "cache_creation": 0,
                "output": 1, "indexer_small_model_total": 0, "complete": True,
            },
            "cost_complete": True,
            "estimated_cost_usd": 0.01,
            "model_calls": 1,
            "total_wall_clock_sec": 10.0,
            "delivery_wall_clock_sec": 11.0,
            "npu_test_runs": 0,
            "manual_interventions": 0,
        },
        evidence="synthetic",
    )
    evidence = {
        "schema_version": "mindie-value-evidence/2",
        "suite_identity": digest(public_suite(suite)),
        "private_suite_sha256": digest(suite),
        "plan_sha256": bundle["plan_sha256"],
        "formal_baseline_sha256": digest(bundle["formal_baseline"]),
        "final_release_sha256": digest(bundle["final_release"]),
        "benchmark": benchmark,
        "runner_release": {
            key: benchmark[key] for key in ("revision", "version", "source_sha256")
        },
        "component_revisions": {
            name: target["revision"] for name, target in bundle["final_repositories"].items()
        },
        "runtime_manifests": [projected_manifest],
        "receipts": [receipt.model_dump(mode="json")],
        "excluded_attempts": [],
        "component_cases": [],
        "native_plugins": [],
        "setup_reports": [],
    }
    report = evaluate_product(suite, evidence)
    # P1/P2 are intentionally absent; the real P0 context must still bind.
    assert report["cohorts"]["real-binding-cohort"]["matrix"]["groups"]["P0_bare_host"]["denominator"]["observed"] == 1
    assert not report["release_gate"]
    assert any(reason.startswith("missing_required_cell") for reason in report["blocking_reasons"])
