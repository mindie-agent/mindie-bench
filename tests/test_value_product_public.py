from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from mindie_bench.io import digest, read_json, write_json
from mindie_bench.runner.matrix import ProductProfile
from mindie_bench.runner.matrix_contracts import BudgetEnvelope, MatchingConditions
from mindie_bench.value_product import (
    build_product_plan,
    collect_product,
    evaluate_product,
    verify_publication_product,
)

BENCHMARK = {
    "version": "0.3.0",
    "revision": "c" * 40,
    "source_sha256": "d" * 64,
}
FORMAL = {
    "diagnostics": ("0.4.3", "1b5a0eeb79d498c39002fc1853271c17e0f26e66"),
    "knowledge": ("0.12.2", "4e328305323eafc7fd43ecc0c53eb3cc0541fdef"),
    "remote-dev": ("0.10.1", "a6649222aae4851deec5774513f8319aece09051"),
    "coordinator": ("0.5.3", "34812c2d385f3ff112be3a597d27fd3a0c15c104"),
    "npu-top": ("0.1.9", "9bbf6de590b12fe1109040e0ef86817bd2c5f3e0"),
    "codex": ("0.0.2", "f1acb69ba336f75ae2d909443ca118835b038b9c"),
    "claude-code": ("0.0.2", "78f34dd9b09e877298225fa243ea4c28978c9dd6"),
    "kimi": ("0.0.2", "0e78f0422162a61e8c56850e049f9e5fc4d290b5"),
    "dsh": ("0.0.2", "d3a1e1282356b65b51fba49072c1842918746f8f"),
    "mindie-bench": ("0.3.0", "68d90d7a689f836c61a64ab054a0282e7361fd16"),
}
REPOSITORIES = {
    "diagnostics": "mindie-agent/diagnostics",
    "knowledge": "mindie-agent/knowledge",
    "remote-dev": "mindie-agent/remote-dev",
    "coordinator": "mindie-agent/coordinator",
    "npu-top": "mindie-agent/npu-top",
    "codex": "mindie-agent/mindie-agent-codex",
    "claude-code": "mindie-agent/mindie-agent-cc",
    "kimi": "mindie-agent/mindie-agent-kimi",
    "dsh": "mindie-agent/mindie-agent-dsh",
    "mindie-bench": "mindie-agent/mindie-bench",
}
PRODUCT_DEPENDENCIES = ["knowledge", "remote-dev", "diagnostics"]
INSTRUMENTATION = ["coordinator", "npu-top"]
RUNTIME_MODULES = ["knowledge", "remote-dev", "diagnostics", "npu-top", "coordinator"]

from test_value_product import (  # noqa: E402
    _sha,
    cohort_budget,
    cohort_conditions,
    collected,
    collector,
    make_suite,
    plugin_identity,
    receipt,
    runtime_manifest,
    runtime_modules,
    source,
    task,
    write_evidence,
)


def test_valid_slower_flat_product_matrix_uses_separate_identities_and_budgets(tmp_path):
    suite, inputs, extra, suite_path, inputs_path, report = collected(tmp_path)
    assert report["release_gate"], report["blocking_reasons"]
    cohort = report["cohorts"]["fused-shared-budget"]
    assert set(cohort["matrix"]["groups"]) == {"P0_bare_host", "P1_v002_baseline", "P2_v003_full"}
    assert cohort["matrix"]["groups"]["P2_v003_full"]["budget"]["max_wall_clock_sec"] == 3361.65
    assert report["final_release"]["status"] == "product_release_candidate"
    assert "published_at" not in report["final_release"]["provenance"]
    assert report["final_release"]["revision"] == (
        report["final_release"]["native_plugin"]["revision"]
    )
    assert report["final_release"]["artifact_sha256"] == (
        report["final_release"]["native_plugin"]["artifact_sha256"]
    )
    assert report["final_release"]["combination_manifest_sha256"] == (
        report["final_release"]["provenance"]["source_manifest_sha256"]
    )
    assert report["formal_baseline"]["product_version"] == "0.0.2"
    assert report["final_release"]["product_version"] == "0.0.3"
    assert report["formal_baseline"]["revision"] != report["final_release"]["revision"]
    sources = {item["group"]: item["source"] for item in report["profiles"].values()}
    assert set(sources["P1_v002_baseline"]["runtime_modules"]) == set(RUNTIME_MODULES)
    assert len(sources["P2_v003_full"]["runtime_modules"]) == 5
    assert sources["P1_v002_baseline"]["runtime_modules"]["coordinator"] == (
        sources["P2_v003_full"]["runtime_modules"]["coordinator"]
    )
    assert sources["P1_v002_baseline"]["benchmark"] == sources["P2_v003_full"]["benchmark"]
    assert sources["P1_v002_baseline"]["runtime_modules"]["knowledge"] != (
        sources["P2_v003_full"]["runtime_modules"]["knowledge"]
    )
    assert report["preparation"][-1]["status"] == "not_usable"
    # Every comparison is slower than its baseline (<1.0x); that is still publishable.
    speedups = [item["wall_clock_speedup_on_mutual_success"] for item in report["comparisons"]]
    assert all(value is not None and value < 1.0 for value in speedups)
    serialized = json.dumps(report)
    assert "/private/" not in serialized
    assert "private-service" not in serialized

def test_over_budget_success_and_slowdown_survive_product_reports_and_public_rebuild(tmp_path):
    from mindie_bench.value_product_views import (
        component_markdown_product,
        component_view_product,
        homepage_markdown_product,
        markdown_product,
    )

    suite, inputs, _ = make_suite(tmp_path)
    for name in inputs["receipts"]:
        path = tmp_path / name
        raw = read_json(path)
        if raw["environment_fingerprint"]["experiment_context"]["cell"]["group"] == "P2_v003_full":
            raw["primary_metrics"]["delivery_wall_clock_sec"] = 4000
            raw["auxiliary_metrics"].update(
                agent_budget_exhausted=True,
                lifecycle_budget=dict(
                    overrun=True, declared_total_sec=3361.65, measured_total_sec=4000,
                    overrun_sec=638.35,
                ),
            )
            write_json(path, raw)
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "out", collector_source=collector(),
    )
    assert report["release_gate"], report["blocking_reasons"]
    comparison = next(c for c in report["comparisons"] if c["candidate_group"] == "P2_v003_full")
    assert comparison["denominator"]["mutually_solved"] > 0
    assert comparison["wall_clock_speedup_on_mutual_success"] == pytest.approx(110 / 4000)
    budget = comparison["budget_observations"]["candidate"]
    assert budget["completed_pass_over_budget"] == comparison["denominator"]["mutually_solved"]
    assert budget["exceeded_attempts"] == budget["observed_attempts"]
    assert budget["observations"][0]["overruns"][0]["observed"] == 4000
    group = report["cohorts"]["fused-shared-budget"]["matrix"]["groups"]["P2_v003_full"]
    assert group["denominator"]["pass"] > 0
    assert group["budget_overruns"][0]["declared"] == 3361.65
    assert "Successful overruns" in markdown_product(report)
    assert "超预算次数" in homepage_markdown_product(report, language="zh")
    target = next(item for item in report["final_release"]["repositories"]
                  if item["repository"] == "mindie-agent/mindie-bench")
    component = component_markdown_product(component_view_product(report, target))
    assert "Over-budget attempts (baseline/candidate)" in component
    assert f"| 0/{budget['exceeded_attempts']} | measured |" in component
    evidence = read_json(tmp_path / "out/value-evidence.json")
    public = next(row for row in evidence["receipts"] if row["auxiliary_metrics"].get("agent_budget_exhausted"))
    assert public["terminal_status"] == "completed_pass"
    assert public["auxiliary_metrics"]["lifecycle_budget"]["overrun"] is True
    rebuilt = evaluate_product(read_json(tmp_path / "out/value-suite.json"), evidence)
    assert rebuilt["comparisons"] == report["comparisons"]


def test_two_scenarios_keep_their_own_3361_65_and_2401_65_budgets(tmp_path):
    specs = [
        {
            "id": "fused-shared-budget",
            "scenario": "fixed_case_optimization",
            "task": task(
                "fused-rmsnorm-gate-dispatch-performance",
                family="performance", digest_value=_sha("task:fused"), oracle="1" * 64,
                max_turns=40, npu_runs=8, timeout=1800.0, verifier=600.0, wall=3361.65,
            ),
        },
        {
            "id": "idle-gdn-shared-budget",
            "scenario": "problem_diagnosis",
            "task": task(
                "k3-gdn-idle-dp-state",
                family="diagnosis", digest_value=_sha("task:gdn"), oracle="2" * 64,
                max_turns=40, npu_runs=8, timeout=1200.0, verifier=240.0, wall=2401.65,
            ),
        },
    ]
    suite, inputs, extra, suite_path, inputs_path, report = collected(tmp_path, cohorts_spec=specs)
    assert report["release_gate"], report["blocking_reasons"]
    assert set(report["cohorts"]) == {"fused-shared-budget", "idle-gdn-shared-budget"}
    assert report["cohorts"]["fused-shared-budget"]["matrix"]["groups"]["P0_bare_host"]["budget"]["max_wall_clock_sec"] == 3361.65
    assert report["cohorts"]["idle-gdn-shared-budget"]["matrix"]["groups"]["P0_bare_host"]["budget"]["max_wall_clock_sec"] == 2401.65
    assert set(report["plan_digests"]) == set(report["cohorts"])
    rendered = (tmp_path / "out/value-results.md").read_text()
    assert "fused-shared-budget" in rendered and "idle-gdn-shared-budget" in rendered
    from mindie_bench.value_publish import write_component_views

    manifest = {
        "product_version": "0.0.3",
        "repositories": [
            {"repository": target["repository"], "version": target["version"], "revision": target["revision"]}
            for target in report["final_release"]["repositories"]
        ],
    }
    files = write_component_views(report, manifest, tmp_path / "views")
    assert files
    assert "idle-gdn-shared-budget" in (
        tmp_path / "views/knowledge/product-0.0.3-value-results.md"
    ).read_text()


def test_distinct_p1_p2_and_candidate_provenance_are_enforced(tmp_path):
    suite, inputs, extra = make_suite(tmp_path)
    same = copy.deepcopy(suite)
    same["final_release"]["repositories"][0]["revision"] = FORMAL["diagnostics"][1]
    with pytest.raises(ValueError, match="combination manifest digest"):
        build_product_plan(same)
    changed_plugin = copy.deepcopy(suite)
    changed_plugin["final_release"]["native_plugin"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="product revision must be the runnable DSH plugin"):
        build_product_plan(changed_plugin)
    formal_kind = copy.deepcopy(suite)
    formal_kind["final_release"]["provenance"]["kind"] = "formal_release"
    with pytest.raises(ValueError, match="product release candidate"):
        build_product_plan(formal_kind)
    published = copy.deepcopy(suite)
    published["final_release"]["provenance"]["published_at"] = "2026-10-20T00:00:00Z"
    with pytest.raises(ValueError, match="must not claim a publication time"):
        build_product_plan(published)
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "out",
        collector_source=collector(),
    )
    release_manifest = {
        "schema_version": "mindie-release-manifest/1",
        "product_version": "0.0.3",
        "repositories": [
            {"repository": target["repository"], "version": target["version"], "revision": target["revision"]}
            for target in report["final_release"]["repositories"]
        ],
    }
    manifest_path = tmp_path / "release-manifest.json"
    write_json(manifest_path, release_manifest)
    evidence_path = tmp_path / "out/value-evidence.json"
    verified = verify_publication_product(
        suite=suite_path, evidence=evidence_path, manifest=manifest_path,
        collector_source=collector(),
    )
    assert verified["release_gate"]
    changed = copy.deepcopy(release_manifest)
    changed["repositories"][0]["revision"] = "0" * 40
    write_json(manifest_path, changed)
    with pytest.raises(ValueError, match="candidate release source differ"):
        verify_publication_product(
            suite=suite_path, evidence=evidence_path, manifest=manifest_path,
            collector_source=collector(),
        )


def test_unusable_old_preparation_cannot_back_a_current_p1_source(tmp_path):
    suite, inputs, extra = make_suite(tmp_path)
    suite["cohorts"][0]["profiles"][1]["source"]["preparation_id"] = "old-p1-observed"
    with pytest.raises(ValueError, match="unusable preparation evidence"):
        build_product_plan(suite)


def test_historical_old_runner_unknown_keeps_original_identity_and_cost(tmp_path):
    suite, inputs, extra, suite_path, inputs_path, report = collected(tmp_path)
    historical = copy.deepcopy(extra["receipts"][0])
    historical.update(
        trial_id="l2-summary-model-catalog_G2_2_6a7c127708f7",
        instance_id="l2-summary-model-catalog",
        repetition=2,
        terminal_status="unknown",
        failure_reason="uncertain_external_effects",
    )
    historical["primary_metrics"].update(
        reward=None,
        resolved=None,
        cost_complete=False,
        estimated_cost_usd=None,
        model_calls=30,
        total_wall_clock_sec=86.41823729203315,
        delivery_wall_clock_sec=292.0006795419613,
        npu_test_runs=None,
        tokens={
            "input_uncached": 93917,
            "input_cached": 1108864,
            "cache_creation": 0,
            "output": 8628,
            "indexer_small_model_total": 0,
            "complete": False,
        },
    )
    historical["environment_fingerprint"].update(
        benchmark_version="0.3.0",
        benchmark_source_sha256="89d827fff805197f59983324a3350dfc2033e4df78a320e2f86cbb78fd02f155",
        component_revisions={
            "coordinator": "daf7fa38b431187c04a6dea4341df4a066100953",
            "diagnostics": "1b5a0eeb79d498c39002fc1853271c17e0f26e66",
            "dsh": "d3a1e1282356b65b51fba49072c1842918746f8f",
            "knowledge": "4e328305323eafc7fd43ecc0c53eb3cc0541fdef",
            "mindie-bench": "03e74644d4d2a118ebad5d6d10db67b0ea989b75",
            "npu-top": "9bbf6de590b12fe1109040e0ef86817bd2c5f3e0",
            "remote-dev": "a6649222aae4851deec5774513f8319aece09051",
        },
        budget={
            "max_npu_runs": 12, "max_output_bytes": 33554432,
            "max_turns": 35, "timeout_sec": 600.0, "verifier_timeout_sec": 90.0,
        },
    )
    historical["environment_fingerprint"].pop("experiment_context", None)
    old_path = tmp_path / "historical.json"
    write_json(old_path, historical)
    inputs["excluded_attempts"] = [
        {"reason": "Historical TLS unknown retained", "path": old_path.name}
    ]
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "with-history",
        collector_source=collector(),
    )
    assert report["release_gate"], report["blocking_reasons"]
    assert report["attempts"]["historical"] == 1
    assert report["attempts"]["all_executed"] == 4
    record = report["historical_attempts"][0]
    assert record["receipt"]["trial_id"] == "l2-summary-model-catalog_G2_2_6a7c127708f7"
    assert record["benchmark_runner"]["version"] == "0.3.0"
    assert record["benchmark_runner"]["revision"] == "03e74644d4d2a118ebad5d6d10db67b0ea989b75"
    byte_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
    canonical_sha = digest(read_json(old_path))
    assert byte_sha != canonical_sha
    assert record["source_receipt_sha256"] == byte_sha
    assert record["canonical_receipt_sha256"] == canonical_sha
    assert report["resources"]["cost_complete"] is False
    assert report["resources"]["observed_cost_usd_lower_bound"] is not None
    # Selected cells still reject an old runner.
    changed = read_json(tmp_path / inputs["receipts"][0])
    changed["environment_fingerprint"]["benchmark_version"] = "0.1.0"
    write_json(tmp_path / inputs["receipts"][0], changed)
    with pytest.raises(ValueError, match="common benchmark runner differs"):
        collect_product(
            suite=suite_path, inputs=inputs_path, output=tmp_path / "strict",
            collector_source=collector(),
        )


def test_required_functional_claim_blocks_all_failure_but_allows_flat_or_slower(tmp_path):
    suite, inputs, extra = make_suite(tmp_path)
    # Default fixture is slower/flat but has a P2 pass; it must publish.
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    base = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "ok",
        collector_source=collector(),
    )
    assert base["release_gate"]
    # Flip the P2 functional arm to failure; P1/P0 passes cannot replace it.
    p2_path = next(
        tmp_path / name for name in inputs["receipts"]
        if read_json(tmp_path / name)["environment_fingerprint"]["experiment_context"]["cell"]["group"]
        == "P2_v003_full"
    )
    failed = read_json(p2_path)
    failed.update(terminal_status="completed_fail", failure_reason="oracle_failed")
    failed["primary_metrics"].update(reward=0.0, resolved=False)
    write_json(p2_path, failed)
    blocked = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "blocked",
        collector_source=collector(),
    )
    assert not blocked["release_gate"]
    assert "functional_claim_failed:fused-shared-budget" in blocked["blocking_reasons"]


def test_missing_cell_duplicate_and_changed_binding_are_rejected(tmp_path):
    suite, inputs, extra = make_suite(tmp_path)
    missing = copy.deepcopy(inputs)
    missing["receipts"] = missing["receipts"][:-1]
    suite_path, inputs_path = write_evidence(tmp_path, suite, missing)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "missing",
        collector_source=collector(),
    )
    assert not report["release_gate"]
    assert report["cohorts"]["fused-shared-budget"]["matrix"]["overall"]["planned"] == 3
    assert report["cohorts"]["fused-shared-budget"]["matrix"]["overall"]["missing"] == 1
    duplicate = copy.deepcopy(inputs)
    duplicate["receipts"].append(duplicate["receipts"][0])
    suite_path, inputs_path = write_evidence(tmp_path, suite, duplicate)
    with pytest.raises(ValueError, match="counted more than once"):
        collect_product(
            suite=suite_path, inputs=inputs_path, output=tmp_path / "duplicate",
            collector_source=collector(),
        )
    changed = read_json(tmp_path / inputs["receipts"][0])
    changed["environment_fingerprint"]["base_model"] = "different-model"
    write_json(tmp_path / inputs["receipts"][0], changed)
    with pytest.raises(ValueError, match="model or common benchmark runner differs"):
        collect_product(
            suite=suite_path, inputs=inputs_path, output=tmp_path / "changed",
            collector_source=collector(),
        )
    changed["environment_fingerprint"]["base_model"] = "deepseek-flash"
    changed["environment_fingerprint"]["experiment_context"]["cell"]["group"] = "P2_v003_full"
    write_json(tmp_path / inputs["receipts"][0], changed)
    with pytest.raises(ValueError, match="different product plan cell|different product binding"):
        collect_product(
            suite=suite_path, inputs=inputs_path, output=tmp_path / "binding",
            collector_source=collector(),
        )


def test_component_runner_keeps_accepted_sha_distinct_from_common_runner(tmp_path):
    suite, inputs, extra = make_suite(tmp_path)
    accepted = {
        "revision": "0a3c9fd81fb37a1b4a52807d35509daca777ebf2",
        "version": "0.2.0",
        "source_sha256": "7" * 64,
        "dirty": False,
    }
    case = {
        "id": "coordinator-contention",
        "component": "coordinator",
        "status": "pass",
        "scope": "Accepted component functional scenario.",
        "baseline": "uncoordinated",
        "candidate": "coordinator",
        "repetitions": 1,
        "limitations": ["Functional check only."],
        "measurements": [{"repetition": 1, "status": "pass", "baseline": {}, "candidate": {}}],
        "source_revisions": {
            "mindie-coordinator": {
                "version": FORMAL["coordinator"][0],
                "revision": extra["releases"]["final"]["coordinator"][1],
                "editable": False,
            }
        },
    }
    report_path = tmp_path / "component-report.json"
    write_json(report_path, {"cases": [case], "runner_source": accepted, "runner_source_after": accepted})
    suite["component_cases"] = [{
        "id": "coordinator-contention", "repetitions": 1,
        "distribution": "mindie-coordinator", "component": "coordinator",
    }]
    inputs["component_reports"] = [report_path.name]
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "out",
        collector_source=collector(),
    )
    assert report["release_gate"], report["blocking_reasons"]
    assert report["component_runner_reports"][0]["runner_source"]["revision"] == accepted["revision"]
    assert accepted["revision"] != report["benchmark"]["revision"]


def test_required_p3_missing_blocks_and_real_public_collect_verify(tmp_path):
    from test_p3_lineage import build_lineage

    from mindie_bench.runner.matrix import ProductCell
    from mindie_bench.runner.p3_lineage import P3LineageBinding

    root, manifest = build_lineage(tmp_path / "p3-chain")
    fused_task = task(
        "fused-rmsnorm-gate-dispatch-performance", family="performance",
        digest_value=_sha("task:fused"), oracle="1" * 64,
        max_turns=40, npu_runs=8, timeout=1800.0, verifier=600.0, wall=3361.65,
    )
    p3_task = {
        "task_id": manifest.target.task_id,
        "task_family": manifest.target.matrix_task_family,
        "hardware_tier": manifest.target.hardware_tier,
        "task_digest": manifest.downstream.task_sha256,
        "oracle_digest": manifest.downstream.oracle_sha256,
        "execution_tier": "l3",
        "role": "eval",
        "source_budget": copy.deepcopy(fused_task["source_budget"]),
    }
    spec = {
        "id": "fused-shared-budget",
        "scenario": "fixed_case_optimization",
        "task": fused_task,
        "extra_tasks": [p3_task],
    }
    suite, inputs, extra = make_suite(tmp_path, cohorts_spec=[spec])
    cohort_conditions_value = cohort_conditions([fused_task, p3_task])
    p3_source = source(
        group="P2_v003_full",
        modules=runtime_modules(extra["releases"]["final"]),
        plugin=plugin_identity("0.0.3", extra["releases"]["final"]["dsh"][1], "p2"),
        catalog_units=3,
        preparation_id="current-p2",
        releases=extra["releases"],
    )
    suite["p3"] = {
        "required": True,
        "scope": "Real closed-family downstream only.",
        "lineage_manifest": str(root / "missing-lineage.json"),
        "manifest_content_identity_sha256": manifest.content_identity(),
        "comparison_cohort": "fused-shared-budget",
        "profile": {
            "profile_id": manifest.target.profile_id,
            "protocol": manifest.target.protocol,
            "fidelity": "matched",
            "cohort": manifest.target.cohort,
            "conditions": cohort_conditions_value,
            "budget": cohort_budget(fused_task),
            "source": p3_source,
        },
        "task": p3_task,
        "repetition": 1,
        "order": 0,
    }
    suite["comparisons"].append({
        "id": "p2-p3", "cohort": "fused-shared-budget",
        "baseline": "P2_v003_full", "candidate": "P3_v003_flywheel",
        "required": True, "scope": "Conditional reuse only.",
    })
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    missing = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "missing-p3",
        collector_source=collector(),
    )
    assert not missing["release_gate"]
    assert missing["p3"]["status"] == "failed"
    assert "required_p3_lineage_or_observation_missing" in missing["blocking_reasons"]
    missing_evidence = read_json(tmp_path / "missing-p3/value-evidence.json")
    assert "/private/" not in json.dumps(missing_evidence)
    assert str(root) not in json.dumps(missing_evidence)

    # Local collect must verify original artifact bytes, not a declaration.
    capture_path = Path(manifest.capture.receipt_locator)
    original_capture = capture_path.read_bytes()
    capture_path.write_text("{}", encoding="utf-8")
    write_json(root / "lineage.json", manifest.model_dump(mode="json"))
    suite["p3"]["lineage_manifest"] = str(root / "lineage.json")
    tampered = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "tampered-p3",
        collector_source=collector(),
    )
    assert not tampered["release_gate"]
    assert tampered["p3"]["status"] == "failed"
    tampered_evidence = read_json(tmp_path / "tampered-p3/value-evidence.json")
    tampered_suite = read_json(tmp_path / "tampered-p3/value-suite.json")
    assert "p3_public_lineage" not in tampered_evidence
    assert "lineage_manifest" not in tampered_suite["p3"]
    assert tampered_suite["p3"]["public_lineage_evidence"] == (
        "value-evidence.json:p3_public_lineage"
    )
    capture_path.write_bytes(original_capture)

    # Real lineage: add the P3 receipt/runtime manifest and collect once.
    binding = P3LineageBinding.prepare(manifest)
    p3_profile = ProductProfile(
        profile_id=manifest.target.profile_id,
        group="P3_v003_flywheel",
        experiment_profile=manifest.content_identity(),
        protocol=manifest.target.protocol,
        fidelity="matched",
        cohort=manifest.target.cohort,
        conditions=MatchingConditions(**cohort_conditions_value),
        budget=BudgetEnvelope(**cohort_budget(fused_task)),
        p3_lineage=binding,
    )
    cell = ProductCell(
        "P3_v003_flywheel", p3_profile.profile_id, p3_profile.cohort,
        p3_profile.protocol, p3_task["task_id"], p3_task["task_family"],
        p3_task["hardware_tier"], 1, 0,
    )
    p3_manifest = runtime_manifest(
        {
            "profile_id": p3_profile.profile_id,
            "group": p3_profile.group.value,
            "conditions": p3_profile.conditions.to_dict(),
            "budget": p3_profile.budget.to_dict(),
        },
        p3_source, [p3_task], tmp_path,
    )
    p3_row = receipt(
        profile={
            "profile_id": p3_profile.profile_id, "group": p3_profile.group.value,
            "cohort": p3_profile.cohort, "protocol": p3_profile.protocol,
            "conditions": p3_profile.conditions.to_dict(),
            "budget": p3_profile.budget.to_dict(),
        },
        source_identity=p3_source, cell=cell, task_row=p3_task,
        manifest=p3_manifest,
        profile_document=p3_profile.to_dict(),
        delivery=130.0,
    )
    p3_row["environment_fingerprint"]["experiment_context"]["p3_admission"] = binding.admission
    p3_path = tmp_path / "receipt-p3.json"
    write_json(p3_path, p3_row)
    inputs["receipts"].append(p3_path.name)
    inputs["runtime_manifests"].append(f"runtime-{p3_profile.profile_id}.json")
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "real-out",
        collector_source=collector(),
    )
    assert report["release_gate"], report["blocking_reasons"]
    assert report["p3"]["status"] == "admitted"
    assert report["p3"]["public_evidence_only"] is True
    evidence = read_json(tmp_path / "real-out/value-evidence.json")
    public_suite = read_json(tmp_path / "real-out/value-suite.json")
    assert "p3_public_lineage" in evidence
    consumer = evidence["p3_public_lineage"]["public_receipts"]["consumer"]
    assert "raw_capture_receipt" not in consumer
    assert "raw_trial_receipt" not in consumer
    assert consumer["raw_capture_receipt_sha256"]
    assert consumer["raw_trial_receipt_sha256"]
    assert "lineage_manifest" not in public_suite["p3"]
    assert public_suite["p3"]["public_lineage_evidence"] == (
        "value-evidence.json:p3_public_lineage"
    )
    assert public_suite["p3"]["manifest_content_identity_sha256"] == manifest.content_identity()
    assert evidence["suite_identity"] == digest(public_suite)
    assert evidence["private_suite_sha256"] == hashlib.sha256(suite_path.read_bytes()).hexdigest()
    serialized = json.dumps(evidence)
    assert "/private/" not in serialized
    assert str(root) not in serialized
    assert "/Volumes" not in serialized
    assert "/private/" not in json.dumps(public_suite)
    assert str(root) not in json.dumps(public_suite)
    # Remove all private P3 originals; public CI verification uses the OUTPUT suite.
    shutil.move(str(root), str(tmp_path / "p3-private-removed"))
    recomputed = evaluate_product(public_suite, evidence)
    assert recomputed["release_gate"], recomputed["blocking_reasons"]
    assert recomputed["suite_identity"] == digest(public_suite)
    assert recomputed["private_suite_sha256"] == evidence["private_suite_sha256"]

    # Logical cross-binding: recomputing the outer public hash cannot hide a
    # changed token, family or timeline because public receipts and ledger are
    # independently re-checked.
    def resealed(mutate):
        candidate = copy.deepcopy(evidence)
        mutation = candidate["p3_public_lineage"]
        mutate(mutation)
        mutation["public_lineage_sha256"] = digest(
            {key: value for key, value in mutation.items() if key != "public_lineage_sha256"}
        )
        return candidate

    def changed_token(lineage):
        lineage["public_receipts"]["seed"]["primary_metrics"]["tokens"]["input_uncached"] += 1

    def changed_cost(lineage):
        lineage["public_receipts"]["review"]["estimated_cost_usd"] += 1.0

    def changed_downstream(lineage):
        admission = lineage["admission"]
        ledger = admission["cost_ledger"]
        downstream = ledger["roles"]["downstream"]
        downstream.update(
            basis="observed",
            complete=True,
            estimated_cost_usd=1.0,
            price_basis="priced",
            tokens_basis="complete",
            input_tokens=0,
            cached_input_tokens=0,
            cache_creation_tokens=0,
            output_tokens=0,
            tokens_total=0,
        )
        admission.update(
            lifecycle_cost_complete=True,
            lifecycle_claim_ready=True,
            unknown_usage=[],
            unpriced_usage=[],
            prerequisite_unknown_usage=[],
            limitations=[],
        )
        totals = {}
        for field in ("model_calls", "tokens_total", "estimated_cost_usd", "npu_runs"):
            values = [
                row.get(field)
                for row in ledger["roles"].values()
                if row.get(field) is not None
            ]
            totals[field] = sum(values) if values else None
        ledger["known_totals"] = totals
        ledger["public_ledger_sha256"] = digest(
            {
                "roles": ledger["roles"],
                "attribution": ledger["attribution"],
                "unknown_usage": ledger["unknown_usage"],
                "prerequisite_unknown_usage": ledger["prerequisite_unknown_usage"],
                "unpriced_usage": ledger["unpriced_usage"],
                "known_totals": ledger["known_totals"],
            }
        )

    def changed_family(lineage):
        lineage["public_receipts"]["review"]["seed_family_canonical"] = (
            lineage["public_receipts"]["review"]["heldout_family_canonical"]
        )

    def changed_timeline(lineage):
        lineage["admission"]["timeline"]["corpus_frozen_at"] = lineage["admission"][
            "timeline"
        ]["seed_completed_at"]

    for mutation, reason in (
        (changed_token, "seed."),
        (changed_cost, "review.estimated_cost_usd"),
        (changed_downstream, "p3_public_downstream_basis_invalid"),
        (changed_family, "p3_public_review_seed_family_mismatch"),
        (changed_timeline, "p3_public_absolute_timeline_order_invalid"),
    ):
        blocked = evaluate_product(public_suite, resealed(mutation))
        assert not blocked["release_gate"]
        reasons = [*blocked["blocking_reasons"], *blocked["p3"].get("reasons", [])]
        assert any(reason in item for item in reasons), reasons
    release_manifest = {
        "schema_version": "mindie-release-manifest/1",
        "product_version": "0.0.3",
        "repositories": [
            {"repository": target["repository"], "version": target["version"], "revision": target["revision"]}
            for target in recomputed["final_release"]["repositories"]
        ],
    }
    manifest_path = tmp_path / "release-manifest.json"
    write_json(manifest_path, release_manifest)
    verified = verify_publication_product(
        suite=tmp_path / "real-out/value-suite.json",
        evidence=tmp_path / "real-out/value-evidence.json",
        manifest=manifest_path, collector_source=collector(),
    )
    assert verified["release_gate"]


