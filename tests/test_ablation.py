from __future__ import annotations

import hashlib

import pytest

from mindie_bench.cli import main
from mindie_bench.evaluation.ablation import AblationFactor, AblationGate, AblationRun


def run(identity, factors, duration, **changes):
    values = dict(
        run_id=identity,
        active_factors=factors,
        conditions={"soc": "contract-fixture", "shape": "batch-16", "cann": "fixed"},
        run_status="passed",
        correctness_status="passed",
        measurement_valid=True,
        samples_ms=[duration, duration, duration, duration],
        evidence_refs=[f"runs/{identity}/receipt.json"],
    )
    values.update(changes)
    if values["samples_ms"]:
        case = hashlib.sha256(b"contract-fixture-case").hexdigest()
        values["timing"] = {
            "clock": "host_monotonic_ns",
            "synchronizer": "contract-fixture",
            "warmup_rounds": 1,
            "buffer_sha256": {case: ["b" * 64]},
            "samples": [
                {
                    "shape_sha256": case,
                    "buffer_index": 0,
                    "started_ns": index * 1_000_000_000,
                    "ended_ns": index * 1_000_000_000 + int(value * 1e6),
                    "synchronized_before": True,
                    "synchronized_after": True,
                }
                for index, value in enumerate(values["samples_ms"])
            ],
        }
        values["sample_groups"] = [
            {
                "case_sha256": case,
                "block_id": f"block-{block}",
                "source_sample_indices": list(range(block * 2, block * 2 + 2)),
                "source_ref": f"contract-fixture/{identity}.json",
                "source_sha256": "a" * 64,
                "grouping_derivation_ref": "contract-fixture/two-sample-blocks",
                "grouping_derivation_sha256": "c" * 64,
            }
            for block in range(len(values["samples_ms"]) // 2)
        ]
    return AblationRun(**values)


def test_redundant_zero_loo_factors_cannot_both_be_pruned_into_a_regression():
    report = AblationGate(min_speedup=1.05).evaluate(
        task_id="redundant-bottleneck",
        factors=[AblationFactor(factor_id="A"), AblationFactor(factor_id="B")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A", "B"], 90),
        ablations=[run("without-A", ["B"], 90), run("without-B", ["A"], 90)],
        pruning_runs=[run("prune-A", ["B"], 90), run("prune-B", [], 100)],
    )
    assert [factor.conditional_delta_p50 for factor in report.factors] == [0, 0]
    assert report.status == "failed"
    assert not report.pruned_combination_verified
    assert report.pruning_verifications[0].performance_verified
    assert not report.pruned_combination.performance_verified
    assert report.runs[-1].samples_ms == [100] * 4


def test_pruning_requires_one_fresh_verification_per_removal():
    options = dict(
        task_id="redundant",
        factors=[AblationFactor(factor_id="A"), AblationFactor(factor_id="B")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A", "B"], 90),
        ablations=[run("without-A", ["B"], 90), run("without-B", ["A"], 90)],
    )
    with pytest.raises(ValueError, match="one factor at a time"):
        AblationGate().evaluate(**options, pruning_runs=[run("drop-both", [], 90)])
    report = AblationGate().evaluate(**options, pruning_runs=[run("drop-A", ["B"], 90)])
    assert report.status == "passed"
    assert report.pruned_combination_verified
    assert report.pruned_combination.retained_factors == ["B"]
    with pytest.raises(ValueError, match="distinct run_id"):
        AblationGate().evaluate(**options, pruning_runs=[run("without-A", ["B"], 90)])


def test_correctness_and_dependencies_are_retained_without_speedup_votes():
    factors = [
        AblationFactor(factor_id="tiling", depends_on=["align"]),
        AblationFactor(factor_id="align", role="correctness_required"),
    ]
    options = dict(
        task_id="aligned",
        factors=factors,
        baseline=run("base", [], 100),
        full_candidate=run("full", ["tiling", "align"], 70),
        ablations=[run("without-tiling", ["align"], 100)],
    )
    report = AblationGate().evaluate(**options)
    assert report.status == "passed"
    assert report.factors[1].disposition == "retain_required_dependency"
    assert report.factors[1].conditional_delta_p50 is None
    with pytest.raises(ValueError, match="correctness requirement"):
        AblationGate().evaluate(**options, pruning_runs=[run("without-align", ["tiling"], 60)])


@pytest.mark.parametrize("status", ["infra_failed", "unknown", "cancelled"])
def test_execution_failures_do_not_become_zero_contribution(status):
    report = AblationGate().evaluate(
        task_id="uncertain",
        factors=[AblationFactor(factor_id="A")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A"], 90),
        ablations=[
            run(
                "without-A",
                [],
                1,
                run_status=status,
                measurement_valid=False,
                correctness_status="not_measured",
                samples_ms=[],
            )
        ],
    )
    assert report.status == "inconclusive"
    assert report.factors[0].conditional_delta_p50 is None
    assert report.factors[0].disposition == "retain_pending_evidence"
    assert report.runs[-1].run_status == status


def test_different_conditions_are_not_a_matched_ablation():
    with pytest.raises(ValueError, match="matched environment"):
        AblationGate().evaluate(
            task_id="mismatch",
            factors=[AblationFactor(factor_id="A")],
            baseline=run("base", [], 100),
            full_candidate=run("full", ["A"], 90),
            ablations=[run("without-A", [], 100, conditions={"soc": "different-chip"})],
        )


def test_inseparable_changes_are_evaluated_as_one_group():
    report = AblationGate().evaluate(
        task_id="grouped",
        factors=[
            AblationFactor(
                factor_id="layout-and-kernel", role="inseparable_group", members=["layout", "kernel"]
            )
        ],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["layout-and-kernel"], 50),
        ablations=[run("without-group", [], 100)],
    )
    assert report.status == "passed"
    assert report.factors[0].disposition == "retain_verified_contributor"


def test_observed_correctness_dependency_is_retained_without_inventing_timing():
    report = AblationGate().evaluate(
        task_id="discovered-dependency",
        factors=[AblationFactor(factor_id="A")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A"], 90),
        ablations=[
            run(
                "without-A",
                [],
                0,
                run_status="failed",
                correctness_status="failed",
                measurement_valid=False,
                samples_ms=[],
            )
        ],
    )
    assert report.status == "passed"
    assert report.factors[0].disposition == "retain_required_by_correctness_evidence"
    assert report.factors[0].conditional_delta_p50 is None


def test_full_candidate_correctness_failure_is_reported_as_failure():
    report = AblationGate().evaluate(
        task_id="incorrect-full",
        factors=[AblationFactor(factor_id="A")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A"], 90, run_status="failed", correctness_status="failed"),
    )
    assert report.status == "failed"
    assert not report.pruned_combination.correctness_verified


def test_cli_preserves_failed_and_inconclusive_gate_outcomes(tmp_path, capsys):
    import json

    payload = dict(
        task_id="cli-contract",
        factors=[dict(factor_id="A")],
        baseline=run("base", [], 100).model_dump(),
        full_candidate=run("full", ["A"], 110).model_dump(),
        ablations=[run("without-A", [], 100).model_dump()],
    )
    source, target = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(payload))
    assert main(["ablation-evaluate", "--input", str(source), "--output", str(target)]) == 1
    assert json.loads(target.read_text())["status"] == "failed"
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    payload["full_candidate"] = run(
        "full", ["A"], 90, measurement_valid=False, measurement_limitations=["incomplete telemetry"]
    ).model_dump()
    source.write_text(json.dumps(payload))
    assert main(["ablation-evaluate", "--input", str(source), "--output", str(target)]) == 2
    assert json.loads(target.read_text())["status"] == "inconclusive"


def test_all_retained_contributor_has_an_independent_final_verification_without_fake_pruning():
    options = dict(
        task_id="positive-factor",
        factors=[AblationFactor(factor_id="A")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A"], 60),
        ablations=[run("without-A", [], 100)],
    )
    report = AblationGate(min_speedup=1.05).evaluate(
        **options, final_candidate=run("final-retained", ["A"], 60)
    )
    assert report.status == "passed"
    assert report.final_combination_verified
    assert report.final_combination.verification_run_id == "final-retained"
    assert report.final_combination.retained_factors == ["A"]
    assert not report.pruned_combination_verified
    assert report.pruning_verifications == []
    assert report.factors[0].disposition == "retain_verified_contributor"
    with pytest.raises(ValueError, match="distinct run_id"):
        AblationGate().evaluate(**options, final_candidate=run("full", ["A"], 60))
    with pytest.raises(ValueError, match="retained combination"):
        AblationGate().evaluate(**options, final_candidate=run("wrong-final", [], 60))


def test_a_removed_factor_still_needs_its_fresh_combination_measurement():
    report = AblationGate().evaluate(
        task_id="unmeasured-removal",
        factors=[AblationFactor(factor_id="A"), AblationFactor(factor_id="B")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A", "B"], 70),
        ablations=[run("without-A", ["B"], 70), run("without-B", ["A"], 100)],
        pruning_runs=[
            run(
                "remove-A",
                ["B"],
                70,
                measurement_valid=False,
                measurement_limitations=["actual comparison-window telemetry missing"],
            )
        ],
    )
    assert report.status == "inconclusive"
    assert not report.pruned_combination_verified
    assert not report.final_combination_verified


def test_point_estimate_cannot_clear_retention_when_observed_blocks_cross_the_loss_limit():
    report = AblationGate(min_speedup=1.05).evaluate(
        task_id="independent-block-variation",
        factors=[AblationFactor(factor_id="A")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A"], 60),
        ablations=[run("without-A", [], 100)],
        final_candidate=run("final", ["A"], 60, samples_ms=[50, 50, 70, 70]),
    )
    final = report.final_combination
    assert final.final_speedup_geomean == pytest.approx(100 / 60)
    assert final.retention_comparison.cases[0].observed_block_ratio_lower == pytest.approx(60 / 70)
    assert report.status == "inconclusive"
    assert not final.uncertainty_verified and not report.final_combination_verified


def _workload_run(identity, factors, durations):
    value = run(identity, factors, durations[0]).model_dump(mode="json")
    samples, groups, buffers = [], [], {}
    for number, duration in enumerate(durations):
        part = run(identity, factors, duration).model_dump(mode="json")
        case = hashlib.sha256(f"case-{number}".encode()).hexdigest()
        offset = len(samples)
        for sample in part["timing"]["samples"]:
            sample["shape_sha256"] = case
            sample["started_ns"] += offset * 1_000_000_000
            sample["ended_ns"] += offset * 1_000_000_000
        samples.extend(part["timing"]["samples"])
        for group in part["sample_groups"]:
            group["case_sha256"] = case
            group["source_sample_indices"] = [i + offset for i in group["source_sample_indices"]]
        groups.extend(part["sample_groups"])
        buffers[case] = ["b" * 64]
    value["timing"].update(samples=samples, buffer_sha256=buffers)
    value["samples_ms"] = [(row["ended_ns"] - row["started_ns"]) / 1e6 for row in samples]
    value["sample_groups"] = groups
    return AblationRun.model_validate(value)


def test_case_regression_is_not_hidden_by_geomean_improvement_or_mixed_workload_iqr():
    report = AblationGate(min_speedup=1.05).evaluate(
        task_id="different-workload-scales",
        factors=[AblationFactor(factor_id="A")],
        baseline=_workload_run("base", [], [1000, 10]),
        full_candidate=_workload_run("full", ["A"], [900, 9]),
        ablations=[_workload_run("without-A", [], [1000, 10])],
        final_candidate=_workload_run("final", ["A"], [810, 9.9]),
    )
    final = report.final_combination
    assert final.final_speedup_geomean > report.full_candidate_speedup
    assert report.noise_floor_rel_iqr == 0
    assert min(case.observed_block_ratio_lower for case in final.retention_comparison.cases) < 0.985
    assert report.status == "inconclusive" and not report.final_combination_verified


def test_flat_legacy_samples_do_not_prove_case_matched_uncertainty():
    final = run("final", ["A"], 60).model_dump(mode="json")
    final.pop("timing")
    final.pop("sample_groups")
    report = AblationGate().evaluate(
        task_id="legacy-ungrouped",
        factors=[AblationFactor(factor_id="A")],
        baseline=run("base", [], 100),
        full_candidate=run("full", ["A"], 60),
        ablations=[run("without-A", [], 100)],
        final_candidate=AblationRun.model_validate(final),
    )
    assert report.final_combination.final_speedup_p50 == pytest.approx(100 / 60)
    assert report.final_combination.final_speedup_geomean is None
    assert report.status == "inconclusive" and not report.final_combination_verified
