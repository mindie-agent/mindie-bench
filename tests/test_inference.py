from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.evaluation.inference import (
    CandidateMeasurementFailure,
    evaluate_inference,
    measure_latency_ms,
    measure_workloads,
)
from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.hf_sync.exporter import public_row, public_task_row
from mindie_bench.inference_models import InferenceContract, InferenceEvidence
from mindie_bench.io import read_json
from mindie_bench.models import ExperimentConfig, PerformanceContract, TaskSpec, TrialReceipt
from mindie_bench.runner.verifier import parse_verifier, remote_verifier
from mindie_bench.scorer.metrics import summarize

NONCE, ORACLE = "native-inference-nonce", "a" * 64


def contract(category="optimization"):
    return InferenceContract(
        category=category, engine="test-engine", model="test-model", baseline_commit="b" * 40,
        baseline_source_sha256="2" * 64,
        runtime_sha256="c" * 64, device_class="cpu", required_checks=["numerics", "serving"],
        measurement_rounds=2,
        workloads=[] if category == "model_support" else [
            dict(name="decode", metric="tokens_per_sec", direction="higher", workload_sha256="d" * 64),
            dict(name="prefill", metric="ttft_ms", direction="lower", workload_sha256="e" * 64),
            dict(name="guard", metric="latency_ms", direction="lower", role="guard",
                 min_ratio=0.95, workload_sha256="f" * 64),
        ],
    )


def evidence(spec, *, speed=2.0, checks=None, guard=1.0):
    def measure(arm, workload):
        if arm == "baseline":
            return 100.0
        ratio = guard if workload.role == "guard" else speed
        return 100.0 * ratio if workload.direction == "higher" else 100.0 / ratio

    return InferenceEvidence(
        run_nonce=NONCE, oracle_manifest_sha256=ORACLE, baseline_commit=spec.baseline_commit,
        baseline_source_sha256=spec.baseline_source_sha256,
        runtime_sha256=spec.runtime_sha256, device_class=spec.device_class,
        candidate_sha256="1" * 64, baseline_healthy=True,
        checks=checks or dict(numerics=True, serving=True), workloads=measure_workloads(spec, measure),
    )


def parse(spec, observed, *, reward=None, **kwargs):
    reward = float(evaluate_inference(spec, observed)["accepted"]) if reward is None else reward
    return parse_verifier(
        0 if reward else 1, json.dumps(dict(reward=reward, inference_evidence=observed.model_dump())),
        inference_contract=spec, performance_nonce=NONCE, oracle_manifest_sha256=ORACLE, **kwargs,
    )


def test_warmup_then_counterbalanced_pairs_execute_each_declared_workload():
    spec, calls = contract(), []

    def measure(arm, workload):
        calls.append((workload.name, arm))
        return 1.0

    observed = measure_workloads(spec, measure)
    assert calls == [(workload.name, arm) for workload in spec.workloads for arm in (
        "baseline", "candidate", "baseline", "candidate", "candidate", "baseline",
    )]
    assert [pair.order for pair in observed["decode"].pairs] == ["baseline_candidate", "candidate_baseline"]


def test_real_latency_measurement_waits_for_completion():
    import threading

    pending, completed = [], []

    def operation():
        worker = threading.Thread(target=lambda: completed.append(sum(range(100000))))
        pending.append(worker)
        worker.start()

    def synchronize():
        while pending:
            pending.pop().join()

    assert measure_latency_ms(operation, synchronize) > 0
    assert completed == [sum(range(100000))] and not pending


def test_paired_median_and_metric_directions_not_ratio_of_aggregate_means():
    spec, observed = contract(), evidence(contract())
    observed.workloads["decode"].pairs[0].baseline = 1.0
    observed.workloads["decode"].pairs[0].candidate = 2.0
    observed.workloads["decode"].pairs[1].baseline = 1000.0
    observed.workloads["decode"].pairs[1].candidate = 500.0
    result = evaluate_inference(spec, observed)
    assert result["workloads"]["decode"]["ratio"] == 1.25
    assert result["workloads"]["prefill"]["ratio"] == 2.0
    assert result["measured_speedup"] == pytest.approx((1.25 * 2)**0.5)
    assert parse(spec, observed).reward == 1


@pytest.mark.parametrize("category", ["optimization", "regression"])
def test_slowdown_and_subthreshold_gain_remain_measured(category):
    spec = contract(category)
    for speed in (0.5, 1.01):
        observed = evidence(spec, speed=speed)
        outcome = evaluate_inference(spec, observed)
        assert outcome["score"] == pytest.approx(speed)
        assert not outcome["accepted"] and parse(spec, observed).reward == 0


def test_guard_rejection_cannot_be_hidden_by_scored_speedup():
    spec = contract()
    observed = evidence(spec, speed=3, guard=0.9)
    result = evaluate_inference(spec, observed)
    assert result["score"] == 1 and result["measured_speedup"] == pytest.approx(3)
    assert "guard_regressed:guard" in result["failed_gates"]
    assert parse(spec, observed).reward == 0
    with pytest.raises(ValueError, match="contradicts inference"):
        parse(spec, observed, reward=1)


def test_correctness_failure_is_neutral_but_missing_guard_is_not_evidence():
    spec = contract()
    observed = evidence(spec, checks=dict(numerics=False, serving=True))
    observed.workloads = {}
    assert evaluate_inference(spec, observed)["score"] == 1
    assert parse(spec, observed).reward == 0
    measured = evidence(spec)
    del measured.workloads["guard"]
    with pytest.raises(ValueError, match="guard workload"):
        parse(spec, measured)


def test_unchanged_source_cannot_earn_gain_from_measurement_noise():
    spec = contract()
    observed = evidence(spec, speed=1.1)
    observed.candidate_sha256 = spec.baseline_source_sha256
    result = evaluate_inference(spec, observed)
    assert result["score"] == 1 and result["failed_gates"] == ["unchanged_candidate"]
    assert parse(spec, observed).reward == 0


def test_support_requires_all_checks_and_preserves_partial_fraction():
    spec = contract("model_support")
    observed = evidence(spec, checks=dict(numerics=True, serving=False))
    result = evaluate_inference(spec, observed)
    assert result["test_pass_fraction"] == 0.5 and result["score"] == 0
    assert parse(spec, observed).reward == 0
    del observed.checks["serving"]
    with pytest.raises(ValueError, match="exactly the required"):
        parse(spec, observed)


def test_candidate_failure_retained_and_infrastructure_not_relabelled():
    spec = contract()

    def measure(arm, workload):
        if arm == "candidate":
            raise CandidateMeasurementFailure("engine failed to serve")
        return 1.0

    observed = evidence(spec)
    observed.workloads = measure_workloads(spec, measure)
    assert evaluate_inference(spec, observed)["failed_gates"] == ["candidate_measurement_failed"]
    assert parse(spec, observed).reward == 0

    def broken_baseline(arm, workload):
        raise ConnectionError("device transport lost")

    with pytest.raises(ConnectionError, match="device transport"):
        measure_workloads(spec, broken_baseline)
    observed.baseline_healthy = False
    with pytest.raises(ValueError, match="baseline is unhealthy"):
        parse(spec, observed)


def test_nonce_frozen_oracle_and_runtime_bind_the_judge_observations():
    spec = contract()
    for field, value in (("run_nonce", "different-run-nonce"), ("oracle_manifest_sha256", "0" * 64),
                         ("runtime_sha256", "0" * 64), ("device_class", "other-device")):
        observed = evidence(spec)
        setattr(observed, field, value)
        with pytest.raises(ValueError):
            parse(spec, observed, reward=1)
    observed = evidence(spec)
    observed.workloads["decode"].pairs.pop()
    with pytest.raises(ValueError, match="complete counterbalanced"):
        parse(spec, observed)


def test_inference_does_not_replace_existing_native_performance_gates():
    spec = contract()
    with pytest.raises(ValueError):
        parse(spec, evidence(spec), performance_contract=PerformanceContract(
            private_shape_sha256=["a" * 64, "b" * 64],
        ))
    assert parse_verifier(0, '{"reward":1}').reward == 1
    with pytest.raises(ValueError, match="declared inference"):
        parse_verifier(0, json.dumps(dict(reward=1, inference_evidence=evidence(spec).model_dump())))


def receipt(task="task", repetition=1, *, spec=None, observed=None, status=None, group="G0"):
    spec = spec or contract()
    observed = observed or evidence(spec)
    accepted = evaluate_inference(spec, observed)["accepted"]
    status = status or ("completed_pass" if accepted else "completed_fail")
    valid = status.startswith("completed")
    return TrialReceipt(
        trial_id=f"{task}-{repetition}-{group}", instance_id=task, repetition=repetition,
        tier="l2", ablation_group=group, knowledge_gradient="no_hit", task_digest=task + "-digest",
        experiment_digest="experiment", environment_fingerprint={"inference_contract": spec.model_dump()},
        terminal_status=status, evidence="synthetic",
        failure_reason=None if status == "completed_pass" else "original native failure",
        primary_metrics=dict(reward=float(status == "completed_pass") if valid else None,
                             resolved=status == "completed_pass" if valid else None,
                             performance_metrics={"inference_evidence": observed.model_dump()}),
    )


def cell(row):
    return dict(instance_id=row.instance_id, tier=row.tier, group=row.ablation_group.value,
                repetition=row.repetition, role=row.task_role, task_digest=row.task_digest,
                evidence=row.evidence,
                inference_category=row.environment_fingerprint["inference_contract"]["category"])


def track(rows, planned=None, name="optimization"):
    result = summarize(rows, expected_cells=planned)
    return result["inference"]["cohorts"][0]["tracks"][name]


def test_native_score_task_weight_not_attempt_weight_and_legacy_metrics_unchanged():
    spec = contract("model_support")
    rows = [receipt("easy", n, spec=spec) for n in (1, 2, 3)] + [
        receipt("hard", spec=spec, observed=evidence(spec, checks=dict(numerics=True, serving=False)))
    ]
    result = summarize(rows, expected_cells=[cell(row) for row in rows])
    assert result["pass_at_1"] == 0.75
    assert result["inference"]["cohorts"][0]["tracks"]["model_support"]["score"] == 0.5
    assert track(rows, name="model_support")["score"] is None


def test_native_score_counts_missing_unknown_blocked_and_extra_cells():
    first, second = receipt("a"), receipt("b", status="unknown")
    planned = [cell(first), cell(second)]
    result = track([first, second], planned)
    assert result["observed_score"] == 2 and result["score"] is None
    assert track([first], planned)["score"] is None
    assert track([first, receipt("extra")], [cell(first)])["score"] is None
    slowdown = receipt("b", observed=evidence(contract(), speed=0.5))
    assert track([first, slowdown], planned)["score"] == 1


def test_other_native_failure_neutralizes_gain_and_public_export_keeps_contract():
    row = receipt(status="completed_fail")
    assert track([row], [cell(row)])["score"] == 1
    assert public_row(row)["environment_fingerprint"]["inference_contract"] == contract().model_dump()


@pytest.mark.parametrize("speed", [0.5, 1.01])
def test_native_workspace_gate_applies_below_inference_threshold_but_budget_is_separate(speed):
    from mindie_bench.evaluation.performance import measure_rotating_buffers

    row = receipt(observed=evidence(contract(), speed=speed))
    shapes = {"a" * 64: [1, 2], "b" * 64: [3, 4]}
    clock = iter(range(0, 10000, 10))
    measured = measure_rotating_buffers(
        shapes, lambda shape, buffer: None, lambda: None, buffer_identity=str,
        samples_per_shape=2, clock_ns=lambda: next(clock),
    ).model_dump()
    admission = PerformanceContract(
        private_shape_sha256=list(shapes), require_aclgraph_capture=False,
        require_e2e_warm_server=False, max_workspace_delta_bytes=0,
    )
    row.environment_fingerprint["performance_contract"] = admission.model_dump()
    row.primary_metrics["performance_metrics"].update(
        speedup_ratio=1.0, memory_workspace_delta_mb=0.000001,
        performance_evidence=dict(
            run_nonce=NONCE, oracle_manifest_sha256=ORACLE, device_identity="synthetic-contract-only",
            baseline=measured, candidate=measured,
            workspace=dict(method="synchronized_allocator_peak", baseline_bytes=10, candidate_bytes=11,
                           synchronized_before=True, synchronized_after=True),
        ),
    )
    assert track([row], [cell(row)])["score"] == 1
    # Recompute the existing gate, not a copied pass/fail label in the receipt.
    row.primary_metrics["performance_metrics"]["performance_evidence"]["workspace"]["candidate_bytes"] = 10
    row.primary_metrics["performance_metrics"]["memory_workspace_delta_mb"] = 0.0
    assert track([row], [cell(row)])["score"] == pytest.approx(speed)
    row.environment_fingerprint["inference_native_stop_reason"] = "max_turns"
    assert track([row], [cell(row)])["score"] == pytest.approx(speed)
    exported = public_row(row)["environment_fingerprint"]
    assert exported["performance_contract"] == admission.model_dump()
    assert exported["inference_native_stop_reason"] == "max_turns"


@pytest.mark.parametrize("speed", [0.5, 1.02, 2.0])
@pytest.mark.parametrize("stop", ["timeout", "max_turns", "max_npu_runs"])
def test_budget_stop_and_delivery_overrun_preserve_measured_speed_and_verdict(speed, stop):
    row = receipt(observed=evidence(contract(), speed=speed))
    row.auxiliary_metrics = dict(
        native_process={"stop_reason": stop},
        lifecycle_budget=dict(overrun=True, measured_total_sec=110, declared_total_sec=100, overrun_sec=10),
    )
    before = row.model_dump()
    report = summarize([row], expected_cells=[cell(row)])
    cohort = report["inference"]["cohorts"][0]
    assert cohort["tracks"]["optimization"]["score"] == pytest.approx(speed)
    assert cohort["tracks"]["optimization"]["complete"] is True
    assert report["pass_at_1"] == float(speed == 2.0)
    assert report["budget"]["agent_budget_exhausted_attempts"] == 1
    assert report["budget"]["delivery_budget_exceeded_attempts"] == 1
    assert report["budget"]["completed_pass_over_delivery_budget"] == int(speed == 2.0)
    assert report["budget"]["pass_at_1_within_delivery_budget"] == 0
    assert public_row(row)["budget"]["delivery_overrun_sec"] == 10
    assert row.model_dump() == before


def test_historical_overrun_uses_explicit_business_verdict_without_rewriting_receipt():
    row = receipt(status="completed_fail")
    row.auxiliary_metrics = dict(
        business_terminal_status="completed_pass", business_reward=1.0,
        lifecycle_budget=dict(overrun=True, measured_total_sec=110, declared_total_sec=100, overrun_sec=10),
    )
    assert track([row], [cell(row)])["score"] == 2
    assert row.terminal_status.value == "completed_fail"
    assert row.primary_metrics["reward"] == 0
    del row.auxiliary_metrics["business_terminal_status"]
    assert track([row], [cell(row)])["score"] == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_evidence_cohorts_scope_the_original_plan_and_keep_unclassified_missing_cells(legacy):
    from mindie_bench.scorer.comparator import grouped_summaries

    rows = [receipt("synthetic"), receipt("verified")]
    rows[1].evidence = "verified"
    planned = [cell(row) for row in rows]
    if legacy:
        for item in planned:
            del item["evidence"]
    cohorts = summarize(rows, expected_cells=planned)["inference"]["cohorts"]
    groups = grouped_summaries(rows, expected_cells=planned)
    for cohort in cohorts:
        assert cohort["tracks"]["optimization"]["score"] == 2
        assert cohort["tracks"]["optimization"]["tasks"] == 1
        assert groups[f"l2/{cohort['evidence']}/G0"]["matrix_coverage"]["complete"] is True
    missing = cell(receipt("missing"))
    if legacy:
        del missing["evidence"]
    planned.append(missing)
    groups = grouped_summaries(rows, expected_cells=planned)
    assert groups["l2/synthetic/G0"]["inference"]["cohorts"][0]["tracks"]["optimization"]["score"] is None
    verified = groups["l2/verified/G0"]
    assert verified["matrix_coverage"]["complete"] is (not legacy)


def test_cohorts_keep_g0_g3_and_model_conditions_separate():
    rows = [receipt(group="G0"), receipt(group="G3")]
    rows[1].environment_fingerprint["base_model"] = "other-model"
    result = summarize(rows)["inference"]
    assert len(result["cohorts"]) == 2
    assert {cohort["group"] for cohort in result["cohorts"]} == {"G0", "G3"}


def test_native_revision_and_cell_integrity_prevent_double_counting():
    row = receipt()
    with pytest.raises(ValueError, match="duplicate native"):
        summarize([row, row])
    changed = receipt(repetition=2)
    changed.task_digest = "changed"
    with pytest.raises(ValueError, match="mixed inference task"):
        summarize([row, changed])
    planned = cell(row)
    planned["task_digest"] = "changed"
    with pytest.raises(ValueError, match="differs from its declared plan"):
        summarize([row], expected_cells=[planned])


def test_existing_score_command_uses_the_original_native_plan(tmp_path, capsys):
    from mindie_bench.cli import main
    from mindie_bench.io import write_json

    row = receipt()
    trial = tmp_path / row.trial_id
    trial.mkdir()
    write_json(trial / "trial_receipt.json", row.model_dump(mode="json"))
    write_json(tmp_path / "eval_manifest.json", dict(
        tasks=[{"path": "task"}], groups=["G0"], repetitions=1, cells=[cell(row)],
    ))
    assert main(["score", "--input", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)["l2/synthetic/G0"]
    assert report["pass_at_1"] == 1
    assert report["inference"]["cohorts"][0]["tracks"]["optimization"]["score"] == 2
    # Re-read the same receipts after an original plan declares a missing task.
    missing = cell(receipt("missing"))
    write_json(tmp_path / "eval_manifest.json", dict(
        tasks=[{"path": "task"}, {"path": "missing"}], groups=["G0"], repetitions=1,
        cells=[cell(row), missing],
    ))
    assert main(["score", "--input", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)["l2/synthetic/G0"]
    assert report["inference"]["cohorts"][0]["tracks"]["optimization"]["score"] is None


@pytest.mark.parametrize("with_plan", [False, True])
def test_compare_cli_uses_each_original_plan_like_score(tmp_path, capsys, with_plan):
    from mindie_bench.cli import main
    from mindie_bench.io import write_json
    from mindie_bench.scorer.comparator import CONTROL_KEYS

    for group in ("G0", "G1"):
        row = receipt(group=group)
        row.environment_fingerprint.update({key: None for key in CONTROL_KEYS})
        row.primary_metrics["localization_penalty_turns"] = 25
        directory = tmp_path / group
        write_json(directory / row.trial_id / "trial_receipt.json", row.model_dump(mode="json"))
        if with_plan:
            write_json(directory / "eval_manifest.json", dict(
                tasks=[{"path": "task"}], groups=[group], repetitions=1, cells=[cell(row)],
            ))
        assert main(["score", "--input", str(directory)]) == 0
        scored = json.loads(capsys.readouterr().out)[f"l2/synthetic/{group}"]
        assert scored["inference"]["cohorts"][0]["tracks"]["optimization"]["score"] == (2 if with_plan else None)
    output = tmp_path / "comparison"
    assert main(["compare", "--baseline", str(tmp_path / "G0"), "--candidate", str(tmp_path / "G1"),
                 "--output", str(output)]) == 0
    report = read_json(output / "comparison.json")
    for side in ("baseline", "candidate"):
        values = report[side]["inference"]["cohorts"][0]["tracks"]["optimization"]
        assert values["score"] == (2 if with_plan else None)
        assert values["complete"] is with_plan
    assert report["comparison_coverage"]["complete"] == (True if with_plan else None)


def example_module():
    path = Path(__file__).resolve().parents[1] / "tasks/inference/materialize.py"
    spec = importlib.util.spec_from_file_location("inference_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def examples(tmp_path):
    module = example_module()
    output = tmp_path / "lifecycle"
    tasks = [Path(path) for path in module.materialize(output)]
    return module, output, tasks


def test_real_example_judges_use_native_task_contracts_and_measured_cpu_work(examples):
    module, output, tasks = examples
    for task in tasks:
        spec = TaskSpec.load(task)
        workspace = Path(spec.environment["l2"].repo_path)
        frozen = freeze_oracle(task, spec.oracle)
        env = dict(os.environ, MINDIE_BENCH_ORACLE_ROOT=str(task), MINDIE_BENCH_WORKSPACE=str(workspace),
                   MINDIE_BENCH_PERFORMANCE_NONCE=NONCE, MINDIE_BENCH_ORACLE_SHA256=frozen.sha256)
        # All unmodified sources fail, including noisy baseline-vs-itself timings.
        before = subprocess.run([sys.executable, "-I", str(task / "verifier/judge.py")],
                                env=env, capture_output=True, text=True, timeout=60)
        assert before.returncode == 1, before.stderr
        (workspace / "engine.py").write_text(module.GOLD, encoding="utf-8")
        after = subprocess.run([sys.executable, "-I", str(task / "verifier/judge.py")],
                               env=env, capture_output=True, text=True, timeout=60)
        assert after.returncode in (0, 1), after.stderr
        verdict = parse_verifier(after.returncode, after.stdout, inference_contract=spec.inference,
                                 performance_nonce=NONCE, oracle_manifest_sha256=frozen.sha256)
        outcome = evaluate_inference(spec.inference, verdict.inference_evidence)
        assert outcome["correctness_passed"]
        if spec.inference.category != "model_support":
            assert outcome["measured_speedup"] > 0
            assert len(verdict.inference_evidence.workloads) == 3
        assert public_task_row(task)["inference"]["category"] == spec.inference.category


@pytest.mark.skipif(sys.platform == "win32", reason="native runner requires Linux/macOS OS isolation")
def test_native_matrix_and_soundness_use_inference_contract_end_to_end(examples, capsys):
    from mindie_bench.cli import main
    from mindie_bench.runner.soundness import check_soundness
    from mindie_bench.scorer.comparator import grouped_summaries, load_receipts

    _, output, tasks = examples
    config = ExperimentConfig.load(output / "experiment.toml")
    # No artificial timing threshold is asserted here; soundness is deterministic
    # on the support task. All three tasks still run their real measured judges.
    proof = check_soundness(tasks[0], config, tier="l2", output=output / "proof")
    assert proof["before_reward"] == 0 and proof["after_reward"] == 1
    assert main([
        "run", "--tasks", str(output / "tasks"), "--config", str(output / "experiment.toml"),
        "--tier", "l2", "--groups", "G0", "--repetitions", "1", "--concurrency", "1",
        "--output", str(output / "run"),
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    rows = load_receipts(output / "run")
    assert all(row.terminal_status.value.startswith("completed") for row in rows), [row.errors for row in rows]
    plan = read_json(output / "run/eval_manifest.json")["cells"]
    result = grouped_summaries(rows, expected_cells=plan)["l2/synthetic/G0"]["inference"]
    assert result["cohorts"][0]["tracks"]["optimization"]["complete"]
    assert result["cohorts"][0]["tracks"]["model_support"]["score"] == 1
    saved = read_json(output / "run/summary.json")["l2/synthetic/G0"]["inference"]
    assert saved == result
    assert printed["l2/synthetic/G0"]["inference"] == saved


def test_remote_private_judge_forwards_contract_and_run_binding(examples, tmp_path):
    _, _, tasks = examples
    task = tasks[0]
    spec = TaskSpec.load(task)
    frozen = freeze_oracle(task, spec.oracle)
    trial = tmp_path / "remote-trial"
    trial.mkdir()

    class Transport:
        operation_unknown = False

        def verify(self, script, *, env, timeout):
            import base64

            measured = evidence(spec.inference, checks={name: True for name in spec.inference.required_checks})
            measured.run_nonce = env["MINDIE_BENCH_PERFORMANCE_NONCE"]
            measured.oracle_manifest_sha256 = env["MINDIE_BENCH_ORACLE_SHA256"]
            raw = json.dumps(dict(reward=1, inference_evidence=measured.model_dump())).encode()
            envelope = dict(schema_version="mindie-bench-private-oracle/1", nonce=measured.run_nonce,
                            manifest_sha256=frozen.sha256, probes=[], before={}, after={}, tampered=None,
                            original_error=None, timed_out=False,
                            judge=dict(returncode=0, stdout_b64=base64.b64encode(raw).decode(), stderr_b64=""))
            return dict(exit_code=0, stdout_b64=base64.b64encode(json.dumps(envelope).encode()).decode(), stderr_b64="")

    verdict, _ = remote_verifier(task, Transport(), trial, 60, frozen, spec.oracle, None, spec.inference)
    assert verdict.reward == 1 and verdict.inference_evidence.run_nonce != NONCE
