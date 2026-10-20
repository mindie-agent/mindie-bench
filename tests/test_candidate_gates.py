from __future__ import annotations

from test_ablation import run as measured_run

from mindie_bench.evaluation.ablation import AblationFactor, AblationGate
from mindie_bench.evaluation.gates import (
    CorrectnessContract,
    CorrectnessEvidence,
    ExecutionContract,
    ExecutionEvidence,
    GateContract,
    MeasurementWindow,
    TelemetryEvidence,
    evaluate_gates,
    telemetry_from_c2,
)


def evidence():
    window = MeasurementWindow(start_ts=10, end_ts=20)
    contract = GateContract(
        task_id="varlen-state-fixture",
        require_telemetry=True,
        telemetry_host="contract-fixture-host",
        telemetry_devices=[0],
        measurement_uses_execution_window=True,
        execution=ExecutionContract(
            run_id="run-candidate",
            patch_digest="a" * 64,
            binary_sha256="b" * 64,
            module_path="/sandbox/candidate/kernel.so",
            target_call_path="state.forward",
            input_ids=["packed", "idle-rank"],
            window=window,
        ),
        correctness=CorrectnessContract(
            required_cases=["packed-varlen", "dp-idle"], max_error={"max_abs": 0.001}
        ),
    )
    execution = ExecutionEvidence(
        run_id="run-candidate",
        patch_digest="a" * 64,
        loaded_binary_sha256="b" * 64,
        loaded_module_path="/sandbox/candidate/kernel.so",
        evidence_ref="trusted/dispatch.json",
        calls=[
            dict(input_id="packed", call_path="state.forward", timestamp=12),
            dict(input_id="idle-rank", call_path="state.forward", timestamp=18),
        ],
    )
    correctness = CorrectnessEvidence(
        run_id="run-candidate",
        evidence_ref="trusted/numerics.json",
        cases=[
            dict(case_id="packed-varlen", input_id="packed", errors={"max_abs": 0.0001}, passed=True),
            dict(case_id="dp-idle", input_id="idle-rank", errors={"max_abs": 0}, passed=True),
        ],
    )
    telemetry = TelemetryEvidence(
        run_id="run-candidate",
        host="contract-fixture-host",
        devices=[0],
        window=window,
        coverage_ratio=1,
        throttling_detected=False,
        resource_interference_detected=False,
        freq_mhz_min=1500,
        power_max_w=240,
        temp_max_c=86,
        evidence_ref="trusted/telemetry.json",
    )
    return contract, dict(execution=execution, correctness=correctness, telemetry=telemetry)


def test_same_named_old_binary_and_outside_window_calls_do_not_prove_execution():
    contract, values = evidence()
    values["execution"].loaded_binary_sha256 = "c" * 64
    result = evaluate_gates(contract, **values)
    assert result.execution.status == "failed"
    assert "candidate_identity_mismatch:binary_sha256" in result.execution.reasons
    values["execution"].loaded_binary_sha256 = "b" * 64
    values["execution"].calls[0].timestamp = 9
    result = evaluate_gates(contract, **values)
    assert result.execution.status == "failed"
    assert "target_calls_missing_for_inputs_in_measurement_window" in result.execution.reasons


def test_task_specific_cases_and_error_thresholds_determine_correctness():
    contract, values = evidence()
    assert evaluate_gates(contract, **values).status == "passed"
    values["correctness"].cases.pop()
    assert evaluate_gates(contract, **values).correctness.status == "inconclusive"
    values["correctness"].cases[0].errors["max_abs"] = 0.1
    assert evaluate_gates(contract, **values).correctness.status == "failed"


def test_temperature_alone_neither_proves_nor_refutes_measurement_health():
    contract, values = evidence()
    assert evaluate_gates(contract, **values).measurement.status == "passed"
    values["telemetry"].temp_max_c = 50
    values["telemetry"].throttling_detected = True
    assert evaluate_gates(contract, **values).measurement.status == "failed"
    values["telemetry"].throttling_detected = None
    result = evaluate_gates(contract, **values)
    assert result.measurement.status == "inconclusive"
    assert "measurement_limitation:throttling_detected_not_measured" in result.measurement.reasons


def test_partial_or_unavailable_telemetry_preserves_measurement_limitation():
    contract, values = evidence()
    values["telemetry"].window.start_ts = 12
    values["telemetry"].coverage_ratio = 0.8
    assert evaluate_gates(contract, **values).measurement.status == "inconclusive"
    values["telemetry"] = None
    assert evaluate_gates(contract, **values).status == "inconclusive"
    contract.require_telemetry = False
    result = evaluate_gates(contract, **values)
    assert result.status == "passed"
    assert result.measurement.status == "not_applicable"


def test_correctness_from_another_run_cannot_validate_current_candidate():
    contract, values = evidence()
    values["correctness"].run_id = "another-run"
    assert evaluate_gates(contract, **values).correctness.status == "failed"


def test_pre_timing_telemetry_cannot_cover_a_later_measured_window():
    contract, values = evidence()
    contract.measurement_uses_execution_window = False
    contract.measurement_window = MeasurementWindow(start_ts=30, end_ts=40)
    result = evaluate_gates(contract, **values)
    assert result.execution.status == "passed"
    assert result.measurement.status == "inconclusive"
    assert "measurement_limitation:telemetry_summary_window_mismatch" in result.measurement.reasons
    # A wider 100-second summary may hide all missing samples in the measured interval.
    values["telemetry"].window = MeasurementWindow(start_ts=0, end_ts=100)
    values["telemetry"].coverage_ratio = 0.95
    assert evaluate_gates(contract, **values).measurement.status == "inconclusive"
    values["telemetry"].window = MeasurementWindow(start_ts=30, end_ts=40)
    values["telemetry"].coverage_ratio = 1.0
    assert evaluate_gates(contract, **values).measurement.status == "passed"


def test_aggregate_correctness_requires_full_input_coverage_and_frozen_oracle():
    contract, values = evidence()
    contract.oracle_manifest_sha256 = "d" * 64
    contract.correctness.required_cases = ["frozen-oracle-suite"]
    values["correctness"] = CorrectnessEvidence(
        run_id=contract.execution.run_id,
        evidence_ref="trusted/raw-verdict.json",
        cases=[
            {
                "case_id": "frozen-oracle-suite",
                "scope": "aggregate",
                "input_ids": ["packed", "idle-rank"],
                "errors": {"max_abs": 0.0001},
                "passed": True,
                "oracle_manifest_sha256": "d" * 64,
                "coverage_evidence_ref": "trusted/frozen-oracle-assertion-coverage.json",
            }
        ],
    )
    assert evaluate_gates(contract, **values).correctness.status == "passed"
    values["correctness"].cases[0].input_ids.pop()
    assert evaluate_gates(contract, **values).correctness.status == "failed"
    values["correctness"].cases[0].input_ids.append("idle-rank")
    values["correctness"].cases[0].errors.clear()
    assert evaluate_gates(contract, **values).correctness.status == "inconclusive"


def test_c2_adapter_preserves_missing_registers_and_binds_device_selection():
    contract, values = evidence()
    payload = {
        "server": {"host": "contract-fixture-host", "id": 1, "name": "fixture"},
        "window": {"start_ts": 10, "end_ts": 20, "requested_devices": [0], "raw_sample_count": 3},
        "coverage_ratio": 1.0,
        "throttling_detected": None,
        "resource_interference_detected": None,
        "freq_mhz_min": None,
        "power_max_w": 230,
        "temp_max_c": 60,
        "measurement_limitations": ["explicit frequency and throttling registers not captured"],
        "devices": [],
    }
    values["telemetry"] = telemetry_from_c2(payload, run_id="run-candidate", evidence_ref="c2.json")
    result = evaluate_gates(contract, **values)
    assert result.measurement.status == "inconclusive"
    assert values["telemetry"].freq_mhz_min is None
    values["telemetry"].devices = [1]
    assert evaluate_gates(contract, **values).measurement.status == "failed"


def test_zero_leave_one_out_delta_does_not_replace_final_combination_verification():
    contract, values = evidence()
    contract.require_ablation = True

    def run(identity, active, duration):
        return measured_run(identity, active, duration)

    ablation = AblationGate().evaluate(
        task_id=contract.task_id,
        factors=[AblationFactor(factor_id="A"), AblationFactor(factor_id="B")],
        baseline=run("base", [], 100),
        full_candidate=run(contract.execution.run_id, ["A", "B"], 90),
        ablations=[run("without-A", ["B"], 90), run("without-B", ["A"], 90)],
    )
    assert ablation.status == "passed"
    assert not ablation.pruned_combination_verified
    assert all(factor.disposition == "candidate_for_sequential_pruning" for factor in ablation.factors)
    assert ablation.pruned_combination.retained_factors == ["A", "B"]
    result = evaluate_gates(contract, **values, ablation=ablation)
    assert result.ablation.status == "inconclusive"
    assert result.status == "inconclusive"
    assert result.ablation.reasons == ["final_combination_not_verified"]


def test_all_retained_final_run_can_pass_ablation_gate_and_binds_execution_identity():
    contract, values = evidence()
    contract.require_ablation = True

    def measured(identity, active, duration):
        return measured_run(identity, active, duration)

    ablation = AblationGate(min_speedup=1.05).evaluate(
        task_id=contract.task_id,
        factors=[AblationFactor(factor_id="A")],
        baseline=measured("base", [], 100),
        full_candidate=measured("initial-full", ["A"], 60),
        ablations=[measured("without-A", [], 100)],
        final_candidate=measured(contract.execution.run_id, ["A"], 60),
    )
    assert ablation.final_combination_verified and not ablation.pruned_combination_verified
    assert evaluate_gates(contract, **values, ablation=ablation).ablation.status == "passed"
    changed = ablation.model_copy(deep=True)
    changed.runs[-1].timing.samples[0].started_ns += 1
    changed.runs[-1].timing.samples[0].ended_ns += 1
    rejected_samples = evaluate_gates(contract, **values, ablation=changed)
    assert rejected_samples.ablation.status == "failed"
    assert "ablation_comparison_sample_binding_mismatch" in rejected_samples.ablation.reasons
    ablation.final_combination.verification_run_id = "another-final-run"
    rejected = evaluate_gates(contract, **values, ablation=ablation)
    assert rejected.ablation.status == "failed"
    assert rejected.ablation.reasons == ["ablation_candidate_identity_mismatch"]
