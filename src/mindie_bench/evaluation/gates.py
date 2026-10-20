"""Task-scoped execution, correctness, measurement and ablation gates.

[Role] Bind independently captured evidence to a declared candidate and scope.
[Invariants] Missing telemetry is inconclusive, not healthy; names alone do not
prove candidate execution. Gate success covers only the declared task contract.
[Timeouts & Locks] Pure evaluation; all collection belongs to bounded callers.
[Side-Effect Policy] No candidate imports, runtime control or hardware writes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from mindie_bench.evaluation.ablation import AblationReceipt
from mindie_bench.evaluation.ablation_statistics import (
    compare_timings,
    observed_lower_bounds_pass,
    timing_identity,
)
from mindie_bench.evaluation.performance import TimingRun, validate_performance
from mindie_bench.models import PerformanceContract, StrictModel, VerifierResult

SHA = r"^[0-9a-f]{64}$"


class MeasurementWindow(StrictModel):
    start_ts: float
    end_ts: float

    @model_validator(mode="after")
    def ordered(self):
        if self.end_ts <= self.start_ts:
            raise ValueError("measurement window must have positive duration")
        return self


class ExecutionContract(StrictModel):
    run_id: str
    patch_digest: str = Field(pattern=SHA)
    binary_sha256: str = Field(pattern=SHA)
    module_path: str
    target_call_path: str
    input_ids: list[str] = Field(min_length=1)
    window: MeasurementWindow


class ExecutionCall(StrictModel):
    input_id: str
    call_path: str
    timestamp: float


class ExecutionEvidence(StrictModel):
    run_id: str
    patch_digest: str = Field(pattern=SHA)
    loaded_binary_sha256: str = Field(pattern=SHA)
    loaded_module_path: str
    calls: list[ExecutionCall]
    evidence_ref: str


class CorrectnessContract(StrictModel):
    # Cases reflect the task's supported paths, e.g. packed-varlen or idle-dp;
    # unrelated kernels must not acquire logits or distributed requirements.
    required_cases: list[str] = Field(min_length=1)
    max_error: dict[str, float] = Field(min_length=1)


class CorrectnessCase(StrictModel):
    case_id: str
    input_id: str | None = None
    input_ids: list[str] = Field(default_factory=list)
    scope: Literal["per_input", "aggregate"] = "per_input"
    oracle_manifest_sha256: str | None = Field(default=None, pattern=SHA)
    coverage_evidence_ref: str | None = None
    errors: dict[str, float]
    passed: bool


class CorrectnessEvidence(StrictModel):
    run_id: str
    cases: list[CorrectnessCase]
    evidence_ref: str


class TelemetryEvidence(StrictModel):
    run_id: str
    host: str
    devices: list[int] = Field(min_length=1)
    window: MeasurementWindow
    coverage_ratio: float = Field(ge=0, le=1)
    throttling_detected: bool | None = None
    resource_interference_detected: bool | None = None
    freq_mhz_min: float | None = Field(default=None, ge=0)
    power_max_w: float | None = Field(default=None, ge=0)
    temp_max_c: float | None = None
    measurement_limitations: list[str] = Field(default_factory=list)
    evidence_ref: str


class GateContract(StrictModel):
    task_id: str
    execution: ExecutionContract
    correctness: CorrectnessContract
    performance: PerformanceContract | None = None
    performance_nonce: str | None = None
    oracle_manifest_sha256: str | None = Field(default=None, pattern=SHA)
    require_telemetry: bool = False
    telemetry_host: str | None = None
    telemetry_devices: list[int] = Field(default_factory=list)
    min_telemetry_coverage: float = Field(default=0.95, gt=0, le=1)
    measurement_window: MeasurementWindow | None = None
    # Older callers may opt in when execution and measurement truly share a window.
    measurement_uses_execution_window: bool = False
    require_ablation: bool = False
    max_ablation_relative_loss: float = Field(default=0.015, ge=0, lt=1)
    performance_reference_label: str = "verifier baseline"
    performance_candidate_label: str = "verifier candidate"

    @model_validator(mode="after")
    def performance_binding(self):
        if self.performance is not None and (
            not self.performance_nonce or not self.oracle_manifest_sha256
        ):
            raise ValueError("performance gate requires the judge nonce and frozen oracle identity")
        if self.require_telemetry and (not self.telemetry_host or not self.telemetry_devices):
            raise ValueError("telemetry gate requires the measured host and device selection")
        if self.measurement_window is not None and self.measurement_uses_execution_window:
            raise ValueError("declare one measurement window source")
        return self


class GateResult(StrictModel):
    status: Literal["passed", "failed", "inconclusive", "not_applicable"]
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class GateReport(StrictModel):
    schema_version: Literal["mindie-bench-gates/1"] = "mindie-bench-gates/1"
    task_id: str
    run_id: str
    status: Literal["passed", "failed", "inconclusive"]
    execution: GateResult
    correctness: GateResult
    measurement: GateResult
    ablation: GateResult
    performance_comparison: dict | None = None
    claim_scope: str = (
        "declared inputs, metrics and measurement window; semantic review remains separate"
    )


class GateRequest(StrictModel):
    contract: GateContract
    execution: ExecutionEvidence | None = None
    correctness: CorrectnessEvidence | None = None
    telemetry: TelemetryEvidence | None = None
    performance_verdict: VerifierResult | None = None
    ablation: AblationReceipt | None = None

    def evaluate(self) -> GateReport:
        return evaluate_gates(
            self.contract,
            execution=self.execution,
            correctness=self.correctness,
            telemetry=self.telemetry,
            performance_verdict=self.performance_verdict,
            ablation=self.ablation,
        )


def telemetry_from_c2(payload: dict, *, run_id: str, evidence_ref: str) -> TelemetryEvidence:
    """Consume npu-top's C2 window without filling absent registers with zeros."""
    window = payload["window"]
    return TelemetryEvidence(
        run_id=run_id,
        host=payload["server"]["host"],
        devices=window["requested_devices"],
        window=MeasurementWindow(start_ts=window["start_ts"], end_ts=window["end_ts"]),
        coverage_ratio=payload["coverage_ratio"],
        throttling_detected=payload.get("throttling_detected"),
        resource_interference_detected=payload.get("resource_interference_detected"),
        freq_mhz_min=payload.get("freq_mhz_min"),
        power_max_w=payload.get("power_max_w"),
        temp_max_c=payload.get("temp_max_c"),
        measurement_limitations=payload["measurement_limitations"],
        evidence_ref=evidence_ref,
    )


def execution_gate(contract: ExecutionContract, evidence: ExecutionEvidence | None) -> GateResult:
    if evidence is None:
        return GateResult(status="inconclusive", reasons=["execution_evidence_missing"])
    reasons = []
    for label, actual, expected in (
        ("run_id", evidence.run_id, contract.run_id),
        ("patch_digest", evidence.patch_digest, contract.patch_digest),
        ("binary_sha256", evidence.loaded_binary_sha256, contract.binary_sha256),
        ("module_path", evidence.loaded_module_path, contract.module_path),
    ):
        if actual != expected:
            reasons.append(f"candidate_identity_mismatch:{label}")
    observed_inputs = {
        call.input_id
        for call in evidence.calls
        if call.call_path == contract.target_call_path
        and contract.window.start_ts <= call.timestamp <= contract.window.end_ts
    }
    if not set(contract.input_ids) <= observed_inputs:
        reasons.append("target_calls_missing_for_inputs_in_measurement_window")
    return GateResult(
        status="failed" if reasons else "passed", reasons=reasons, evidence_refs=[evidence.evidence_ref]
    )


def correctness_gate(contract: GateContract, evidence: CorrectnessEvidence | None) -> GateResult:
    if evidence is None:
        return GateResult(status="inconclusive", reasons=["correctness_evidence_missing"])
    reasons = []
    if evidence.run_id != contract.execution.run_id:
        reasons.append("correctness_run_id_mismatch")
    cases = {case.case_id: case for case in evidence.cases}
    if len(cases) != len(evidence.cases):
        reasons.append("duplicate_correctness_case")
    missing = set(contract.correctness.required_cases) - cases.keys()
    for case_id in set(contract.correctness.required_cases) & cases.keys():
        case = cases[case_id]
        if case.scope == "aggregate":
            if set(case.input_ids) != set(contract.execution.input_ids) or case.input_id is not None:
                reasons.append(f"aggregate_correctness_coverage_mismatch:{case_id}")
            if not case.coverage_evidence_ref:
                missing.add(f"{case_id}:aggregate_coverage_evidence")
            if (
                not contract.oracle_manifest_sha256
                or case.oracle_manifest_sha256 != contract.oracle_manifest_sha256
            ):
                reasons.append(f"aggregate_correctness_oracle_mismatch:{case_id}")
        elif case.input_id not in contract.execution.input_ids or case.input_ids:
            reasons.append(f"correctness_input_not_bound:{case_id}")
        if not case.passed:
            reasons.append(f"correctness_failed:{case_id}")
        for metric, limit in contract.correctness.max_error.items():
            if metric not in case.errors:
                missing.add(f"{case_id}:{metric}")
            elif not 0 <= case.errors[metric] <= limit:
                reasons.append(f"error_above_contract:{case_id}:{metric}")
    return GateResult(
        status="failed" if reasons else "inconclusive" if missing else "passed",
        reasons=reasons + [f"correctness_not_measured:{item}" for item in sorted(missing)],
        evidence_refs=[evidence.evidence_ref],
    )


def measurement_gate(
    contract: GateContract,
    telemetry: TelemetryEvidence | None,
    verdict: VerifierResult | None,
) -> GateResult:
    if contract.performance is None and not contract.require_telemetry:
        return GateResult(status="not_applicable")
    failed, limitations, refs = [], [], []
    measured_window = contract.measurement_window
    if measured_window is None and contract.measurement_uses_execution_window:
        measured_window = contract.execution.window
    if measured_window is None:
        limitations.append("measurement_window_not_declared")
    if contract.performance is not None:
        if verdict is None:
            limitations.append("performance_evidence_missing")
        else:
            # Retain the existing nonce, oracle, rotating-buffer and sync checks.
            # Malformed/mismatched judge evidence remains a visible caller error.
            result = validate_performance(
                verdict,
                contract.performance,
                run_nonce=contract.performance_nonce,
                oracle_manifest_sha256=contract.oracle_manifest_sha256,
            )
            failed.extend(result["failed_gates"])
    if contract.require_telemetry:
        if telemetry is None:
            limitations.append("telemetry_missing")
        else:
            refs.append(telemetry.evidence_ref)
            if telemetry.run_id != contract.execution.run_id:
                failed.append("telemetry_run_id_mismatch")
            if telemetry.host != contract.telemetry_host or set(telemetry.devices) != set(
                contract.telemetry_devices
            ):
                failed.append("telemetry_device_selection_mismatch")
            if measured_window is not None and (
                telemetry.window.start_ts != measured_window.start_ts
                or telemetry.window.end_ts != measured_window.end_ts
            ):
                limitations.append("telemetry_summary_window_mismatch")
            if telemetry.coverage_ratio < contract.min_telemetry_coverage:
                limitations.append("insufficient_telemetry_coverage")
            limitations.extend(telemetry.measurement_limitations)
            for name in ("throttling_detected", "resource_interference_detected"):
                value = getattr(telemetry, name)
                if value is True:
                    failed.append(name)
                elif value is None:
                    limitations.append(f"{name}_not_measured")
            for name in ("freq_mhz_min", "power_max_w"):
                if getattr(telemetry, name) is None:
                    limitations.append(f"{name}_not_measured")
            # Temperature alone is neither a throttling detector nor a health proof.
    return GateResult(
        status="failed" if failed else "inconclusive" if limitations else "passed",
        reasons=failed + [f"measurement_limitation:{item}" for item in limitations],
        evidence_refs=refs,
    )


def evaluate_gates(
    contract: GateContract,
    *,
    execution: ExecutionEvidence | None = None,
    correctness: CorrectnessEvidence | None = None,
    telemetry: TelemetryEvidence | None = None,
    performance_verdict: VerifierResult | None = None,
    ablation: AblationReceipt | None = None,
) -> GateReport:
    if not contract.require_ablation:
        ablation_result = GateResult(status="not_applicable")
    elif ablation is None:
        ablation_result = GateResult(status="inconclusive", reasons=["ablation_evidence_missing"])
    else:
        final = ablation.final_combination or ablation.pruned_combination
        verified = (
            ablation.schema_version == "mindie-bench-ablation/2"
            and ablation.final_combination_verified
            and final.correctness_verified
            and final.performance_verified
            and final.uncertainty_verified
        )
        binding_errors = []
        if verified:
            runs = {run.run_id: run for run in ablation.runs}
            baseline, initial, measured = (
                runs.get(identity)
                for identity in (
                    ablation.baseline_run_id,
                    ablation.full_candidate_run_id,
                    final.verification_run_id,
                )
            )
            if baseline is None or initial is None or measured is None:
                binding_errors.append("ablation_comparison_run_missing")
            elif (
                compare_timings(baseline, measured) != final.baseline_comparison
                or compare_timings(initial, measured) != final.retention_comparison
                or measured.timing is None
                or timing_identity(measured.timing) != final.final_timing_sha256
            ):
                binding_errors.append("ablation_comparison_sample_binding_mismatch")
            if not observed_lower_bounds_pass(
                final.baseline_comparison, ablation.min_speedup
            ) or not observed_lower_bounds_pass(
                final.retention_comparison, 1 - ablation.max_relative_loss
            ):
                binding_errors.append("ablation_uncertainty_requirement_not_met")
            if ablation.max_relative_loss > contract.max_ablation_relative_loss:
                binding_errors.append("ablation_retention_threshold_mismatch")
            if contract.performance is not None:
                if ablation.min_speedup < contract.performance.min_speedup_ratio:
                    binding_errors.append("ablation_speed_threshold_mismatch")
                candidate = (
                    (performance_verdict.performance_evidence or {}).get("candidate")
                    if performance_verdict
                    else None
                )
                if candidate is None:
                    binding_errors.append("ablation_final_performance_samples_missing")
                else:
                    try:
                        candidate_sha = timing_identity(TimingRun.model_validate(candidate))
                    except ValueError:
                        binding_errors.append("ablation_final_performance_samples_invalid")
                    else:
                        if candidate_sha != final.final_timing_sha256:
                            binding_errors.append("ablation_final_performance_samples_mismatch")
        if (
            ablation.task_id != contract.task_id
            or final.verification_run_id != contract.execution.run_id
        ):
            ablation_result = GateResult(
                status="failed", reasons=["ablation_candidate_identity_mismatch"]
            )
        elif binding_errors:
            ablation_result = GateResult(status="failed", reasons=binding_errors)
        elif ablation.status == "passed" and not verified:
            ablation_result = GateResult(
                status="inconclusive", reasons=["final_combination_not_verified"]
            )
        else:
            ablation_result = GateResult(status=ablation.status, reasons=ablation.reasons)
    gates = dict(
        execution=execution_gate(contract.execution, execution),
        correctness=correctness_gate(contract, correctness),
        measurement=measurement_gate(contract, telemetry, performance_verdict),
        ablation=ablation_result,
    )
    statuses = {gate.status for gate in gates.values()}
    comparison = None
    if performance_verdict and performance_verdict.performance_evidence:
        evidence = performance_verdict.performance_evidence
        try:
            reference_sha = timing_identity(TimingRun.model_validate(evidence.get("baseline")))
            candidate_sha = timing_identity(TimingRun.model_validate(evidence.get("candidate")))
        except ValueError:
            comparison = {"comparison_status": "inconclusive", "reason": "timing_identity_invalid"}
        else:
            comparison = {
                "reference_label": contract.performance_reference_label,
                "candidate_label": contract.performance_candidate_label,
                "estimator": "geomean_of_per_case_arithmetic_mean_ratios",
                "speedup_geomean": performance_verdict.speedup_ratio,
                "reference_timing_sha256": reference_sha,
                "candidate_timing_sha256": candidate_sha,
                "comparison_status": gates["measurement"].status,
            }
    return GateReport(
        task_id=contract.task_id,
        run_id=contract.execution.run_id,
        status="failed"
        if "failed" in statuses
        else "inconclusive"
        if "inconclusive" in statuses
        else "passed",
        performance_comparison=comparison,
        **gates,
    )
