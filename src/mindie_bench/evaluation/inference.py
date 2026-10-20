"""Reusable native judge primitives for support, optimization and regression tasks.

[Role] Collect counterbalanced A/B measurements and derive lifecycle acceptance.
[Invariants] Every declared guard is measured; absent evidence never becomes 1x.
[Timeouts & Locks] No scheduler or retry loop; runs inside the existing isolated judge.
[Side-Effect Policy] Only caller-supplied trusted measurement callbacks execute.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable

from mindie_bench.inference_models import (
    InferenceContract,
    InferenceEvidence,
    InferenceWorkload,
    PairedMeasurement,
    WorkloadMeasurements,
)
from mindie_bench.io import digest


class CandidateMeasurementFailure(RuntimeError):
    """A judge-established candidate business failure, not a transport/device fault."""


def measure_latency_ms(operation: Callable, synchronize: Callable, *, clock=time.perf_counter_ns) -> float:
    """Time a completed operation; device completion belongs to synchronize()."""
    synchronize()
    start = clock()
    operation()
    synchronize()
    elapsed = clock() - start
    if elapsed <= 0:
        raise ValueError("measurement clock did not advance")
    return elapsed / 1_000_000


def measure_workloads(
    contract: InferenceContract,
    measure: Callable[[str, InferenceWorkload], float],
) -> dict[str, WorkloadMeasurements]:
    """Warm each arm, then alternate AB/BA in paired rounds for every workload.

    The callback measures the declared metric using the same requests/settings
    for both arms. It owns server readiness and device synchronization. Only an
    explicitly classified candidate business failure becomes a failed arm;
    baseline or unexpected errors propagate as infrastructure faults.
    """
    observations = {}
    for workload in contract.workloads:
        warmup_error = None
        for _ in range(contract.warmup_rounds):
            for arm in ("baseline", "candidate"):
                try:
                    value = measure(arm, workload)
                    # Apply the same finite/positive numeric contract to warmup.
                    PairedMeasurement(round=0, order="baseline_candidate", baseline=value, candidate=value)
                except CandidateMeasurementFailure as error:
                    if arm == "baseline":
                        raise
                    warmup_error = str(error) or "candidate warmup failed"
        pairs = []
        for index in range(contract.measurement_rounds):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            values, candidate_error = {}, warmup_error
            for arm in order:
                if arm == "candidate" and warmup_error:
                    values[arm] = None
                    continue
                try:
                    values[arm] = measure(arm, workload)
                except CandidateMeasurementFailure as error:
                    if arm == "baseline":
                        raise
                    values[arm] = None
                    candidate_error = str(error) or "candidate measurement failed"
            pairs.append(PairedMeasurement(
                round=index, order="_".join(order), baseline=values["baseline"],
                candidate=values["candidate"], candidate_error=candidate_error,
            ))
        observations[workload.name] = WorkloadMeasurements(
            workload_sha256=workload.workload_sha256,
            warmup_rounds=contract.warmup_rounds, pairs=pairs,
        )
    return observations


def evaluate_inference(contract: InferenceContract, evidence: InferenceEvidence) -> dict:
    """Recompute from paired observations; no agent-supplied scalar score is trusted."""
    for key in ("baseline_commit", "baseline_source_sha256", "runtime_sha256", "device_class"):
        if getattr(evidence, key) != getattr(contract, key):
            raise ValueError(f"inference {key} differs from the declared baseline/runtime")
    if not evidence.baseline_healthy:
        raise ValueError("inference baseline is unhealthy; no business score is available")
    if set(evidence.checks) != set(contract.required_checks):
        raise ValueError("inference correctness evidence must cover exactly the required checks")
    correctness = all(evidence.checks.values())
    result = dict(
        category=contract.category,
        contract_sha256=digest(contract.model_dump(mode="json")),
        evidence_sha256=digest(evidence.model_dump(mode="json")),
        correctness_passed=correctness,
        test_pass_fraction=sum(evidence.checks.values()) / len(evidence.checks),
        measured_speedup=None, score=None, accepted=False, failed_gates=[], workloads={},
    )
    if contract.category == "model_support":
        if evidence.workloads:
            raise ValueError("support evidence must not contain undeclared workloads")
        result.update(score=float(correctness), accepted=correctness,
                      failed_gates=[] if correctness else ["correctness_failed"])
        return result
    # Correctness rejection is a measured failure. Performance need not run, but
    # any partial diagnostics are deliberately not used as a speedup score.
    if not correctness:
        result.update(score=1.0, failed_gates=["correctness_failed"])
        return result
    if evidence.candidate_sha256 == contract.baseline_source_sha256:
        result.update(score=1.0, failed_gates=["unchanged_candidate"])
        return result
    if set(evidence.workloads) != {workload.name for workload in contract.workloads}:
        raise ValueError("inference evidence must cover every scored AND guard workload")
    scored, failed, candidate_failed = [], [], False
    for workload in contract.workloads:
        measured = evidence.workloads[workload.name]
        if (measured.workload_sha256 != workload.workload_sha256
                or measured.warmup_rounds != contract.warmup_rounds):
            raise ValueError("inference workload identity or warmup differs from its contract")
        if len(measured.pairs) != contract.measurement_rounds or any(
            pair.round != index or pair.order != (
                "baseline_candidate" if index % 2 == 0 else "candidate_baseline"
            ) for index, pair in enumerate(measured.pairs)
        ):
            raise ValueError("inference measurement needs complete counterbalanced paired rounds")
        ratios = [
            (pair.candidate / pair.baseline if workload.direction == "higher"
             else pair.baseline / pair.candidate)
            for pair in measured.pairs if pair.candidate is not None
        ]
        if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
            raise ValueError("derived performance ratios must remain finite and positive")
        complete = len(ratios) == contract.measurement_rounds
        ratio = statistics.median(ratios) if complete else None
        candidate_failed |= not complete
        guard_passed = ratio >= workload.min_ratio if ratio is not None and workload.role == "guard" else None
        result["workloads"][workload.name] = dict(
            role=workload.role, metric=workload.metric, direction=workload.direction,
            paired_ratios=ratios, ratio=ratio, guard_passed=guard_passed,
        )
        if workload.role == "guard" and guard_passed is False:
            failed.append("guard_regressed:" + workload.name)
        elif workload.role == "scored" and ratio is not None:
            scored.append(ratio)
    if candidate_failed:
        result.update(score=1.0, failed_gates=["candidate_measurement_failed", *failed])
        return result
    speedup = statistics.geometric_mean(scored)
    if speedup < contract.min_speedup_ratio:
        failed.append("speedup_below_requirement")
    # Correct measured slowdowns remain visible. A guard failure rejects the
    # candidate even if a scored workload improves; diagnostics retain the ratio.
    result.update(
        measured_speedup=speedup, score=1.0 if any(f.startswith("guard_") for f in failed) else speedup,
        accepted=not failed, failed_gates=failed,
    )
    return result


def validate_inference(contract, evidence, *, run_nonce, oracle_manifest_sha256) -> dict:
    evidence = InferenceEvidence.model_validate(evidence)
    if evidence.run_nonce != run_nonce or evidence.oracle_manifest_sha256 != oracle_manifest_sha256:
        raise ValueError("inference evidence belongs to another run or frozen oracle")
    return evaluate_inference(contract, evidence)
