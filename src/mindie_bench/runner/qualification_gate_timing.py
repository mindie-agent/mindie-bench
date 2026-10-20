"""Sustained ABBA and producer-consumer module gate evaluation.

[Role]
Evaluate G2 paired interleaved ACLGraph timing evidence and G3 subgraph / TP
module paired verification without executing commands or touching a device.

[Invariants]
- Numerical qualification is decided before latency; a failed oracle leaves
  latency unmeasured rather than reporting a speedup.
- G2 accepts only warmup plus multi-round `baseline, candidate, candidate,
  baseline` ordering, a frozen baseline digest and sustained-clock evidence.
  Short-burst clocks never become sustained optimization claims.
- G3 rejects a module when a required guard, boundary ownership check, executed
  path or paired hidden-input identity is missing or failed.
- Missing guard measurement is `unknown`, not healthy.

[Timeouts & Locks]
- Pure arithmetic; the caller supplies an already validated timeout plan.

[Side-Effect Policy]
- No subprocess, file write, model call or hardware access.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from typing import Any

from mindie_bench.runner.qualification_gate_models import (
    G2Spec,
    G3Spec,
    GateResult,
    GateState,
    StageReceipt,
)


def frozen_baseline_digest(spec: G2Spec) -> str:
    """Bind the frozen baseline identity to the exact measured baseline rows."""
    baseline = [
        (sample.round_index, sample.phase, sample.duration_ns)
        for sample in spec.samples
        if sample.variant == "baseline"
    ]
    payload = json.dumps(
        {
            "baseline_id": spec.baseline_id,
            "rounds": spec.rounds,
            "warmup_rounds": spec.warmup_rounds,
            "baseline": sorted(baseline),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluate_g2_sustained_abba(
    spec: G2Spec, *, timeout_plan: dict[str, Any] | None = None
) -> GateResult:
    if type(spec.rounds) is not int or spec.rounds < 1:
        return GateResult(
            "G2",
            GateState.FAILED,
            [StageReceipt("interleaving", GateState.FAILED,
                          "rounds must be a positive integer")],
            timeout_plan=timeout_plan,
        )
    if type(spec.warmup_rounds) is not int or spec.warmup_rounds < 1:
        return GateResult(
            "G2",
            GateState.FAILED,
            [StageReceipt("interleaving", GateState.FAILED,
                          "at least one warmup round is required")],
            timeout_plan=timeout_plan,
        )
    stages: list[StageReceipt] = []
    expected = ("baseline", "candidate", "candidate", "baseline")
    pattern_problems = []
    for round_index in range(1, spec.rounds + 1):
        measured = [
            sample
            for sample in spec.samples
            if sample.phase == "measured" and sample.round_index == round_index
        ]
        if tuple(sample.variant for sample in measured) != expected:
            pattern_problems.append(
                f"round {round_index}: {tuple(sample.variant for sample in measured)}"
            )
        if any(sample.duration_ns <= 0 for sample in measured):
            pattern_problems.append(f"round {round_index}: nonpositive duration")
        if any(sample.aclgraph_replays < 1 for sample in measured):
            pattern_problems.append(f"round {round_index}: no ACLGraph replay")
    for round_index in range(1, spec.warmup_rounds + 1):
        warm = {
            sample.variant
            for sample in spec.samples
            if sample.phase == "warmup" and sample.round_index == round_index
        }
        if warm != {"baseline", "candidate"}:
            pattern_problems.append(f"warmup round {round_index}: variants={sorted(warm)}")
    stages.append(
        StageReceipt(
            "interleaving",
            GateState.FAILED if pattern_problems else GateState.COMPLETED,
            "; ".join(pattern_problems)
            if pattern_problems
            else "warmup plus ABBA baseline/candidate/candidate/baseline ordering verified",
        )
    )
    expected_digest = frozen_baseline_digest(spec)
    frozen_ok = (
        bool(re.fullmatch(r"[0-9a-f]{64}", spec.frozen_baseline_sha256))
        and expected_digest == spec.frozen_baseline_sha256
    )
    stages.append(
        StageReceipt(
            "frozen_baseline",
            GateState.COMPLETED if frozen_ok else GateState.FAILED,
            "frozen baseline digest matches"
            if frozen_ok
            else "baseline timings do not match the frozen baseline digest",
            {"expected": spec.frozen_baseline_sha256, "observed": expected_digest},
        )
    )
    numerical = spec.numerical
    numerical_ok = numerical.qualified
    stages.append(
        StageReceipt(
            "numerical",
            GateState.COMPLETED if numerical_ok else GateState.FAILED,
            "baseline and candidate satisfy numerical plus sentinel oracle"
            if numerical_ok
            else "numerical qualification failed; latency is not admissible",
            {
                "baseline_passed": numerical.baseline_passed,
                "candidate_passed": numerical.candidate_passed,
                "sentinel_ok": numerical.sentinel_ok,
                "max_abs_error": numerical.max_abs_error,
                "tolerance": numerical.tolerance,
                "cosine_similarity": numerical.cosine_similarity,
            },
        )
    )
    measured = [sample for sample in spec.samples if sample.phase == "measured"]
    total_measured_sec = sum(sample.duration_ns for sample in measured) / 1_000_000_000
    clocks = [sample.clock_mhz for sample in measured]
    sustained_problems = []
    if total_measured_sec < spec.min_sustained_window_sec:
        sustained_problems.append(
            f"measured window {total_measured_sec:.3f}s < {spec.min_sustained_window_sec:g}s"
        )
    if any(not sample.sustained for sample in measured):
        sustained_problems.append("a measured sample is marked burst-only")
    if spec.sustained_clock_mhz is not None:
        low, high = spec.sustained_clock_mhz
        if any(clock is None or not low <= clock <= high for clock in clocks):
            sustained_problems.append(
                f"clock evidence outside sustained range [{low:g}, {high:g}] MHz"
            )
    sustained_ok = not sustained_problems and bool(measured)
    stages.append(
        StageReceipt(
            "sustained",
            GateState.COMPLETED if sustained_ok else GateState.FAILED,
            "; ".join(sustained_problems)
            if sustained_problems
            else "sustained multi-round window accepted",
            {"measured_window_sec": round(total_measured_sec, 6), "clocks_mhz": clocks},
        )
    )
    round_speedups: list[float] = []
    baseline_medians: list[float] = []
    candidate_medians: list[float] = []
    for round_index in range(1, spec.rounds + 1):
        before = [
            sample.duration_ns
            for sample in measured
            if sample.round_index == round_index and sample.variant == "baseline"
        ]
        after = [
            sample.duration_ns
            for sample in measured
            if sample.round_index == round_index and sample.variant == "candidate"
        ]
        if len(before) == 2 and len(after) == 2:
            base_median, candidate_median = statistics.median(before), statistics.median(after)
            baseline_medians.append(base_median)
            candidate_medians.append(candidate_median)
            round_speedups.append(base_median / candidate_median)
    speedup = statistics.geometric_mean(round_speedups) if round_speedups else None
    if not numerical_ok:
        latency_state = GateState.UNKNOWN
        latency_detail = "latency not evaluated because numerical qualification failed"
    elif not sustained_ok:
        latency_state = GateState.FAILED
        latency_detail = "short-burst or non-sustained timing cannot support an optimization claim"
    elif speedup is None or speedup < spec.min_speedup_ratio:
        latency_state = GateState.FAILED
        latency_detail = "sustained latency does not reach the declared ratio"
    else:
        latency_state = GateState.COMPLETED
        latency_detail = "sustained paired latency improvement accepted"
    stages.append(
        StageReceipt(
            "latency",
            latency_state,
            latency_detail,
            {
                "round_speedups": [round(value, 6) for value in round_speedups],
                "geometric_speedup": speedup,
                "min_speedup_ratio": spec.min_speedup_ratio,
                "baseline_round_medians_ns": baseline_medians,
                "candidate_round_medians_ns": candidate_medians,
            },
        )
    )
    accepted = (
        numerical_ok
        and sustained_ok
        and speedup is not None
        and speedup >= spec.min_speedup_ratio
    )
    if any(stage.state is GateState.FAILED for stage in stages):
        state = GateState.FAILED
    elif any(stage.state is GateState.UNKNOWN for stage in stages):
        state = GateState.UNKNOWN
    else:
        state = GateState.COMPLETED
    return GateResult(
        "G2",
        state,
        stages,
        {
            "numerical_qualified": numerical_ok,
            "sustained_valid": sustained_ok,
            "geometric_speedup": speedup,
            "accepted": accepted,
        },
        [f"{stage.name}: {stage.detail}" for stage in stages
         if stage.state is not GateState.COMPLETED],
        timeout_plan,
    )


def evaluate_g3_subgraph_module_gate(
    spec: G3Spec, *, timeout_plan: dict[str, Any] | None = None
) -> GateResult:
    stages: list[StageReceipt] = []
    boundary_problems = []
    if not spec.boundary_pairs:
        boundary_problems.append("no producer-consumer boundary pair declared")
    for pair in spec.boundary_pairs:
        if pair.logical_identity != pair.observed_identity or not pair.equivalent:
            boundary_problems.append(
                f"{pair.producer}->{pair.consumer}: logical boundary is not equivalent"
            )
        if pair.executed_calls < 1:
            boundary_problems.append(
                f"{pair.producer}->{pair.consumer}: consumer branch never executed"
            )
        if not pair.ownership_preserved:
            boundary_problems.append(
                f"{pair.producer}->{pair.consumer}: tensor ownership changed across boundary"
            )
        if not pair.numerical_passed:
            boundary_problems.append(
                f"{pair.producer}->{pair.consumer}: boundary numerical check failed"
            )
    stages.append(
        StageReceipt(
            "boundary",
            GateState.FAILED if boundary_problems else GateState.COMPLETED,
            "; ".join(boundary_problems)
            if boundary_problems
            else "producer-consumer logical boundary and ownership verified",
        )
    )
    runs = {run.variant: run for run in spec.paired_runs}
    module_problems = []
    if set(runs) != {"baseline", "candidate"}:
        module_problems.append("both baseline and candidate module runs are required")
    else:
        before, after = runs["baseline"], runs["candidate"]
        if before.inputs_sha256 != after.inputs_sha256:
            module_problems.append("paired module runs used different hidden inputs")
        if not before.numerical_passed or not after.numerical_passed:
            module_problems.append("paired numerical qualification failed")
        if not before.sentinel_ok or not after.sentinel_ok:
            module_problems.append("paired sentinel qualification failed")
        if before.executed_calls < 1 or after.executed_calls < 1:
            module_problems.append("module path was not actually executed")
    stages.append(
        StageReceipt(
            "module",
            GateState.FAILED if module_problems else GateState.COMPLETED,
            "; ".join(module_problems)
            if module_problems
            else "paired module numerical, sentinel and execution path verified",
        )
    )
    topology = spec.topology
    topology_problems = []
    ranks = list(topology.observed_ranks)
    if topology.tp < 1 or topology.die_count < 1:
        topology_problems.append("TP and die count must be positive")
    elif topology.die_count % topology.tp:
        topology_problems.append("die count is not divisible by TP")
    if len(set(ranks)) != len(ranks):
        topology_problems.append("observed ranks contain duplicates")
    if len(ranks) != topology.die_count:
        topology_problems.append(
            f"observed {len(ranks)} ranks but topology declares {topology.die_count} dies"
        )
    stages.append(
        StageReceipt(
            "topology",
            GateState.FAILED if topology_problems else GateState.COMPLETED,
            "; ".join(topology_problems)
            if topology_problems
            else "TP/die topology and all observed ranks verified",
        )
    )
    guard_rows = []
    guard_problems = []
    guard_unknown = False
    for guard in spec.guard_workloads:
        if not guard.required:
            continue
        row = {
            "name": guard.name,
            "measurement_valid": guard.measurement_valid,
            "passed": guard.passed,
        }
        if not guard.measurement_valid:
            guard_unknown = True
            guard_problems.append(f"{guard.name}: guard measurement invalid")
        elif not guard.passed:
            guard_problems.append(f"{guard.name}: required guard failed")
        elif (
            guard.baseline_ns is not None
            and guard.candidate_ns is not None
            and guard.baseline_ns > 0
            and guard.candidate_ns / guard.baseline_ns > guard.max_regression_ratio
        ):
            row["regression_ratio"] = guard.candidate_ns / guard.baseline_ns
            guard_problems.append(f"{guard.name}: guard regression above tolerance")
        guard_rows.append(row)
    if not any(guard.required for guard in spec.guard_workloads):
        guard_problems.append("no required guard workload declared")
    guard_state = (
        GateState.UNKNOWN
        if guard_unknown
        else GateState.FAILED
        if guard_problems
        else GateState.COMPLETED
    )
    stages.append(
        StageReceipt(
            "guards",
            guard_state,
            "; ".join(guard_problems) if guard_problems else "all required guard workloads held",
            {"guards": guard_rows},
        )
    )
    if any(stage.state is GateState.FAILED for stage in stages):
        state = GateState.FAILED
    elif any(stage.state is GateState.UNKNOWN for stage in stages):
        state = GateState.UNKNOWN
    else:
        state = GateState.COMPLETED
    return GateResult(
        "G3",
        state,
        stages,
        {"guards": guard_rows},
        [f"{stage.name}: {stage.detail}" for stage in stages
         if stage.state is not GateState.COMPLETED],
        timeout_plan,
    )
