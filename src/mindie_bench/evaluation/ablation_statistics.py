"""Case-matched summaries of independently measured ablation stages.

[Role] Keep workload mixtures, raw timing identity and observed block variation distinct.
[Invariants] A block envelope is an observed range, never a confidence interval.
Independent stages are not paired. Every sample belongs to exactly one case/block.
[Timeouts & Locks] Pure in-memory statistics; no waits, subprocesses or locks.
[Side-Effect Policy] Returns derived evidence without changing raw measurements.
"""

from __future__ import annotations

import math
import statistics
from typing import Literal

from pydantic import Field

from mindie_bench.evaluation.performance import TimingRun
from mindie_bench.io import digest
from mindie_bench.models import StrictModel

SHA = r"^[0-9a-f]{64}$"


class SampleGroup(StrictModel):
    case_sha256: str = Field(pattern=SHA)
    block_id: str = Field(min_length=1)
    source_sample_indices: list[int] = Field(min_length=2)
    source_ref: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA)
    grouping_derivation_ref: str = Field(min_length=1)
    grouping_derivation_sha256: str = Field(pattern=SHA)


class CaseComparison(StrictModel):
    case_sha256: str
    reference_samples: int
    candidate_samples: int
    reference_mean_ms: float
    candidate_mean_ms: float
    speedup_ratio: float
    reference_block_means_ms: list[float]
    candidate_block_means_ms: list[float]
    observed_block_ratio_lower: float
    observed_block_ratio_upper: float
    reference_within_block_max_rel_iqr: float
    candidate_within_block_max_rel_iqr: float


class TimingComparison(StrictModel):
    estimator: Literal["geomean_of_per_case_arithmetic_mean_ratios"] = (
        "geomean_of_per_case_arithmetic_mean_ratios"
    )
    reference_run_id: str
    candidate_run_id: str
    reference_timing_sha256: str
    candidate_timing_sha256: str
    clock: Literal["host_monotonic_ns"]
    speedup_geomean: float
    cases: list[CaseComparison]
    uncertainty_scope: str = (
        "observed independent block-mean envelope; not a confidence interval; "
        "no cross-stage block pairing or between-run replication claim"
    )


def timing_identity(timing: TimingRun) -> str:
    return digest(timing.model_dump(mode="json"))


def milliseconds(timing: TimingRun) -> list[float]:
    return [(sample.ended_ns - sample.started_ns) / 1e6 for sample in timing.samples]


def validate_groups(timing: TimingRun, groups: list[SampleGroup], samples_ms: list[float]):
    if milliseconds(timing) != samples_ms:
        raise ValueError("ablation samples differ from the bound raw timing intervals")
    indices = [index for group in groups for index in group.source_sample_indices]
    if sorted(indices) != list(range(len(timing.samples))):
        raise ValueError("case/block groups must cover every raw sample exactly once")
    keys = [(group.case_sha256, group.block_id) for group in groups]
    if len(set(keys)) != len(keys):
        raise ValueError("case/block identities must be unique within a stage")
    for group in groups:
        if any(
            timing.samples[index].shape_sha256 != group.case_sha256
            for index in group.source_sample_indices
        ):
            raise ValueError("case/block group does not match the raw sample case")


def _case_blocks(run, case):
    return [
        [run.samples_ms[index] for index in group.source_sample_indices]
        for group in run.sample_groups
        if group.case_sha256 == case
    ]


def _relative_iqr(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (q3 - q1) / statistics.median(values)


def compare_timings(reference, candidate) -> TimingComparison | None:
    """Use independent observed blocks; return no conclusion without block identity."""
    if not reference.timing or not candidate.timing:
        return None
    if not reference.sample_groups or not candidate.sample_groups:
        return None
    reference_cases = {group.case_sha256 for group in reference.sample_groups}
    candidate_cases = {group.case_sha256 for group in candidate.sample_groups}
    if reference_cases != candidate_cases or reference.timing.clock != candidate.timing.clock:
        raise ValueError("conditional timing comparisons require the same cases and clock")
    cases = []
    for case in sorted(reference_cases):
        left, right = _case_blocks(reference, case), _case_blocks(candidate, case)
        if len(left) < 2 or len(right) < 2:
            return None
        lm, rm = [statistics.mean(x) for x in left], [statistics.mean(x) for x in right]
        lv, rv = [v for block in left for v in block], [v for block in right for v in block]
        if (
            len(lv) != len(rv)
            or reference.timing.warmup_rounds != candidate.timing.warmup_rounds
            or len(reference.timing.buffer_sha256[case]) != len(candidate.timing.buffer_sha256[case])
        ):
            raise ValueError("case comparisons require matching sample, warmup and buffer counts")
        cases.append(
            CaseComparison(
                case_sha256=case,
                reference_samples=len(lv),
                candidate_samples=len(rv),
                reference_mean_ms=statistics.mean(lv),
                candidate_mean_ms=statistics.mean(rv),
                speedup_ratio=statistics.mean(lv) / statistics.mean(rv),
                reference_block_means_ms=lm,
                candidate_block_means_ms=rm,
                observed_block_ratio_lower=min(lm) / max(rm),
                observed_block_ratio_upper=max(lm) / min(rm),
                reference_within_block_max_rel_iqr=max(map(_relative_iqr, left)),
                candidate_within_block_max_rel_iqr=max(map(_relative_iqr, right)),
            )
        )
    return TimingComparison(
        reference_run_id=reference.run_id,
        candidate_run_id=candidate.run_id,
        reference_timing_sha256=timing_identity(reference.timing),
        candidate_timing_sha256=timing_identity(candidate.timing),
        clock=reference.timing.clock,
        speedup_geomean=math.exp(statistics.mean(math.log(row.speedup_ratio) for row in cases)),
        cases=cases,
    )


def observed_lower_bounds_pass(comparison: TimingComparison | None, threshold: float) -> bool:
    return comparison is not None and all(
        row.observed_block_ratio_lower >= threshold for row in comparison.cases
    )
