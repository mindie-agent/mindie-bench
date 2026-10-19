"""Release benchmark interchange contracts.

[Role] Describe repository-owned workloads and their public numeric measurements.
[Invariants] Functional checks, microbenchmarks and end-to-end trials stay distinct.
Missing or failed required evidence never becomes a successful release gate.
[Timeouts & Locks] Each case declares its execution budget; no locks or I/O here.
[Side-Effect Policy] Public payloads exclude transcripts, environment variables and logs.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import Field, model_validator

from mindie_bench.models import StrictModel

Identifier = str
Status = Literal["pass", "fail", "unknown", "unsupported"]


class Metric(StrictModel):
    name: Identifier = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    unit: str = Field(pattern=r"^[a-zA-Z0-9_./%-]+$")
    value: float = Field(allow_inf_nan=False)
    iterations: int = Field(gt=0, strict=True)


class Measurement(StrictModel):
    status: Status
    reason_code: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")
    metrics: list[Metric] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_metrics(self):
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("a measurement cannot repeat a metric")
        if self.status != "pass" and not self.reason_code:
            raise ValueError("non-pass measurements require a public reason code")
        return self


class ReleaseCase(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    kind: Literal["functional", "microbenchmark", "e2e"]
    required: bool = True
    repetitions: int = Field(default=5, ge=1, strict=True)
    timeout_seconds: float = Field(gt=0, le=3600, allow_inf_nan=False)
    command: list[str] = Field(min_length=1)
    side_effects: Literal["local", "read_only", "external_write"] = "local"


class ReleaseSuite(StrictModel):
    schema_version: Literal["mindie-release-suite/1"]
    suite: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    publishable: Literal[True]
    cases: list[ReleaseCase] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_cases(self):
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("release suite repeats a case ID")
        if not any(case.required for case in self.cases):
            raise ValueError("a release suite needs at least one required case")
        return self


class ReleaseTarget(StrictModel):
    repository: str = Field(pattern=r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    version: str = Field(min_length=1)
    required_cases: list[str] = Field(min_length=1)
    platforms: list[str] = Field(min_length=1)
    runner_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")


class ReleaseManifest(StrictModel):
    schema_version: Literal["mindie-release-manifest/1"]
    product_version: str = Field(min_length=1)
    repositories: list[ReleaseTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_repositories(self):
        names = [target.repository for target in self.repositories]
        if len(set(names)) != len(names):
            raise ValueError("release manifest repeats a repository")
        return self


class Source(StrictModel):
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dirty: bool


class Attempt(StrictModel):
    repetition: int = Field(ge=1)
    status: Status
    returncode: int | None = None
    process_seconds: float | None = Field(default=None, ge=0)
    reason_code: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")
    metrics: list[Metric] = Field(default_factory=list)
    evidence_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class MetricSummary(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    unit: str = Field(pattern=r"^[a-zA-Z0-9_./%-]+$")
    samples: list[float] = Field(min_length=1)
    iterations: list[int] = Field(min_length=1)
    median: float
    mean: float
    min: float
    max: float


class CaseResult(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    kind: Literal["functional", "microbenchmark", "e2e"]
    required: bool
    repetitions_planned: int = Field(ge=1)
    repetitions_completed: int = Field(ge=1)
    status: Status
    attempts: list[Attempt] = Field(min_length=1)
    metrics: list[MetricSummary]

    @model_validator(mode="after")
    def complete_pass(self):
        if self.status == "pass" and (
            self.repetitions_completed != self.repetitions_planned
            or len(self.attempts) != self.repetitions_completed
            or any(attempt.status != "pass" for attempt in self.attempts)
            or any(attempt.returncode != 0 for attempt in self.attempts)
            or any(attempt.evidence_sha256 is None for attempt in self.attempts)
            or (self.kind != "functional" and not self.metrics)
            or any(
                {(metric.name, metric.unit) for metric in attempt.metrics}
                != {(metric.name, metric.unit) for metric in self.metrics}
                for attempt in self.attempts
            )
        ):
            raise ValueError("passing case lacks complete successful measurement evidence")
        for metric in self.metrics:
            observed = [
                item for attempt in self.attempts if attempt.status == "pass"
                for item in attempt.metrics if item.name == metric.name
            ]
            values = [item.value for item in observed]
            if (
                not values or metric.samples != values
                or metric.iterations != [item.iterations for item in observed]
                or any(item.unit != metric.unit for item in observed)
                or metric.median != statistics.median(values)
                or metric.mean != statistics.mean(values)
                or metric.min != min(values) or metric.max != max(values)
            ):
                raise ValueError("metric summary disagrees with raw measurement evidence")
        return self


class Environment(StrictModel):
    platform: str
    machine: str
    os_release: str
    runner_python: str
    workload_python: str
    cpu_count: int | None


class Tool(StrictModel):
    name: Literal["mindie-bench"]
    version: str
    source: Source | None = None
    source_after: Source | None = None


class BenchmarkReport(StrictModel):
    schema_version: Literal["mindie-release-benchmark/1", "mindie-release-benchmark/2"]
    repository: str = Field(pattern=r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
    version: str = Field(pattern=r"^[a-zA-Z0-9_.+-]+$")
    source: Source
    source_after: Source
    suite: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    suite_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool: Tool
    environment: Environment
    started_at: str
    completed_at: str
    cases: list[CaseResult] = Field(min_length=1)
    release_gate: bool
    blocking_reasons: list[str]

    @model_validator(mode="after")
    def gate_matches_evidence(self):
        if self.release_gate and self.schema_version == "mindie-release-benchmark/2":
            before, after = self.tool.source, self.tool.source_after
            if (before is None or after is None or before.dirty or after.dirty
                    or before.revision != after.revision):
                raise ValueError("release gate contradicts runner source evidence")
        if self.release_gate and (
            self.source.dirty or self.source_after.dirty
            or self.source.revision != self.source_after.revision
            or self.blocking_reasons
            or not any(case.required for case in self.cases)
            or any(case.required and case.status != "pass" for case in self.cases)
        ):
            raise ValueError("release gate contradicts source or required case evidence")
        return self
