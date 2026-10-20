"""Native inference lifecycle contracts, independent of any external benchmark.

[Role] Declare lifecycle goals, scored/guard workloads and judge observations.
[Invariants] Binary task acceptance is separate from measured performance gain.
[Timeouts & Locks] No I/O; the existing task verifier budget owns measurement.
[Side-Effect Policy] Validate data only; never load candidate code or start servers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = r"^[0-9a-f]{64}$"


class InferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class InferenceWorkload(InferenceModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    metric: str = Field(min_length=1)
    direction: Literal["higher", "lower"]
    role: Literal["scored", "guard"] = "scored"
    workload_sha256: str = Field(pattern=SHA256)
    min_ratio: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def guard_has_floor(self):
        if (self.role == "guard") != (self.min_ratio is not None):
            raise ValueError("only guard workloads must declare a minimum ratio")
        return self


class InferenceContract(InferenceModel):
    category: Literal["model_support", "optimization", "regression"]
    engine: str = Field(min_length=1)
    model: str = Field(min_length=1)
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    baseline_source_sha256: str = Field(pattern=SHA256)
    runtime_sha256: str = Field(pattern=SHA256)
    device_class: str = Field(min_length=1)
    required_checks: list[str] = Field(min_length=1)
    workloads: list[InferenceWorkload] = Field(default_factory=list)
    warmup_rounds: int = Field(default=1, ge=1)
    measurement_rounds: int = Field(default=4, ge=2)
    min_speedup_ratio: float = Field(default=1.03, gt=1)

    @model_validator(mode="after")
    def lifecycle_requirements(self):
        if len(set(self.required_checks)) != len(self.required_checks) or not all(self.required_checks):
            raise ValueError("correctness check names must be nonempty and unique")
        names = [workload.name for workload in self.workloads]
        if len(set(names)) != len(names):
            raise ValueError("workload names must be unique")
        if self.category == "model_support":
            if self.workloads:
                raise ValueError("model support is scored on required checks, not speedups")
        elif not any(workload.role == "scored" for workload in self.workloads):
            raise ValueError("optimization/regression requires a scored workload")
        if self.measurement_rounds % 2:
            raise ValueError("paired measurement needs an even number of counterbalanced rounds")
        return self


class PairedMeasurement(InferenceModel):
    round: int = Field(ge=0)
    order: Literal["baseline_candidate", "candidate_baseline"]
    baseline: float = Field(gt=0, strict=True)
    candidate: float | None = Field(default=None, gt=0, strict=True)
    candidate_error: str | None = None

    @model_validator(mode="after")
    def candidate_outcome(self):
        if (self.candidate is None) != bool(self.candidate_error):
            raise ValueError("candidate measurement requires either a value or an explicit failure")
        return self


class WorkloadMeasurements(InferenceModel):
    workload_sha256: str = Field(pattern=SHA256)
    warmup_rounds: int = Field(ge=1)
    pairs: list[PairedMeasurement] = Field(min_length=2)


class InferenceEvidence(InferenceModel):
    schema_version: Literal["mindie-bench-inference/1"] = "mindie-bench-inference/1"
    run_nonce: str = Field(min_length=16)
    oracle_manifest_sha256: str = Field(pattern=SHA256)
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    baseline_source_sha256: str = Field(pattern=SHA256)
    runtime_sha256: str = Field(pattern=SHA256)
    device_class: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=SHA256)
    baseline_healthy: bool = Field(strict=True)
    checks: dict[str, bool] = Field(strict=True)
    workloads: dict[str, WorkloadMeasurements] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def explicit_check_outcomes(cls, value):
        if isinstance(value, dict) and any(type(v) is not bool for v in value.get("checks", {}).values()):
            raise ValueError("correctness outcomes must be measured booleans")
        return value
