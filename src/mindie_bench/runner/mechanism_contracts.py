"""Immutable inputs and observations for controlled mechanism ablations.

[Role] Bind one-factor arms to per-task controls and retain comparison denominators.
[Invariants] Raw evidence, model, task and total budget are fixed inputs; delivered
knowledge and completion tokens are outcomes. Missing values remain missing.
[Timeouts & Locks] Pure record construction and validation, no waits or locks.
[Side-Effect Policy] No I/O, requests, workers or external execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from mindie_bench.io import digest
from mindie_bench.runner.compaction_contracts import SemanticEvent
from mindie_bench.runner.protocol_json import JsonRecord, record_payload

MECHANISM_ABLATION_PROTOCOL = "mindie-bench-mechanism-ablation/1"


class Mechanism(StrEnum):
    SINGLE_AGENT_VS_ROLES = "single_agent_vs_roles"
    LEDGER = "ledger"
    RETRIEVAL = "retrieval"
    PARALLEL_EXPLORATION = "parallel_exploration"
    STRATEGY_SELECTION = "strategy_selection"


MECHANISMS = (
    Mechanism.SINGLE_AGENT_VS_ROLES,
    Mechanism.LEDGER,
    Mechanism.RETRIEVAL,
    Mechanism.PARALLEL_EXPLORATION,
)

ALL_MECHANISMS = (
    *MECHANISMS,
    Mechanism.STRATEGY_SELECTION,
)


MECHANISM_FIELDS = {
    Mechanism.SINGLE_AGENT_VS_ROLES: "roles",
    Mechanism.LEDGER: "ledger",
    Mechanism.RETRIEVAL: "retrieval",
    Mechanism.PARALLEL_EXPLORATION: "exploration",
    Mechanism.STRATEGY_SELECTION: "strategy_selection",
}


CONFIG_FIELDS = ("roles", "ledger", "retrieval", "exploration", "strategy_selection")


class ArmRole(StrEnum):
    CONTROL = "control"
    TREATMENT = "treatment"


class AblationOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INFRA = "infra"
    UNKNOWN = "unknown"


_OUTCOME_ALIASES = {
    "completed_pass": AblationOutcome.PASS,
    "completed_fail": AblationOutcome.FAIL,
    "infra_failed": AblationOutcome.INFRA,
    "unknown": AblationOutcome.UNKNOWN,
}


def _normalize_outcome(value: AblationOutcome | str) -> AblationOutcome:
    if isinstance(value, AblationOutcome):
        return value
    text = str(getattr(value, "value", value))
    if text in _OUTCOME_ALIASES:
        return _OUTCOME_ALIASES[text]
    try:
        return AblationOutcome(text)
    except ValueError as error:
        raise ValueError(f"unknown ablation outcome: {value!r}") from error


@dataclass(frozen=True)
class FixedControls(JsonRecord):
    """Fixed inputs bound to one task; output volume is not among them."""

    model_id: str
    inference_profile: str
    tool_digest: str
    raw_knowledge_digest: str
    evidence_digest: str
    task_start_digest: str
    semantic_event: SemanticEvent
    model_call_budget: int
    npu_run_budget: int
    max_wall_clock_sec: float
    knowledge_token_budget: int = 0
    child_model_call_budget: int = 0
    organizer_model_call_budget: int = 0
    review_model_call_budget: int = 0

    def __post_init__(self):
        for name in (
            "model_id",
            "inference_profile",
            "tool_digest",
            "raw_knowledge_digest",
            "evidence_digest",
            "task_start_digest",
        ):
            if not getattr(self, name):
                raise ValueError(f"fixed control {name} must be non-empty")
        object.__setattr__(self, "semantic_event", SemanticEvent(self.semantic_event))
        if type(self.model_call_budget) is not int or self.model_call_budget <= 0:
            raise ValueError("model_call_budget must be a positive integer")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.child_model_call_budget,
                self.organizer_model_call_budget,
                self.review_model_call_budget,
            )
        ):
            raise ValueError("child/organizer/review budgets must be non-negative integers")
        reserved = (
            self.child_model_call_budget
            + self.organizer_model_call_budget
            + self.review_model_call_budget
        )
        if reserved > self.model_call_budget:
            raise ValueError("child/organizer/review model calls must fit the total model-call budget")
        if type(self.npu_run_budget) is not int or self.npu_run_budget < 0:
            raise ValueError("npu_run_budget must be a non-negative integer")
        if not math.isfinite(self.max_wall_clock_sec) or self.max_wall_clock_sec < 0:
            raise ValueError("max_wall_clock_sec must be finite and non-negative")
        if type(self.knowledge_token_budget) is not int or self.knowledge_token_budget < 0:
            raise ValueError("knowledge_token_budget must be a non-negative integer")

    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class ArmConfig(JsonRecord):
    roles: str
    ledger: str
    retrieval: str
    exploration: str
    strategy_selection: str = "default"

    def __post_init__(self):
        for name in CONFIG_FIELDS:
            if not getattr(self, name):
                raise ValueError(f"arm configuration {name} must be non-empty")

    def changed_fields(self, other: ArmConfig) -> tuple[str, ...]:
        return tuple(name for name in CONFIG_FIELDS if getattr(self, name) != getattr(other, name))


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    mechanism: Mechanism
    control: ArmConfig
    treatment: ArmConfig
    control_id: str = "control"
    treatment_id: str = "treatment"

    def __post_init__(self):
        if not self.pair_id:
            raise ValueError("pair_id must be non-empty")
        object.__setattr__(self, "mechanism", Mechanism(self.mechanism))
        if not self.control_id or not self.treatment_id:
            raise ValueError("arm IDs must be non-empty")
        if self.control_id == self.treatment_id:
            raise ValueError("control and treatment arm IDs must differ")


@dataclass(frozen=True)
class MechanismPair(JsonRecord):
    pair_id: str
    mechanism: Mechanism
    control: ArmConfig
    treatment: ArmConfig
    control_id: str
    treatment_id: str

    def __post_init__(self):
        object.__setattr__(self, "mechanism", Mechanism(self.mechanism))
        expected = MECHANISM_FIELDS[self.mechanism]
        changed = self.control.changed_fields(self.treatment)
        if changed != (expected,):
            raise ValueError(f"pair {self.pair_id!r} must change only {expected!r}; changed {changed}")

    @property
    def changed_field(self) -> str:
        return MECHANISM_FIELDS[self.mechanism]

    @property
    def held_constant(self) -> tuple[str, ...]:
        return tuple(name for name in CONFIG_FIELDS if name != self.changed_field)

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload.update(
            changed_field=self.changed_field,
            held_constant=list(self.held_constant),
        )
        return payload


@dataclass(frozen=True)
class TaskControls(JsonRecord):
    task_id: str
    controls: FixedControls

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task controls require a task id")

    def digest(self) -> str:
        return self.controls.digest()


@dataclass(frozen=True)
class AblationTask(JsonRecord):
    task_id: str
    task_family: str
    hardware_tier: str

    def __post_init__(self):
        for name in ("task_id", "task_family", "hardware_tier"):
            if not getattr(self, name):
                raise ValueError(f"ablation task {name} must be non-empty")


@dataclass(frozen=True)
class AblationCell(JsonRecord):
    pair_id: str
    mechanism: Mechanism
    arm_role: ArmRole
    arm_id: str
    task_id: str
    task_family: str
    hardware_tier: str
    repetition: int
    order: int
    controls_digest: str

    def key(self) -> tuple[str, ArmRole, str, int]:
        return (self.pair_id, self.arm_role, self.task_id, self.repetition)


@dataclass(frozen=True)
class MechanismAblationPlan(JsonRecord):
    protocol: str
    pairs: tuple[MechanismPair, ...]
    tasks: tuple[AblationTask, ...]
    repetitions: int
    controls_by_task: tuple[TaskControls, ...]
    cells: tuple[AblationCell, ...]

    def controls_for(self, task_id: str) -> FixedControls:
        for item in self.controls_by_task:
            if item.task_id == task_id:
                return item.controls
        raise ValueError(f"no fixed controls for task: {task_id}")

    def controls_digests(self) -> dict[str, str]:
        return {item.task_id: item.digest() for item in self.controls_by_task}

    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload["controls_digest_by_task"] = self.controls_digests()
        return payload


@dataclass(frozen=True)
class AblationObservation(JsonRecord):
    pair_id: str
    arm_role: ArmRole
    task_id: str
    repetition: int
    status: AblationOutcome | str
    completion_tokens: int | None = None
    wall_clock_sec: float | None = None
    npu_runs: int | None = None
    knowledge_tokens: int | None = None
    first_effective_action: str | None = None
    controls_digest: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.pair_id or not self.task_id:
            raise ValueError("ablation observation requires pair and task IDs")
        object.__setattr__(self, "arm_role", ArmRole(self.arm_role))
        object.__setattr__(self, "status", _normalize_outcome(self.status))
        if type(self.repetition) is not int or self.repetition < 1:
            raise ValueError("ablation observation requires a positive repetition")
        for name in ("completion_tokens", "npu_runs", "knowledge_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.wall_clock_sec is not None and (
            not math.isfinite(self.wall_clock_sec) or self.wall_clock_sec < 0
        ):
            raise ValueError("wall_clock_sec must be finite and non-negative or None")

    def key(self) -> tuple[str, ArmRole, str, int]:
        return (self.pair_id, self.arm_role, self.task_id, self.repetition)


@dataclass(frozen=True)
class AblationDenominator(JsonRecord):
    planned: int = 0
    observed: int = 0
    missing: int = 0
    passed: int = 0
    failed: int = 0
    infra: int = 0
    unknown: int = 0

    @property
    def valid(self) -> int:
        return self.passed + self.failed

    @property
    def completion_rate(self) -> float | None:
        return self.passed / self.valid if self.valid else None

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload.pop("passed", None)
        payload.pop("failed", None)
        payload.update(
            {
                "pass": self.passed,
                "fail": self.failed,
                "valid": self.valid,
                "completion_rate": self.completion_rate,
            }
        )
        return payload


@dataclass(frozen=True)
class PairAssessment(JsonRecord):
    pair_id: str
    mechanism: Mechanism
    changed_field: str
    held_constant: tuple[str, ...]
    controls_digest_by_task: dict[str, str]
    controls_binding_complete: bool
    control: AblationDenominator
    treatment: AblationDenominator
    combined: AblationDenominator
    planned_pairs: int
    paired_tasks: int
    independent_paired_families: int
    paired_valid_outcomes: int
    paired_pass_delta: int | None
    control_tokens: int | None
    treatment_tokens: int | None
    token_delta: int | None
    token_sample_count: int
    token_complete: bool
    control_wall_clock_sec: float | None
    treatment_wall_clock_sec: float | None
    wall_clock_delta_sec: float | None
    wall_clock_sample_count: int
    wall_clock_complete: bool
    knowledge_tokens_control: int | None
    knowledge_tokens_treatment: int | None
    knowledge_tokens_delta: int | None
    knowledge_token_sample_count: int
    knowledge_token_complete: bool
    observations: tuple[AblationObservation, ...]
    small_sample: bool
    attribution: Literal["descriptive_only", "insufficient"]
    limitations: tuple[str, ...]
    interactions: tuple[str, ...]


@dataclass(frozen=True)
class MechanismAblationReport(JsonRecord):
    protocol: str
    plan_digest: str
    controls_digest_by_task: dict[str, str]
    pairs: tuple[PairAssessment, ...]
    overall: AblationDenominator
    observations: tuple[AblationObservation, ...]
    task_families: tuple[str, ...]
    independent_task_families: int
    by_task_family: dict[str, AblationDenominator]
    by_hardware_tier: dict[str, AblationDenominator]
    interactions: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload["interpretation"] = (
            "one-factor matched pairs describe conditional differences; they do not "
            "identify interactions or attribute an end-to-end total to one component; "
            "delivered knowledge/output volume is an outcome, not a fixed input"
        )
        return payload
