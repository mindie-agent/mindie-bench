"""Immutable product-matrix profiles, observations and report contracts.

[Role] Define the separate product-level comparison protocol and result schemas.
[Invariants] Legacy component groups retain their original meaning; product
profiles fix conditions and the matched downstream budget plus separately
accounted prerequisite flywheel cost, preserve missing data, and keep faithful
external baselines separate from matched comparisons. Equal downstream budget is
conditional reuse only; a same-lifecycle total needs a separately predeclared
allocation.
[Timeouts & Locks] Pure record construction and validation, no waits or locks.
[Side-Effect Policy] No I/O, model requests or hardware execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from mindie_bench.io import digest
from mindie_bench.models import Group, TerminalStatus
from mindie_bench.runner.p3_lineage import P3LineageBinding
from mindie_bench.runner.protocol_json import JsonRecord, record_payload

PRODUCT_MATRIX_PROTOCOL = "mindie-bench-product-matrix/1"


class ProductGroup(StrEnum):
    P0_BARE_HOST = "P0_bare_host"
    P1_V002_BASELINE = "P1_v002_baseline"
    P2_V003_FULL = "P2_v003_full"
    P3_V003_FLYWHEEL = "P3_v003_flywheel"
    EXT_MODEL_AGENT = "P_ext_model_agent"
    EXT_CANNBOT = "P_ext_cannbot"
    EXT_RAW_SKILLS = "P_ext_raw_skills"


MATCHED_PRODUCT_GROUPS = (
    ProductGroup.P0_BARE_HOST,
    ProductGroup.P1_V002_BASELINE,
    ProductGroup.P2_V003_FULL,
    ProductGroup.P3_V003_FLYWHEEL,
)


EXTERNAL_PRODUCT_GROUPS = (
    ProductGroup.EXT_MODEL_AGENT,
    ProductGroup.EXT_CANNBOT,
    ProductGroup.EXT_RAW_SKILLS,
)


ALL_PRODUCT_GROUPS = MATCHED_PRODUCT_GROUPS + EXTERNAL_PRODUCT_GROUPS


PRODUCT_GROUP_DESCRIPTIONS = {
    ProductGroup.P0_BARE_HOST: "bare host, no MindIE Agent components",
    ProductGroup.P1_V002_BASELINE: "MindIE Agent v0.0.2",
    ProductGroup.P2_V003_FULL: "MindIE Agent v0.0.3 complete",
    ProductGroup.P3_V003_FLYWHEEL: "seed -> capture -> evaluation with a frozen lineage",
    ProductGroup.EXT_MODEL_AGENT: "external Model Agent, faithful native configuration",
    ProductGroup.EXT_CANNBOT: "external CANNBot, faithful native configuration",
    ProductGroup.EXT_RAW_SKILLS: "raw external skill collection, faithful native configuration",
}


class CellOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INFRA = "infra"
    UNKNOWN = "unknown"


_TERMINAL_OUTCOMES = {
    TerminalStatus.PASS: CellOutcome.PASS,
    TerminalStatus.FAIL: CellOutcome.FAIL,
    TerminalStatus.INFRA: CellOutcome.INFRA,
    TerminalStatus.UNKNOWN: CellOutcome.UNKNOWN,
}


_OUTCOME_ALIASES = {
    "completed_pass": CellOutcome.PASS,
    "completed_fail": CellOutcome.FAIL,
    "infra_failed": CellOutcome.INFRA,
    "unknown": CellOutcome.UNKNOWN,
}


def _as_product_group(value: ProductGroup | str) -> ProductGroup:
    if isinstance(value, Group):
        raise TypeError(
            "models.Group is component-increment G0..G3 semantics; select a ProductGroup "
            "with an independent experiment profile instead"
        )
    try:
        return value if isinstance(value, ProductGroup) else ProductGroup(value)
    except ValueError as error:
        raise ValueError(f"unknown product group: {value!r}") from error


def normalize_outcome(value: CellOutcome | TerminalStatus | str) -> CellOutcome:
    if isinstance(value, CellOutcome):
        return value
    if isinstance(value, TerminalStatus):
        return _TERMINAL_OUTCOMES[value]
    text = str(getattr(value, "value", value))
    if text in _OUTCOME_ALIASES:
        return _OUTCOME_ALIASES[text]
    try:
        return CellOutcome(text)
    except ValueError as error:
        raise ValueError(f"unknown cell outcome: {value!r}") from error


@dataclass(frozen=True)
class MatrixTask(JsonRecord):
    task_id: str
    task_family: str
    hardware_tier: str

    def __post_init__(self):
        for name in ("task_id", "task_family", "hardware_tier"):
            if not getattr(self, name):
                raise ValueError(f"matrix task {name} must be non-empty")


_PREREQUISITE_COST_ROLES = ("seed", "capture", "organizer", "review")


@dataclass(frozen=True)
class BudgetEnvelope(JsonRecord):
    """Matched downstream budget plus visible prerequisite flywheel cost.

    Seed and capture usage is additional P3 prerequisite cost, not a free part of
    the downstream envelope. It is therefore excluded from ``comparison_dict``
    (P2/P3 remain matched on the same downstream total) but retained in every
    report and cross-checked against the P3 cost ledger before execution.
    """

    total_model_calls: int
    foreground_model_calls: int = 0
    child_model_calls: int = 0
    organizer_model_calls: int = 0
    review_model_calls: int = 0
    npu_runs: int = 0
    human_interventions: int = 0
    max_wall_clock_sec: float = 0.0
    max_cost_usd: float | None = None
    lifecycle_model_calls: int | None = None
    lifecycle_cost_usd: float | None = None
    seed_model_calls: int = 0
    capture_model_calls: int = 0
    capture_organizer_model_calls: int = 0
    independent_review_model_calls: int = 0
    seed_npu_runs: int = 0
    seed_cost_usd: float = 0.0
    capture_cost_usd: float = 0.0
    capture_organizer_cost_usd: float = 0.0
    independent_review_cost_usd: float = 0.0
    organizer_cost_usd: float = 0.0
    review_cost_usd: float = 0.0
    unknown_cost_roles: tuple[str, ...] = ()

    def __post_init__(self):
        values = (
            self.total_model_calls,
            self.foreground_model_calls,
            self.child_model_calls,
            self.organizer_model_calls,
            self.review_model_calls,
            self.npu_runs,
            self.human_interventions,
            self.seed_model_calls,
            self.capture_model_calls,
            self.capture_organizer_model_calls,
            self.independent_review_model_calls,
            self.seed_npu_runs,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("budget fields must be non-negative integers")
        if self.total_model_calls <= 0:
            raise ValueError("total_model_calls must be positive")
        reserved = sum(values[1:5])
        if reserved > self.total_model_calls:
            raise ValueError(
                "foreground/child/organizer/review model calls must fit the total model budget"
            )
        if self.max_cost_usd is not None and (
            not math.isfinite(self.max_cost_usd) or self.max_cost_usd < 0
        ):
            raise ValueError("max_cost_usd must be finite and non-negative or None")
        if (self.lifecycle_model_calls is None) != (self.lifecycle_cost_usd is None):
            raise ValueError("lifecycle model-call and cost allocations must be declared together")
        if self.lifecycle_model_calls is not None:
            if type(self.lifecycle_model_calls) is not int or self.lifecycle_model_calls < 0:
                raise ValueError("lifecycle_model_calls must be a non-negative integer or None")
            minimum = (
                self.total_model_calls
                + self.seed_model_calls
                + self.capture_model_calls
                + self.capture_organizer_model_calls
                + self.independent_review_model_calls
            )
            if self.lifecycle_model_calls < minimum:
                raise ValueError(
                    "lifecycle_model_calls must include downstream plus Seed/capture/in-capture organizer/independent review"
                )
        if self.lifecycle_cost_usd is not None and (
            not math.isfinite(self.lifecycle_cost_usd) or self.lifecycle_cost_usd < 0
        ):
            raise ValueError("lifecycle_cost_usd must be finite and non-negative or None")
        if self.lifecycle_cost_usd is not None:
            downstream_cost = (
                self.max_cost_usd
                if self.max_cost_usd is not None
                else self.organizer_cost_usd + self.review_cost_usd
            )
            minimum = (
                downstream_cost
                + self.seed_cost_usd
                + self.capture_cost_usd
                + self.capture_organizer_cost_usd
                + self.independent_review_cost_usd
            )
            if self.lifecycle_cost_usd < minimum:
                raise ValueError("lifecycle_cost_usd must include Seed/Organizer overhead")
        for name in (
            "max_wall_clock_sec",
            "seed_cost_usd",
            "capture_cost_usd",
            "capture_organizer_cost_usd",
            "independent_review_cost_usd",
            "organizer_cost_usd",
            "review_cost_usd",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        roles = tuple(self.unknown_cost_roles)
        if any(role not in _PREREQUISITE_COST_ROLES for role in roles):
            raise ValueError(f"unknown_cost_roles must be drawn from {_PREREQUISITE_COST_ROLES}")
        if len(set(roles)) != len(roles) or tuple(sorted(roles)) != roles:
            raise ValueError("unknown_cost_roles must be unique and sorted")
        object.__setattr__(self, "unknown_cost_roles", roles)

    def comparison_dict(self) -> dict:
        """Common downstream envelope, including the enforced cost cap; role shares are choices."""
        return {
            "total_model_calls": self.total_model_calls,
            "npu_runs": self.npu_runs,
            "human_interventions": self.human_interventions,
            "max_wall_clock_sec": self.max_wall_clock_sec,
            "max_cost_usd": self.max_cost_usd,
        }

    def task_derived_dict(self) -> dict:
        """Fields derivable from the task; role shares and caps are plan choices."""
        return {
            "total_model_calls": self.total_model_calls,
            "npu_runs": self.npu_runs,
            "human_interventions": self.human_interventions,
            "max_wall_clock_sec": self.max_wall_clock_sec,
        }

    def ledger_dict(self) -> dict:
        """Downstream plus Seed/capture/organizer/review identity for the P3 ledger."""
        return {
            "total_model_calls": self.total_model_calls,
            "npu_runs": self.npu_runs,
            "seed_model_calls": self.seed_model_calls,
            "capture_model_calls": self.capture_model_calls,
            "capture_organizer_model_calls": self.capture_organizer_model_calls,
            "independent_review_model_calls": self.independent_review_model_calls,
            "seed_npu_runs": self.seed_npu_runs,
            "seed_cost_usd": self.seed_cost_usd,
            "capture_cost_usd": self.capture_cost_usd,
            "capture_organizer_cost_usd": self.capture_organizer_cost_usd,
            "independent_review_cost_usd": self.independent_review_cost_usd,
            "unknown_cost_roles": list(self.unknown_cost_roles),
        }


@dataclass(frozen=True)
class MatchingConditions(JsonRecord):
    model_id: str
    inference_profile: str
    task_start_digest: str
    oracle_digest: str
    input_digest: str
    knowledge_cutoff: str
    tool_permission_digest: str
    persistence_permission_digest: str

    def __post_init__(self):
        for name in (
            "model_id",
            "inference_profile",
            "task_start_digest",
            "oracle_digest",
            "input_digest",
            "knowledge_cutoff",
            "tool_permission_digest",
            "persistence_permission_digest",
        ):
            if not getattr(self, name):
                raise ValueError(f"matching condition {name} must be non-empty")

    def fingerprint(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class ProductProfile(JsonRecord):
    """One product column backed by its own experiment profile.
    `experiment_profile` is an opaque, independent reference. It is never
    derived from a legacy `models.Group`.
    """

    profile_id: str
    group: ProductGroup
    experiment_profile: str
    protocol: str
    fidelity: Literal["matched", "faithful_external"]
    cohort: str
    conditions: MatchingConditions
    budget: BudgetEnvelope
    preserved_capabilities: tuple[str, ...] = ()
    notes: str = ""
    p3_lineage: P3LineageBinding | None = None

    def __post_init__(self):
        if isinstance(self.group, Group):
            raise TypeError(
                "legacy models.Group uses component-increment G0..G3 semantics and cannot "
                "back a product profile"
            )
        object.__setattr__(self, "group", _as_product_group(self.group))
        for name in ("profile_id", "experiment_profile", "protocol", "cohort"):
            if not getattr(self, name):
                raise ValueError(f"product profile {name} must be non-empty")
        if self.experiment_profile in {group.value for group in Group}:
            raise ValueError(
                "experiment_profile must reference an independent profile, not a legacy Group value"
            )
        if self.fidelity not in {"matched", "faithful_external"}:
            raise ValueError("fidelity must be 'matched' or 'faithful_external'")
        if self.fidelity == "faithful_external" and not self.preserved_capabilities:
            raise ValueError(
                "faithful external profiles must declare the native capabilities they preserve"
            )
        if any(not value for value in self.preserved_capabilities):
            raise ValueError("preserved capabilities must be non-empty names")
        if self.p3_lineage is not None and not isinstance(self.p3_lineage, P3LineageBinding):
            raise TypeError("p3_lineage must be a prepared P3LineageBinding")
        if self.group == ProductGroup.P3_V003_FLYWHEEL and self.p3_lineage is None:
            raise ValueError(
                "P3 product profiles require a closed-family lineage admission; "
                "a product label alone is not a flywheel"
            )
        if self.group != ProductGroup.P3_V003_FLYWHEEL and self.p3_lineage is not None:
            raise ValueError("only the P3 product profile may carry a P3 lineage admission")
        if self.p3_lineage is not None:
            target = self.p3_lineage.manifest.target
            if (target.profile_id, target.cohort, target.protocol) != (
                self.profile_id,
                self.cohort,
                self.protocol,
            ):
                raise ValueError("P3 lineage target differs from its product profile")

    def comparison_fingerprint(self) -> str:
        return digest(
            {
                "conditions": self.conditions.to_dict(),
                "budget": self.budget.comparison_dict(),
            }
        )

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload.update(
            description=PRODUCT_GROUP_DESCRIPTIONS[self.group],
            conditions_fingerprint=self.comparison_fingerprint(),
        )
        return payload


@dataclass(frozen=True)
class ProductCell(JsonRecord):
    group: ProductGroup
    profile_id: str
    cohort: str
    protocol: str
    task_id: str
    task_family: str
    hardware_tier: str
    repetition: int
    order: int

    def key(self) -> tuple[ProductGroup, str, int]:
        return (self.group, self.task_id, self.repetition)


@dataclass(frozen=True)
class ProductMatrixPlan(JsonRecord):
    protocol: str
    cells: tuple[ProductCell, ...]
    profiles: tuple[ProductProfile, ...]
    tasks: tuple[MatrixTask, ...]
    repetitions: int

    def profile_by_group(self) -> dict[ProductGroup, ProductProfile]:
        return {profile.group: profile for profile in self.profiles}

    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class ProductObservation(JsonRecord):
    group: ProductGroup
    task_id: str
    repetition: int
    status: CellOutcome | TerminalStatus | str
    receipt_id: str | None = None
    model_calls: int | None = None
    tokens_total: int | None = None
    estimated_cost_usd: float | None = None
    total_wall_clock_sec: float | None = None
    npu_runs: int | None = None
    manual_interventions: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "group", _as_product_group(self.group))
        object.__setattr__(self, "status", normalize_outcome(self.status))
        if not self.task_id or type(self.repetition) is not int or self.repetition < 1:
            raise ValueError("observations require a task ID and positive repetition")
        for name in ("model_calls", "tokens_total", "npu_runs", "manual_interventions"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        for name in ("estimated_cost_usd", "total_wall_clock_sec"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative or None")

    def key(self) -> tuple[ProductGroup, str, int]:
        return (self.group, self.task_id, self.repetition)


@dataclass(frozen=True)
class CellDenominator(JsonRecord):
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
class MeasurementTotal(JsonRecord):
    """Observed sum over every measured attempt, never an imputed zero or success-only mean."""

    planned_count: int = 0
    available_count: int = 0
    sum_observed: int | float | None = None
    complete: bool = False


@dataclass(frozen=True)
class ProductGroupReport(JsonRecord):
    group: ProductGroup
    profile_id: str
    experiment_profile: str
    protocol: str
    fidelity: str
    cohort: str
    conditions: MatchingConditions
    budget: BudgetEnvelope
    denominator: CellDenominator
    by_task_family: dict[str, CellDenominator]
    by_hardware_tier: dict[str, CellDenominator]
    missing_cells: tuple[dict, ...]
    observations: tuple[ProductObservation, ...]
    tokens_total: MeasurementTotal
    cost_usd_total: MeasurementTotal
    model_calls_total: MeasurementTotal
    wall_clock_sec_total: MeasurementTotal
    npu_runs_total: MeasurementTotal
    manual_interventions_total: MeasurementTotal
    p3_lineage: dict | None = None
    budget_overruns: tuple[dict, ...] = ()
    budget_contract: dict | None = None

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload.update(
            description=PRODUCT_GROUP_DESCRIPTIONS[self.group],
            conditions_fingerprint=self.conditions.fingerprint(),
        )
        return payload


@dataclass(frozen=True)
class ProductMatrixReport(JsonRecord):
    protocol: str
    plan_digest: str
    observations: tuple[ProductObservation, ...]
    groups: dict[str, ProductGroupReport]
    overall: CellDenominator
    matched_overall: CellDenominator
    by_cohort: dict[str, CellDenominator]
    matched_cohorts: dict[str, tuple[str, ...]]
    faithful_external_groups: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload.update(
            denominator_scope=(
                "overall is a whole-matrix inventory, not a product average; matched_overall "
                "and each group report are the comparison denominators"
            ),
            interpretation=(
                "external faithful profiles are separate columns; they are not pooled into "
                "a single external average, and missing/infra/unknown cells remain in every denominator; "
                "observed totals include measured fail/infra/unknown attempts and exclude None"
            ),
        )
        return payload


@dataclass(frozen=True)
class MatchedPair:
    task_id: str
    task_family: str
    hardware_tier: str
    repetition: int
    left: ProductObservation | None
    right: ProductObservation | None

    @property
    def both_observed(self) -> bool:
        return self.left is not None and self.right is not None

    @property
    def both_valid(self) -> bool:
        return bool(
            self.left
            and self.right
            and self.left.status in {CellOutcome.PASS, CellOutcome.FAIL}
            and self.right.status in {CellOutcome.PASS, CellOutcome.FAIL}
        )
