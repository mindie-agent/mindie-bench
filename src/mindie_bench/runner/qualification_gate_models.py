"""Declarative G0-G3 gate contracts and honest receipt states.

[Role]
Hold the immutable specifications and receipts shared by the qualification gate
runner and its sustained-ABBA / subgraph-module evaluators.

[Invariants]
- `completed`, `failed` and `unknown` are distinct; a timed-out command or an
  indeterminate PID is never converted into success.
- Timeout arithmetic is validated before a specification can run.
- Specs name hidden inputs by digest and never embed oracle contents.

[Timeouts & Locks]
- Pure data validation only; no subprocess, clock, lock or hardware access.

[Side-Effect Policy]
- No I/O. Construction only normalizes caller-declared immutable values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

SCHEMA = "mindie-bench-qualification-gate/1"


class GateState(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TimeoutArithmeticError(ValueError):
    """The outer gate deadline cannot cover every inner stage plus cleanup."""


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def plan_gate_timeouts(
    stage_budgets: Mapping[str, float],
    *,
    cleanup_grace_sec: float,
    outer_timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Validate and expose the gate timeout hierarchy before execution."""
    if not stage_budgets:
        raise ValueError("a gate requires at least one inner stage budget")
    budgets = {}
    for name, value in stage_budgets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("stage names must be nonempty text")
        _positive(value, f"stage budget {name}")
        budgets[name] = float(value)
    if (
        isinstance(cleanup_grace_sec, bool)
        or not isinstance(cleanup_grace_sec, (int, float))
        or not math.isfinite(cleanup_grace_sec)
        or cleanup_grace_sec < 0
    ):
        raise ValueError("cleanup_grace_sec must be a nonnegative number")
    inner = sum(budgets.values())
    minimum = inner + cleanup_grace_sec
    outer = minimum if outer_timeout_sec is None else float(outer_timeout_sec)
    if outer < minimum:
        raise TimeoutArithmeticError(
            f"outer_timeout {outer:g}s < sum(inner_stage_budgets) {inner:g}s "
            f"+ cleanup_grace {cleanup_grace_sec:g}s"
        )
    return {
        "stage_budgets": budgets,
        "inner_stage_budget_sec": round(inner, 6),
        "cleanup_grace_sec": float(cleanup_grace_sec),
        "outer_timeout_sec": round(outer, 6),
        "valid": True,
    }


@dataclass(frozen=True)
class GateCommand:
    """One declared command with its own inner budget and sentinel contract."""

    argv: tuple[str, ...]
    timeout_sec: float
    purpose: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    expect_stdout: tuple[str, ...] = ()
    expect_stderr: tuple[str, ...] = ()
    forbidden_stdout: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.argv or any(not isinstance(part, str) or not part for part in self.argv):
            raise ValueError("a gate command requires nonempty argv text")
        _positive(self.timeout_sec, "command timeout_sec")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "env", {str(k): str(v) for k, v in dict(self.env).items()})
        for name in ("expect_stdout", "expect_stderr", "forbidden_stdout"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))


@dataclass(frozen=True)
class CommandReceipt:
    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None

    @property
    def state(self) -> GateState:
        if self.timed_out or self.exit_code is None:
            return GateState.UNKNOWN
        return GateState.COMPLETED if self.exit_code == 0 else GateState.FAILED


@dataclass
class StageReceipt:
    name: str
    state: GateState
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class GateResult:
    gate: str
    state: GateState
    stages: list[StageReceipt]
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    timeout_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "gate": self.gate,
            "state": self.state.value,
            "stages": [stage.to_dict() for stage in self.stages],
            "metrics": self.metrics,
            "reasons": self.reasons,
            "timeout_plan": self.timeout_plan,
        }


@dataclass(frozen=True)
class G0Spec:
    contract_tests: tuple[GateCommand, ...] = ()
    manifests: tuple[str, ...] = ()
    generated_json: tuple[str, ...] = ()
    ldd_targets: tuple[str, ...] = ()
    symbol_checks: tuple[GateCommand, ...] = ()
    stage_budgets: Mapping[str, float] = field(default_factory=dict)
    outer_timeout_sec: float | None = None
    cleanup_grace_sec: float = 5.0


@dataclass(frozen=True)
class NativeOperatorArtifacts:
    name: str
    aicore_object: str | None
    aicpu_registration: str | None


@dataclass(frozen=True)
class G1Spec:
    source_root: str
    jobs: int = 8
    build_system: str = "make"
    build_target: str = "native-gate-opp"
    build_command: GateCommand | None = None
    build_sentinel: tuple[str, ...] = ()
    debug_lock_paths: tuple[str, ...] = ()
    strict_locks: bool = True
    eager_qualification: GateCommand | None = None
    aclgraph_qualification: GateCommand | None = None
    stage_budgets: Mapping[str, float] = field(default_factory=dict)
    outer_timeout_sec: float | None = None
    cleanup_grace_sec: float = 5.0


@dataclass
class DebugLock:
    path: str
    pid: int | None
    pgid: int | None
    state: GateState
    detail: str


@dataclass(frozen=True)
class G2Spec:
    baseline_id: str
    candidate_id: str
    frozen_baseline_sha256: str
    samples: tuple["AbbaSample", ...]
    rounds: int
    numerical: "NumericalQualification"
    warmup_rounds: int = 1
    min_speedup_ratio: float = 1.0
    min_sustained_window_sec: float = 10.0
    sustained_clock_mhz: tuple[float, float] | None = None
    stage_budgets: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class G3Spec:
    module_id: str
    topology: "ModuleTopology"
    boundary_pairs: tuple["BoundaryPair", ...]
    paired_runs: tuple["ModuleRun", ...]
    guard_workloads: tuple["GuardWorkload", ...]
    stage_budgets: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NumericalQualification:
    baseline_passed: bool
    candidate_passed: bool
    sentinel_ok: bool
    max_abs_error: float | None
    tolerance: float
    cosine_similarity: float | None = None

    @property
    def qualified(self) -> bool:
        return (
            self.baseline_passed
            and self.candidate_passed
            and self.sentinel_ok
            and self.max_abs_error is not None
            and self.tolerance >= 0
            and self.max_abs_error <= self.tolerance
        )


@dataclass(frozen=True)
class AbbaSample:
    variant: str
    phase: str
    round_index: int
    duration_ns: int
    aclgraph_replays: int
    clock_mhz: float | None = None
    sustained: bool = False


@dataclass(frozen=True)
class ModuleTopology:
    tp: int
    die_count: int
    observed_ranks: tuple[int, ...]


@dataclass(frozen=True)
class BoundaryPair:
    producer: str
    consumer: str
    boundary: str
    logical_identity: str
    observed_identity: str
    equivalent: bool
    ownership_preserved: bool
    numerical_passed: bool
    executed_calls: int


@dataclass(frozen=True)
class ModuleRun:
    variant: str
    run_id: str
    inputs_sha256: str
    numerical_passed: bool
    sentinel_ok: bool
    executed_calls: int


@dataclass(frozen=True)
class GuardWorkload:
    name: str
    measurement_valid: bool
    passed: bool
    baseline_ns: int | None = None
    candidate_ns: int | None = None
    max_regression_ratio: float = 1.0
    required: bool = True
