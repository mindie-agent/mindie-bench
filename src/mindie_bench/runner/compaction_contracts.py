"""Immutable host compaction and fresh-context resume contracts.

[Role] Describe native trigger receipts, resumed observations and retry conditions.
[Invariants] Session identity, evidence age and missing observations are explicit;
native triggers do not by themselves prove restored reasoning or task success.
[Timeouts & Locks] Pure record construction and validation, no waits or locks.
[Side-Effect Policy] Protocol callbacks are declared here, never invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from mindie_bench.runner.protocol_json import JsonRecord, record_payload

COMPACTION_PROTOCOL = "mindie-bench-compaction/1"


OLD_EVIDENCE_AGE = 30


class CompactionMode(StrEnum):
    NATIVE = "native_compaction"
    FRESH = "fresh_context_resume"


class SemanticEvent(StrEnum):
    IN_FLIGHT_RUN = "in_flight_run"
    NEGATIVE_RESULT = "negative_result"
    BEST_UPDATED = "best_updated"


ALL_SEMANTIC_EVENTS = (
    SemanticEvent.IN_FLIGHT_RUN,
    SemanticEvent.NEGATIVE_RESULT,
    SemanticEvent.BEST_UPDATED,
)


class ProbeStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class RetryDecision(StrEnum):
    INVALID_DUPLICATE = "invalid_duplicate"
    VALID_RETRY = "valid_retry"
    VALID_REVALIDATION = "valid_revalidation"


@dataclass(frozen=True)
class HostCapabilities(JsonRecord):
    native_compaction: bool
    fresh_session: bool = True
    native_events: frozenset[SemanticEvent] = field(
        default_factory=lambda: frozenset(ALL_SEMANTIC_EVENTS)
    )
    host: str = "unknown"

    def __post_init__(self):
        if not self.host:
            raise ValueError("host name must be non-empty")
        object.__setattr__(
            self,
            "native_events",
            frozenset(SemanticEvent(event) for event in self.native_events),
        )


@dataclass(frozen=True)
class KnownFacts(JsonRecord):
    digest: str
    materials: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.digest:
            raise ValueError("known-facts digest must be non-empty")
        if any(not material for material in self.materials):
            raise ValueError("persisted material names must be non-empty")
        if len(set(self.materials)) != len(self.materials):
            raise ValueError("persisted material names must be unique")

    def material_set(self) -> frozenset[str]:
        return frozenset(self.materials)


@dataclass(frozen=True)
class NativeCompactionReceipt(JsonRecord):
    host: str
    session_id: str
    event: SemanticEvent
    host_receipt_id: str
    process_restarted: bool = False

    def __post_init__(self):
        if not self.host:
            raise ValueError("native compaction receipt requires a host name")
        if not self.session_id:
            raise ValueError("native compaction receipt requires a real session id")
        object.__setattr__(self, "event", SemanticEvent(self.event))
        if not self.host_receipt_id:
            raise ValueError("native compaction receipt requires a host receipt id")
        if self.process_restarted:
            raise ValueError("native compaction cannot be a restarted process")


@dataclass(frozen=True)
class FreshSession(JsonRecord):
    session_id: str
    known_facts_digest: str
    materials: tuple[str, ...]
    resumed_from: str | None = None

    def __post_init__(self):
        if not self.session_id:
            raise ValueError("fresh session requires a new session id")
        if not self.known_facts_digest:
            raise ValueError("fresh session requires a known-facts digest")
        if len(set(self.materials)) != len(self.materials):
            raise ValueError("fresh session persisted materials must be unique")


@dataclass(frozen=True)
class ResumeExpectation(JsonRecord):
    original_run_id: str
    expected_baseline: str
    required_evidence_ids: tuple[str, ...] = ()
    min_old_evidence: int = 0
    evidence_event_indices: tuple[tuple[str, int], ...] = ()
    recovery_event_index: int | None = None
    expected_first_effective_action: str | None = None
    expected_final_result: str | None = None

    def __post_init__(self):
        if not self.original_run_id or not self.expected_baseline:
            raise ValueError("resume expectation requires the original run and baseline")
        if len(set(self.required_evidence_ids)) != len(self.required_evidence_ids):
            raise ValueError("required evidence IDs must be unique")
        if type(self.min_old_evidence) is not int or self.min_old_evidence < 0:
            raise ValueError("min_old_evidence must be a non-negative integer")
        evidence_ids = [item[0] for item in self.evidence_event_indices]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence event indices must be unique per evidence id")
        if any(
            not evidence or type(index) is not int or index < 0
            for evidence, index in self.evidence_event_indices
        ):
            raise ValueError("evidence event indices require non-empty ids and non-negative indices")
        if self.recovery_event_index is not None and (
            type(self.recovery_event_index) is not int or self.recovery_event_index < 0
        ):
            raise ValueError("recovery_event_index must be a non-negative integer or None")

    def evidence_index_map(self) -> dict[str, int]:
        return dict(self.evidence_event_indices)


@dataclass(frozen=True)
class ResumeProbe(JsonRecord):
    """Observed resume facts; `None` means unobserved, never an invented value."""

    session_id: str
    known_facts_digest: str
    original_run_id_queried: str | None = None
    resubmitted_in_flight: bool | None = None
    baseline: str | None = None
    recovered_evidence_ids: tuple[str, ...] | None = None
    first_effective_action: str | None = None
    final_result: str | None = None

    def __post_init__(self):
        if not self.session_id or not self.known_facts_digest:
            raise ValueError("resume probe requires session identity and known-facts digest")
        if self.resubmitted_in_flight is not None and type(self.resubmitted_in_flight) is not bool:
            raise ValueError("resubmitted_in_flight must be bool or None")
        if self.recovered_evidence_ids is not None:
            if len(set(self.recovered_evidence_ids)) != len(self.recovered_evidence_ids):
                raise ValueError("recovered evidence IDs must be unique")
            if any(not evidence for evidence in self.recovered_evidence_ids):
                raise ValueError("recovered evidence IDs must be non-empty")


@dataclass(frozen=True)
class ResumeAssessment(JsonRecord):
    status: ProbeStatus
    original_run_queried: bool | None
    repeated_in_flight_submit: bool | None
    baseline_matches: bool | None
    old_evidence_recovered: int | None
    old_evidence_recovered_ids: tuple[str, ...]
    recovered_evidence_count: int | None
    old_evidence_required: int
    first_effective_action: str | None
    final_result: str | None
    missing_observations: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class NativeCompactionResult(JsonRecord):
    trigger_status: ProbeStatus
    session_id: str
    receipts: tuple[NativeCompactionReceipt, ...]
    resume_assessment: ResumeAssessment | None
    reason: str | None
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload.update(
            mode=CompactionMode.NATIVE.value,
            interpretation=(
                "trigger_status only proves a same-session native compaction trigger; "
                "resume_assessment separately evaluates recovered old evidence and resumption"
            ),
        )
        return payload


@dataclass(frozen=True)
class FreshContextResumeResult(JsonRecord):
    status: ProbeStatus
    original_session_id: str
    fresh_session: FreshSession | None
    assessment: ResumeAssessment | None
    new_session: bool | None
    same_known_facts: bool | None
    same_materials: bool | None
    reason: str | None
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = record_payload(self)
        payload["mode"] = CompactionMode.FRESH.value
        return payload


@dataclass(frozen=True)
class CompactionSuiteResult(JsonRecord):
    protocol: str
    native: NativeCompactionResult
    fresh: FreshContextResumeResult

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "native": self.native.to_dict(),
            "fresh": self.fresh.to_dict(),
            "interpretation": (
                "native compaction and fresh-context resume are separate tests; a trigger "
                "receipt or a fresh-session structural check is not task-success or G3 evaluation"
            ),
        }


@dataclass(frozen=True)
class AttemptRecord(JsonRecord):
    proposition: str
    environment: str
    input_digest: str
    implementation: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("proposition", "environment", "input_digest", "implementation"):
            if not getattr(self, name):
                raise ValueError(f"attempt {name} must be non-empty")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("attempt evidence IDs must be unique")

    def conditions(self) -> dict[str, str]:
        return {
            "proposition": self.proposition,
            "environment": self.environment,
            "input_digest": self.input_digest,
            "implementation": self.implementation,
        }

    def evidence_set(self) -> frozenset[str]:
        return frozenset(self.evidence_ids)


@dataclass(frozen=True)
class RetryAssessment(JsonRecord):
    decision: RetryDecision
    retry_allowed: bool
    prior_conclusion_applicable: bool
    same_conditions: bool
    requires_revalidation: bool
    new_evidence: tuple[str, ...]
    changed_conditions: tuple[str, ...]


class CompactionAdapter(Protocol):
    """Host callbacks. Implementations may perform real host work."""

    def capabilities(self) -> HostCapabilities: ...
    def trigger_native_compaction(
        self, session_id: str, event: SemanticEvent
    ) -> NativeCompactionReceipt | Mapping[str, Any] | None: ...
    def probe_native_resume(
        self, session_id: str, expectation: ResumeExpectation
    ) -> ResumeProbe | Mapping[str, Any] | None: ...
    def open_fresh_session(self, known_facts: KnownFacts) -> FreshSession | Mapping[str, Any]: ...
    def probe_resume(
        self, session: FreshSession, expectation: ResumeExpectation
    ) -> ResumeProbe | Mapping[str, Any]: ...
