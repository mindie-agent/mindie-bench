"""Receipt-driven host-native compaction and fresh-context resume tests.

[Role]
Expose two separated continuation tests: host-native compaction through a
supported host callable, and fresh-context resume through a new session.

[Invariants]
- Native compaction requires an advertised capability, a trigger callable and a
  receipt for the same real session identity; a new process is never substituted.
- A native trigger receipt proves only the trigger and session identity. Resume
  acceptance requires a separate same-session probe; both statuses are reported
  separately and neither is task success.
- Missing capability returns `unavailable`; an invalid or absent receipt/probe
  returns `inconclusive`. Neither is reported as a successful G3 continuation.
- Fresh resume uses a different session id with the same known-facts digest and
  the same available persisted materials.
- Evidence is "old" only when a frozen expectation supplies its event index and
  the recovery/truncation index and its age exceeds 30 events; unknown age is
  missing evidence, never an old-evidence pass.
- Resume assessment separates original-run query, resubmission, baseline,
  old-evidence recovery, first action and final result from missing observations.
- A repeat is invalid only when proposition, environment, input and implementation
  match with no new evidence; changed conditions require revalidation.

[Timeouts & Locks]
- Pure in-memory callbacks; the host adapter owns real operations, timeouts and locks.

[Side-Effect Policy]
- No request, network, process, history replay or external message; tests use an in-memory adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mindie_bench.runner.compaction_contracts import (
    ALL_SEMANTIC_EVENTS as ALL_SEMANTIC_EVENTS,
)
from mindie_bench.runner.compaction_contracts import (
    COMPACTION_PROTOCOL as COMPACTION_PROTOCOL,
)
from mindie_bench.runner.compaction_contracts import (
    OLD_EVIDENCE_AGE as OLD_EVIDENCE_AGE,
)
from mindie_bench.runner.compaction_contracts import (
    AttemptRecord as AttemptRecord,
)
from mindie_bench.runner.compaction_contracts import (
    CompactionAdapter as CompactionAdapter,
)
from mindie_bench.runner.compaction_contracts import (
    CompactionMode as CompactionMode,
)
from mindie_bench.runner.compaction_contracts import (
    CompactionSuiteResult as CompactionSuiteResult,
)
from mindie_bench.runner.compaction_contracts import (
    FreshContextResumeResult as FreshContextResumeResult,
)
from mindie_bench.runner.compaction_contracts import (
    FreshSession as FreshSession,
)
from mindie_bench.runner.compaction_contracts import (
    HostCapabilities as HostCapabilities,
)
from mindie_bench.runner.compaction_contracts import (
    KnownFacts as KnownFacts,
)
from mindie_bench.runner.compaction_contracts import (
    NativeCompactionReceipt as NativeCompactionReceipt,
)
from mindie_bench.runner.compaction_contracts import (
    NativeCompactionResult as NativeCompactionResult,
)
from mindie_bench.runner.compaction_contracts import (
    ProbeStatus as ProbeStatus,
)
from mindie_bench.runner.compaction_contracts import (
    ResumeAssessment as ResumeAssessment,
)
from mindie_bench.runner.compaction_contracts import (
    ResumeExpectation as ResumeExpectation,
)
from mindie_bench.runner.compaction_contracts import (
    ResumeProbe as ResumeProbe,
)
from mindie_bench.runner.compaction_contracts import (
    RetryAssessment as RetryAssessment,
)
from mindie_bench.runner.compaction_contracts import (
    RetryDecision as RetryDecision,
)
from mindie_bench.runner.compaction_contracts import (
    SemanticEvent as SemanticEvent,
)


def _coerce(cls, value: object, label: str):
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**value)
    raise TypeError(f"{label} must be a {cls.__name__} or mapping")


def _capabilities(adapter: object) -> HostCapabilities:
    callback = getattr(adapter, "capabilities", None)
    if not callable(callback):
        return HostCapabilities(
            native_compaction=False, fresh_session=False, host=type(adapter).__name__
        )
    return _coerce(HostCapabilities, callback(), "adapter capabilities")


def _validate_session_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("session id must be a non-empty string")
    return value


def trigger_native_compaction(
    adapter: object,
    *,
    session_id: str,
    event: SemanticEvent | str,
) -> NativeCompactionResult:
    """Trigger one native compaction event on a host that explicitly supports it."""
    session_id = _validate_session_id(session_id)
    event = SemanticEvent(event)
    capabilities = _capabilities(adapter)
    base_limitations = (
        "native compaction receipt proves the trigger and session identity, not resume or task success",
        "a new process or replayed history is not native compaction",
    )
    trigger = getattr(adapter, "trigger_native_compaction", None)
    if not capabilities.native_compaction or not callable(trigger):
        return NativeCompactionResult(
            trigger_status=ProbeStatus.UNAVAILABLE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason="host does not advertise a callable native-compaction interface",
            limitations=base_limitations,
        )
    if event not in capabilities.native_events:
        return NativeCompactionResult(
            trigger_status=ProbeStatus.UNAVAILABLE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason=f"host does not support native compaction at {event.value}",
            limitations=base_limitations,
        )
    receipt = trigger(session_id, event)
    if receipt is None:
        return NativeCompactionResult(
            trigger_status=ProbeStatus.INCONCLUSIVE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason="host returned no native compaction receipt",
            limitations=base_limitations,
        )
    try:
        parsed = _coerce(NativeCompactionReceipt, receipt, "native compaction receipt")
    except (TypeError, ValueError) as error:
        return NativeCompactionResult(
            trigger_status=ProbeStatus.INCONCLUSIVE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason=f"invalid native compaction receipt: {error}",
            limitations=base_limitations,
        )
    if parsed.session_id != session_id:
        return NativeCompactionResult(
            trigger_status=ProbeStatus.INCONCLUSIVE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason="native compaction receipt changed the real session identity",
            limitations=base_limitations,
        )
    if parsed.event != event:
        return NativeCompactionResult(
            trigger_status=ProbeStatus.INCONCLUSIVE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason="native compaction receipt is for a different semantic event",
            limitations=base_limitations,
        )
    if capabilities.host != "unknown" and parsed.host != capabilities.host:
        return NativeCompactionResult(
            trigger_status=ProbeStatus.INCONCLUSIVE,
            session_id=session_id,
            receipts=(),
            resume_assessment=None,
            reason="native compaction receipt came from a different host",
            limitations=base_limitations,
        )
    return NativeCompactionResult(
        trigger_status=ProbeStatus.PASS,
        session_id=session_id,
        receipts=(parsed,),
        resume_assessment=None,
        reason=None,
        limitations=base_limitations,
    )


def _native_resume_assessment(
    adapter: object,
    *,
    session_id: str,
    expectation: ResumeExpectation,
    known_facts: KnownFacts | None,
) -> tuple[ResumeAssessment | None, str | None]:
    callback = getattr(adapter, "probe_native_resume", None)
    if not callable(callback):
        return (
            None,
            "host has no same-session native resume probe; trigger status is not resume acceptance",
        )
    observed = callback(session_id, expectation)
    if observed is None:
        return (
            None,
            "host returned no same-session native resume probe; trigger status is not resume acceptance",
        )
    try:
        probe = _coerce(ResumeProbe, observed, "native resume probe")
    except (TypeError, ValueError) as error:
        return None, f"invalid native resume probe: {error}"
    if probe.session_id != session_id:
        return None, "native resume probe belongs to a different session"
    if known_facts is not None and probe.known_facts_digest != known_facts.digest:
        return None, "native resume probe reported a different known-facts digest"
    return evaluate_resume(probe, expectation), None


def run_native_compaction_events(
    adapter: object,
    *,
    session_id: str,
    events: Sequence[SemanticEvent | str] = ALL_SEMANTIC_EVENTS,
    expectation: ResumeExpectation | None = None,
    known_facts: KnownFacts | None = None,
) -> NativeCompactionResult:
    """Trigger semantic events and, when supplied, probe same-session resume separately."""
    if not events:
        raise ValueError("at least one semantic event is required")
    session_id = _validate_session_id(session_id)
    results = [
        trigger_native_compaction(adapter, session_id=session_id, event=event) for event in events
    ]
    receipts = tuple(receipt for result in results for receipt in result.receipts)
    statuses = {result.trigger_status for result in results}
    if statuses == {ProbeStatus.PASS}:
        trigger_status = ProbeStatus.PASS
    elif statuses == {ProbeStatus.UNAVAILABLE}:
        trigger_status = ProbeStatus.UNAVAILABLE
    else:
        trigger_status = ProbeStatus.INCONCLUSIVE
    reasons = [result.reason for result in results if result.reason]
    limitations = list(
        dict.fromkeys(limitation for result in results for limitation in result.limitations)
    )
    assessment = None
    if expectation is None:
        limitations.append("native resume assessment was not requested; receipts prove trigger only")
    elif trigger_status != ProbeStatus.PASS:
        limitations.append("native resume assessment not attempted without a passing trigger receipt")
    else:
        assessment, resume_reason = _native_resume_assessment(
            adapter,
            session_id=session_id,
            expectation=expectation,
            known_facts=known_facts,
        )
        if resume_reason:
            reasons.append(resume_reason)
            limitations.append(resume_reason)
    limitations.append(
        "trigger status and same-session resume assessment are separate; neither is task success"
    )
    return NativeCompactionResult(
        trigger_status=trigger_status,
        session_id=session_id,
        receipts=receipts,
        resume_assessment=assessment,
        reason="; ".join(dict.fromkeys(reasons)) or None,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def evaluate_resume(probe: ResumeProbe, expectation: ResumeExpectation) -> ResumeAssessment:
    """Assess observable resume facts without filling in unobserved values."""
    missing = []
    failures = []
    if probe.original_run_id_queried is None:
        missing.append("original_run_id_query")
        original_run_queried = None
    elif probe.original_run_id_queried != expectation.original_run_id:
        failures.append("wrong_original_run_id")
        original_run_queried = False
    else:
        original_run_queried = True
    if probe.resubmitted_in_flight is None:
        missing.append("in_flight_resubmission")
    elif probe.resubmitted_in_flight:
        failures.append("duplicate_in_flight_submit")
    if probe.baseline is None:
        missing.append("baseline")
        baseline_matches = None
    else:
        baseline_matches = probe.baseline == expectation.expected_baseline
        if not baseline_matches:
            failures.append("wrong_baseline")
    recovered_count = None
    old_count = None
    old_ids: tuple[str, ...] = ()
    if probe.recovered_evidence_ids is None:
        missing.append("recovered_evidence")
    else:
        recovered = tuple(probe.recovered_evidence_ids)
        recovered_count = len(recovered)
        absent = sorted(set(expectation.required_evidence_ids) - set(recovered))
        if absent:
            failures.append("missing_required_evidence:" + ",".join(absent))
        index_by_id = expectation.evidence_index_map()
        recovery_index = expectation.recovery_event_index
        old = []
        unknown_age = []
        for evidence_id in recovered:
            evidence_index = index_by_id.get(evidence_id)
            if evidence_index is None or recovery_index is None:
                unknown_age.append(evidence_id)
            elif recovery_index - evidence_index > OLD_EVIDENCE_AGE:
                old.append(evidence_id)
        old_count = len(old)
        old_ids = tuple(old)
        if unknown_age:
            missing.append("evidence_event_age:" + ",".join(unknown_age))
        if old_count < expectation.min_old_evidence and not unknown_age:
            failures.append(f"old_evidence_below_minimum:{old_count}<{expectation.min_old_evidence}")
    if expectation.expected_first_effective_action is not None:
        if probe.first_effective_action is None:
            missing.append("first_effective_action")
        elif probe.first_effective_action != expectation.expected_first_effective_action:
            failures.append("unexpected_first_effective_action")
    if probe.final_result is None:
        missing.append("final_result")
    elif (
        expectation.expected_final_result is not None
        and probe.final_result != expectation.expected_final_result
    ):
        failures.append("unexpected_final_result")
    if failures:
        status = ProbeStatus.FAIL
    elif missing:
        status = ProbeStatus.INCONCLUSIVE
    else:
        status = ProbeStatus.PASS
    return ResumeAssessment(
        status=status,
        original_run_queried=original_run_queried,
        repeated_in_flight_submit=probe.resubmitted_in_flight,
        baseline_matches=baseline_matches,
        old_evidence_recovered=old_count,
        old_evidence_recovered_ids=old_ids,
        recovered_evidence_count=recovered_count,
        old_evidence_required=expectation.min_old_evidence,
        first_effective_action=probe.first_effective_action,
        final_result=probe.final_result,
        missing_observations=tuple(missing),
        failures=tuple(failures),
    )


def run_fresh_context_resume(
    adapter: object,
    *,
    original_session_id: str,
    known_facts: KnownFacts,
    expectation: ResumeExpectation,
) -> FreshContextResumeResult:
    """Open a genuinely new session with the same known facts and persisted material."""
    original_session_id = _validate_session_id(original_session_id)
    capabilities = _capabilities(adapter)
    limitations = (
        "fresh-context resume is not host-native compaction",
        "only supplied persisted materials are available; full history is not read",
    )
    opener = getattr(adapter, "open_fresh_session", None)
    if not capabilities.fresh_session or not callable(opener):
        return FreshContextResumeResult(
            status=ProbeStatus.UNAVAILABLE,
            original_session_id=original_session_id,
            fresh_session=None,
            assessment=None,
            new_session=None,
            same_known_facts=None,
            same_materials=None,
            reason="host does not advertise a callable fresh-session interface",
            limitations=limitations,
        )
    opened = opener(known_facts)
    if opened is None:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=None,
            assessment=None,
            new_session=None,
            same_known_facts=None,
            same_materials=None,
            reason="host returned no fresh session",
            limitations=limitations,
        )
    try:
        session = _coerce(FreshSession, opened, "fresh session")
    except (TypeError, ValueError) as error:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=None,
            assessment=None,
            new_session=None,
            same_known_facts=None,
            same_materials=None,
            reason=f"invalid fresh session: {error}",
            limitations=limitations,
        )
    new_session = session.session_id != original_session_id
    same_known_facts = session.known_facts_digest == known_facts.digest
    same_materials = set(session.materials) == set(known_facts.materials)
    structural_reason = None
    if not new_session:
        structural_reason = "fresh resume reused the original session id"
    elif session.resumed_from not in {None, original_session_id}:
        structural_reason = "fresh session resumed from an unexpected original session"
    elif not same_known_facts:
        structural_reason = "fresh session changed the known-facts digest"
    elif not same_materials:
        structural_reason = "fresh session changed the available persisted materials"
    if structural_reason:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=session,
            assessment=None,
            new_session=new_session,
            same_known_facts=same_known_facts,
            same_materials=same_materials,
            reason=structural_reason,
            limitations=limitations,
        )
    probe_callback = getattr(adapter, "probe_resume", None)
    if not callable(probe_callback):
        return FreshContextResumeResult(
            status=ProbeStatus.UNAVAILABLE,
            original_session_id=original_session_id,
            fresh_session=session,
            assessment=None,
            new_session=new_session,
            same_known_facts=same_known_facts,
            same_materials=same_materials,
            reason="host does not expose a resume observation callback",
            limitations=limitations,
        )
    observed = probe_callback(session, expectation)
    if observed is None:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=session,
            assessment=None,
            new_session=new_session,
            same_known_facts=same_known_facts,
            same_materials=same_materials,
            reason="host returned no resume probe",
            limitations=limitations,
        )
    try:
        probe = _coerce(ResumeProbe, observed, "resume probe")
    except (TypeError, ValueError) as error:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=session,
            assessment=None,
            new_session=new_session,
            same_known_facts=same_known_facts,
            same_materials=same_materials,
            reason=f"invalid resume probe: {error}",
            limitations=limitations,
        )
    if probe.session_id != session.session_id:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=session,
            assessment=None,
            new_session=new_session,
            same_known_facts=same_known_facts,
            same_materials=same_materials,
            reason="resume probe belongs to a different session",
            limitations=limitations,
        )
    if probe.known_facts_digest != known_facts.digest:
        return FreshContextResumeResult(
            status=ProbeStatus.INCONCLUSIVE,
            original_session_id=original_session_id,
            fresh_session=session,
            assessment=None,
            new_session=new_session,
            same_known_facts=same_known_facts,
            same_materials=same_materials,
            reason="resume probe reported a different known-facts digest",
            limitations=limitations,
        )
    assessment = evaluate_resume(probe, expectation)
    return FreshContextResumeResult(
        status=assessment.status,
        original_session_id=original_session_id,
        fresh_session=session,
        assessment=assessment,
        new_session=new_session,
        same_known_facts=same_known_facts,
        same_materials=same_materials,
        reason=None,
        limitations=limitations,
    )


def run_compaction_suite(
    adapter: object,
    *,
    native_session_id: str,
    known_facts: KnownFacts,
    expectation: ResumeExpectation,
    events: Sequence[SemanticEvent | str] = ALL_SEMANTIC_EVENTS,
) -> CompactionSuiteResult:
    """Run both continuation modes and keep trigger and resume results separate."""
    return CompactionSuiteResult(
        protocol=COMPACTION_PROTOCOL,
        native=run_native_compaction_events(
            adapter,
            session_id=native_session_id,
            events=events,
            expectation=expectation,
            known_facts=known_facts,
        ),
        fresh=run_fresh_context_resume(
            adapter,
            original_session_id=native_session_id,
            known_facts=known_facts,
            expectation=expectation,
        ),
    )


def evaluate_retry(prior: AttemptRecord, retry: AttemptRecord) -> RetryAssessment:
    """Decide whether a repeated attempt is justified or an invalid duplicate."""
    prior_conditions = prior.conditions()
    retry_conditions = retry.conditions()
    changed = tuple(
        name for name in prior_conditions if prior_conditions[name] != retry_conditions[name]
    )
    same_conditions = not changed
    new_evidence = tuple(sorted(retry.evidence_set() - prior.evidence_set()))
    if same_conditions and not new_evidence:
        decision = RetryDecision.INVALID_DUPLICATE
        retry_allowed = False
    elif same_conditions:
        decision = RetryDecision.VALID_RETRY
        retry_allowed = True
    else:
        decision = RetryDecision.VALID_REVALIDATION
        retry_allowed = True
    return RetryAssessment(
        decision=decision,
        retry_allowed=retry_allowed,
        prior_conclusion_applicable=same_conditions,
        same_conditions=same_conditions,
        requires_revalidation=not same_conditions,
        new_evidence=new_evidence,
        changed_conditions=changed,
    )
