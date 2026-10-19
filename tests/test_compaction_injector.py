from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_bench.runner.compaction_injector import (
    ALL_SEMANTIC_EVENTS,
    COMPACTION_PROTOCOL,
    OLD_EVIDENCE_AGE,
    AttemptRecord,
    FreshSession,
    HostCapabilities,
    KnownFacts,
    NativeCompactionReceipt,
    ProbeStatus,
    ResumeExpectation,
    ResumeProbe,
    RetryDecision,
    SemanticEvent,
    evaluate_resume,
    evaluate_retry,
    run_compaction_suite,
    run_fresh_context_resume,
    run_native_compaction_events,
)


class MemoryHost:
    def __init__(
        self,
        *,
        native: bool = True,
        fresh: bool = True,
        receipt: NativeCompactionReceipt | None = None,
        fresh_session: FreshSession | None = None,
        probe: ResumeProbe | None = None,
        native_probe: ResumeProbe | None = None,
    ):
        self.native = native
        self.fresh = fresh
        self.receipt = receipt
        self.fresh_session = fresh_session
        self.probe = probe
        self.native_probe = native_probe
        self.native_calls = []
        self.fresh_calls = []
        self.probe_calls = []
        self.native_probe_calls = []

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            native_compaction=self.native,
            fresh_session=self.fresh,
            host="memory-host",
        )

    def trigger_native_compaction(self, session_id, event):
        self.native_calls.append((session_id, event))
        return self.receipt

    def probe_native_resume(self, session_id, expectation):
        self.native_probe_calls.append((session_id, expectation))
        return self.native_probe

    def open_fresh_session(self, known_facts):
        self.fresh_calls.append(known_facts)
        return self.fresh_session

    def probe_resume(self, session, expectation):
        self.probe_calls.append((session, expectation))
        return self.probe


def evidence_ids(count: int = 31) -> tuple[str, ...]:
    return tuple(f"evidence-{index:02d}" for index in range(count))


def expectation(evidence_count: int = 31) -> ResumeExpectation:
    ids = evidence_ids(evidence_count)
    return ResumeExpectation(
        original_run_id="run-1",
        expected_baseline="baseline-7",
        required_evidence_ids=ids,
        min_old_evidence=evidence_count,
        evidence_event_indices=tuple((evidence, index) for index, evidence in enumerate(ids)),
        recovery_event_index=100,
        expected_first_effective_action="query-original-run",
        expected_final_result="completed_pass",
    )


def good_probe(
    *,
    session_id: str = "session-fresh",
    digest: str = "facts-sha",
    evidence_count: int = 31,
) -> ResumeProbe:
    return ResumeProbe(
        session_id=session_id,
        known_facts_digest=digest,
        original_run_id_queried="run-1",
        resubmitted_in_flight=False,
        baseline="baseline-7",
        recovered_evidence_ids=evidence_ids(evidence_count),
        first_effective_action="query-original-run",
        final_result="completed_pass",
    )


def known_facts() -> KnownFacts:
    return KnownFacts(
        digest="facts-sha",
        materials=("PROGRESS.md", "experiment-ledger.jsonl", "run-manifest.json"),
    )


def test_native_compaction_is_unavailable_without_supported_callable():
    host = MemoryHost(native=False)
    result = run_native_compaction_events(host, session_id="session-1")
    assert result.trigger_status == ProbeStatus.UNAVAILABLE
    assert result.receipts == ()
    assert result.resume_assessment is None
    assert host.native_calls == []
    assert any("not native compaction" in limitation for limitation in result.limitations)


def test_native_compaction_receipt_must_keep_session_and_event_identity():
    receipt = NativeCompactionReceipt(
        host="memory-host",
        session_id="session-1",
        event=SemanticEvent.IN_FLIGHT_RUN,
        host_receipt_id="receipt-1",
    )
    host = MemoryHost(receipt=receipt)
    result = run_native_compaction_events(
        host, session_id="session-1", events=[SemanticEvent.IN_FLIGHT_RUN]
    )
    assert result.trigger_status == ProbeStatus.PASS
    assert result.receipts == (receipt,)
    assert host.native_calls == [("session-1", SemanticEvent.IN_FLIGHT_RUN)]

    wrong_session = replace(receipt, session_id="session-2")
    rejected = run_native_compaction_events(
        MemoryHost(receipt=wrong_session),
        session_id="session-1",
        events=[SemanticEvent.IN_FLIGHT_RUN],
    )
    assert rejected.trigger_status == ProbeStatus.INCONCLUSIVE
    assert rejected.receipts == ()
    assert "session identity" in (rejected.reason or "")

    wrong_event = replace(receipt, event=SemanticEvent.BEST_UPDATED)
    rejected = run_native_compaction_events(
        MemoryHost(receipt=wrong_event),
        session_id="session-1",
        events=[SemanticEvent.IN_FLIGHT_RUN],
    )
    assert rejected.trigger_status == ProbeStatus.INCONCLUSIVE


def test_all_three_semantic_events_are_explicitly_triggered():
    receipts = [
        NativeCompactionReceipt(
            host="memory-host",
            session_id="session-1",
            event=event,
            host_receipt_id=f"receipt-{event.value}",
        )
        for event in ALL_SEMANTIC_EVENTS
    ]

    class EventHost(MemoryHost):
        def trigger_native_compaction(self, session_id, event):
            self.native_calls.append((session_id, event))
            return receipts[len(self.native_calls) - 1]

    host = EventHost()
    result = run_native_compaction_events(host, session_id="session-1")
    assert result.trigger_status == ProbeStatus.PASS
    assert [receipt.event for receipt in result.receipts] == list(ALL_SEMANTIC_EVENTS)
    assert [event for _, event in host.native_calls] == list(ALL_SEMANTIC_EVENTS)


def test_missing_native_receipt_is_inconclusive_not_success():
    result = run_native_compaction_events(MemoryHost(receipt=None), session_id="session-1")
    assert result.trigger_status == ProbeStatus.INCONCLUSIVE
    assert "no native compaction receipt" in (result.reason or "")


def test_native_trigger_status_and_same_session_resume_assessment_are_separate():
    receipt = NativeCompactionReceipt(
        host="memory-host",
        session_id="session-1",
        event=SemanticEvent.NEGATIVE_RESULT,
        host_receipt_id="receipt-1",
    )
    host = MemoryHost(
        receipt=receipt,
        native_probe=good_probe(session_id="session-1", evidence_count=1),
    )
    result = run_native_compaction_events(
        host,
        session_id="session-1",
        events=[SemanticEvent.NEGATIVE_RESULT],
        expectation=expectation(1),
        known_facts=known_facts(),
    )
    assert result.trigger_status == ProbeStatus.PASS
    assert result.resume_assessment is not None
    assert result.resume_assessment.status == ProbeStatus.PASS
    payload = result.to_dict()
    assert payload["trigger_status"] == "pass"
    assert payload["resume_assessment"]["status"] == "pass"
    assert host.native_probe_calls[0][0] == "session-1"

    no_probe = run_native_compaction_events(
        MemoryHost(receipt=receipt),
        session_id="session-1",
        events=[SemanticEvent.NEGATIVE_RESULT],
        expectation=expectation(1),
        known_facts=known_facts(),
    )
    assert no_probe.trigger_status == ProbeStatus.PASS
    assert no_probe.resume_assessment is None
    assert any(
        "trigger status is not resume acceptance" in limitation for limitation in no_probe.limitations
    )


def test_fresh_resume_requires_new_session_same_known_facts_and_materials():
    host = MemoryHost(
        fresh_session=FreshSession(
            session_id="session-fresh",
            known_facts_digest="facts-sha",
            materials=("PROGRESS.md", "experiment-ledger.jsonl", "run-manifest.json"),
            resumed_from="session-1",
        ),
        probe=good_probe(),
    )
    result = run_fresh_context_resume(
        host,
        original_session_id="session-1",
        known_facts=known_facts(),
        expectation=expectation(),
    )
    assert result.status == ProbeStatus.PASS
    assert result.new_session is True
    assert result.same_known_facts is True
    assert result.same_materials is True
    assert result.assessment is not None
    assert result.assessment.original_run_queried is True
    assert result.assessment.repeated_in_flight_submit is False
    assert result.assessment.baseline_matches is True
    assert result.assessment.old_evidence_recovered == 31
    assert len(result.assessment.old_evidence_recovered_ids) == 31
    assert result.assessment.final_result == "completed_pass"
    assert result.to_dict()["mode"] == "fresh_context_resume"
    assert result.to_dict()["assessment"]["status"] == "pass"


@pytest.mark.parametrize(
    "session,expected_reason",
    [
        (
            FreshSession(
                session_id="session-1",
                known_facts_digest="facts-sha",
                materials=("PROGRESS.md", "experiment-ledger.jsonl", "run-manifest.json"),
            ),
            "original session id",
        ),
        (
            FreshSession(
                session_id="session-fresh",
                known_facts_digest="different-facts",
                materials=("PROGRESS.md", "experiment-ledger.jsonl", "run-manifest.json"),
            ),
            "known-facts digest",
        ),
        (
            FreshSession(
                session_id="session-fresh",
                known_facts_digest="facts-sha",
                materials=("PROGRESS.md",),
            ),
            "persisted materials",
        ),
    ],
)
def test_fresh_resume_rejects_reused_session_or_changed_material(session, expected_reason):
    host = MemoryHost(fresh_session=session, probe=good_probe())
    result = run_fresh_context_resume(
        host,
        original_session_id="session-1",
        known_facts=known_facts(),
        expectation=expectation(),
    )
    assert result.status == ProbeStatus.INCONCLUSIVE
    assert expected_reason in (result.reason or "")
    assert result.assessment is None
    assert host.probe_calls == []


def test_resume_assessment_separates_missing_observations_from_failures():
    missing_everything = ResumeProbe(session_id="session-fresh", known_facts_digest="facts-sha")
    assessment = evaluate_resume(missing_everything, expectation(evidence_count=3))
    assert assessment.status == ProbeStatus.INCONCLUSIVE
    assert assessment.original_run_queried is None
    assert "original_run_id_query" in assessment.missing_observations
    assert "baseline" in assessment.missing_observations
    assert assessment.failures == ()

    failed = ResumeProbe(
        session_id="session-fresh",
        known_facts_digest="facts-sha",
        original_run_id_queried="run-2",
        resubmitted_in_flight=True,
        baseline="baseline-8",
        recovered_evidence_ids=("evidence-00",),
        first_effective_action="submit-again",
        final_result=None,
    )
    assessment = evaluate_resume(failed, expectation(evidence_count=31))
    assert assessment.status == ProbeStatus.FAIL
    assert "wrong_original_run_id" in assessment.failures
    assert "duplicate_in_flight_submit" in assessment.failures
    assert "wrong_baseline" in assessment.failures
    assert any(item.startswith("old_evidence_below_minimum") for item in assessment.failures)
    assert "final_result" in assessment.missing_observations
    assert assessment.repeated_in_flight_submit is True
    assert assessment.baseline_matches is False


def test_one_key_evidence_from_forty_events_ago_counts_as_old():
    old_expectation = ResumeExpectation(
        original_run_id="run-1",
        expected_baseline="baseline-7",
        required_evidence_ids=("key-evidence",),
        min_old_evidence=1,
        evidence_event_indices=(("key-evidence", 10),),
        recovery_event_index=50,
        expected_first_effective_action="query-original-run",
        expected_final_result="completed_pass",
    )
    probe = ResumeProbe(
        session_id="session-fresh",
        known_facts_digest="facts-sha",
        original_run_id_queried="run-1",
        resubmitted_in_flight=False,
        baseline="baseline-7",
        recovered_evidence_ids=("key-evidence",),
        first_effective_action="query-original-run",
        final_result="completed_pass",
    )
    assessment = evaluate_resume(probe, old_expectation)
    assert 50 - 10 > OLD_EVIDENCE_AGE
    assert assessment.status == ProbeStatus.PASS
    assert assessment.old_evidence_recovered == 1
    assert assessment.old_evidence_recovered_ids == ("key-evidence",)


def test_thirty_one_recent_evidence_ids_do_not_satisfy_old_evidence_target():
    ids = evidence_ids(31)
    recent = expectation(31)
    recent = replace(
        recent,
        evidence_event_indices=tuple((evidence, 100 + index) for index, evidence in enumerate(ids)),
        recovery_event_index=130,
    )
    assessment = evaluate_resume(good_probe(), recent)
    assert assessment.status == ProbeStatus.FAIL
    assert assessment.old_evidence_recovered == 0
    assert any(item.startswith("old_evidence_below_minimum") for item in assessment.failures)


def test_missing_evidence_age_is_inconclusive_not_an_old_evidence_pass():
    unknown_age = replace(
        expectation(1),
        evidence_event_indices=(),
        recovery_event_index=None,
    )
    assessment = evaluate_resume(good_probe(evidence_count=1), unknown_age)
    assert assessment.status == ProbeStatus.INCONCLUSIVE
    assert assessment.old_evidence_recovered == 0
    assert any(item.startswith("evidence_event_age") for item in assessment.missing_observations)
    assert not any("old_evidence_below_minimum" in item for item in assessment.failures)


def test_missing_first_effective_action_is_inconclusive_not_confirmed_failure():
    probe = replace(good_probe(evidence_count=1), first_effective_action=None)
    assessment = evaluate_resume(probe, expectation(1))
    assert assessment.status == ProbeStatus.INCONCLUSIVE
    assert "first_effective_action" in assessment.missing_observations
    assert assessment.failures == ()


def test_resume_probe_must_belong_to_the_new_session_and_known_facts():
    host = MemoryHost(
        fresh_session=FreshSession(
            session_id="session-fresh",
            known_facts_digest="facts-sha",
            materials=("PROGRESS.md", "experiment-ledger.jsonl", "run-manifest.json"),
        ),
        probe=good_probe(session_id="other-session"),
    )
    result = run_fresh_context_resume(
        host,
        original_session_id="session-1",
        known_facts=known_facts(),
        expectation=expectation(),
    )
    assert result.status == ProbeStatus.INCONCLUSIVE
    assert "different session" in (result.reason or "")


def test_retry_is_invalid_only_for_matching_conditions_without_new_evidence():
    prior = AttemptRecord(
        proposition="module.py accepted the fix",
        environment="A2/CANN-8.0",
        input_digest="input-1",
        implementation="patch-1",
        evidence_ids=("evidence-1",),
    )
    no_new_evidence = replace(prior)
    decision = evaluate_retry(prior, no_new_evidence)
    assert decision.decision == RetryDecision.INVALID_DUPLICATE
    assert decision.retry_allowed is False
    assert decision.prior_conclusion_applicable is True
    assert decision.requires_revalidation is False

    new_evidence = replace(prior, evidence_ids=("evidence-1", "evidence-2"))
    decision = evaluate_retry(prior, new_evidence)
    assert decision.decision == RetryDecision.VALID_RETRY
    assert decision.retry_allowed is True
    assert decision.new_evidence == ("evidence-2",)

    changed_environment = replace(prior, environment="A3/CANN-8.1")
    decision = evaluate_retry(prior, changed_environment)
    assert decision.decision == RetryDecision.VALID_REVALIDATION
    assert decision.retry_allowed is True
    assert decision.prior_conclusion_applicable is False
    assert decision.requires_revalidation is True
    assert decision.changed_conditions == ("environment",)

    changed_input = replace(prior, input_digest="input-2")
    assert evaluate_retry(prior, changed_input).decision == RetryDecision.VALID_REVALIDATION


def test_compaction_suite_keeps_trigger_and_resume_results_separate():
    host = MemoryHost(
        receipt=NativeCompactionReceipt(
            host="memory-host",
            session_id="session-1",
            event=SemanticEvent.NEGATIVE_RESULT,
            host_receipt_id="receipt-negative",
        ),
        native_probe=good_probe(session_id="session-1", evidence_count=1),
        fresh_session=FreshSession(
            session_id="session-fresh",
            known_facts_digest="facts-sha",
            materials=("PROGRESS.md", "experiment-ledger.jsonl", "run-manifest.json"),
            resumed_from="session-1",
        ),
        probe=good_probe(evidence_count=1),
    )
    suite = run_compaction_suite(
        host,
        native_session_id="session-1",
        known_facts=known_facts(),
        expectation=expectation(1),
        events=[SemanticEvent.NEGATIVE_RESULT],
    )
    assert suite.protocol == COMPACTION_PROTOCOL
    assert suite.native.trigger_status == ProbeStatus.PASS
    assert suite.native.resume_assessment.status == ProbeStatus.PASS
    assert suite.fresh.status == ProbeStatus.PASS
    payload = suite.to_dict()
    assert payload["native"]["mode"] == "native_compaction"
    assert payload["native"]["trigger_status"] == "pass"
    assert payload["native"]["resume_assessment"]["status"] == "pass"
    assert payload["fresh"]["mode"] == "fresh_context_resume"
    assert payload["native"]["receipts"][0]["event"] == "negative_result"


def test_native_compaction_rejects_a_restarted_process_as_a_receipt():
    with pytest.raises(ValueError, match="restarted process"):
        NativeCompactionReceipt(
            host="memory-host",
            session_id="session-1",
            event=SemanticEvent.IN_FLIGHT_RUN,
            host_receipt_id="receipt-1",
            process_restarted=True,
        )
