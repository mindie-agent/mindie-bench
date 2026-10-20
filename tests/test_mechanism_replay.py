"""Mechanism replay M1-01 ~ M6-01 contracts.

Tier 0 covers isolated pre-decision views and structured decisions. The M6
revision test runs the production organizer over a real SQLite GradedRepository.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from mindie_bench.cli import main as cli_main
from mindie_bench.evaluation.knowledge_tiered import distill_selected
from mindie_bench.evaluation.knowledge_tiered_support import Engine, GradedUnit, Store
from mindie_bench.evaluation.mechanism_replay import (
    CASES,
    DECISION_SCHEMA,
    RELATED_BODY_BUDGET_BYTES,
    ReplayFact,
    collect_m6_repository_evidence,
    evaluate_mechanism_replay,
    prepare_learner_prompt_view,
    run_host_mechanism_replay,
    verify_prompt_isolation,
)

M1 = "M1-01"
M2 = "M2-01"
M3 = "M3-01"
M4 = "M4-01"
M5 = "M5-01"
M6 = "M6-01"


def _decision(case_id: str, action: str, hypothesis: str, locators: list[str], **extra):
    return {
        "schema_version": DECISION_SCHEMA,
        "case_id": case_id,
        "action": action,
        "hypothesis_id": hypothesis,
        "evidence_locators": locators,
        **extra,
    }


def _m1_decision() -> dict:
    return _decision(
        M1, "phase0_fusion_triage", "H1_fusion_boundary",
        ["T1:theoretical-expectation.md:124-145", "T3:53"],
        phase0={
            "fusion_sites": [
                {"kind": "producer_output_fusion", "producer": "MLAPO", "consumer": "FlashMLA",
                 "eliminates_gm_roundtrip": True},
                {"kind": "consumer_writeback_fusion", "producer": "FlashMLA", "consumer": "pack",
                 "eliminates_gm_roundtrip": True},
            ],
            "hard_boundaries": ["collective_barrier", "ub_capacity_limit"],
            "critical_path": {"expression": "max(P+C, Q+G)", "on_exposed_path": True},
            "single_kernel_tweaks_deferred": True,
        },
        next_probe={
            "kind": "local_forward_comparison",
            "discriminates": ["H1_fusion_boundary", "H3_collective_barrier"],
            "estimated_hours": 0.2,
            "alternative_rejected": "cross_version_bisect",
            "repeats_prior_negative": False,
        },
    )


def _m2_decision() -> dict:
    return _decision(
        M2, "locate_first_equivalent_boundary", "H4_downstream_order",
        ["T2:737", "T3:20203"],
        first_divergence={
            "boundary": "moe partial sum at logical token",
            "alignment": "logical_token_state",
            "physical_block_id_used": False,
            "checks": ["shape", "dtype", "stride", "layout", "lifetime"],
        },
        downstream_contract={
            "collective_matched": True,
            "order_compared": True,
            "stages": ["allgather", "reduction", "rmsnorm"],
            "module_declared_correct_from_collective_only": False,
        },
        simulation={
            "cycle_bound_kind": "max_cycle_mean",
            "additive_cycle_bound": False,
            "unmeasured_stages": ["epilogue"],
            "missing_durations": None,
            "critical_path": {"expression": "max(P+C, Q+G)"},
            "clock_claim": "burst_only",
            "sustained_claim": False,
        },
    )


def _m3_decision() -> dict:
    return _decision(
        M3, "observer_control_design", "H1_observer_perturbation",
        ["T3:3028", "T3:22797", "T3:22844"],
        observer={
            "uninstrumented_baseline": True,
            "minimal_intervention": True,
            "observer_only_control": True,
            "observation_changes_result": True,
            "candidate_branch_execution_count": 0,
            "claim": "inconclusive",
            "inconclusive_retained": True,
            "dfx_disabled_final_verification": True,
            "draft_claimed_executed": False,
            "dump_pass_called_fix": False,
        },
    )


def _m4_decision() -> dict:
    return _decision(
        M4, "sequence_reproduction", "H1_stateful_history",
        ["T3:23706", "T4:4793", "T1:eplb-shm-0916/FINDINGS.md"],
        reproduction={
            "fresh_process": {"passes": True},
            "historical_process": {"fails": True},
            "only_final_input": False,
            "full_sequence": ["req1", "req2", "req3", "req4"],
            "minimal_sequence": ["req1", "req2", "req3"],
            "minimal_sequence_replayed": True,
            "reset": {
                "verified": True,
                "covers": ["kv_cache", "conv_state", "slot_owner"],
                "treated_as_complete": False,
            },
            "serial_passes": True,
            "concurrent_deadlocks": True,
            "concurrency_preserved": True,
            "claimed_serial_sufficient": False,
            "claimed_fixed_after_fresh_process": False,
        },
    )


def _m5_decision() -> dict:
    return _decision(
        M5, "independent_oracle_and_scoped_acceptance", "H1_shared_defect",
        ["T2:911", "T3:3028", "T3:20178"],
        oracle={
            "independent": True,
            "shared_defect": "quantized_rope",
            "uses_shared_defect_as_proof": False,
            "acceptance_rate": 1.0,
            "acceptance_rate_used_as_correctness": False,
            "exit_zero_used_as_accuracy": False,
            "health_used_as_accuracy": False,
            "candidate_branch_execution_count": 0,
            "claim": "inconclusive",
            "zero_execution_patch_claimed_fix": False,
        },
        scopes={
            "first_token": {"state": "passed"},
            "continued_decode": {"state": "failed"},
            "concurrent": {"state": "not_measured"},
        },
        first_token_generalized=False,
        gate_plan={
            "g0": "run", "g1": "run", "g2": "run", "g3": "run",
            "g1_incremental_build": True, "g2_sustained_abba": True,
            "manual_glue_required": False,
        },
    )


@pytest.mark.tier0
def test_prompt_views_expose_only_predecision_facts_hypotheses_and_t1_t5_locators():
    allowed = {
        "schema_version", "case_id", "pre_decision_facts", "competing_hypotheses",
        "provenance_locators",
    }
    for case_id, case in CASES.items():
        view = prepare_learner_prompt_view(case_id)
        assert set(view) == allowed
        assert case.sources and set(case.sources) <= {"T1", "T2", "T3", "T4", "T5"}
        observation = verify_prompt_isolation(view, case)
        assert observation["isolated"] is True, observation
        surface = json.dumps(view)
        for marker in case.private_judge["leak_markers"]:
            assert marker not in surface
        assert all(fact["provenance"].split(":", 1)[0] in case.sources
                   for fact in view["pre_decision_facts"])


@pytest.mark.tier0
def test_a_public_copy_of_a_private_observation_is_rejected_before_prompting():
    original = CASES[M1]
    leaky = dataclasses.replace(
        original,
        facts=original.facts + (
            ReplayFact("F-leak", "The independent pack disappeared after 21-23us.", "T1:leaked"),
        ),
    )
    with pytest.raises(ValueError, match="leaks private judge material"):
        prepare_learner_prompt_view(leaky)


@pytest.mark.tier0
def test_m1_phase0_fusion_triage_beats_single_kernel_and_expensive_bisect():
    good = evaluate_mechanism_replay(M1, _m1_decision())
    assert good["state"] == "completed", good
    assert good["accepted"] is True
    bad = _m1_decision()
    bad["action"] = "tune_single_kernel_buffer"
    bad["next_probe"] = {**bad["next_probe"], "kind": "cross_version_bisect",
                         "alternative_rejected": None, "estimated_hours": 6}
    result = evaluate_mechanism_replay(M1, bad)
    assert result["state"] == "failed"
    failed_ids = {item["id"] for item in result["criteria"] if item["passed"] is False}
    assert {"action_allowed", "m1_distinguishing_next_probe"} <= failed_ids


@pytest.mark.tier0
def test_m1_repeating_the_prior_negative_experiment_is_not_progress():
    decision = _m1_decision()
    decision["next_probe"]["repeats_prior_negative"] = True
    result = evaluate_mechanism_replay(M1, decision)
    assert result["state"] == "failed"
    assert any(item["id"] == "m1_avoids_repeat" and item["passed"] is False
               for item in result["criteria"])

    bogus = _m1_decision()
    bogus["next_probe"]["discriminates"] = ["BOGUS_1", "BOGUS_2"]
    bogus_res = evaluate_mechanism_replay(M1, bogus)
    assert bogus_res["state"] == "failed"
    assert any(item["id"] == "m1_distinguishing_next_probe" and item["passed"] is False
               for item in bogus_res["criteria"])

    custom = _m1_decision()
    custom["hypothesis_id"] = "H_custom_gm_pack_chain"
    custom["hypothesis_statement"] = (
        "Producer FIXPIPE cast and pack writeback fusion eliminate the exposed GM round-trip."
    )
    custom["phase0"]["critical_path"] = {"expression": "max(Q+G, P+C)", "on_exposed_path": True}
    custom["next_probe"]["discriminates"] = [
        "H_custom_gm_pack_chain",
        "H3_collective_barrier",
    ]
    assert evaluate_mechanism_replay(M1, custom)["state"] == "completed"


@pytest.mark.tier0
def test_m2_requires_logical_alignment_collective_downstream_and_nonadditive_simulation():
    good = evaluate_mechanism_replay(M2, _m2_decision())
    assert good["state"] == "completed", good
    physical = _m2_decision()
    physical["first_divergence"].update(
        {"alignment": "physical_block_id", "physical_block_id_used": True}
    )
    result = evaluate_mechanism_replay(M2, physical)
    assert result["state"] == "failed"
    assert any(item["id"] == "m2_logical_first_divergence" and item["passed"] is False
               for item in result["criteria"])

    fabricated = _m2_decision()
    fabricated["simulation"].update({"missing_durations": 12.5, "additive_cycle_bound": True})
    result = evaluate_mechanism_replay(M2, fabricated)
    assert result["state"] == "failed"
    assert any(item["id"] == "m2_nonadditive_simulation" and item["passed"] is False
               for item in result["criteria"])

    burst = _m2_decision()
    burst["simulation"].update({"clock_claim": "sustained", "sustained_claim": True})
    assert evaluate_mechanism_replay(M2, burst)["state"] == "failed"


@pytest.mark.tier0
def test_m3_observer_controls_reject_dump_pass_as_fix_and_zero_execution_attribution():
    good = evaluate_mechanism_replay(M3, _m3_decision())
    assert good["state"] == "completed", good
    dump_fix = _m3_decision()
    dump_fix["observer"].update(
        {"claim": "fix", "dump_pass_called_fix": True, "inconclusive_retained": False}
    )
    result = evaluate_mechanism_replay(M3, dump_fix)
    assert result["state"] == "failed"
    assert any(item["id"] == "m3_branch_execution_claim" and item["passed"] is False
               for item in result["criteria"])

    zero_claim = _m3_decision()
    zero_claim["observer"].update({"claim": "fix", "candidate_branch_execution_count": 0})
    assert evaluate_mechanism_replay(M3, zero_claim)["state"] == "failed"

    fabricated_fix = _m3_decision()
    fabricated_fix["observer"].update({
        "claim": "fix",
        "candidate_branch_execution_count": 5,
        "dump_pass_called_fix": False,
        "inconclusive_retained": False,
    })
    assert evaluate_mechanism_replay(M3, fabricated_fix)["state"] == "failed"

    omitted_count = _m3_decision()
    omitted_count["observer"].pop("candidate_branch_execution_count")
    assert evaluate_mechanism_replay(M3, omitted_count)["state"] == "completed"


@pytest.mark.tier0
def test_m4_reproduces_history_minimizes_sequence_and_preserves_concurrency():
    good = evaluate_mechanism_replay(M4, _m4_decision())
    assert good["state"] == "completed", good
    only_last = _m4_decision()
    only_last["reproduction"].update(
        {"only_final_input": True, "full_sequence": ["req4"],
         "minimal_sequence": ["req4"], "minimal_sequence_replayed": True}
    )
    assert evaluate_mechanism_replay(M4, only_last)["state"] == "failed"

    serial = _m4_decision()
    serial["reproduction"].update({"claimed_serial_sufficient": True,
                                   "concurrency_preserved": False})
    result = evaluate_mechanism_replay(M4, serial)
    assert result["state"] == "failed"
    assert any(item["id"] == "m4_concurrency_boundary" and item["passed"] is False
               for item in result["criteria"])

    fresh_fix = _m4_decision()
    fresh_fix["reproduction"]["claimed_fixed_after_fresh_process"] = True
    assert evaluate_mechanism_replay(M4, fresh_fix)["state"] == "failed"


@pytest.mark.tier0
def test_m5_rejects_shared_defect_acceptance_health_and_first_token_generalization():
    good = evaluate_mechanism_replay(M5, _m5_decision())
    assert good["state"] == "completed", good
    acceptance = _m5_decision()
    acceptance["oracle"].update(
        {"acceptance_rate_used_as_correctness": True, "exit_zero_used_as_accuracy": True}
    )
    result = evaluate_mechanism_replay(M5, acceptance)
    assert result["state"] == "failed"
    assert any(item["id"] == "m5_independent_oracle" and item["passed"] is False
               for item in result["criteria"])
    assert any(item["id"] == "m5_actual_path" and item["passed"] is False
               for item in result["criteria"])

    zero = _m5_decision()
    zero["oracle"].update({"claim": "fix", "zero_execution_patch_claimed_fix": True})
    assert evaluate_mechanism_replay(M5, zero)["state"] == "failed"

    fabricated_m5 = _m5_decision()
    fabricated_m5["oracle"].update({
        "claim": "fix",
        "candidate_branch_execution_count": 5,
        "zero_execution_patch_claimed_fix": False,
    })
    assert evaluate_mechanism_replay(M5, fabricated_m5)["state"] == "failed"

    generalized = _m5_decision()
    generalized["first_token_generalized"] = True
    generalized["scopes"]["concurrent"]["state"] = "passed"
    assert evaluate_mechanism_replay(M5, generalized)["state"] == "failed"

    manual = _m5_decision()
    manual["gate_plan"]["manual_glue_required"] = True
    assert evaluate_mechanism_replay(M5, manual)["state"] == "failed"


# ------------------------------------------------------------------- M6 real repo
def _u1_body(repeats: int = 74) -> str:
    branch = (
        "Zeroer remains excluded for the 910C decode branch when prefix_cache=off. "
        "The rejected PV2, Score2, S256 and vmula attempts are folded into the compact "
        "exclusion table with their measured conditions.\n"
    )
    return (branch * repeats) + (
        "Human correction: do not change the scheduling contract without technical evidence. "
        "The old baseline means the immediately previous run. 4:1 and parent/subslot are "
        "superseded layouts and must remain inactive recommendations.\n"
    )


def _u1_unit(*, repeats: int = 74) -> GradedUnit:
    body = _u1_body(repeats)
    metadata = {
        "schema": "mindie-graded-unit/1",
        "id": "unit-history-conditioned-zeroer-exclusion",
        "revision": 1,
        "tier": "L3",
        "category": "workflow",
        "evidence_strength": "single_observation",
        "domain_path": "diagnostics/state",
        "title": "History-conditioned exclusion and correction records",
        "inherits_from": [],
        "conditions": {
            "soc": "Ascend910C",
            "workload_scope": "decode",
            "prefix_cache": "off",
        },
        "triggers": {"error_signatures": [], "keywords": ["zeroer", "prefix", "layout"]},
        "provenance": {
            "source_type": "session_distill",
            "source_locator": "fixture@abcdef1:active-state.md",
            "source_digest": hashlib.sha256(body.encode()).hexdigest(),
            "evidence_refs": ["t4-11236", "t4-16041", "t4-17326"],
            "supersedes": [],
        },
    }
    return GradedUnit.from_parts(metadata, body, f"{metadata['domain_path']}/{metadata['id']}.md")


def _u2_body() -> str:
    return (
        "Active layout and retained constraints: the 910C decode branch keeps prefix_cache=off "
        "as the primary path. The excluded zeroer hypothesis remains excluded; do not reopen "
        "it under unchanged conditions.\n\n"
        "Verified exclusion table: PV2, Score2, S256 and vmula were rejected by paired "
        "measurements. 4:1 and parent/subslot are superseded and must not be revived.\n\n"
        "Corrections: a human instruction is not independent technical proof. The old "
        "baseline means the immediately previous run. A prefix-cache-enabled branch is "
        "recorded only as a scoped, untested branch.\n"
    )


def _distill_proposal(unit: GradedUnit, *, body: str, session: str, evidence_content: str):
    metadata = {key: value for key, value in unit.metadata.items() if key != "revision"}
    metadata = deepcopy(metadata)
    metadata["provenance"]["source_digest"] = hashlib.sha256(evidence_content.encode()).hexdigest()
    return {
        "schema_version": "mindie-distill-decision/1",
        "session_digest": session,
        "evidence_inputs": ["authorized_transcript"],
        "action": "propose_graded_units",
        "drop_reason": None,
        "units": [
            {
                "target_unit_id": unit.id,
                "target_tier": metadata["tier"],
                "category": metadata["category"],
                "evidence_strength": metadata["evidence_strength"],
                "domain_path": metadata["domain_path"],
                "operation": "merge_into_existing",
                "frontmatter": metadata,
                "markdown_body": body,
            }
        ],
    }


def _m6_decision(unit: GradedUnit, before_body: str, after_body: str) -> dict:
    return _decision(
        M6, "revise_existing_unit", "H1_revise_compact",
        ["T4:11236", "T4:24224", "T1:ACTIVE_STATE.md"],
        revision={
            "target_unit_id": unit.id,
            "previous_revision": 1,
            "operation": "merge_into_existing",
            "new_unit_created": False,
            "active_unit_count_delta": 0,
            "related_unit_snapshot": {
                "body_complete": True,
                "body_excerpt_bytes": len(before_body.encode()),
                "body_sha256": hashlib.sha256(before_body.encode()).hexdigest(),
            },
            "append_only": False,
            "rewrite_to_compact": True,
            "before_body_bytes": len(before_body.encode()),
            "after_body_bytes": len(after_body.encode()),
            "net_body_bytes": len(after_body.encode()) - len(before_body.encode()),
            "max_body_bytes": 8192,
            "body_sha256": hashlib.sha256(after_body.encode()).hexdigest(),
            "old_conditions": dict(unit.metadata["conditions"]),
            "new_branch_query": "prefix cache enabled branch scoped untested",
            "old_query": "zeroer excluded 4:1 parent subslot correction",
            "superseded_layouts": ["4:1", "parent/subslot"],
            "revived_layouts": [],
            "excluded_hypotheses": ["zeroer"],
            "reopened_hypotheses": [],
            "reopen_conditions_changed": False,
        },
        evidence={
            "counter_evidence_retained": [
                {"claim": "zeroer excluded", "source": "T4:11236"},
                {"claim": "4:1 superseded", "source": "T4:17326"},
                {"claim": "parent/subslot superseded", "source": "T4:19868"},
            ],
            "human_corrections": ["T4:16041", "T4:24062"],
            "technical_proofs": ["T4:17326", "T4:19868"],
            "separated": True,
        },
    )


@pytest.mark.tier1
def test_m6_incomplete_related_body_cannot_merge_past_the_16k_boundary(tmp_path):
    gitleaks = shutil.which("gitleaks") or next(
        (str(path) for path in Path.home().glob(".cache/mindie-agent/gitleaks/*/*/gitleaks")),
        None,
    )
    if not gitleaks:
        pytest.skip("installed gitleaks is required for the production organizer")
    store = Store(tmp_path / "store", "vllm-ascend")
    unit = _u1_unit(repeats=85)
    assert len(unit.body.encode()) > RELATED_BODY_BUDGET_BYTES
    store.graded.install([unit])
    before = store.graded.explain(store.graded.ref(unit.id))["content"]
    evidence_content = (
        "Verified milestone: the large prior body is not fully visible; a merge must be refused "
        "rather than reconstructing unseen counter-evidence."
    )
    session = "f" * 64
    proposal = _distill_proposal(
        unit, body=_u2_body(), session=session, evidence_content=evidence_content
    )
    engine = Engine(store, redactor_executable=gitleaks)
    with pytest.raises(ValueError, match="complete related unit"):
        engine.distill_selected(
            [{
                "kind": "authorized_transcript",
                "outcome": "verified",
                "source_locator": unit.metadata["provenance"]["source_locator"],
                "content": evidence_content,
                "evidence_refs": list(unit.metadata["provenance"]["evidence_refs"]),
                "repetitions": 1,
            }],
            session_digest=session,
            task_state="verified_milestone",
            related_refs=[unit.id],
            proposal=proposal,
        )
    after = store.graded.explain(store.graded.ref(unit.id))["content"]
    assert after == before
    decision = _m6_decision(unit, before, _u2_body())
    decision["revision"]["related_unit_snapshot"]["body_complete"] = False
    result = evaluate_mechanism_replay(M6, decision)
    assert result["state"] == "failed"
    assert any(item["id"] == "m6_body_complete_budget" and item["passed"] is False
               for item in result["criteria"])


@pytest.mark.tier1
def test_m6_real_graded_repository_u1_to_u2_retains_conditions_and_anti_bloat(tmp_path):
    gitleaks = shutil.which("gitleaks") or next(
        (str(path) for path in Path.home().glob(".cache/mindie-agent/gitleaks/*/*/gitleaks")),
        None,
    )
    if not gitleaks:
        pytest.skip("installed gitleaks is required for the production organizer")
    store = Store(tmp_path / "store", "vllm-ascend")
    unit = _u1_unit()
    assert len(unit.body.encode()) <= RELATED_BODY_BUDGET_BYTES
    store.graded.install([unit])
    before_view = store.graded.explain(store.graded.ref(unit.id))
    before_body = before_view["content"]
    before_active = len(store.graded._load().units)
    after_body = _u2_body()
    evidence_content = (
        "Verified milestone: zeroer stays excluded under the same conditions; 4:1 and "
        "parent/subslot remain superseded while the old metric baseline definition is retained."
    )
    session = "d" * 64
    proposal = _distill_proposal(
        unit, body=after_body, session=session, evidence_content=evidence_content
    )
    applied = distill_selected(
        store=store,
        inputs=[{
            "kind": "authorized_transcript",
            "outcome": "verified",
            "source_locator": unit.metadata["provenance"]["source_locator"],
            "content": evidence_content,
            "evidence_refs": list(unit.metadata["provenance"]["evidence_refs"]),
            "repetitions": 1,
        }],
        session_digest=session,
        task_state="verified_milestone",
        proposal=proposal,
        redactor_executable=gitleaks,
    )
    assert applied["status"] == "applied", applied
    assert applied["admitted_unit_ids"] == [unit.id]
    current = store.graded.explain(store.graded.ref(unit.id))
    assert current["revision"] == 2
    assert current["content"] == after_body
    assert current["frontmatter"]["conditions"] == unit.metadata["conditions"]
    assert len(current["content"].encode()) < len(before_body.encode())
    assert len(store.graded._load().units) == before_active == 1
    evidence = collect_m6_repository_evidence(
        store.graded,
        unit_id=unit.id,
        old_conditions=unit.metadata["conditions"],
        new_query="prefix cache enabled branch scoped untested",
        old_query="zeroer excluded 4:1 parent subslot correction",
        before_body=before_body,
        before_active_units=before_active,
    )
    assert evidence["old_ref_status"] == "removed_or_superseded"
    assert evidence["new_branch_recalled"] is True
    assert evidence["old_conditions_recalled"] is True
    decision = _m6_decision(unit, before_body, after_body)
    result = evaluate_mechanism_replay(
        M6, decision, repository=store.graded,
        revision_context={"before_body": before_body, "before_active_units": before_active,
                          "before_conditions": unit.metadata["conditions"]},
    )
    assert result["state"] == "completed", result

    # Missing repository evidence is unknown, not an invented pass.
    unknown = evaluate_mechanism_replay(M6, decision)
    assert unknown["state"] == "unknown"
    assert any(item["id"] == "m6_repository_revision" and item["passed"] is None
               for item in unknown["criteria"])

    # A revived superseded layout and a mismatched body hash must both fail.
    revived = deepcopy(decision)
    revived["revision"]["revived_layouts"] = ["4:1"]
    assert evaluate_mechanism_replay(
        M6, revived, repository=store.graded,
        revision_context={"before_body": before_body, "before_active_units": before_active,
                          "before_conditions": unit.metadata["conditions"]},
    )["state"] == "failed"
    wrong_hash = deepcopy(decision)
    wrong_hash["revision"]["body_sha256"] = "0" * 64
    assert evaluate_mechanism_replay(
        M6, wrong_hash, repository=store.graded,
        revision_context={"before_body": before_body, "before_active_units": before_active,
                          "before_conditions": unit.metadata["conditions"]},
    )["state"] == "failed"

    # Omitting copy-pasted byte/hash fields in decision still passes via repository evidence.
    minimal_decision = deepcopy(decision)
    for key in ("before_body_bytes", "after_body_bytes", "net_body_bytes", "body_sha256"):
        minimal_decision["revision"].pop(key, None)
    minimal_decision["revision"]["related_unit_snapshot"].pop("body_excerpt_bytes", None)
    minimal_decision["revision"]["related_unit_snapshot"].pop("body_sha256", None)
    assert evaluate_mechanism_replay(
        M6, minimal_decision, repository=store.graded,
        revision_context={"before_body": before_body, "before_active_units": before_active,
                          "before_conditions": unit.metadata["conditions"]},
    )["state"] == "completed"


@pytest.mark.tier0
def test_host_runner_and_cli_dispatch_mechanism_replay(tmp_path):
    seen_prompts = []

    def fake_host_runner(prompt_view):
        seen_prompts.append(prompt_view)
        return {
            "decision": _m1_decision(),
            "execution_metrics": {
                "wall_clock_ms": 14.5,
                "prompt_tokens": 310,
                "completion_tokens": 120,
                "tool_calls": ["read_file"],
                "commands_executed": ["pytest -q"],
            },
        }

    host_report = run_host_mechanism_replay(M1, runner=fake_host_runner)
    assert host_report["state"] == "completed"
    assert len(seen_prompts) == 1 and seen_prompts[0]["case_id"] == M1
    assert host_report["execution_metrics"]["prompt_tokens"] == 310
    assert host_report["execution_metrics"]["tool_calls"] == ["read_file"]

    prompt_file = tmp_path / "m3_prompt.json"
    assert cli_main(["mechanism-replay", "--case", M3, "--prompt-only", "--output", str(prompt_file)]) == 0
    prompt_json = json.loads(prompt_file.read_text(encoding="utf-8"))
    assert prompt_json["case_id"] == M3
    assert "Dump changes stream timing" not in json.dumps(prompt_json)

    decision_file = tmp_path / "m3_decision.json"
    decision_file.write_text(json.dumps(_m3_decision()), encoding="utf-8")
    out_file = tmp_path / "m3_eval.json"
    assert cli_main([
        "mechanism-replay", "--case", M3, "--decision", str(decision_file), "--output", str(out_file)
    ]) == 0
    eval_json = json.loads(out_file.read_text(encoding="utf-8"))
    assert eval_json["state"] == "completed"
    assert "execution_metrics" in eval_json
