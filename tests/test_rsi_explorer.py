from __future__ import annotations

import json
import uuid
from copy import deepcopy

import pytest

from mindie_bench.runner.rsi_explorer import (
    append_ledger_and_render_progress,
    build_debiased_rsi_context,
    evaluate_exploration_axes,
    evaluate_hidden_input_generalization,
    evaluate_rsi_candidate_gates,
    execute_rsi_compile_and_timing,
    plan_rsi_compile_and_timing,
    run_rsi_gate_campaign,
    should_escalate_to_rsi,
    verify_sequential_ablation,
)


def test_debiasing_keeps_foundation_from_l3_and_removes_nested_l3_body():
    # RoutingIndex returns complete parents inside inherited_rules, and may
    # suppress a standalone foundation row when a workflow already includes it.
    parent = {"id": "unit-foundation", "revision": 1, "tier": "L1",
              "content": "Preserve the input dtype.", "conditions": {"soc": "A2"}}
    workflow = {"id": "unit-workflow", "revision": 1, "tier": "L3",
                "content": "Always retry the old tiling.", "inherited_rules": [parent]}
    mutation = {"id": "unit-mutation", "revision": 1, "tier": "L4",
                "excerpt": "Change the reduction schedule.", "inherited_rules": [workflow, parent]}
    retrieved = [workflow, mutation]
    original = deepcopy(retrieved)
    context = build_debiased_rsi_context(
        goal="Improve reduction", retrieved_units=retrieved,
        bottleneck_summary={}, ledger_entries=[], conditions={"soc": "A2"},
    )
    assert [row["id"] for row in context["l1_foundation"]["units"]] == [parent["id"]]
    assert "Always retry the old tiling" not in json.dumps(context)
    assert context["stripped_l3_ids"] == [workflow["id"]]
    assert retrieved == original


def test_invalid_measurement_does_not_refute_a_candidate_or_trigger_exploration():
    attempt = evaluate_rsi_candidate_gates(
        run_id="remote-run-12", candidate_id="tiling-a", mutation_summary="Use a wider tile",
        conditions={"soc": "A2", "shape": [32, 4096]}, compile_ok=True, oracle_passed=True,
        measurement_passed=False, speedup_ratio=None,
    )
    assert attempt["verdict"] == "inconclusive"
    assert attempt["conditional_avoid"] == []
    assert attempt["gates"]["gate3_ablation_gate"]["passed"] is None
    assert not should_escalate_to_rsi([attempt, attempt])[0]


def test_negative_experience_is_reused_only_under_the_measured_conditions():
    attempt = evaluate_rsi_candidate_gates(
        run_id="remote-run-13", candidate_id="tiling-a", mutation_summary="Use a wider tile",
        conditions={"soc": "A2", "shape": [32, 4096]}, compile_ok=True, oracle_passed=True,
        measurement_passed=True, speedup_ratio=0.95, ablation_retained=True,
    )
    def context(conditions):
        return build_debiased_rsi_context(goal="Improve reduction", retrieved_units=[],
                                         bottleneck_summary={}, ledger_entries=[attempt],
                                         conditions=conditions)
    same = context({"soc": "A2", "shape": [32, 4096], "dtype": "bf16"})
    assert same["conditional_avoid"][0]["conditions"] == attempt["conditions"]
    assert same["conditional_avoid"][0]["run_id"] == "remote-run-13"
    assert context({"soc": "A3", "shape": [32, 4096]})["conditional_avoid"] == []
    assert context({"soc": "A2", "shape": [1024, 4096]})["conditional_avoid"] == []
    assert context({"soc": "A2"})["conditional_avoid"] == []
    corrected = {**attempt, "verdict": "supported", "outcome": "verified"}
    revised = build_debiased_rsi_context(
        goal="Improve reduction", retrieved_units=[], bottleneck_summary={},
        ledger_entries=[attempt, corrected], conditions=attempt["conditions"],
    )
    assert revised["conditional_avoid"] == []


def test_l3_stall_requires_a_valid_experiment_before_escalating():
    values = {"run_id": "remote-l3", "candidate_id": "l3-trial", "mutation_summary": "change tile",
              "conditions": {"soc": "A2"}, "compile_ok": True, "oracle_passed": True,
              "measurement_passed": True, "speedup_ratio": 1.01, "ablation_retained": True}
    gate1 = evaluate_rsi_candidate_gates(**{**values, "compile_ok": False})
    escalate, reason = should_escalate_to_rsi([gate1, dict(gate1)])
    assert escalate is False
    assert "execution_incomplete" in reason

    gate2 = evaluate_rsi_candidate_gates(**{**values, "measurement_passed": False})
    escalate, reason = should_escalate_to_rsi([gate2, dict(gate2)])
    assert escalate is False
    assert "execution_incomplete" in reason

    slow = evaluate_rsi_candidate_gates(**values)
    assert should_escalate_to_rsi([slow, dict(slow)])[0] is False
    escalate, reason = should_escalate_to_rsi([slow, {**slow, "run_id": "remote-l3-next"}])
    assert escalate is True
    assert "speedup_within_noise" in reason


def test_l3_stall_requires_consecutive_evidence_and_resets_on_success():
    values = {"run_id": "remote-l3", "candidate_id": "l3-trial", "mutation_summary": "change tile",
              "conditions": {"soc": "A2"}, "compile_ok": True, "oracle_passed": True,
              "measurement_passed": True, "speedup_ratio": 1.0, "ablation_retained": True}
    stalled = evaluate_rsi_candidate_gates(**values)
    healthy = evaluate_rsi_candidate_gates(**{**values, "speedup_ratio": 1.2, "run_id": "remote-l3-healthy"})
    third, fourth = {**stalled, "run_id": "remote-l3-third"}, {**stalled, "run_id": "remote-l3-fourth"}
    assert should_escalate_to_rsi([stalled, healthy, third])[0] is False
    assert should_escalate_to_rsi([stalled], max_stalled_attempts=2)[0] is False
    assert should_escalate_to_rsi([stalled, healthy, third, fourth])[0] is True
    assert should_escalate_to_rsi([healthy, healthy])[0] is False
    assert should_escalate_to_rsi([], max_stalled_attempts=2)[0] is False


def test_debiased_context_strips_l3_and_mounts_l1_l2_l4_and_refuted_avoid():
    retrieved = [
        {
            "tier": "L1",
            "unit_id": "l1-foundation",
            "axioms": ["preserve-dtype"],
            "measurement_rules": ["synchronize-before-timing"],
            "inherited_rules": ["never-edit-sealed-oracle"],
            "foundation_baseline": {"model": "baseline-a", "device": 0},
        },
        {
            "tier": "L2",
            "unit_id": "l2-bottleneck",
            "observations": [{"bottleneck": "attention-memory-bound"}],
        },
        {
            "tier": "L3",
            "unit_id": "l3-workflow-bias",
            "workflow": "retry-the-same-heuristic-tiling",
            "inherited_rules": ["l3-bias-that-must-not-mount"],
        },
        {"tier": "L4", "unit_id": "l4-mutator", "operators": ["fuse-producer-consumer"]},
    ]
    ledger = [
        {
            "candidate_id": "old-refuted",
            "run_id": "old-run",
            "verdict": "refuted",
            "conditions": {"soc": "A2"},
            "mutation_summary": "duplicate L3 tiling",
            "failure_reason": "slower than baseline",
        },
        {
            "candidate_id": "old-supported",
            "run_id": "supported-run",
            "verdict": "supported",
            "conditional_avoid": ["refuted-only constraints must not mount"],
        },
        {
            "candidate_id": "derived-refuted",
            "run_id": "derived-run",
            "verdict": "refuted",
            "conditions": {"soc": "A2"},
            "mutation_summary": "oversized fused tile",
            "failure_reason": "workspace delta above requirement",
        },
    ]

    context = build_debiased_rsi_context(
        goal="beat baseline attention throughput",
        retrieved_units=retrieved,
        bottleneck_summary={"observations": [{"bottleneck": "host-side bubble"}]},
        ledger_entries=ledger,
        conditions={"soc": "A2"},
    )

    assert context["stripped_l3_ids"] == ["l3-workflow-bias"]
    assert context["stripped_tiers"] == ["L3"]
    assert context["mounted_tiers"] == ["L1", "L2", "L4"]
    assert context["foundation_axioms"] == ["preserve-dtype"]
    assert context["measurement_rules"] == ["synchronize-before-timing"]
    assert context["inherited_rules"] == ["never-edit-sealed-oracle"]
    assert context["foundation_baseline"] == {"model": "baseline-a", "device": 0}
    assert context["l1_foundation"]["inherited_rules"] == ["never-edit-sealed-oracle"]
    assert context["l1_foundation"]["foundation_baseline"] == {
        "model": "baseline-a",
        "device": 0,
    }
    assert context["l2_bottleneck"]["units"][0]["unit_id"] == "l2-bottleneck"
    assert context["bottleneck_observations"] == [
        {"bottleneck": "host-side bubble"},
        {"bottleneck": "attention-memory-bound"},
    ]
    assert context["mutator_operators"] == ["fuse-producer-consumer"]
    assert context["l4_mutators"]["units"][0]["unit_id"] == "l4-mutator"
    assert [rule["candidate_id"] for rule in context["conditional_avoid"]] == ["old-refuted", "derived-refuted"]
    assert all(rule["conditions"] == {"soc": "A2"} for rule in context["conditional_avoid"])
    serialized = json.dumps(context)
    assert "retry-the-same-heuristic-tiling" not in serialized
    assert "l3-bias-that-must-not-mount" not in serialized


def test_plan_compiles_on_cpu_then_times_only_proved_candidates_serially():
    candidates = [
        {"candidate_id": "proved-a", "compile_ok": True},
        {"candidate_id": "proved-b", "compile_proof": {"status": "passed"}},
        {"candidate_id": "failed-compile", "compile_ok": False},
        {"candidate_id": "missing-proof"},
    ]
    plan = plan_rsi_compile_and_timing(candidates, state_dir="/tmp/rsi-campaign-a", npu_device=2, max_compile_slots=2)

    compile_phase, timing_phase = plan["phases"]
    assert compile_phase is plan["phase1_compile"]
    assert compile_phase["parallel"] is True
    assert compile_phase["resources"] == {"npu_count": 0}
    assert compile_phase["max_parallel"] == 2
    assert compile_phase["progress_owner"] == "CoordinatorService"
    assert compile_phase["progress_owner_detail"]["role"] == "single_progress_owner"
    assert compile_phase["state_dir"] == plan["state_dir"]
    assert compile_phase["state_dir"] == compile_phase["progress_owner_detail"]["state_dir"]
    assert compile_phase["waves"] == [["proved-a", "proved-b"], ["failed-compile", "missing-proof"]]
    assert all(len(wave) <= 2 for wave in compile_phase["waves"])

    assert timing_phase["parallel"] is False
    assert timing_phase["resources"] == {"devices": [2]}
    assert timing_phase["candidates"] == ["proved-a", "proved-b"]
    assert all(item["parallel"] is False for item in timing_phase["items"])
    assert timing_phase["skipped_candidates"] == [
        {"candidate_id": "failed-compile", "reason": "compile_proof_not_succeeded"},
        {"candidate_id": "missing-proof", "reason": "compile_proof_missing"},
    ]

    assert plan["state_dir"] == "/tmp/rsi-campaign-a"
    other = plan_rsi_compile_and_timing(candidates, state_dir="/tmp/rsi-campaign-b", npu_device=2, max_compile_slots=2)
    assert other["state_dir"] != plan["state_dir"]


@pytest.mark.parametrize(
    ("failure_kwargs", "gate", "verdict"),
    [
        ({"compile_ok": False}, 1, "inconclusive"),
        ({"oracle_passed": False}, 1, "refuted"),
        ({"measurement_passed": False}, 2, "inconclusive"),
        ({"ablation_retained": None}, 3, "inconclusive"),
        ({"ablation_retained": False}, 3, "refuted"),
        ({"speedup_ratio": 0.95, "min_speedup_ratio": 1.05}, 4, "refuted"),
    ],
)
def test_gate_failure_keeps_execution_and_hypothesis_outcomes_separate(failure_kwargs, gate, verdict):
    values = {
        "candidate_id": f"candidate-gate-{gate}",
        "run_id": f"remote-gate-{gate}",
        "mutation_summary": "change one launch-config knob",
        "conditions": {"device": "A2", "shape": "decode"},
        "compile_ok": True,
        "oracle_passed": True,
        "measurement_passed": True,
        "speedup_ratio": 1.2,
        "ablation_retained": True,
    }
    values.update(failure_kwargs)
    entry = evaluate_rsi_candidate_gates(**values)

    assert entry["verdict"] == verdict
    assert entry["gate_reached"] == gate
    assert entry["distill_ready"] is False
    if verdict == "refuted":
        assert entry["conditional_avoid"][0]["candidate_id"] == values["candidate_id"]
    else:
        assert entry["conditional_avoid"] == []
    assert entry["run_id"] == values["run_id"]
    assert all(item["passed"] is None for item in list(entry["gates"].values())[gate:])
    assert entry["conditions"] == values["conditions"]

    # A valid gain above noise but below target is reusable evidence, not a
    # refutation and not a conditional avoidance rule.
    subtarget = evaluate_rsi_candidate_gates(
        run_id="remote-gate-subtarget", candidate_id="candidate-subtarget",
        mutation_summary="change one launch-config knob",
        conditions={"device": "A2", "shape": "decode"}, compile_ok=True,
        oracle_passed=True, measurement_passed=True, speedup_ratio=1.04,
        min_speedup_ratio=1.05, ablation_retained=True,
    )
    assert subtarget["verdict"] == "subtarget_valid"
    assert subtarget["distill_ready"] is True
    assert subtarget["conditional_avoid"] == []


def test_four_gate_pass_is_supported_and_distill_ready():
    entry = evaluate_rsi_candidate_gates(
        run_id="remote-winner",
        candidate_id="winner",
        mutation_summary="fuse attention epilogue",
        conditions={"device": "A2", "shape": "decode", "dtype": "fp16"},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=True,
        speedup_ratio=1.23,
        min_speedup_ratio=1.05,
        ablation_retained=True,
    )

    assert entry["verdict"] == "supported"
    assert entry["gate_reached"] == 4
    assert entry["distill_ready"] is True
    assert entry["conditional_avoid"] == []
    assert entry["speedup_ratio"] == pytest.approx(1.23)
    assert all(gate["passed"] for gate in entry["gates"].values())


def test_ledger_and_progress_persist_supported_and_refuted_directions(tmp_path):
    refuted = evaluate_rsi_candidate_gates(
        run_id="remote-refuted",
        candidate_id="refuted-a",
        mutation_summary="oversized tiling",
        conditions={"device": "A2"},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=True,
        speedup_ratio=0.91,
        ablation_retained=True,
    )
    supported = evaluate_rsi_candidate_gates(
        run_id="remote-supported",
        candidate_id="supported-b",
        mutation_summary="fused epilogue",
        conditions={"device": "A2"},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=True,
        speedup_ratio=1.31,
        ablation_retained=True,
    )
    incomplete = evaluate_rsi_candidate_gates(
        run_id="remote-incomplete", candidate_id="noisy-timing", mutation_summary="wider tile",
        conditions={"device": "A2"}, compile_ok=True, oracle_passed=True,
        measurement_passed=False, speedup_ratio=None,
    )
    ledger_path = tmp_path / "experiment-ledger.jsonl"
    progress_path = tmp_path / "PROGRESS.md"

    append_ledger_and_render_progress(
        ledger_path,
        progress_path,
        refuted,
        campaign_id="campaign-rsi-1",
        goal="Improve decode attention on A2",
    )
    append_ledger_and_render_progress(
        ledger_path,
        progress_path,
        supported,
        campaign_id="campaign-rsi-1",
        goal="Improve decode attention on A2",
    )
    append_ledger_and_render_progress(
        ledger_path, progress_path, incomplete,
        campaign_id="campaign-rsi-1", goal="Improve decode attention on A2",
    )

    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [row["verdict"] for row in rows] == ["refuted", "supported", "inconclusive"]
    assert [row["candidate_id"] for row in rows] == ["refuted-a", "supported-b", "noisy-timing"]

    progress = progress_path.read_text()
    assert "# RSI Progress: campaign-rsi-1" in progress
    assert "## Goal" in progress and "Improve decode attention on A2" in progress
    assert "## Best Supported Candidate" in progress
    assert "supported-b" in progress
    assert "1.31" in progress
    assert "## Active Hypotheses" in progress
    assert "## Refuted Directions" in progress
    assert "avoid refuted-a" in progress
    assert "avoid noisy-timing" not in progress
    assert "## Experiments To Complete" in progress
    assert 'when {"device": "A2"}' in progress
    completed = {**supported, "run_id": incomplete["run_id"], "candidate_id": incomplete["candidate_id"]}
    append_ledger_and_render_progress(
        ledger_path, progress_path, completed,
        campaign_id="campaign-rsi-1", goal="Improve decode attention on A2",
    )
    assert len(ledger_path.read_text().splitlines()) == 4
    assert progress_path.read_text().split("## Experiments To Complete")[1].strip() == "- none"


def test_information_progress_resets_stall_and_never_escalates_as_a_stall():
    stalled = evaluate_rsi_candidate_gates(
        run_id="remote-stalled", candidate_id="tiling-a", mutation_summary="retry wider tile",
        conditions={"soc": "A2"}, compile_ok=True, oracle_passed=True,
        measurement_passed=True, speedup_ratio=0.99, ablation_retained=True,
    )
    observed = {**stalled, "run_id": "remote-observed", "verdict": "refuted",
                "new_observation": True, "failure_reason": "same failure"}
    divergent = {**stalled, "run_id": "remote-divergent", "verdict": "refuted",
                 "divergent_layer_identified": True, "failure_reason": "same failure"}
    escalate, reason = should_escalate_to_rsi([stalled, observed])
    assert escalate is False
    assert "l3_information_progress" in reason
    assert should_escalate_to_rsi([divergent, divergent])[0] is False
    assert should_escalate_to_rsi([stalled, observed, stalled])[0] is False
    assert should_escalate_to_rsi(
        [stalled, {**stalled, "run_id": "remote-stalled-2"}]
    )[0] is True


def test_subtarget_speedup_keeps_valid_evidence_and_never_becomes_conditional_avoid():
    entry = evaluate_rsi_candidate_gates(
        run_id="remote-subtarget", candidate_id="tiling-b", mutation_summary="narrower tile",
        conditions={"soc": "A2", "shape": [32, 4096]}, compile_ok=True, oracle_passed=True,
        measurement_passed=True, speedup_ratio=1.04, min_speedup_ratio=1.05,
        ablation_retained=True, preserve_subtarget_evidence=True, return_tier="L3",
    )
    assert entry["verdict"] == "subtarget_valid"
    assert entry["outcome"] == "speedup_below_target_valid_evidence"
    assert entry["distill_ready"] is True
    assert entry["conditional_avoid"] == []
    assert entry["return_tier"] == "L3"
    assert entry["axes"] == {
        "evidence_validity": "valid",
        "goal_attainment": "not_attained",
        "reusability": "reusable_observation",
    }
    context = build_debiased_rsi_context(
        goal="Improve reduction", retrieved_units=[], bottleneck_summary={},
        ledger_entries=[entry], conditions=entry["conditions"],
    )
    assert context["conditional_avoid"] == []

    noisy = evaluate_rsi_candidate_gates(
        run_id="remote-noise", candidate_id="tiling-c", mutation_summary="no-op launch change",
        conditions={"soc": "A2"}, compile_ok=True, oracle_passed=True,
        measurement_passed=True, speedup_ratio=1.005, min_speedup_ratio=1.05,
        ablation_retained=True, preserve_subtarget_evidence=True,
    )
    assert noisy["verdict"] == "inconclusive"
    assert noisy["outcome"] == "speedup_within_noise"
    assert noisy["distill_ready"] is False
    assert noisy["conditional_avoid"] == []


def test_backward_compatible_below_target_still_refutes_without_opt_in():
    values = {
        "run_id": "remote-legacy", "candidate_id": "tiling-legacy",
        "mutation_summary": "wider tile", "conditions": {"soc": "A2"},
        "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
        "speedup_ratio": 1.04, "min_speedup_ratio": 1.05, "ablation_retained": True,
    }
    entry = evaluate_rsi_candidate_gates(**values)
    assert entry["verdict"] == "subtarget_valid"
    assert entry["outcome"] == "speedup_below_target_valid_evidence"
    assert entry["axes"]["evidence_validity"] == "valid"
    assert entry["axes"]["goal_attainment"] == "not_attained"
    assert entry["axes"]["reusability"] == "reusable_observation"
    assert entry["distill_ready"] is True
    assert entry["conditional_avoid"] == []

    legacy = evaluate_rsi_candidate_gates(**values, preserve_subtarget_evidence=False)
    assert legacy["verdict"] == "refuted"
    assert legacy["outcome"] == "speedup_below_target"
    assert legacy["axes"]["reusability"] == "reusable_negative_boundary"
    assert legacy["conditional_avoid"][0]["candidate_id"] == "tiling-legacy"


@pytest.mark.parametrize(
    ("kwargs", "evidence", "goal", "reuse", "verdict"),
    [
        ({"compile_ok": False}, "compile_failed", "not_applicable", "inconclusive", "inconclusive"),
        ({"oracle_passed": False}, "oracle_failed", "not_applicable",
         "reusable_negative_boundary", "refuted"),
        ({"measurement_passed": False}, "measurement_invalid", "not_applicable",
         "inconclusive", "inconclusive"),
        ({"ablation_retained": None}, "valid", "not_applicable", "inconclusive", "inconclusive"),
        ({"speedup_ratio": 1.3, "return_tier": "L1"}, "valid", "attained",
         "reusable_observation", "supported"),
        ({"speedup_ratio": 1.3, "return_tier": "L4"}, "valid", "attained",
         "reusable_mutator", "supported"),
    ],
)
def test_every_outcome_carries_three_independent_axes(kwargs, evidence, goal, reuse, verdict):
    values = {"compile_ok": True, "oracle_passed": True, "measurement_passed": True,
              "speedup_ratio": 1.2, "ablation_retained": True, "return_tier": "L4"}
    values.update(kwargs)
    result = evaluate_exploration_axes(**values)
    assert result["verdict"] == verdict
    assert result["axes"] == {"evidence_validity": evidence, "goal_attainment": goal,
                              "reusability": reuse}
    assert result["evidence_validity"] == evidence
    assert result["return_tier"] == values["return_tier"]


def test_timing_uses_the_assigned_device_lease_instead_of_npu_zero():
    candidates = [{"candidate_id": "proved", "compile_ok": True}]
    plan = plan_rsi_compile_and_timing(
        candidates, state_dir="/tmp/rsi-lease", npu_device=0,
        assigned_devices=[3, 4],
    )
    timing = plan["timing_wave"]
    assert plan["npu_device"] == 3
    assert plan["assigned_devices"] == [3, 4]
    assert timing["device"] == 3
    assert timing["resources"] == {"devices": [3, 4]}
    assert timing["items"][0]["resources"] == {"devices": [3, 4]}
    fallback = plan_rsi_compile_and_timing(
        candidates, state_dir="/tmp/rsi-lease", npu_device=2, assigned_devices=[],
    )
    assert fallback["assigned_devices"] == [2]
    assert fallback["timing_wave"]["resources"] == {"devices": [2]}


def test_isolation_mode_declares_whether_a_native_subcontext_really_exists():
    def context(native_subcontext=False):
        return build_debiased_rsi_context(
            goal="Explore a new mechanism", retrieved_units=[], bottleneck_summary={},
            ledger_entries=[], conditions={}, native_subcontext=native_subcontext,
        )
    assert context()["isolation_mode"] == "in_session_strategy_shift"
    assert context(native_subcontext=True)["isolation_mode"] == "native_subcontext"


def test_progress_keeps_subtarget_evidence_under_active_hypotheses(tmp_path):
    subtarget = evaluate_rsi_candidate_gates(
        run_id="remote-subtarget-progress", candidate_id="valid-small-win",
        mutation_summary="one launch-config knob", conditions={"device": "A2"},
        compile_ok=True, oracle_passed=True, measurement_passed=True,
        speedup_ratio=1.04, min_speedup_ratio=1.05, ablation_retained=True,
        preserve_subtarget_evidence=True,
    )
    append_ledger_and_render_progress(
        tmp_path / "experiment-ledger.jsonl", tmp_path / "PROGRESS.md", subtarget,
        campaign_id="campaign-subtarget", goal="Keep valid evidence below target",
    )
    progress = (tmp_path / "PROGRESS.md").read_text()
    assert "## Active Hypotheses" in progress
    assert "valid-small-win" in progress
    assert "## Refuted Directions" in progress
    assert "avoid valid-small-win" not in progress


def test_subtarget_evidence_does_not_require_a_finished_ablation_to_be_valid():
    entry = evaluate_rsi_candidate_gates(
        run_id="remote-subtarget-pending-ablation", candidate_id="small-win",
        mutation_summary="narrower tile", conditions={"soc": "A2"},
        compile_ok=True, oracle_passed=True, measurement_passed=True,
        speedup_ratio=1.04, min_speedup_ratio=1.05,
        preserve_subtarget_evidence=True,
    )
    assert entry["verdict"] == "subtarget_valid"
    assert entry["gate_reached"] == 4
    assert entry["gates"]["gate3_ablation_gate"]["passed"] is None
    assert entry["distill_ready"] is True


def test_non_speedup_goals_use_goal_met_and_diagnostic_gain_instead_of_ratio():
    values = {"compile_ok": True, "oracle_passed": True, "measurement_passed": True,
              "speedup_ratio": None, "goal_type": "correctness"}
    attained = evaluate_exploration_axes(**values, goal_met=True)
    assert attained["verdict"] == "supported"
    assert attained["axes"] == {"evidence_validity": "valid", "goal_attainment": "attained",
                                "reusability": "reusable_observation"}
    assert attained["distill_ready"] is True

    diagnostic = evaluate_exploration_axes(
        **{**values, "goal_type": "diagnose"}, goal_met=False, diagnostic_gain=True)
    assert diagnostic["verdict"] == "subtarget_valid"
    assert diagnostic["outcome"] == "diagnostic_gain_valid_evidence"
    assert diagnostic["axes"]["reusability"] == "reusable_observation"
    assert diagnostic["distill_ready"] is True

    refuted = evaluate_exploration_axes(**values, goal_met=False)
    assert refuted["verdict"] == "refuted"
    assert refuted["outcome"] == "goal_not_met"
    assert refuted["axes"]["reusability"] == "reusable_negative_boundary"

    unverified = evaluate_exploration_axes(**values, goal_met=None)
    assert unverified["verdict"] == "inconclusive"
    assert unverified["outcome"] == "goal_unverified"
    assert unverified["axes"]["goal_attainment"] == "not_applicable"
    assert unverified["distill_ready"] is False



class _ContractLease:
    """Test-only lease whose identity is still externally validated."""

    def __init__(self, execution_id, devices):
        self.execution_id = execution_id
        self.devices = list(devices)

    def receipt(self):
        return {
            "execution_id": self.execution_id,
            "devices": list(self.devices),
            "state": "running",
            "authority": "test-contract-authority",
        }

    def verify(self):
        return True


class _ContractAuthority:
    mode = "test_contract_authority"
    evidence_class = "test_contract_authority"

    def __init__(self):
        self.issued = []

    def acquire(self, devices, *, phase, candidate, timeout_seconds):
        execution_id = uuid.uuid4().hex + uuid.uuid4().hex
        self.issued.append(execution_id)
        return _ContractLease(execution_id, devices)

    def validate_result(self, lease, result, *, phase, devices):
        if result.get("run_id") != lease.execution_id:
            raise ValueError("result does not preserve the admitted execution ID")

    def release(self, lease):
        return lease.receipt()



def test_execute_rsi_compile_and_timing_runs_cpu_waves_then_serial_npu_timing(tmp_path):
    compile_calls: list[str] = []
    timing_calls: list[tuple[str, int, list[int]]] = []
    active = {"current": 0, "maximum": 0}

    def compile_runner(candidate):
        compile_calls.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "compile-fail":
            return {"compile_ok": False, "failure_reason": "bisheng compile error"}
        if candidate["candidate_id"] == "missing-proof":
            return {}
        return {
            "compile_ok": True,
            "compile_proof": {"status": "passed"},
            "run_id": f"compile-{candidate['candidate_id']}",
            "oracle_passed": True,
        }

    def timing_runner(candidate):
        active["current"] += 1
        active["maximum"] = max(active["maximum"], active["current"])
        timing_calls.append(
            (candidate["candidate_id"], candidate["npu_device"], candidate["assigned_devices"])
        )
        active["current"] -= 1
        return {
            "run_id": candidate["coordinator_lease"].execution_id,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.3 if candidate["candidate_id"] == "winner" else 1.08,
            "ablation_retained": True,
            "workspace_delta_bytes": 2048,
            "guard_workloads": [
                {"name": "uniform-guard", "regression_ratio": 1.01, "max_regression_ratio": 1.05}
            ],
        }

    candidates = [
        {"candidate_id": "pre-proved", "compile_ok": True, "mutation_summary": "ready"},
        {"candidate_id": "winner", "mutation_summary": "fuse epilogue"},
        {"candidate_id": "compile-fail", "mutation_summary": "invalid tile"},
        {"candidate_id": "missing-proof", "mutation_summary": "not compiled"},
    ]
    result = execute_rsi_compile_and_timing(
        candidates,
        compile_runner=compile_runner,
        timing_runner=timing_runner,
        lease_authority=_ContractAuthority(),
        state_dir=tmp_path / "rsi-executor",
        npu_device=3,
        max_compile_slots=2,
        ledger_path=tmp_path / "experiment-ledger.jsonl",
        progress_path=tmp_path / "PROGRESS.md",
        goal="Raise decode throughput",
    )

    # Waves are planned in input order, but a pre-proved candidate is not compiled again.
    assert compile_calls == ["winner", "compile-fail", "missing-proof"]
    assert result["compile_order"] == compile_calls
    assert result["timing_order"] == ["pre-proved", "winner"]
    assert timing_calls == [("pre-proved", 3, [3]), ("winner", 3, [3])]
    assert active["maximum"] == 1
    assert result["plan"]["phase2_timing"]["candidates"] == ["pre-proved", "winner"]
    assert [row["candidate_id"] for row in result["evaluations"]] == [
        "pre-proved", "winner", "compile-fail", "missing-proof"
    ]
    assert [row["verdict"] for row in result["evaluations"]] == [
        "supported", "supported", "inconclusive", "inconclusive"
    ]
    assert [row["gate_reached"] for row in result["evaluations"]] == [4, 4, 1, 1]
    assert result["best_candidate"]["candidate_id"] == "winner"
    assert result["best_candidate"]["speedup_ratio"] == pytest.approx(1.3)

    ledger = [
        json.loads(line)
        for line in (tmp_path / "experiment-ledger.jsonl").read_text().splitlines()
    ]
    assert [row["candidate_id"] for row in ledger] == [
        "pre-proved", "winner", "compile-fail", "missing-proof"
    ]
    progress = (tmp_path / "PROGRESS.md").read_text()
    assert "# RSI Progress: rsi-executor" in progress
    assert "Raise decode throughput" in progress
    assert "winner" in progress
    assert "## Experiments To Complete" in progress
    assert "compile_failed" in progress
    assert "compile-fail" in progress and "missing-proof" in progress


def test_execute_rsi_compile_and_timing_fails_closed_on_workspace_and_guard(tmp_path):
    def timing_runner(candidate):
        lease_id = candidate["coordinator_lease"].execution_id
        if candidate["candidate_id"] == "overflow":
            return {
                "run_id": lease_id, "oracle_passed": True,
                "measurement_passed": True, "speedup_ratio": 1.2,
                "ablation_retained": True,
                "workspace_delta_bytes": {"0": 1024, "1": 8192},
            }
        if candidate["candidate_id"] == "guard-regression":
            return {
                "run_id": lease_id, "oracle_passed": True,
                "measurement_passed": True, "speedup_ratio": 1.2,
                "ablation_retained": True, "workspace_delta_bytes": 1024,
                "guard_workloads": [
                    {"name": "uniform-guard", "regression_ratio": 1.2,
                     "max_regression_ratio": 1.05}
                ],
            }
        return {
            "run_id": lease_id, "oracle_passed": True,
            "measurement_passed": True, "speedup_ratio": 1.2,
            "ablation_retained": True, "workspace_delta_bytes": 1024,
            "guard_workloads": [
                {"name": "uniform-guard", "baseline_ms": 100.0, "candidate_ms": 101.0,
                 "max_regression_ratio": 1.05}
            ],
        }

    result = execute_rsi_compile_and_timing(
        [
            {"candidate_id": "overflow", "compile_ok": True},
            {"candidate_id": "guard-regression", "compile_ok": True},
            {"candidate_id": "clean", "compile_ok": True},
        ],
        timing_runner=timing_runner,
        lease_authority=_ContractAuthority(),
        state_dir=tmp_path / "rsi-guards",
        max_workspace_delta_bytes=4096,
    )
    entries = {row["candidate_id"]: row for row in result["evaluations"]}

    assert entries["overflow"]["verdict"] == "inconclusive"
    assert entries["overflow"]["gate_reached"] == 2
    assert entries["overflow"]["failure_reason"] == "workspace_delta_above_limit"
    assert entries["overflow"]["gates"]["gate2_measurement_guard"] == {
        "passed": False, "reasons": ["workspace_delta_above_limit"]
    }
    assert entries["guard-regression"]["verdict"] == "inconclusive"
    assert entries["guard-regression"]["gate_reached"] == 2
    assert "guard_workload_regressed:uniform-guard" in entries["guard-regression"]["failure_reason"]
    assert entries["clean"]["verdict"] == "supported"
    assert result["best_candidate"]["candidate_id"] == "clean"


def test_hidden_input_generalization_classifies_all_five_categories_cleanly():
    cases = [
        {"case_id": "dynamic-tail", "bucket": "dynamic-shape-257", "shape": [2, 257],
         "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": 100.0, "opt_latency_us": 50.0},
        {"case_id": "high-dynamic-range", "bucket": "hdr-fp16",
         "ref_ok": True, "opt_ok": True, "max_abs_err": 1e-4, "max_rel_err": 1e-4,
         "ref_latency_us": 1000.0, "opt_latency_us": 250.0},
        {"case_id": "nonfinite-latency", "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": float("nan"), "opt_latency_us": 10.0},
        {"case_id": "regressed-tail", "bucket": "tail-17", "ref_ok": True, "opt_ok": False,
         "parity_ok": True, "ref_latency_us": 100.0, "opt_latency_us": 100.0},
        {"case_id": "both-fail", "ref_ok": False, "opt_ok": False, "parity_ok": True},
        {"case_id": "fixed-baseline", "ref_ok": False, "opt_ok": True, "parity_ok": True},
        {"case_id": "timeout", "status": "timeout"},
        {"case_id": "nan-parity", "ref_ok": True, "opt_ok": True,
         "max_abs_err": float("inf")},
    ]
    summary = evaluate_hidden_input_generalization(cases)

    assert summary["schema_version"] == "mindie-bench-l4-generalization/1"
    assert summary["category_counts"] == {
        "both_pass": 3, "opt_regression": 2, "both_fail": 1,
        "opt_improvement": 1, "evaluation_error": 1,
    }
    assert summary["total_cases"] == len(cases)
    assert summary["generalization_ok"] is False
    assert summary["failed_case_ids"] == [
        "regressed-tail", "both-fail", "fixed-baseline", "timeout", "nan-parity",
    ]
    # Only the two finite both_pass ratios contribute: sqrt(2 * 4), never a
    # regression or failing case.
    assert summary["both_pass_speedup"] == pytest.approx(8 ** 0.5)
    classified = {row["case_id"]: row for row in summary["cases"]}
    assert classified["dynamic-tail"]["bucket"] == "dynamic-shape-257"
    assert classified["dynamic-tail"]["shape"] == [2, 257]
    assert classified["dynamic-tail"]["speedup_ratio"] == pytest.approx(2.0)
    assert classified["high-dynamic-range"]["category"] == "both_pass"
    assert classified["nonfinite-latency"]["category"] == "both_pass"
    assert classified["nonfinite-latency"]["speedup_ratio"] is None
    assert classified["nan-parity"]["category"] == "opt_regression"
    assert classified["nan-parity"]["parity_ok"] is False

    clean = evaluate_hidden_input_generalization([
        {"case_id": "pass", "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": 90.0, "opt_latency_us": 45.0},
        {"case_id": "improved", "ref_ok": False, "opt_ok": True, "parity_ok": True},
    ])
    assert clean["generalization_ok"] is True
    assert clean["both_pass_speedup"] == pytest.approx(2.0)
    assert clean["failed_case_ids"] == ["improved"]
    assert evaluate_hidden_input_generalization([
        {"case_id": "both-fail", "ref_ok": False, "opt_ok": False, "parity_ok": True},
    ])["generalization_ok"] is False


def test_sequential_ablation_contract_c4_retains_unverified_candidates():
    kwargs = {
        "baseline_run_id": "baseline",
        "candidate_run_id": "candidate",
        "baseline_latency_us": 1000.0,
        "combined_latency_us": 500.0,
        "factor_trials": [
            {"factor_name": "tiling", "run_id": "loo-tiling",
             "compile_ok": True, "correctness_ok": True, "latency_us": 800.0},
            {"factor_name": "launch", "run_id": "loo-launch",
             "compile_ok": True, "correctness_ok": True, "latency_us": 502.0},
            {"factor_name": "required", "run_id": "loo-required",
             "compile_ok": False, "correctness_ok": True, "latency_us": 500.0},
            {"factor_name": "oracle", "run_id": "loo-oracle",
             "compile_ok": True, "correctness_ok": False, "latency_us": 500.0},
        ],
    }
    retained = verify_sequential_ablation(**kwargs)

    assert retained["schema_version"] == "mindie-bench-ablation/1"
    assert retained["pruned_combination_verified"] is False
    assert retained["ablation_retained"] is True
    assert retained["pruned_factors"] == []
    assert retained["retained_factors"] == ["tiling", "launch", "required", "oracle"]
    decisions = {row["factor_name"]: row for row in retained["factor_decisions"]}
    assert decisions["tiling"]["status"] == "essential"
    assert decisions["launch"]["status"] == "candidate_for_sequential_pruning"
    assert all(row["decision"] == "RETAIN" for row in decisions.values())
    assert decisions["launch"]["contribution_us"] == pytest.approx(2.0)
    assert retained["net_speedup"] == pytest.approx(2.0)

    missing_identity = verify_sequential_ablation(
        **kwargs,
        pruned_combination_run={"run_id": "", "verified": True},
    )
    assert missing_identity["pruned_combination_verified"] is False
    assert missing_identity["ablation_retained"] is True
    assert all(row["decision"] == "RETAIN" for row in missing_identity["factor_decisions"])

    implied = verify_sequential_ablation(
        **kwargs,
        pruned_combination_run={
            "run_id": "pruned-implied", "compile_ok": True,
            "correctness_ok": True, "latency_us": 502.0,
        },
    )
    assert implied["pruned_combination_verified"] is True
    assert implied["verification_run_id"] == "pruned-implied"
    assert implied["pruned_factors"] == ["launch"]
    assert implied["retained_factors"] == ["tiling", "required", "oracle"]

    verified = verify_sequential_ablation(
        **kwargs,
        pruned_combination_run={
            "run_id": "pruned-verified", "verified": True,
            "compile_ok": True, "correctness_ok": True, "latency_us": 505.0,
        },
    )
    assert verified["pruned_combination_verified"] is True
    assert verified["ablation_retained"] is False
    verified_decisions = {row["factor_name"]: row for row in verified["factor_decisions"]}
    assert verified_decisions["launch"]["decision"] == "PRUNE"
    assert verified_decisions["tiling"]["decision"] == "RETAIN"


def test_candidate_gates_consume_generalization_and_ablation_receipts():
    values = {
        "run_id": "remote-evidence", "candidate_id": "evidence",
        "mutation_summary": "fuse epilogue", "conditions": {"soc": "A2"},
        "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
        "speedup_ratio": 1.3, "ablation_retained": True,
    }
    generalized = evaluate_rsi_candidate_gates(
        **values, hidden_input_summary={"generalization_ok": True},
    )
    assert generalized["verdict"] == "supported"
    assert generalized["generalization_ok"] is True

    regressed = evaluate_rsi_candidate_gates(
        **values, hidden_input_summary={"generalization_ok": False},
    )
    assert regressed["verdict"] == "refuted"
    assert regressed["gate_reached"] == 1
    assert regressed["gates"]["gate1_compile_oracle"]["passed"] is False
    assert "hidden_input_generalization_failed" in regressed["gates"]["gate1_compile_oracle"]["reasons"]
    assert regressed["failure_reason"] == "hidden_input_generalization_failed"

    unverified = evaluate_rsi_candidate_gates(
        **values, ablation_receipt={"ablation_retained": False},
    )
    assert unverified["verdict"] == "refuted"
    assert unverified["gate_reached"] == 3
    assert unverified["gates"]["gate3_ablation_gate"]["passed"] is False
    assert "ablation_receipt_not_retained" in unverified["gates"]["gate3_ablation_gate"]["reasons"]
    assert unverified["ablation_receipt"] == {"ablation_retained": False}

    retained = evaluate_rsi_candidate_gates(
        **values, ablation_receipt={"pruned_combination_verified": False},
    )
    assert retained["verdict"] == "supported"
    assert retained["gates"]["gate3_ablation_gate"]["passed"] is True


def test_run_rsi_gate_campaign_closes_loop_and_isolates_accounting_errors(tmp_path):
    stalled = evaluate_rsi_candidate_gates(
        run_id="stalled-1", candidate_id="old-tiling", mutation_summary="retry tile",
        conditions={"soc": "A2"}, compile_ok=True, oracle_passed=True,
        measurement_passed=True, speedup_ratio=1.0, ablation_retained=True,
    )
    ledger_entries = [stalled, {**stalled, "run_id": "stalled-2"}]
    units = [
        {"tier": "L1", "unit_id": "l1", "axioms": ["preserve-dtype"],
         "measurement_rules": ["synchronize-before-timing"]},
        {"tier": "L2", "unit_id": "l2", "observations": ["vector-bound"]},
        {"tier": "L3", "unit_id": "l3-bias", "workflow": "retry-the-stale-tiling"},
        {"tier": "L4", "unit_id": "l4", "operators": ["fuse-producer-consumer"]},
    ]
    hidden_cases = [
        {"case_id": "dynamic-tail", "ref_ok": True, "opt_ok": True, "parity_ok": True,
         "ref_latency_us": 100.0, "opt_latency_us": 50.0},
    ]
    ablation_spec = {
        "baseline_run_id": "baseline", "candidate_run_id": "candidate",
        "baseline_latency_us": 1000.0, "combined_latency_us": 500.0,
        "factor_trials": [
            {"factor_name": "tiling", "run_id": "loo-tiling",
             "compile_ok": True, "correctness_ok": True, "latency_us": 800.0},
            {"factor_name": "launch", "run_id": "loo-launch",
             "compile_ok": True, "correctness_ok": True, "latency_us": 502.0},
        ],
    }

    class GateStub:
        gate_specs = {"G0": "g0", "G1": "g1", "G2": "g2", "G3": "g3"}

        def __init__(self):
            self.calls = []

        def run_gate(self, gate, spec):
            self.calls.append(gate)
            return {"gate": gate, "state": "completed", "stages": [], "reasons": []}

    gate_runner = GateStub()

    def runner(candidate):
        if candidate.get("phase") == "compile":
            return {"compile_ok": True, "compile_proof": {"status": "passed"},
                    "run_id": "compile-c1", "oracle_passed": True}
        return {"run_id": candidate["coordinator_lease"].execution_id,
                "oracle_passed": True, "measurement_passed": True,
                "speedup_ratio": 1.4, "ablation_retained": True}

    feedback_entries = []

    def broken_distill(entry):
        raise RuntimeError("distill accounting unavailable")

    def collect_feedback(entry):
        feedback_entries.append(entry)
        return {"accepted": True}

    campaign = run_rsi_gate_campaign(
        goal="Improve decode attention",
        units=units,
        ledger_entries=ledger_entries,
        bottleneck_summary={"observations": ["host-side bubble"]},
        conditions={"soc": "A2"},
        consecutive_limit=2,
        command_plan={"candidates": [{"candidate_id": "c1", "mutation_summary": "fuse epilogue"}]},
        runner=runner,
        lease_authority=_ContractAuthority(),
        gate_runner=gate_runner,
        hidden_input_cases=hidden_cases,
        ablation_spec=ablation_spec,
        ledger_path=tmp_path / "experiment-ledger.jsonl",
        progress_path=tmp_path / "PROGRESS.md",
        distill_callback=broken_distill,
        feedback_callback=collect_feedback,
    )

    assert campaign["should_escalate"] is True
    assert campaign["escalation"]["should_escalate"] is True
    assert campaign["escalation"]["consecutive_limit"] == 2
    assert campaign["context"]["stripped_l3_ids"] == ["l3-bias"]
    assert "retry-the-stale-tiling" not in json.dumps(campaign["context"])
    pack = campaign["context"]["subagent_fact_pack"]
    assert pack["schema_version"] == "mindie-rsi-subagent-pack/1"
    assert [row["run_id"] for row in pack["failed_attempts"]] == ["stalled-1", "stalled-2"]
    assert pack["foundation"]["axioms"] == ["preserve-dtype"]

    assert campaign["execution"]["compile_order"] == ["c1"]
    assert campaign["execution"]["timing_order"] == ["c1"]
    assert gate_runner.calls == ["G0", "G1", "G2", "G3"]
    assert campaign["qualification_ok"] is True
    assert all(result["state"] == "completed" for result in campaign["qualification_gates"].values())

    assert campaign["verdict"] == "supported"
    assert campaign["campaign_passed"] is True
    assert campaign["evaluation"]["verdict"] == "supported"
    assert campaign["evaluation"]["execution_evidence"]["production_adapter"] is False
    assert campaign["evaluation"]["execution_evidence"]["physical_acceptance"] is False
    assert campaign["hidden_input_summary"]["generalization_ok"] is True
    assert campaign["ablation_receipt"]["ablation_retained"] is True
    assert campaign["ablation_receipt"]["pruned_combination_verified"] is False

    assert campaign["ledger_appended"] is True
    ledger = [json.loads(line) for line in (tmp_path / "experiment-ledger.jsonl").read_text().splitlines()]
    assert [row["candidate_id"] for row in ledger] == ["c1"]
    assert ledger[0]["verdict"] == "supported"
    assert "c1" in (tmp_path / "PROGRESS.md").read_text()

    # Principle 14: accounting callbacks are isolated from the business verdict
    # and the gate receipts. Feedback still runs after distillation fails.
    assert campaign["accounting_warnings"] == [
        "distill_callback:RuntimeError:distill accounting unavailable"
    ]
    assert campaign["feedback_result"] == {"accepted": True}
    assert feedback_entries == [campaign["evaluation"]]
    assert campaign["verdict"] == "supported"
    assert campaign["gates"]["gate4_distill_or_refute"]["passed"] is True


def test_apple_rsi_and_geak_safeguards_category_gate_jit_isolation_speculative_and_harvest():
    from mindie_bench.runner.rsi_explorer import (
        build_isolated_jit_cache_env,
        classify_candidate_patch_shape,
        evaluate_speculative_acceptance,
        harvest_candidate_patch,
        run_rsi_gate_campaign,
    )

    # 1. Apple category_gate.py: pure_flag_flip, config_default_flip, artifact_only, source_edit
    flag_diff = "diff --git a/launch_server.sh b/launch_server.sh\n+++ b/launch_server.sh\n+--chunked-prefill-size 8192\n"
    flag_res = classify_candidate_patch_shape(flag_diff, reject_narrow_flips=True)
    assert flag_res["category"] == "pure_flag_flip"
    assert flag_res["rejected_by_narrow_gate"] is True

    cfg_diff = "diff --git a/model_config.py b/model_config.py\n+++ b/model_config.py\n+MAX_NUM_SEQS = 256\n"
    cfg_res = classify_candidate_patch_shape(cfg_diff, reject_narrow_flips=True)
    assert cfg_res["category"] == "config_default_flip"
    assert cfg_res["rejected_by_narrow_gate"] is True

    src_diff = (
        "diff --git a/kernels/rmsnorm_triton.py b/kernels/rmsnorm_triton.py\n"
        "+++ b/kernels/rmsnorm_triton.py\n"
        "+def fused_rmsnorm(x, w):\n"
        "+    return tl.rsqrt(x)\n"
    )
    src_res = classify_candidate_patch_shape(src_diff, reject_narrow_flips=True)
    assert src_res["category"] == "source_edit"
    assert src_res["rejected_by_narrow_gate"] is False

    # 2. Apple _paired_helper.py: isolated JIT cache env per side
    prist_env = build_isolated_jit_cache_env("/tmp/rsi-state", "pristine")
    cand_env = build_isolated_jit_cache_env("/tmp/rsi-state", "candidate-1")
    assert prist_env["TRITON_CACHE_DIR"] != cand_env["TRITON_CACHE_DIR"]
    assert prist_env["ASCEND_CACHE_PATH"] != cand_env["ASCEND_CACHE_PATH"]

    # 3. Apple acceptance.py: speculative decoding draft count & acceptance length guard
    no_drafts = evaluate_speculative_acceptance(
        {"num_drafts": 0, "mean_acceptance_length": 0.0},
        {"num_drafts": 64, "mean_acceptance_length": 3.2},
    )
    assert no_drafts["acceptance_ok"] is False and no_drafts["reason"] == "no_drafts"

    degraded = evaluate_speculative_acceptance(
        {"num_drafts": 64, "mean_acceptance_length": 2.4},
        {"num_drafts": 64, "mean_acceptance_length": 3.2},
        rtol=0.05,
    )
    assert degraded["acceptance_ok"] is False and degraded["reason"] == "acceptance_degraded"

    healthy = evaluate_speculative_acceptance(
        {"num_drafts": 64, "mean_acceptance_length": 3.15},
        {"num_drafts": 64, "mean_acceptance_length": 3.2},
        rtol=0.05,
    )
    assert healthy["acceptance_ok"] is True

    # 4. GEAK kernel_lane.js: measurement-anchored patch harvesting rescues timed-out disk patches
    rescued = harvest_candidate_patch({"timed_out": True}, patch_text=src_diff, baseline_floor=1.0)
    assert rescued["should_verify"] is True and rescued["harvest_mode"] == "disk_patch_rescued"

    skipped = harvest_candidate_patch(
        {"measurement_valid": True, "speedup_ratio": 0.94}, patch_text=src_diff, baseline_floor=1.0
    )
    assert skipped["should_verify"] is False and skipped["harvest_mode"] == "skip_verified_below_floor"

    # 5. Campaign integration: narrow flag flip rejected at G3; incumbent_speedup prevents false winner
    rejected_campaign = run_rsi_gate_campaign(
        candidate={
            "candidate_id": "flag-gamer",
            "compile_ok": True,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.35,
            "diff_text": flag_diff,
        },
        reject_narrow_flips=True,
    )
    assert rejected_campaign["verdict"] == "refuted"
    assert rejected_campaign["patch_classification"]["category"] == "pure_flag_flip"
    assert "narrow_patch_rejected:pure_flag_flip" in rejected_campaign["evaluation"]["failure_reason"]

    below_incumbent = run_rsi_gate_campaign(
        candidate={
            "candidate_id": "c-sub-incumbent",
            "compile_ok": True,
            "oracle_passed": True,
            "measurement_passed": True,
            "speedup_ratio": 1.5,
            "ablation_retained": True,
            "diff_text": src_diff,
        },
        incumbent_speedup=2.5,
        reject_narrow_flips=True,
    )
    assert below_incumbent["verdict"] == "supported"
    assert below_incumbent["incumbent_comparison"]["improved_over_incumbent"] is False
    assert below_incumbent["incumbent_comparison"]["became_winner"] is False
    assert below_incumbent["reprofile_required"] is False
