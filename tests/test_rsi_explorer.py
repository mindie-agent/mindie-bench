from __future__ import annotations

import json
from copy import deepcopy

import pytest

from mindie_bench.runner.rsi_explorer import (
    append_ledger_and_render_progress,
    build_debiased_rsi_context,
    evaluate_rsi_candidate_gates,
    plan_rsi_compile_and_timing,
    should_escalate_to_rsi,
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
    assert "speedup_below_target" in reason


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
        ({"speedup_ratio": 1.01, "min_speedup_ratio": 1.05}, 4, "refuted"),
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
