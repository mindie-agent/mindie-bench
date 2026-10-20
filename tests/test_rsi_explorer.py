from __future__ import annotations

import json

import pytest

from mindie_bench.runner.rsi_explorer import (
    append_ledger_and_render_progress,
    build_debiased_rsi_context,
    evaluate_rsi_candidate_gates,
    plan_rsi_compile_and_timing,
    should_escalate_to_rsi,
)


def test_l3_stall_detects_gate_failures_and_missed_speedup():
    gate1 = {"compile_ok": False, "oracle_passed": True, "measurement_passed": True}
    escalate, reason = should_escalate_to_rsi([gate1, dict(gate1)])
    assert escalate is True
    assert "gate1" in reason

    gate2 = {"compile_ok": True, "oracle_passed": True, "measurement_passed": False}
    escalate, reason = should_escalate_to_rsi([gate2, dict(gate2)])
    assert escalate is True
    assert "gate2" in reason

    slow = {
        "compile_ok": True,
        "oracle_passed": True,
        "measurement_passed": True,
        "speedup_ratio": 1.01,
        "min_speedup_ratio": 1.05,
    }
    escalate, reason = should_escalate_to_rsi([slow, dict(slow)])
    assert escalate is True
    assert "speedup_below_target" in reason


def test_l3_stall_requires_consecutive_evidence_and_resets_on_success():
    stalled = {"measurement_passed": False, "speedup_ratio": 1.0}
    healthy = {
        "compile_ok": True,
        "oracle_passed": True,
        "measurement_passed": True,
        "speedup_ratio": 1.2,
    }
    assert should_escalate_to_rsi([stalled, healthy, dict(stalled)])[0] is False
    assert should_escalate_to_rsi([stalled], max_stalled_attempts=2)[0] is False
    assert should_escalate_to_rsi([stalled, healthy, stalled, stalled])[0] is True
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
            "verdict": "refuted",
            "conditional_avoid": ["avoid duplicate L3 tiling retries"],
        },
        {
            "candidate_id": "old-supported",
            "verdict": "supported",
            "conditional_avoid": ["refuted-only constraints must not mount"],
        },
        {
            "candidate_id": "derived-refuted",
            "verdict": "refuted",
            "mutation_summary": "oversized fused tile",
            "failure_reason": "workspace delta above requirement",
        },
    ]

    context = build_debiased_rsi_context(
        goal="beat baseline attention throughput",
        retrieved_units=retrieved,
        bottleneck_summary={"observations": [{"bottleneck": "host-side bubble"}]},
        ledger_entries=ledger,
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
    assert "avoid duplicate L3 tiling retries" in context["conditional_avoid"]
    assert any("derived-refuted" in rule for rule in context["conditional_avoid"])
    assert not any("refuted-only" in rule for rule in context["conditional_avoid"])
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
    plan = plan_rsi_compile_and_timing(candidates, npu_device=2, max_compile_slots=2)

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

    assert plan_rsi_compile_and_timing(candidates, npu_device=2, max_compile_slots=2) == plan
    other = plan_rsi_compile_and_timing(candidates, npu_device=3, max_compile_slots=2)
    assert other["state_dir"] != plan["state_dir"]


@pytest.mark.parametrize(
    ("failure_kwargs", "gate"),
    [
        ({"compile_ok": False}, 1),
        ({"oracle_passed": False}, 1),
        ({"measurement_passed": False}, 2),
        ({"ablation_retained": False}, 3),
        ({"speedup_ratio": 1.01, "min_speedup_ratio": 1.05}, 4),
    ],
)
def test_each_failed_gate_is_refuted_with_conditional_avoid(failure_kwargs, gate):
    values = {
        "candidate_id": f"candidate-gate-{gate}",
        "mutation_summary": "change one launch-config knob",
        "conditions": {"device": "A2", "shape": "decode"},
        "compile_ok": True,
        "oracle_passed": True,
        "measurement_passed": True,
        "speedup_ratio": 1.2,
    }
    values.update(failure_kwargs)
    entry = evaluate_rsi_candidate_gates(**values)

    assert entry["verdict"] == "refuted"
    assert entry["gate_reached"] == gate
    assert entry["distill_ready"] is False
    assert entry["conditional_avoid"], "refuted candidates must carry a reuse guard"
    assert values["candidate_id"] in entry["conditional_avoid"][0]
    assert entry["run_id"].startswith("rsi-")
    assert entry["conditions"] == values["conditions"]


def test_four_gate_pass_is_supported_and_distill_ready():
    entry = evaluate_rsi_candidate_gates(
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
        candidate_id="refuted-a",
        mutation_summary="oversized tiling",
        conditions={"device": "A2"},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=False,
        speedup_ratio=0.91,
    )
    supported = evaluate_rsi_candidate_gates(
        candidate_id="supported-b",
        mutation_summary="fused epilogue",
        conditions={"device": "A2"},
        compile_ok=True,
        oracle_passed=True,
        measurement_passed=True,
        speedup_ratio=1.31,
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

    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [row["verdict"] for row in rows] == ["refuted", "supported"]
    assert [row["candidate_id"] for row in rows] == ["refuted-a", "supported-b"]

    progress = progress_path.read_text()
    assert "# RSI Progress: campaign-rsi-1" in progress
    assert "## Goal" in progress and "Improve decode attention on A2" in progress
    assert "## Best Supported Candidate" in progress
    assert "supported-b" in progress
    assert "1.31" in progress
    assert "## Active Hypotheses" in progress
    assert "## Refuted Directions" in progress
    assert "avoid refuted-a" in progress
