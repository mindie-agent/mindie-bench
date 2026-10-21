"""Business contracts for the optional component-value scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_bench.benchmarks import value_diagnostics, value_scenarios


def test_optional_scenarios_join_defaults_only_when_importable(monkeypatch):
    assert {"profiler", "sim"} <= set(value_scenarios.scenario_choices())
    monkeypatch.setattr(value_scenarios, "_optional_scenarios", lambda: [])
    assert value_scenarios.default_scenarios() == ["coordinator", "npu-top", "diagnostics"]
    monkeypatch.setattr(value_scenarios, "_optional_scenarios", lambda: ["profiler", "sim"])
    assert value_scenarios.default_scenarios() == [
        "coordinator", "npu-top", "diagnostics", "profiler", "sim",
    ]


def test_profiler_scenario_separates_slow_rank_from_slow_link_and_preserves_db(tmp_path):
    pytest.importorskip("mindie_profiler")
    from mindie_bench.benchmarks import value_profiler

    result = value_profiler.run(tmp_path, 1)
    row = result["measurements"][0]
    assert row["status"] == "pass"
    assert row["candidate"]["slow_link_dominant_block"] == "Comm"
    assert row["candidate"]["slow_rank_dominant_block"] == "Attention"
    assert row["candidate"]["residual_bytes"] == value_profiler.RESIDUAL_BYTES
    assert row["candidate"]["residual_status"] == "within_target"
    assert row["candidate"]["memory_target_met"] is True
    assert row["candidate"]["db_mutation"] is False
    raw = json.loads(Path(row["raw"]).read_text())
    assert raw["db_sha256_before"] == raw["db_sha256_after"]
    assert raw["comparison"]["comparable"] is True
    assert raw["raw_baseline"]["link"]


def test_sim_scenario_records_accuracy_and_rejects_overflowing_fusion(tmp_path):
    pytest.importorskip("mindie_sim.a3_physical_calibration")
    pytest.importorskip("mindie_sim.operator_estimation")
    from mindie_bench.benchmarks import value_sim

    result = value_sim.run(tmp_path, 1)
    row = result["measurements"][0]
    assert row["status"] == "pass"
    candidate = row["candidate"]
    assert candidate["qwen_case_deterministic"] is True
    assert candidate["a3_graph_ttft_ms"] > 0
    assert candidate["a3_graph_tpot_ms"] > 0
    assert abs(candidate["tpot_error_pct"]) < 20
    assert candidate["operator_plan_backends"] == [
        "gemm_finite_pipeline", "vector_instruction",
    ]
    assert candidate["subgraph_overflow_rejected"] is True
    raw = json.loads(Path(row["raw"]).read_text())
    assert raw["accuracy"]["ttft_absolute_error_ms"] < 0
    assert raw["subgraph_screening"]["overflow"]["feasibility"]["failures"] == [
        "live_buffer_limit",
    ]


def test_diagnostics_triage_is_deterministic_and_redacts_credentials(tmp_path):
    pytest.importorskip("mindie_diagnostics.npu_fault_triage")
    evidence = value_diagnostics.fault_triage_evidence(tmp_path)
    assert evidence["secrets_removed"] is True
    assert evidence["privacy_findings"] > 0
    assert evidence["deterministic"] is True
    assert evidence["root_cause_rank"] == "3"
    assert evidence["root_cause_domain"] == "aicore_memory_or_tiling"
    assert evidence["cascading_timeout_detected"] is True
    assert evidence["cascading_rank"] == "7"
    redacted = Path(evidence["redacted"]).read_text()
    assert "ghp_valuefixture" not in redacted
    assert "10.18.4.22" not in redacted
    assert "/home/alice" not in redacted
