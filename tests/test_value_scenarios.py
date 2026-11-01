"""Business contracts for the optional component-value scenarios."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mindie_bench.benchmarks import value_diagnostics, value_precision, value_scenarios


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


def test_precision_localization_catches_five_real_bugs_without_false_positives(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mindie_diagnostics.precision")
    first = value_precision.precision_localization_evidence(tmp_path)
    first_bytes = Path(first["evidence_path"]).read_bytes()
    second = value_precision.precision_localization_evidence(tmp_path)
    second_bytes = Path(second["evidence_path"]).read_bytes()
    assert first_bytes == second_bytes
    assert first["evidence_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert second["evidence_sha256"] == hashlib.sha256(second_bytes).hexdigest()
    assert first["status"] == second["status"] == "pass"
    assert first["deterministic"] is second["deterministic"] is True
    assert first["cases_evaluated"] == second["cases_evaluated"] == 5
    assert first["pre_fix_bugs_caught"] == second["pre_fix_bugs_caught"] == 5
    assert first["post_fix_false_positives"] == second["post_fix_false_positives"] == 0
    assert first["k3_idle_dp_caught"] is True
    assert first["k3_offending_module"].endswith("layers.0.self_attn")
    assert first["sfa_mtp_alias_caught"] is True
    assert first["kda_workspace_caught"] is True
    assert first["kda_first_stomp_chunk"] == 74
    assert first["norm_truncation_caught"] is True
    assert first["moe_healed_localized"] is True
    assert first["moe_first_diverged_module"] == "layers.1.experts.2"
    payload = json.loads(first_bytes)
    assert payload["cases"] == first["cases"] == second["cases"]
    assert payload["pre_fix_bugs_caught"] == 5
