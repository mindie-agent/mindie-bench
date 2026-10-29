"""The actual sim consumer remains discoverable and preserves modeling boundaries."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mindie_sim")

from mindie_bench.benchmarks import value_scenarios, value_sim
from mindie_bench.value_homepage import homepage_markdown
from mindie_bench.value_resources import project_component_case


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    result = value_sim.run(tmp_path_factory.mktemp("sim-consumer"), 1)
    raw = json.loads(Path(result["measurements"][0]["raw"]).read_text())
    return result, raw


def test_new_sim_is_discovered_without_removed_a3_modules():
    assert "sim" in value_scenarios.default_scenarios()


def test_consumer_runs_public_paths_and_preserves_independent_work_and_boundaries(simulation):
    result, raw = simulation
    row = result["measurements"][0]
    candidate = row["candidate"]
    assert result["status"] == row["status"] == "pass"
    assert candidate["qkv_flops"] == 262144
    assert candidate["attention_flops"] == 18432
    assert candidate["larger_qkv_flops"] == 524288
    assert candidate["larger_attention_flops"] == 69632
    assert raw["operators"]["boundary"] == "sum_of_isolated_device_programs"
    assert raw["device_graph"]["boundary"] == "device_graph"
    assert raw["submitted_graph"]["boundary"] == "submitted_device_graph"
    assert raw["serving"]["metric_boundary"] == "device_token_commit"
    assert candidate["operator_sum_ms"] == pytest.approx(
        raw["device_graph"]["isolated"]["operator_sum_ms"])
    assert candidate["submitted_graph_ms"] > candidate["device_graph_ms"]
    assert candidate["generated_tokens"] == 3
    assert candidate["model_deterministic"]
    assert raw["serving"]["parameters"]["calibration"] == "none"
    assert raw["accuracy"]["status"] == "not_assessed"


def test_cut_graph_requires_input_and_delays_real_consumers(simulation):
    _, raw = simulation
    graph = raw["cut_graph"]
    assert graph["missing_input"]["error"].startswith("graph_dependency_unavailable:")
    assert graph["delayed"]["complete_ms"] - graph["ready"]["complete_ms"] == pytest.approx(5)
    assert min(row["ready_ms"] for row in graph["delayed"]["timeline"]) == 5
    for result in (graph["ready"], graph["delayed"], raw["device_graph"], raw["serving"]):
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        assert not any(result["execution"]["live_allocated"].values())


def test_actual_scenario_survives_public_projection_and_bilingual_rendering(simulation):
    case, _ = simulation
    public = project_component_case(case)
    observed = public["measurements"][0]["candidate"]
    assert observed["device_graph_ms"] == case["measurements"][0]["candidate"]["device_graph_ms"]
    assert observed["missing_dependency_rejected"] and observed["resources_released"]
    assert "raw" not in public["measurements"][0]
    assert not {"a3_graph_tpot_ms", "tpot_error_pct", "subgraph_critical_path_saving_us"} & observed.keys()
    report = {"product_version": "0.8.0", "release_gate": True,
              "studies": [], "component_cases": [public]}
    for language in ("zh", "en"):
        assert homepage_markdown(report, language)
