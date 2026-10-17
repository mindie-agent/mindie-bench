import json

import pytest

from mindie_bench.scorer.components import summarize_component_runtime


def test_component_summary_retains_partial_rpc_and_actual_hardware_observations(tmp_path):
    def row(event, timestamp, **facts):
        return dict(event=event, monotonic_sec=timestamp, trace_id="trial", **facts)

    records = [
        row("component_provenance", 0, component="coordinator", actual={"version": "0.5"}),
        row("coordinator_request", 1, request={"action": "activate"}),
        row("coordinator_response", 3, action="activate", response={"status": "active"}),
        row("coordinator_request", 4, request={"action": "heartbeat"}),
        row("coordinator_error", 5, action="heartbeat", unknown=True),
        row("coordinator_request", 6, request={"action": "release"}),
        row(
            "npu_observation",
            7,
            phase="active",
            selected_devices={"12": {"used_mb": 3200, "aicore_percent": 17}},
            elapsed_sec=0.2,
        ),
        row("npu_observation_rejected", 8),
        row(
            "diagnostics",
            9,
            result={"record_ref": "owned-record", "logging_failed": False, "dropped_records": 0},
            elapsed_sec=0.03,
        ),
    ]
    path = tmp_path / "components.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = summarize_component_runtime(path)
    assert result["coordinator"]["activate"]["latency"]["mean_sec"] == 2
    assert result["coordinator"]["heartbeat"]["acknowledged"] == 0
    assert result["coordinator"]["release"]["unacknowledged"] == 1
    assert result["npu_top"]["accepted_samples"] == 1
    assert result["npu_top"]["rejected_samples"] == 1
    assert result["npu_top"]["peak_observed_aicore_percent"] == {"12": 17}
    assert result["diagnostics"]["recorded_events"] == 1


def test_component_summary_rejects_an_outcome_without_a_request(tmp_path):
    path = tmp_path / "components.jsonl"
    path.write_text(
        json.dumps(
            dict(
                event="coordinator_response",
                action="release",
                response={"status": "released"},
                monotonic_sec=1,
                trace_id="trial",
            )
        )
    )
    with pytest.raises(ValueError, match="no corresponding request"):
        summarize_component_runtime(path)


def test_five_component_attribution_and_amortized_k_projection():
    from test_compare_seed_cost import row

    from mindie_bench.scorer.comparator import comparison

    baseline = [row("G1", tokens=100, seconds=60)]
    candidate = [
        row("G3", tokens=40, seconds=25),
        row("G3", "seed", tokens=120, seconds=70),
    ]
    result = comparison(baseline, candidate)
    attr = result["five_component_attribution"]
    assert set(attr["candidate"]) == {
        "remote-dev",
        "knowledge",
        "coordinator",
        "npu-top",
        "diagnostics",
    }
    proj = result["amortized_k_projection"]
    assert proj["break_even_k_tokens"] == pytest.approx(121 / 60)
    assert proj["by_k"]["1"]["net_token_savings"] < 0
    assert proj["by_k"]["3"]["net_token_savings"] > 0
    assert proj["by_k"]["10"]["net_delivery_sec_savings"] > 0
