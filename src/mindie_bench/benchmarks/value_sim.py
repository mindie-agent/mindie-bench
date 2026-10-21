"""Measure physical A3 graph prediction, operator planning and fusion screening.

[Role] Drive installed ``mindie_sim`` on the checked-in Qwen2.5-7B-Instruct/A3 workload.
[Invariants] Modeled durations stay separate from measured anchors; no uncalibrated value is relabeled as device time.
[Timeouts & Locks] Pure finite CPU work; each invocation is measured locally with no background service.
[Side-Effect Policy] Owned output files only; no device, network, subprocess or source-checkout mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from .value_common import source_revisions, write_json

A3_SKU = "Ascend910_9362"
MIB = 1024 * 1024
MEASURED_ANCHORS = {
    "source": "mindie-sim/docs/large-operator-calibration.md measured A3 910C c=1 graph 128/32",
    "prompt_tokens": 128,
    "output_tokens": 32,
    "graph_ttft_ms": 50.85,
    "graph_tpot_ms": 15.71,
}

QWEN_CASE = {
    "name": "qwen2.5-7b-instruct-a3-tp1-graph",
    "machine": {
        "name": "a3-ascend910-9362-one-die",
        "hardware_key": A3_SKU,
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 1,
        "device": {
            "hbm_gib": 64,
            "cube_flops_per_s": 2.70336e14,
            "vector_ops_per_s": 1.6896e13,
            "hbm_bw_bytes_per_s": 1.2e12,
            "mte_bw_bytes_per_s": 2.4e12,
            "launch_us": 20.0,
            "l2_bytes": 201326592,
            "aic_count": 20,
            "aiv_count": 40,
        },
        "compute_peaks": {"bf16": 2.70336e14, "fp16": 2.70336e14},
        "links": [{
            "name": "d2d",
            "channel": "d2d_intra_node",
            "capacity_gbps": 196,
            "latency_us": 2.5,
            "members": [0],
        }],
    },
    "model": {
        "name": "Qwen2.5-7B-Instruct",
        "architecture": "gqa",
        "layers": 28,
        "hidden_size": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate_size": 18944,
        "vocab_size": 152064,
        "activation_dtype": "bf16",
        "weight_dtype": "bf16",
        "kv_dtype": "bf16",
        "compute_dtype": "bf16",
    },
    "workload": {
        "requests": [{"id": "qwen25-7b-one", "prompt_tokens": 128, "output_tokens": 4}],
    },
    "runtime": {
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 16,
        "max_model_len": 8192,
        "enable_chunked_prefill": True,
        "kv_block_size": 128,
        "gpu_memory_utilization": 0.9,
        "tp": 1,
        "ep": 1,
        "dp": 1,
        "pp": 1,
        "graph_buckets": [1, 2, 4, 8, 16],
        "graph_mode": "force",
        "moe_recipe": "auto",
    },
    "slo": {"ttft_ms": 60000, "tpot_ms": 6000, "e2e_ms": 180000},
}

QWEN_CALIBRATION = {
    "schema_version": "mindie-sim.physical-calibration/1",
    "hardware_key": A3_SKU,
    "arch_family": "A3",
    "machine": {
        "hardware_key": A3_SKU,
        "arch_family": "A3",
        "aic_count": 20,
        "aiv_count": 40,
        "clock_mhz": 1650.0,
        "ops_per_cycle": 8192.0,
        "peak_bw_bytes_per_s": 1.2e12,
        "l2_bytes": 67108864,
        "vec_elements_per_cycle_per_aiv": 128.0,
    },
    "fit_config": {"fit_policy": "qwen2.5-7b-instruct-public-benchmark-fixture"},
    "families": {
        "gemm/device_event_warm": {
            "family": "gemm",
            "recipe": "torch_npu_managed_runner",
            "protocol_signature": "",
            "timing_scope": "device_event_warm",
            "model_kind": "gemm",
            "model_form": "wave_max",
            "fit_level": "family_protocol",
            "block_m": 32,
            "block_n": 256,
            "launch_ms": 0.0177,
            "eta_compute": 0.837,
            "eta_memory": 0.814,
            "vec_ops_per_s": None,
            "memory_model": "hbm",
            "l2_usable_bytes": 0,
            "l2_bw_bytes_per_s": None,
            "compute_dtype": "bf16",
            "samples": 367,
            "raw_event_count": 2202,
            "train_wape_pct": 16.6,
            "cv_wape_pct": 17.65,
            "median_ape_pct": 20.3,
            "p90_ape_pct": 51.8,
            "max_ape_pct": 85.2,
            "binding_share": {"compute": 0.5, "memory": 0.5},
            "binding_counts": {"compute": 180, "memory": 187},
            "identified": {"launch": True, "tile_geometry": True, "eta_compute": True},
            "pinned_parameters": [],
            "intervals": {},
            "breakdown": {},
            "model_selection": {},
            "memory_model_selection": {},
            "quality": "trusted",
            "warnings": [],
        },
    },
    "provenance": {
        "generator": "benchmarks/fixtures/qwen2.5-7b-instruct.json",
        "condition_count": 367,
        "raw_event_count": 2202,
        "fitted_utc": "2026-10-21T00:00:00+00:00",
        "training_hash": "sha256:qwen2.5-7b-instruct-public-benchmark-fixture",
        "heldout_guard": {"status": "clean"},
    },
    "unmodeled": [],
}

OPERATOR_PLAN = {
    "name": "synthetic-vector-gemm-vector",
    "source": "synthetic_fixture: arithmetic demonstration; no Ascend SKU or measured latency claim",
    "machine": {
        "name": "synthetic-two-core-device",
        "arch_family": "A5",
        "num_devices": 1,
        "dies_per_package": 1,
        "device": {
            "hbm_gib": 1,
            "cube_flops_per_s": 10000000000,
            "vector_ops_per_s": 1000000000,
            "hbm_bw_bytes_per_s": 4000000000,
            "mte_bw_bytes_per_s": 2000000000,
            "mte1_bw_bytes_per_s": 2000000000,
            "launch_us": 1,
            "aic_count": 2,
            "aiv_count": 2,
            "ub_bytes": 8192,
            "l1_bytes": 8192,
            "l0a_bytes": 2048,
            "l0b_bytes": 2048,
            "l0c_bytes": 2048,
        },
        "compute_peaks": {"bf16": 10000000000},
        "metadata": {"source": "synthetic_fixture; not hardware calibration"},
    },
    "estimators": {
        "strict": True,
        "vector_profile": {
            "name": "synthetic-two-core-vector",
            "source": "synthetic_fixture: independent arithmetic rates, not an ISA calibration",
            "architectures": ["A5"],
            "frequency_hz": 1000000000,
            "simd_bytes": 16,
            "vector_cores": 2,
            "ub_bytes_per_core": 8192,
            "tile_elements": 128,
            "buffers": 2,
            "instructions": [
                {"name": "Cast", "dtypes": ["bf16", "fp32"]},
                {"name": "Mul", "dtypes": ["fp32"]},
                {"name": "Neg", "dtypes": ["fp32"]},
                {"name": "Add", "dtypes": ["fp32"]},
                {"name": "Exp", "pipe": "sfu", "throughput_cycles": 4,
                 "latency_cycles": 4, "dtypes": ["fp32"]},
                {"name": "Div", "pipe": "sfu", "throughput_cycles": 4,
                 "latency_cycles": 4, "dtypes": ["fp32"]},
            ],
        },
        "gemm_profile": {
            "source": "synthetic_fixture: BF16 arithmetic rates, not measured kernel parameters",
            "core_count": 2,
            "tile_m": 16,
            "tile_n": 16,
            "tile_k": 32,
            "l1_k": 64,
            "l1_bytes": 8192,
            "l0a_bytes": 2048,
            "l0b_bytes": 2048,
            "l0c_bytes": 2048,
            "l2_bytes_per_s_per_core": 1000000000,
            "mte2_bytes_per_s_per_core": 1000000000,
            "mte1_a_bytes_per_s_per_core": 1000000000,
            "mte1_b_bytes_per_s_per_core": 1000000000,
            "mte3_bytes_per_s_per_core": 1000000000,
            "fix_bytes_per_s_per_core": 1000000000,
        },
    },
    "operators": [
        {
            "op_id": "pre",
            "family": "vector_chain",
            "phase": "prefill",
            "stage": 0,
            "layer": 0,
            "requests": ["synthetic"],
            "semantic_shape": [2560],
            "local_shape": [2560],
            "padded_shape": [2560],
            "launch_shape": [2560],
            "dtype": "bf16",
            "hbm_bytes": 10240,
            "mte2_bytes": 5120,
            "mte3_bytes": 5120,
            "vector_ops": 10240,
            "metadata": {"vector_flow": {
                "elements": 2560,
                "input_bytes_per_element": 2,
                "output_bytes_per_element": 2,
                "ub_bytes_per_element": 20,
                "instructions": [
                    {"name": "cast_in", "op": "Cast", "src_dtype": "bf16", "dst_dtype": "fp32"},
                    {"name": "multiply", "op": "Mul", "dtype": "fp32", "deps": ["cast_in"]},
                    {"name": "negate", "op": "Neg", "dtype": "fp32", "deps": ["multiply"]},
                    {"name": "cast_out", "op": "Cast", "src_dtype": "fp32",
                     "dst_dtype": "bf16", "deps": ["negate"]},
                ],
            }},
        },
        {
            "op_id": "gemm",
            "family": "gemm",
            "phase": "prefill",
            "stage": 0,
            "layer": 0,
            "requests": ["synthetic"],
            "semantic_shape": [32, 48, 80],
            "local_shape": [32, 48, 80],
            "padded_shape": [32, 48, 80],
            "launch_shape": [32, 48, 80],
            "dtype": "bf16",
            "flops": 245760,
            "hbm_bytes": 15872,
            "mte2_bytes": 12800,
            "mte3_bytes": 3072,
            "deps": ["pre"],
        },
        {
            "op_id": "post",
            "family": "vector_chain",
            "phase": "prefill",
            "stage": 0,
            "layer": 0,
            "requests": ["synthetic"],
            "semantic_shape": [1536],
            "local_shape": [1536],
            "padded_shape": [1536],
            "launch_shape": [1536],
            "dtype": "bf16",
            "hbm_bytes": 6144,
            "mte2_bytes": 3072,
            "mte3_bytes": 3072,
            "vector_ops": 4608,
            "deps": ["gemm"],
            "metadata": {"vector_flow": {
                "elements": 1536,
                "input_bytes_per_element": 2,
                "output_bytes_per_element": 2,
                "ub_bytes_per_element": 16,
                "instructions": [
                    {"name": "cast_in", "op": "Cast", "src_dtype": "bf16", "dst_dtype": "fp32"},
                    {"name": "negate", "op": "Neg", "dtype": "fp32", "deps": ["cast_in"]},
                    {"name": "cast_out", "op": "Cast", "src_dtype": "fp32",
                     "dst_dtype": "bf16", "deps": ["negate"]},
                ],
            }},
        },
    ],
}

FUSION_DAG = {
    "name": "value-producer-consumer-cast",
    "nodes": [
        {"id": "producer", "kind": "matmul", "duration_us": 10.0},
        {"id": "consumer", "kind": "aiv_cast", "duration_us": 5.0},
        {"id": "tail", "kind": "attention", "duration_us": 20.0},
    ],
    "edges": [
        {
            "src": "producer",
            "dst": "consumer",
            "tensor": {"bytes": 1 * MIB, "dtype": "float32", "layout": "ND"},
        },
        {"src": "consumer", "dst": "tail"},
    ],
    "capacities_kib": {"L1": 512, "L0A": 64, "L0B": 64, "L0C": 256, "UB": 248},
}
FUSION_CANDIDATE = {
    "name": "elide-cast",
    "producer": "producer",
    "consumer": "consumer",
    "remove_nodes": ["consumer"],
    "remove_edges": [{"src": "producer", "dst": "consumer"}],
    "live_buffers": [{"name": "UB scratch", "buffer": "UB", "size_kib": 128}],
}
FUSION_OVERFLOW = {
    **FUSION_CANDIDATE,
    "name": "elide-cast-overflow",
    "live_buffers": [{"name": "UB scratch", "buffer": "UB", "size_kib": 512}],
}


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


def _validate_qwen(payload: dict) -> None:
    assert payload["status"] == "modeled"
    assert payload["served_requests"] == 1 and payload["unserved_requests"] == 0
    assert payload["output_tokens"] == 4
    calibration = payload["physical_calibration"]
    assert calibration["status"] == "configured"
    assert calibration["timing_scope"] == "device_event_warm"
    assert calibration["applied_op_count"] > 0
    assert calibration["unmodeled_op_count"] > 0
    request = payload["requests"][0]
    assert request["status"] == "served" and request["generated_tokens"] == 4
    ttft, tpot = float(request["ttft_ms"]), float(request["tpot_ms"])
    assert math.isfinite(ttft) and ttft > 0
    assert math.isfinite(tpot) and tpot > 0
    assert _close(request["e2e_ms"], ttft + (request["generated_tokens"] - 1) * tpot)
    assert _close(payload["metrics"]["ttft_ms"]["mean"], ttft)
    assert _close(payload["metrics"]["tpot_ms"]["mean"], tpot)
    assert len(payload["traces"]) == request["generated_tokens"]


def _validate_operator_plan(payload: dict) -> None:
    assert payload["status"] == "modeled"
    assert payload["execution"]["status"] == "modeled"
    assert _close(payload["elapsed_ms"], payload["execution"]["makespan_ms"])
    assert [op["op_id"] for op in payload["operators"]] == ["pre", "gemm", "post"]
    assert all(op["operator_estimation"]["status"] == "analytic_plan"
               for op in payload["operators"])
    assert {op["execution_plan"]["backend"] for op in payload["operators"]} == {
        "vector_instruction", "gemm_finite_pipeline",
    }
    assert all(op["operator_estimation"]["not_npu_measured"] for op in payload["operators"])


def _validate_fusion(payload: dict, *, feasible: bool) -> None:
    assert payload["status"] == ("modeled" if feasible else "infeasible")
    assert payload["feasibility"]["feasible"] is feasible
    assert _close(payload["removable_gm_bytes"], 2 * MIB)
    assert _close(payload["latency"]["critical_path_saving_us"], 5.0)
    if feasible:
        assert payload["feasibility"]["failures"] == []
        assert payload["buffer_capacities_kib"]["UB"] == 248
    else:
        assert payload["feasibility"]["failures"] == ["live_buffer_limit"]
        gate = next(item for item in payload["feasibility"]["gates"]
                    if item["code"] == "live_buffer_limit")
        assert gate["detail"]["diagnostics"][0]["code"] == "buffer_overflow"


def _timed(call):
    started = time.perf_counter()
    value = call()
    return value, (time.perf_counter() - started) * 1000


def run(output: Path, repetitions: int):
    from mindie_sim import analyze_subgraph_fusion
    from mindie_sim.a3_framework import predict
    from mindie_sim.a3_physical_calibration import PhysicalCalibration
    from mindie_sim.operator_estimation import operator_plan_report

    calibration = PhysicalCalibration.from_mapping(QWEN_CALIBRATION)
    measurements = []
    for repetition in range(1, repetitions + 1):
        root = output / f"sim-{repetition}"
        root.mkdir(parents=True)
        payload, qwen_ms = _timed(lambda: predict(QWEN_CASE, physical_calibration=calibration))
        _validate_qwen(payload)
        repeat_payload, _ = _timed(
            lambda: predict(QWEN_CASE, physical_calibration=calibration)
        )
        assert json.dumps(payload, sort_keys=True) == json.dumps(repeat_payload, sort_keys=True)
        request = payload["requests"][0]
        ttft = float(request["ttft_ms"])
        tpot = float(request["tpot_ms"])
        ttft_error = ttft - MEASURED_ANCHORS["graph_ttft_ms"]
        tpot_error = tpot - MEASURED_ANCHORS["graph_tpot_ms"]
        operator, operator_ms = _timed(lambda: operator_plan_report(OPERATOR_PLAN))
        _validate_operator_plan(operator)
        fusion, fusion_ms = _timed(lambda: analyze_subgraph_fusion(FUSION_DAG, FUSION_CANDIDATE))
        _validate_fusion(fusion, feasible=True)
        overflow, overflow_ms = _timed(
            lambda: analyze_subgraph_fusion(FUSION_DAG, FUSION_OVERFLOW)
        )
        _validate_fusion(overflow, feasible=False)
        raw = {
            "qwen_case_sha256": _digest(QWEN_CASE),
            "physical_calibration_sha256": _digest(QWEN_CALIBRATION),
            "operator_plan_sha256": _digest(OPERATOR_PLAN),
            "qwen_prediction": {
                "status": payload["status"],
                "timing_scope": payload["timing_scope"],
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "e2e_ms": request["e2e_ms"],
                "traces": len(payload["traces"]),
                "physical_calibration": payload["physical_calibration"],
                "metrics": payload["metrics"],
            },
            "measured_anchor": MEASURED_ANCHORS,
            "accuracy": {
                "ttft_absolute_error_ms": ttft_error,
                "ttft_error_pct": ttft_error / MEASURED_ANCHORS["graph_ttft_ms"] * 100,
                "tpot_absolute_error_ms": tpot_error,
                "tpot_error_pct": tpot_error / MEASURED_ANCHORS["graph_tpot_ms"] * 100,
                "scope": "published A3 c=1 graph anchors; the fixed client/scheduler floor is not modeled",
            },
            "operator_plan": operator,
            "subgraph_screening": {
                "feasible": fusion,
                "overflow": overflow,
            },
            "latency_ms": {
                "qwen_prediction": qwen_ms,
                "operator_plan": operator_ms,
                "subgraph_feasible": fusion_ms,
                "subgraph_overflow": overflow_ms,
            },
        }
        write_json(root / "raw.json", raw)
        measurements.append({
            "repetition": repetition,
            "status": "pass",
            "raw": str(root / "raw.json"),
            "baseline": {
                "source": "published A3 Qwen2.5-7B-Instruct c=1 graph measurement",
                "prompt_tokens": MEASURED_ANCHORS["prompt_tokens"],
                "output_tokens": MEASURED_ANCHORS["output_tokens"],
                "graph_ttft_ms": MEASURED_ANCHORS["graph_ttft_ms"],
                "graph_tpot_ms": MEASURED_ANCHORS["graph_tpot_ms"],
            },
            "candidate": {
                "qwen_case_deterministic": True,
                "a3_graph_ttft_ms": ttft,
                "a3_graph_tpot_ms": tpot,
                "ttft_absolute_error_ms": ttft_error,
                "ttft_error_pct": ttft_error / MEASURED_ANCHORS["graph_ttft_ms"] * 100,
                "tpot_absolute_error_ms": tpot_error,
                "tpot_error_pct": tpot_error / MEASURED_ANCHORS["graph_tpot_ms"] * 100,
                "operator_plan_elapsed_ms": operator["elapsed_ms"],
                "operator_plan_backends": sorted(
                    {op["execution_plan"]["backend"] for op in operator["operators"]}
                ),
                "subgraph_removable_gm_bytes": fusion["removable_gm_bytes"],
                "subgraph_critical_path_saving_us": fusion["latency"]["critical_path_saving_us"],
                "subgraph_overflow_rejected": True,
                "qwen_prediction_latency_ms": qwen_ms,
                "operator_plan_latency_ms": operator_ms,
                "subgraph_screen_latency_ms": fusion_ms,
                "subgraph_overflow_latency_ms": overflow_ms,
            },
        })
    return {
        "id": "sim-a3-graph-and-boundary-screening",
        "component": "sim",
        "status": "pass",
        "scope": "real A3 physical-calibration predict, finite operator planning and subgraph boundary gates",
        "baseline": "Published A3 c=1 Qwen2.5-7B-Instruct graph TTFT/TPOT anchors plus a raw producer-consumer DAG.",
        "candidate": "Production sim models graph latency, plans Vector/GEMM operators and rejects an overflowing UB fusion.",
        "repetitions": repetitions,
        "measurements": measurements,
        "limitations": [
            "Predicted TTFT includes no client/scheduler floor; its absolute error is recorded, not hidden.",
            "The Qwen physical calibration artifact is a checked-in public fixture, not a new NPU measurement.",
            "Operator-plan and subgraph inputs are synthetic arithmetic fixtures.",
            "Latency is CPU wall-clock for these pure-Python calls, not device throughput.",
        ],
        "source_revisions": source_revisions(("mindie-sim",)),
    }
