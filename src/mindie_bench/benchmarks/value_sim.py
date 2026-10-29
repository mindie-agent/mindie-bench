"""Exercise the public operator, graph and serving boundaries of mindie-sim.

[Role] Run a finite neutral model through the installed causal engine.
[Invariants] Synthetic resource inputs test behavior, never measured accuracy.
  Isolated sums, device graphs and token-commit metrics keep separate meanings.
[Timeouts & Locks] Bounded CPU-only model calls; no services, waits or locks.
[Side-Effect Policy] Writes caller-owned evidence only; no device or network use.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from .value_common import source_revisions, write_json

CASE = {
    "machine": {
        "name": "synthetic-component-control",
        "architecture": "declared",
        "num_devices": 1,
        "device": {
            "memory_bytes": 1024**3,
            "compute_units": 4,
            "compute_peaks": {"bf16": 16e9, "fp32": 8e9},
            "rates": {"memory": 8e9, "vector": 4e9},
            "launch_us": 2,
        },
        "assumptions": {
            "parameters": "Synthetic arithmetic control; no hardware calibration or SKU claim."
        },
    },
    "model": {
        "name": "component-gqa-control",
        "layers": 2,
        "hidden_size": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 256,
        "activation_dtype": "bf16",
        "weight_dtype": "bf16",
        "kv_dtype": "bf16",
        "compute_dtype": "bf16",
    },
    "workload": {
        "requests": [{"id": "control", "prompt_tokens": 8, "output_tokens": 3}],
    },
    "runtime": {
        "max_num_batched_tokens": 16,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "kv_block_size": 16,
        "graph_mode": "none",
    },
}


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _timed(call):
    started = time.perf_counter()
    value = call()
    return value, (time.perf_counter() - started) * 1000


def _released(payload):
    execution = payload["execution"]
    return (
        execution["live_tasks"] == execution["live_programs"] == 0
        and not any(execution["live_allocated"].values())
    )


def _check_model(operators, graph, serving, tokens):
    model = CASE["model"]
    qkv = sum(op.flops for op in operators if op.family == "qkv_projection")
    attention = sum(op.flops for op in operators if op.family == "attention")
    expected_qkv = (
        model["layers"] * 2 * tokens * model["hidden_size"]
        * (model["num_heads"] + 2 * model["num_kv_heads"]) * model["head_dim"]
    )
    expected_attention = (
        model["layers"] * 4 * (tokens * (tokens + 1) // 2)
        * model["num_heads"] * model["head_dim"]
    )
    assert qkv == expected_qkv and attention == expected_attention
    assert graph["boundary"] == "device_graph"
    assert graph["duration_ms"] + 1e-9 >= graph["bounds"]["combined_ms"]
    assert not any(row["family"] in {"sampling", "token_commit"}
                   for row in graph["timeline"])
    assert serving["status"] == "modeled"
    assert serving["metric_boundary"] == "device_token_commit"
    assert serving["metrics"]["served_requests"] == 1
    assert serving["metrics"]["unserved_requests"] == 0
    request = serving["requests"][0]
    generated = CASE["workload"]["requests"][0]["output_tokens"]
    assert request["status"] == "served" and request["generated_tokens"] == generated
    assert _close(request["e2e_ms"],
                  request["ttft_ms"] + (generated - 1) * request["tpot_ms"])
    assert serving["parameters"]["calibration"] == "none"
    assert serving["parameters"]["operator_occurrences"]["measured_primitive"] == 0
    assert _released(graph) and _released(serving)
    return expected_qkv, expected_attention


def run(output: Path, repetitions: int):
    from mindie_sim import (
        graph_costs,
        operator_costs,
        predict,
        target_graphs,
        target_operators,
    )

    tokens = CASE["workload"]["requests"][0]["prompt_tokens"]
    case_digest = _digest(CASE)
    measurements = []
    for repetition in range(1, repetitions + 1):
        root = output / f"sim-{repetition}"
        root.mkdir(parents=True)
        operators = target_operators(CASE, phase="prefill", tokens=tokens, scope="model")
        isolated, operator_ms = _timed(lambda: operator_costs(CASE, operators))
        graphs = target_graphs(CASE, phase="prefill", tokens=tokens, scope="model")
        graph, graph_ms = _timed(lambda: graph_costs(CASE, graphs))
        submitted = graph_costs(CASE, graphs, include_submission=True, isolated=False)
        serving, prediction_ms = _timed(
            lambda: predict(CASE, trace_limit=0, event_limit=0))
        repeated = predict(CASE, trace_limit=0, event_limit=0)
        expected_qkv, expected_attention = _check_model(operators, graph, serving, tokens)
        assert serving == repeated and _digest(CASE) == case_digest
        assert isolated["boundary"] == "sum_of_isolated_device_programs"
        assert _close(isolated["operator_sum_ms"], graph["isolated"]["operator_sum_ms"])
        assert isolated["operator_occurrences"] == graph["operator_occurrences"]
        assert submitted["boundary"] == "submitted_device_graph"
        assert submitted["duration_ms"] > graph["duration_ms"]
        assert _released(submitted)

        # A cut body graph needs an actual input arrival; a missing producer
        # cannot disappear merely because isolated operator costs are available.
        body = target_graphs(CASE, phase="prefill", tokens=tokens, scope="body")
        try:
            graph_costs(CASE, body, isolated=False)
        except ValueError as error:
            if not str(error).startswith("graph_dependency_unavailable:"):
                raise
            rejection = {"type": type(error).__name__, "error": str(error)}
        else:
            raise AssertionError("cut graph accepted an undeclared external input")
        ready = graph_costs(CASE, body, external_inputs={"analysis_input": 0}, isolated=False)
        delayed = graph_costs(CASE, body, external_inputs={"analysis_input": 5}, isolated=False)
        input_shift = delayed["complete_ms"] - ready["complete_ms"]
        assert _close(input_shift, 5)
        assert all(row["ready_ms"] >= 5 for row in delayed["timeline"])
        assert _released(ready) and _released(delayed)

        larger = target_operators(CASE, phase="prefill", tokens=2 * tokens, scope="model")
        larger_qkv = sum(op.flops for op in larger if op.family == "qkv_projection")
        larger_attention = sum(op.flops for op in larger if op.family == "attention")
        assert larger_qkv == 2 * expected_qkv
        assert larger_attention == (
            CASE["model"]["layers"] * 4 * (2 * tokens * (2 * tokens + 1) // 2)
            * CASE["model"]["num_heads"] * CASE["model"]["head_dim"]
        )
        request = serving["requests"][0]
        raw = {
            "case": CASE,
            "case_sha256": case_digest,
            "operators": isolated,
            "device_graph": graph,
            "submitted_graph": submitted,
            "serving": serving,
            "cut_graph": {"ready": ready, "delayed": delayed, "missing_input": rejection},
            "accuracy": {
                "status": "not_assessed",
                "reason": "synthetic resource control has no paired heldout timing observations",
            },
            "latency_ms": {
                "operator_analysis": operator_ms,
                "graph_analysis": graph_ms,
                "prediction": prediction_ms,
            },
        }
        write_json(root / "raw.json", raw)
        measurements.append({
            "repetition": repetition,
            "status": "pass",
            "raw": str(root / "raw.json"),
            "baseline": {
                "prefill_tokens": tokens,
                "expected_qkv_flops": expected_qkv,
                "expected_attention_flops": expected_attention,
                "external_input_ready_ms": 5,
            },
            "candidate": {
                "model_deterministic": serving == repeated,
                "analytic_without_calibration": True,
                "qkv_flops": sum(op.flops for op in operators if op.family == "qkv_projection"),
                "attention_flops": sum(op.flops for op in operators if op.family == "attention"),
                "larger_qkv_flops": larger_qkv,
                "larger_attention_flops": larger_attention,
                "operator_occurrences": isolated["operator_occurrences"],
                "operator_sum_ms": isolated["operator_sum_ms"],
                "device_graph_ms": graph["duration_ms"],
                "submitted_graph_ms": submitted["duration_ms"],
                "serving_ttft_ms": request["ttft_ms"],
                "serving_tpot_ms": request["tpot_ms"],
                "generated_tokens": request["generated_tokens"],
                "external_input_shift_ms": input_shift,
                "missing_dependency_rejected": True,
                "resources_released": True,
                "operator_analysis_latency_ms": operator_ms,
                "graph_analysis_latency_ms": graph_ms,
                "prediction_latency_ms": prediction_ms,
            },
        })
    return {
        "id": "sim-causal-modeling-boundaries",
        "component": "sim",
        "status": "pass",
        "scope": "public operator, pure-graph and serving contracts on a synthetic resource control",
        "baseline": "Independent GQA work arithmetic and a declared external-input arrival.",
        "candidate": "Installed causal engine composes model work and preserves graph/serving boundaries.",
        "repetitions": repetitions,
        "measurements": measurements,
        "limitations": [
            "Pass means structural consumer checks succeeded, not measured timing accuracy.",
            "Resource rates are synthetic; no A3/GPU specification or old fitted parameters are used.",
            "Independent operator sums, device/submitted graph spans and token-commit metrics differ.",
            "No fusion feasibility or optimization saving is inferred from graph lower bounds.",
            "Latency fields measure local CPU analysis time; no device execution or calibration occurs.",
        ],
        "source_revisions": source_revisions(("mindie-sim",)),
    }
