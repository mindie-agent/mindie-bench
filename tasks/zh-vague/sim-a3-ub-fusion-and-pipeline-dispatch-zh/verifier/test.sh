#!/bin/bash
set -eu
if command -v mindie-profiler >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY="/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python"
fi
exec "$PY" -I -B -c '
import importlib.util, json, math, os, sys
from pathlib import Path

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    mod = load_mod("a3_sim_planner", root / "vllm_ascend_diag/a3_sim_planner.py")
    res = mod.evaluate_a3_simulation_suite()
    fl = res["four_layer"]
    assert math.isclose(fl["hap"]["t_hw_us"], 508.40048484848484, rel_tol=1e-7), f"bad hap t_hw_us: {fl['hap']}"
    assert math.isclose(fl["hap"]["hap_score"], 0.8429360983266381, rel_tol=1e-7), f"bad hap_score: {fl['hap']}"
    assert math.isclose(fl["pipeline"]["minimum_period_ns"], 350.0, rel_tol=1e-7), f"bad pipeline: {fl['pipeline']}"
    assert math.isclose(fl["pipeline"]["minimum_tile_ii_ns"], 350.0, rel_tol=1e-7)
    assert fl["fusion"]["feasibility"]["failures"] == [], f"unexpected fusion failures: {fl['fusion']}"
    assert math.isclose(fl["fusion"]["removable_gm_bytes"], 2 * 1024 * 1024, rel_tol=1e-7), f"bad removable_gm_bytes: {fl['fusion']}"
    assert math.isclose(fl["fusion"]["latency"]["critical_path_saving_us"], 5.0, rel_tol=1e-7), f"bad saving_us: {fl['fusion']}"
    assert math.isclose(fl["transfer"]["lower_bound_us"], 344.8921632653061, rel_tol=1e-7)
    assert [t["tier"] for t in fl["collective"]["tiers"]] == ["d2d_intra_package", "d2d_intra_node", "d2d_inter_node"], f"bad tiers: {fl['collective']}"
    assert math.isclose(fl["collective"]["lower_bound_us"], 2216.780863295949, rel_tol=1e-7)
    qp = res["qwen_physical"]
    assert qp["physical_calibration"]["status"] == "configured", f"bad cal status: {qp['physical_calibration']}"
    assert qp["physical_calibration"]["timing_scope"] == "device_event_warm"
    assert qp["physical_calibration"]["applied_op_count"] > 0
    req = qp["requests"][0]
    assert math.isclose(req["e2e_ms"], req["ttft_ms"] + (req["generated_tokens"] - 1) * req["tpot_ms"], rel_tol=1e-7), f"bad e2e_ms: {req}"
    op = res["operator_plan"]
    assert math.isclose(op["elapsed_ms"], op["execution"]["makespan_ms"], rel_tol=1e-7), f"bad op elapsed_ms: {op}"
    assert all(o["operator_estimation"]["not_npu_measured"] is True for o in op["operators"])
    res_overflow = mod.evaluate_a3_simulation_suite(ub_scratch_override_kib=300)
    assert len(res_overflow["four_layer"]["fusion"]["feasibility"]["failures"]) > 0, "UB overflow must produce feasibility failure"
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
