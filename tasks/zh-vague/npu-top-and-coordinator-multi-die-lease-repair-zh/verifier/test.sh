#!/bin/bash
set -eu
if [ -n "${MINDIE_BENCH_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_PYTHON}" ]; then
  PY="${MINDIE_BENCH_PYTHON}"
elif [ -n "${MINDIE_BENCH_CPU_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_CPU_PYTHON}" ]; then
  PY="${MINDIE_BENCH_CPU_PYTHON}"
elif command -v mindie-profiler >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif [ -x "/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python" ]; then
  PY="/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python"
else
  PY="python3"
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
    mod = load_mod("multi_die_lease_allocator", root / "vllm_ascend_diag/multi_die_lease_allocator.py")
    res = mod.reconcile_multi_die_and_leases()
    assert res["a2_parsed"]["devices"][0]["hbm"]["used_mb"] == 32768
    assert res["a2_parsed"]["usage_override_probe"][0]["aicore_percent"] == 55
    assert res["a2_parsed"]["usage_override_probe"][1]["aicore_percent"] == 70
    a3_devs = res["a3_parsed"]["devices"]
    assert [c["phy_id"] for c in a3_devs[0]["chips"]] == [0, 1], f"bad card 0 chips: {a3_devs[0]}"
    assert [c["phy_id"] for c in a3_devs[1]["chips"]] == [2, 3], f"bad card 1 chips: {a3_devs[1]}"
    assert [c["phy_id"] for c in a3_devs[2]["chips"]] == [4, 5], f"bad card 2 chips: {a3_devs[2]}"
    assert a3_devs[0]["hbm"] == {"used_mb": 43343, "total_mb": 131072}, f"bad card 0 hbm: {a3_devs[0]}"
    assert a3_devs[0]["processes"][0]["phy_id"] == 1, f"bad proc phy_id: {a3_devs[0]['processes']}"
    assert res["s1_dual_die"]["visible_devices_by_role"]["default"] == "2,3", f"bad s1_dual_die: {res['s1_dual_die']}"
    assert res["s1_pd_pair"]["visible_devices_by_role"] == {"prefill": "2,3", "decode": "4,5"}, f"bad s1_pd_pair: {res['s1_pd_pair']}"
    assert res["placement_ok"]["runtime_ids"] == ["runtime-32", "runtime-36"], f"bad placement_ok: {res['placement_ok']}"
    assert res["placement_busy"]["runtime_ids"] == [] and res["placement_busy"]["reason"], f"bad placement_busy: {res['placement_busy']}"
    dcmi = res["dcmi_decoded"]
    assert dcmi["1"]["thermal_limit"] is False
    assert dcmi["4"]["thermal_limit"] is True
    assert dcmi["8"]["power_limit"] is True
    assert dcmi[str((1 << 0) | (1 << 63))]["unknown_bits"] == [63]
    assert dcmi["ret_neg1"]["thermal_limit"] is None
    orig = json.loads((root / "vllm_ascend_diag/fixtures/npu_coordinator_case.json").read_text())["dual_die_smi"]
    perturbed = orig.replace(
        "| 0     2                   | 0000:9E:00.0  | 0           0    / 0          3343  / 65536        |",
        "| 0     2                   | 0000:9E:00.0  | 0           0    / 0          12288 / 65536        |",
    )
    res_p = mod.reconcile_multi_die_and_leases(dual_die_smi_override=perturbed)
    assert res_p["s1_dual_die"]["visible_devices_by_role"]["default"] == "4,5", f"bad perturbed s1_dual_die: {res_p['s1_dual_die']}"
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
