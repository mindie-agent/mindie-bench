# [Role] Independent behavioral oracle for a controlled task workspace.
# [Invariants] Rewards require executed input/output checks; source shape is never a success criterion.
# [Timeouts & Locks] The runner bounds this subprocess; no external service calls.
# [Side-Effect Policy] Candidate imports execute only in the supplied verification workspace.
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    mod = load_mod("profile_reconciler", root / "vllm_ascend_diag/profile_reconciler.py")
    res = mod.reconcile_profile_and_hbm()
    rep = res["profile_report"]
    assert rep["step_accounting"]["steady_steps"] == [4, 5], (
        f"bad steady_steps: {rep['step_accounting']}"
    )
    w0 = rep["window_accounting"][0]
    assert math.isclose(w0["busy_union_ms"], 0.04, rel_tol=1e-6), f"bad w0 busy_union_ms: {w0}"
    assert math.isclose(w0["kernel_sum_ms"], 0.061, rel_tol=1e-6), f"bad w0 kernel_sum_ms: {w0}"
    assert math.isclose(w0["host_gap_ms"], 0.005, rel_tol=1e-6), f"bad w0 host_gap_ms: {w0}"
    assert math.isclose(w0["exposed_comm_ms"], 0.01, rel_tol=1e-6), f"bad w0 exposed_comm_ms: {w0}"
    w2 = next(
        w for w in rep["window_accounting"] if w.get("rank_index", w.get("rank")) == 1 and w["step"] == 4
    )
    assert math.isclose(w2["busy_union_ms"], 0.05, rel_tol=1e-6), f"bad w2 busy_union_ms: {w2}"
    assert math.isclose(w2["exposed_comm_ms"], 0.025, rel_tol=1e-6), f"bad w2 exposed_comm_ms: {w2}"
    slowest = rep["step_accounting"]["slowest_rank"]
    assert slowest["rank_index"] == 1, f"bad slowest rank_index: {slowest}"
    assert slowest["dominant_excess_block"] == "Comm", f"bad dominant_excess_block: {slowest}"
    assert math.isclose(slowest["busy_union_ms_p50"], 0.0475, rel_tol=1e-6), (
        f"bad busy_union_ms_p50: {slowest}"
    )
    comp = res["comparison"]
    assert comp["comparable"] is True, f"not comparable: {comp}"
    assert comp["matching"]["steps"]["status"] == "same_steps", f"bad step matching: {comp}"
    mem = res["memory_attribution"]
    assert mem["target_met"] is True, f"target_met should be True: {mem}"
    dev0 = mem["devices"][0]
    assert math.isclose(dev0["known_sum_bytes"], 4_100_000_000, rel_tol=1e-9), (
        f"bad known_sum_bytes: {dev0}"
    )
    assert math.isclose(dev0["residual"]["bytes"], 150_000_000, rel_tol=1e-9), (
        f"bad residual bytes: {dev0}"
    )
    assert dev0["residual"]["status"] == "within_target", f"bad residual status: {dev0}"
    cmap = {c["key"]: c for c in dev0["components"]}
    for k in ("baseline_observed", "weights", "kv_cache", "graph"):
        assert cmap[k]["included_in_sum"] is True, f"{k} must be included_in_sum"
    for k in ("hccl", "runtime", "slog", "activation"):
        assert cmap[k]["included_in_sum"] is False, f"{k} must not be included_in_sum"
    res2 = mod.reconcile_profile_and_hbm(after_infer_override_bytes=4_550_000_000)
    assert res2["memory_attribution"]["target_met"] is False, "perturbed peak must fail target_met"
    assert math.isclose(
        res2["memory_attribution"]["devices"][0]["residual"]["bytes"], 450_000_000, rel_tol=1e-9
    )
    assert res2["memory_attribution"]["devices"][0]["residual"]["status"] == "above_target"
    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "controlled_offline_fixture",
                "native_kernel_bound": None,
                "measured_npu_speedup": None,
                "speedup_ratio": None,
                "tpot_ms": None,
            }
        )
    )
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc), "failure_type": type(exc).__name__}))
    raise SystemExit(1)
