#!/bin/bash
set -eu
exec /tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python -I -B -c 'import ast, importlib.util, json, os, shutil, sys, tempfile
from pathlib import Path

def _arm(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for sub in ("checks", "correctness", "metrics"):
        if isinstance(d.get(sub), dict):
            out.update(d[sub])
    out.update(d)
    return out

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    ledger_path = root / "experiment-ledger.jsonl"
    progress_path = root / "PROGRESS.md"
    assert ledger_path.is_file(), "Missing experiment-ledger.jsonl"
    assert progress_path.is_file(), "Missing PROGRESS.md"
    ledger_text = ledger_path.read_text(encoding="utf-8")
    for req_arm in ("full_candidate", "minus_q_fusion", "minus_view_alias", "minus_device_guard"):
        assert req_arm in ledger_text, f"experiment-ledger.jsonl missing arm {req_arm}"
    assert len(progress_path.read_text(encoding="utf-8").strip()) >= 40, "PROGRESS.md is empty or too short"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for sub in ("vllm_ascend", "logs"):
            if (root / sub).exists():
                shutil.copytree(root / sub, td_path / sub)
        shutil.copy2(ledger_path, td_path / "experiment-ledger.jsonl")
        shutil.copy2(progress_path, td_path / "PROGRESS.md")

        target = td_path / "vllm_ascend/models/deepseek_v4/sparse_mla_decode.py"
        src = target.read_text(encoding="utf-8")
        tree = ast.parse(src)
        unp = ast.unparse(tree)
        assert "fused_q_dequant_rmsnorm_rope_split" in unp, "Missing L4 Factor 1 fused_q_dequant_rmsnorm_rope_split op call"

        spec = importlib.util.spec_from_file_location("sparse_mla_decode", target)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_ablation_suite"), "Missing run_ablation_suite() in sparse_mla_decode.py"
        suite = mod.run_ablation_suite()
        assert isinstance(suite, dict), f"run_ablation_suite() must return a dict keyed by arm name, got {type(suite).__name__}"
        for req_arm in ("full_candidate", "minus_q_fusion", "minus_view_alias", "minus_device_guard"):
            assert req_arm in suite, f"Missing required leave-one-out arm: {req_arm}"

        full = _arm(suite["full_candidate"])
        mq = _arm(suite["minus_q_fusion"])
        mv = _arm(suite["minus_view_alias"])
        mg = _arm(suite["minus_device_guard"])

        full_json = json.dumps(suite["full_candidate"])
        mg_json = json.dumps(suite["minus_device_guard"])
        full_ok = (
            full.get("correctness_ok") is True
            or full.get("no_invalid_gm_address") is True
            or full.get("completed") is True
            or full.get("verdict") in ("supported", "keep")
        ) and "AICore invalid GM address" not in full_json
        assert full["tensor_move_count"] == 0 and full["host_d2h_sync_count"] == 0 and full_ok, f"Invalid full_candidate metrics: {full}"
        assert mq["tensor_move_count"] >= 3 and mq["tpot_ms"] > full["tpot_ms"], f"Invalid minus_q_fusion metrics: {mq}"
        assert mv["tensor_move_count"] >= 2 and mv["tpot_ms"] > full["tpot_ms"], f"Invalid minus_view_alias metrics: {mv}"
        mg_fail = (
            mg.get("invalid_gm_triggered") is True
            or mg.get("correctness_ok") is False
            or mg.get("no_invalid_gm_address") is False
            or mg.get("completed") is False
            or "invalid GM address" in mg_json
            or "negative block id" in mg_json
        )
        assert mg["host_d2h_sync_count"] >= 1 and mg_fail, f"Invalid minus_device_guard metrics: {mg}"

    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
