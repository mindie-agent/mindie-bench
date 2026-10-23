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
    mod = load_mod("plog_fault_reconciler", root / "vllm_ascend_diag/plog_fault_reconciler.py")
    res = mod.reconcile_plog_and_outbox(simulate_post_timeout=True)
    t = res["triage"]
    assert t["fault_domain"] == "device_oom_or_driver", f"bad fault_domain: {t}"
    assert t["error_codes"][0] == "E89999", f"bad error_codes: {t}"
    assert "HBM ECC" in t["root_cause_line"], f"bad root_cause_line: {t}"
    assert t["root_cause_rank"] == "0", f"bad root_cause_rank: {t}"
    assert t["device_id"] == "0", f"bad device_id: {t}"
    assert t["failed_operator"] == "MatMulV2", f"bad failed_operator: {t}"
    assert t["decoded_codes"][0]["subsystem"] == "driver_memory", f"bad decoded_codes: {t}"
    assert t["cascading_timeout_detected"] is True, f"missing cascading_timeout_detected: {t}"
    assert t["cascading_rank"] == "1", f"bad cascading_rank: {t}"
    assert len(t["cascading_timeout_lines"]) == 2, f"bad cascading_timeout_lines: {t}"
    assert "npu-smi" in t["recommended_tools"] and "HCCL_ENTRY_LOG_ENABLE=1" in t["recommended_tools"], f"bad tools: {t}"
    assert res["remaining_secrets"] == [], f"leaked secrets: {res['remaining_secrets']}"
    assert res["scan_finding_count"] == 0, f"non-zero scan findings: {res['scan_finding_count']}"
    assert "torch_npu" in res["sanitized_log"] and "MatMulV2" in res["sanitized_log"]
    oa = res["outbox_audit"]
    assert oa["state_after_timeout"] == "uncertain", f"bad state_after_timeout: {oa}"
    assert oa["occurrences_after_duplicate"] == 2, f"bad occurrences: {oa}"
    assert len(oa["incident_ids"]) == 2, f"bad incident_ids: {oa}"
    assert oa["blind_retry_prevented"] is True, f"blind_retry_prevented must be True: {oa}"
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
