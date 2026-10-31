# [Role] Independent behavioral oracle for a controlled task workspace.
# [Invariants] Rewards require executed input/output checks; source shape is never a success criterion.
# [Timeouts & Locks] The runner bounds this subprocess; no external service calls.
# [Side-Effect Policy] Candidate imports execute only in the supplied verification workspace.
import importlib.util
import json
import os
import sys
import tempfile
import traceback
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
    assert any(
        k in str(t["fault_domain"]).lower() for k in ("device", "driver", "hbm", "oom", "hardware")
    ), f"bad fault_domain: {t}"
    assert t["error_codes"][0] == "E89999", f"bad error_codes: {t}"
    assert "HBM ECC" in t["root_cause_line"], f"bad root_cause_line: {t}"
    assert t["root_cause_rank"] == "0", f"bad root_cause_rank: {t}"
    assert t["device_id"] == "0", f"bad device_id: {t}"
    assert t["failed_operator"] == "MatMulV2", f"bad failed_operator: {t}"
    assert any(
        k in str(t["decoded_codes"][0]["subsystem"]).lower()
        for k in ("driver", "memory", "hbm", "device")
    ), f"bad decoded_codes: {t}"
    assert t["cascading_timeout_detected"] is True, f"missing cascading_timeout_detected: {t}"
    assert t["cascading_rank"] == "1", f"bad cascading_rank: {t}"
    assert len(t["cascading_timeout_lines"]) == 2, f"bad cascading_timeout_lines: {t}"
    assert any(
        "npu-smi" in str(x) or "HCCL_ENTRY_LOG_ENABLE" in str(x) for x in t["recommended_tools"]
    ), f"bad tools: {t}"
    assert res["remaining_secrets"] == [], f"leaked secrets: {res['remaining_secrets']}"
    assert res["scan_finding_count"] == 0, f"non-zero scan findings: {res['scan_finding_count']}"
    assert "torch_npu" in res["sanitized_log"] and "MatMulV2" in res["sanitized_log"]
    oa = res["outbox_audit"]
    assert oa["state_after_timeout"] == "uncertain", f"bad state_after_timeout: {oa}"
    assert oa["occurrences_after_duplicate"] == 2, f"bad occurrences: {oa}"
    assert len(oa["incident_ids"]) == 2, f"bad incident_ids: {oa}"
    assert oa["blind_retry_prevented"] is True, f"blind_retry_prevented must be True: {oa}"
    # Replay another fault identity and a completed POST. A constant result must fail.
    original = json.loads((root / "vllm_ascend_diag/fixtures/diagnostics_case.json").read_text())
    for driver_rank, cascade_rank, device in ((3, 7, 5), (2, 6, 4)):
        plog = original["hccl_hbm_plog"].replace("rank[0]", f"rank[{driver_rank}]")
        plog = plog.replace("rank[1]", f"rank[{cascade_rank}]")
        plog = plog.replace("deviceId:0", f"deviceId:{device}").replace("MatMulV2", "Add")
        changed = mod.reconcile_plog_and_outbox(plog_override=plog, simulate_post_timeout=False)
        triage = changed["triage"]
        assert triage["root_cause_rank"] == str(driver_rank), f"root rank must follow input: {triage}"
        assert triage["cascading_rank"] == str(cascade_rank), f"cascade rank must follow input: {triage}"
        assert triage["device_id"] == str(device), f"device must follow input: {triage}"
        assert triage["failed_operator"] == "Add", f"operator must follow input: {triage}"
        assert changed["outbox_audit"]["state_after_timeout"] == "published", (
            "successful POST must remain published"
        )
    with tempfile.TemporaryDirectory(prefix="mindie-diagnostics-oracle-") as directory:
        fixture = json.loads(json.dumps(original))
        fixture["incident"]["operation_id_1"] = "a182c7e9fb10436d8ab53f2057e6c941"
        fixture["incident"]["operation_id_2"] = "b293d8fa0c21547e9bc6403168f7da52"
        fixture["redaction_line"] = fixture["redaction_line"].replace(
            "abcDEF1234567890token", "anotherSecret987654321"
        )
        fixture["planted_secrets"][0] = "anotherSecret987654321"
        fixture_path = Path(directory) / "case.json"
        fixture_path.write_text(json.dumps(fixture))
        changed = mod.reconcile_plog_and_outbox(fixture_path=fixture_path)
        assert changed["outbox_audit"]["incident_ids"] == [
            "a182c7e9fb10436d8ab53f2057e6c941",
            "b293d8fa0c21547e9bc6403168f7da52",
        ], "valid input incident IDs must both survive duplicate enqueue"
        assert "anotherSecret987654321" not in changed["sanitized_log"], (
            "changed Bearer credential leaked"
        )
        assert changed["remaining_secrets"] == [], "changed fixture secret was not redacted"
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
    traceback.print_exc()
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc), "failure_type": type(exc).__name__}))
    raise SystemExit(1)
