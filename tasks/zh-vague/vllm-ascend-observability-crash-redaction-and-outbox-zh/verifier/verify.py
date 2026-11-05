# [Role] Independent CPU behavior oracle for crash log redaction & outbox state honesty (Stream C - DG1).
# [Invariants] Only executed behavioral assertions establish success; no NPU hardware required.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources and workspace log; emits a JSON verdict only.

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
TARGET_REL = "vllm_ascend/logger.py"
ALLOWED_FILES = {TARGET_REL}


def check_diff_scope() -> None:
    if not (ROOT / ".git").exists():
        return
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    disallowed = {p for p in changed if p not in ALLOWED_FILES and not p.startswith("tests/")}
    if disallowed:
        raise AssertionError(f"Modified files outside allowed scope {sorted(ALLOWED_FILES)}: {sorted(disallowed)}")


def load_logger_redaction_and_outbox():
    path = ROOT / TARGET_REL
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if not any(name in {"DEFAULT_LOGGING_CONFIG"} for name in targets):
                wanted_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name != "_configure_ascend_logger":
            wanted_nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "CrashDiagnosticOutbox":
            wanted_nodes.append(node)

    ns: dict[str, Any] = {
        "Any": Any,
        "Callable": Callable,
        "re": re,
        "os": os,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod_ast = ast.Module(body=[future, *wanted_nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod_ast), str(path), "exec"), ns)
    if "redact_crash_diagnostic_text" not in ns or "CrashDiagnosticOutbox" not in ns:
        raise AssertionError(
            f"{TARGET_REL} must define redact_crash_diagnostic_text and CrashDiagnosticOutbox"
        )
    return ns["redact_crash_diagnostic_text"], ns["CrashDiagnosticOutbox"]


def verify_redaction(redact_fn) -> None:
    sample_log = (
        "Worker PID: 48291 | Source: /Users/maoxx241/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:412\n"
        "Coordinator: ssh://root@177.31.14.88:22 | Peer: hwadmin@192.168.13.165\n"
        "HCCL Master: 173.125.1.2:29500 | Node IP: 192.168.13.156\n"
        "Auth: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ascend_secret_payload\n"
        "Key: sk-live-9f8e7d6c5b4a3210fedcba9876543210 | Env: ASCEND_LAUNCH_TOKEN=tok_ascend_prod_778899\n"
        "Host Log: /home/ascend_ops/ascend/plog/plog-48291.log\n"
        "[ERROR] RUNTIME retCode=507015 | HCCL error code EI0006 | ASCENDCL error code EZ9999 in vllm_ascend/ops/mla.py:128\n"
    )
    scrubbed = redact_fn(sample_log)

    leaked_tokens = [
        "177.31.14.88",
        "192.168.13.165",
        "173.125.1.2",
        "192.168.13.156",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "sk-live-9f8e7d6c5b4a3210fedcba9876543210",
        "tok_ascend_prod_778899",
        "/Users/maoxx241",
        "/home/ascend_ops",
    ]
    for secret in leaked_tokens:
        assert secret not in scrubbed, f"Redaction leaked sensitive value '{secret}' in output: {scrubbed}"

    for placeholder in ("<REDACTED_SSH>", "<REDACTED_IP>", "<REDACTED_TOKEN>", "<REDACTED_HOME>"):
        assert placeholder in scrubbed, f"Expected placeholder {placeholder} in redacted text"

    preserved = [
        "507015",
        "EI0006",
        "EZ9999",
        "vllm_ascend/worker/model_runner_v1.py:412",
        "vllm_ascend/ops/mla.py:128",
    ]
    for token in preserved:
        assert token in scrubbed, f"Redaction corrupted required diagnostic identifier '{token}'"


def verify_outbox_state_machine(outbox_cls) -> None:
    outbox = outbox_cls()
    enq = outbox.enqueue("crash-001", "Node 192.168.13.165 token sk-secret123456 error 507015")
    assert "192.168.13.165" not in enq["payload"] and "sk-secret123456" not in enq["payload"]
    assert "507015" in enq["payload"]
    assert enq["status"] == "pending" and enq["receipt_unconfirmed"] is False

    calls = []

    def timeout_sink(payload: str) -> str:
        calls.append(payload)
        raise TimeoutError("sink gateway timed out after request body was transmitted")

    after_timeout = outbox.flush_entry("crash-001", timeout_sink)
    assert after_timeout["status"] == "uncertain", (
        f"TimeoutError must transition outbox entry to 'uncertain', got {after_timeout['status']}"
    )
    assert after_timeout["receipt_unconfirmed"] is True, (
        "TimeoutError must mark receipt_unconfirmed=True"
    )
    assert after_timeout["attempts"] == 1 and len(calls) == 1

    # Automatic retry or flush_pending MUST NOT blindly replay an uncertain write
    replayed = outbox.flush_pending(lambda p: "ok-receipt")
    assert replayed == [], f"flush_pending must not replay uncertain entries, got {replayed}"
    assert outbox.get_entry("crash-001")["attempts"] == 1

    blocked = False
    try:
        outbox.flush_entry("crash-001", lambda p: "ok-receipt")
    except RuntimeError:
        blocked = True
    assert blocked, "flush_entry on an 'uncertain' entry without allow_uncertain_replay=True must raise RuntimeError"

    # Non-timeout OSError (e.g. ConnectionRefusedError) transitions to 'failed' and CAN be retried
    outbox.enqueue("crash-002", "CANN EZ9999 on /home/tester/work/vllm_ascend/ops/mla.py:40")

    def conn_refused_sink(payload: str) -> str:
        raise ConnectionRefusedError("local collector socket not listening")

    after_conn_err = outbox.flush_entry("crash-002", conn_refused_sink)
    assert after_conn_err["status"] in {"failed", "pending"} and after_conn_err["receipt_unconfirmed"] is False

    flushed = outbox.flush_pending(lambda p: "receipt-002")
    assert len(flushed) == 1 and flushed[0]["status"] == "sent" and flushed[0]["receipt"] == "receipt-002"


def main() -> None:
    try:
        check_diff_scope()
        redact_fn, outbox_cls = load_logger_redaction_and_outbox()
        verify_redaction(redact_fn)
        verify_outbox_state_machine(outbox_cls)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "correctness_verified": False,
                    "error_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "validation_scope": "cpu_behavior",
                    "performance_status": "unmeasured",
                }
            )
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "cpu_behavior",
                "performance_status": "unmeasured",
                "native_kernel_bound": None,
                "measured_npu_speedup": None,
                "speedup_ratio": None,
            }
        )
    )


if __name__ == "__main__":
    main()
