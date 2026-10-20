"""[Role] One conservative delivery budget for product trials from bounded owners.
[Invariants] The total covers execution plus every sequential settlement,
verifier and cleanup branch; preparation spends the execution allowance; a
measured overrun is reported without hiding time. The same predeclared total is
consumed by every arm and is never recomputed from the arm's actual services.
[Timeouts & Locks] Pure arithmetic over process, gateway, remote_slot,
component_runtime, knowledge_broker and native_product bounds; no waits.
[Side-Effect Policy] No I/O. Callers persist the returned terms.
"""

from __future__ import annotations

from mindie_bench.runner.gateway import MODEL_SETTLEMENT_GRACE_SEC

# runner/process.py: wait 2s, group cleanup wait 5s, two reader joins 1s each.
AGENT_TERMINATION_SEC = 9.0
# runner/gateway.py: model settlement wait plus the 0.25s post-close drain.
MODEL_SETTLEMENT_DRAIN_SEC = 0.25
# runner/gateway.py __exit__: server thread join.
GATEWAY_JOIN_SEC = 2.0
# runner/remote_slot.py: quiesce request budget.
REMOTE_QUIESCE_SEC = 30.0
# runner/remote_slot.py execute: verifier host wait is verifier limit + 15s.
VERIFIER_REMOTE_HOST_GRACE_SEC = 15.0
# runner/remote_slot.py close: 40s receipt + 5s wait + 5s kill/reap.
REMOTE_CLOSE_IO_SEC = 50.0
# integrations/component_runtime.py: 60s call + 5s terminate + 5s kill/reap.
COMPONENT_CALL_SEC = 70.0
# component_runtime.stop_monitor joins for 75s (one in-flight call plus margin).
COMPONENT_MONITOR_STOP_SEC = 75.0
# component_runtime.finish healthy branch: observe 3, release 2, diagnostic 1.
COMPONENT_FINISH_OBSERVE_CALLS = 3
COMPONENT_FINISH_RELEASE_CALLS = 2
COMPONENT_FINISH_DIAGNOSTIC_CALLS = 1
# remote.verify observes verifier_before on remote-dev, npu-top and coordinator.
REMOTE_VERIFIER_OBSERVE_CALLS = 3
# knowledge_broker: at most 8 sessions, each 5s EOF drain + 5s reap + 0.05s exchange.
KNOWLEDGE_SESSION_CAP = 8
KNOWLEDGE_SESSION_CLOSE_SEC = 10.05
# native_product.py: SIGTERM 5s + SIGKILL/reap 5s.
NATIVE_SERVICE_STOP_SEC = 10.0
# Capture gateway: finish_inference plus __exit__ finish_remote.
CAPTURE_SETTLEMENT_SEC = MODEL_SETTLEMENT_GRACE_SEC + MODEL_SETTLEMENT_DRAIN_SEC + MODEL_SETTLEMENT_GRACE_SEC
CONTRACT = "phase-budget/2"


def product_phase_budget(
    execution_sec: float,
    verifier_sec: float,
    *,
    remote: bool,
    native_product: bool,
    knowledge_sessions: int = 0,
    capture_timeout_sec: float | None = None,
) -> dict:
    """Return the conservative total and every sequential post-agent term.

    `execution_sec` includes preparation and agent execution. The reserve is a
    property of the predeclared cohort, not of the arm that happens to execute.
    """
    for name, value in (("execution_sec", execution_sec), ("verifier_sec", verifier_sec)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if isinstance(knowledge_sessions, bool) or not isinstance(knowledge_sessions, int):
        raise ValueError("knowledge_sessions must be an integer")
    if not 0 <= knowledge_sessions <= KNOWLEDGE_SESSION_CAP:
        raise ValueError(f"knowledge_sessions must be within 0..{KNOWLEDGE_SESSION_CAP}")
    terms = {
        "execution_including_preparation": float(execution_sec),
        "gateway_model_settlement": MODEL_SETTLEMENT_GRACE_SEC + MODEL_SETTLEMENT_DRAIN_SEC,
        "gateway_remote_settlement": MODEL_SETTLEMENT_GRACE_SEC,
        "agent_termination": AGENT_TERMINATION_SEC,
        "gateway_join": GATEWAY_JOIN_SEC,
        "verifier": float(verifier_sec),
    }
    if remote:
        terms.update(
            remote_quiesce=REMOTE_QUIESCE_SEC,
            verifier_remote_host_grace=VERIFIER_REMOTE_HOST_GRACE_SEC,
            verifier_before_observe=REMOTE_VERIFIER_OBSERVE_CALLS * COMPONENT_CALL_SEC,
            remote_close_io=REMOTE_CLOSE_IO_SEC,
            component_monitor_stop=COMPONENT_MONITOR_STOP_SEC,
            component_finish_observe=COMPONENT_FINISH_OBSERVE_CALLS * COMPONENT_CALL_SEC,
            component_finish_release=COMPONENT_FINISH_RELEASE_CALLS * COMPONENT_CALL_SEC,
            component_finish_diagnostic=COMPONENT_FINISH_DIAGNOSTIC_CALLS * COMPONENT_CALL_SEC,
        )
    if knowledge_sessions:
        terms["knowledge_close"] = knowledge_sessions * KNOWLEDGE_SESSION_CLOSE_SEC
    if native_product:
        terms["native_service_stop"] = NATIVE_SERVICE_STOP_SEC
    if capture_timeout_sec is not None:
        if (
            isinstance(capture_timeout_sec, bool)
            or not isinstance(capture_timeout_sec, (int, float))
            or capture_timeout_sec <= 0
        ):
            raise ValueError("capture_timeout_sec must be positive when provided")
        terms["capture"] = float(capture_timeout_sec)
        terms["capture_settlement"] = CAPTURE_SETTLEMENT_SEC
    post_agent = sum(value for name, value in terms.items() if name != "execution_including_preparation")
    return {
        "contract": CONTRACT,
        "execution_allowance_sec": float(execution_sec),
        "post_agent_reserve_sec": post_agent,
        "total_delivery_sec": float(execution_sec) + post_agent,
        "terms": terms,
    }


def lifecycle_overrun(declared_total_sec: float, measured_total_sec: float) -> dict:
    """Report the measured lifecycle against the predeclared total."""
    for name, value in (
        ("declared_total_sec", declared_total_sec),
        ("measured_total_sec", measured_total_sec),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    overrun = max(0.0, float(measured_total_sec) - float(declared_total_sec))
    return {
        "declared_total_sec": float(declared_total_sec),
        "measured_total_sec": float(measured_total_sec),
        "overrun_sec": overrun,
        "overrun": overrun > 0,
    }
