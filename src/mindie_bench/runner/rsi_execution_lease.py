"""RSI coordinator-lease admission, ownership verification and cleanup.

[Role]
Turn an execution authority into one verified lease while the calling executor
still owns the local reference, including exact cleanup when verification fails
after submission.

[Invariants]
- A submitted execution is never abandoned because assignment to the caller's
  local lease variable did not complete.
- A command that completed during verification switches to terminal collection
  on the same execution ID; it is never re-submitted.
- Release requires observed release evidence. A missing or failed observation is
  a cleanup failure, not a successful release.

[Timeouts & Locks]
- Admission and cleanup budgets are supplied by the caller/authority; this
  module adds no waits or local locks and performs no retries.

[Side-Effect Policy]
- Delegates acquisition and release to the supplied authority. It writes no
  files and never touches a device.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mindie_bench.runner.rsi_measurement import (
    _TERMINAL_EXECUTION_STATES,
    RSIExecutionError,
    RSIExecutionUnavailable,
    _receipt,
    _receipt_execution_id,
    _result_failure,
)


def _authority_contract(authority: Any) -> dict[str, Any]:
    """Require an independently verifiable authority, not a receipt callback."""
    if authority is None:
        return {}
    validate = getattr(authority, "validate_result", None)
    if not callable(validate):
        raise RSIExecutionUnavailable(
            "execution authority has no independent validate_result contract"
        )
    evidence_class = getattr(authority, "evidence_class", None)
    if not isinstance(evidence_class, str) or not evidence_class.strip():
        raise RSIExecutionUnavailable(
            "execution authority does not declare an evidence class"
        )
    if getattr(authority, "physical_acceptance", False) is True:
        raise RSIExecutionUnavailable(
            "an execution authority cannot declare physical acceptance; "
            "the parent reviewer owns that determination"
        )
    return {
        "mode": str(getattr(authority, "mode", evidence_class)),
        "evidence_class": evidence_class.strip(),
        "production_adapter": bool(getattr(authority, "production_adapter", False)),
        "physical_acceptance": False,
        "verification": "independent validate_result",
    }


def _expected_run_id(lease: Any) -> str:
    value = getattr(lease, "execution_id", None)
    if isinstance(value, str) and value:
        return value
    try:
        receipt = _receipt(lease)
    except RSIExecutionError:
        return ""
    return _receipt_execution_id(receipt)


def _lease_state(lease: Any) -> str:
    return str(_receipt(lease).get("state") or "").strip().lower()


def _lease_state_now(lease: Any) -> str:
    """Read current coordinator state without asserting that it is active."""
    refresh = getattr(lease, "refresh_state", None)
    if callable(refresh):
        return str(refresh() or "").strip().lower()
    return _lease_state(lease)


def _is_terminal_transition(lease: Any, exc: BaseException) -> bool:
    """Distinguish a completed command from an unverifiable active lease."""
    details = getattr(exc, "details", None)
    if not isinstance(details, Mapping):
        return False
    state = str(details.get("state") or "").strip().lower()
    if details.get("collectable") is not True and state not in _TERMINAL_EXECUTION_STATES:
        return False
    try:
        return _lease_state_now(lease) in _TERMINAL_EXECUTION_STATES
    except Exception:
        return False


def _cleanup_after_acquire(authority: Any, lease: Any, identity: str) -> dict[str, Any]:
    """Release a submitted lease that never reached the caller's local variable."""
    try:
        reply = _release(authority, lease)
    except Exception as exc:
        cleanup = dict(getattr(exc, "cleanup", {}) or {})
        cleanup.setdefault("execution_id", identity)
        cleanup.setdefault("released", False)
        cleanup.setdefault("cleanup_error", _result_failure(exc))
        return cleanup
    return {
        "execution_id": str(reply.get("execution_id") or identity),
        "released": (reply.get("released") is True
                     or reply.get("resources_released") is True
                     or str(reply.get("state") or "").lower() == "released"),
        "state": reply.get("state"),
        "resources_released": reply.get("resources_released") is True,
    }


def _attach_cleanup(exc: BaseException, cleanup: Mapping[str, Any], identity: str) -> None:
    current = dict(getattr(exc, "cleanup", {}) or {})
    current.update({key: value for key, value in cleanup.items() if value is not None})
    exc.cleanup = current
    if not getattr(exc, "execution_id", ""):
        exc.execution_id = identity


def _acquire(authority: Any, devices: Sequence[int], *, phase: str,
             candidate: Mapping[str, Any], timeout_seconds: float) -> Any:
    _authority_contract(authority)
    acquire = getattr(authority, "acquire", None)
    if not callable(acquire):
        raise RSIExecutionUnavailable("execution authority has no acquire method")
    lease = acquire(list(devices), phase=phase, candidate=candidate,
                    timeout_seconds=timeout_seconds)
    identity = ""
    try:
        identity = _expected_run_id(lease)
        if not identity:
            raise RSIExecutionError("coordinator lease has no observed execution identity")
        state = _lease_state_now(lease)
        if state == "running":
            if not getattr(lease, "verified_active", False):
                verify = getattr(lease, "verify", None)
                if not callable(verify):
                    raise RSIExecutionUnavailable(
                        f"coordinator lease {identity} cannot verify its active ownership"
                    )
                try:
                    verified = verify()
                except Exception as exc:
                    if _is_terminal_transition(lease, exc):
                        # The command completed after admission. Keep the same
                        # execution for exact collection instead of losing it.
                        return lease
                    raise
                if not verified:
                    raise RSIExecutionError(
                        f"coordinator lease {identity} did not verify as active",
                        execution_id=identity,
                    )
        elif state not in _TERMINAL_EXECUTION_STATES:
            raise RSIExecutionUnavailable(
                f"coordinator lease {identity} is in non-active state {state!r}",
                execution_id=identity,
                details={"state": state},
            )
        return lease
    except Exception as exc:
        cleanup = _cleanup_after_acquire(authority, lease, identity)
        _attach_cleanup(exc, cleanup, identity)
        raise


def _release(authority: Any, lease: Any) -> Mapping[str, Any]:
    release = getattr(authority, "release", None)
    if callable(release):
        value = release(lease)
    else:
        value = lease.release()
    if not isinstance(value, Mapping):
        value = _receipt(lease)
    released = (
        value.get("released") is True
        or value.get("resources_released") is True
        or str(value.get("state") or "").lower() == "released"
    )
    if not released:
        cleanup = value.get("cleanup")
        if isinstance(cleanup, Mapping) and any(
            cleanup.get(key) for key in ("status_error", "stop_error", "release_error")
        ):
            raise RSIExecutionError(
                "coordinator cleanup reported an error",
                execution_id=str(value.get("execution_id") or _expected_run_id(lease)),
                cleanup=dict(cleanup),
            )
        raise RSIExecutionError(
            "coordinator release did not confirm resource release",
            execution_id=str(value.get("execution_id") or _expected_run_id(lease)),
            cleanup=dict(value),
        )
    return dict(value)


__all__ = ["_acquire", "_authority_contract", "_expected_run_id", "_lease_state", "_release"]
