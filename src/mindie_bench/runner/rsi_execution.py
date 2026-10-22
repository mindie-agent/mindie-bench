# [Side-Effect Policy] Writes only explicitly owned trial artifacts; no network or unowned mutation.

"""RSI compile/timing execution with real coordinator-lease admission.

[Role]
Execute RSI candidates: CPU-only compiles may run concurrently, while every
device compile/JIT and every timing run must hold a coordinator lease while its
callback runs. A completed coordinator execution may have its exact result
collected, but it cannot authorize a later arbitrary callback.

[Invariants]
- Timing and device compile require a verifiable authority contract with an
  independent `validate_result`; a receipt-only callback dictionary is never
  authority. Production TaskClient evidence is marked separately from test or
  external lease contracts, and physical acceptance is never inferred.
- A running lease authorizes a callback. A terminal execution may only be
  collected through the authority's exact-execution collector.
- A timing result must carry the run ID of the admitted lease. Missing run,
  requested workspace or guard evidence is invalid, never success.
- CPU-only compiles receive `npu_count == 0` and may overlap; timing is serial.
  A CPU compile timeout is bounded and visible; a device-compile timeout blocks
  later timing because cleanup is unknown.
- Candidate failures remain visible. A valid measurement is not erased merely
  because local release failed; cleanup failure is reported separately, blocks
  later NPU timing, and never sets `released` without observed release evidence.

[Timeouts & Locks]
- Lease admission is bounded per call (0..600s). CPU compile concurrency is
  bounded by `max_compile_slots`. The default outer compile timeout is derived
  from the production adapter's admission + command + release/verification +
  bounded cleanup budget plus grace, and an undersized explicit outer timeout is
  rejected. No local lockfile or shadow scheduler exists; the coordinator owns
  device locks.

[Side-Effect Policy]
- Delegates coordinator effects to the supplied authority/adapter. Ledger and
  progress writes occur only when both paths are explicitly supplied.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from mindie_bench.runner.rsi_execution_lease import (
    _acquire,
    _authority_contract,
    _expected_run_id,
    _lease_state,
    _release,
)
from mindie_bench.runner.rsi_explorer import (
    _candidate_identity,
    _compile_proved,
    append_ledger_and_render_progress,
    evaluate_rsi_candidate_gates,
    plan_rsi_compile_and_timing,
)
from mindie_bench.runner.rsi_measurement import (
    _TERMINAL_EXECUTION_STATES,
    RSIExecutionError,
    RSIExecutionUnavailable,
    _compile_requires_device,
    _measurement_problems,
    _merge_result,
    _reason,
    _receipt,
    _receipt_execution_id,
    _required_guards,
    _result_failure,
)
from mindie_bench.runner.rsi_taskclient import TaskClientLease, TaskClientRSIExecutor

_COMPILE_RESULT_FIELDS = ("compile_ok", "compile_proof", "run_id", "oracle_passed",
                          "failure_reason")
_TIMING_RESULT_FIELDS = (
    "run_id", "oracle_passed", "measurement_passed", "speedup_ratio",
    "ablation_retained", "workspace_delta_bytes", "guard_workloads",
    "failure_reason", "coordinator_execution",
)
_TIMING_EVIDENCE_FIELDS = (
    "run_id", "measurement_passed", "speedup_ratio", "ablation_retained",
    "workspace_delta_bytes", "guard_workloads",
)
_COMPILE_TIMEOUT_REASON = "compile_callback_timeout"
_DEFAULT_COMPILE_TIMEOUT_SECONDS = 1800.0
_COMPILE_BUDGET_GRACE_SECONDS = 10.0


def _compile_timeout_floor(authority: Any, lease_timeout_seconds: float) -> float | None:
    """Return the production inner budget plus outer cleanup grace."""
    budget = getattr(authority, "compile_wait_budget_seconds", None)
    if not callable(budget):
        return None
    value = budget(lease_timeout_seconds=lease_timeout_seconds)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise RSIExecutionUnavailable(
            "execution authority reported an invalid compile wait budget"
        )
    return float(value) + _COMPILE_BUDGET_GRACE_SECONDS


def _collect_completed(authority: Any, lease: Any, *, phase: str,
                       candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Collect an already-terminal execution; never invoke a callback for it."""
    collector = getattr(authority, "collect_completed", None)
    if not callable(collector):
        raise RSIExecutionUnavailable(
            f"completed execution {_expected_run_id(lease)} cannot authorize a "
            "later callback and the authority provides no exact result collector",
            execution_id=_expected_run_id(lease),
        )
    result = collector(lease, phase=phase, candidate=candidate)
    if not isinstance(result, Mapping):
        raise RSIExecutionError("authority completed-result collector returned no object")
    return result


def _validate(authority: Any, lease: Any, result: Mapping[str, Any], *,
              phase: str, devices: Sequence[int],
              expected_run_id: str | None = None) -> None:
    _authority_contract(authority)
    validate = getattr(authority, "validate_result", None)
    if not callable(validate):
        raise RSIExecutionUnavailable("execution authority has no independent verifier")
    validate(lease, result, phase=phase, devices=list(devices))
    expected = expected_run_id if expected_run_id is not None else _expected_run_id(lease)
    observed = str(result.get("run_id") or "").strip()
    if not observed:
        raise RSIExecutionError("callback result has no observed run_id")
    if expected and observed != expected:
        raise RSIExecutionError(
            f"callback run_id {observed!r} differs from coordinator lease {expected!r}"
        )


def _cleanup_detail(exc: BaseException, phase: str) -> dict[str, Any]:
    cleanup = getattr(exc, "cleanup", None)
    return {
        "phase": phase,
        "execution_id": str(getattr(exc, "execution_id", "") or ""),
        "error": _result_failure(exc),
        "cleanup": dict(cleanup) if isinstance(cleanup, Mapping) else {},
    }


def _record_cleanup(candidate: dict[str, Any], detail: Mapping[str, Any]) -> None:
    candidate.setdefault("cleanup_failures", []).append(dict(detail))
    candidate["cleanup_ok"] = False


def _compile_one(
    candidate: dict[str, Any],
    *,
    compile_runner: Any,
    authority: Any,
    devices: Sequence[int],
    lease_timeout_seconds: float,
    cancel_event: threading.Event | None = None,
) -> bool:
    """Return whether any lease cleanup remained healthy."""
    if _compile_proved(candidate) == "passed" or candidate.get("compile_ok") is False:
        return True
    if compile_runner is None:
        return True
    requires_device = _compile_requires_device(candidate)
    reasoning: list[str] = []
    lease = None
    cleanup_ok = True
    compile_execution_id = ""
    try:
        if requires_device:
            if authority is None:
                raise RSIExecutionUnavailable(
                    "device compilation requires a coordinator lease; none was supplied"
                )
            lease = _acquire(authority, devices, phase="compile", candidate=candidate,
                             timeout_seconds=lease_timeout_seconds)
            compile_execution_id = _expected_run_id(lease)
        compile_input = {
            **candidate, "phase": "compile", "device_compile": requires_device,
            "devices": list(devices) if requires_device else [],
            "assigned_devices": list(devices) if requires_device else [],
            "npu_device": devices[0] if requires_device else None,
        }
        if lease is not None:
            compile_input["coordinator_lease"] = lease
            compile_input["coordinator_lease_receipt"] = _receipt(lease)
            if _lease_state(lease) in _TERMINAL_EXECUTION_STATES:
                result = _collect_completed(
                    authority, lease, phase="compile", candidate=candidate,
                )
            else:
                result = compile_runner(compile_input)
        else:
            result = compile_runner(compile_input)
        if cancel_event is not None and cancel_event.is_set():
            raise RSIExecutionUnavailable(_COMPILE_TIMEOUT_REASON)
        if not isinstance(result, Mapping):
            raise RSIExecutionError("compile callback returned no result object")
        if lease is not None:
            _validate(authority, lease, result, phase="compile", devices=devices,
                      expected_run_id=compile_execution_id)
        result_cleanup = result.get("coordinator_cleanup")
        if isinstance(result_cleanup, Mapping) and result_cleanup:
            candidate["attempted_cleanup"] = dict(result_cleanup)
            if result_cleanup.get("released") is not True:
                cleanup_ok = False
                _record_cleanup(candidate, {
                    "phase": "compile",
                    "execution_id": str(
                        result_cleanup.get("execution_id") or compile_execution_id
                    ),
                    "error": "coordinator_release_unconfirmed",
                    "cleanup": dict(result_cleanup),
                })
        _merge_result(candidate, result, _COMPILE_RESULT_FIELDS)
        if candidate.get("run_id"):
            candidate["compile_run_id"] = candidate["run_id"]
    except Exception as exc:
        reasoning.append(_result_failure(exc))
        failed_execution_id = str(getattr(exc, "execution_id", "") or "")
        if failed_execution_id:
            candidate["attempted_execution_id"] = failed_execution_id
        cleanup = getattr(exc, "cleanup", None)
        if isinstance(cleanup, Mapping) and cleanup:
            candidate["attempted_cleanup"] = dict(cleanup)
            if cleanup.get("status_error") or cleanup.get("stop_error") or cleanup.get("release_error"):
                reasoning.append(
                    "coordinator_cleanup_failed:"
                    + str(cleanup.get("stop_error") or cleanup.get("release_error")
                          or cleanup.get("status_error"))
                )
                cleanup_ok = False
            if cleanup.get("released") is not True:
                cleanup_ok = False
                _record_cleanup(candidate, _cleanup_detail(exc, "compile"))
        if candidate.get("compile_ok") is not False:
            candidate["compile_ok"] = False
    finally:
        if lease is not None:
            try:
                _release(authority, lease)
            except Exception as exc:
                cleanup_ok = False
                detail = _cleanup_detail(exc, "compile")
                _record_cleanup(candidate, detail)
                # A successful compile remains successful; cleanup is separate.
                if candidate.get("compile_ok") is not True:
                    candidate["compile_ok"] = False
                    reasoning.append(f"coordinator_cleanup_failed:{detail['error']}")
    if reasoning:
        candidate["failure_reason"] = _reason(candidate.get("failure_reason"), reasoning)
    return cleanup_ok


def _clear_prefilled_timing(candidate: dict[str, Any]) -> None:
    """Only the timing execution may supply timing evidence this call."""
    prefilled: dict[str, Any] = {}
    for key in _TIMING_EVIDENCE_FIELDS:
        if key not in candidate:
            continue
        value = candidate.pop(key)
        if key == "run_id" and value == candidate.get("compile_run_id"):
            continue
        prefilled[key] = value
    if prefilled:
        candidate["prefilled_timing_evidence"] = prefilled


def _reject_timing(candidate: dict[str, Any], reason: str) -> None:
    """Invalidate prefilled timing claims; never let caller data become a run."""
    candidate["measurement_passed"] = False
    prior = str(candidate.get("run_id") or "").strip()
    if prior and not isinstance(candidate.get("coordinator_lease_receipt"), Mapping):
        if str(candidate.get("compile_run_id") or "") != prior:
            candidate["declared_run_id"] = prior
        candidate["run_id"] = ""
    candidate["failure_reason"] = _reason(candidate.get("failure_reason"), [reason])


def _timing_one(
    candidate: dict[str, Any],
    *,
    timing_runner: Any,
    authority: Any,
    devices: Sequence[int],
    lease_timeout_seconds: float,
    timing_attempts: list[dict[str, Any]],
) -> bool:
    """Return whether any lease cleanup remained healthy."""
    declared_guards = _required_guards(candidate, None)
    if declared_guards:
        candidate["declared_required_guard_workloads"] = declared_guards
    _clear_prefilled_timing(candidate)
    if timing_runner is None:
        _reject_timing(candidate, "timing_runner_unavailable")
        timing_attempts.append({
            "candidate_id": candidate.get("candidate_id"),
            "acquired": False,
            "failure_reason": "timing_runner_unavailable",
        })
        return True
    if authority is None:
        _reject_timing(candidate, "coordinator_lease_unavailable")
        timing_attempts.append({
            "candidate_id": candidate.get("candidate_id"),
            "acquired": False,
            "failure_reason": "coordinator_lease_unavailable",
        })
        return True
    _authority_contract(authority)
    candidate_id = str(candidate.get("candidate_id") or "")
    lease = None
    reasoning: list[str] = []
    acquired = False
    cleanup_ok = True
    execution_id = ""
    collection_mode = ""
    try:
        lease = _acquire(authority, devices, phase="timing", candidate=candidate,
                         timeout_seconds=lease_timeout_seconds)
        acquired = True
        execution_id = _expected_run_id(lease)
        lease_receipt = _receipt(lease)
        expected = execution_id
        timing_input = {
            **candidate, "phase": "timing", "devices": list(devices),
            "assigned_devices": list(devices), "npu_device": devices[0],
            "coordinator_lease": lease,
            "coordinator_lease_receipt": lease_receipt,
        }
        if _lease_state(lease) in _TERMINAL_EXECUTION_STATES:
            # The command already completed. Collect its exact coordinator
            # result; do not let arbitrary callback code run after release.
            collection_mode = "completed_execution"
            result = _collect_completed(
                authority, lease, phase="timing", candidate=candidate,
            )
        else:
            result = timing_runner(timing_input)
        if not isinstance(result, Mapping):
            raise RSIExecutionError("timing callback returned no result object")
        _validate(authority, lease, result, phase="timing", devices=devices,
                  expected_run_id=expected)
        observed = str(result.get("run_id") or "").strip()
        if not observed:
            raise RSIExecutionError("timing result has no observed run_id")
        if expected and observed != expected:
            raise RSIExecutionError(
                f"timing run_id {observed!r} differs from coordinator lease {expected!r}"
            )
        _merge_result(candidate, result, _TIMING_RESULT_FIELDS)
        candidate["coordinator_lease_receipt"] = _receipt(lease)
        candidate["measurement_collection"] = collection_mode or "running_callback"
    except Exception as exc:
        reasoning.append(_result_failure(exc))
        execution_id = execution_id or str(getattr(exc, "execution_id", "") or "")
        if execution_id:
            candidate["attempted_execution_id"] = execution_id
        cleanup = getattr(exc, "cleanup", None)
        if isinstance(cleanup, Mapping) and cleanup:
            candidate["attempted_cleanup"] = dict(cleanup)
            if cleanup.get("status_error") or cleanup.get("stop_error") or cleanup.get("release_error"):
                reasoning.append(
                    "coordinator_cleanup_failed:"
                    + str(cleanup.get("stop_error") or cleanup.get("release_error")
                          or cleanup.get("status_error"))
                )
                cleanup_ok = False
            if cleanup.get("released") is not True:
                cleanup_ok = False
                detail = _cleanup_detail(exc, "timing")
                if not detail["execution_id"] and execution_id:
                    detail["execution_id"] = execution_id
                _record_cleanup(candidate, detail)
        _reject_timing(candidate, _result_failure(exc))
    finally:
        if lease is not None:
            try:
                released = _release(authority, lease)
                candidate["coordinator_release"] = dict(released) if released else {}
                # Only a successful release observation refreshes the receipt;
                # failed cleanup must keep resources_released false.
                try:
                    candidate["coordinator_lease_receipt"] = _receipt(lease)
                except RSIExecutionError:
                    pass
            except Exception as exc:
                cleanup_ok = False
                detail = _cleanup_detail(exc, "timing")
                if not detail["execution_id"] and execution_id:
                    detail["execution_id"] = execution_id
                _record_cleanup(candidate, detail)
                # Keep a validated remote measurement; cleanup is separate.
                if candidate.get("measurement_passed") is not True:
                    candidate["measurement_passed"] = False
                    reasoning.append(f"coordinator_cleanup_failed:{detail['error']}")
        timing_attempts.append({
            "candidate_id": candidate_id,
            "execution_id": execution_id or (candidate.get("attempted_execution_id") or ""),
            "acquired": acquired,
            "collection_mode": collection_mode,
            "released": "coordinator_release" in candidate,
            "cleanup_ok": cleanup_ok,
            "failure_reason": _reason(candidate.get("failure_reason"), reasoning),
        })
    if reasoning:
        candidate["failure_reason"] = _reason(candidate.get("failure_reason"), reasoning)
    return cleanup_ok


def _validate_options(max_compile_slots: int, max_workspace_delta_bytes: int | None,
                      lease_timeout_seconds: float, compile_timeout_seconds: float,
                      required_guard_workloads: Sequence[str] | None,
                      min_speedup_ratio: float) -> None:
    if type(max_compile_slots) is not int or max_compile_slots < 1:
        raise ValueError("max_compile_slots must be a positive integer")
    if max_workspace_delta_bytes is not None:
        if type(max_workspace_delta_bytes) is not int or max_workspace_delta_bytes < 0:
            raise ValueError("max_workspace_delta_bytes must be a non-negative integer")
    if (isinstance(lease_timeout_seconds, bool)
            or not isinstance(lease_timeout_seconds, (int, float))
            or not 0 <= lease_timeout_seconds <= 600):
        raise ValueError("lease_timeout_seconds must be between 0 and 600")
    if (isinstance(compile_timeout_seconds, bool)
            or not isinstance(compile_timeout_seconds, (int, float))
            or not 0 < compile_timeout_seconds <= 24 * 3600):
        raise ValueError("compile_timeout_seconds must be positive and bounded")
    minimum = float(min_speedup_ratio)
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("min_speedup_ratio must be finite and positive")
    if required_guard_workloads is not None:
        if isinstance(required_guard_workloads, (str, bytes)) or any(
            not isinstance(name, str) or not name.strip()
            for name in required_guard_workloads
        ):
            raise ValueError("required_guard_workloads must contain non-empty names")


def execute_rsi_compile_and_timing(
    candidates: Sequence[Mapping[str, Any]],
    *,
    compile_runner: Any = None,
    timing_runner: Any = None,
    execution_backend: Any = None,
    lease_authority: Any = None,
    state_dir: str | Path = "/tmp/mindie-rsi-coordinator",
    npu_device: int = 0,
    assigned_devices: Sequence[int] | None = None,
    max_compile_slots: int = 4,
    min_speedup_ratio: float = 1.05,
    max_workspace_delta_bytes: int | None = None,
    required_guard_workloads: Sequence[str] | None = None,
    lease_timeout_seconds: float = 300.0,
    compile_timeout_seconds: float | None = None,
    ledger_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    goal: str = "Optimize operator throughput",
) -> dict[str, Any]:
    """Run CPU compile waves, then serial coordinator-leased NPU timing.

    CPU-only compiles run concurrently without a device lease. Any candidate
    declaring device compilation, and every timing operation, is admitted
    through a real coordinator lease. A completed command may have its exact
    result collected; it never authorizes a later arbitrary callback.

    ``execution_backend`` is the production convenience form: it must provide
    ``compile_runner``, ``timing_runner``, ``acquire``/``release`` and an
    independent ``validate_result``. The separate
    ``compile_runner``/``timing_runner`` plus ``lease_authority`` form supports
    an existing verified lease (for example a remote slot).

    ``state_dir`` is retained as historical plan/ledger metadata. It is not a
    coordinator lock or a source of lease authority; the installed coordinator
    or an explicitly verified existing lease owns device exclusion.
    """
    if execution_backend is not None:
        if compile_runner is not None or timing_runner is not None or lease_authority is not None:
            raise ValueError(
                "execution_backend already provides runners and authority; "
                "do not also pass callbacks or lease_authority"
            )
        compile_runner = getattr(execution_backend, "compile_runner", None)
        timing_runner = getattr(execution_backend, "timing_runner", None)
        lease_authority = execution_backend
    compile_floor = (
        _compile_timeout_floor(lease_authority, lease_timeout_seconds)
        if lease_authority is not None else None
    )
    if compile_timeout_seconds is None:
        compile_timeout_seconds = (
            compile_floor if compile_floor is not None
            else _DEFAULT_COMPILE_TIMEOUT_SECONDS
        )
    elif (compile_floor is not None
          and not isinstance(compile_timeout_seconds, bool)
          and isinstance(compile_timeout_seconds, (int, float))
          and compile_timeout_seconds < compile_floor):
        raise ValueError(
            f"compile_timeout_seconds {compile_timeout_seconds:g}s is below the "
            f"production inner budget {compile_floor - _COMPILE_BUDGET_GRACE_SECONDS:g}s "
            f"plus {_COMPILE_BUDGET_GRACE_SECONDS:g}s grace"
        )
    _validate_options(max_compile_slots, max_workspace_delta_bytes,
                      lease_timeout_seconds, compile_timeout_seconds,
                      required_guard_workloads, min_speedup_ratio)
    authority_contract = _authority_contract(lease_authority) if lease_authority else {}
    records = [dict(candidate) for candidate in candidates]
    plan = plan_rsi_compile_and_timing(
        candidates, state_dir=state_dir, npu_device=npu_device,
        max_compile_slots=max_compile_slots, assigned_devices=assigned_devices,
    )
    by_id = {_candidate_identity(candidate, index): candidate
             for index, candidate in enumerate(records)}
    devices = list(plan["assigned_devices"])
    compile_order: list[str] = []
    compile_attempts: list[str] = []
    compile_timeouts: list[str] = []
    compile_cleanup_failures: list[dict[str, Any]] = []
    device_cleanup_unknown = False

    for wave in plan["phase1_compile"]["waves"]:
        runnable: list[str] = []
        for candidate_id in wave:
            candidate = by_id[candidate_id]
            if _compile_proved(candidate) == "passed" or candidate.get("compile_ok") is False:
                continue
            if compile_runner is None:
                continue
            runnable.append(candidate_id)
            compile_attempts.append(candidate_id)
        if not runnable:
            continue
        compile_order.extend(runnable)
        workers = min(max_compile_slots, len(runnable))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rsi-compile")
        events = {candidate_id: threading.Event() for candidate_id in runnable}
        deadline = time.monotonic() + compile_timeout_seconds
        futures = {
            candidate_id: pool.submit(
                _compile_one, by_id[candidate_id], compile_runner=compile_runner,
                authority=lease_authority, devices=devices,
                lease_timeout_seconds=lease_timeout_seconds,
                cancel_event=events[candidate_id],
            )
            for candidate_id in runnable
        }
        try:
            for candidate_id in runnable:
                remaining = deadline - time.monotonic()
                if remaining <= 0 and not futures[candidate_id].done():
                    events[candidate_id].set()
                    futures[candidate_id].cancel()
                    candidate = by_id[candidate_id]
                    candidate["compile_ok"] = False
                    candidate["compile_timeout_seconds"] = compile_timeout_seconds
                    candidate["failure_reason"] = _reason(
                        candidate.get("failure_reason"), [_COMPILE_TIMEOUT_REASON],
                    )
                    compile_timeouts.append(candidate_id)
                    if _compile_requires_device(candidate):
                        device_cleanup_unknown = True
                    continue
                try:
                    cleanup_ok = futures[candidate_id].result(timeout=max(0.0, remaining))
                except FutureTimeout:
                    events[candidate_id].set()
                    futures[candidate_id].cancel()
                    candidate = by_id[candidate_id]
                    candidate["compile_ok"] = False
                    candidate["compile_timeout_seconds"] = compile_timeout_seconds
                    candidate["failure_reason"] = _reason(
                        candidate.get("failure_reason"), [_COMPILE_TIMEOUT_REASON],
                    )
                    compile_timeouts.append(candidate_id)
                    if _compile_requires_device(candidate):
                        device_cleanup_unknown = True
                else:
                    if not cleanup_ok:
                        detail = {
                            "phase": "compile",
                            "candidate_id": candidate_id,
                            "cleanup": by_id[candidate_id].get("cleanup_failures", []),
                        }
                        compile_cleanup_failures.append(detail)
                        if _compile_requires_device(by_id[candidate_id]):
                            device_cleanup_unknown = True
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    final_plan = plan_rsi_compile_and_timing(
        records, state_dir=state_dir, npu_device=npu_device,
        max_compile_slots=max_compile_slots, assigned_devices=assigned_devices,
    )
    timing_order: list[str] = []
    timing_attempts: list[dict[str, Any]] = []
    cleanup_failures: list[dict[str, Any]] = list(compile_cleanup_failures)
    if device_cleanup_unknown:
        cleanup_failures.append({
            "phase": "compile",
            "reason": "device_compile_timeout_cleanup_unknown",
        })
    timing_blocked = (
        "device_compile_cleanup_unknown" if device_cleanup_unknown
        else "coordinator_cleanup_failed" if compile_cleanup_failures
        else ""
    )
    for item in final_plan["phase2_timing"]["items"]:
        candidate_id = item["candidate_id"]
        candidate = by_id[candidate_id]
        timing_order.append(candidate_id)
        if timing_blocked:
            _reject_timing(candidate, f"timing_not_attempted:{timing_blocked}")
            timing_attempts.append({
                "candidate_id": candidate_id, "acquired": False,
                "cleanup_ok": False, "failure_reason": timing_blocked,
            })
            continue
        cleanup_ok = _timing_one(
            candidate, timing_runner=timing_runner, authority=lease_authority,
            devices=list(final_plan["assigned_devices"]),
            lease_timeout_seconds=lease_timeout_seconds,
            timing_attempts=timing_attempts,
        )
        cleanup_failures.extend(candidate.get("cleanup_failures", []))
        if not cleanup_ok:
            timing_blocked = "coordinator_cleanup_failed"
            if not candidate.get("cleanup_failures"):
                cleanup_failures.append({
                    "phase": "timing",
                    "candidate_id": candidate_id,
                    "reason": "coordinator_cleanup_failed",
                })

    evaluations: list[dict[str, Any]] = []
    for candidate_id, candidate in by_id.items():
        compile_ok = _compile_proved(candidate) == "passed"
        workspace_limit = max_workspace_delta_bytes
        declared_limit = candidate.get("max_workspace_delta_bytes")
        problems: list[str] = []
        if workspace_limit is None and declared_limit is not None:
            if type(declared_limit) is int and declared_limit >= 0:
                workspace_limit = declared_limit
            else:
                problems.append("workspace_limit_invalid")
        guards = _required_guards(candidate, required_guard_workloads)
        expected_run_id = ""
        receipt = candidate.get("coordinator_lease_receipt")
        if isinstance(receipt, Mapping):
            expected_run_id = _receipt_execution_id(receipt)
        if compile_ok:
            problems.extend(_measurement_problems(
                candidate, workspace_limit=workspace_limit,
                required_guards=guards, expected_run_id=expected_run_id,
            ))
            if problems:
                candidate["measurement_passed"] = False
        failure_reason = _reason(candidate.get("failure_reason"), problems)
        entry = evaluate_rsi_candidate_gates(
            run_id=str(candidate.get("run_id") or ""),
            candidate_id=candidate_id,
            mutation_summary=str(candidate.get("mutation_summary", "")),
            conditions=dict(candidate.get("conditions") or {}),
            compile_ok=compile_ok,
            oracle_passed=candidate.get("oracle_passed") is True,
            measurement_passed=candidate.get("measurement_passed") is True,
            speedup_ratio=candidate.get("speedup_ratio"),
            min_speedup_ratio=min_speedup_ratio,
            ablation_retained=candidate.get("ablation_retained"),
            failure_reason=failure_reason,
            t_cand_us=candidate.get("t_cand_us", candidate.get("cand_latency_us", candidate.get("latency_us"))),
            t_baseline_us=candidate.get("t_baseline_us", candidate.get("baseline_perf_us", candidate.get("baseline_latency_us"))),
            t_hw_us=candidate.get("t_hw_us"),
            roofline_floor_us=candidate.get("roofline_floor_us"),
        )
        if problems and entry["gate_reached"] == 2:
            entry["gates"]["gate2_measurement_guard"]["reasons"] = list(problems)
        if candidate.get("cleanup_failures"):
            entry["cleanup_failures"] = list(candidate["cleanup_failures"])
            entry["cleanup_ok"] = False
        entry["execution_evidence"] = {
            "evidence_class": authority_contract.get("evidence_class", "unavailable"),
            "production_adapter": authority_contract.get("production_adapter", False),
            "physical_acceptance": False,
            "cleanup_ok": not bool(candidate.get("cleanup_failures")),
            "collection_mode": candidate.get("measurement_collection", "not_collected"),
        }
        evaluations.append(entry)
        if ledger_path is not None and progress_path is not None:
            append_ledger_and_render_progress(
                Path(ledger_path), Path(progress_path), entry,
                campaign_id=Path(state_dir).name or "rsi", goal=goal,
            )

    supported = [entry for entry in evaluations if entry["verdict"] == "supported"]
    best_candidate = sorted(
        supported, key=lambda entry: (-float(entry.get("speedup_ratio") or 0.0),
                                      str(entry["candidate_id"]))
    )[0] if supported else None
    authority_summary = {
        "available": lease_authority is not None,
        "mode": authority_contract.get("mode", "unavailable"),
        "evidence_class": authority_contract.get("evidence_class", "unavailable"),
        "production_adapter": authority_contract.get("production_adapter", False),
        "physical_acceptance": False,
        "verification": authority_contract.get("verification", "none"),
        "lease_timeout_seconds": lease_timeout_seconds,
        "compile_timeout_seconds": compile_timeout_seconds,
        "compile_timeout_budget": {
            "inner_seconds": (
                compile_floor - _COMPILE_BUDGET_GRACE_SECONDS
                if compile_floor is not None else None
            ),
            "grace_seconds": _COMPILE_BUDGET_GRACE_SECONDS,
            "outer_seconds": compile_timeout_seconds,
            "source": "execution_authority" if compile_floor is not None else "default",
        },
        "timing_attempts": timing_attempts,
        "leases_acquired": sum(1 for item in timing_attempts if item.get("acquired")),
        "leases_released": sum(1 for item in timing_attempts if item.get("released")),
        "device_compile_cleanup_unknown": device_cleanup_unknown,
        "cleanup_ok": not cleanup_failures and not device_cleanup_unknown,
        "cleanup_failures": cleanup_failures,
    }
    return {
        "plan": final_plan, "compile_order": compile_order,
        "compile_attempts": compile_attempts, "compile_timeouts": compile_timeouts,
        "timing_order": timing_order, "timing_attempts": timing_attempts,
        "execution_authority": authority_summary,
        "candidates": records, "evaluations": evaluations,
        "best_candidate": best_candidate,
    }


__all__ = [
    "RSIExecutionError", "RSIExecutionUnavailable", "TaskClientLease",
    "TaskClientRSIExecutor", "execute_rsi_compile_and_timing",
]
