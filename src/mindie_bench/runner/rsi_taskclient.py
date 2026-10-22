# [Side-Effect Policy] Writes only explicitly owned trial artifacts; no network or unowned mutation.

"""Production RSI coordinator adapter over `mindie_coordinator.TaskClient`.

[Role]
Submit compile/timing commands to the installed coordinator, hold the real
execution lease while a callback may run, independently collect the completed
coordinator execution, and verify role/device assignments from observed service
payloads.

[Invariants]
- Only a 64-hex execution ID returned by TaskClient is accepted; no local or
  callback-declared lease identity is created.
- A running lease authorizes callback execution. A terminal execution may only
  have its exact coordinator result collected; it never authorizes a later
  arbitrary callback after release.
- Result validation re-queries the exact execution through TaskClient and
  compares caller-supplied measurements with the observed payload. A caller
  dictionary alone is never coordinator evidence.
- A validated terminal result is retained even when release accounting has not
  yet caught up. Cleanup is reported separately; `released` is only true after
  an observed release, and later NPU timing remains blocked by the executor.
- The adapter never retries a failed or uncertain submission; cleanup failures
  preserve the execution ID and concrete stop/release observations.

[Timeouts & Locks]
- Lease admission is bounded by `lease_wait_seconds` (0..600s); release and
  command waits are bounded separately. Cleanup after a result or failed
  admission is bounded to 30s and reports its concrete stop/release
  observations. `compile_wait_budget_seconds` derives the inner admission,
  command, two release observations and cleanup sum for the outer scheduler.
  The coordinator owns device locks.

[Side-Effect Policy]
- Submits, observes, stops and releases its own TaskClient executions.
  It writes no local files and does not touch an NPU itself.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mindie_bench.runner.rsi_measurement import (
    _TERMINAL_EXECUTION_STATES,
    RSIExecutionError,
    RSIExecutionUnavailable,
    _assigned_devices,
    _assignment_devices,
    _compile_requires_device,
    _devices,
    _execution_receipt,
    _failure_reason,
    _mapping,
    _payload,
    _reason,
    _reply_failure_reason,
)

_EXECUTION_ID = re.compile(r"[0-9a-f]{64}\Z")
_CLEANUP_LIMIT_SECONDS = 30.0
_COMPARE_FIELDS = (
    "compile_ok", "oracle_passed", "measurement_passed", "speedup_ratio",
    "ablation_retained", "workspace_delta_bytes", "guard_workloads",
    "failure_reason",
)


def _terminal_payload(reply: Mapping[str, Any], *, execution_id: str, phase: str,
                      devices: Sequence[int]) -> dict[str, Any]:
    """Validate one terminal service reply without treating release as the result.

    Release accounting is a cleanup observation. A terminal succeeded command
    remains a validated measurement even when the local release wait timed out;
    callers must report cleanup separately and never synthesize ``released``.
    """
    if reply.get("execution_id") not in {None, execution_id}:
        raise RSIExecutionError(
            "coordinator release receipt belongs to another execution",
            execution_id=execution_id,
        )
    state = reply.get("state")
    if state not in _TERMINAL_EXECUTION_STATES:
        raise RSIExecutionError(
            f"execution {execution_id} did not reach a terminal state "
            f"(state {state!r}, wait_timed_out={reply.get('wait_timed_out') is True})",
            execution_id=execution_id,
        )
    payload = _payload(reply.get("stdout"))
    declared = payload.get("run_id")
    if declared not in (None, "") and str(declared) != execution_id:
        raise RSIExecutionError(
            f"command declared run_id {declared!r} but coordinator execution is "
            f"{execution_id}",
            execution_id=execution_id,
        )
    if phase == "compile":
        if state != "succeeded" and payload.get("compile_ok") is not False:
            raise RSIExecutionUnavailable(
                "compile execution did not succeed: "
                + (_reply_failure_reason(reply) or str(state)),
                execution_id=execution_id,
            )
        if state == "succeeded" and "compile_ok" not in payload:
            payload["compile_ok"] = True
        if payload.get("compile_ok") is False and state != "succeeded":
            payload["failure_reason"] = _reason(
                payload.get("failure_reason"),
                [_reply_failure_reason(reply) or "compile_execution_failed"],
            )
    elif state != "succeeded":
        raise RSIExecutionUnavailable(
            f"timing execution {execution_id} ended {state!r}: "
            + (_reply_failure_reason(reply) or "no failure text"),
            execution_id=execution_id,
            details={"state": state},
        )
    elif not payload:
        raise RSIExecutionError(
            f"timing execution {execution_id} produced no structured result",
            execution_id=execution_id,
        )
    payload["run_id"] = execution_id
    payload["coordinator_execution"] = _execution_receipt(
        reply, execution_id, phase, devices,
    )
    return payload


def _reply_observation(reply: Any) -> dict[str, Any]:
    if not isinstance(reply, Mapping):
        return {"value": _failure_reason(reply)}
    return {
        key: reply.get(key) for key in (
            "execution_id", "state", "resources_released", "wait_timed_out",
            "error", "cancel_requested",
        ) if key in reply
    }


def _same_value(left: Any, right: Any) -> bool:
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_value(left[key], right[key]) for key in left
        )
    if (isinstance(left, Sequence) and not isinstance(left, (str, bytes))
            and isinstance(right, Sequence) and not isinstance(right, (str, bytes))):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right)
        )
    try:
        return json.dumps(left, sort_keys=True, default=str) == json.dumps(
            right, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        return left == right


class TaskClientLease:
    """One real coordinator execution holding or having held a lease."""

    def __init__(self, adapter: "TaskClientRSIExecutor", execution_id: str,
                 devices: Sequence[int], phase: str, candidate_id: str,
                 admitted: Mapping[str, Any], *, terminal_at_acquire: bool = False):
        self.adapter = adapter
        self.execution_id = execution_id
        self.devices = tuple(int(item) for item in devices)
        self.phase = phase
        self.candidate_id = candidate_id
        self.admitted = dict(admitted)
        self.terminal_at_acquire = terminal_at_acquire
        self.released = False
        self.verified_active = False
        self.result: dict[str, Any] | None = None
        self.cleanup: dict[str, Any] = {}

    @property
    def client(self):
        return self.adapter.client

    def receipt(self) -> dict[str, Any]:
        receipt = _execution_receipt(self.admitted, self.execution_id, self.phase, self.devices)
        receipt.update(
            candidate_id=self.candidate_id,
            released=self.released,
            terminal_at_acquire=self.terminal_at_acquire,
            resources_released=self.released or self.admitted.get("resources_released") is True,
            cleanup=dict(self.cleanup) if self.cleanup else None,
        )
        return receipt

    def _observe_status(self) -> Mapping[str, Any]:
        status = self.client.observe(self.execution_id, action="status", refresh=False)
        if not isinstance(status, Mapping):
            raise RSIExecutionError(
                f"coordinator returned no status for {self.execution_id}",
                execution_id=self.execution_id,
            )
        if status.get("execution_id") not in {None, self.execution_id}:
            raise RSIExecutionError(
                "coordinator status belongs to another execution",
                execution_id=self.execution_id,
            )
        self.admitted = {**self.admitted, **dict(status)}
        if status.get("resources_released") is True:
            self.released = True
        return status

    def refresh_state(self) -> str:
        """Observe the exact execution and report terminal completion as-is.

        Unlike ``verify_active`` this is not an ownership assertion: a command
        that completed after admission is returned as terminal so the caller can
        collect its exact result instead of treating the lease as failed.
        """
        state = str(self._observe_status().get("state") or "").strip().lower()
        if state in _TERMINAL_EXECUTION_STATES:
            self.terminal_at_acquire = True
        return state

    def verify_active(self) -> bool:
        """A callback may run only while the exact execution is still running."""
        status = self._observe_status()
        state = str(status.get("state") or "")
        if state != "running":
            raise RSIExecutionUnavailable(
                f"coordinator execution {self.execution_id} is {state!r}, not running; "
                "a released/terminal lease cannot authorize a callback",
                execution_id=self.execution_id,
                details={"state": state, "collectable": state in _TERMINAL_EXECUTION_STATES},
            )
        assigned = _assigned_devices(status)
        if not assigned:
            assigned = _assigned_devices(self.admitted)
        if self.phase == "timing" and set(assigned) != set(self.devices):
            raise RSIExecutionError(
                f"coordinator did not assign {list(self.devices)} for timing "
                f"(observed {assigned})",
                execution_id=self.execution_id,
                details={"assigned_devices": assigned},
            )
        if self.phase == "compile" and self.devices and set(assigned) != set(self.devices):
            raise RSIExecutionError(
                f"coordinator did not assign {list(self.devices)} for device compile "
                f"(observed {assigned})",
                execution_id=self.execution_id,
                details={"assigned_devices": assigned},
            )
        self.verified_active = True
        return True

    def verify(self) -> bool:
        return self.verify_active()

    def _completed_result(self, reply: Mapping[str, Any]) -> dict[str, Any]:
        payload = _terminal_payload(
            reply, execution_id=self.execution_id, phase=self.phase,
            devices=self.devices,
        )
        self.result = dict(payload)
        self.admitted = {**self.admitted, **dict(reply)}
        if payload["coordinator_execution"].get("resources_released") is True:
            self.released = True
        return dict(payload)

    def collect_result(self, *, refresh: bool = False) -> dict[str, Any]:
        """Collect the exact completed execution from TaskClient.

        This is coordinator observation, not trust in a caller-supplied result
        receipt. `refresh=True` always issues the query again.
        """
        if self.result is not None and not refresh:
            return dict(self.result)
        reply = self.adapter.wait_released(self.execution_id)
        if not isinstance(reply, Mapping):
            raise RSIExecutionError(
                "coordinator returned no release receipt",
                execution_id=self.execution_id,
            )
        return self._completed_result(reply)

    def wait_result(self) -> dict[str, Any]:
        return self.collect_result()

    def release(self) -> dict[str, Any]:
        if self.released:
            return self.receipt()
        cleanup: dict[str, Any] = {
            "execution_id": self.execution_id, "stop": None, "stop_error": None,
            "release": None, "release_error": None, "released": False,
        }
        status: Mapping[str, Any] = {}
        try:
            status = self.client.observe(self.execution_id, action="status", refresh=False)
            cleanup["status"] = _reply_observation(status)
            if (isinstance(status, Mapping)
                    and status.get("execution_id") not in {None, self.execution_id}):
                cleanup["status_error"] = "coordinator status belongs to another execution"
                status = {}
            if isinstance(status, Mapping) and status.get("resources_released") is True:
                self.released = True
                self.admitted = {**self.admitted, **dict(status)}
                cleanup["released"] = True
                self.cleanup = cleanup
                return self.receipt()
        except Exception as exc:
            cleanup["status_error"] = _failure_reason(f"{type(exc).__name__}: {exc}")
        try:
            cleanup["stop"] = _reply_observation(
                self.client.observe(self.execution_id, action="stop")
            )
        except Exception as exc:
            cleanup["stop_error"] = _failure_reason(f"{type(exc).__name__}: {exc}")
        try:
            reply = self.adapter.wait_released(
                self.execution_id,
                timeout_seconds=self.adapter.cleanup_wait_seconds(),
            )
            cleanup["release"] = _reply_observation(reply)
            if (isinstance(reply, Mapping)
                    and reply.get("execution_id") not in {None, self.execution_id}):
                cleanup["release_error"] = "coordinator release receipt belongs to another execution"
                reply = {}
            if isinstance(reply, Mapping) and reply.get("resources_released") is True:
                self.released = True
                self.admitted = {**self.admitted, **dict(reply)}
                cleanup["released"] = True
                self.cleanup = cleanup
                return self.receipt()
        except Exception as exc:
            cleanup["release_error"] = _failure_reason(f"{type(exc).__name__}: {exc}")
        self.cleanup = cleanup
        raise RSIExecutionError(
            f"coordinator execution {self.execution_id} remains unreleased",
            execution_id=self.execution_id,
            cleanup=cleanup,
            details={"stop_error": cleanup["stop_error"], "release_error": cleanup["release_error"]},
        )


class TaskClientRSIExecutor:
    """Production adapter: TaskClient submissions are the lease authority."""

    mode = "mindie_coordinator.task_client"
    evidence_class = "coordinator_execution"
    production_adapter = True
    physical_acceptance = False

    def __init__(
        self,
        client: Any,
        *,
        sources: Mapping[str, str] | None = None,
        topology: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
        command_timeout_seconds: int = 1800,
        lease_wait_seconds: float = 300.0,
        release_wait_seconds: float = 600.0,
        release_timeout_seconds: float = 1800.0,
    ):
        for name in ("run", "wait", "observe"):
            if not callable(getattr(client, name, None)):
                raise ValueError(f"TaskClient adapter requires a callable {name} method")
        if not 0 <= lease_wait_seconds <= 600:
            raise ValueError("lease_wait_seconds must be between 0 and 600")
        if not 0 <= release_wait_seconds <= 600:
            raise ValueError("release_wait_seconds must be between 0 and 600")
        if not 0 < release_timeout_seconds <= 24 * 3600:
            raise ValueError("release_timeout_seconds must be positive and bounded")
        if type(command_timeout_seconds) is not int or command_timeout_seconds < 1:
            raise ValueError("command_timeout_seconds must be a positive integer")
        self.client = client
        self.sources = dict(sources) if sources is not None else None
        self.topology = dict(topology) if topology is not None else None
        self.env = dict(env) if env is not None else None
        self.command_timeout_seconds = command_timeout_seconds
        self.lease_wait_seconds = float(lease_wait_seconds)
        self.release_wait_seconds = float(release_wait_seconds)
        self.release_timeout_seconds = float(release_timeout_seconds)

    def cleanup_wait_seconds(self) -> float:
        """Bound post-result stop/release accounting separately from command time."""
        return min(self.release_timeout_seconds, _CLEANUP_LIMIT_SECONDS)

    def compile_wait_budget_seconds(self, lease_timeout_seconds: float | None = None) -> float:
        """Worst-case inner compile wait before the outer scheduler deadline.

        The executor admits (``lease_timeout_seconds``), the command has its
        business timeout, the result is collected once, validation re-queries the
        exact execution, and release cleanup gets only its bounded budget.
        """
        admission = (self.lease_wait_seconds if lease_timeout_seconds is None
                     else float(lease_timeout_seconds))
        release = self.release_timeout_seconds
        return (admission + float(self.command_timeout_seconds)
                + release + release + self.cleanup_wait_seconds())

    @staticmethod
    def _command(candidate: Mapping[str, Any], phase: str) -> str:
        value = candidate.get(f"{phase}_command")
        if value is None:
            value = _mapping(candidate.get("commands")).get(phase)
        if not isinstance(value, str) or not value.strip():
            raise RSIExecutionUnavailable(
                f"candidate {candidate.get('candidate_id', '<unknown>')!r} has no "
                f"{phase}_command for the production coordinator"
            )
        return value

    @staticmethod
    def _script_file(candidate: Mapping[str, Any], phase: str) -> str | None:
        value = candidate.get(f"{phase}_script_file")
        return str(value) if isinstance(value, (str, Path)) and str(value).strip() else None

    @staticmethod
    def _resources(candidate: Mapping[str, Any], phase: str,
                   devices: Sequence[int]) -> dict[str, Any]:
        if phase == "compile" and not _compile_requires_device(candidate):
            return {"npu_count": 0}
        selected = _devices(devices)
        if not selected:
            raise RSIExecutionUnavailable("coordinator lease requires at least one device")
        resources: dict[str, Any] = {"devices": selected}
        service_port = candidate.get("service_port")
        if type(service_port) is int and 0 <= service_port < 65536:
            resources["service_port"] = service_port
        return resources

    def _compile_topology(self, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
        """CPU compile may select a host but never inherit gang/device planning."""
        explicit = _mapping(candidate.get("topology"))
        configured = explicit or self.topology or {}
        host = configured.get("host")
        if host is None and explicit:
            host = (self.topology or {}).get("host")
        if not isinstance(host, str) or not host.strip():
            roles = configured.get("roles")
            hosts = {
                str(role.get("host")).strip()
                for role in roles or () if isinstance(role, Mapping) and role.get("host")
            } if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)) else set()
            if len(hosts) == 1:
                host = hosts.pop()
            elif configured:
                raise RSIExecutionUnavailable(
                    "CPU-only compile requires a single explicit host; "
                    "device/gang topology is not admissible"
                )
            else:
                return None
        return {"host": host.strip()}

    def _runtime_topology(self, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
        return _mapping(candidate.get("topology")) or self.topology

    def _submit(self, candidate: Mapping[str, Any], phase: str,
                devices: Sequence[int]) -> tuple[str, Mapping[str, Any], bool]:
        cpu_compile = phase == "compile" and not _compile_requires_device(candidate)
        kwargs: dict[str, Any] = {
            "sources": self.sources,
            "env": {**(self.env or {}), **_mapping(candidate.get("env"))},
            "resources": self._resources(candidate, phase, devices),
            "topology": self._compile_topology(candidate) if cpu_compile
            else self._runtime_topology(candidate),
            "timeout_seconds": self.command_timeout_seconds,
        }
        script_file = self._script_file(candidate, phase)
        if script_file:
            kwargs["script_file"] = script_file
        else:
            kwargs["command"] = self._command(candidate, phase)
        reply = self.client.run(**kwargs)
        if not isinstance(reply, Mapping):
            raise RSIExecutionUnavailable("TaskClient run returned no admission receipt")
        execution_id = reply.get("execution_id")
        if not isinstance(execution_id, str) or not _EXECUTION_ID.fullmatch(execution_id):
            raise RSIExecutionUnavailable(
                "TaskClient did not return an actual 64-hex execution ID; no lease is claimed"
            )
        return execution_id, reply, cpu_compile

    def _cleanup_execution(self, execution_id: str) -> dict[str, Any]:
        cleanup: dict[str, Any] = {
            "execution_id": execution_id, "stop": None, "stop_error": None,
            "release": None, "release_error": None, "released": False,
        }
        try:
            stopped = self.client.observe(execution_id, action="stop")
            cleanup["stop"] = _reply_observation(stopped)
            if (isinstance(stopped, Mapping)
                    and stopped.get("execution_id") not in {None, execution_id}):
                cleanup["stop_error"] = "coordinator stop reply belongs to another execution"
        except Exception as exc:
            cleanup["stop_error"] = _failure_reason(f"{type(exc).__name__}: {exc}")
        try:
            reply = self.wait_released(
                execution_id, timeout_seconds=self.cleanup_wait_seconds()
            )
            cleanup["release"] = _reply_observation(reply)
            if (isinstance(reply, Mapping)
                    and reply.get("execution_id") not in {None, execution_id}):
                cleanup["release_error"] = "coordinator release receipt belongs to another execution"
            else:
                cleanup["released"] = (
                    isinstance(reply, Mapping) and reply.get("resources_released") is True
                )
        except Exception as exc:
            cleanup["release_error"] = _failure_reason(f"{type(exc).__name__}: {exc}")
        return cleanup

    def _raise_after_cleanup(self, message: str, execution_id: str,
                             cleanup: Mapping[str, Any], *, cause: BaseException | None = None):
        error = RSIExecutionError(message, execution_id=execution_id, cleanup=cleanup)
        if cause is not None:
            raise error from cause
        raise error

    def acquire(self, devices: Sequence[int], *, phase: str, candidate: Mapping[str, Any],
                timeout_seconds: float | None = None) -> TaskClientLease:
        lease_timeout = self.lease_wait_seconds if timeout_seconds is None else float(timeout_seconds)
        if not 0 <= lease_timeout <= 600:
            raise ValueError("lease timeout must be between 0 and 600 seconds")
        candidate_id = str(candidate.get("candidate_id") or "")
        execution_id, admitted, _cpu = self._submit(candidate, phase, devices)
        try:
            running = self.client.wait(
                execution_id, until="running", timeout_seconds=lease_timeout,
            )
        except Exception as exc:
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"coordinator lease observation failed for {execution_id}: "
                f"{type(exc).__name__}: {exc}",
                execution_id, cleanup, cause=exc,
            )
            raise AssertionError("unreachable")
        if not isinstance(running, Mapping):
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"coordinator returned no lease observation for {execution_id}",
                execution_id, cleanup,
            )
            raise AssertionError("unreachable")
        if running.get("execution_id") not in {None, execution_id}:
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                "coordinator lease observation belongs to another execution",
                execution_id, cleanup,
            )
            raise AssertionError("unreachable")
        state = str(running.get("state") or "")
        if running.get("wait_timed_out") is True or state not in (
            "running", *_TERMINAL_EXECUTION_STATES
        ):
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"coordinator lease for {execution_id} was not granted within "
                f"{lease_timeout:g}s (state {state!r})",
                execution_id, cleanup,
            )
            raise AssertionError("unreachable")
        lease = TaskClientLease(
            self, execution_id, devices, phase, candidate_id, running,
            terminal_at_acquire=state in _TERMINAL_EXECUTION_STATES,
        )
        if state == "running":
            try:
                lease.verify_active()
            except RSIExecutionUnavailable as exc:
                details = getattr(exc, "details", {})
                if details.get("collectable"):
                    # The command completed between the running wait and the
                    # active check. It is collectable, but never callback-
                    # authorizing; switch the same exact execution to terminal
                    # collection mode.
                    terminal = self.client.observe(execution_id, action="status", refresh=False)
                    terminal_state = str(terminal.get("state") or "") if isinstance(terminal, Mapping) else ""
                    if terminal_state in _TERMINAL_EXECUTION_STATES:
                        lease.terminal_at_acquire = True
                        lease.admitted = {**lease.admitted, **dict(terminal)}
                        return lease
                cleanup = self._cleanup_execution(execution_id)
                self._raise_after_cleanup(
                    f"coordinator running lease did not verify for {execution_id}: "
                    f"{type(exc).__name__}: {exc}",
                    execution_id, cleanup, cause=exc,
                )
                raise AssertionError("unreachable")
            except Exception as exc:
                cleanup = self._cleanup_execution(execution_id)
                self._raise_after_cleanup(
                    f"coordinator running lease did not verify for {execution_id}: "
                    f"{type(exc).__name__}: {exc}",
                    execution_id, cleanup, cause=exc,
                )
                raise AssertionError("unreachable")
        return lease

    def wait_released(self, execution_id: str, *,
                      timeout_seconds: float | None = None) -> Mapping[str, Any]:
        """Observe the same execution until release, never resubmit it."""
        limit = self.release_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        deadlined = time.monotonic() + limit
        while True:
            remaining = deadlined - time.monotonic()
            if remaining <= 0:
                raise RSIExecutionError(
                    f"coordinator execution {execution_id} did not release within {limit:g}s",
                    execution_id=execution_id,
                )
            reply = self.client.wait(
                execution_id, until="released",
                timeout_seconds=max(0.1, min(self.release_wait_seconds, remaining)),
            )
            if not isinstance(reply, Mapping):
                raise RSIExecutionError(
                    "coordinator returned no release receipt", execution_id=execution_id,
                )
            if reply.get("wait_timed_out") is not True:
                return reply
            if remaining <= self.release_wait_seconds:
                return reply

    def collect_completed(self, lease: Any, *, phase: str = "",
                          candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(lease, TaskClientLease):
            raise RSIExecutionError("completed result collection requires a TaskClient lease")
        return lease.collect_result(refresh=True)

    def compile_runner(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        lease = candidate.get("coordinator_lease")
        if lease is not None:
            if not isinstance(lease, TaskClientLease):
                raise RSIExecutionError("device compile requires a TaskClient lease")
            return lease.wait_result()
        if _compile_requires_device(candidate):
            raise RSIExecutionUnavailable("device compile requires a coordinator lease")
        execution_id, _admitted, _cpu = self._submit(candidate, "compile", ())
        try:
            reply = self.wait_released(execution_id)
        except Exception as exc:
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"CPU compile execution {execution_id} did not finish: "
                f"{type(exc).__name__}: {exc}",
                execution_id, cleanup, cause=exc,
            )
            raise AssertionError("unreachable")
        if not isinstance(reply, Mapping):
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"CPU compile execution {execution_id} returned no receipt",
                execution_id, cleanup,
            )
            raise AssertionError("unreachable")
        if reply.get("execution_id") not in {None, execution_id}:
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                "CPU compile release receipt belongs to another execution",
                execution_id, cleanup,
            )
            raise AssertionError("unreachable")
        observed_devices = _assigned_devices(reply)
        echo_devices = _assignment_devices(reply)
        if observed_devices or echo_devices:
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"CPU-only compile {execution_id} unexpectedly acquired devices "
                f"{observed_devices or echo_devices}",
                execution_id, cleanup,
            )
            raise AssertionError("unreachable")
        try:
            payload = _terminal_payload(
                reply, execution_id=execution_id, phase="compile", devices=(),
            )
        except Exception as exc:
            cleanup = self._cleanup_execution(execution_id)
            self._raise_after_cleanup(
                f"CPU compile execution {execution_id} has no terminal result: "
                f"{type(exc).__name__}: {exc}",
                execution_id, cleanup, cause=exc,
            )
            raise AssertionError("unreachable")
        if payload["coordinator_execution"].get("resources_released") is not True:
            payload["coordinator_cleanup"] = self._cleanup_execution(execution_id)
        return dict(payload)

    def timing_runner(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        lease = candidate.get("coordinator_lease")
        if not isinstance(lease, TaskClientLease):
            raise RSIExecutionUnavailable(
                "timing requires a real TaskClient coordinator lease"
            )
        return lease.wait_result()

    def validate_result(self, lease: Any, result: Mapping[str, Any], *, phase: str,
                        devices: Sequence[int]) -> None:
        if not isinstance(lease, TaskClientLease):
            raise RSIExecutionError("result is not bound to a TaskClient lease")
        if not isinstance(result, Mapping):
            raise RSIExecutionError("callback returned no result object")
        execution_id = lease.execution_id
        observed = lease.collect_result(refresh=True)
        observed_receipt = observed.get("coordinator_execution")
        if not isinstance(observed_receipt, Mapping):
            raise RSIExecutionError(
                "coordinator observation has no execution receipt",
                execution_id=execution_id,
            )
        run_id = str(result.get("run_id") or "")
        if run_id != execution_id:
            raise RSIExecutionError(
                f"run_id {run_id!r} is not the coordinator execution {execution_id}",
                execution_id=execution_id,
            )
        state = observed_receipt.get("state")
        if phase == "timing" and state != "succeeded":
            raise RSIExecutionError(
                f"timing execution state is {state!r}", execution_id=execution_id,
            )
        if phase == "compile" and state not in {"succeeded", "failed"}:
            raise RSIExecutionError(
                f"compile execution state is {state!r}", execution_id=execution_id,
            )
        if (phase == "compile" and state == "failed"
                and result.get("compile_ok") is not False):
            raise RSIExecutionError(
                "compile execution failed but caller claims compile success",
                execution_id=execution_id,
            )
        # Resource release is cleanup accounting, not measurement evidence. The
        # executor records an unresolved cleanup and blocks later NPU timing.
        assigned = _devices(observed_receipt.get("granted_devices"))
        expected = set(int(item) for item in devices)
        if phase == "timing" and set(assigned) != expected:
            raise RSIExecutionError(
                f"timing observation assigned {assigned}, requested {list(devices)}",
                execution_id=execution_id,
            )
        if phase == "compile" and result.get("compile_ok") is not False:
            if set(assigned) != expected:
                raise RSIExecutionError(
                    f"compile observation assigned {assigned}, requested {list(devices)}",
                    execution_id=execution_id,
                )
        for key in _COMPARE_FIELDS:
            if key not in result:
                continue
            if key == "failure_reason" and not str(result[key] or "").strip():
                continue
            if not _same_value(result[key], observed.get(key)):
                raise RSIExecutionError(
                    f"caller result differs from coordinator observation for {key}",
                    execution_id=execution_id,
                )
        caller_receipt = result.get("coordinator_execution")
        if caller_receipt is not None:
            if not isinstance(caller_receipt, Mapping):
                raise RSIExecutionError(
                    "caller-supplied execution receipt is not an object",
                    execution_id=execution_id,
                )
            for key in ("execution_id", "state", "granted_devices"):
                if key in caller_receipt and not _same_value(
                    caller_receipt[key], observed_receipt.get(key)
                ):
                    raise RSIExecutionError(
                        f"caller-supplied receipt differs from coordinator observation for {key}",
                        execution_id=execution_id,
                    )
            if (caller_receipt.get("resources_released") is True
                    and observed_receipt.get("resources_released") is not True):
                raise RSIExecutionError(
                    "caller-supplied receipt claims release that coordinator did not observe",
                    execution_id=execution_id,
                )

    def release(self, lease: Any) -> dict[str, Any]:
        if not isinstance(lease, TaskClientLease):
            raise RSIExecutionError("cannot release an unknown lease")
        return lease.release()
