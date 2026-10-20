"""RSI measurement-evidence validation and lease receipt parsing.

[Role]
Separate observed coordinator evidence from candidate-supplied claims: parse
execution receipts, workspace/guard measurements and failure reasons used by
the RSI executor and the TaskClient adapter.

[Invariants]
- An unobserved run ID is empty, never synthesized from a candidate name.
- Observed top-level role views, `target.environment`/`target.env` and their
  per-role device identities are the grant. Assignment echoes never override a
  role/target grant; candidate requests are never treated as observations.
- A requested workspace/guard measurement that is absent is a problem, not a
  pass. Unrequested gates add no requirements; global guard requirements
  augment candidate-declared ones.
- Failure reasons are preserved in their original order and deduplicated.

[Timeouts & Locks]
- Pure functions; no waits, locks, subprocesses or coordinator calls.

[Side-Effect Policy]
- No I/O or mutation. Callers own all execution and ledger effects.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

_LEASE_IDENTITY_KEYS = ("execution_id", "run_id", "lease_id", "task_id")
_TERMINAL_EXECUTION_STATES = frozenset({
    "succeeded", "failed", "timeout", "cancelled", "inconclusive", "expired",
    "released",
})


class RSIExecutionError(RuntimeError):
    """An execution or lease receipt is insufficient for measurement."""

    def __init__(self, message: str, *, execution_id: str = "",
                 cleanup: Mapping[str, Any] | None = None,
                 details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.execution_id = execution_id
        self.cleanup = dict(cleanup or {})
        self.details = dict(details or {})


class RSIExecutionUnavailable(RSIExecutionError):
    """No production execution authority or no admissible lease was available."""


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _devices(value: Any) -> list[int]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Mapping):
        raw = list(value.values())
    elif isinstance(value, Sequence):
        raw = list(value)
    elif type(value) is int:
        raw = [value]
    else:
        raw = []
    devices: list[int] = []
    for item in raw:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number < 0 or str(item).strip() not in {str(number), f"{number}.0"}:
            continue
        if number not in devices:
            devices.append(number)
    return devices


def _receipt(lease: Any) -> dict[str, Any]:
    value = getattr(lease, "receipt", None)
    value = value() if callable(value) else value
    if not isinstance(value, Mapping):
        raise RSIExecutionError("coordinator lease did not provide a receipt")
    return dict(value)


def _receipt_execution_id(value: Mapping[str, Any]) -> str:
    for key in _LEASE_IDENTITY_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _failure_reason(value: Any, limit: int = 800) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())[:limit]
    return "" if text == "None" else text


def _reason(existing: Any, additions: Sequence[str]) -> str:
    parts: list[str] = []
    current = _failure_reason(existing)
    if current:
        parts.append(current)
    for addition in additions:
        text = _failure_reason(addition)
        if text and text not in parts:
            parts.append(text)
    return "; ".join(parts)


def _merge_result(candidate: dict[str, Any], result: Mapping[str, Any],
                  fields: Sequence[str]) -> None:
    for key in fields:
        if key not in result:
            continue
        value = result[key]
        if key == "failure_reason":
            if not str(value or "").strip():
                continue
            candidate[key] = _reason(candidate.get(key), [str(value)])
            continue
        candidate[key] = value


def _compile_requires_device(candidate: Mapping[str, Any]) -> bool:
    for key in ("device_compile", "compile_on_device", "requires_device", "jit_on_device"):
        if candidate.get(key) is True:
            return True
    for key in ("compile_npu_count", "npu_count"):
        count = candidate.get(key)
        if type(count) is int and count > 0:
            return True
    return bool(_devices(candidate.get("compile_devices")))


def _workspace_values(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        raw = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raw = [value]
    numbers = [_number(item) for item in raw]
    # A partially measured rank set is still a missing measurement; never
    # accept the numeric subset as if every requested rank had reported.
    if not numbers or any(number is None for number in numbers):
        return []
    return [number for number in numbers if number is not None]


def _guard_values(guard: Mapping[str, Any]) -> float | None:
    ratio = _number(guard.get("regression_ratio"))
    baseline = _number(guard.get("baseline_ms") if guard.get("baseline_ms") is not None
                       else guard.get("baseline"))
    candidate = _number(guard.get("candidate_ms") if guard.get("candidate_ms") is not None
                        else guard.get("candidate"))
    if ratio is None and baseline is not None and baseline > 0 and candidate is not None:
        ratio = candidate / baseline
    return ratio


def _guard_problems(guards: Any, required: Sequence[str]) -> list[str]:
    if guards is not None and (isinstance(guards, (str, bytes))
                               or not isinstance(guards, Sequence)):
        return ["guard_measurement_invalid"]
    rows = [dict(row) for row in guards or () if isinstance(row, Mapping)]
    by_name: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for row in rows:
        name = str(row.get("name") or row.get("workload") or "").strip()
        if not name:
            problems.append("guard_identity_missing")
            continue
        if name in by_name:
            problems.append(f"guard_duplicate:{name}")
            continue
        by_name[name] = row
        if row.get("measurement_valid") is False:
            problems.append(f"guard_measurement_invalid:{name}")
        if row.get("passed") is False:
            problems.append(f"guard_workload_failed:{name}")
        ratio = _guard_values(row)
        if ratio is None:
            problems.append(f"guard_measurement_missing:{name}")
            continue
        tolerance = _number(row.get("max_regression_ratio"))
        tolerance = 1.0 if tolerance is None else tolerance
        if ratio > tolerance + 1e-12:
            problems.append(f"guard_workload_regressed:{name}")
    for name in required:
        if name not in by_name:
            problems.append(f"guard_workload_missing:{name}")
    return problems


def _required_guards(candidate: Mapping[str, Any],
                     explicit: Sequence[str] | None) -> list[str]:
    """Union global, declared and per-candidate required guard names.

    An explicit global list augments candidate requirements; it never replaces
    them. The snapshot from before timing evidence was cleared is included so a
    candidate-declared guard cannot disappear with its prefilled measurement.
    """
    names: list[str] = []
    sources = (
        explicit,
        candidate.get("declared_required_guard_workloads"),
        candidate.get("required_guard_workloads"),
    )
    for source in sources:
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
            continue
        for item in source:
            name = str(item).strip()
            if name and name not in names:
                names.append(name)
    for row in candidate.get("guard_workloads") or ():
        if isinstance(row, Mapping) and row.get("required") is True:
            name = str(row.get("name") or row.get("workload") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _measurement_problems(
    candidate: Mapping[str, Any],
    *,
    workspace_limit: int | None,
    required_guards: Sequence[str],
    expected_run_id: str,
) -> list[str]:
    problems: list[str] = []
    run_id = str(candidate.get("run_id") or "").strip()
    if not run_id:
        problems.append("observed_run_id_missing")
    elif expected_run_id and run_id != expected_run_id:
        problems.append("observed_run_id_differs_from_coordinator")
    if candidate.get("measurement_passed") is not True:
        problems.append("measurement_not_passed")
    if workspace_limit is not None:
        values = _workspace_values(candidate.get("workspace_delta_bytes"))
        if not values:
            problems.append("workspace_delta_not_measured")
        elif max(values) > workspace_limit:
            problems.append("workspace_delta_above_limit")
    observed_guards = candidate.get("guard_workloads")
    if observed_guards is not None or required_guards:
        problems.extend(_guard_problems(observed_guards, required_guards))
    return problems


def _result_failure(value: Any) -> str:
    return _failure_reason(f"{type(value).__name__}: {value}")


def _payload(text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for chunk in (text.strip(), *reversed(lines)):
        try:
            value = json.loads(chunk)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _role_devices(value: Mapping[str, Any], *,
                  allow_value_env: bool = False) -> tuple[list[int], str]:
    """Read granted devices from the observed coordinator role/target payload.

    Never use candidate requests. The production service emits role views with
    `target.environment` (and older/newer peers may emit `target.env` or a
    top-level `devices` field); assignment is retained only as a fallback
    observation, not as the primary grant.
    """
    target = _mapping(value.get("target"))
    environment = _mapping(target.get("environment")) or _mapping(target.get("env"))
    if not environment:
        environment = _mapping(value.get("environment"))
    if not environment and allow_value_env:
        environment = _mapping(value.get("env"))
    direct = _devices(value.get("devices"))
    if direct:
        return direct, "role.devices"
    from_env = _devices(environment.get("ASCEND_RT_VISIBLE_DEVICES"))
    if from_env:
        return from_env, "role.target.environment.ASCEND_RT_VISIBLE_DEVICES"
    generic = _devices(environment.get("devices"))
    if generic:
        return generic, "role.target.environment.devices"
    return [], ""


def _role_identities(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    roles = value.get("roles")
    if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)):
        for index, raw in enumerate(roles):
            role = _mapping(raw)
            if not role:
                continue
            devices, source = _role_devices(role)
            target = _mapping(role.get("target"))
            endpoint = target.get("endpoint") if isinstance(target.get("endpoint"), Mapping) else {}
            identities.append({
                "name": str(role.get("name") or f"role-{index}"),
                "host": role.get("host") or endpoint.get("host"),
                "runtime_id": role.get("runtime_id") or target.get("runtime_id"),
                "state": role.get("state") or target.get("state"),
                "devices": devices,
                "device_source": source,
            })
    target = _mapping(value.get("target"))
    if identities and not any(item["devices"] for item in identities) and target:
        devices, source = _role_devices(target, allow_value_env=True)
        if devices:
            identities.append({
                "name": "target", "host": None, "runtime_id": target.get("runtime_id"),
                "state": target.get("state"), "devices": devices,
                "device_source": source,
            })
    if not identities and target:
        devices, source = _role_devices(target, allow_value_env=True)
        identities.append({
            "name": str(target.get("name") or "target"),
            "host": (target.get("endpoint") or {}).get("host")
            if isinstance(target.get("endpoint"), Mapping) else None,
            "runtime_id": target.get("runtime_id"),
            "state": target.get("state"),
            "devices": devices,
            "device_source": source,
        })
    if not identities:
        devices = _devices(value.get("granted_devices"))
        if devices:
            identities.append({
                "name": "granted", "host": None, "runtime_id": None,
                "state": value.get("state"), "devices": devices,
                "device_source": "granted_devices",
            })
    return identities


def _assignment_devices(value: Mapping[str, Any]) -> list[int]:
    """Echo devices are diagnostic only; they are never a grant."""
    assignment = _mapping(value.get("assignment"))
    devices: list[int] = []
    roles = assignment.get("roles")
    if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)):
        for raw in roles:
            for item in _devices(_mapping(raw).get("devices")):
                if item not in devices:
                    devices.append(item)
    return devices


def _assigned_devices(value: Mapping[str, Any]) -> list[int]:
    devices: list[int] = []
    for identity in _role_identities(value):
        for item in identity["devices"]:
            if item not in devices:
                devices.append(item)
    return devices


def _execution_receipt(reply: Mapping[str, Any], execution_id: str, phase: str,
                       devices: Sequence[int]) -> dict[str, Any]:
    identities = _role_identities(reply)
    return {
        "schema_version": "mindie-bench-rsi-execution/1",
        "authority": "mindie_coordinator.task_client",
        "execution_id": execution_id,
        "phase": phase,
        "devices": list(devices),
        "roles": identities,
        "granted_devices": _assigned_devices(reply),
        "state": reply.get("state"),
        "resources_released": reply.get("resources_released") is True,
        "service": reply.get("service"),
        "source_snapshot_id": reply.get("source_snapshot_id"),
        "observed_at": reply.get("observed_at"),
    }

def _reply_failure_reason(reply: Mapping[str, Any]) -> str:
    for key in ("error", "stderr", "tail", "stdout"):
        value = reply.get(key)
        if isinstance(value, str) and value.strip():
            return _failure_reason(value)
    return ""
