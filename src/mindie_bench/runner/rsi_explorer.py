"""L4 recursive self-improvement context, scheduling, gates and ledger.

[Role]
Turn stalled L3 heuristic work into L4 RSI: mount L1 foundation, L2 bottleneck
and L4 mutators, schedule CPU-only compiles before exclusive NPU timing, then
record four-gate outcomes.

[Invariants]
- L3 workflow units never enter L4 context; only IDs survive in `stripped_l3_ids`.
- Refuted candidates always carry `conditional_avoid`; only supported candidates
  are `distill_ready`.
- Compiles are CPU-only (`npu_count == 0`); NPU timing is serial, exclusive to
  the declared device, and admitted only for proved compiles.
- The ledger is append-only; PROGRESS.md is deterministic and atomically replaced.

[Timeouts & Locks]
- Pure planning/evaluation helpers perform no waits and take no locks.
- The ledger owns one short O_APPEND write and one atomic markdown replacement;
  callers own all experiment and NPU timeouts.

[Side-Effect Policy]
- Only `append_ledger_and_render_progress` writes: the JSONL ledger and PROGRESS.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "mindie-bench-rsi/1"
DEFAULT_MIN_SPEEDUP_RATIO = 1.05
_COMPILE_PASS = frozenset({"compiled", "ok", "pass", "passed", "success", "succeeded"})


def _seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value) if isinstance(value, Sequence) else [value]


def _unit_id(unit: Mapping[str, Any], index: int) -> str:
    raw = unit.get("unit_id", unit.get("id", unit.get("workflow_id")))
    return str(raw) if raw not in (None, "") else f"unit-{index}"


def _tier(unit: Mapping[str, Any]) -> str:
    return str(unit.get("tier", "")).strip().upper()


def _number(attempt: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        try:
            value = float(attempt[key])
        except (KeyError, TypeError, ValueError):
            continue
        return value if math.isfinite(value) else None
    return None


def _gate(attempt: Mapping[str, Any], name: str, aliases: Sequence[str]) -> bool | None:
    for key in (name, *aliases):
        value = attempt.get(key)
        if isinstance(value, Mapping):
            value = value.get("passed", value.get("status"))
        if value is not None:
            return str(value).lower() in _COMPILE_PASS if isinstance(value, str) else bool(value)
    return None


def _stall_reason(attempt: Mapping[str, Any]) -> str | None:
    gate1 = _gate(attempt, "gate1_passed", ("gate1_ok", "compile_and_oracle_passed", "gate1"))
    if gate1 is None and attempt.get("compile_ok") is not None:
        oracle = attempt.get("oracle_passed")
        gate1 = bool(attempt["compile_ok"]) and bool(oracle if oracle is not None else True)
    if gate1 is False:
        return "gate1_compile_or_oracle_failed"
    gate2 = _gate(attempt, "gate2_passed", ("gate2_ok", "measurement_passed", "measurement_valid", "gate2"))
    if gate2 is False:
        return "gate2_measurement_guard_failed"
    speedup = _number(attempt, ("speedup_ratio", "speedup", "measured_speedup"))
    target = _number(attempt, ("min_speedup_ratio", "target_speedup_ratio", "target_speedup"))
    target = DEFAULT_MIN_SPEEDUP_RATIO if target is None else target
    if speedup is None:
        return "speedup_not_measured"
    if speedup < target:
        return f"speedup_below_target:{speedup:.6g}<{target:.6g}"
    reached = attempt.get("gate_reached")
    if isinstance(reached, int) and reached < 4:
        return f"gate{reached}_failed"
    return None


def should_escalate_to_rsi(
    l3_attempts: Sequence[Mapping[str, Any]], *, max_stalled_attempts: int = 2
) -> tuple[bool, str]:
    """Escalate after consecutive L3 attempts fail gates or miss target speedup."""
    if type(max_stalled_attempts) is not int or max_stalled_attempts < 1:
        raise ValueError("max_stalled_attempts must be a positive integer")
    if not l3_attempts:
        return False, "no_l3_attempts_recorded"
    stalled, reason = 0, "l3_workflow_progressing"
    for attempt in l3_attempts:
        attempt_reason = _stall_reason(attempt)
        if attempt_reason is None:
            stalled, reason = 0, "l3_workflow_progressing"
        else:
            stalled, reason = stalled + 1, attempt_reason
    if stalled >= max_stalled_attempts:
        return True, f"l3_stalled_for_{stalled}_attempts:{reason}"
    return False, f"l3_stalled_{stalled}_of_{max_stalled_attempts}:{reason}"


def _rules(unit: Mapping[str, Any], *keys: str) -> list[Any]:
    return [item for key in keys for item in _seq(unit.get(key))]


def _conditional_avoid(entry: Mapping[str, Any]) -> list[str]:
    rules = [
        json.dumps(rule, sort_keys=True, ensure_ascii=False, default=str)
        if isinstance(rule, Mapping) else str(rule).strip()
        for rule in _seq(entry.get("conditional_avoid"))
    ]
    rules = [rule for rule in rules if rule]
    if rules:
        return rules
    candidate = str(entry.get("candidate_id") or entry.get("run_id") or "unknown-candidate")
    direction = str(entry.get("mutation_summary") or entry.get("direction") or "").strip()
    failure = str(entry.get("failure_reason") or entry.get("gate_reached") or "").strip()
    detail = "; ".join(item for item in (direction, failure) if item)
    return [f"avoid {candidate}: {detail}" if detail else f"avoid {candidate}"]


def build_debiased_rsi_context(
    *,
    goal: str,
    retrieved_units: Sequence[Mapping[str, Any]],
    bottleneck_summary: Mapping[str, Any],
    ledger_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mount L1/L2/L4 evidence while excluding routine L3 heuristic workflow bias."""
    stripped: list[str] = []
    units: dict[str, list[dict[str, Any]]] = {"L1": [], "L2": [], "L4": []}
    axioms: list[Any] = []
    measurement: list[Any] = []
    inherited: list[Any] = []
    baselines: list[Any] = []
    observations: list[Any] = []
    operators: list[Any] = []
    for index, unit in enumerate(retrieved_units):
        tier, identity = _tier(unit), _unit_id(unit, index)
        if tier == "L3":
            stripped.append(identity)
            continue
        if tier in units:
            units[tier].append(dict(unit))
            for parent in _rules(unit, "inherited_rules"):
                if not isinstance(parent, Mapping) or _tier(parent) != "L3":
                    if parent not in inherited:
                        inherited.append(parent)
        if tier == "L1":
            axioms.extend(_rules(unit, "axioms", "foundation_axioms"))
            measurement.extend(_rules(unit, "measurement_rules", "measurement_axioms"))
            if "foundation_baseline" in unit:
                baselines.append(unit["foundation_baseline"])
            elif unit.get("route") == "foundation_baseline":
                baselines.append(dict(unit))
            foundation = unit.get("foundation")
            if isinstance(foundation, Mapping):
                axioms.extend(_rules(foundation, "axioms", "foundation_axioms"))
                measurement.extend(_rules(foundation, "measurement_rules"))
                inherited.extend(_rules(foundation, "inherited_rules"))
        elif tier == "L2":
            observations.extend(
                _rules(unit, "observations", "bottleneck_observations", "bottleneck")
            )
        elif tier == "L4":
            operators.extend(_rules(unit, "mutators", "operators", "mutation_operators"))
    summary = dict(bottleneck_summary)
    observations = _seq(summary.get("observations")) + observations
    baseline: Any = baselines[0] if len(baselines) == 1 else baselines or None
    avoid = [
        rule
        for entry in ledger_entries
        if str(entry.get("verdict", "")).strip().lower() == "refuted"
        for rule in _conditional_avoid(entry)
    ]
    foundation = {
        "units": units["L1"],
        "axioms": axioms,
        "measurement_rules": measurement,
        "inherited_rules": inherited,
        "foundation_baseline": baseline,
    }
    bottleneck = {"units": units["L2"], "summary": summary, "observations": observations}
    mutators = {"units": units["L4"], "operators": operators}
    return {
        "schema_version": SCHEMA_VERSION, "goal": goal, "mounted_tiers": ["L1", "L2", "L4"],
        "stripped_tiers": ["L3"], "stripped_l3_ids": stripped, "l1_foundation": foundation,
        "l2_bottleneck": bottleneck, "l4_mutators": mutators, "foundation_axioms": axioms,
        "measurement_rules": measurement, "inherited_rules": inherited,
        "foundation_baseline": baseline, "bottleneck_summary": summary,
        "bottleneck_observations": observations, "mutator_operators": operators,
        "conditional_avoid": list(dict.fromkeys(avoid)),
    }


def _compile_proof(candidate: Mapping[str, Any]) -> bool | None:
    for key in ("compile_ok", "compile_passed", "compiled", "compile_success"):
        if key in candidate:
            return bool(candidate[key])
    proof = candidate.get("compile_proof")
    if isinstance(proof, bool):
        return proof
    if isinstance(proof, Mapping):
        for key in ("passed", "ok", "succeeded", "success", "status"):
            if key in proof:
                value = proof[key]
                return str(value).lower() in _COMPILE_PASS if key == "status" else bool(value)
    if candidate.get("compile_status") is not None:
        return str(candidate["compile_status"]).lower() in _COMPILE_PASS
    return None


def _item(candidate_id: str, phase: str, resources: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "phase": phase,
        "parallel": phase == "compile",
        "resources": resources,
        **extra,
    }


def plan_rsi_compile_and_timing(
    candidates: Sequence[Mapping[str, Any]], *, npu_device: int = 0, max_compile_slots: int = 3
) -> dict[str, Any]:
    """Plan a CPU compile wave, then a serial exclusive-device timing wave."""
    if type(max_compile_slots) is not int or max_compile_slots < 1:
        raise ValueError("max_compile_slots must be a positive integer")
    if type(npu_device) is not int or npu_device < 0:
        raise ValueError("npu_device must be a non-negative integer")
    ids = [str(c.get("candidate_id", c.get("id", c.get("unit_id"))) or f"candidate-{i}")
           for i, c in enumerate(candidates)]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    proofs = {identity: _compile_proof(candidate) for identity, candidate in zip(ids, candidates)}
    seed = {"candidate_ids": ids, "npu_device": npu_device, "max_compile_slots": max_compile_slots}
    digest = hashlib.sha256(json.dumps(seed, sort_keys=True).encode()).hexdigest()[:16]
    state_dir = str(Path(tempfile.gettempdir()) / f"mindie-bench-rsi-compile-{digest}")
    compile_items = [_item(identity, "compile", {"npu_count": 0},
                           compile_proof_succeeded=proofs[identity] is True) for identity in ids]
    waves = [ids[offset : offset + max_compile_slots]
             for offset in range(0, len(ids), max_compile_slots)]
    timing_ids = [identity for identity in ids if proofs[identity] is True]
    owner = {"service": "CoordinatorService", "role": "single_progress_owner",
             "state_dir": state_dir}
    compile_phase = {
        "phase": "compile", "name": "compile", "phase_index": 1,
        "mode": "parallel_cpu_compile", "parallel": True,
        "resources": {"npu_count": 0}, "max_parallel": max_compile_slots,
        "max_compile_slots": max_compile_slots, "progress_owner": "CoordinatorService",
        "progress_owner_detail": owner, "coordinator": owner,
        "coordinator_service": "CoordinatorService",
        "state_dir": state_dir, "candidates": ids, "items": compile_items, "waves": waves,
        "initial_wave": waves[0] if waves else [],
    }
    timing_phase = {
        "phase": "timing", "name": "timing", "phase_index": 2,
        "mode": "serial_exclusive_npu_timing",
        "parallel": False, "resources": {"devices": [npu_device]}, "device": npu_device,
        "depends_on": "compile_proof_succeeded", "candidates": timing_ids,
        "items": [_item(identity, "timing", {"devices": [npu_device]}, exclusive_npu=True)
                  for identity in timing_ids],
        "skipped_candidates": [
            {"candidate_id": identity, "reason": "compile_proof_not_succeeded"
             if proofs[identity] is False else "compile_proof_missing"}
            for identity in ids if proofs[identity] is not True
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION, "npu_device": npu_device,
        "max_compile_slots": max_compile_slots, "state_dir": state_dir,
        "progress_owner": "CoordinatorService", "progress_owner_detail": owner,
        "compile_wave": compile_phase, "timing_wave": timing_phase,
        "phase1_compile": compile_phase, "phase2_timing": timing_phase,
        "phases": [compile_phase, timing_phase],
    }


def _run_id(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False)
    return "rsi-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def evaluate_rsi_candidate_gates(
    *,
    candidate_id: str,
    mutation_summary: str,
    conditions: Mapping[str, Any],
    compile_ok: bool,
    oracle_passed: bool,
    measurement_passed: bool,
    speedup_ratio: float,
    min_speedup_ratio: float = DEFAULT_MIN_SPEEDUP_RATIO,
    ablation_retained: bool = True,
    failure_reason: str = "",
) -> dict[str, Any]:
    """Evaluate Gate 1..4 and return one supported/refuted ledger entry."""
    try:
        speedup, minimum = float(speedup_ratio), float(min_speedup_ratio)
    except (TypeError, ValueError) as error:
        raise ValueError("speedup_ratio and min_speedup_ratio must be numeric") from error
    if not math.isfinite(speedup) or speedup < 0 or not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("speedup ratios must be finite and non-negative with a positive target")
    gate1 = [reason for ok, reason in
             ((compile_ok, "compile_failed"), (oracle_passed, "oracle_freezer_rejected")) if not ok]
    gates = {
        "gate1_compile_oracle": gate1,
        "gate2_measurement_guard": [] if measurement_passed else ["measurement_guard_rejected"],
        "gate3_ablation_gate": [] if ablation_retained else ["ablation_not_retained"],
        "gate4_distill_or_refute": (
            [] if speedup >= minimum else [f"speedup_below_target:{speedup:.6g}<{minimum:.6g}"]
        ),
    }
    failed = next(((index, reasons) for index, reasons in enumerate(gates.values(), 1) if reasons), None)
    gate_reached, gate_reasons = failed or (4, [])
    verdict = "supported" if failed is None else "refuted"
    normalized = dict(conditions)
    reason_text = "; ".join(gate_reasons)
    if failure_reason.strip():
        reason_text = f"{reason_text}; {failure_reason.strip()}" if reason_text else failure_reason.strip()
    avoid = [f"avoid {candidate_id} ({mutation_summary.strip() or 'unspecified mutation'}): "
             f"{reason_text or 'gate_failed'}"] if verdict == "refuted" else []
    identity = {
        "candidate_id": candidate_id, "mutation_summary": mutation_summary,
        "conditions": normalized, "compile_ok": compile_ok, "oracle_passed": oracle_passed,
        "measurement_passed": measurement_passed, "speedup_ratio": speedup,
        "min_speedup_ratio": minimum, "ablation_retained": ablation_retained,
        "failure_reason": failure_reason, "verdict": verdict,
    }
    return {
        "schema_version": SCHEMA_VERSION, "run_id": _run_id(identity),
        "candidate_id": candidate_id, "mutation_summary": mutation_summary,
        "verdict": verdict, "gate_reached": gate_reached, "speedup_ratio": speedup,
        "min_speedup_ratio": minimum, "conditions": normalized,
        "gates": {
            name: {"passed": not reasons, "reasons": reasons} for name, reasons in gates.items()
        },
        "conditional_avoid": avoid, "distill_ready": verdict == "supported",
        "failure_reason": (failure_reason.strip() if failure_reason.strip()
                           else reason_text if verdict == "refuted" else ""),
    }


def _artifact(path: Path, name: str) -> Path:
    return path / name if path.is_dir() else path


def _append_json_line(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"ledger line {number} is not a JSON object")
            entries.append(value)
    return entries


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _speedup(entry: Mapping[str, Any]) -> float:
    try:
        value = float(entry.get("speedup_ratio", 0.0))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _line(value: Any) -> str:
    return " ".join(str(value).split())


def _render_progress(entries: Sequence[Mapping[str, Any]], *, campaign_id: str, goal: str) -> str:
    supported = [entry for entry in entries if str(entry.get("verdict", "")).lower() == "supported"]
    refuted = [entry for entry in entries if str(entry.get("verdict", "")).lower() == "refuted"]
    supported.sort(key=lambda entry: (-_speedup(entry), str(entry.get("candidate_id", ""))))
    lines = [f"# RSI Progress: {campaign_id}", "", "## Goal", "",
             goal.strip() or "(not recorded)", "", "## Best Supported Candidate", ""]
    if supported:
        best = supported[0]
        lines.extend([f"- candidate: `{best.get('candidate_id', 'unknown')}`",
                      f"- speedup_ratio: `{_speedup(best):.6g}`",
                      f"- run_id: `{best.get('run_id', 'unknown')}`",
                      f"- mutation: {_line(best.get('mutation_summary', '')) or '(not recorded)'}"])
    else:
        lines.append("- none (no candidate has passed all four gates)")
    lines.extend(["", "## Active Hypotheses", ""])
    lines.extend([f"- `{entry.get('candidate_id', 'unknown')}`: "
                  f"{_line(entry.get('mutation_summary', '')) or '(no mutation summary)'} "
                  f"(speedup `{_speedup(entry):.6g}`, run `{entry.get('run_id', 'unknown')}`)"
                  for entry in supported] or ["- none"])
    lines.extend(["", "## Refuted Directions", ""])
    directions = sorted({rule for entry in refuted for rule in _conditional_avoid(entry)})
    lines.extend([f"- {direction}" for direction in directions] or ["- none"])
    return "\n".join(lines).rstrip() + "\n"


def append_ledger_and_render_progress(
    ledger_path: Path,
    progress_path: Path,
    entry: Mapping[str, Any],
    *,
    campaign_id: str,
    goal: str,
) -> None:
    """Atomically append one ledger result and rewrite deterministic PROGRESS.md."""
    ledger = _artifact(Path(ledger_path), "experiment-ledger.jsonl")
    _append_json_line(ledger, entry)
    text = _render_progress(_read_ledger(ledger), campaign_id=campaign_id, goal=goal)
    _atomic_write_text(_artifact(Path(progress_path), "PROGRESS.md"), text)
