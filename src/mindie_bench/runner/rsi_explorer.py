"""L4 recursive self-improvement context, scheduling, gates and ledger.

[Role]
Turn stalled L3 heuristic work into L4 RSI: mount L1 foundation, L2 bottleneck
and L4 mutators, schedule CPU-only compiles before exclusive NPU timing, then
record four-gate outcomes.

[Invariants]
- L3 workflow units never enter L4 context; only IDs survive in `stripped_l3_ids`.
- A new observation or a located divergence layer is information progress, not
  a stalled attempt; only consecutive non-informative attempts escalate.
- Negative experience applies only to its observed conditions. Invalid
  measurements and valid sub-target speedups are not refutations; only
  supported and sub-target-valid candidates are `distill_ready`.
- Speedup is the default objective. Correctness, localization and diagnosis
  tasks report `goal_met`/`diagnostic_gain` instead of a synthetic ratio.
- Compiles are CPU-only (`npu_count == 0`); NPU timing is serial, exclusive to
  the assigned device lease, and admitted only for proved compiles.
- The ledger is append-only; PROGRESS.md is deterministic and atomically replaced.

[Timeouts & Locks]
- Pure planning/evaluation helpers perform no waits and take no locks.
- The ledger owns one short O_APPEND write and one atomic markdown replacement;
  callers own all experiment and NPU timeouts.

[Side-Effect Policy]
- Only `append_ledger_and_render_progress` writes: the JSONL ledger and PROGRESS.md.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "mindie-bench-rsi/1"
DEFAULT_MIN_SPEEDUP_RATIO = 1.05
RETURN_TIERS = frozenset({"L1", "L2", "L3", "L4"})
_COMPILE_PASS = frozenset({"compiled", "ok", "pass", "passed", "success", "succeeded"})


def _seq(value: Any) -> list[Any]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence) else [value]

def _unit_id(unit: Mapping[str, Any], index: int) -> str:
    raw = unit.get("unit_id", unit.get("id", unit.get("workflow_id")))
    return str(raw) if raw not in (None, "") else f"unit-{index}"


def _tier(unit: Mapping[str, Any]) -> str:
    return str(unit.get("tier", "")).strip().upper()

def should_escalate_to_rsi(
    l3_attempts: Sequence[Mapping[str, Any]], *, max_stalled_attempts: int = 2
) -> tuple[bool, str]:
    """Escalate on completed, non-informative experiments; repair failures first."""
    if type(max_stalled_attempts) is not int or max_stalled_attempts < 1:
        raise ValueError("max_stalled_attempts must be a positive integer")
    if not l3_attempts:
        return False, "no_l3_attempts_recorded"
    stalled, reason = 0, "l3_workflow_progressing"
    incomplete_outcomes = {"compile_failed", "measurement_invalid", "ablation_incomplete"}
    # Polling or resuming one remote run cannot create a second experiment.
    for attempt in {row["run_id"]: row for row in l3_attempts}.values():
        outcome = attempt.get("outcome")
        validity = attempt.get("axes", {}).get("evidence_validity", "valid")
        if attempt.get("new_observation") is True or attempt.get("divergent_layer_identified") is True:
            stalled, reason = 0, "l3_information_progress"
        elif (outcome in incomplete_outcomes or validity != "valid"
              or (attempt["verdict"] == "inconclusive"
                  and outcome != "speedup_within_noise")):
            # Only genuinely incomplete execution asks for repair. A completed
            # below-target or within-noise run is a valid, non-attaining result.
            stalled, reason = 0, "l3_execution_incomplete"
        elif attempt["verdict"] == "supported":
            stalled, reason = 0, "l3_workflow_progressing"
        else:
            stalled, reason = stalled + 1, attempt.get("failure_reason") or "speedup_below_target"
    if stalled >= max_stalled_attempts:
        return True, f"l3_stalled_for_{stalled}_attempts:{reason}"
    return False, f"l3_stalled_{stalled}_of_{max_stalled_attempts}:{reason}"


def _rules(unit: Mapping[str, Any], *keys: str) -> list[Any]:
    return [item for key in keys for item in _seq(unit.get(key))]


def _same_conditions(required: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Ledger conditions are measured facts, not version-range rule expressions."""
    return bool(required) and all(
        key in observed and value is not None and value != "unknown"
        and (_same_conditions(value, observed[key])
             if isinstance(value, Mapping) and isinstance(observed[key], Mapping)
             else value == observed[key])
        for key, value in required.items()
    )


def _conditional_avoid(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    invalid = entry["verdict"] != "refuted" or not entry["conditions"]
    invalid |= entry.get("outcome") == "speedup_below_target_valid_evidence"
    if invalid:
        return []
    return [{"candidate_id": entry["candidate_id"], "run_id": entry["run_id"],
             "conditions": dict(entry["conditions"]), "mutation_summary": entry["mutation_summary"],
             "reason": entry["failure_reason"]}]


def build_debiased_rsi_context(
    *, goal: str, retrieved_units: Sequence[Mapping[str, Any]],
    bottleneck_summary: Mapping[str, Any], ledger_entries: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Any], native_subcontext: bool = False,
) -> dict[str, Any]:
    """Mount L1/L2/L4 evidence while excluding routine L3 heuristic workflow bias."""
    stripped: list[str] = []
    units: dict[str, list[dict[str, Any]]] = {"L1": [], "L2": [], "L4": []}
    axioms, measurement, inherited = [], [], []
    baselines, observations, operators = [], [], []
    mounted: set[tuple[str, Any]] = set()

    def mount(unit: Mapping[str, Any]) -> dict[str, Any] | None:
        # Parents are returned inside query rows. Preserve their foundation even
        # when their L3 child is removed, and remove L3 from every returned copy.
        cleaned = dict(unit)
        if "inherited_rules" in unit:
            parents = []
            for parent in _seq(unit["inherited_rules"]):
                kept = mount(parent) if isinstance(parent, Mapping) else parent
                if kept is not None:
                    parents.append(kept)
            cleaned["inherited_rules"] = parents
        if isinstance(unit.get("foundation"), Mapping):
            cleaned["foundation"] = mount(unit["foundation"])
        tier, identity = _tier(unit), _unit_id(unit, len(mounted))
        if tier == "L3":
            if identity not in stripped:
                stripped.append(identity)
            return None
        key = (identity, unit.get("revision"))
        if tier in units and key not in mounted:
            units[tier].append(cleaned)
            mounted.add(key)
        return cleaned

    for unit in retrieved_units:
        mount(unit)
    for tier, mounted_units in units.items():
        for unit in mounted_units:
            for parent in _rules(unit, "inherited_rules"):
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
                observations.extend(_rules(unit, "observations", "bottleneck_observations", "bottleneck"))
            elif tier == "L4":
                operators.extend(_rules(unit, "mutators", "operators", "mutation_operators"))
    summary = dict(bottleneck_summary)
    observations = _seq(summary.get("observations")) + observations
    baseline: Any = baselines[0] if len(baselines) == 1 else baselines or None
    avoid = [
        rule
        for entry in {row["run_id"]: row for row in ledger_entries}.values()
        if entry["verdict"] == "refuted" and _same_conditions(entry["conditions"], conditions)
        for rule in _conditional_avoid(entry)
    ]
    foundation = {"units": units["L1"], "axioms": axioms, "measurement_rules": measurement,
                  "inherited_rules": inherited, "foundation_baseline": baseline}
    bottleneck = {"units": units["L2"], "summary": summary, "observations": observations}
    mutators = {"units": units["L4"], "operators": operators}
    return {
        "schema_version": SCHEMA_VERSION, "goal": goal, "mounted_tiers": ["L1", "L2", "L4"],
        "stripped_tiers": ["L3"], "stripped_l3_ids": stripped, "l1_foundation": foundation,
        "l2_bottleneck": bottleneck, "l4_mutators": mutators, "foundation_axioms": axioms,
        "measurement_rules": measurement, "inherited_rules": inherited,
        "foundation_baseline": baseline, "bottleneck_summary": summary,
        "bottleneck_observations": observations, "mutator_operators": operators,
        "conditional_avoid": avoid,
        "isolation_mode": "native_subcontext" if native_subcontext else "in_session_strategy_shift",
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
    return {"candidate_id": candidate_id, "phase": phase, "parallel": phase == "compile",
            "resources": resources, **extra}


def _device_lease(npu_device: int, assigned_devices: Sequence[int] | None) -> list[int]:
    """A timing wave uses the full assigned lease and never silently falls back to NPU 0."""
    devices = list(assigned_devices or [])
    if devices:
        if any(type(device) is not int or device < 0 for device in devices) or len(set(devices)) != len(devices):
            raise ValueError("assigned_devices must be unique non-negative integers")
        return devices
    if type(npu_device) is not int or npu_device < 0:
        raise ValueError("npu_device must be a non-negative integer")
    return [npu_device]


def plan_rsi_compile_and_timing(
    candidates: Sequence[Mapping[str, Any]], *, state_dir: str | Path,
    npu_device: int = 0, max_compile_slots: int = 3,
    assigned_devices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Plan a CPU compile wave, then a serial exclusive-device timing wave."""
    if type(max_compile_slots) is not int or max_compile_slots < 1:
        raise ValueError("max_compile_slots must be a positive integer")
    devices = _device_lease(npu_device, assigned_devices)
    primary = devices[0]
    ids = [str(c.get("candidate_id", c.get("id", c.get("unit_id"))) or f"candidate-{i}")
           for i, c in enumerate(candidates)]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    proofs = {identity: _compile_proof(candidate) for identity, candidate in zip(ids, candidates)}
    state_dir = str(state_dir)
    compile_items = [_item(identity, "compile", {"npu_count": 0},
                           compile_proof_succeeded=proofs[identity] is True) for identity in ids]
    waves = [ids[offset : offset + max_compile_slots]
             for offset in range(0, len(ids), max_compile_slots)]
    timing_ids = [identity for identity in ids if proofs[identity] is True]
    owner = {"service": "CoordinatorService", "role": "single_progress_owner",
             "state_dir": state_dir}
    compile_phase = {
        "phase": "compile", "name": "compile", "phase_index": 1,
        "mode": "parallel_cpu_compile", "parallel": True, "resources": {"npu_count": 0},
        "max_parallel": max_compile_slots, "max_compile_slots": max_compile_slots,
        "progress_owner": "CoordinatorService", "progress_owner_detail": owner,
        "coordinator": owner, "coordinator_service": "CoordinatorService",
        "state_dir": state_dir, "candidates": ids, "items": compile_items, "waves": waves,
        "initial_wave": waves[0] if waves else [],
    }
    timing_phase = {
        "phase": "timing", "name": "timing", "phase_index": 2,
        "mode": "serial_exclusive_npu_timing", "parallel": False,
        "resources": {"devices": devices}, "device": primary,
        "depends_on": "compile_proof_succeeded", "candidates": timing_ids,
        "items": [_item(identity, "timing", {"devices": devices}, exclusive_npu=True)
                  for identity in timing_ids],
        "skipped_candidates": [
            {"candidate_id": identity, "reason": "compile_proof_not_succeeded"
             if proofs[identity] is False else "compile_proof_missing"}
            for identity in ids if proofs[identity] is not True
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION, "npu_device": primary, "assigned_devices": devices,
        "max_compile_slots": max_compile_slots, "state_dir": state_dir,
        "progress_owner": "CoordinatorService", "progress_owner_detail": owner,
        "compile_wave": compile_phase, "timing_wave": timing_phase,
        "phase1_compile": compile_phase, "phase2_timing": timing_phase,
        "phases": [compile_phase, timing_phase],
    }


def evaluate_exploration_axes(
    *, compile_ok: bool, oracle_passed: bool, measurement_passed: bool,
    speedup_ratio: float | None, min_speedup_ratio: float = DEFAULT_MIN_SPEEDUP_RATIO,
    ablation_retained: bool | None = None, preserve_subtarget_evidence: bool = True,
    noise_margin: float = 0.01, return_tier: str = "L4",
    goal_type: str = "speedup", goal_met: bool | None = None,
    diagnostic_gain: bool = False,
) -> dict[str, Any]:
    """Separate evidence validity, goal attainment and reuse on every outcome.

    Speedup is the default objective. Correctness, localization and non-timing
    performance tasks declare another ``goal_type`` and pass ``goal_met`` plus
    ``diagnostic_gain`` instead of inventing a synthetic speedup ratio.
    """
    if return_tier not in RETURN_TIERS:
        raise ValueError("return_tier must be one of L1, L2, L3, L4")
    if not isinstance(goal_type, str) or not goal_type.strip():
        raise ValueError("goal_type must be nonempty text")
    if not math.isfinite(noise_margin) or noise_margin < 0:
        raise ValueError("noise_margin must be finite and non-negative")
    minimum = float(min_speedup_ratio)
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("min_speedup_ratio must be finite and positive")
    normalized_goal = goal_type.strip().lower()
    speedup_goal = normalized_goal == "speedup"
    speedup = None if speedup_ratio is None else float(speedup_ratio)
    measured = (compile_ok and oracle_passed and measurement_passed
                and (not speedup_goal or (
                    speedup is not None and math.isfinite(speedup) and speedup >= 0)))
    if not compile_ok:
        state = ("compile_failed", "inconclusive", "compile_failed",
                 "not_applicable", "inconclusive", 1)
    elif not oracle_passed:
        state = ("oracle_failed", "refuted", "oracle_failed", "not_applicable",
                 "reusable_negative_boundary", 1)
    elif not measured:
        state = ("measurement_invalid", "inconclusive", "measurement_invalid",
                 "not_applicable", "inconclusive", 2)
    elif not speedup_goal:
        if goal_met is True:
            # Correctness and localization conclusions are reusable observations
            # (contracts, branches, verification steps), not L4 mutators.
            state = ("valid", "supported", "verified", "attained",
                     "reusable_observation", 4)
        elif diagnostic_gain:
            # A distinguishing observation or first-divergence localization is
            # reusable evidence even when the original objective is unmet.
            state = ("valid", "subtarget_valid", "diagnostic_gain_valid_evidence",
                     "not_attained", "reusable_observation", 4)
        elif goal_met is False:
            state = ("valid", "refuted", "goal_not_met", "not_attained",
                     "reusable_negative_boundary", 4)
        else:
            state = ("valid", "inconclusive", "goal_unverified", "not_applicable",
                     "inconclusive", 4)
    elif speedup >= minimum:
        if ablation_retained is None:
            state = ("valid", "inconclusive", "ablation_pending", "not_applicable",
                     "inconclusive", 3)
        elif not ablation_retained:
            state = ("valid", "refuted", "ablation_not_retained", "not_applicable",
                     "reusable_negative_boundary", 3)
        else:
            reuse = "reusable_mutator" if return_tier == "L4" else "reusable_observation"
            state = ("valid", "supported", "verified", "attained", reuse, 4)
    elif (preserve_subtarget_evidence and speedup > 1.0 + noise_margin
          and ablation_retained is not False):
        state = ("valid", "subtarget_valid", "speedup_below_target_valid_evidence",
                 "not_attained", "reusable_observation", 4)
    elif ablation_retained is None:
        state = ("valid", "inconclusive", "ablation_pending", "not_applicable",
                 "inconclusive", 3)
    elif preserve_subtarget_evidence and 1.0 < speedup <= 1.0 + noise_margin:
        state = ("valid", "inconclusive", "speedup_within_noise",
                 "not_attained", "inconclusive", 4)
    else:
        state = ("valid", "refuted", "speedup_below_target",
                 "not_attained", "reusable_negative_boundary", 4)
    evidence, verdict, outcome, goal, reuse, gate = state
    axes = {"evidence_validity": evidence, "goal_attainment": goal, "reusability": reuse}
    return {
        "axes": axes, **axes, "return_tier": return_tier, "verdict": verdict,
        "outcome": outcome, "gate_reached": gate, "goal_type": normalized_goal,
        "distill_ready": verdict in {"supported", "subtarget_valid"},
    }

def evaluate_rsi_candidate_gates(
    *, run_id: str, candidate_id: str, mutation_summary: str,
    conditions: Mapping[str, Any], compile_ok: bool, oracle_passed: bool,
    measurement_passed: bool, speedup_ratio: float | None,
    min_speedup_ratio: float = DEFAULT_MIN_SPEEDUP_RATIO,
    ablation_retained: bool | None = None, failure_reason: str = "",
    preserve_subtarget_evidence: bool = True, noise_margin: float = 0.01,
    return_tier: str = "L4",
) -> dict[str, Any]:
    """Judge observed results using the executor's run ID, without synthetic proof."""
    result = evaluate_exploration_axes(
        compile_ok=compile_ok, oracle_passed=oracle_passed,
        measurement_passed=measurement_passed, speedup_ratio=speedup_ratio,
        min_speedup_ratio=min_speedup_ratio, ablation_retained=ablation_retained,
        preserve_subtarget_evidence=preserve_subtarget_evidence,
        noise_margin=noise_margin, return_tier=return_tier,
    )
    minimum = float(min_speedup_ratio)
    speedup = None if speedup_ratio is None else float(speedup_ratio)
    measured = (compile_ok and oracle_passed and measurement_passed
                and speedup is not None and math.isfinite(speedup) and speedup >= 0)
    # Gates stop at the first incomplete or failed stage. A later stage did not
    # pass merely because its caller supplied a default True value.
    checks = [
        ("gate1_compile_oracle", compile_ok and oracle_passed),
        ("gate2_measurement_guard", measured),
        ("gate3_ablation_gate", ablation_retained),
        ("gate4_distill_or_refute", measured and speedup >= minimum),
    ]
    gates = {name: {"passed": None, "reasons": []} for name, _ in checks}
    verdict, outcome, gate_reached = result["verdict"], result["outcome"], result["gate_reached"]
    for number, (name, passed) in enumerate(checks, 1):
        if number < gate_reached:
            gates[name] = {"passed": passed, "reasons": []}
        elif number == gate_reached:
            gates[name] = {"passed": passed, "reasons": [] if passed else [outcome]}
            break
    entry = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "candidate_id": candidate_id, "mutation_summary": mutation_summary,
        "verdict": verdict, "outcome": outcome, "gate_reached": gate_reached,
        "speedup_ratio": speedup if measured else None,
        "min_speedup_ratio": minimum, "conditions": dict(conditions), "gates": gates,
        "axes": result["axes"], "return_tier": result["return_tier"],
        "distill_ready": result["distill_ready"],
        "failure_reason": failure_reason.strip() or (outcome if verdict != "supported" else ""),
    }
    entry["conditional_avoid"] = _conditional_avoid(entry)
    return entry


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
    entries = list({entry["run_id"]: entry for entry in entries}.values())
    supported = [entry for entry in entries if str(entry.get("verdict", "")).lower() == "supported"]
    subtarget = [entry for entry in entries if entry.get("verdict") == "subtarget_valid"]
    refuted = [entry for entry in entries if str(entry.get("verdict", "")).lower() == "refuted"]
    incomplete = [entry for entry in entries if entry.get("verdict") == "inconclusive"]
    supported.sort(key=lambda entry: (-_speedup(entry), str(entry.get("candidate_id", ""))))
    subtarget.sort(key=lambda entry: (-_speedup(entry), str(entry.get("candidate_id", ""))))
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
    hypotheses = supported + subtarget
    lines.extend([f"- `{entry.get('candidate_id', 'unknown')}`: "
                  f"{_line(entry.get('mutation_summary', '')) or '(no mutation summary)'} "
                  f"(speedup `{_speedup(entry):.6g}`, run `{entry.get('run_id', 'unknown')}`)"
                  for entry in hypotheses] or ["- none"])
    lines.extend(["", "## Refuted Directions", ""])
    directions = [f"avoid {rule['candidate_id']} when "
                  f"{json.dumps(rule['conditions'], sort_keys=True, ensure_ascii=False)}: "
                  f"{_line(rule['reason'])} (run `{rule['run_id']}`)"
                  for entry in refuted for rule in _conditional_avoid(entry)]
    lines.extend([f"- {direction}" for direction in directions] or ["- none"])
    lines.extend(["", "## Experiments To Complete", ""])
    lines.extend([f"- `{entry['candidate_id']}`: {_line(entry['failure_reason'])} "
                  f"(run `{entry['run_id']}`)" for entry in incomplete] or ["- none"])
    return "\n".join(lines).rstrip() + "\n"


def append_ledger_and_render_progress(
    ledger_path: Path, progress_path: Path, entry: Mapping[str, Any],
    *, campaign_id: str, goal: str,
) -> None:
    """Atomically append one ledger result and rewrite deterministic PROGRESS.md."""
    ledger = _artifact(Path(ledger_path), "experiment-ledger.jsonl")
    _append_json_line(ledger, entry)
    text = _render_progress(_read_ledger(ledger), campaign_id=campaign_id, goal=goal)
    _atomic_write_text(_artifact(Path(progress_path), "PROGRESS.md"), text)
