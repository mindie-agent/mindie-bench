"""Shared campaign adapters: fact packs, candidate plans, runners and G0-G3 gates.

[Role]
Keep `rsi_campaign` focused on L4-02/L4-04 verdicts by isolating the mechanical
translations between knowledge fact packs, compile/timing callbacks, candidate
plan payloads and QualificationGateRunner receipts.

[Invariants]
- A fact pack always carries `mindie-rsi-subagent-pack/1` and never mounts L3
  heuristic bodies.
- Candidate plans are normalized, de-duplicated by candidate identity, and a
  compile proof carried by a plan is never silently treated as missing.
- A compatibility shim maps methods/attributes only; it never fabricates a
  compile, timing, correctness or qualification result.
- Qualification gate states are read verbatim; failed/unknown is never promoted
  to completed.

[Timeouts & Locks]
- Pure in-memory normalization. Callers own every compile, timing and gate
  timeout; this module never starts a process, waits or locks.

[Side-Effect Policy]
- No file, network, knowledge or device writes; values returned are derived
  copies for the campaign caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_SUBAGENT_PACK = "mindie-rsi-subagent-pack/1"
_PASS_WORDS = frozenset({"1", "true", "yes", "ok", "pass", "passed", "success", "succeeded", "completed"})
_GATE_OK = frozenset({"completed", "passed", "pass", "ok", "success", "succeeded"})
_GATE_FAILED = frozenset({"failed", "fail", "error", "regressed"})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    return [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence) else [value]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _PASS_WORDS if isinstance(value, str) else bool(value)


def build_subagent_fact_pack(
    *,
    context: Mapping[str, Any],
    goal: str,
    retrieved_units: Sequence[Mapping[str, Any]] = (),
    bottleneck_summary: Mapping[str, Any] | None = None,
    ledger_entries: Sequence[Mapping[str, Any]] = (),
    conditions: Mapping[str, Any] | None = None,
    native_subcontext: bool = False,
) -> dict[str, Any]:
    """Return the stable `mindie-rsi-subagent-pack/1` fact pack for either host."""
    _ = (retrieved_units, bottleneck_summary)
    shared = dict(conditions or {})
    foundation = _mapping(context.get("l1_foundation"))
    bottleneck = _mapping(context.get("l2_bottleneck"))
    mutators = _mapping(context.get("l4_mutators"))
    attempts = [
        {key: _mapping(entry).get(key) for key in
         ("run_id", "candidate_id", "verdict", "outcome", "conditions", "failure_reason", "mutation_summary")}
        for entry in ledger_entries or ()
    ]
    avoid = [dict(rule) for rule in _seq(context.get("conditional_avoid")) if isinstance(rule, Mapping)]
    facts = [
        *({"kind": "foundation_axiom", "value": value} for value in _seq(context.get("foundation_axioms"))),
        *({"kind": "measurement_rule", "value": value} for value in _seq(context.get("measurement_rules"))),
        *({"kind": "inherited_rule", "value": value} for value in _seq(context.get("inherited_rules"))),
        *({"kind": "bottleneck_observation", "value": value}
          for value in _seq(context.get("bottleneck_observations"))),
        *({"kind": "mutator_operator", "value": value} for value in _seq(context.get("mutator_operators"))),
    ]
    return {
        "schema_version": SCHEMA_SUBAGENT_PACK,
        "schema": SCHEMA_SUBAGENT_PACK,
        "source": "mindie-bench.runner.rsi_explorer",
        "goal": goal,
        "conditions": shared,
        "isolation_mode": context.get(
            "isolation_mode", "native_subcontext" if native_subcontext else "in_session_strategy_shift"
        ),
        "native_subcontext": bool(native_subcontext),
        "foundation": foundation,
        "bottleneck": bottleneck,
        "mutators": mutators,
        "failed_attempts": attempts,
        "conditional_avoid": avoid,
        "stripped_l3_ids": list(_seq(context.get("stripped_l3_ids"))),
        "facts": facts,
        "fact_lines": [f"{fact['kind']}: {fact['value']}" for fact in facts],
        "notes": [
            "Only L1/L2/L4 facts are mounted; L3 recommendation steps are withheld.",
            "Conditional avoidance applies only to matching measured conditions.",
        ],
    }


def _candidate_rows(plan: Any, candidates: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize a plan, phase list, or explicit candidate sequence into rows."""
    if callable(plan):
        plan = plan()
    rows: list[dict[str, Any]] = [dict(row) for row in candidates or () if isinstance(row, Mapping)]
    if isinstance(plan, Mapping):
        raw = plan.get("candidates", plan.get("candidate"))
        if raw is None:
            raw = []
            for phase in _seq(plan.get("phases")):
                raw.extend(_seq(_mapping(phase).get("items")) or _seq(_mapping(phase).get("candidates")))
        if isinstance(raw, Mapping):
            raw = [raw]
        for item in _seq(raw):
            if isinstance(item, str):
                rows.append({"candidate_id": item})
            elif isinstance(item, Mapping):
                row = dict(item)
                row.setdefault("candidate_id", row.get("id") or row.get("name"))
                if "compile_proof_succeeded" in row and "compile_ok" not in row:
                    row["compile_ok"] = bool(row["compile_proof_succeeded"])
                rows.append(row)
        if not rows and any(key in plan for key in ("candidate_id", "run_id", "mutation_summary")):
            rows.append(dict(plan))
    elif isinstance(plan, Sequence) and not isinstance(plan, (str, bytes)):
        for index, item in enumerate(plan):
            if isinstance(item, Mapping):
                row = dict(item)
                row.setdefault("candidate_id", row.get("id") or f"candidate-{index}")
                rows.append(row)
            elif isinstance(item, str):
                rows.append({"candidate_id": item})
    merged: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = str(row.get("candidate_id") or row.get("id") or f"candidate-{index}")
        row["candidate_id"] = identity
        merged.setdefault(identity, {}).update(row)
    return list(merged.values())


def _runner_callbacks(runner: Any):
    """Adapt a callable, callback mapping, or compile/timing object to two callbacks."""
    if runner is None:
        return None, None
    if isinstance(runner, Mapping):
        return runner.get("compile_runner") or runner.get("compile"), (
            runner.get("timing_runner") or runner.get("timing") or runner.get("run")
        )
    compile_runner = next(
        (getattr(runner, name) for name in ("compile_runner", "compile") if callable(getattr(runner, name, None))),
        None,
    )
    timing_runner = next(
        (getattr(runner, name) for name in ("timing_runner", "timing", "run")
         if callable(getattr(runner, name, None))),
        None,
    )
    if callable(runner) and compile_runner is None and timing_runner is None:
        return (
            lambda candidate: runner({**candidate, "phase": "compile"}),
            lambda candidate: runner({**candidate, "phase": "timing"}),
        )
    return compile_runner, timing_runner


def _compile_ok(candidate: Mapping[str, Any]) -> bool:
    for key in ("compile_ok", "compile_passed", "compiled", "compile_success"):
        if key in candidate and candidate[key] is not None:
            return _flag(candidate[key])
    proof = candidate.get("compile_proof")
    if isinstance(proof, bool):
        return proof
    if isinstance(proof, Mapping):
        if "passed" in proof:
            return _flag(proof["passed"])
        if "status" in proof:
            return str(proof["status"]).lower() in _PASS_WORDS
    return str(candidate.get("compile_status", "")).lower() in _PASS_WORDS


def _source_speedup(source: Mapping[str, Any]) -> float | None:
    ratio = _number(source.get("speedup_ratio"))
    if ratio is not None:
        return ratio
    baseline = _number(source.get("baseline_latency_us", source.get("ref_latency_us")))
    optimized = _number(source.get("latency_us", source.get("opt_latency_us")))
    return baseline / optimized if baseline and optimized and optimized > 0 else None


def _gate_state(value: Any) -> str:
    raw = value.get("state", value.get("status")) if isinstance(value, Mapping) else getattr(value, "state", value)
    return str(getattr(raw, "value", raw)).strip().lower()


def _gate_result(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"state": str(value)}


def _run_qualification_gates(
    gate_runner: Any, specs: Any, candidate: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute declared G0-G3 specs when a runner is available; never invent states."""
    if gate_runner is None:
        return {}
    specs = specs if specs is not None else getattr(gate_runner, "gate_specs", None)
    specs = specs if specs is not None else candidate.get("gate_specs")
    if isinstance(specs, Mapping) and specs:
        run_gate = getattr(gate_runner, "run_gate", None)
        if callable(run_gate):
            return {str(gate).upper(): _gate_result(run_gate(gate, spec)) for gate, spec in specs.items()}
        if callable(gate_runner):
            return {str(gate).upper(): _gate_result(gate_runner(gate, spec)) for gate, spec in specs.items()}
        return {str(gate).upper(): _gate_result(spec) for gate, spec in specs.items()}
    run_all = getattr(gate_runner, "run_all", None)
    if not callable(run_all) and not callable(gate_runner):
        return {}
    result = run_all(candidate) if callable(run_all) else gate_runner(candidate)
    if isinstance(result, Mapping) and any(str(key).upper() in {"G0", "G1", "G2", "G3"} for key in result):
        return {str(key).upper(): _gate_result(value) for key, value in result.items()}
    payload = _gate_result(result)
    return {str(payload.get("gate") or "campaign").upper(): payload}


_ARTIFACT_EXTS = frozenset({".pyc", ".pyo", ".so", ".o", ".a", ".spv", ".cubin", ".fatbin", ".ptx", ".log"})
_FORBIDDEN_DIRS = ("__pycache__", ".ssh", ".codex", ".claude", "benchmark_results", ".git")
_CONFIG_EXTS = frozenset({".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"})
_SOURCE_EXTS = frozenset({".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".asc", ".cce", ".mlir", ".triton"})
_CODE_SIGNAL_TOKENS = ("def ", "class ", "tl.", "torch.", "triton.", "for ", "while ", "if ", "return ", "acl", "AscendC")


def classify_candidate_patch_shape(
    diff_text_or_files: str | Sequence[Any] | Mapping[str, Any] | None,
    *,
    reject_narrow_flips: bool = True,
) -> dict[str, Any]:
    """Classify candidate edits per Apple ml-rsi-inference-bench `category_gate.py`.

    Prevents gaming throughput tasks via launch-script flag edits (`pure_flag_flip`),
    scalar default literal tweaks (`config_default_flip`), or build artifacts (`artifact_only`).
    """
    files: list[tuple[str, list[str]]] = []
    if isinstance(diff_text_or_files, str):
        current_path: str | None = None
        current_lines: list[str] = []
        for raw_line in diff_text_or_files.splitlines():
            if raw_line.startswith("diff --git "):
                if current_path is not None:
                    files.append((current_path, current_lines))
                parts = raw_line.split()
                current_path = parts[-1].removeprefix("b/") if parts else "unknown"
                current_lines = []
            elif raw_line.startswith("+++ ") and not raw_line.startswith("+++ /dev/null"):
                target = raw_line[4:].strip().removeprefix("b/")
                if current_path is None:
                    current_path, current_lines = target, []
                else:
                    current_path = target
            elif (raw_line.startswith("+") or raw_line.startswith("-")) and not raw_line.startswith(("+++", "---")):
                if current_path is None:
                    current_path = "patch.py"
                current_lines.append(raw_line[1:].strip())
        if current_path is not None:
            files.append((current_path, current_lines))
    elif isinstance(diff_text_or_files, Mapping):
        for path, body in diff_text_or_files.items():
            lines = body.splitlines() if isinstance(body, str) else [str(item) for item in _seq(body)]
            files.append((str(path), lines))
    elif isinstance(diff_text_or_files, Sequence):
        for item in diff_text_or_files:
            if isinstance(item, str):
                files.append((item, []))
            elif isinstance(item, Mapping):
                path = str(item.get("path") or item.get("file") or "unknown")
                body = item.get("added_lines") or item.get("diff") or ()
                lines = body.splitlines() if isinstance(body, str) else [str(x) for x in _seq(body)]
                files.append((path, lines))

    source_files, flag_files, config_files, ignored_artifacts = [], [], [], []
    for raw_path, changed in files:
        norm = raw_path.strip().replace("\\", "/")
        lower = norm.lower()
        name = lower.rsplit("/", 1)[-1]
        dot = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        if any(seg in lower.split("/") for seg in _FORBIDDEN_DIRS) or name.startswith(".env") or dot in _ARTIFACT_EXTS:
            ignored_artifacts.append(norm)
            continue
        if dot in {".sh", ".bash", ".zsh"} or name.startswith(("launch_", "run_server", "start_")):
            flag_files.append(norm)
            continue
        if dot in _CONFIG_EXTS:
            config_files.append(norm)
            continue
        if dot == ".py" and ("config" in name or "args" in name or "settings" in name):
            has_logic = any(any(tok in line for tok in _CODE_SIGNAL_TOKENS) for line in changed)
            if changed and not has_logic:
                config_files.append(norm)
                continue
        if dot in _SOURCE_EXTS or changed:
            source_files.append(norm)
        else:
            config_files.append(norm)

    if not files:
        category = "empty"
    elif not source_files and not flag_files and not config_files:
        category = "artifact_only"
    elif source_files and not flag_files and not config_files:
        category = "source_edit"
    elif source_files:
        category = "mixed"
    elif flag_files and not config_files:
        category = "pure_flag_flip"
    else:
        category = "config_default_flip"
    rejected = bool(
        reject_narrow_flips and category in {"empty", "artifact_only", "pure_flag_flip", "config_default_flip"}
    )
    return {
        "category": category,
        "substantive_source_edit": bool(source_files),
        "rejected_by_narrow_gate": rejected,
        "source_files": source_files,
        "flag_files": flag_files,
        "config_files": config_files,
        "ignored_artifact_files": ignored_artifacts,
    }


def build_isolated_jit_cache_env(state_dir: str | any, side_or_candidate_id: str) -> dict[str, str]:
    """Return isolated JIT/compiler cache paths so paired AB/BA runs never share `.so`/Triton caches."""
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(side_or_candidate_id or "default"))
    root = f"{str(state_dir).rstrip('/')}/jit_cache_{slug}"
    return {
        "TRITON_CACHE_DIR": f"{root}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{root}/inductor",
        "ASCEND_CACHE_PATH": f"{root}/ascend_atc",
        "MINDIE_JIT_CACHE_DIR": f"{root}/mindie_jit",
    }


def evaluate_speculative_acceptance(
    candidate_metrics: Mapping[str, Any],
    pristine_metrics: Mapping[str, Any] | None = None,
    *,
    rtol: float = 0.05,
) -> dict[str, Any]:
    """Verify speculative decoding (MTP/DSpark) draft count and mean acceptance length."""
    cand = _mapping(candidate_metrics)
    prist = _mapping(pristine_metrics)
    cand_drafts = _number(cand.get("num_drafts", cand.get("draft_steps", cand.get("spec_steps"))))
    prist_drafts = _number(prist.get("num_drafts", prist.get("draft_steps", prist.get("spec_steps"))))
    cand_acc = _number(cand.get("mean_acceptance_length", cand.get("acceptance_length", cand.get("accept_len"))))
    prist_acc = _number(prist.get("mean_acceptance_length", prist.get("acceptance_length", prist.get("accept_len"))))
    if cand_drafts is not None and cand_drafts <= 0 and (prist_drafts is None or prist_drafts > 0):
        return {
            "acceptance_ok": False, "reason": "no_drafts", "ratio": 0.0,
            "cand_acceptance_length": cand_acc, "pristine_acceptance_length": prist_acc,
        }
    if cand_acc is not None and prist_acc is not None and prist_acc > 0:
        ratio = cand_acc / prist_acc
        ok = ratio >= (1.0 - max(0.0, float(rtol))) - 1e-9
        return {
            "acceptance_ok": ok, "reason": "ok" if ok else "acceptance_degraded", "ratio": ratio,
            "cand_acceptance_length": cand_acc, "pristine_acceptance_length": prist_acc,
        }
    return {
        "acceptance_ok": True, "reason": "ok", "ratio": None,
        "cand_acceptance_length": cand_acc, "pristine_acceptance_length": prist_acc,
    }


def harvest_candidate_patch(
    worker_result: Mapping[str, Any] | None,
    *,
    patch_text: str = "",
    patch_files: Sequence[str] = (),
    baseline_floor: float = 1.0,
) -> dict[str, Any]:
    """Measurement-anchored patch harvesting (GEAK `kernel_lane.js`).

    Never discard a candidate patch on disk just because the engineer subagent timed out
    before returning its JSON summary; only skip verification when the worker cleanly
    reported a valid measurement below `baseline_floor`.
    """
    row = _mapping(worker_result)
    disk_patch_present = bool((patch_text and patch_text.strip()) or list(patch_files or ()))
    measurement_valid = _flag(row.get("measurement_valid", row.get("measurement_passed")), default=False)
    speedup = _number(row.get("speedup_ratio", row.get("speedup")))
    timed_out = _flag(row.get("timed_out", row.get("worker_timed_out")), default=False)
    if measurement_valid and speedup is not None and speedup <= float(baseline_floor):
        return {
            "should_verify": False,
            "harvest_mode": "skip_verified_below_floor",
            "reason": "valid_measurement_below_floor",
            "disk_patch_present": disk_patch_present,
            "worker_timed_out": timed_out,
        }
    if disk_patch_present:
        mode = "disk_patch_rescued" if (timed_out or not row or not measurement_valid) else "worker_and_disk_ready"
        return {
            "should_verify": True,
            "harvest_mode": mode,
            "reason": "candidate_patch_available_for_independent_verification",
            "disk_patch_present": True,
            "worker_timed_out": timed_out,
        }
    return {
        "should_verify": bool(measurement_valid and speedup is not None and speedup > float(baseline_floor)),
        "harvest_mode": "worker_return_only" if row else "empty",
        "reason": "no_disk_patch_found" if not row else "worker_reported_candidate",
        "disk_patch_present": False,
        "worker_timed_out": timed_out,
    }


__all__ = [
    "build_isolated_jit_cache_env",
    "build_subagent_fact_pack",
    "classify_candidate_patch_shape",
    "evaluate_speculative_acceptance",
    "harvest_candidate_patch",
]
