"""Hardware-Anchored Performance (HAP), operator complexity scoring, and Sentry trajectory auditing.

[Role]
Distill `cann-bench` Roofline-anchored operator evaluation (HAP, Eq.3/Eq.4
composite scoring, L1-L5 operator complexity taxonomy) and `cannbot-sentry`
(`sift` + `insight`) dual-verification trajectory auditing (process compliance,
result consistency, and 6-lane engineering swimlane attribution) into `mindie-bench`.

[Invariants]
- HAP (`compute_hap_score`) is a saturating Roofline-anchored metric, not a raw
  speedup ratio: `T_cand == T_baseline -> 0.5`, `T_cand == T_HW -> 1.0`,
  `T_cand > T_baseline -> (0, 0.5)`, `T_cand < T_baseline -> > 0.5`.
- Invalid/non-positive anchors (`T_cand <= 0`, `T_HW <= 0`, `T_baseline <= 0`,
  `NaN`, `inf`) or non-positive denominator `(T_cand - T_HW) + (T_baseline - T_HW) <= 0`
  or non-positive numerator `T_baseline - T_HW <= 0` always return `None`, never
  `+-inf` or negative scores.
- Operator composite score (`EachOperatorScore`) zeroes out the entire operator
  when `compile_passed` is False (`delta_pass == 0`) or when anti-cheat flags
  (`cpu_fallback_detected` / `no_npu_kernel_detected`) trigger.
- Sentry dual-verification requires both `process_compliance` (ordered workflow
  stages with `step_index`/`event_id` back-links) and `result_consistency`
  (reported claims reconciled against `verification_receipt`).

[Timeouts & Locks]
- Pure in-memory evaluation and deterministic trajectory auditing; zero
  subprocesses, zero network I/O, zero locks, and zero sleeps.

[Side-Effect Policy]
- Pure functional transformations returning new dictionaries/dataclasses; never
  mutates caller inputs or touches the filesystem.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SCHEMA_CANN_SENTRY = "mindie-bench-cann-sentry/1"
CANONICAL_SWIMLANES: tuple[str, ...] = ("explore", "design", "code", "compile", "verify", "distill")

OPERATOR_COMPLEXITY_TAXONOMY: dict[str, dict[str, Any]] = {
    "L1": {"level": "L1", "rank": 1, "category": "Element-wise / Activation", "operators": ("Exp", "SwiGLU", "RMSNorm", "Gelu", "Sigmoid", "Silu", "Add", "Sqrt")},
    "L2": {"level": "L2", "rank": 2, "category": "Reduction / Normalization / Gather-Scatter", "operators": ("Softmax", "LayerNorm", "Gather", "Scatter", "ImageResizeNormalize", "GridSampler3D")},
    "L3": {"level": "L3", "rank": 3, "category": "High-dimensional Tiling / Conv / MoE / Matmul", "operators": ("TopK", "Matmul", "GroupedMatmul", "Conv2D", "NMS", "MoeReRouting")},
    "L4": {"level": "L4", "rank": 4, "category": "Complex Fused / Attention / Sequence", "operators": ("FlashAttention", "GatedDeltaNet", "SelectiveScan", "LinearCrossEntropy", "AllGatherMatmul", "AttentionResidual", "LSTM", "MLA", "MLAProlog")},
    "L5": {"level": "L5", "rank": 5, "category": "Stateful / Sparse Attention / Dynamic Control Flow", "operators": ("KimiDeltaAttention", "LightningIndexer", "SparseFlashAttention", "TensorProgramVm", "LactTttChunk")},
}

_PASS_WORDS = frozenset({"1", "true", "yes", "ok", "pass", "passed", "success", "succeeded", "completed", "supported", "verified", "attained"})
_FAIL_WORDS = frozenset({"0", "false", "no", "fail", "failed", "error", "timeout", "refuted", "regressed"})


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())


_OPERATOR_TO_LEVEL: dict[str, str] = {
    _norm_token(op): level for level, info in OPERATOR_COMPLEXITY_TAXONOMY.items() for op in info["operators"]
}


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) if math.isfinite(value) else default
    text = str(value).strip().lower()
    if text in _PASS_WORDS:
        return True
    if text in _FAIL_WORDS:
        return False
    return default


def _level_info(level_key: str, matched_op: str | None) -> dict[str, Any]:
    spec = OPERATOR_COMPLEXITY_TAXONOMY[level_key]
    return {
        "complexity_level": spec["level"], "rank": spec["rank"],
        "category": spec["category"], "canonical_operators": list(spec["operators"]),
        "matched_operator": matched_op or None,
    }


def classify_operator_complexity(operator_or_level: str | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify an operator name, path, or level tag into the L1..L5 taxonomy."""
    if isinstance(operator_or_level, Mapping):
        raw_level = str(operator_or_level.get("complexity_level") or operator_or_level.get("level") or operator_or_level.get("rel_path") or "").strip()
        raw_op = str(operator_or_level.get("operator") or operator_or_level.get("operator_name") or operator_or_level.get("name") or "").strip()
    else:
        raw_level = raw_op = str(operator_or_level or "").strip()

    for candidate in (raw_level, raw_op):
        if not candidate:
            continue
        first_seg = candidate.split("/", 1)[0].strip().upper()
        if first_seg in OPERATOR_COMPLEXITY_TAXONOMY:
            return _level_info(first_seg, raw_op)
        m = re.match(r"^(?:LEVEL[_\-\s]*)?([1-5])$", first_seg)
        if m:
            return _level_info(f"L{m.group(1)}", raw_op)

    for candidate in (raw_op, raw_level):
        if not candidate:
            continue
        leaf = candidate.rsplit("/", 1)[-1].strip()
        norm = _norm_token(leaf)
        if norm in _OPERATOR_TO_LEVEL:
            return _level_info(_OPERATOR_TO_LEVEL[norm], leaf)
        for lvl, kws in (
            ("L5", ("kimidelta", "lightningindexer", "sparseflash", "tensorprogramvm", "lactttt")),
            ("L4", ("flashattention", "gateddeltanet", "selectivescan", "linearcrossentropy", "attention", "mla", "lstm")),
            ("L3", ("topk", "groupedmatmul", "matmul", "conv", "moe", "nms", "gemm")),
            ("L2", ("softmax", "layernorm", "gather", "scatter", "reduce")),
            ("L1", ("exp", "swiglu", "rmsnorm", "gelu", "silu", "relu", "add", "sqrt")),
        ):
            if any(k in norm for k in kws):
                return _level_info(lvl, leaf)
    return _level_info("L1", raw_op)


def compute_hap_score(
    t_cand_us: float | int | None, t_baseline_us: float | int | None, t_hw_us: float | int | None,
) -> float | None:
    """Compute `cann-bench` Hardware-Anchored Performance (HAP) score `(T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))`."""
    cand, base, hw = _finite_float(t_cand_us), _finite_float(t_baseline_us), _finite_float(t_hw_us)
    if cand is None or base is None or hw is None or cand <= 0.0 or base <= 0.0 or hw <= 0.0:
        return None
    numerator, denominator = base - hw, (cand - hw) + (base - hw)
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    score = numerator / denominator
    return score if math.isfinite(score) and score > 0.0 else None


def _extract_case_eval(case: Any) -> tuple[bool, float | None, bool, str | None]:
    """Extract (accuracy_passed, hap_score, anti_cheat_triggered, level_hint) from one case."""
    if isinstance(case, (tuple, list)) and len(case) >= 2:
        acc_ok = _bool_flag(case[0], default=False)
        raw_hap = _finite_float(case[1]) if acc_ok else None
        return acc_ok, (raw_hap if raw_hap is not None and raw_hap > 0.0 else None), False, None
    if not isinstance(case, Mapping):
        return False, None, False, None

    anti_cheat = bool(
        case.get("cpu_fallback_detected") or case.get("no_npu_kernel_detected")
        or case.get("zeroed_by_cpu_fallback") or case.get("zeroed_by_no_npu_perf")
    )
    level_hint = str(
        case.get("complexity_level") or case.get("level") or case.get("operator") or case.get("rel_path") or ""
    ).strip() or None

    acc_ok: bool | None = None
    for key in ("accuracy_passed", "passed", "success", "oracle_passed", "correctness_ok", "accuracy_ok", "delta_accuracy"):
        if key in case and case[key] is not None:
            acc_ok = _bool_flag(case[key])
            break
    if acc_ok is None:
        acc_ok = _bool_flag(case["status"], default=False) if "status" in case and case["status"] is not None else True
    if not acc_ok:
        return False, None, anti_cheat, level_hint

    for hap_key in ("hap_score", "hap", "sol_score", "score"):
        if hap_key in case:
            val = _finite_float(case[hap_key])
            if val is not None and val > 0.0:
                return True, val, anti_cheat, level_hint
            break

    cand = _finite_float(case.get("t_cand_us", case.get("cand_latency_us", case.get("latency_us", case.get("elapsed_us", case.get("opt_latency_us"))))))
    base = _finite_float(case.get("t_baseline_us", case.get("baseline_perf_us", case.get("baseline_latency_us", case.get("ref_latency_us")))))
    hw = _finite_float(case.get("t_hw_us", case.get("roofline_floor_us", case.get("hw_latency_us"))))
    return True, compute_hap_score(cand, base, hw), anti_cheat, level_hint


def compute_operator_composite_score(
    compile_passed: bool,
    case_results: Sequence[Any] | None,
    *,
    w_compile: float = 0.2,
    w_accuracy: float = 0.3,
    w_perf: float = 0.5,
    operator: str | None = None,
    complexity_level: str | None = None,
    total_cases: int | None = None,
) -> dict[str, Any]:
    """Compute `cann-bench` composite operator score (`EachOperatorScore`, Eq. 4)."""
    wc, wf, wp = _finite_float(w_compile), _finite_float(w_accuracy), _finite_float(w_perf)
    if wc is None or wf is None or wp is None or wc < 0.0 or wf < 0.0 or wp < 0.0:
        raise ValueError("w_compile, w_accuracy, and w_perf must be finite non-negative numbers")

    cases = list(case_results or ())
    n_cases = max(int(total_cases) if total_cases is not None else len(cases), len(cases))
    first_hint = next(
        (
            item.get("complexity_level") or item.get("level") or item.get("operator") or item.get("rel_path")
            for item in cases if isinstance(item, Mapping)
            and (item.get("complexity_level") or item.get("level") or item.get("operator") or item.get("rel_path"))
        ),
        None,
    )
    complexity_info = classify_operator_complexity(complexity_level or operator or first_hint)
    resolved_level = complexity_info["complexity_level"]

    per_case_hap: list[float | None] = []
    valid_haps: list[float] = []
    passed_cases = 0
    missing_hap_cases = 0
    anti_cheat = False
    level_buckets: dict[str, list[tuple[bool, float | None]]] = {lvl: [] for lvl in OPERATOR_COMPLEXITY_TAXONOMY}

    for item in cases:
        acc_ok, hap_i, item_ac, item_hint = _extract_case_eval(item)
        anti_cheat = anti_cheat or item_ac
        item_level = classify_operator_complexity(item_hint or resolved_level)["complexity_level"]
        if not compile_passed or not acc_ok:
            per_case_hap.append(None)
            level_buckets[item_level].append((False, None))
            continue
        passed_cases += 1
        per_case_hap.append(hap_i)
        level_buckets[item_level].append((True, hap_i))
        if hap_i is None:
            missing_hap_cases += 1
        else:
            valid_haps.append(hap_i)

    if len(per_case_hap) < n_cases:
        per_case_hap.extend([None] * (n_cases - len(per_case_hap)))

    if n_cases <= 0 or not compile_passed or anti_cheat:
        comp_score = acc_score = perf_score = total_score = 0.0
        mean_hap = None
        pass_rate = 0.0 if (n_cases <= 0 or not compile_passed) else (passed_cases / n_cases)
    else:
        comp_score = wc * 100.0
        acc_score = (passed_cases / n_cases) * wf * 100.0
        hap_sum = sum(valid_haps)
        perf_score = (hap_sum / n_cases) * wp * 100.0
        total_score = comp_score + acc_score + perf_score
        mean_hap = (hap_sum / len(valid_haps)) if valid_haps else None
        pass_rate = passed_cases / n_cases

    complexity_summary: dict[str, dict[str, Any]] = {}
    for lvl, spec in OPERATOR_COMPLEXITY_TAXONOMY.items():
        bucket = level_buckets[lvl]
        b_total = len(bucket)
        b_passed = sum(1 for ok, _ in bucket if ok)
        b_haps = [h for ok, h in bucket if ok and h is not None]
        b_score = ((wc + (b_passed / b_total) * wf + (sum(b_haps) / b_total) * wp) * 100.0
                   if (b_total > 0 and compile_passed and not anti_cheat) else 0.0)
        complexity_summary[lvl] = {
            "level": lvl, "rank": spec["rank"], "category": spec["category"],
            "canonical_operators": list(spec["operators"]), "case_count": b_total,
            "passed_cases": b_passed, "accuracy_pass_rate": (b_passed / b_total) if b_total > 0 else 0.0,
            "mean_hap": (sum(b_haps) / len(b_haps)) if b_haps else None,
            "each_operator_score": b_score,
        }

    return {
        "schema_version": SCHEMA_CANN_SENTRY, "operator": operator or complexity_info["matched_operator"] or "",
        "compile_passed": bool(compile_passed), "each_operator_score": total_score,
        "total_score": total_score, "score": total_score, "compilation_score": comp_score,
        "accuracy_score": acc_score, "function_score": acc_score, "performance_score": perf_score,
        "mean_hap": mean_hap, "hap_scores": per_case_hap, "per_case_scores": per_case_hap,
        "accuracy_pass_rate": pass_rate, "pass_rate": pass_rate,
        "passed_cases": passed_cases if compile_passed else 0, "total_cases": max(0, n_cases),
        "missing_hap_cases": missing_hap_cases, "zeroed_by_anti_cheat": anti_cheat,
        "complexity_level": resolved_level, "complexity_category": complexity_info["category"],
        "complexity_summary": complexity_summary, "weights": {"w_compile": wc, "w_accuracy": wf, "w_perf": wp},
    }


@dataclass
class RSICandidateResult(Mapping[str, Any]):
    """Structured RSI candidate evaluation result carrying both speedup and HAP score."""

    run_id: str
    candidate_id: str
    verdict: str = "inconclusive"
    outcome: str = "measurement_invalid"
    gate_reached: int = 1
    speedup_ratio: float | None = None
    hap_score: float | None = None
    t_cand_us: float | None = None
    t_baseline_us: float | None = None
    t_hw_us: float | None = None
    roofline_floor_us: float | None = None
    min_speedup_ratio: float = 1.05
    compile_ok: bool = False
    oracle_passed: bool = False
    measurement_passed: bool = False
    ablation_retained: bool | None = None
    distill_ready: bool = False
    mutation_summary: str = ""
    failure_reason: str = ""
    return_tier: str = "L4"
    conditions: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    axes: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hw = _finite_float(self.t_hw_us if self.t_hw_us is not None else self.roofline_floor_us)
        if hw is not None:
            self.t_hw_us = self.roofline_floor_us = hw
        base, cand, speedup = _finite_float(self.t_baseline_us), _finite_float(self.t_cand_us), _finite_float(self.speedup_ratio)
        if speedup is None and base is not None and cand is not None and base > 0.0 and cand > 0.0:
            self.speedup_ratio = speedup = base / cand
        elif speedup is not None and speedup > 0.0:
            if cand is None and base is not None and base > 0.0:
                self.t_cand_us = cand = base / speedup
            elif base is None and cand is not None and cand > 0.0:
                self.t_baseline_us = base = cand * speedup
        if self.hap_score is None and hw is not None and cand is not None and base is not None:
            if self.compile_ok and self.oracle_passed and self.measurement_passed:
                self.hap_score = compute_hap_score(cand, base, hw)

    def to_dict(self) -> dict[str, Any]:
        keys = ("run_id", "candidate_id", "mutation_summary", "verdict", "outcome", "gate_reached", "speedup_ratio", "hap_score", "t_cand_us", "t_baseline_us", "t_hw_us", "roofline_floor_us", "min_speedup_ratio", "compile_ok", "oracle_passed", "measurement_passed", "ablation_retained", "distill_ready", "failure_reason", "return_tier")
        base_dict = {k: getattr(self, k) for k in keys}
        base_dict.update(conditions=dict(self.conditions), gates=dict(self.gates), axes=dict(self.axes))
        base_dict.update(self.extra)
        return base_dict

    @classmethod
    def from_gate_entry(cls, entry: Mapping[str, Any]) -> RSICandidateResult:
        gates = entry.get("gates") if isinstance(entry.get("gates"), Mapping) else {}
        g1, g2, g3 = gates.get("gate1_compile_oracle", {}).get("passed"), gates.get("gate2_measurement_guard", {}).get("passed"), gates.get("gate3_ablation_gate", {}).get("passed")
        known = set(cls.__dataclass_fields__) - {"extra"}
        return cls(
            run_id=str(entry.get("run_id") or ""), candidate_id=str(entry.get("candidate_id") or ""),
            verdict=str(entry.get("verdict") or "inconclusive"), outcome=str(entry.get("outcome") or ""), gate_reached=int(entry.get("gate_reached") or 1),
            speedup_ratio=_finite_float(entry.get("speedup_ratio")), hap_score=_finite_float(entry.get("hap_score")),
            t_cand_us=_finite_float(entry.get("t_cand_us")), t_baseline_us=_finite_float(entry.get("t_baseline_us")),
            t_hw_us=_finite_float(entry.get("t_hw_us")), roofline_floor_us=_finite_float(entry.get("roofline_floor_us")),
            min_speedup_ratio=float(_finite_float(entry.get("min_speedup_ratio")) or 1.05),
            compile_ok=bool(entry.get("compile_ok", g1 is True)), oracle_passed=bool(entry.get("oracle_passed", g1 is True)),
            measurement_passed=bool(entry.get("measurement_passed", g2 is True)), ablation_retained=entry.get("ablation_retained", g3),
            distill_ready=bool(entry.get("distill_ready", False)), mutation_summary=str(entry.get("mutation_summary") or ""),
            failure_reason=str(entry.get("failure_reason") or ""), return_tier=str(entry.get("return_tier") or "L4"),
            conditions=dict(entry.get("conditions") or {}), gates=dict(gates), axes=dict(entry.get("axes") or {}),
            extra={k: v for k, v in entry.items() if k not in known},
        )
    def __getitem__(self, key: str) -> Any: return self.to_dict()[key]
    def __iter__(self) -> Iterator[str]: return iter(self.to_dict())
    def __len__(self) -> int: return len(self.to_dict())


_ACTION_TOKEN_MAP: dict[str, set[str]] = {
    "knowledge_query": {"knowledge_query", "knowledge", "query_knowledge", "retrieve_knowledge", "knowledge_broker", "fact_pack", "l1_foundation", "l2_bottleneck", "l4_mutators"},
    "code_mutation": {"code_mutation", "mutate", "edit", "edit_file", "write", "write_file", "patch", "apply_patch", "multiedit", "source_edit", "implement"},
    "compile_check": {"compile", "compile_check", "build", "cmake", "bisheng", "ascendc_compile", "jit_compile", "phase1_compile", "g0", "g1"},
    "npu_timing": {"npu_timing", "timing", "phase2_timing", "benchmark_timing", "perf_timing", "measure_latency", "latency_bench", "g3", "g3_timing"},
    "oracle_check": {"oracle", "oracle_check", "accuracy", "accuracy_check", "correctness", "golden", "allclose", "g2"},
    "generalization_check": {"generalization", "generalization_check", "hidden_input", "hidden_input_generalization", "post_freeze", "holdout"},
    "ablation_check": {"ablation", "ablation_check", "leave_one_out", "loo", "sequential_ablation", "factor_trial", "c4_ablation"},
    "distill_action": {"distill", "ledger", "progress", "append_ledger", "append_ledger_and_render_progress", "export_knowledge", "commit_knowledge"},
}


def _classify_event_lane_and_actions(raw: Mapping[str, Any]) -> tuple[str, set[str]]:
    explicit_lane = str(raw.get("lane") or raw.get("swimlane") or raw.get("phase") or "").strip().lower()
    stage_str = str(raw.get("stage") or "").strip().lower()
    tool_str = str(raw.get("tool_name") or raw.get("tool") or raw.get("action") or raw.get("type") or raw.get("name") or "").strip().lower()
    cmd_str = str(raw.get("command") or raw.get("command_line") or (raw.get("input", {}) if isinstance(raw.get("input"), Mapping) else {}).get("command", "") or "").strip().lower()
    combined = " ".join(part for part in (explicit_lane, stage_str, tool_str, cmd_str) if part)
    tokens = set(re.findall(r"[a-z0-9_]+", combined))

    actions = {act for act, kw in _ACTION_TOKEN_MAP.items() if tokens & kw}
    if "knowledge" in combined:
        actions.add("knowledge_query")
    if explicit_lane == "code" or stage_str in {"code", "code_mutation", "mutate", "edit"}:
        actions.add("code_mutation")
    if explicit_lane == "compile" or stage_str in {"compile", "compile_check", "build"}:
        actions.add("compile_check")
    if "npu_timing" in combined or stage_str in {"timing", "npu_timing"}:
        actions.add("npu_timing")
    if "hidden_input" in combined or "generalization" in combined:
        actions.add("generalization_check")
    if "ablation" in combined or "leave_one_out" in combined:
        actions.add("ablation_check")
    if explicit_lane == "distill" or stage_str in {"distill", "ledger", "report"}:
        actions.add("distill_action")

    lane = (
        explicit_lane if explicit_lane in CANONICAL_SWIMLANES else
        stage_str if stage_str in CANONICAL_SWIMLANES else
        "distill" if "distill_action" in actions else
        "verify" if (actions & {"npu_timing", "oracle_check", "generalization_check", "ablation_check"} or tokens & {"verify", "test", "pytest", "verifier", "benchmark", "eval", "gate"}) else
        "compile" if "compile_check" in actions else
        "code" if "code_mutation" in actions else
        "design" if (tokens & {"design", "plan", "hypothesis", "architect", "schedule", "tiling_plan", "debiased_context"}) else
        "explore"
    )
    return lane, actions


def _event_succeeded(raw: Mapping[str, Any]) -> bool:
    if raw.get("is_error") is True:
        return False
    for key in ("succeeded", "ok", "passed", "compile_ok", "compile_passed", "success", "status"):
        if key in raw and raw[key] is not None:
            return _bool_flag(raw[key], default=True)
    return True


def _build_swimlane_attribution(normalized_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lanes: dict[str, dict[str, Any]] = {
        lane: {"lane": lane, "turns": 0, "event_count": 0, "tool_calls": 0, "tokens": 0, "input_tokens": 0, "output_tokens": 0, "duration_ms": 0.0, "duration_seconds": 0.0, "token_share": 0.0, "duration_share": 0.0, "step_indices": [], "event_ids": []}
        for lane in CANONICAL_SWIMLANES
    }
    total_turns = total_tool_calls = total_tokens = 0
    total_duration_ms = 0.0
    for evt in normalized_events:
        b = lanes[evt["lane"]]
        for k in ("turns", "tool_calls", "input_tokens", "output_tokens", "tokens"):
            b[k] += int(evt[k])
        b["event_count"] += 1
        b["duration_ms"] += float(evt["duration_ms"])
        b["duration_seconds"] = b["duration_ms"] / 1000.0
        b["step_indices"].append(evt["step_index"])
        b["event_ids"].append(evt["event_id"])
        total_turns += int(evt["turns"])
        total_tool_calls += int(evt["tool_calls"])
        total_tokens += int(evt["tokens"])
        total_duration_ms += float(evt["duration_ms"])
    for b in lanes.values():
        b["token_share"] = (b["tokens"] / total_tokens) if total_tokens > 0 else 0.0
        b["duration_share"] = (b["duration_ms"] / total_duration_ms) if total_duration_ms > 0.0 else 0.0
    return {
        "lane_order": list(CANONICAL_SWIMLANES), "lanes": lanes,
        "lane_tokens": {lane: lanes[lane]["tokens"] for lane in CANONICAL_SWIMLANES},
        "total_turns": total_turns, "total_events": len(normalized_events),
        "total_tool_calls": total_tool_calls, "total_tokens": total_tokens,
        "total_duration_ms": total_duration_ms, "total_duration_seconds": total_duration_ms / 1000.0,
        "active_lanes": [lane for lane in CANONICAL_SWIMLANES if lanes[lane]["event_count"] > 0],
    }


def compare_swimlane_attributions(with_knowledge: Mapping[str, Any], without_knowledge: Mapping[str, Any]) -> dict[str, Any]:
    """Compute per-lane and total A/B deltas (`with_knowledge` vs `without_knowledge`)."""
    wk = with_knowledge.get("swimlane_attribution", with_knowledge) if isinstance(with_knowledge, Mapping) else {}
    wok = without_knowledge.get("swimlane_attribution", without_knowledge) if isinstance(without_knowledge, Mapping) else {}
    wk_lanes = wk.get("lanes", {}) if isinstance(wk.get("lanes"), Mapping) else {}
    wok_lanes = wok.get("lanes", {}) if isinstance(wok.get("lanes"), Mapping) else {}
    lane_deltas = {
        lane: {
            "tokens_delta": int(wk_lanes.get(lane, {}).get("tokens") or 0) - int(wok_lanes.get(lane, {}).get("tokens") or 0),
            "duration_ms_delta": float(wk_lanes.get(lane, {}).get("duration_ms") or 0.0) - float(wok_lanes.get(lane, {}).get("duration_ms") or 0.0),
            "tool_calls_delta": int(wk_lanes.get(lane, {}).get("tool_calls") or 0) - int(wok_lanes.get(lane, {}).get("tool_calls") or 0),
            "with_knowledge_tokens": int(wk_lanes.get(lane, {}).get("tokens") or 0),
            "without_knowledge_tokens": int(wok_lanes.get(lane, {}).get("tokens") or 0),
        }
        for lane in CANONICAL_SWIMLANES
    }
    tok_a, tok_b = int(wk.get("total_tokens") or 0), int(wok.get("total_tokens") or 0)
    return {
        "total_tokens_delta": tok_a - tok_b,
        "total_duration_ms_delta": float(wk.get("total_duration_ms") or 0.0) - float(wok.get("total_duration_ms") or 0.0),
        "token_reduction_ratio": ((tok_b - tok_a) / tok_b) if tok_b > 0 else 0.0, "lane_deltas": lane_deltas,
    }


def _make_check(check_id: str, category: str, verdict: str, evidence_events: Sequence[Mapping[str, Any]], evidence: str, *, reason: str = "") -> dict[str, Any]:
    steps, eids = [int(e["step_index"]) for e in evidence_events], [str(e["event_id"]) for e in evidence_events]
    return {
        "check_id": check_id, "category": category, "verdict": verdict,
        "passed": True if verdict == "PASS" else (False if verdict == "FAIL" else None),
        "method": "PROGRAMMATIC", "step_indices": steps, "turn_refs": steps, "event_ids": eids,
        "evidence_backlinks": [{"step_index": s, "event_id": i} for s, i in zip(steps, eids)],
        "reason": reason or ("ok" if verdict == "PASS" else check_id), "evidence": evidence,
    }


def _normalize_events(events: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw_item in enumerate(events or ()):
        if not isinstance(raw_item, Mapping):
            continue
        step_index = int(raw_item.get("step_index", raw_item.get("turn_index", raw_item.get("index", raw_item.get("seq", idx)))))
        event_id = str(raw_item.get("event_id") or raw_item.get("id") or raw_item.get("call_id") or raw_item.get("capture_id") or f"evt-{step_index}")
        lane, actions = _classify_event_lane_and_actions(raw_item)
        in_tok = int(_finite_float(raw_item.get("input_tokens")) or 0)
        out_tok = int(_finite_float(raw_item.get("output_tokens")) or 0)
        tok = int(_finite_float(raw_item.get("tokens", raw_item.get("total_tokens"))) or (in_tok + out_tok))
        if any(k in raw_item for k in ("duration_ms", "elapsed_ms", "latency_ms")):
            dur_ms = float(_finite_float(raw_item.get("duration_ms", raw_item.get("elapsed_ms", raw_item.get("latency_ms")))) or 0.0)
        elif any(k in raw_item for k in ("duration_seconds", "duration_s", "seconds", "elapsed_s")):
            dur_ms = float(_finite_float(raw_item.get("duration_seconds", raw_item.get("duration_s", raw_item.get("seconds", raw_item.get("elapsed_s"))))) or 0.0) * 1000.0
        else:
            dur_ms = 0.0
        raw_tcalls = raw_item.get("tool_calls")
        tool_calls = max(0, raw_tcalls) if isinstance(raw_tcalls, int) and not isinstance(raw_tcalls, bool) else (len(raw_tcalls) if isinstance(raw_tcalls, Sequence) and not isinstance(raw_tcalls, (str, bytes)) else 1)
        normalized.append({
            "raw": dict(raw_item), "step_index": step_index, "event_id": event_id,
            "lane": lane, "actions": actions, "succeeded": _event_succeeded(raw_item),
            "tokens": max(0, tok), "input_tokens": max(0, in_tok), "output_tokens": max(0, out_tok),
            "duration_ms": max(0.0, dur_ms), "tool_calls": tool_calls,
            "turns": max(1, int(_finite_float(raw_item.get("turns")) or 1)),
        })
    return normalized


def audit_sentry_trajectory(
    events: Sequence[Mapping[str, Any]] | None,
    declared_contract: Mapping[str, Any] | None = None,
    verification_receipt: Mapping[str, Any] | None = None,
    *,
    reported_claims: Mapping[str, Any] | None = None,
    speedup_rtol: float = 0.05,
) -> dict[str, Any]:
    """Execute `cannbot-sentry` dual-verification audit + 6-lane swimlane attribution."""
    contract, receipt = dict(declared_contract or {}), dict(verification_receipt or {})
    normalized = _normalize_events(events)
    swimlane_attribution = _build_swimlane_attribution(normalized)
    baseline_traj = contract.get("without_knowledge_events") or contract.get("baseline_swimlane")
    if isinstance(baseline_traj, Mapping):
        swimlane_attribution["ab_comparison"] = compare_swimlane_attributions(swimlane_attribution, baseline_traj)
    elif isinstance(baseline_traj, Sequence) and not isinstance(baseline_traj, (str, bytes)):
        swimlane_attribution["ab_comparison"] = compare_swimlane_attributions(
            swimlane_attribution, _build_swimlane_attribution(_normalize_events(baseline_traj))
        )

    req_raw = contract.get("required_stages") or contract.get("required_order") or contract.get("workflow_stages") or ()
    req_stages = [str(s).strip().lower() for s in (req_raw if isinstance(req_raw, Sequence) and not isinstance(req_raw, str) else [req_raw])]
    req_knowledge = _bool_flag(contract.get("require_knowledge_before_mutation", contract.get("require_pre_mutation_retrieval", contract.get("knowledge_before_mutation", True))), default=True) or any("knowledge" in s or "retrieve" in s for s in req_stages)
    req_compile_before_timing = _bool_flag(contract.get("require_compile_before_timing", contract.get("compile_before_timing", True)), default=True) or any("compile" in s for s in req_stages)
    req_generalization = _bool_flag(contract.get("require_generalization_gate", contract.get("require_hidden_input_generalization", contract.get("hidden_input_generalization", False))), default=False) or any("generalization" in s or "hidden_input" in s for s in req_stages)
    factor_count = int(_finite_float(contract.get("factor_count", receipt.get("factor_count"))) or 1)
    req_ablation = _bool_flag(contract.get("require_ablation_gate", contract.get("require_ablation", contract.get("multi_factor", False))), default=False) or factor_count > 1 or any("ablation" in s or "leave_one_out" in s for s in req_stages)

    knowledge_evts = [e for e in normalized if "knowledge_query" in e["actions"]]
    mutation_evts = [e for e in normalized if "code_mutation" in e["actions"]]
    compile_evts = [e for e in normalized if "compile_check" in e["actions"]]
    ok_compile_evts = [e for e in compile_evts if e["succeeded"]]
    timing_evts = [e for e in normalized if "npu_timing" in e["actions"]]
    gen_evts = [e for e in normalized if "generalization_check" in e["actions"]]
    ablation_evts = [e for e in normalized if "ablation_check" in e["actions"]]

    checks: dict[str, dict[str, Any]] = {}
    if not req_knowledge:
        checks["knowledge_before_mutation"] = _make_check("knowledge_before_mutation", "step", "N/A", [], "not required by contract")
    elif mutation_evts:
        first_mut = mutation_evts[0]
        prior_k = [e for e in knowledge_evts if e["step_index"] < first_mut["step_index"]]
        checks["knowledge_before_mutation"] = (
            _make_check("knowledge_before_mutation", "step", "PASS", [prior_k[0], first_mut], f"knowledge_query at step {prior_k[0]['step_index']} preceded code_mutation at step {first_mut['step_index']}")
            if prior_k else
            _make_check("knowledge_before_mutation", "prohibition", "FAIL", [first_mut, *knowledge_evts], f"code_mutation at step {first_mut['step_index']} occurred without prior knowledge_query", reason="knowledge_query_after_code_mutation" if knowledge_evts else "missing_knowledge_query_before_mutation")
        )
    elif knowledge_evts:
        checks["knowledge_before_mutation"] = _make_check("knowledge_before_mutation", "step", "PASS", knowledge_evts, "knowledge_query executed")
    elif any("knowledge" in s for s in req_stages):
        checks["knowledge_before_mutation"] = _make_check("knowledge_before_mutation", "prohibition", "FAIL", [], "required knowledge_query missing", reason="missing_knowledge_query")
    else:
        checks["knowledge_before_mutation"] = _make_check("knowledge_before_mutation", "step", "N/A", [], "no code mutation occurred")

    if not (req_compile_before_timing and (timing_evts or any("timing" in s for s in req_stages))):
        checks["compile_before_timing"] = _make_check("compile_before_timing", "step", "N/A", [], "no npu_timing stage invoked")
    elif not timing_evts:
        checks["compile_before_timing"] = _make_check("compile_before_timing", "step", "FAIL", compile_evts, "required npu_timing missing", reason="missing_npu_timing")
    else:
        timing_violations, reason = [], ""
        for t_evt in timing_evts:
            prior_c = [c for c in ok_compile_evts if c["step_index"] < t_evt["step_index"]]
            prior_m = [m for m in mutation_evts if m["step_index"] < t_evt["step_index"]]
            if not prior_c:
                timing_violations.append(t_evt)
                reason = "npu_timing_before_compile_check"
            elif prior_m and prior_m[-1]["step_index"] > prior_c[-1]["step_index"]:
                timing_violations.extend([prior_m[-1], t_evt])
                reason = "uncompiled_code_mutation_before_npu_timing"
        checks["compile_before_timing"] = (
            _make_check("compile_before_timing", "step", "PASS", [ok_compile_evts[-1], timing_evts[0]], f"compile_check at step {ok_compile_evts[-1]['step_index']} preceded npu_timing at step {timing_evts[0]['step_index']}")
            if not timing_violations else
            _make_check("compile_before_timing", "prohibition", "FAIL", [*compile_evts, *timing_violations], f"npu_timing violated compile-before-timing ordering ({reason})", reason=reason)
        )

    for cid, req_flag, evts, ok_msg, miss_reason, fail_reason in (
        ("hidden_input_generalization", req_generalization, gen_evts, "hidden_input_generalization verified", "missing_hidden_input_generalization_check", "failed_or_stale_hidden_input_generalization"),
        ("multi_factor_ablation", req_ablation, ablation_evts, "sequential ablation verified", "missing_multi_factor_ablation_check", "failed_or_stale_multi_factor_ablation"),
    ):
        if req_flag:
            ok_evts = [x for x in evts if x["succeeded"]]
            if ok_evts and (not mutation_evts or ok_evts[-1]["step_index"] > mutation_evts[-1]["step_index"]):
                checks[cid] = _make_check(cid, "step", "PASS", ok_evts, ok_msg)
            else:
                r = miss_reason if not evts else fail_reason
                checks[cid] = _make_check(cid, "prohibition", "FAIL", evts or mutation_evts[-1:], f"{cid} check failed ({r})", reason=r)
        else:
            checks[cid] = _make_check(cid, "step", "PASS" if evts else "N/A", evts, f"{cid} observed" if evts else "not required")

    if mutation_evts and _bool_flag(contract.get("require_post_mutation_verification", True), default=True):
        last_mut = mutation_evts[-1]
        post_verify = [
            e for e in normalized
            if e["step_index"] > last_mut["step_index"]
            and (e["lane"] in {"compile", "verify"} or bool(e["actions"] & {"compile_check", "npu_timing", "oracle_check", "generalization_check", "ablation_check"}))
        ]
        checks["post_mutation_verification"] = (
            _make_check("post_mutation_verification", "step", "PASS", [last_mut, post_verify[-1]], "code_mutation verified")
            if post_verify else
            _make_check("post_mutation_verification", "prohibition", "FAIL", [last_mut], "trailing code_mutation had no subsequent verification", reason="unverified_trailing_code_mutation")
        )

    findings = list(checks.values())
    judged = [f for f in findings if f["verdict"] in {"PASS", "FAIL"}]
    violations = [f for f in findings if f["verdict"] == "FAIL"]
    proc_status = "UNKNOWN" if (not normalized and not judged) else ("VIOLATED" if violations else "OK")
    compliance_rate = None if proc_status == "UNKNOWN" else ((sum(1 for f in judged if f["verdict"] == "PASS") / len(judged)) if judged else 1.0)
    proc_compliant = proc_status == "OK"
    process_compliance = {
        "compliant": proc_compliant, "passed": proc_compliant, "status": proc_status,
        "compliance_rate": compliance_rate, "checks": checks, "findings": findings,
        "violations": violations, "violation_reasons": [v["reason"] for v in violations],
        "evidence_backlinks": {cid: chk["evidence_backlinks"] for cid, chk in checks.items()},
    }

    claim_keys = (
        "reported_speedup", "claimed_speedup", "reported_hap_score", "claimed_hap_score",
        "reported_verdict", "claimed_verdict", "reported_outcome", "claimed_outcome",
        "reported_accuracy_passed", "claimed_accuracy_passed", "reported_oracle_passed",
        "reported_compile_passed", "claimed_compile_passed", "reported_compile_ok",
        "reported_measurement_passed", "claimed_measurement_passed",
        "reported_ablation_retained", "claimed_ablation_retained",
    )
    claims: dict[str, Any] = {}
    for src in (*(e["raw"] for e in normalized), contract, receipt):
        for sub in ("reported_claims", "claimed_result", "agent_claim", "agent_report", "claims"):
            if isinstance(src.get(sub), Mapping):
                claims.update(src[sub])
        for k in claim_keys:
            if k in src and src[k] is not None:
                claims[k] = src[k]
    if isinstance(reported_claims, Mapping):
        claims.update(reported_claims)

    claimed_speedup = _finite_float(claims.get("claimed_speedup", claims.get("reported_speedup", claims.get("speedup_ratio", claims.get("speedup")))))
    claimed_hap = _finite_float(claims.get("claimed_hap_score", claims.get("reported_hap_score", claims.get("hap_score"))))
    claimed_verdict = str(claims.get("claimed_verdict", claims.get("reported_verdict", claims.get("verdict", ""))) or "").strip().lower() or None
    claimed_outcome = str(claims.get("claimed_outcome", claims.get("reported_outcome", claims.get("outcome", ""))) or "").strip().lower() or None
    claimed_acc = next((_bool_flag(claims[k]) for k in ("claimed_accuracy_passed", "reported_accuracy_passed", "reported_oracle_passed", "accuracy_passed", "oracle_passed") if k in claims and claims[k] is not None), None)
    claimed_compile = next((_bool_flag(claims[k]) for k in ("claimed_compile_passed", "reported_compile_passed", "reported_compile_ok", "compile_passed", "compile_ok") if k in claims and claims[k] is not None), None)
    claimed_meas = next((_bool_flag(claims[k]) for k in ("claimed_measurement_passed", "reported_measurement_passed", "measurement_passed") if k in claims and claims[k] is not None), None)

    actual_speedup = _finite_float(receipt.get("speedup_ratio", receipt.get("actual_speedup", receipt.get("speedup"))))
    actual_hap = _finite_float(receipt.get("hap_score", receipt.get("actual_hap_score"))) or compute_hap_score(
        _finite_float(receipt.get("t_cand_us", receipt.get("latency_us"))),
        _finite_float(receipt.get("t_baseline_us", receipt.get("baseline_perf_us", receipt.get("baseline_latency_us")))),
        _finite_float(receipt.get("t_hw_us", receipt.get("roofline_floor_us"))),
    )
    actual_verdict = str(receipt.get("verdict") or "").strip().lower() or None
    actual_compile = _bool_flag(receipt.get("compile_ok", receipt.get("compile_passed"))) if any(k in receipt for k in ("compile_ok", "compile_passed")) else None
    actual_oracle = _bool_flag(receipt.get("oracle_passed", receipt.get("accuracy_passed", receipt.get("correctness_ok")))) if any(k in receipt for k in ("oracle_passed", "accuracy_passed", "correctness_ok")) else None
    actual_gen = _bool_flag(receipt.get("generalization_ok", (receipt.get("hidden_input_summary") if isinstance(receipt.get("hidden_input_summary"), Mapping) else {}).get("generalization_ok"))) if ("generalization_ok" in receipt or isinstance(receipt.get("hidden_input_summary"), Mapping)) else None
    actual_meas = _bool_flag(receipt.get("measurement_passed", receipt.get("timing_valid"))) if any(k in receipt for k in ("measurement_passed", "timing_valid")) else None
    actual_ablation = _bool_flag(receipt.get("ablation_retained", (receipt.get("ablation_receipt") if isinstance(receipt.get("ablation_receipt"), Mapping) else {}).get("ablation_retained"))) if ("ablation_retained" in receipt or isinstance(receipt.get("ablation_receipt"), Mapping)) else None
    anti_cheat_receipt = bool(receipt.get("cpu_fallback_detected") or receipt.get("no_npu_kernel_detected") or receipt.get("zeroed_by_cpu_fallback") or receipt.get("zeroed_by_no_npu_perf"))

    discrepancies: list[dict[str, Any]] = []
    tol = max(0.0, float(_finite_float(speedup_rtol) or 0.05))
    if not receipt and (claimed_speedup is not None or claimed_verdict in {"supported", "attained"} or claimed_acc is True):
        discrepancies.append({"code": "missing_verification_receipt", "field": "verification_receipt", "claimed": claims, "actual": None, "evidence": "agent claimed verified speedup/success without verification_receipt"})
    if claimed_speedup is not None and receipt:
        if actual_speedup is None or actual_meas is False or actual_compile is False:
            discrepancies.append({"code": "unverified_speedup_claim", "field": "speedup_ratio", "claimed": claimed_speedup, "actual": actual_speedup, "evidence": f"claimed speedup {claimed_speedup} has no valid measured speedup"})
        elif abs(claimed_speedup - actual_speedup) > max(tol * max(abs(actual_speedup), 1.0), 1e-6):
            discrepancies.append({"code": "speedup_mismatch", "field": "speedup_ratio", "claimed": claimed_speedup, "actual": actual_speedup, "evidence": f"claimed speedup {claimed_speedup:.4f} != verified {actual_speedup:.4f}"})
    if claimed_hap is not None and receipt and (actual_hap is None or abs(claimed_hap - actual_hap) > max(tol * max(abs(actual_hap), 1.0), 1e-6)):
        discrepancies.append({"code": "hap_score_mismatch", "field": "hap_score", "claimed": claimed_hap, "actual": actual_hap, "evidence": f"claimed hap_score {claimed_hap} != verified {actual_hap}"})
    if (claimed_acc is True or claimed_verdict in {"supported", "attained"} or claimed_outcome == "verified") and (actual_oracle is False or actual_gen is False):
        discrepancies.append({"code": "unverified_accuracy_claim", "field": "oracle_passed", "claimed": True, "actual": False, "evidence": "claimed accuracy pass contradicts oracle/generalization failure"})
    if (claimed_compile is True or claimed_verdict in {"supported", "attained"}) and actual_compile is False:
        discrepancies.append({"code": "unverified_compile_claim", "field": "compile_ok", "claimed": True, "actual": False, "evidence": "claimed compile pass contradicts compile_ok=False"})
    if (claimed_meas is True or claimed_verdict in {"supported", "attained"}) and actual_meas is False:
        discrepancies.append({"code": "unverified_measurement_claim", "field": "measurement_passed", "claimed": True, "actual": False, "evidence": "claimed valid measurement contradicts measurement_passed=False"})
    if claimed_verdict is not None and actual_verdict is not None and claimed_verdict != actual_verdict:
        discrepancies.append({"code": "verdict_mismatch", "field": "verdict", "claimed": claimed_verdict, "actual": actual_verdict, "evidence": f"claimed verdict {claimed_verdict!r} != verified {actual_verdict!r}"})
    if claimed_verdict in {"supported", "attained"} and actual_ablation is False:
        discrepancies.append({"code": "unverified_ablation_claim", "field": "ablation_retained", "claimed": True, "actual": False, "evidence": "claimed supported verdict contradicts ablation_retained=False"})
    if anti_cheat_receipt:
        discrepancies.append({"code": "cpu_fallback_or_no_npu_kernel", "field": "anti_cheat", "claimed": "npu_kernel", "actual": "cpu_fallback", "evidence": "CPU fallback or missing NPU kernel detected"})

    event_speedups = [s for e in normalized if any(k in e["raw"] for k in ("speedup_ratio", "measured_speedup")) for s in [_finite_float(e["raw"].get("speedup_ratio", e["raw"].get("measured_speedup")))] if s is not None]
    if event_speedups and actual_speedup is not None and claimed_speedup is None and abs(event_speedups[-1] - actual_speedup) > max(tol * max(abs(actual_speedup), 1.0), 1e-6):
        discrepancies.append({"code": "trajectory_receipt_speedup_mismatch", "field": "speedup_ratio", "claimed": actual_speedup, "actual": event_speedups[-1], "evidence": "trajectory measured speedup diverges from receipt"})
    event_oracles = [_bool_flag(e["raw"].get("oracle_passed", e["raw"].get("accuracy_passed"))) for e in normalized if any(k in e["raw"] for k in ("oracle_passed", "accuracy_passed"))]
    if event_oracles and event_oracles[-1] is False and actual_oracle is True and claimed_acc is None:
        discrepancies.append({"code": "trajectory_receipt_accuracy_mismatch", "field": "oracle_passed", "claimed": True, "actual": False, "evidence": "trajectory oracle_passed=False contradicts receipt"})

    consistent = len(discrepancies) == 0
    result_consistency = {
        "consistent": consistent, "passed": consistent, "reward_hacking_detected": not consistent,
        "discrepancies": discrepancies, "violations": discrepancies, "discrepancy_codes": [d["code"] for d in discrepancies],
        "claimed_summary": {"speedup_ratio": claimed_speedup, "hap_score": claimed_hap, "verdict": claimed_verdict, "outcome": claimed_outcome, "accuracy_passed": claimed_acc, "compile_passed": claimed_compile, "measurement_passed": claimed_meas},
        "verified_summary": {"speedup_ratio": actual_speedup, "hap_score": actual_hap, "verdict": actual_verdict, "compile_ok": actual_compile, "oracle_passed": actual_oracle, "generalization_ok": actual_gen, "measurement_passed": actual_meas, "ablation_retained": actual_ablation},
    }
    result_pass = bool(receipt and actual_compile is not False and actual_oracle is not False and actual_gen is not False and actual_meas is not False and actual_ablation is not False and (actual_verdict in {None, "supported", "subtarget_valid", "attained", "pass", "passed"}))
    calibration_verdict = "PROCESS_UNKNOWN" if proc_status == "UNKNOWN" else (("TRUSTED" if result_pass else "FIXABLE") if (proc_compliant and consistent) else ("HIGH_SCORE_SUSPECT" if (result_pass or claimed_verdict == "supported") else "SKILL_FAILED"))
    dual_ok = proc_compliant and consistent
    return {"schema_version": SCHEMA_CANN_SENTRY, "audit_passed": dual_ok and result_pass, "dual_verification_passed": dual_ok, "calibration_verdict": calibration_verdict, "process_compliance": process_compliance, "result_consistency": result_consistency, "swimlane_attribution": swimlane_attribution}


def audit_inference_benchmark_validity(benchmark_record: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Audit end-to-end `vllm-ascend` / MindIE inference benchmark records for methodology distortions."""
    rec = {**dict(benchmark_record or {}), **kwargs}
    violations: list[dict[str, Any]] = []
    normalized_metrics: dict[str, Any] = {}
    def _add(code: str, field_name: str, message: str, remediation: str) -> None:
        violations.append({"code": code, "severity": "error", "field": field_name, "message": message, "remediation": remediation})

    # 1. Unexcluded compilation / ACLGraph bucket capture warmup steps polluting steady-state TPOT/TTFT
    warmup_val = next((rec[k] for k in ("warmup_steps", "warmup_requests", "num_warmup", "warmup_iters") if k in rec and rec[k] is not None), None)
    warmup_count = int(_finite_float(warmup_val) or 0) if warmup_val is not None else None
    warmup_polluted = (warmup_count is not None and warmup_count <= 0) or bool(rec.get("include_warmup_in_metrics") or rec.get("warmup_included") or rec.get("compile_or_capture_included") or rec.get("aclgraph_capture_in_timed_window"))
    first_ttft, steady_ttft = _finite_float(rec.get("first_request_ttft_ms")), _finite_float(rec.get("steady_state_ttft_ms", rec.get("ttft_ms")))
    if first_ttft is not None and steady_ttft is not None and steady_ttft > 0.0 and first_ttft > 5.0 * steady_ttft and bool(rec.get("include_first_request", False)):
        warmup_polluted = True
    if warmup_polluted:
        _add("unexcluded_warmup_or_graph_capture", "warmup_steps", "Compilation or ACLGraph bucket capture warmup steps were not excluded from the timed TTFT/TPOT measurement window.", "Run >= 2-3 warmup requests covering all target batch/token buckets before starting the timed window so torch.npugraph_ex / ACLGraph capture is excluded.")

    # 2. Dummy / zero / uncalibrated synthetic weights used in quantized performance benchmarks
    quant_str, w_bit = str(rec.get("quantization") or rec.get("quant_method") or "").strip().lower(), _finite_float(rec.get("w_bit"))
    is_quantized = bool(quant_str and quant_str not in {"none", "null", "false", "fp16", "bf16", "float16", "bfloat16"}) or (w_bit is not None and w_bit <= 8)
    weights_src = str(rec.get("load_format") or rec.get("weights_source") or rec.get("weight_mode") or "").strip().lower()
    is_dummy_weights = weights_src in {"dummy", "random", "zero", "zeros", "synthetic", "uncalibrated"} or bool(rec.get("dummy_weights") or rec.get("zero_weights") or rec.get("synthetic_weights")) or (rec.get("calibrated_weights") is False)
    if is_quantized and is_dummy_weights:
        _add("dummy_or_uncalibrated_quant_weights", "load_format", f"Quantized benchmark ({quant_str or f'w{int(w_bit or 8)}'}) used {weights_src or 'dummy/uncalibrated'} weights, distorting dynamic quant scales, activation sparsity, and MoE routing.", "Benchmark quantized (W8A8/W4A8/MXFP8/MXFP4) models only with real msmodelslim-calibrated weights (--load-format auto/safetensors).")

    # 3. Speculative decoding (MTP / DSpark / DFlash / Eagle3) step latency reported without acceptance metrics
    spec_cfg = rec.get("speculative_config") if isinstance(rec.get("speculative_config"), Mapping) else {}
    spec_method = str(rec.get("speculative_method") or spec_cfg.get("method") or "").strip().lower()
    num_spec = int(_finite_float(rec.get("num_speculative_tokens", spec_cfg.get("num_speculative_tokens"))) or 0)
    is_speculative = bool(spec_method and spec_method not in {"none", "false", "0"}) or num_spec > 0 or bool(rec.get("speculative_enabled"))
    mean_acc_len = _finite_float(rec.get("mean_acceptance_length", rec.get("acceptance_length", rec.get("accepted_tokens_per_step"))))
    acc_rate = _finite_float(rec.get("acceptance_rate", rec.get("draft_acceptance_rate", rec.get("alpha"))))
    step_lat_ms, reported_tpot_ms = _finite_float(rec.get("step_latency_ms", rec.get("draft_step_latency_ms"))), _finite_float(rec.get("tpot_ms", rec.get("reported_tpot_ms")))
    if mean_acc_len is None and acc_rate is not None and num_spec > 0:
        mean_acc_len = 1.0 + num_spec * acc_rate
    if step_lat_ms is not None and mean_acc_len is not None and mean_acc_len > 0.0:
        normalized_metrics["effective_tpot_ms"] = step_lat_ms / mean_acc_len
        normalized_metrics["mean_acceptance_length"] = mean_acc_len
    if is_speculative and ((mean_acc_len is None and acc_rate is None) or (step_lat_ms is not None and reported_tpot_ms is not None and mean_acc_len is not None and mean_acc_len > 1.05 and abs(reported_tpot_ms - step_lat_ms) < 1e-3)):
        _add("speculative_decoding_missing_acceptance_metrics", "mean_acceptance_length", f"Speculative decoding ({spec_method or 'mtp/eagle'}) reported step latency as TPOT without effective draft acceptance length (`mean_acceptance_length = 1 + sum(alpha)`).", "Report mean_acceptance_length and acceptance_rate on real prompts and compute effective_tpot_ms = step_latency_ms / mean_acceptance_length.")

    # 4. Short output_len (< 32) misrepresented as steady-state long-context decode TPOT
    out_len_val = next((rec[k] for k in ("output_len", "max_tokens", "output_tokens", "decode_len") if k in rec and rec[k] is not None), None)
    out_len = int(_finite_float(out_len_val) or 0) if out_len_val is not None else None
    claim_mode = str(rec.get("benchmark_claim") or rec.get("claim_type") or rec.get("profile_mode") or "").strip().lower()
    if out_len is not None and 0 < out_len < 32 and (claim_mode in {"steady_state_tpot", "decode_tpot", "long_context_decode", "steady_state"} or bool(rec.get("claims_steady_state_tpot")) or ("steady_state_tpot_ms" in rec)):
        _add("short_output_len_tpot_distortion", "output_len", f"Short output window (output_len={out_len} < 32) is misrepresented as steady-state decode TPOT.", "Use output_len >= 128 (minimum >= 32) decode steps when measuring steady-state decode TPOT.")

    valid = len(violations) == 0
    return {
        "schema_version": SCHEMA_CANN_SENTRY, "valid": valid, "status": "valid" if valid else "distorted",
        "verdict": "TRUSTED" if valid else "DISTORTED", "violations": violations,
        "violation_codes": [v["code"] for v in violations],
        "remediation_steps": [v["remediation"] for v in violations], "normalized_metrics": normalized_metrics,
    }


__all__ = [
    "CANONICAL_SWIMLANES", "OPERATOR_COMPLEXITY_TAXONOMY", "RSICandidateResult",
    "SCHEMA_CANN_SENTRY", "audit_inference_benchmark_validity", "audit_sentry_trajectory",
    "classify_operator_complexity", "compare_swimlane_attributions", "compute_hap_score",
    "compute_operator_composite_score",
]
