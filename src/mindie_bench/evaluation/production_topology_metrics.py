"""Distributed invariance, stress-distribution and SLA gates for S0-S3.

[Role]
Evaluate asymmetric-DP batch matrices, MoE skew, varlen/continuous batching,
concurrency ladders, TP/DP/ACLGraph parallelism invariance, warm-server SLA,
per-rank workspace and guard-workload evidence.

[Invariants]
- Missing measurements are `not_measured`, never a zero or a pass.
- `HCCL_EXEC_TIMEOUT`, idle-rank non-participation, `conv_state` / `ssm_state`
  drift, KL/top-1 divergence, checksum mismatch, SLA miss, workspace overflow and
  guard regression fail closed. `L4-03` names the workspace/guard boundary.
- Metrics are descriptive only when no threshold is declared; defaults exist for
  p95 TTFT/TPOT, goodput, comm/overlap ratio and rank skew.

[Timeouts & Locks]
- Pure in-memory evaluation with no sleeps, subprocesses or locks.

[Side-Effect Policy]
- Reads caller-provided evidence only; never contacts an NPU or writes a file.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "mindie-bench-production-topology/1"
L4_03 = "L4-03"
_ASYM_MATRICES = ("[B,0]", "[0,B]", "[1,B]", "[0,0]")
_CONCURRENCY_LEVELS = (1, 16, 64)
_DEFAULT_TARGETS = {
    "ttft_p95_ms": ("max", 5000.0),
    "tpot_p95_ms": ("max", 500.0),
    "goodput_tokens_per_sec": ("min", 1.0),
    "comm_ratio": ("max", 0.8),
    "overlap_ratio": ("min", 0.1),
    "rank_skew_ratio": ("max", 1.5),
}
def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    seen: set[Any] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)

def _seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    return list(value) if isinstance(value, Sequence) else [value]

def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _state_map(value: Any) -> dict[int, str]:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            try:
                result[int(key)] = str(item)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: str(item) for index, item in enumerate(value)}
    return {}


def _batch_matrix(case: Mapping[str, Any]) -> tuple[int, ...] | None:
    value = next(
        (case[key] for key in ("batch_per_rank", "batch", "batch_sizes", "batches",
                               "matrix", "per_rank_batch") if key in case),
        None,
    )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _case_failures(case: Mapping[str, Any], matrix: Sequence[int]) -> list[str]:
    failures = []
    text = " ".join(
        str(case.get(key, "")) for key in ("error", "stderr", "status", "message", "exception")
    ).upper()
    if "HCCL_EXEC_TIMEOUT" in text or case.get("timeout") is True:
        failures.append("HCCL_EXEC_TIMEOUT")
    participants = next(
        (case[key] for key in ("collective_participants", "participating_ranks",
                               "collective_ranks", "ranks") if key in case),
        None,
    )
    if participants is not None and isinstance(participants, Sequence):
        present = {int(item) for item in participants if str(item).lstrip("-").isdigit()}
        if set(range(len(matrix))) - present:
            failures.append("missing_idle_rank_collective_participation")
    if case.get("idle_rank_participated") is False or case.get("missing_idle_rank_collective") is True:
        failures.append("missing_idle_rank_collective_participation")
    for name in ("conv_state", "ssm_state"):
        if case.get(f"{name}_corruption") is True or case.get("recurrent_state_corruption") is True:
            failures.append(f"{name}_corruption")
        if case.get(f"{name}_unchanged") is False:
            failures.append(f"{name}_corruption")
    before = _state_map(case.get("state_checksums_before") or case.get("state_before"))
    after = _state_map(case.get("state_checksums_after") or case.get("state_after"))
    for rank, checksum in before.items():
        if rank < len(matrix) and matrix[rank] == 0 and after.get(rank, checksum) != checksum:
            failures.append("recurrent_state_corruption")
    return failures


def _evaluate_asymmetric(block: Mapping[str, Any]) -> dict[str, Any]:
    cases = block.get("cases") or block.get("runs") or block.get("matrix_cases") or []
    if isinstance(cases, Mapping):
        cases = [{"name": key, **value} for key, value in cases.items() if isinstance(value, Mapping)]
    rows, seen, failures = [], set(), []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        matrix = _batch_matrix(case)
        if matrix is None:
            failures.append("missing_batch_matrix")
            continue
        seen.add(matrix)
        row_failures = _case_failures(case, matrix)
        failures.extend(row_failures)
        rows.append(
            {"name": case.get("name") or case.get("case_id"), "matrix": list(matrix),
             "failures": row_failures}
        )
    checks = (
        (any(row[0] > 0 and row[1] == 0 for row in seen if len(row) == 2), "[B,0]"),
        (any(row[0] == 0 and row[1] > 0 for row in seen if len(row) == 2), "[0,B]"),
        (any(row[0] == 1 and row[1] >= 1 for row in seen if len(row) == 2), "[1,B]"),
        ((0, 0) in seen, "[0,0]"),
    )
    failures.extend(f"missing_matrix:{label}" for present, label in checks if not present)
    failures = list(_unique(failures))
    return {
        "passed": bool(rows) and not failures,
        "cases": rows,
        "matrices": [list(row) for row in sorted(seen)],
        "required_matrices": list(_ASYM_MATRICES),
        "failures": failures,
    }


def _rank_times(block: Mapping[str, Any]) -> list[float]:
    value = next(
        (block[key] for key in ("rank_step_ms", "rank_times_ms", "step_ms_by_rank",
                                "steps_ms", "rank_step_time") if key in block),
        None,
    )
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        return [number for _, item in ordered if (number := _num(item)) is not None]
    result = []
    for item in _seq(value):
        if isinstance(item, Mapping):
            number = _num(item.get("step_ms") or item.get("ms") or item.get("value"))
        else:
            number = _num(item)
        if number is not None:
            result.append(number)
    return result


def _evaluate_moe(block: Mapping[str, Any], target: float) -> dict[str, Any]:
    times = _rank_times(block)
    ratio = _num(block.get("rank_skew_ratio"))
    median = statistics.median(times) if times else None
    if ratio is None and times and median and median > 0:
        ratio = max(times) / median
    stragglers = (
        [index for index, value in enumerate(times) if median and value > median * target]
        if times and median else []
    )
    counts = block.get("expert_counts") or block.get("expert_load")
    max_share = None
    if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)) and counts:
        total = sum(item for item in counts if isinstance(item, (int, float)) and item >= 0)
        max_share = max(counts) / total if total else None
    failures = []
    if ratio is None:
        failures.append("rank_skew_ratio_not_measured")
    elif ratio > target:
        failures.append("straggler_rank_skew")
    if stragglers:
        failures.append("straggler_ranks")
    if block.get("eplb_failed") is True or block.get("all_to_all_failed") is True:
        failures.append("expert_load_balance_failed")
    return {
        "passed": not failures,
        "rank_step_ms": times,
        "median_rank_step_ms": median,
        "max_rank_step_ms": max(times) if times else None,
        "rank_skew_ratio": round(ratio, 6) if ratio is not None else None,
        "straggler_ranks": stragglers,
        "max_expert_share": round(max_share, 6) if max_share is not None else None,
        "failures": failures,
    }


def _evaluate_varlen(block: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    if block.get("varlen_prefill") is None and block.get("prefill") is None:
        failures.append("varlen_prefill_not_measured")
    if block.get("continuous_batching_decode") is None and block.get("decode") is None:
        failures.append("continuous_batching_decode_not_measured")
    if block.get("continuous_batching") is False:
        failures.append("continuous_batching_disabled")
    if block.get("shape_error") or block.get("error"):
        failures.append("mixed_shape_failure")
    lengths = block.get("lengths") or block.get("request_lengths") or []
    return {
        "passed": not failures,
        "lengths": list(lengths) if isinstance(lengths, Sequence) else [],
        "failures": failures,
    }


def _levels(block: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    value = block.get("levels") or block.get("results") or block.get("ladder") or {}
    rows: dict[int, Mapping[str, Any]] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            try:
                rows[int(key)] = item if isinstance(item, Mapping) else {"value": item}
            except (TypeError, ValueError):
                continue
    for item in _seq(value if not isinstance(value, Mapping) else None):
        if isinstance(item, Mapping):
            try:
                rows[int(item.get("level") or item.get("concurrency"))] = item
            except (TypeError, ValueError):
                continue
    return rows


def _evaluate_concurrency(block: Mapping[str, Any]) -> dict[str, Any]:
    rows = _levels(block)
    failures = []
    for level in _CONCURRENCY_LEVELS:
        if level not in rows:
            failures.append(f"missing_concurrency:{level}")
            continue
        row = rows[level]
        if row.get("passed") is False or row.get("success") is False or row.get("error"):
            failures.append(f"concurrency_failed:{level}")
    return {
        "passed": not failures,
        "required_levels": list(_CONCURRENCY_LEVELS),
        "observed_levels": sorted(rows),
        "goodput_tokens_per_sec": {
            level: _num(row.get("goodput_tokens_per_sec") or row.get("goodput"))
            for level, row in rows.items()
        },
        "failures": failures,
    }


def _distribution_name(key: str) -> str:
    lowered = key.casefold()
    if "asymmetric" in lowered or "dp_batch" in lowered or "batch_matrix" in lowered:
        return "asymmetric_dp"
    if "moe" in lowered or "expert" in lowered:
        return "moe_skew"
    if "varlen" in lowered or "continuous" in lowered:
        return "varlen_continuous"
    if "concurren" in lowered or "ladder" in lowered:
        return "concurrency_ladder"
    return key


def _distributions(stress: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(stress, Mapping):
        result = {}
        for key, value in stress.items():
            name = _distribution_name(str(key))
            if isinstance(value, Mapping):
                result[name] = value
            elif name == "asymmetric_dp" and isinstance(value, Sequence):
                result[name] = {"cases": list(value)}
        return result
    result = {}
    for item in _seq(stress):
        if isinstance(item, Mapping):
            key = str(item.get("distribution") or item.get("kind") or item.get("name") or "case")
            result[_distribution_name(key)] = item
    return result


def _metric(metrics: Mapping[str, Any], name: str, aliases: Sequence[str] = ()) -> float | None:
    return next(
        (number for key in (name, *aliases) if (number := _num(metrics.get(key))) is not None),
        None,
    )


def _sla_metrics(
    metrics: Mapping[str, Any], stress: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    ttft = [value for item in _seq(metrics.get("ttft_ms") or metrics.get("ttft_samples_ms"))
            if (value := _num(item)) is not None]
    tpot = [value for item in _seq(metrics.get("tpot_ms") or metrics.get("itl_ms")
                                   or metrics.get("tpot_samples_ms"))
            if (value := _num(item)) is not None]
    values = {
        "ttft_p50_ms": _metric(metrics, "ttft_p50_ms", ("p50_ttft_ms", "ttft_p50"))
        or (_percentile(ttft, 0.50) if ttft else None),
        "ttft_p95_ms": _metric(metrics, "ttft_p95_ms", ("p95_ttft_ms", "ttft_p95"))
        or (_percentile(ttft, 0.95) if ttft else None),
        "tpot_p50_ms": _metric(metrics, "tpot_p50_ms", ("p50_tpot_ms", "tpot_p50"))
        or (_percentile(tpot, 0.50) if tpot else None),
        "tpot_p95_ms": _metric(metrics, "tpot_p95_ms", ("p95_tpot_ms", "tpot_p95"))
        or (_percentile(tpot, 0.95) if tpot else None),
        "goodput_tokens_per_sec": _metric(
            metrics, "goodput_tokens_per_sec", ("goodput", "output_tokens_per_sec")
        ),
    }
    if values["goodput_tokens_per_sec"] is None:
        output = _num(metrics.get("output_tokens"))
        elapsed = _num(metrics.get("elapsed_sec") or metrics.get("duration_sec"))
        if output is not None and elapsed and elapsed > 0:
            values["goodput_tokens_per_sec"] = output / elapsed
    comm = _metric(metrics, "comm_time_ms", ("communication_ms", "comm_ms"))
    step = _metric(metrics, "step_time_ms", ("step_ms",))
    values["comm_ratio"] = _metric(metrics, "comm_ratio")
    if values["comm_ratio"] is None and comm is not None and step and step > 0:
        values["comm_ratio"] = comm / step
    values["overlap_ratio"] = _metric(metrics, "overlap_ratio", ("overlap",))
    if values["overlap_ratio"] is None and comm:
        overlap = _num(metrics.get("overlap_time_ms") or metrics.get("overlapped_ms"))
        compute = _num(metrics.get("compute_time_ms") or metrics.get("compute_ms"))
        if overlap is not None:
            values["overlap_ratio"] = min(max(overlap / comm, 0.0), 1.0)
        elif compute is not None and step is not None:
            values["overlap_ratio"] = min(max((comm + compute - step) / comm, 0.0), 1.0)
    moe = stress.get("moe_skew") or stress.get("moe") or {}
    values["rank_skew_ratio"] = _metric(metrics, "rank_skew_ratio")
    if values["rank_skew_ratio"] is None and isinstance(moe, Mapping):
        values["rank_skew_ratio"] = _num(moe.get("rank_skew_ratio"))
    missing = [name for name, value in values.items() if value is None]
    return {
        name: (round(value, 6) if value is not None else None)
        for name, value in values.items()
    }, missing


def _targets(sla_targets: Mapping[str, Any] | None, metrics: Mapping[str, Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {
        name: {"direction": direction, "value": threshold}
        for name, (direction, threshold) in _DEFAULT_TARGETS.items()
        if metrics.get(name) is not None
    }
    for key, value in (sla_targets or {}).items():
        name = key
        for prefix in ("max_", "min_"):
            name = name.removeprefix(prefix)
        direction = "min" if key.startswith("min_") else "max"
        threshold = value
        if isinstance(value, Mapping):
            threshold = value.get("value", value.get("max", value.get("min")))
            direction = "min" if "min" in value else direction
        threshold = _num(threshold)
        if threshold is not None:
            targets[name] = {"direction": direction, "value": threshold}
    return targets


def _sla_gates(metrics: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    results, failures = {}, []
    for name, target in targets.items():
        value = metrics.get(name)
        if value is None:
            results[name] = {"status": "not_measured", "threshold": target}
            failures.append(f"{name}_not_measured")
            continue
        passed = (
            value >= target["value"] if target["direction"] == "min"
            else value <= target["value"]
        )
        results[name] = {
            "status": "passed" if passed else "failed",
            "value": value,
            "threshold": target,
        }
        if not passed:
            failures.append(f"{name}_miss")
    return {"passed": not failures, "targets": dict(targets), "results": results,
            "failures": failures}


def _variant_value(variant: Mapping[str, Any], key: str, *, aggregate: str) -> float | None:
    value = variant.get(key)
    if isinstance(value, Mapping):
        values = [number for item in value.values() if (number := _num(item)) is not None]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [number for item in value if (number := _num(item)) is not None]
    else:
        values = [number] if (number := _num(value)) is not None else []
    return (max(values) if aggregate == "max" else min(values)) if values else None

def _state_match(variant: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    expected = variant.get("expected_rank_state_checksums") or reference.get(
        "rank_state_checksums"
    ) or reference.get("state_checksums")
    actual = variant.get("rank_state_checksums") or variant.get("state_checksums")
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    expected_map, actual_map = _state_map(expected), _state_map(actual)
    if not expected_map or not actual_map:
        return False
    if len(expected_map) == 1 and len(actual_map) > 1:
        return all(value == next(iter(expected_map.values())) for value in actual_map.values())
    return all(
        actual_map.get(key) == value for key, value in expected_map.items() if key in actual_map
    )


def _mode(variant: Mapping[str, Any]) -> str:
    name = str(variant.get("mode") or variant.get("graph_mode") or variant.get("name") or "")
    return "aclgraph" if "aclgraph" in name.casefold() or "graph" in name.casefold() else "eager"

def _softmax(row: Sequence[float]) -> list[float]:
    maximum = max(row)
    exponents = [math.exp(value - maximum) for value in row]
    total = sum(exponents)
    return [value / total for value in exponents] if total else [0.0 for _ in row]

def _logits_metrics(
    reference: Mapping[str, Any], variant: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    expected = reference.get("logits") or reference.get("reference_logits")
    actual = variant.get("logits") or variant.get("candidate_logits")
    if not isinstance(expected, Sequence) or not isinstance(actual, Sequence):
        return None, None
    if not expected or not actual:
        return None, None
    pairs = (
        list(zip(expected, actual, strict=False))
        if isinstance(expected[0], Sequence) and isinstance(actual[0], Sequence)
        else [(expected, actual)]
    )
    divergences, agreements = [], []
    for expected_row, actual_row in pairs:
        if len(expected_row) != len(actual_row) or not expected_row:
            continue
        left, right = _softmax(expected_row), _softmax(actual_row)
        divergences.append(
            sum(
                probability * math.log(probability / max(candidate, 1e-300))
                for probability, candidate in zip(left, right, strict=True)
                if probability > 0
            )
        )
        agreements.append(
            1.0
            if max(range(len(left)), key=left.__getitem__)
            == max(range(len(right)), key=right.__getitem__)
            else 0.0
        )
    return (
        (max(divergences) if divergences else None),
        (sum(agreements) / len(agreements) if agreements else None),
    )

def _parallelism_gate(
    reference: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    *,
    max_kl: float,
    min_top1: float,
    require_aclgraph: bool,
) -> dict[str, Any]:
    failures, rows = [], []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        kl = _variant_value(variant, "kl_divergence", aggregate="max")
        top1 = _variant_value(variant, "top1_agreement", aggregate="min")
        computed_kl, computed_top1 = _logits_metrics(reference, variant)
        if kl is None:
            kl = computed_kl
        if top1 is None:
            top1 = computed_top1
        states_ok = _state_match(variant, reference)
        row_failures = []
        if variant.get("error") or variant.get("timeout"):
            row_failures.append("variant_failed")
        if kl is None:
            row_failures.append("kl_divergence_not_measured")
        elif kl > max_kl:
            row_failures.append("kl_divergence_exceeded")
        if top1 is None:
            row_failures.append("top1_agreement_not_measured")
        elif top1 < min_top1:
            row_failures.append("top1_agreement_below_floor")
        if not states_ok:
            row_failures.append("rank_state_checksum_mismatch")
        rows.append(
            {
                "name": variant.get("name") or variant.get("run_id") or f"variant-{len(rows)}",
                "mode": _mode(variant),
                "tp": int(variant.get("tp", 1)),
                "dp": int(variant.get("dp", 1)),
                "kl_divergence": kl,
                "top1_agreement": top1,
                "rank_state_checksums_match": states_ok,
                "passed": not row_failures,
                "failures": row_failures,
            }
        )
        failures.extend(row_failures)
    if not rows:
        failures.append("no_parallel_variant_measured")
    if not any(row["tp"] > 1 or row["dp"] > 1 for row in rows):
        failures.append("no_tp_gt_1_or_dp_gt_1_variant")
    if require_aclgraph and not any(row["mode"] == "aclgraph" for row in rows):
        failures.append("aclgraph_variant_not_measured")
    eager_pairs = {(int(reference.get("tp", 1)), int(reference.get("dp", 1)))}
    eager_pairs.update(
        (row["tp"], row["dp"]) for row in rows if row["mode"] == "eager"
    )
    if any((row["tp"], row["dp"]) not in eager_pairs
           for row in rows if row["mode"] == "aclgraph"):
        failures.append("aclgraph_has_no_eager_peer")
    failures = list(_unique(failures))
    return {
        "passed": not failures,
        "reference": {
            "tp": int(reference.get("tp", 1)),
            "dp": int(reference.get("dp", 1)),
            "mode": _mode(reference),
            "rank_state_checksums": reference.get("rank_state_checksums")
            or reference.get("state_checksums"),
        },
        "max_kl_divergence": max_kl,
        "min_top1_agreement": min_top1,
        "variants": rows,
        "failures": failures,
    }

def _workspace_gate(
    evidence: Mapping[str, Any],
    metrics: Mapping[str, Any],
    limit: int | None,
    guards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deltas = (
        metrics.get("rank_workspace_delta_bytes")
        or evidence.get("rank_workspace_delta_bytes")
        or evidence.get("workspace_delta_bytes_by_rank")
    )
    if isinstance(deltas, Mapping):
        values = [number for item in deltas.values() if (number := _num(item)) is not None]
    else:
        values = [number for item in _seq(deltas) if (number := _num(item)) is not None]
    explicit = _num(
        metrics.get("max_rank_workspace_delta_bytes")
        if metrics.get("max_rank_workspace_delta_bytes") is not None
        else evidence.get("max_rank_workspace_delta_bytes")
    )
    maximum = max([*values, *([explicit] if explicit is not None else [])], default=None)
    failures = []
    if limit is not None and maximum is None:
        failures.append("workspace_delta_not_measured")
    elif limit is not None and maximum is not None and maximum > limit:
        failures.append("max_rank_workspace_delta_above_limit")
    guard_rows = []
    for guard in guards:
        if not isinstance(guard, Mapping):
            continue
        name = str(guard.get("name") or guard.get("workload") or "guard")
        row_failures = []
        measured = guard.get("measurement_valid", True)
        baseline = _num(guard.get("baseline_ms") or guard.get("baseline"))
        candidate = _num(guard.get("candidate_ms") or guard.get("candidate"))
        ratio = _num(guard.get("regression_ratio"))
        tolerance = _num(guard.get("max_regression_ratio"))
        tolerance = 1.0 if tolerance is None else tolerance
        if ratio is None and baseline and candidate is not None:
            ratio = candidate / baseline
        if measured is False:
            row_failures.append("guard_measurement_invalid")
        if guard.get("passed") is False:
            row_failures.append("guard_workload_failed")
        if ratio is not None and ratio > tolerance + 1e-12:
            row_failures.append("guard_workload_regressed")
        guard_rows.append(
            {"name": name, "ratio": round(ratio, 6) if ratio is not None else None,
             "max_regression_ratio": tolerance, "passed": not row_failures,
             "failures": row_failures}
        )
        failures.extend(row_failures)
    failures = list(_unique(failures))
    return {
        "code": L4_03,
        "passed": not failures,
        "max_rank_workspace_delta_bytes": maximum,
        "max_workspace_delta_bytes": limit,
        "workspace_delta_bytes_by_rank": values,
        "guard_workloads": guard_rows,
        "failures": failures,
    }

def evaluate_distributed_invariance_and_sla(
    evidence: Mapping[str, Any] | None = None,
    *,
    reference: Mapping[str, Any] | None = None,
    variants: Sequence[Mapping[str, Any]] = (),
    stress: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
    guard_workloads: Sequence[Mapping[str, Any]] = (),
    sla_targets: Mapping[str, Any] | None = None,
    max_workspace_delta_bytes: int | None = None,
    max_rank_workspace_delta_bytes: int | None = None,
    max_kl_divergence: float = 1e-3,
    min_top1_agreement: float = 0.999,
    require_aclgraph: bool = True,
    require_all_stress: bool = True,
) -> dict[str, Any]:
    """Evaluate the production stress matrix, invariance, SLA and L4-03 guards."""
    data = dict(evidence or {})
    reference_run = {
        **_mapping(data.get("reference") or data.get("reference_run")),
        **dict(reference or {}),
    }
    variant_rows = list(variants) if variants else list(
        data.get("variants")
        or data.get("parallel_variants")
        or data.get("parallel")
        or data.get("candidates")
        or data.get("parallel_runs")
        or data.get("aclgraph")
        or []
    )
    if isinstance(data.get("aclgraph_variants"), Sequence):
        variant_rows.extend(data["aclgraph_variants"])
    if isinstance(variant_rows, Mapping):
        variant_rows = [{"name": key, **value} for key, value in variant_rows.items()]
    stress_rows = _distributions(
        stress
        if stress is not None
        else data.get("stress")
        or data.get("stress_distributions")
        or data.get("stress_inputs")
        or data.get("distributions")
    )
    sla_block = _mapping(data.get("sla"))
    metric_rows = {
        **(_mapping(sla_block.get("metrics")) if "metrics" in sla_block else sla_block),
        **_mapping(data.get("metrics") or data.get("sla_metrics") or data.get("system_metrics")),
        **dict(metrics or {}),
    }
    raw_guards = guard_workloads if guard_workloads else (
        data.get("guard_workloads") or data.get("guards") or []
    )
    if isinstance(raw_guards, Mapping):
        guards = [{"name": key, **value} for key, value in raw_guards.items()
                  if isinstance(value, Mapping)]
    else:
        guards = list(raw_guards)
    limit = max_workspace_delta_bytes
    if limit is None:
        limit = max_rank_workspace_delta_bytes
    if limit is None:
        limit = data.get("max_workspace_delta_bytes") or data.get("max_rank_workspace_delta_bytes")
        if limit is None:
            limit = _mapping(data.get("sla_targets")).get("max_workspace_delta_bytes")
    limit = int(limit) if isinstance(limit, (int, float)) and not isinstance(limit, bool) else None
    targets_input = {
        **_mapping(sla_block.get("targets")),
        **_mapping(data.get("sla_targets")),
        **dict(sla_targets or {}),
    }
    if limit is None:
        target_limit = _num(
            targets_input.get("max_rank_workspace_delta_bytes")
            or targets_input.get("max_workspace_delta_bytes")
        )
        limit = int(target_limit) if target_limit is not None else None
    asym = _evaluate_asymmetric(stress_rows.get("asymmetric_dp") or {})
    computed_metrics, missing = _sla_metrics(metric_rows, stress_rows)
    moe_block = stress_rows.get("moe_skew") or stress_rows.get("moe") or {}
    moe_target = _num(_mapping(targets_input).get("rank_skew_ratio")) or 1.5
    moe = _evaluate_moe(moe_block, moe_target)
    if computed_metrics["rank_skew_ratio"] is None:
        computed_metrics["rank_skew_ratio"] = moe["rank_skew_ratio"]
        if computed_metrics["rank_skew_ratio"] is not None and "rank_skew_ratio" in missing:
            missing.remove("rank_skew_ratio")
    varlen = _evaluate_varlen(
        stress_rows.get("varlen_continuous") or stress_rows.get("varlen") or {}
    )
    concurrency = _evaluate_concurrency(
        stress_rows.get("concurrency_ladder") or stress_rows.get("concurrency") or {}
    )
    targets = _targets(targets_input, computed_metrics)
    sla = _sla_gates(computed_metrics, targets)
    workspace = _workspace_gate(data, metric_rows, limit, guards)
    computed_metrics["max_rank_workspace_delta_bytes"] = workspace[
        "max_rank_workspace_delta_bytes"
    ]
    invariance = _parallelism_gate(
        reference_run,
        variant_rows,
        max_kl=max_kl_divergence,
        min_top1=min_top1_agreement,
        require_aclgraph=require_aclgraph,
    )
    stages = {
        "asymmetric_dp": asym,
        "moe_skew": moe,
        "varlen_continuous": varlen,
        "concurrency_ladder": concurrency,
        "parallelism_invariance": invariance,
        "sla": sla,
        L4_03: workspace,
    }
    failures = []
    if not asym["passed"]:
        failures.append("asymmetric_dp")
    if not moe["passed"]:
        failures.append("moe_skew")
    if not varlen["passed"]:
        failures.append("varlen_continuous")
    if not concurrency["passed"]:
        failures.append("concurrency_ladder")
    if not invariance["passed"]:
        failures.append("parallelism_invariance")
    if not sla["passed"]:
        failures.append("sla")
    if not workspace["passed"]:
        failures.append(L4_03)
    if require_all_stress:
        for name in ("asymmetric_dp", "moe_skew", "varlen_continuous", "concurrency_ladder"):
            if name not in stress_rows and name not in failures:
                failures.append(name)
    status = "passed" if not failures else "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "verdict": status,
        "passed": not failures,
        "failed_gates": failures,
        "failures": failures,
        "stages": stages,
        **stages,
        **computed_metrics,
        "gates": {name: bool(stage["passed"]) for name, stage in stages.items()},
        "stress_distributions": stress_rows,
        "parallelism_invariance": invariance,
        "parallelism_invariance_gate": invariance,
        "metrics": computed_metrics,
        "missing_metrics": missing,
        "sla": sla,
        "sla_targets": targets,
        "l4_03": workspace,
        "l4_03_guard": workspace,
        "workspace": workspace,
        "guard_workloads": workspace["guard_workloads"],
        "max_rank_workspace_delta_bytes": workspace["max_rank_workspace_delta_bytes"],
    }

__all__ = ["L4_03", "SCHEMA_VERSION", "evaluate_distributed_invariance_and_sla"]
