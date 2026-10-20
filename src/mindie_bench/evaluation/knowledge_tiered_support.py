"""Shared production bindings and bounded helpers for tiered knowledge evaluation.

[Role]
Bind `mindie_knowledge.graded` once and provide repository, condition, L4 guard,
feedback identity and distillation helpers to `knowledge_tiered`.
[Invariants]
- Every condition, index, usage and candidate decision is delegated to production.
- No helper persists a parallel catalog, manifest, event ledger or decision state.
[Timeouts & Locks]
- Pure local work; callers own SQLite, process and distillation deadlines.
[Side-Effect Policy]
- Corpus copies write only under the caller's workspace; helpers never touch the
  selected source corpus or publish knowledge.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_KNOWLEDGE_ROOT = Path("/Users/maoxx241/code/mindie-release-20261014/knowledge")
_KNOWLEDGE_SRC = _KNOWLEDGE_ROOT / "src"  # historical src-layout fallback


def _load_graded():
    from mindie_knowledge.graded.conditions import match_conditions  # noqa: F401
    from mindie_knowledge.graded.explore import (  # noqa: F401
        assess_stagnation,
        build_exploration_context,
        evaluate_exploration_candidate,
    )
    from mindie_knowledge.graded.feedback import (  # noqa: F401
        record_usage_event,
        summarize_usage_events,
    )
    from mindie_knowledge.graded.index import RoutingIndex  # noqa: F401
    from mindie_knowledge.graded.maintenance import growth_metrics  # noqa: F401
    from mindie_knowledge.graded.repository import GradedRepository  # noqa: F401
    from mindie_knowledge.graded.units import GradedUnit  # noqa: F401
    from mindie_knowledge.loop.engine import Engine  # noqa: F401
    from mindie_knowledge.loop.store import Store  # noqa: F401

    return locals()


try:
    _graded = _load_graded()
except ImportError:
    # A present-but-broken installation is an error, not a silent shadowing.
    if importlib.util.find_spec("mindie_knowledge") is not None:
        raise
    # Lightweight import fallback only; never duplicate graded logic.
    for _candidate in (_KNOWLEDGE_SRC, _KNOWLEDGE_ROOT):
        if _candidate.is_dir() and str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
    _graded = _load_graded()

GradedRepository = _graded["GradedRepository"]
RoutingIndex = _graded["RoutingIndex"]
GradedUnit = _graded["GradedUnit"]
Store = _graded["Store"]
Engine = _graded["Engine"]
growth_metrics = _graded["growth_metrics"]
match_conditions = _graded["match_conditions"]
record_usage_event = _graded["record_usage_event"]
summarize_usage_events = _graded["summarize_usage_events"]
evaluate_exploration_candidate = _graded["evaluate_exploration_candidate"]
assess_stagnation = _graded["assess_stagnation"]
build_exploration_context = _graded["build_exploration_context"]

SCHEMA_VERSION = "mindie-bench-knowledge-tiered/1"
CONDITIONS = ("no_knowledge", "oracle", "normal_retrieval", "with_distractors")
STATUSES = ("matched", "mismatch", "condition_unknown")
BUCKETS = ("both_pass", "opt_regression", "both_fail", "opt_improvement", "evaluation_error")
EVAL_OUTCOMES = ("supported", "refuted", "inconclusive")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _seq(value: Any) -> list[Any]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence) else [value]


def _mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _ratio(numerator: int, denominator: int, *, missing: int = 0) -> dict:
    return {
        "value": round(numerator / denominator, 6) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "missing": missing,
    }


def _status(value: Any) -> str:
    return {
        "unknown": "condition_unknown",
        "condition_unknown": "condition_unknown",
        "mismatched": "mismatch",
        "mismatch": "mismatch",
        "matched": "matched",
    }.get(str(value), str(value))


def _unit_id(ref: Any) -> str:
    text = str(ref).strip()
    if text.startswith("mindie://"):
        text = text.rsplit("/units/", 1)[-1]
    return text.split("@", 1)[0]


def _revision(ref: Any) -> int | None:
    text = str(ref).strip()
    if "@" not in text:
        return None
    tail = text.rsplit("@", 1)[1]
    return int(tail) if tail.isdigit() else None


def _domain(corpus: Path) -> str:
    contract = corpus / "publication-contract.json"
    if contract.is_file():
        value = json.loads(contract.read_text())
        if isinstance(value.get("domain"), str):
            return value["domain"]
    return "bench-knowledge"


def _copy_corpus(source: Path, destination: Path) -> Path:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    return destination


def _distractor_unit(value: Mapping) -> GradedUnit:
    if isinstance(value.get("unit"), Mapping):
        value = value["unit"]
    frontmatter = value.get("frontmatter") or value.get("metadata")
    if frontmatter is None:
        frontmatter = {
            key: item
            for key, item in value.items()
            if key not in {"body", "label", "kind", "query", "observed", "expected_status", "reason"}
        }
    body = value.get("body") or value.get("markdown_body")
    if not isinstance(frontmatter, Mapping) or not isinstance(body, str):
        raise ValueError("a distractor requires frontmatter and body")
    return GradedUnit.from_parts(dict(frontmatter), body, value.get("path") or None)


def _inject_distractors(corpus: Path, distractors: Sequence[Mapping]) -> list[dict]:
    injected = []
    for item in distractors:
        unit = _distractor_unit(item)
        path = corpus / (unit.path or f"{unit.metadata['domain_path']}/{unit.id}.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit.render(), encoding="utf-8")
        injected.append(
            {
                "unit_id": unit.id,
                "label": str(item.get("label") or item.get("kind") or "distractor"),
                "expected_status": _status(item.get("expected_status", "mismatch")),
                "observed": dict(item.get("observed") or {}),
                "unit_conditions": dict(unit.metadata["conditions"]),
                "path": path.relative_to(corpus).as_posix(),
            }
        )
    return injected


def _repository(workspace: Path, corpus: Path, domain: str | None = None):
    store = Store(workspace / "store", domain or _domain(corpus))
    store.graded.install_root(corpus)
    return store, store.graded


def _pin(repository: GradedRepository, ref: Any) -> str:
    text = str(ref).strip()
    index = repository._load()
    if text.startswith("mindie://"):
        tail = text.split("/units/", 1)[-1]
        ident, _, revision = tail.partition("@")
        if not revision:
            unit = index.units.get(ident)
            if unit is None:
                raise ValueError(f"unknown graded unit: {ident}")
            text = repository.ref(ident, unit.revision)
        return text
    ident, _, revision = text.partition("@")
    unit = index.units.get(ident)
    if unit is None:
        raise ValueError(f"unknown graded unit: {ident}")
    return repository.ref(ident, int(revision) if revision.isdigit() else unit.revision)


def _loaded_view(repository: GradedRepository, refs: Sequence[Any], conditions: Mapping | None,
                 *, token_budget: int = 4096, known_refs: Sequence[Any] = ()) -> dict:
    pinned = [_pin(repository, ref) for ref in refs]
    if not pinned:
        return {"units": [], "requires": [], "unresolved": [], "usage": {}}
    return repository.explain(
        refs=pinned,
        conditions=dict(conditions or {}),
        known_refs=[_pin(repository, ref) for ref in known_refs],
        token_budget=token_budget,
    )


def _view_ids(view: Mapping) -> set[str]:
    return {
        str(item.get("id"))
        for bucket in ("units", "requires")
        for item in _seq(view.get(bucket))
        if item.get("id")
    }


def _case_status(index: RoutingIndex, ref: Any, conditions: Mapping) -> str:
    unit = index.units.get(_unit_id(ref))
    if unit is None:
        return "not_found"
    return _status(match_conditions(unit.metadata["conditions"], dict(conditions)).status)


def _premise_complete(index, ref, conditions, loaded_ids: set[str]) -> tuple[bool, str]:
    ident = _unit_id(ref)
    unit = index.units.get(ident)
    if unit is None:
        return False, "unknown_unit"
    closure = [*index.closure.get(ident, []), ident]
    if not set(closure) <= loaded_ids:
        return False, "missing_body"
    statuses = [match_conditions(index.units[node].metadata["conditions"], dict(conditions)) for node in closure]
    if any(item.status != "matched" for item in statuses):
        return False, "missing_condition"
    return True, "complete"


def _natural_trigger(case: Mapping) -> bool:
    return not bool(case.get("retrieval_expected", True)) or case.get("query") is None


def _source_hash(root: Path) -> str:
    from mindie_bench.io import tree_hash

    return tree_hash(root)


def freeze_l4_candidate(candidate_id: str, artifact: Any = None) -> dict:
    """Freeze the candidate identity before any held-out result is classified."""
    if isinstance(artifact, Mapping) and artifact.get("artifact_sha256"):
        digest = str(artifact["artifact_sha256"])
    elif isinstance(artifact, Path) or (isinstance(artifact, str) and Path(artifact).exists()):
        path = Path(artifact)
        if path.is_dir():
            digest = _sha({str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
                           for item in sorted(path.rglob("*")) if item.is_file()})
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
    elif isinstance(artifact, str) and len(artifact) == 64:
        digest = artifact
    else:
        digest = _sha({"candidate_id": candidate_id, "artifact": artifact})
    return {
        "candidate_id": candidate_id,
        "artifact_sha256": digest,
        "frozen": True,
        "frozen_at_unix": time.time(),
    }


def _correct(result: Any) -> bool | None:
    if not isinstance(result, Mapping):
        return None
    if result.get("evaluation_error") or result.get("error"):
        return False
    for key in ("correct", "oracle_passed", "passed", "ok", "success"):
        if key in result:
            return bool(result[key])
    if "exit_code" in result:
        return result["exit_code"] == 0
    return None


def _duration_ms(result: Any) -> float | None:
    if not isinstance(result, Mapping):
        return None
    for key in ("duration_ms", "latency_ms", "elapsed_ms", "runtime_ms", "time_ms"):
        value = result.get(key)
        if type(value) in (int, float) and value > 0:
            return float(value)
    for key, scale in (("duration_sec", 1000.0), ("duration_ns", 1e-6)):
        value = result.get(key)
        if type(value) in (int, float) and value > 0:
            return float(value) * scale
    return None


def _classify(baseline: Any, candidate: Any) -> tuple[str, float | None, str]:
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return "evaluation_error", None, "missing_result"
    if any(
        row.get("evaluation_error")
        or row.get("status") in {"evaluation_error", "infra_failed", "measurement_invalid"}
        for row in (baseline, candidate)
    ):
        return "evaluation_error", None, "evaluation_error"
    b, c = _correct(baseline), _correct(candidate)
    if b is None or c is None:
        return "evaluation_error", None, "missing_correctness"
    if b and c:
        left, right = _duration_ms(baseline), _duration_ms(candidate)
        ratio = left / right if left is not None and right is not None else None
        return "both_pass", ratio, "both implementations passed"
    if b and not c:
        return "opt_regression", None, "candidate failed an input the baseline passed"
    if not b and c:
        return "opt_improvement", None, "candidate corrected a baseline failure"
    return "both_fail", None, "both implementations failed"


def _guard_results(guards, classifications, default_tolerance):
    rows = []
    failures = []
    for item in _seq(guards):
        if isinstance(item, str):
            item = {"input_id": item}
        ident = item.get("input_id") or item.get("workload")
        tolerance = item.get("tolerance", default_tolerance)
        row = classifications.get(ident)
        if row is None:
            failures.append({"input_id": ident, "reason": "guard_missing"})
            rows.append({"input_id": ident, "passed": False, "reason": "guard_missing"})
            continue
        if row["bucket"] not in {"both_pass", "opt_improvement"}:
            failures.append({"input_id": ident, "reason": "guard_correctness_failed"})
            rows.append({"input_id": ident, "passed": False, "reason": "guard_correctness_failed"})
            continue
        ratio = row["speedup_ratio"]
        if ratio is None:
            failures.append({"input_id": ident, "reason": "guard_speedup_missing"})
            rows.append({"input_id": ident, "passed": False, "reason": "guard_speedup_missing"})
        elif ratio < 1.0 - tolerance:
            failure = {"input_id": ident, "reason": "guard_regression", "ratio": ratio,
                       "tolerance": tolerance}
            failures.append(failure)
            rows.append({"input_id": ident, "passed": False, **failure})
        else:
            rows.append({"input_id": ident, "passed": True, "ratio": ratio, "tolerance": tolerance})
    return {"passed": not failures, "rows": rows, "tolerance": default_tolerance}, failures


def _scope_results(out_of_scope, classifications):
    identifiers = list(out_of_scope) if isinstance(out_of_scope, Mapping) else list(_seq(out_of_scope))
    rows, violations = [], []
    for ident in identifiers:
        candidate = classifications.get(ident)
        result = candidate.get("candidate_result") if isinstance(candidate, Mapping) else None
        used_fallback = bool(
            isinstance(result, Mapping)
            and (result.get("fallback") is True or result.get("used_fallback") is True
                 or str(result.get("path")) == "fallback")
        )
        row = {"input_id": ident, "used_fallback": used_fallback,
               "bucket": candidate.get("bucket") if candidate else None}
        if candidate is None or candidate["bucket"] == "evaluation_error":
            violations.append({"input_id": ident, "reason": "missing_result"})
        elif not used_fallback and candidate["bucket"] in {"both_pass", "opt_improvement"}:
            violations.append({"input_id": ident, "reason": "optimized_path_applied_outside_scope"})
        rows.append(row)
    return {"checked": identifiers, "rows": rows, "violations": violations,
            "correct_rate": _ratio(len(rows) - len(violations), len(rows))["value"] if rows else None}


def _ablation_report(ablation, conditions, candidate_id, run_id, minimum, noise):
    def decision(row, label):
        row = _mapping(row)
        return evaluate_exploration_candidate(
            run_id=str(row.get("run_id") or f"{run_id}-{label}"),
            candidate_id=str(row.get("candidate_id") or f"{candidate_id}-{label}"),
            mutation_summary=str(row.get("mutation_summary") or label),
            conditions=dict(row.get("conditions") or conditions),
            compile_ok=bool(row.get("compile_ok", True)),
            oracle_passed=bool(row.get("oracle_passed", row.get("correct", True))),
            measurement_passed=bool(row.get("measurement_passed", True)),
            speedup_ratio=row.get("speedup_ratio"),
            min_speedup_ratio=float(row.get("min_speedup_ratio", minimum)),
            noise_margin=float(row.get("noise_margin", noise)),
            ablation_retained=row.get("ablation_retained", True),
            target_tier="L4",
            failure_reason=str(row.get("failure_reason") or ""),
        )

    full = decision(ablation.get("full") or ablation.get("full_candidate") or {}, "full")
    final = decision(ablation.get("final_combination") or {}, "final")
    final_row = _mapping(ablation.get("final_combination"))
    full_row = _mapping(ablation.get("full") or ablation.get("full_candidate"))
    independent = bool(final_row.get("verified") is True and not final_row.get("reuses_full_candidate"))
    if final_row.get("run_id") and full_row.get("run_id"):
        independent = independent and final_row["run_id"] != full_row["run_id"]
    contributions = []
    raw_loo = ablation.get("leave_one_out")
    loo = (
        _mapping(raw_loo)
        if isinstance(raw_loo, Mapping)
        else {row["factor_id"]: row for row in _seq(raw_loo) if row.get("factor_id")}
    )
    for factor in _seq(ablation.get("factors")) or list(loo):
        row = _mapping(loo.get(factor))
        result = decision(row, f"without-{factor}")
        full_gain = full.get("speedup_ratio")
        without = result.get("speedup_ratio")
        contribution = (full_gain - without) if full_gain is not None and without is not None else None
        if contribution is None:
            label = "unknown"
        elif contribution <= noise:
            label = "within_noise_preserve"
        elif result["verdict"] in {"refuted", "inconclusive", "compile_failed", "oracle_failed"}:
            label = "dependency_or_failure"
        else:
            label = "main_effect"
        contributions.append({"factor_id": str(factor), "contribution": contribution,
                              "classification": label, "decision": result})
    accepted = (
        full["verdict"] in {"attained", "subtarget_valid"}
        and final["verdict"] in {"attained", "subtarget_valid"}
        and independent
    )
    return {
        "full": full,
        "final_combination": final,
        "independent_final_validation": independent,
        "leave_one_out": contributions,
        "factors": [row["factor_id"] for row in contributions],
        "accepted": accepted,
        "verdict": "verified" if accepted else "rejected_or_unverified",
    }


def _primary_outcome(receipt: Mapping) -> str:
    value = receipt.get("primary_outcome") or receipt.get("verification_outcome") or receipt.get("terminal_status")
    verification = _mapping(receipt.get("verification"))
    value = value or verification.get("outcome") or verification.get("status") or receipt.get("outcome") or receipt.get("status")
    text = str(value or "").strip().lower()
    if text in {"completed_pass", "pass", "passed", "success", "succeeded", "completed", "ok"}:
        return "completed_pass"
    if text in {"completed_fail", "fail", "failed", "failure", "regressed"}:
        return "completed_fail"
    if text in {"infra_failed", "measurement_invalid", "infrastructure_failed"}:
        return "infra_failed"
    return "unknown"


def _primary_verified(primary: str) -> bool:
    return primary in {"completed_pass", "completed_fail"}


def _evaluation_outcome(primary: str, explicit: str | None) -> str:
    if primary in {"infra_failed", "unknown"}:
        return "inconclusive"
    if explicit in EVAL_OUTCOMES:
        return explicit
    return {"completed_pass": "supported", "completed_fail": "refuted"}.get(primary, "inconclusive")


def _record_event(store, *, root_hash, ref, event_type, run_id, outcome=None,
                  step_id="", reason="", evidence_source="agent_declared"):
    return record_usage_event(
        store,
        root_hash=root_hash,
        ref=ref,
        event_type=event_type,
        run_id=run_id,
        outcome=outcome,
        step_id=step_id,
        reason=reason,
        evidence_source=evidence_source,
    )


def _find_redactor() -> str:
    cached = sorted((Path.home() / ".cache/mindie-agent/gitleaks").glob("*/gitleaks"))
    found = shutil.which("gitleaks")
    if found:
        return str(Path(found).resolve())
    if cached:
        return str(cached[-1].resolve())
    raise ValueError("an installed gitleaks scanner is required for distillation")


def _repository_units(repository: GradedRepository) -> list:
    return list(repository._load().units.values())


def _rare_recall(repository: GradedRepository, probes: Sequence[Mapping], *, limit: int = 5) -> dict:
    rows, recalled = [], 0
    for probe in probes:
        expected = _unit_id(probe.get("expected_unit_id") or probe.get("unit_id") or "")
        conditions = dict(probe.get("conditions") or {})
        query = str(probe.get("query") or "")
        result = repository.query(
            query, conditions=conditions, limit=limit, token_budget=2048,
            include_unknown=False, navigation_only=True,
        )
        found = {
            str(item["id"])
            for bucket in ("results", "cards", "foundation_cards")
            for item in _seq(result.get(bucket))
        }
        view = _loaded_view(repository, [expected], conditions, token_budget=4096) if expected else {"units": []}
        loaded = _view_ids(view)
        row = {
            "probe_id": probe.get("probe_id") or expected,
            "query": query,
            "conditions": conditions,
            "expected_unit_id": expected,
            "retrieved_refs": sorted(found),
            "loaded_refs": sorted(loaded),
            "recalled": bool(expected and expected in found and expected in loaded),
            "evidence": "repository.query+explain",
        }
        recalled += int(row["recalled"])
        rows.append(row)
    return {"probes": rows, "recall_rate": recalled / len(rows) if rows else None,
            "recalled": recalled, "total": len(rows)}
