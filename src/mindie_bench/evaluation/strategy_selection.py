# [Side-Effect Policy] Writes only explicitly owned trial artifacts; no network or unowned mutation.

"""Conditional strategy selection pilot (Section 4.7 & Section 16).

[Role]
Project isolated task-only (Arm B) and task-plus-evidence (Arm C) selector views
from a shared decision point, judge the real selector decisions, and only with a
caller-supplied continuation executor run A/B/C continuation cells and cost ledgers.

[Invariants]
- Arm B views expose only the task, declared conditions and body-free navigation
  cards; Arm C adds only pre-decision observations. Post-decision explanations,
  leak markers and private judge fields never reach a selector.
- The private judge stays in the evaluator; executors receive a judge-free case.
  Without an executor, cells stay `not_executed`: no task outcome, continuation
  usage, NPU run or device occupancy is reported.
- Selector spend is the selector's own `usage.model_calls`; unknown spend stays `None`
  but reserves `selector_call_budget`. Elapsed time is measured; a late decision is
  kept and flagged over budget. Verified passes stay separate from budget compliance.
- Missing usage stays `None`. NPU card/Die hours (A3: 2 Dies = 1 Card) come only
  from executor-measured occupancy seconds, never from wall clock or run counts.

[Timeouts & Locks]
- `outer_timeout_sec >= business_window_sec + verifier_timeout_sec + cleanup_grace_sec`
  (default `725s >= 600s + 120s + 5s`); `selector_timeout_sec <= 30s`. Callables own I/O.

[Side-Effect Policy]
- Never mutates source corpora or fabricates hardware/model receipts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from mindie_bench.evaluation.knowledge_tiered_support import (
    RoutingIndex,
    _ratio,
    _sha,
    build_strategy_navigation_cards,
    read_units,
    summarize_arm_pareto_and_cop,
)
from mindie_bench.evaluation.strategy_selection_cases import (  # noqa: F401
    ALL_STRATEGY_CASES,
    CANDIDATE_UNIT_IDS,
    DEV_CASES,
    PILOT_CASES,
    PreDecisionObservation,
    StrategyCase,
    _strip_revision,
    claude_cli_strategy_selector,
    evaluate_strategy_continuation,
    evaluate_strategy_selection,
    resolve_strategy_case,
    verify_pilot_lineage_and_soundness,
)
from mindie_bench.runner.compaction_contracts import SemanticEvent
from mindie_bench.runner.mechanism_ablation import (
    analyze_mechanism_ablations,
    build_mechanism_pairs,
    plan_mechanism_ablation,
)
from mindie_bench.runner.mechanism_contracts import (
    AblationObservation,
    AblationOutcome,
    AblationTask,
    ArmConfig,
    ArmRole,
    FixedControls,
    Mechanism,
    PairSpec,
)

SCHEMA_VERSION = "mindie-bench-strategy-selection-pilot/1"
SELECTOR_VIEW_SCHEMA = "mindie-bench-strategy-selector-view/1"
ARMS = ("A", "B", "C")
ARM_LABELS = {"A": "default_path", "B": "task_only_select", "C": "task_plus_evidence_select"}
FORBIDDEN_SELECTOR_KEYS = {
    "post_decision", "post_decision_observations", "perturbation_explanation",
    "private_judge", "reference_solution", "known_bad_solution",
    "expected_strategy", "scoring", "leak_markers",
}

@dataclass(frozen=True)
class StrategySelectionBudget:
    model_call_budget: int = 12
    selector_call_budget: int = 1
    npu_run_budget: int = 2
    selector_timeout_sec: float = 30.0
    business_window_sec: float = 600.0
    verifier_timeout_sec: float = 120.0
    cleanup_grace_sec: float = 5.0
    outer_timeout_sec: float = 725.0
    knowledge_token_budget: int = 4096

    def __post_init__(self):
        if type(self.model_call_budget) is not int or self.model_call_budget < 2:
            raise ValueError("model_call_budget must be an int >= 2")
        if type(self.selector_call_budget) is not int or not (1 <= self.selector_call_budget < self.model_call_budget):
            raise ValueError("selector_call_budget must fit strictly inside model_call_budget")
        if type(self.npu_run_budget) is not int or self.npu_run_budget < 1:
            raise ValueError("npu_run_budget must be a positive int")
        if not (0 < self.selector_timeout_sec <= 30.0):
            raise ValueError("selector_timeout_sec must be in (0, 30]")
        if self.business_window_sec < self.selector_timeout_sec:
            raise ValueError("business_window_sec must be >= selector_timeout_sec")
        if self.cleanup_grace_sec < 5.0:
            raise ValueError("cleanup_grace_sec must be >= 5.0s")
        min_outer = self.business_window_sec + self.verifier_timeout_sec + self.cleanup_grace_sec
        if self.outer_timeout_sec < min_outer:
            raise ValueError(
                f"outer_timeout_sec ({self.outer_timeout_sec}) must be >= "
                f"business_window_sec + verifier_timeout_sec + cleanup_grace_sec ({min_outer})"
            )

def load_combined_routing_index(corpus_roots: Sequence[str | Path]) -> RoutingIndex:
    """Build a unified RoutingIndex across domain corpora (`vllm-ascend` + `operators`)."""
    units = []
    for root in corpus_roots:
        units.extend(read_units(Path(root)))
    return RoutingIndex(units)

# -------------------------------------- Input Projection & Strict Isolation
def _walk_mapping_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_mapping_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_mapping_keys(item)

def verify_selector_isolation(
    view: Mapping[str, Any],
    case: StrategyCase,
    *,
    arm: Literal["B", "C"],
) -> dict[str, Any]:
    """Verify Arm B (task-only) and Arm C (task+evidence) context isolation."""
    if arm not in ("B", "C"):
        raise ValueError(f"selector isolation applies to arms B and C, got {arm!r}")
    keys = list(_walk_mapping_keys(view))
    forbidden_keys = sorted({key for key in keys if key.casefold() in FORBIDDEN_SELECTOR_KEYS})
    serialized = json.dumps(view, sort_keys=True, ensure_ascii=False, default=str)
    leak_markers = [str(marker) for marker in case.private_judge.get("leak_markers", ())]
    post_explanation = str(case.private_judge.get("post_decision_explanation", ""))
    leaked_markers = [marker for marker in leak_markers if marker and marker in serialized]
    if post_explanation and post_explanation in serialized:
        leaked_markers.append("post_decision_explanation")

    evidence_leaks: list[str] = []
    if arm == "B":
        if "pre_decision_observations" in view or "observed_conditions" in view:
            evidence_leaks.append("arm B view contains pre-decision observation fields")
        for obs in case.pre_decision_observations:
            if obs.statement in serialized or obs.anchor in serialized:
                evidence_leaks.append(f"arm B leaked observation {obs.observation_id}")
    else:
        obs_rows = view.get("pre_decision_observations")
        if not isinstance(obs_rows, Sequence) or not obs_rows:
            evidence_leaks.append("arm C view is missing pre_decision_observations")

    cards = view.get("candidate_navigation_cards") or ()
    body_leaks = [
        str(card.get("id")) for card in cards
        if isinstance(card, Mapping) and ("body" in card or "content" in card or "excerpt" in card)
    ]
    issues = [
        *(f"forbidden key: {key}" for key in forbidden_keys),
        *(f"leak marker: {item}" for item in leaked_markers),
        *evidence_leaks,
        *(f"navigation card leaked inline body: {ident}" for ident in body_leaks),
    ]
    return {"isolated": not issues, "arm": arm, "issues": issues, "checked_leak_markers": leak_markers}

def prepare_strategy_selector_view(
    case: str | StrategyCase,
    *,
    arm: Literal["B", "C"],
    routing_index: RoutingIndex,
) -> dict[str, Any]:
    """Construct the isolated tool-less selector input for Arm B or Arm C."""
    selected = resolve_strategy_case(case)
    if arm not in ("B", "C"):
        raise ValueError(f"selector view requires arm 'B' or 'C', got {arm!r}")
    visible_conditions = (
        dict(selected.declared_conditions)
        if arm == "B"
        else {**dict(selected.declared_conditions), **dict(selected.observed_conditions)}
    )
    if build_strategy_navigation_cards is None:
        raise RuntimeError(
            "installed mindie_knowledge lacks build_strategy_navigation_cards; "
            "install the pinned knowledge extra"
        )
    cards = build_strategy_navigation_cards(
        routing_index, selected.candidate_ids, conditions=visible_conditions,
    )
    view: dict[str, Any] = {
        "schema_version": SELECTOR_VIEW_SCHEMA, "case_id": selected.case_id,
        "decision_point_id": selected.decision_point_id, "selector_arm": arm,
        "tool_access": "none", "max_model_calls": 1,
        "problem_statement": selected.problem_statement,
        "declared_conditions": dict(selected.declared_conditions),
        "candidate_navigation_cards": cards,
        "default_option": {
            "id": "default",
            "description": (
                "Retain the default diagnostic path when candidate conditions are "
                "mismatched, when uninstrumented baseline/missing facts must be probed "
                "first, or when no specialized workflow matches."
            ),
        },
        "selection_instructions": (
            "Select a candidate unit ID only when visible facts match its conditions and "
            "observable triggers; if required conditions are missing (`condition_unknown`), "
            "or if execution evidence contradicts the candidate's premise (`mismatch`), "
            "select `default` and name the minimal discriminating probe."
        ),
        "output_schema": {
            "selected_strategy_ref": "candidate unit ID or 'default'",
            "rationale": "concise evidence-grounded reason",
            "applicability_basis": "matched | condition_unknown | mismatch",
            "missing_conditions_to_probe": ["optional condition keys"],
        },
    }
    if arm == "C":
        view["observed_conditions"] = dict(selected.observed_conditions)
        view["pre_decision_observations"] = [
            {"observation_id": obs.observation_id, "statement": obs.statement, "provenance": obs.anchor}
            for obs in selected.pre_decision_observations
        ]
    isolation = verify_selector_isolation(view, selected, arm=arm)
    if not isolation["isolated"]:
        raise ValueError(
            f"strategy selector view for arm {arm} failed isolation: "
            + "; ".join(isolation["issues"])
        )
    return view

# ------------------------------------ Reference Selector & Executor Boundary
def condition_aware_selector(view: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic rule selector on the visible view; it makes no model call."""
    return {**_condition_aware_decision(view), "usage": {"model_calls": 0}}


def _condition_aware_decision(view: Mapping[str, Any]) -> dict[str, Any]:
    cards = list(view.get("candidate_navigation_cards") or ())
    arm = view.get("selector_arm")
    problem_lower = str(view.get("problem_statement", "")).casefold()
    obs_text = " ".join(
        str(item.get("statement", "")) for item in (view.get("pre_decision_observations") or ())
    ).casefold()

    if arm == "C":
        if "enforce-eager" in obs_text and ("reproduces the exact same" in obs_text or "eager mode shows" in obs_text):
            return {
                "selected_strategy_ref": "default",
                "rationale": "Eager mode already reproduces the failure; graph capture is not the root cause.",
                "applicability_basis": "mismatch", "missing_conditions_to_probe": [],
            }
        if "507035" in obs_text and "uninstrumented baseline" in obs_text:
            return {
                "selected_strategy_ref": "default",
                "rationale": "Uninstrumented baseline crashes with 507035 and AlltoAll dominates; fix correctness before fusion.",
                "applicability_basis": "mismatch", "missing_conditions_to_probe": [],
            }

    matched_cards = [card for card in cards if card.get("match_status") == "matched"]
    if matched_cards:
        chosen = matched_cards[0]
        return {
            "selected_strategy_ref": chosen["ref"],
            "rationale": f"All declared/observed conditions matched for {chosen['id']}.",
            "applicability_basis": "matched", "missing_conditions_to_probe": [],
        }

    for card in cards:
        keywords = [str(k).casefold() for k in card.get("triggers", {}).get("keywords", ())]
        if any(k in problem_lower for k in keywords) and card.get("match_status") != "mismatch":
            return {
                "selected_strategy_ref": card["ref"],
                "rationale": f"Surface task keywords matched {card['id']} with missing conditions {card.get('missing_conditions')}.",
                "applicability_basis": card.get("match_status", "condition_unknown"),
                "missing_conditions_to_probe": list(card.get("missing_conditions") or ()),
            }

    return {
        "selected_strategy_ref": "default", "rationale": "No candidate matched visible conditions.",
        "applicability_basis": "mismatch", "missing_conditions_to_probe": [],
    }


def _executor_case(case: StrategyCase) -> StrategyCase:
    """Continuations see the task and decision point, never the judge or its answers."""
    return replace(case, private_judge={}, reference_solution={}, known_bad_solution={})


def _total(rows: Sequence[Mapping[str, Any]], key: str, digits: int | None = None):
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    total = sum(values)
    return round(total, digits) if digits is not None else total


# ------------------------------------ Balanced Triplet Planning & Execution
@dataclass(frozen=True)
class StrategyCellPlan:
    case_id: str
    family_id: str
    case_kind: str
    hardware_tier: str
    topology_tier: str
    repetition: int
    arm: Literal["A", "B", "C"]
    triplet_order: int
    global_order: int
    controls_digest: str


def plan_strategy_selection_cells(
    cases: Sequence[StrategyCase] = tuple(PILOT_CASES.values()),
    *,
    repetitions: int = 2,
    budget: StrategySelectionBudget = StrategySelectionBudget(),
    model_id: str = "mindie-strategy-pilot-model",
    raw_knowledge_digest: str = "raw-corpus-digest",
) -> tuple[tuple[StrategyCellPlan, ...], dict[str, FixedControls]]:
    """Plan balanced A/B/C triplets across families and reverse order on even repetitions."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not cases:
        raise ValueError("at least one strategy case is required")

    base_orders: tuple[tuple[Literal["A", "B", "C"], ...], ...] = (
        ("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B"),
    )
    controls_by_task: dict[str, FixedControls] = {}
    for case in cases:
        evidence_digest = _sha([obs.__dict__ for obs in case.pre_decision_observations])
        controls_by_task[case.case_id] = FixedControls(
            model_id=model_id,
            inference_profile="temperature=0,reasoning_effort=high",
            tool_digest="native-harness-tools/1",
            raw_knowledge_digest=raw_knowledge_digest,
            evidence_digest=evidence_digest,
            task_start_digest=_sha({
                "case_id": case.case_id, "decision_point_id": case.decision_point_id,
                "declared_conditions": dict(case.declared_conditions),
            }),
            semantic_event=SemanticEvent.NEGATIVE_RESULT,
            model_call_budget=budget.model_call_budget,
            npu_run_budget=budget.npu_run_budget,
            max_wall_clock_sec=budget.outer_timeout_sec,
            knowledge_token_budget=budget.knowledge_token_budget,
            child_model_call_budget=budget.selector_call_budget,
        )

    cells: list[StrategyCellPlan] = []
    global_order = 0
    for repetition in range(1, repetitions + 1):
        for idx, case in enumerate(cases):
            order_tuple = base_orders[idx % len(base_orders)]
            if repetition % 2 == 0:
                order_tuple = tuple(reversed(order_tuple))
            c_digest = controls_by_task[case.case_id].digest()
            for triplet_idx, arm in enumerate(order_tuple):
                cells.append(StrategyCellPlan(
                    case_id=case.case_id, family_id=case.family_id, case_kind=case.case_kind,
                    hardware_tier=case.hardware_tier, topology_tier=case.topology_tier,
                    repetition=repetition, arm=arm, triplet_order=triplet_idx,
                    global_order=global_order, controls_digest=c_digest,
                ))
                global_order += 1
    return tuple(cells), controls_by_task


def _reported(usage: Mapping[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return None if value is None else int(value)


def run_strategy_selection_cell(
    cell: StrategyCellPlan,
    case: StrategyCase,
    *,
    routing_index: RoutingIndex,
    budget: StrategySelectionBudget = StrategySelectionBudget(),
    selector_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = condition_aware_selector,
    executor_fn: Callable[..., Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one (case, repetition, arm) cell with strict budget & error-visibility semantics."""
    selector_status: Literal["not_applicable", "completed", "failed", "unknown"] = "not_applicable"
    selector_error: str | None = None
    selector_decision: Mapping[str, Any] | None = None
    # Arm A runs no selector, so its selector spend is an observed zero.
    selector_model_calls: int | None = 0 if cell.arm == "A" else None
    selector_prompt_tokens: int | None = 0 if cell.arm == "A" else None
    selector_completion_tokens: int | None = 0 if cell.arm == "A" else None
    selector_wall_sec, selector_over_budget = 0.0, None

    if cell.arm in ("B", "C"):
        view = prepare_strategy_selector_view(case, arm=cell.arm, routing_index=routing_index)
        started = clock()
        try:
            raw_sel = dict(selector_fn(view))
        except TimeoutError as exc:
            selector_status, selector_error = "unknown", str(exc)
        except Exception as exc:  # noqa: BLE001 — preserve error and fall back to default continuation
            selector_status, selector_error = "failed", str(exc)
        else:
            # A late decision is still the observed decision; lateness is a budget fact.
            usage = raw_sel.get("usage") if isinstance(raw_sel.get("usage"), Mapping) else {}
            selector_model_calls = _reported(usage, "model_calls")
            selector_prompt_tokens = _reported(usage, "prompt_tokens")
            selector_completion_tokens = _reported(usage, "completion_tokens")
            if selector_model_calls == 0:
                # Tokens are spent per model call; a selector that made none spent none.
                selector_prompt_tokens = selector_prompt_tokens or 0
                selector_completion_tokens = selector_completion_tokens or 0
            selector_decision, selector_status = raw_sel, "completed"
        selector_wall_sec = round(clock() - started, 4)
        selector_over_budget = selector_wall_sec > budget.selector_timeout_sec

    selector_usage = selector_prompt_tokens is not None and selector_completion_tokens is not None
    row = {
        "cell": cell.__dict__, "controls_digest": cell.controls_digest,
        "selector_status": selector_status, "selector_error": selector_error,
        "selector_decision": dict(selector_decision) if selector_decision is not None else None,
        "selection": (
            evaluate_strategy_selection(case, selector_decision.get("selected_strategy_ref"))
            if selector_decision is not None else None
        ),
        "selector_model_calls": selector_model_calls,
        "selector_prompt_tokens": selector_prompt_tokens,
        "selector_completion_tokens": selector_completion_tokens,
        "selector_wall_clock_sec": selector_wall_sec,
        "selector_budget_exceeded": selector_over_budget,
        "npu_cards": case.npu_cards, "npu_dies": case.npu_dies,
    }
    unexecuted = {
        "judgment": None, "budget_compliant": False if selector_over_budget else None,
        "usage_status": "unavailable", "executor_model_calls": None, "total_model_calls": None,
        "executor_prompt_tokens": None, "executor_completion_tokens": None,
        "prompt_tokens": None, "completion_tokens": None, "knowledge_tokens": None,
        "wall_clock_sec": selector_wall_sec, "npu_runs": None,
        "npu_occupancy_sec": None, "npu_card_hours": None, "npu_die_hours": None,
        "uninformative_probes": None, "human_interventions": None,
    }
    if executor_fn is None:
        # No continuation ran, so the selector stage is the whole cell.
        return {
            **row, **unexecuted, "terminal_status": "not_executed",
            "usage_status": "native" if selector_usage else "unavailable",
            "total_model_calls": selector_model_calls,
            "prompt_tokens": selector_prompt_tokens, "completion_tokens": selector_completion_tokens,
        }

    # Unknown selector spend keeps its declared reservation in the shared budget.
    reserved = budget.selector_call_budget if selector_model_calls is None else selector_model_calls
    started = clock()
    try:
        exec_out = dict(
            executor_fn(
                case=_executor_case(case),
                arm=cell.arm,
                selector_decision=selector_decision,
                routing_index=routing_index,
                remaining_model_calls=budget.model_call_budget - reserved,
                remaining_window_sec=budget.business_window_sec - selector_wall_sec,
            )
        )
    except TimeoutError as exc:
        # The dispatched continuation may have spent calls; its totals stay unknown.
        return {
            **row, **unexecuted, "terminal_status": "unknown", "executor_error": str(exc),
            "wall_clock_sec": round(selector_wall_sec + clock() - started, 4),
        }
    exec_wall = clock() - started

    judged = evaluate_strategy_continuation(case, exec_out)
    exec_calls = exec_out.get("model_calls")
    total_calls = (
        None if exec_calls is None or selector_model_calls is None
        else selector_model_calls + int(exec_calls)
    )
    total_wall = round(selector_wall_sec + exec_wall, 4)
    npu_runs = exec_out.get("npu_runs")
    exec_prompt = exec_out.get("prompt_tokens")
    exec_completion = exec_out.get("completion_tokens")
    native = selector_usage and exec_prompt is not None and exec_completion is not None
    occupancy = exec_out.get("npu_occupancy_sec")
    checks = [
        selector_over_budget is not True,
        total_wall <= budget.business_window_sec,
        None if total_calls is None else total_calls <= budget.model_call_budget,
        None if npu_runs is None else npu_runs <= budget.npu_run_budget,
    ]
    return {
        **row,
        "terminal_status": "pass" if judged["task_succeeded"] else "fail",
        "judgment": judged,
        "budget_compliant": False if False in checks else None if None in checks else True,
        "usage_status": "native" if native else "unavailable",
        "executor_model_calls": exec_calls, "total_model_calls": total_calls,
        "executor_prompt_tokens": exec_prompt, "executor_completion_tokens": exec_completion,
        "prompt_tokens": int(exec_prompt) + int(selector_prompt_tokens) if native else None,
        "completion_tokens": int(exec_completion) + int(selector_completion_tokens) if native else None,
        "knowledge_tokens": exec_out.get("knowledge_tokens"),
        "wall_clock_sec": total_wall, "npu_runs": npu_runs, "npu_occupancy_sec": occupancy,
        "npu_card_hours": None if occupancy is None else round(occupancy / 3600.0 * case.npu_cards, 6),
        "npu_die_hours": None if occupancy is None else round(occupancy / 3600.0 * case.npu_dies, 6),
        "uninformative_probes": exec_out.get("uninformative_probes"),
        "human_interventions": exec_out.get("human_interventions"),
    }


def _selection_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ("B", "C"):
        rows = [row for row in results if row["cell"]["arm"] == arm]
        judged = [row["selection"] for row in rows if row["selection"] is not None]
        missing = len(rows) - len(judged)
        arms[arm] = {
            "planned_cells": len(rows), "judged_cells": len(judged),
            "selector_budget_exceeded_cells": sum(row["selector_budget_exceeded"] is True for row in rows),
            "selection_acceptable_rate": _ratio(
                sum(item["selection_acceptable"] for item in judged), len(judged), missing=missing
            ),
            "false_adoption_rate": _ratio(
                sum(item["false_adoption"] for item in judged), len(judged), missing=missing
            ),
        }

    def delta(key: str) -> float | None:
        b, c = arms["B"][key]["value"], arms["C"][key]["value"]
        return None if b is None or c is None else round(c - b, 6)

    return {
        "arms": arms,
        "C_vs_B": {
            "selection_acceptable_delta": delta("selection_acceptable_rate"),
            "false_adoption_delta": delta("false_adoption_rate"),
        },
    }


def run_strategy_selection_pilot(
    *,
    corpus_roots: Sequence[str | Path],
    cases: Sequence[StrategyCase] = tuple(PILOT_CASES.values()),
    repetitions: int = 2,
    budget: StrategySelectionBudget = StrategySelectionBudget(),
    selector_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = condition_aware_selector,
    executor_fn: Callable[..., Mapping[str, Any]] | None = None,
    offline_prep_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge B/C selector decisions; with a real executor, also compare A/B/C continuations."""
    routing_index = load_combined_routing_index(corpus_roots)
    readiness = verify_pilot_lineage_and_soundness(cases)
    cells, controls_by_task = plan_strategy_selection_cells(
        cases,
        repetitions=repetitions,
        budget=budget,
        raw_knowledge_digest=routing_index.generation,
    )
    case_by_id = {case.case_id: case for case in cases}
    results = [
        run_strategy_selection_cell(
            cell,
            case_by_id[cell.case_id],
            routing_index=routing_index,
            budget=budget,
            selector_fn=selector_fn,
            executor_fn=executor_fn,
        )
        for cell in cells
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "readiness": readiness, "budget": budget.__dict__,
        "upper_bounds": {
            "total_cells": len(cells), "selector_calls": sum(1 for c in cells if c.arm in ("B", "C")),
            "max_model_calls": len(cells) * budget.model_call_budget,
            "max_npu_runs": len(cells) * budget.npu_run_budget,
            "max_outer_wall_clock_sec": len(cells) * budget.outer_timeout_sec,
        },
        "continuation": "not_executed" if executor_fn is None else "executed",
        "selection": _selection_summary(results),
    }
    if executor_fn is None:
        return {
            **report,
            "scope_note": (
                "Selector stage only: selector decisions on isolated B/C views judged by the private "
                "judge. No continuation executor was supplied, so no task outcome, continuation "
                "usage, NPU run or device occupancy is reported."
            ),
            "decision_gate": (
                "continuation_not_executed" if readiness["ready"] else "return_to_implementation_fix"
            ),
            "cells": results,
        }

    base_cfg = ArmConfig(
        roles="single_agent", ledger="two_layer", retrieval="graded_conditions",
        exploration="sequential", strategy_selection="default",
    )
    pairs = build_mechanism_pairs([
        PairSpec(
            pair_id="pair-strategy-c-vs-b",
            mechanism=Mechanism.STRATEGY_SELECTION,
            control=ArmConfig(
                roles=base_cfg.roles, ledger=base_cfg.ledger, retrieval=base_cfg.retrieval,
                exploration=base_cfg.exploration, strategy_selection="task_only",
            ),
            treatment=ArmConfig(
                roles=base_cfg.roles, ledger=base_cfg.ledger, retrieval=base_cfg.retrieval,
                exploration=base_cfg.exploration, strategy_selection="task_plus_evidence",
            ),
            control_id="B_task_only",
            treatment_id="C_task_plus_evidence",
        ),
    ], require_all=False)
    ablation_tasks = [
        AblationTask(task_id=c.case_id, task_family=c.family_id, hardware_tier=c.hardware_tier)
        for c in cases
    ]
    c_vs_b_plan = plan_mechanism_ablation(
        pairs, ablation_tasks, controls_by_task=controls_by_task,
        repetitions=repetitions, require_all_mechanisms=False,
    )
    by_case_rep_arm = {
        (row["cell"]["case_id"], row["cell"]["repetition"], row["cell"]["arm"]): row
        for row in results
    }
    c_vs_b_obs = []
    for abl_cell in c_vs_b_plan.cells:
        arm_key = "B" if abl_cell.arm_role == ArmRole.CONTROL else "C"
        row = by_case_rep_arm[(abl_cell.task_id, abl_cell.repetition, arm_key)]
        c_vs_b_obs.append(AblationObservation(
            pair_id=abl_cell.pair_id, arm_role=abl_cell.arm_role, task_id=abl_cell.task_id,
            repetition=abl_cell.repetition, status=AblationOutcome(row["terminal_status"]),
            completion_tokens=row["completion_tokens"], wall_clock_sec=row["wall_clock_sec"],
            npu_runs=row["npu_runs"], knowledge_tokens=row["knowledge_tokens"],
            first_effective_action=(row["judgment"] or {}).get("next_action"),
            controls_digest=row["controls_digest"],
        ))
    c_vs_b_report = analyze_mechanism_ablations(c_vs_b_plan, c_vs_b_obs, min_independent_families=3)

    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [row for row in results if row["cell"]["arm"] == arm]
        valid_rows = [row for row in arm_rows if row["terminal_status"] in {"pass", "fail"}]
        passed = sum(row["terminal_status"] == "pass" for row in valid_rows)
        reasonable = sum(row["judgment"]["selection_reasonable"] is True for row in valid_rows)
        false_adoptions = sum(row["judgment"]["false_adoption"] is True for row in valid_rows)
        trigger_rows = [row for row in valid_rows if row["cell"]["case_kind"] == "trigger_matched"]
        boundary_rows = [row for row in valid_rows if row["cell"]["case_kind"] == "boundary_or_mismatch"]
        arm_summaries[arm] = {
            "arm": arm, "label": ARM_LABELS[arm],
            "planned_cells": len(arm_rows), "valid_cells": len(valid_rows),
            "budget_noncompliant_cells": sum(r["budget_compliant"] is False for r in arm_rows),
            "selector_budget_exceeded_cells": sum(r["selector_budget_exceeded"] is True for r in arm_rows),
            "pass_rate": _ratio(passed, len(valid_rows), missing=len(arm_rows) - len(valid_rows)),
            "selection_reasonable_rate": _ratio(reasonable, len(valid_rows)),
            "false_adoption_rate": _ratio(false_adoptions, len(valid_rows)),
            "trigger_matched_pass_rate": _ratio(sum(r["terminal_status"] == "pass" for r in trigger_rows), len(trigger_rows)),
            "boundary_safe_handling_rate": _ratio(sum(not r["judgment"]["false_adoption"] for r in boundary_rows), len(boundary_rows)),
            "total_model_calls": _total(arm_rows, "total_model_calls"),
            "selector_model_calls": _total(arm_rows, "selector_model_calls"),
            "total_prompt_tokens": _total(arm_rows, "prompt_tokens"),
            "total_completion_tokens": _total(arm_rows, "completion_tokens"),
            "total_wall_clock_sec": _total(arm_rows, "wall_clock_sec", 3),
            "total_npu_runs": _total(arm_rows, "npu_runs"),
            "total_npu_card_hours": _total(arm_rows, "npu_card_hours", 6),
            "total_npu_die_hours": _total(arm_rows, "npu_die_hours", 6),
        }

    negative_transfer_c_vs_a = 0
    cost_regression_on_both_pass_c_vs_a = 0
    c_beats_b_selection = 0
    c_beats_a_pass = 0
    for case in cases:
        for rep in range(1, repetitions + 1):
            row_a = by_case_rep_arm[(case.case_id, rep, "A")]
            row_b = by_case_rep_arm[(case.case_id, rep, "B")]
            row_c = by_case_rep_arm[(case.case_id, rep, "C")]
            if row_a["terminal_status"] == "pass" and row_c["terminal_status"] == "fail":
                negative_transfer_c_vs_a += 1
            if (
                row_a["terminal_status"] == "pass"
                and row_c["terminal_status"] == "pass"
                and row_c["wall_clock_sec"] > row_a["wall_clock_sec"]
            ):
                cost_regression_on_both_pass_c_vs_a += 1
            if (
                (row_c["judgment"] or {}).get("selection_reasonable") is True
                and (row_b["judgment"] or {}).get("selection_reasonable") is False
            ):
                c_beats_b_selection += 1
            if row_c["terminal_status"] == "pass" and row_a["terminal_status"] == "fail":
                c_beats_a_pass += 1

    shared_prefix_wall = sum(float(case.shared_prefix_cost.get("wall_clock_sec", 0.0)) for case in cases)
    shared_prefix_prompt = sum(int(case.shared_prefix_cost.get("prompt_tokens", 0)) for case in cases)
    shared_prefix_completion = sum(int(case.shared_prefix_cost.get("completion_tokens", 0)) for case in cases)

    c_reasonable = arm_summaries["C"]["selection_reasonable_rate"]["value"] or 0.0
    b_reasonable = arm_summaries["B"]["selection_reasonable_rate"]["value"] or 0.0
    c_pass = arm_summaries["C"]["pass_rate"]["value"] or 0.0
    a_pass = arm_summaries["A"]["pass_rate"]["value"] or 0.0

    pareto_and_cop = summarize_arm_pareto_and_cop(arm_summaries, baseline_arm="A")
    if not readiness["ready"]:
        decision_gate = "return_to_implementation_fix"
    elif c_reasonable > b_reasonable and c_pass >= a_pass and negative_transfer_c_vs_a == 0:
        decision_gate = "continue_expanding_families"
    else:
        decision_gate = "stop_default_rollout"

    walls = [arm_summaries[arm]["total_wall_clock_sec"] for arm in ARMS]
    return {
        **report,
        "scope_note": (
            f"Continuation pilot from a shared decision point across {len(cases)} cases "
            f"({len(cells)} executed continuations + {report['upper_bounds']['selector_calls']} "
            "isolated selector calls). Does not claim full-session product cohort superiority "
            "without end-to-end prefix replay."
        ),
        "arms": arm_summaries,
        "comparisons": {
            "C_vs_B": {
                "selection_reasonable_delta": round(c_reasonable - b_reasonable, 6),
                "false_adoption_delta": round(
                    (arm_summaries["C"]["false_adoption_rate"]["value"] or 0.0)
                    - (arm_summaries["B"]["false_adoption_rate"]["value"] or 0.0), 6,
                ),
                "pass_rate_delta": round(c_pass - (arm_summaries["B"]["pass_rate"]["value"] or 0.0), 6),
                "c_beats_b_selection_cells": c_beats_b_selection,
                "paired_ablation": c_vs_b_report.to_dict(),
            },
            "C_vs_A": {
                "pass_rate_delta": round(c_pass - a_pass, 6),
                "c_beats_a_pass_cells": c_beats_a_pass,
                "negative_transfer_count": negative_transfer_c_vs_a,
                "cost_regression_on_both_pass_count": cost_regression_on_both_pass_c_vs_a,
                "continuation_wall_clock_delta_sec": (
                    None if None in (walls[0], walls[2]) else round(walls[2] - walls[0], 3)
                ),
            },
            "trigger_vs_boundary": {
                arm: {
                    "trigger_matched_pass_rate": arm_summaries[arm]["trigger_matched_pass_rate"],
                    "boundary_safe_handling_rate": arm_summaries[arm]["boundary_safe_handling_rate"],
                }
                for arm in ARMS
            },
            "pareto_and_cost_of_pass": pareto_and_cop,
        },
        "cost_ledgers": {
            "continuation_cost_by_arm": {
                arm: {
                    "wall_clock_sec": arm_summaries[arm]["total_wall_clock_sec"],
                    "model_calls": arm_summaries[arm]["total_model_calls"],
                    "selector_calls": arm_summaries[arm]["selector_model_calls"],
                    "prompt_tokens": arm_summaries[arm]["total_prompt_tokens"],
                    "completion_tokens": arm_summaries[arm]["total_completion_tokens"],
                    "npu_runs": arm_summaries[arm]["total_npu_runs"],
                    "npu_card_hours": arm_summaries[arm]["total_npu_card_hours"],
                    "npu_die_hours": arm_summaries[arm]["total_npu_die_hours"],
                    "cost_of_pass": pareto_and_cop["cost_of_pass_by_arm"][arm],
                }
                for arm in ARMS
            },
            "shared_prefix_cost_once": {
                "wall_clock_sec": round(shared_prefix_wall, 3), "prompt_tokens": shared_prefix_prompt,
                "completion_tokens": shared_prefix_completion,
                "source": "declared by the case definitions; not measured by this pilot",
                "accounting_rule": "counted once in physical total; added equally when estimating full-task conditional cost",
            },
            "offline_and_total_experiment_cost": {
                "offline_prep_cost": dict(offline_prep_cost) if offline_prep_cost is not None else None,
                "physical_continuation_wall_clock_sec": (
                    None if None in walls else round(sum(walls), 3)
                ),
                "amortization_cases": len(cases),
            },
        },
        "decision_gate": decision_gate, "cells": results,
    }
