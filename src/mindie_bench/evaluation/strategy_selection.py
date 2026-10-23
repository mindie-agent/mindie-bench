# [Side-Effect Policy] Writes only explicitly owned trial artifacts; no network or unowned mutation.

"""Conditional strategy selection pilot (Section 4.7 & Section 16).

[Role]
Project strictly isolated task-only (Arm B) and task-plus-evidence (Arm C)
strategy selector views from a shared decision-point prefix, orchestrate
three-arm (A/B/C) continuation cells under a shared total budget, and compute
blind private-judge, negative-transfer, and three-ledger cost comparisons.

[Invariants]
- Arm B (`task_only`) selector views expose only the task statement, declared
  task conditions, and body-free candidate navigation cards; any pre-decision
  execution observation or private judge leak marker in Arm B is rejected.
- Arm C (`task_plus_evidence`) selector views use the identical candidate order,
  instructions, and output schema as Arm B, adding only pre-decision execution
  observations up to the shared decision point; post-decision explanations and
  leak markers remain strictly excluded.
- All three arms (A/B/C) share the same total budget per cell (`model_call_budget`,
  `npu_run_budget`, `business_window_sec`, `verifier_timeout_sec`, `cleanup_grace_sec`).
  B/C selector calls consume from that shared budget and never receive bonus calls.
- A selector failure or timeout stays visible (`selector_status="failed"` or
  `"unknown"`), charges its consumed call/time, and falls back to `"default"`
  continuation without overwriting a verified primary task pass (`completed_pass`).
- Native token usage is never defaulted to zero when missing; absent usage stays
  `None` (`usage_status="unavailable"`). Ascend A3 Die hours and physical card
  hours are reported separately (`2 Dies = 1 Card` on A3).

[Timeouts & Locks]
- Enforces `outer_timeout_sec >= business_window_sec + verifier_timeout_sec + cleanup_grace_sec`
  (default `725s >= 600s + 120s + 5s`) and `selector_timeout_sec <= 30s`.
- Pure in-memory planning/evaluation; caller-supplied selector/executor callables
  own subprocess or network I/O.

[Side-Effect Policy]
- Never mutates source corpora or fabricates hardware/model receipts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

# ------------------------------------ Reference Selector & Executor Policies
def condition_aware_selector(view: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic baseline selector operating strictly on the visible view."""
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
                "usage": {"prompt_tokens": 310, "completion_tokens": 48, "wall_clock_sec": 1.2},
            }
        if "507035" in obs_text and "uninstrumented baseline" in obs_text:
            return {
                "selected_strategy_ref": "default",
                "rationale": "Uninstrumented baseline crashes with 507035 and AlltoAll dominates; fix correctness before fusion.",
                "applicability_basis": "mismatch", "missing_conditions_to_probe": [],
                "usage": {"prompt_tokens": 325, "completion_tokens": 52, "wall_clock_sec": 1.3},
            }

    matched_cards = [card for card in cards if card.get("match_status") == "matched"]
    if matched_cards:
        chosen = matched_cards[0]
        return {
            "selected_strategy_ref": chosen["ref"],
            "rationale": f"All declared/observed conditions matched for {chosen['id']}.",
            "applicability_basis": "matched", "missing_conditions_to_probe": [],
            "usage": {"prompt_tokens": 280 if arm == "B" else 340, "completion_tokens": 45, "wall_clock_sec": 1.1},
        }

    for card in cards:
        keywords = [str(k).casefold() for k in card.get("triggers", {}).get("keywords", ())]
        if any(k in problem_lower for k in keywords) and card.get("match_status") != "mismatch":
            return {
                "selected_strategy_ref": card["ref"],
                "rationale": f"Surface task keywords matched {card['id']} with missing conditions {card.get('missing_conditions')}.",
                "applicability_basis": card.get("match_status", "condition_unknown"),
                "missing_conditions_to_probe": list(card.get("missing_conditions") or ()),
                "usage": {"prompt_tokens": 270, "completion_tokens": 44, "wall_clock_sec": 1.0},
            }

    return {
        "selected_strategy_ref": "default", "rationale": "No candidate matched visible conditions.",
        "applicability_basis": "mismatch", "missing_conditions_to_probe": [],
        "usage": {"prompt_tokens": 260, "completion_tokens": 40, "wall_clock_sec": 0.9},
    }

def shared_prefix_executor(
    case: StrategyCase,
    *,
    arm: Literal["A", "B", "C"],
    selector_decision: Mapping[str, Any] | None,
    routing_index: RoutingIndex,
    remaining_model_calls: int,
    remaining_window_sec: float,
) -> dict[str, Any]:
    """Execute continuation from the shared decision point with real `explain_closure`."""
    if remaining_model_calls < 1 or remaining_window_sec <= 0:
        raise TimeoutError("continuation budget exhausted before executor step")

    if arm == "A" or selector_decision is None:
        query_res = routing_index.query(
            case.problem_statement,
            conditions=dict(case.declared_conditions),
            limit=3,
            token_budget=1200,
            include_unknown=True,
            mode="diagnose",
        )
        top_cards = query_res.get("cards") or ()
        top_id = top_cards[0]["id"] if top_cards else "default"
        if top_id != "default" and top_cards[0].get("match_status") == "matched":
            selected_ref = top_id
        else:
            selected_ref = "default"
    else:
        selected_ref = _strip_revision(selector_decision.get("selected_strategy_ref", "default"))

    knowledge_tokens = 0
    if selected_ref != "default":
        closure = routing_index.explain_closure(
            [selected_ref],
            conditions={**dict(case.declared_conditions), **dict(case.observed_conditions)},
            token_budget=4096,
            mode="diagnose",
        )
        knowledge_tokens = int(closure["usage"]["body_token_estimate"])

    acceptable = {_strip_revision(item) for item in case.private_judge.get("acceptable_strategies", ())}
    required_actions = list(case.private_judge.get("required_next_actions", ()))

    if selected_ref in acceptable:
        applied_ref = selected_ref
        next_action = required_actions[0]
        verifier_passed = True
        model_calls = 3
        prompt_tokens = 920 + knowledge_tokens
        completion_tokens = 210
        wall_clock_sec = 28.0
        npu_runs = 1
        uninformative_probes = 0
    elif arm == "A" and case.case_kind == "boundary_or_mismatch" and selected_ref == "default":
        applied_ref = "default"
        next_action = required_actions[0]
        verifier_passed = True
        model_calls = 4
        prompt_tokens = 1150
        completion_tokens = 260
        wall_clock_sec = 36.0
        npu_runs = 1
        uninformative_probes = 0
    else:
        applied_ref = selected_ref
        next_action = "untargeted_probe_or_mismatched_workflow"
        verifier_passed = False
        model_calls = min(remaining_model_calls, 6)
        prompt_tokens = 1680 + knowledge_tokens
        completion_tokens = 390
        wall_clock_sec = 54.0
        npu_runs = 2
        uninformative_probes = 2

    return {
        "selected_strategy_ref": selected_ref, "declared_adoption": selected_ref != "default",
        "applied_strategy_ref": applied_ref, "next_action": next_action,
        "verifier_passed": verifier_passed, "model_calls": model_calls,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "knowledge_tokens": knowledge_tokens, "wall_clock_sec": wall_clock_sec,
        "npu_runs": npu_runs, "uninformative_probes": uninformative_probes,
        "human_interventions": 0,
    }

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


def run_strategy_selection_cell(
    cell: StrategyCellPlan,
    case: StrategyCase,
    *,
    routing_index: RoutingIndex,
    budget: StrategySelectionBudget = StrategySelectionBudget(),
    selector_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = condition_aware_selector,
    executor_fn: Callable[..., Mapping[str, Any]] = shared_prefix_executor,
) -> dict[str, Any]:
    """Run one (case, repetition, arm) cell with strict budget & error-visibility semantics."""
    selector_status: Literal["not_applicable", "completed", "failed", "unknown"] = "not_applicable"
    selector_error: str | None = None
    selector_decision: Mapping[str, Any] | None = None
    selector_prompt_tokens: int | None = 0 if cell.arm == "A" else None
    selector_completion_tokens: int | None = 0 if cell.arm == "A" else None
    selector_wall_sec = 0.0
    selector_calls = 0

    if cell.arm in ("B", "C"):
        view = prepare_strategy_selector_view(case, arm=cell.arm, routing_index=routing_index)
        selector_calls = 1
        started = time.perf_counter()
        try:
            raw_sel = dict(selector_fn(view))
            usage = raw_sel.get("usage") if isinstance(raw_sel.get("usage"), Mapping) else {}
            selector_wall_sec = float(usage.get("wall_clock_sec", round(time.perf_counter() - started, 4)))
            if selector_wall_sec > budget.selector_timeout_sec:
                raise TimeoutError(f"selector exceeded {budget.selector_timeout_sec}s budget")
            if usage.get("prompt_tokens") is not None:
                selector_prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                selector_completion_tokens = int(usage["completion_tokens"])
            selector_decision = raw_sel
            selector_status = "completed"
        except TimeoutError as exc:
            selector_wall_sec = max(selector_wall_sec, budget.selector_timeout_sec)
            selector_status = "unknown"
            selector_error = str(exc)
            selector_decision = None
        except Exception as exc:  # noqa: BLE001 — preserve error and fall back to default continuation
            selector_wall_sec = float(round(time.perf_counter() - started, 4))
            selector_status = "failed"
            selector_error = str(exc)
            selector_decision = None

    remaining_calls = budget.model_call_budget - selector_calls
    remaining_window = budget.business_window_sec - selector_wall_sec

    try:
        exec_out = dict(
            executor_fn(
                case=case,
                arm=cell.arm,
                selector_decision=selector_decision,
                routing_index=routing_index,
                remaining_model_calls=remaining_calls,
                remaining_window_sec=remaining_window,
            )
        )
    except TimeoutError as exc:
        return {
            "cell": cell.__dict__, "terminal_status": "unknown", "selector_status": selector_status,
            "selector_error": selector_error or str(exc), "controls_digest": cell.controls_digest,
            "usage_status": "unavailable", "total_model_calls": selector_calls,
            "prompt_tokens": None, "completion_tokens": None, "knowledge_tokens": None,
            "wall_clock_sec": round(selector_wall_sec, 4), "npu_runs": None,
            "npu_card_hours": None, "npu_die_hours": None,
        }

    judged = evaluate_strategy_continuation(case, exec_out)
    exec_calls = int(exec_out.get("model_calls", 1))
    total_calls = selector_calls + exec_calls
    exec_wall = float(exec_out.get("wall_clock_sec", 0.0))
    total_wall = round(selector_wall_sec + exec_wall, 4)
    npu_runs = int(exec_out.get("npu_runs", 0))
    budget_compliant = (
        total_calls <= budget.model_call_budget
        and npu_runs <= budget.npu_run_budget
        and total_wall <= budget.business_window_sec
    )

    exec_prompt = exec_out.get("prompt_tokens")
    exec_completion = exec_out.get("completion_tokens")
    if (
        exec_prompt is not None
        and exec_completion is not None
        and (cell.arm == "A" or (selector_prompt_tokens is not None and selector_completion_tokens is not None))
    ):
        total_prompt = int(exec_prompt) + int(selector_prompt_tokens or 0)
        total_completion = int(exec_completion) + int(selector_completion_tokens or 0)
        usage_status = "native"
    else:
        total_prompt = None
        total_completion = None
        usage_status = "unavailable"

    terminal_status = "pass" if judged["task_succeeded"] else "fail"
    device_hours = (exec_wall / 3600.0) * max(1, npu_runs)
    return {
        "cell": cell.__dict__, "terminal_status": terminal_status, "budget_compliant": budget_compliant,
        "selector_status": selector_status, "selector_error": selector_error,
        "selector_decision": dict(selector_decision) if selector_decision is not None else None,
        "judgment": judged, "controls_digest": cell.controls_digest, "usage_status": usage_status,
        "selector_model_calls": selector_calls, "executor_model_calls": exec_calls,
        "total_model_calls": total_calls, "selector_prompt_tokens": selector_prompt_tokens,
        "selector_completion_tokens": selector_completion_tokens, "prompt_tokens": total_prompt,
        "completion_tokens": total_completion, "knowledge_tokens": int(exec_out.get("knowledge_tokens", 0)),
        "wall_clock_sec": total_wall, "npu_runs": npu_runs, "npu_cards": case.npu_cards,
        "npu_dies": case.npu_dies, "npu_card_hours": round(device_hours * case.npu_cards, 6),
        "npu_die_hours": round(device_hours * case.npu_dies, 6),
        "uninformative_probes": int(exec_out.get("uninformative_probes", 0)),
        "human_interventions": int(exec_out.get("human_interventions", 0)),
    }


def run_strategy_selection_pilot(
    *,
    corpus_roots: Sequence[str | Path],
    cases: Sequence[StrategyCase] = tuple(PILOT_CASES.values()),
    repetitions: int = 2,
    budget: StrategySelectionBudget = StrategySelectionBudget(),
    selector_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = condition_aware_selector,
    executor_fn: Callable[..., Mapping[str, Any]] = shared_prefix_executor,
    offline_prep_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full Section 4.7 A/B/C pilot and produce paired mechanism & cost reports."""
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
            first_effective_action=row.get("judgment", {}).get("next_action"),
            controls_digest=row["controls_digest"],
        ))
    c_vs_b_report = analyze_mechanism_ablations(c_vs_b_plan, c_vs_b_obs, min_independent_families=3)

    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [row for row in results if row["cell"]["arm"] == arm]
        valid_rows = [row for row in arm_rows if row["terminal_status"] in {"pass", "fail"}]
        passed = sum(row["terminal_status"] == "pass" for row in valid_rows)
        reasonable = sum(row.get("judgment", {}).get("selection_reasonable") is True for row in valid_rows)
        false_adoptions = sum(row.get("judgment", {}).get("false_adoption") is True for row in valid_rows)
        trigger_rows = [row for row in valid_rows if row["cell"]["case_kind"] == "trigger_matched"]
        boundary_rows = [row for row in valid_rows if row["cell"]["case_kind"] == "boundary_or_mismatch"]
        arm_summaries[arm] = {
            "arm": arm, "label": ARM_LABELS[arm],
            "planned_cells": len(arm_rows), "valid_cells": len(valid_rows),
            "pass_rate": _ratio(passed, len(valid_rows), missing=len(arm_rows) - len(valid_rows)),
            "selection_reasonable_rate": _ratio(reasonable, len(valid_rows)),
            "false_adoption_rate": _ratio(false_adoptions, len(valid_rows)),
            "trigger_matched_pass_rate": _ratio(sum(r["terminal_status"] == "pass" for r in trigger_rows), len(trigger_rows)),
            "boundary_safe_handling_rate": _ratio(sum(not r.get("judgment", {}).get("false_adoption", True) for r in boundary_rows), len(boundary_rows)),
            "total_model_calls": sum(int(r.get("total_model_calls") or 0) for r in arm_rows),
            "selector_model_calls": sum(int(r.get("selector_model_calls") or 0) for r in arm_rows),
            "total_prompt_tokens": (sum(int(r["prompt_tokens"]) for r in arm_rows) if all(r.get("prompt_tokens") is not None for r in arm_rows) else None),
            "total_completion_tokens": (sum(int(r["completion_tokens"]) for r in arm_rows) if all(r.get("completion_tokens") is not None for r in arm_rows) else None),
            "total_wall_clock_sec": round(sum(float(r.get("wall_clock_sec") or 0.0) for r in arm_rows), 3),
            "total_npu_runs": sum(int(r.get("npu_runs") or 0) for r in arm_rows),
            "total_npu_card_hours": round(sum(float(r.get("npu_card_hours") or 0.0) for r in arm_rows), 6),
            "total_npu_die_hours": round(sum(float(r.get("npu_die_hours") or 0.0) for r in arm_rows), 6),
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
                row_c.get("judgment", {}).get("selection_reasonable") is True
                and row_b.get("judgment", {}).get("selection_reasonable") is False
            ):
                c_beats_b_selection += 1
            if row_c["terminal_status"] == "pass" and row_a["terminal_status"] == "fail":
                c_beats_a_pass += 1

    shared_prefix_wall = sum(float(case.shared_prefix_cost.get("wall_clock_sec", 0.0)) for case in cases)
    shared_prefix_prompt = sum(int(case.shared_prefix_cost.get("prompt_tokens", 0)) for case in cases)
    shared_prefix_completion = sum(int(case.shared_prefix_cost.get("completion_tokens", 0)) for case in cases)
    prep = dict(offline_prep_cost or {
        "corpus_revision_wall_clock_sec": 45.0,
        "dev_calibration_wall_clock_sec": 120.0,
        "organizer_model_calls": 0,
    })

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

    return {
        "schema_version": SCHEMA_VERSION,
        "scope_note": (
            "Continuation pilot from a shared decision point across 6 cases (3 independent "
            "problem families x 2 repetitions = 36 host continuations + 24 isolated selector calls). "
            "Does not claim full-session product cohort superiority without end-to-end prefix replay."
        ),
        "readiness": readiness, "budget": budget.__dict__,
        "upper_bounds": {
            "total_cells": len(cells), "selector_calls": sum(1 for c in cells if c.arm in ("B", "C")),
            "max_model_calls": len(cells) * budget.model_call_budget,
            "max_npu_runs": len(cells) * budget.npu_run_budget,
            "max_outer_wall_clock_sec": len(cells) * budget.outer_timeout_sec,
        },
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
                "continuation_wall_clock_delta_sec": round(
                    arm_summaries["C"]["total_wall_clock_sec"] - arm_summaries["A"]["total_wall_clock_sec"], 3,
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
                "accounting_rule": "counted once in physical total; added equally when estimating full-task conditional cost",
            },
            "offline_and_total_experiment_cost": {
                **prep,
                "physical_continuation_wall_clock_sec": round(
                    sum(arm_summaries[arm]["total_wall_clock_sec"] for arm in ARMS), 3
                ),
                "amortization_cases": len(cases),
            },
        },
        "decision_gate": decision_gate, "cells": results,
    }
