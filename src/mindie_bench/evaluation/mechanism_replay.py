"""Transcript decision-mechanism replay M1-01 through M6-01.

[Role]
Expose only pre-decision facts to a learner, keep post-decision observations,
perturbation explanations and scoring in a private judge, and evaluate a
structured next-action decision against the Section 4.6 mechanism contract.

[Invariants]
- Prompt views contain only pre-decision facts, competing hypotheses and exact
  T1-T5 provenance locators. Private post-decision observations, perturbation
  explanations, scoring and leak markers never enter a view.
- Evaluation checks structured decisions (fusion boundary, logical alignment,
  observer control, sequence state, independent oracle, scoped counter-evidence)
  rather than keyword hits in prose.
- `M6-01` binds U@1 -> U@2 to a real `GradedRepository`: the old revision must
  be unreadable, old conditions must remain, the new branch must be recalled,
  and active-unit/body growth must respect the 16 KiB complete-body and byte
  budget contracts.
- Missing repository evidence is `unknown`, never a fabricated pass.

[Timeouts & Locks]
- Prompt and decision evaluation are pure in-memory work. The repository
  collector uses the caller-owned `GradedRepository` read APIs only.

[Side-Effect Policy]
- Never mutates a corpus, store, receipt or transcript. The collector is
  read-only; revision application belongs to the production knowledge runtime.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

from mindie_bench.evaluation.mechanism_replay_cases import (
    CASES,
    Hypothesis,
    MechanismCase,
    ReplayFact,
    resolve_case,
)
from mindie_bench.evaluation.mechanism_replay_host import (
    MAX_MERGED_BODY_BYTES,
    SCHEMA,
    _m6_repository,
    collect_m6_repository_evidence,
    evaluate_mechanism_replay,
    run_host_mechanism_replay,
)

PROMPT_SCHEMA = "mindie-bench-mechanism-prompt/1"
DECISION_SCHEMA = "mindie-bench-mechanism-decision/1"
SOURCES = ("T1", "T2", "T3", "T4", "T5")
RELATED_BODY_BUDGET_BYTES = 16 * 1024
__all__ = [
    "CASES", "DECISION_SCHEMA", "Hypothesis", "MAX_MERGED_BODY_BYTES", "MechanismCase",
    "PROMPT_SCHEMA", "RELATED_BODY_BUDGET_BYTES", "ReplayFact", "SCHEMA", "SOURCES",
    "collect_m6_repository_evidence", "evaluate_mechanism_replay", "prepare_learner_prompt_view",
    "resolve_case", "run_host_mechanism_replay", "verify_prompt_isolation",
]

FORBIDDEN_PRIVATE_KEYS = {
    "post_decision", "post_decision_observations", "perturbation_explanation",
    "scoring", "rubric", "expected", "correct_action", "private", "answer",
}


# ------------------------------------------------------------- prompt isolation
def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_keys(item)


def verify_prompt_isolation(view: Mapping, case: MechanismCase) -> dict[str, Any]:
    """Prove a learner view contains no private judge key or leak marker."""
    keys = list(_walk_keys(view))
    forbidden = sorted({key for key in keys if key.casefold() in FORBIDDEN_PRIVATE_KEYS})
    public_text = json.dumps(view, sort_keys=True, ensure_ascii=False, default=str)
    markers = [str(marker) for marker in case.private_judge.get("leak_markers", ())]
    leaked_markers = [marker for marker in markers if marker in public_text]
    fact_anchors = {fact.anchor for fact in case.facts}
    view_anchors = {anchor for values in view.get("provenance_locators", {}).values() for anchor in values}
    unanchored = sorted(view_anchors - fact_anchors)
    unknown_sources = sorted(source for source in view.get("provenance_locators", {}) if source not in SOURCES)
    leaks = [
        *(f"private key: {key}" for key in forbidden),
        *(f"leak marker: {marker}" for marker in leaked_markers),
        *(f"unanchored locator: {anchor}" for anchor in unanchored),
        *(f"unknown source: {source}" for source in unknown_sources),
    ]
    return {"isolated": not leaks, "leaks": leaks, "checked_markers": markers}


def prepare_learner_prompt_view(case: str | MechanismCase) -> dict[str, Any]:
    """Expose only pre-decision facts, competing hypotheses and T1-T5 locators."""
    selected = resolve_case(case) if isinstance(case, str) else case
    anchors: dict[str, list[str]] = {source: [] for source in selected.sources}
    for fact in selected.facts:
        source = fact.anchor.split(":", 1)[0]
        if fact.anchor not in anchors[source]:
            anchors[source].append(fact.anchor)
    view = {
        "schema_version": PROMPT_SCHEMA,
        "case_id": selected.case_id,
        "pre_decision_facts": [
            {"fact_id": fact.fact_id, "statement": fact.statement, "provenance": fact.anchor}
            for fact in selected.facts
        ],
        "competing_hypotheses": [
            {"hypothesis_id": item.hypothesis_id, "statement": item.statement}
            for item in selected.hypotheses
        ],
        "provenance_locators": anchors,
    }
    report = verify_prompt_isolation(view, selected)
    if not report["isolated"]:
        raise ValueError("learner prompt view leaks private judge material: " + "; ".join(report["leaks"]))
    return view


# --------------------------------------------------------------- decision checks
def _dig(decision: Mapping, path: str, default=None):
    value: Any = decision
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _choice(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").split())


def _expr(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).casefold())


def _locators(case: MechanismCase) -> set[str]:
    return {fact.anchor for fact in case.facts}


def _case_domain_tokens(case: MechanismCase) -> set[str]:
    text = " ".join(fact.statement for fact in case.facts) + " " + " ".join(
        item.statement for item in case.hypotheses
    )
    raw = {token.casefold() for token in re.findall(r"[A-Za-z0-9_/-]{3,}", text)}
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "not", "can",
        "two", "only", "after", "while", "into", "than", "yet", "has", "have", "been",
    }
    return {tok for tok in raw if tok not in stopwords}


def _hypothesis_entry_grounded(item: Any, case: MechanismCase, decision: Mapping) -> str | None:
    known_ids = {hyp.hypothesis_id for hyp in case.hypotheses}
    custom_map = decision.get("custom_hypotheses")
    custom_lookup = dict(custom_map) if isinstance(custom_map, Mapping) else {}
    if isinstance(item, str):
        ident = item.strip()
        if not ident:
            return None
        if ident in known_ids:
            return ident
        statement = str(
            custom_lookup.get(ident)
            or (decision.get("hypothesis_statement") if ident == decision.get("hypothesis_id") else "")
            or ident
        ).strip()
    elif isinstance(item, Mapping):
        ident = str(item.get("hypothesis_id") or item.get("id") or item.get("statement") or "").strip()
        if ident in known_ids:
            return ident
        statement = str(item.get("statement") or item.get("rationale") or "").strip()
    else:
        return None
    lower = (ident + " " + statement).casefold()
    if any(bad in lower for bad in ("bogus", "fake", "placeholder", "todo", "ungrounded")):
        return None
    words = {tok.casefold() for tok in re.findall(r"[A-Za-z0-9_/-]{3,}", statement)}
    if len(statement) >= 20 and len(words & _case_domain_tokens(case)) >= 2:
        return ident or statement
    return None


def _critical_path_equivalent(critical: Mapping[str, Any]) -> bool:
    expr = _expr(critical.get("expression"))
    if expr in {
        "max(p+c,q+g)", "max(q+g,p+c)", "max(c+p,q+g)", "max(p+c,g+q)",
        "max(c+p,g+q)", "max(q+g,c+p)", "max(g+q,p+c)", "max(g+q,c+p)",
    }:
        return True
    branches = {_expr(item) for item in (critical.get("branches") or [])}
    return bool(branches & {"p+c", "c+p"}) and bool(branches & {"q+g", "g+q"})


def _evidence(decision: Mapping, case: MechanismCase) -> tuple[bool, str]:
    cited = decision.get("evidence_locators")
    if not isinstance(cited, list) or len(cited) < 2:
        return False, "at least two exact provenance locators are required"
    unknown = [item for item in cited if item not in _locators(case)]
    if unknown:
        return False, f"citations are not in the public provenance set: {unknown}"
    return True, "citations resolve to public pre-decision facts"


def _m1_phase0(decision, case, context):
    phase0 = decision.get("phase0") or {}
    sites = phase0.get("fusion_sites") or []
    site_ok = any(
        any(token in _choice(item.get("kind"))
            for token in ("producer", "writeback", "chain", "helper"))
        and item.get("eliminates_gm_roundtrip") is True
        and bool(item.get("producer")) and bool(item.get("consumer"))
        for item in sites if isinstance(item, Mapping)
    )
    boundaries = " ".join(str(item).casefold() for item in phase0.get("hard_boundaries") or [])
    boundaries_ok = "collective" in boundaries and ("capacity" in boundaries or "ub" in boundaries)
    critical = phase0.get("critical_path") or {}
    path_ok = _critical_path_equivalent(critical) and (
        critical.get("on_exposed_path") is True
    )
    deferred = phase0.get("single_kernel_tweaks_deferred") is True
    passed = site_ok and boundaries_ok and path_ok and deferred
    return passed, (
        f"fusion_site={site_ok} hard_boundaries={boundaries_ok} "
        f"critical_path={path_ok} single_kernel_deferred={deferred}"
    )


def _m1_probe(decision, case, context):
    probe = decision.get("next_probe") or {}
    local = probe.get("kind") in {"local_forward_comparison", "boundary_ablation"}
    raw_items = list(probe.get("discriminates") or [])
    grounded_items = [_hypothesis_entry_grounded(item, case, decision) for item in raw_items]
    discriminating = bool(raw_items) and all(grounded_items) and len(set(grounded_items)) >= 2
    rejected = probe.get("alternative_rejected") == "cross_version_bisect"
    hours = probe.get("estimated_hours")
    cheap = isinstance(hours, (int, float)) and not isinstance(hours, bool) and 0 < hours <= 1
    passed = local and discriminating and rejected and cheap
    return passed, (
        f"local_probe={local} discriminates={discriminating} grounded={grounded_items} "
        f"bisect_rejected={rejected} cost_hours={hours}"
    )


def _m1_no_repeat(decision, case, context):
    probe = decision.get("next_probe") or {}
    passed = probe.get("repeats_prior_negative") is False
    return passed, f"repeats_prior_negative={probe.get('repeats_prior_negative')}"


def _m2_logical(decision, case, context):
    first = decision.get("first_divergence") or {}
    alignment = _choice(first.get("alignment"))
    logical = alignment in {"logical token state", "logical state", "logical token"}
    physical = first.get("physical_block_id_used") is False
    checks = {str(item).casefold() for item in first.get("checks") or []}
    coverage = {"shape", "dtype", "stride", "layout", "lifetime"} <= checks
    boundary = _choice(first.get("boundary"))
    passed = logical and physical and coverage and ("moe" in boundary or "partial sum" in boundary)
    return passed, (
        f"alignment={alignment} physical_id={physical} checks={sorted(checks)} boundary={boundary}"
    )


def _m2_downstream(decision, case, context):
    contract = decision.get("downstream_contract") or {}
    stages = [_choice(item) for item in contract.get("stages") or []]
    order = contract.get("order_compared") is True
    collective = contract.get("collective_matched") is True
    reduction = any("reduction" in item for item in stages)
    rmsnorm = any("rmsnorm" in item for item in stages)
    overclaim = contract.get("module_declared_correct_from_collective_only") is False
    passed = order and collective and reduction and rmsnorm and overclaim
    return passed, (
        f"order={order} collective={collective} reduction={reduction} "
        f"rmsnorm={rmsnorm} collective_only_overclaim={overclaim}"
    )


def _m2_simulation(decision, case, context):
    simulation = decision.get("simulation") or {}
    kind = _choice(simulation.get("cycle_bound_kind"))
    non_additive = simulation.get("additive_cycle_bound") is False
    missing = simulation.get("missing_durations", "<absent>")
    kept_missing = missing is None
    unmeasured = bool(simulation.get("unmeasured_stages"))
    critical = simulation.get("critical_path") or {}
    path_ok = _critical_path_equivalent(critical)
    passed = ("max cycle mean" in kind or "karp" in kind) and non_additive and kept_missing \
        and unmeasured and path_ok
    return passed, (
        f"bound={kind} non_additive={non_additive} missing_durations={missing!r} "
        f"unmeasured={unmeasured} critical_path={path_ok}"
    )


def _m2_no_burst(decision, case, context):
    simulation = decision.get("simulation") or {}
    burst = _choice(simulation.get("clock_claim"))
    accepted = simulation.get("sustained_claim") is False
    passed = burst in {"burst only", "burst", "not measured", "unknown"} and accepted
    return passed, f"clock_claim={burst} sustained_claim={simulation.get('sustained_claim')}"


def _m3_controls(decision, case, context):
    observer = decision.get("observer") or {}
    keys = ("uninstrumented_baseline", "minimal_intervention", "observer_only_control",
            "observation_changes_result")
    missing = [key for key in keys if observer.get(key) is not True]
    return not missing, f"missing_controls={missing}"


def _m3_branch(decision, case, context):
    observer = decision.get("observer") or {}
    claim = _choice(observer.get("claim"))
    observed = {**dict(case.observed_results), **(context.get("observed_results") or {})}
    actual_count = observed.get("candidate_branch_execution_count", 0)
    actual_uninstrumented = observed.get("uninstrumented_passed", False)
    count = observer.get("candidate_branch_execution_count", actual_count)
    count_consistent = isinstance(count, int) and not isinstance(count, bool) and count == actual_count
    dump_as_fix = observer.get("dump_pass_called_fix") is True
    inconclusive = observer.get("inconclusive_retained") is True
    if claim in {"inconclusive", "observer effect"}:
        passed = count_consistent and inconclusive and not dump_as_fix
    elif claim == "fix":
        passed = (
            count_consistent
            and actual_count > 0
            and actual_uninstrumented is True
            and not dump_as_fix
        )
    else:
        passed = False
    return passed, (
        f"claim={claim} branch_count={count} actual_branch_count={actual_count} "
        f"uninstrumented_passed={actual_uninstrumented} inconclusive={inconclusive} dump_as_fix={dump_as_fix}"
    )


def _m3_dfx(decision, case, context):
    observer = decision.get("observer") or {}
    passed = observer.get("dfx_disabled_final_verification") is True \
        and observer.get("draft_claimed_executed") is False
    return passed, (
        f"dfx_disabled_final={observer.get('dfx_disabled_final_verification')} "
        f"draft_claimed_executed={observer.get('draft_claimed_executed')}"
    )


def _subsequence(small: Sequence, large: Sequence) -> bool:
    iterator = iter(large)
    return all(any(item == candidate for candidate in iterator) for item in small)


def _m4_history(decision, case, context):
    repro = decision.get("reproduction") or {}
    observed = {**dict(case.observed_results), **(context.get("observed_results") or {})}
    fresh = repro.get("fresh_process") or {}
    history = repro.get("historical_process") or {}
    fresh_passes = fresh.get("passes", observed.get("fresh_process_passes"))
    history_fails = history.get("fails", observed.get("historical_process_fails"))
    passed = (
        fresh_passes is True
        and history_fails is True
        and fresh_passes == observed.get("fresh_process_passes", True)
        and history_fails == observed.get("historical_process_fails", True)
        and repro.get("only_final_input") is False
    )
    return passed, (
        f"fresh_passes={fresh_passes} history_fails={history_fails} "
        f"only_final_input={repro.get('only_final_input')}"
    )


def _m4_sequence(decision, case, context):
    repro = decision.get("reproduction") or {}
    full = repro.get("full_sequence") or []
    minimal = repro.get("minimal_sequence") or []
    within = bool(full) and bool(minimal) and len(minimal) < len(full) and _subsequence(minimal, full)
    passed = within and repro.get("minimal_sequence_replayed") is True
    return passed, (
        f"minimal={minimal} full={full} subsequence={within} "
        f"replayed={repro.get('minimal_sequence_replayed')}"
    )


def _m4_reset(decision, case, context):
    reset = (decision.get("reproduction") or {}).get("reset") or {}
    covers = {str(item).casefold() for item in reset.get("covers") or []}
    expected = {"kv_cache", "conv_state", "slot_owner"}
    passed = reset.get("verified") is True and expected <= covers \
        and reset.get("treated_as_complete") is False
    return passed, f"verified={reset.get('verified')} covers={sorted(covers)}"


def _m4_concurrency(decision, case, context):
    repro = decision.get("reproduction") or {}
    serial = repro.get("serial_passes") is True
    concurrent = repro.get("concurrent_deadlocks") is True
    preserved = repro.get("concurrency_preserved") is True
    serial_claim = repro.get("claimed_serial_sufficient") is True
    passed = (not (serial and concurrent)) or (preserved and not serial_claim)
    return passed, (
        f"serial={serial} concurrent={concurrent} preserved={preserved} "
        f"serial_sufficient={serial_claim}"
    )


def _m4_no_fix(decision, case, context):
    repro = decision.get("reproduction") or {}
    passed = repro.get("claimed_fixed_after_fresh_process") is False
    return passed, f"fresh_process_as_fix={repro.get('claimed_fixed_after_fresh_process')}"


def _m5_oracle(decision, case, context):
    oracle = decision.get("oracle") or {}
    observed = {**dict(case.observed_results), **(context.get("observed_results") or {})}
    independent = oracle.get("independent") is True
    shared = bool(oracle.get("shared_defect"))
    shared_as_proof = oracle.get("uses_shared_defect_as_proof") is True
    rate_as_correctness = oracle.get("acceptance_rate_used_as_correctness") is True
    accepted = oracle.get("acceptance_rate", observed.get("acceptance_rate"))
    high_acceptance = isinstance(accepted, (int, float)) and accepted >= 1.0
    passed = independent and shared and not shared_as_proof and not rate_as_correctness \
        and high_acceptance
    return passed, (
        f"independent={independent} shared_defect={shared} shared_as_proof={shared_as_proof} "
        f"acceptance_as_correctness={rate_as_correctness} acceptance={accepted}"
    )


def _m5_path(decision, case, context):
    oracle = decision.get("oracle") or {}
    claim = _choice(oracle.get("claim"))
    observed = {**dict(case.observed_results), **(context.get("observed_results") or {})}
    actual_count = observed.get("candidate_branch_execution_count", 0)
    actual_continued = observed.get("continued_decode_passed", False)
    count = oracle.get("candidate_branch_execution_count", actual_count)
    count_consistent = isinstance(count, int) and not isinstance(count, bool) and count == actual_count
    zero_claim = oracle.get("zero_execution_patch_claimed_fix") is True
    exit_zero = oracle.get("exit_zero_used_as_accuracy") is True
    health = oracle.get("health_used_as_accuracy") is True
    if not count_consistent:
        passed = False
    elif claim == "fix" and (actual_count < 1 or actual_continued is not True):
        passed = False
    else:
        passed = not zero_claim and not exit_zero and not health
    return passed, (
        f"claim={claim} branch_count={count} actual_branch_count={actual_count} "
        f"zero_execution_claim={zero_claim} exit_zero={exit_zero} health={health}"
    )


def _m5_scopes(decision, case, context):
    scopes = decision.get("scopes") or {}
    first = _choice((scopes.get("first_token") or {}).get("state"))
    continued = _choice((scopes.get("continued_decode") or {}).get("state"))
    concurrent = _choice((scopes.get("concurrent") or {}).get("state"))
    generalized = decision.get("first_token_generalized") is True
    distinct = {"first_token", "continued_decode", "concurrent"} <= set(scopes)
    passed = first in {"passed", "failed", "not measured"} \
        and continued in {"passed", "failed", "not measured", "unknown"} \
        and concurrent in {"passed", "failed", "not measured", "unknown"} \
        and distinct and not generalized
    return passed, (
        f"first={first} continued={continued} concurrent={concurrent} generalized={generalized}"
    )


def _m5_gates(decision, case, context):
    plan = decision.get("gate_plan") or {}
    complete = all(
        _choice(plan.get(gate)) in {"run", "reuse", "passed", "completed", "executed"}
        for gate in ("g0", "g1", "g2", "g3")
    )
    incremental = plan.get("g1_incremental_build") is True
    abba = plan.get("g2_sustained_abba") is True
    manual = plan.get("manual_glue_required") is True
    passed = complete and incremental and abba and not manual
    return passed, (
        f"gates_complete={complete} g1_incremental={incremental} g2_abba={abba} "
        f"manual_glue={manual}"
    )


def _m6_merge(decision, case, context):
    revision = decision.get("revision") or {}
    passed = revision.get("operation") == "merge_into_existing" \
        and revision.get("new_unit_created") is False \
        and revision.get("active_unit_count_delta") == 0
    return passed, (
        f"operation={revision.get('operation')} new_unit={revision.get('new_unit_created')} "
        f"active_delta={revision.get('active_unit_count_delta')}"
    )


def _m6_body_complete(decision, case, context):
    revision = decision.get("revision") or {}
    snapshot = revision.get("related_unit_snapshot") or {}
    actual = (context.get("m6_repository") or {})
    size = snapshot.get("body_excerpt_bytes", actual.get("before_body_bytes"))
    complete = snapshot.get("body_complete") is True
    within = isinstance(size, int) and 0 < size <= RELATED_BODY_BUDGET_BYTES
    expected_sha = snapshot.get("body_sha256", actual.get("before_body_sha256"))
    hash_matches = expected_sha == actual.get("before_body_sha256") if actual else None
    passed = complete and within and hash_matches is not False
    return passed, (
        f"body_complete={complete} excerpt_bytes={size} <= {RELATED_BODY_BUDGET_BYTES} "
        f"sha_matches={hash_matches}"
    )


def _m6_anti_bloat(decision, case, context):
    revision = decision.get("revision") or {}
    actual = (context.get("m6_repository") or {})
    compact = revision.get("rewrite_to_compact") is True
    append = revision.get("append_only") is True
    adds_valid = revision.get("adds_valid_branch") is True
    after = revision.get("after_body_bytes", actual.get("current_body_bytes"))
    before = revision.get("before_body_bytes", actual.get("before_body_bytes"))
    net = revision.get(
        "net_body_bytes",
        (after - before) if isinstance(after, int) and isinstance(before, int) else None,
    )
    cap = revision.get("max_body_bytes", MAX_MERGED_BODY_BYTES)
    net_ok = isinstance(net, int) and (net <= 0 or adds_valid)
    within = (
        net_ok
        and isinstance(after, int)
        and isinstance(cap, int)
        and 0 < after <= cap <= MAX_MERGED_BODY_BYTES
    )
    passed = compact and not append and within
    return passed, (
        f"rewrite_to_compact={compact} append_only={append} net_bytes={net} "
        f"after_bytes={after} max_body_bytes={cap} adds_valid_branch={adds_valid}"
    )


def _m6_counter_evidence(decision, case, context):
    revision = decision.get("revision") or {}
    evidence = decision.get("evidence") or {}
    retained = evidence.get("counter_evidence_retained") or []
    corrections = evidence.get("human_corrections") or []
    proofs = evidence.get("technical_proofs") or []
    separated = evidence.get("separated") is True
    layouts = {str(item) for item in revision.get("superseded_layouts") or []}
    revived = revision.get("revived_layouts") or []
    excluded = {str(item) for item in revision.get("excluded_hypotheses") or []}
    reopened = revision.get("reopened_hypotheses") or []
    changed = revision.get("reopen_conditions_changed") is True
    passed = bool(retained) and bool(corrections) and bool(proofs) and separated \
        and {"4:1", "parent/subslot"} <= layouts and not revived \
        and "zeroer" in excluded and not reopened and not changed
    return passed, (
        f"retained={len(retained)} corrections={len(corrections)} proofs={len(proofs)} "
        f"separated={separated} superseded={sorted(layouts)} revived={revived} "
        f"excluded={sorted(excluded)} reopened={reopened} reopened_with_new_condition={changed}"
    )


_CRITERIA: dict[str, Callable] = {
    "m1_phase0_fusion": _m1_phase0,
    "m1_distinguishing_next_probe": _m1_probe,
    "m1_avoids_repeat": _m1_no_repeat,
    "m2_logical_first_divergence": _m2_logical,
    "m2_downstream_contract": _m2_downstream,
    "m2_nonadditive_simulation": _m2_simulation,
    "m2_no_short_burst_claim": _m2_no_burst,
    "m3_observer_controls": _m3_controls,
    "m3_branch_execution_claim": _m3_branch,
    "m3_dfx_disabled_final": _m3_dfx,
    "m4_history_pair": _m4_history,
    "m4_minimal_sequence": _m4_sequence,
    "m4_reset_semantics": _m4_reset,
    "m4_concurrency_boundary": _m4_concurrency,
    "m4_no_fresh_process_fix": _m4_no_fix,
    "m5_independent_oracle": _m5_oracle,
    "m5_actual_path": _m5_path,
    "m5_scope_separation": _m5_scopes,
    "m5_gate_runner": _m5_gates,
    "m6_merge_not_create": _m6_merge,
    "m6_body_complete_budget": _m6_body_complete,
    "m6_anti_bloat": _m6_anti_bloat,
    "m6_counter_evidence": _m6_counter_evidence,
    "m6_repository_revision": _m6_repository,
}


# --------------------------------------------------------- M6 repository binding
