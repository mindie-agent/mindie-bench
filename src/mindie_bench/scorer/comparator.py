"""Compare matched trials only; incompatible frozen baselines are rejected."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

from mindie_bench.evaluation.accounting import scope_cells
from mindie_bench.io import digest, read_json, write_json
from mindie_bench.models import TerminalStatus, TrialReceipt
from mindie_bench.scorer.components import summarize_five_components
from mindie_bench.scorer.metrics import summarize

CONTROL_KEYS = (
    "host_os",
    "host_arch",
    "harness_name",
    "harness_version",
    "harness_binary_sha256",
    "base_model",
    "reasoning_effort",
    "model_catalog_sha256",
    "provider_url",
    "backend",
    "budget",
    "base_commit",
    "npu_model",
    "npu_count",
    "remote_environment",
    "benchmark_version",
    "benchmark_source_sha256",
)


def load_receipts(root: Path) -> list[TrialReceipt]:
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("trial_receipt.json"))
    if not paths:
        raise ValueError(f"no receipts found: {root}")
    if root.is_dir():
        for marker in root.rglob("attempt_started.json"):
            if not (marker.parent / "trial_receipt.json").is_file():
                raise ValueError(
                    f"interrupted attempt is UNKNOWN, never automatically replay it: {marker.parent}"
                )
    raw = [read_json(path) for path in paths]
    receipts = [TrialReceipt.model_validate(value) for value in raw]
    if root.is_dir() and (root / "baseline_manifest.json").exists():
        frozen = read_json(root / "baseline_manifest.json")["receipt_sha256"]
        actual = {value["trial_id"]: digest(value) for value in raw}
        if frozen != actual:
            raise ValueError("frozen baseline content failed integrity verification")
    if root.is_dir() and (root / "eval_manifest.json").exists():
        manifest = read_json(root / "eval_manifest.json")
        expected = len(manifest["tasks"]) * len(manifest["groups"]) * manifest["repetitions"]
        if len(receipts) != expected:
            raise ValueError(
                f"incomplete experiment matrix: {len(receipts)} terminal receipts for {expected} cells"
            )
    ids = [r.trial_id for r in receipts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trial IDs; refusing double-counted evidence")
    keys = [(r.instance_id, r.tier, r.ablation_group, r.repetition) for r in receipts]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/group/repetition cells; select a single experiment")
    return receipts


def comparison(
    baseline: list[TrialReceipt],
    candidate: list[TrialReceipt],
    *,
    k: int = 3,
    role: str = "eval",
    expected_cells: list[dict] | None = None,
    blocked_cells: list[dict] | None = None,
    baseline_cells: list[dict] | None = None,
    candidate_cells: list[dict] | None = None,
) -> dict:
    if role not in {"seed", "eval"}:
        raise ValueError("comparison role must be seed or eval")
    complete_inputs = baseline, candidate
    baseline = [row for row in baseline if row.task_role == role]
    candidate = [row for row in candidate if row.task_role == role]

    def indexed(rows):
        if len({r.ablation_group for r in rows}) != 1:
            raise ValueError("compare exactly one ablation group per input")
        result = {(r.instance_id, r.tier, r.repetition): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate comparison cells")
        return result

    left, right = indexed(baseline), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("baseline and candidate must contain the same tasks, tiers, and repetitions")
    if len({(r.tier, r.evidence) for r in baseline + candidate}) != 1:
        raise ValueError("never mix live/replay or synthetic/verified evidence")
    for key in left:
        a, b = left[key], right[key]
        if a.task_digest != b.task_digest:
            raise ValueError(f"task contents changed: {a.instance_id}")
        for field in CONTROL_KEYS:
            if field not in a.environment_fingerprint or field not in b.environment_fingerprint:
                raise ValueError(f"missing controlled fingerprint field: {field}")
            if a.environment_fingerprint[field] != b.environment_fingerprint[field]:
                raise ValueError(f"incompatible baseline: {a.instance_id}: {field} changed")
        shared = set(a.environment_fingerprint.get("components", [])) & set(
            b.environment_fingerprint.get("components", [])
        )
        for component in shared:
            av = a.environment_fingerprint.get("component_revisions", {}).get(component)
            bv = b.environment_fingerprint.get("component_revisions", {}).get(component)
            if not av or av != bv:
                raise ValueError(f"shared component revision changed or missing: {component}")
        skills = [row.environment_fingerprint.get("skills_sha256") for row in (a, b)]
        if all(skills) and skills[0] != skills[1]:
            raise ValueError("shared skill snapshot changed")
    plans = [
        baseline_cells if baseline_cells is not None else expected_cells,
        candidate_cells if candidate_cells is not None else expected_cells,
    ]
    plans_available = all(plan is not None for plan in plans)
    a, b = [
        summarize(
            rows,
            k=k,
            expected_cells=scope_cells(
                plan, rows[0].tier, rows[0].ablation_group.value, role,
                evidence=rows[0].evidence, receipts=rows,
            ),
            blocked_cells=scope_cells(
                blocked_cells, rows[0].tier, rows[0].ablation_group.value, role,
                evidence=rows[0].evidence, receipts=rows,
            ),
        )
        for rows, plan in zip((baseline, candidate), plans, strict=True)
    ]
    if any(
        side["matrix_coverage"].get(field)
        for side in (a, b)
        for field in ("identity_mismatches", "unexpected_terminal_cells", "duplicate_terminal_cells")
    ):
        raise ValueError("comparison cells differ from the declared inventory")
    valid = {TerminalStatus.PASS, TerminalStatus.FAIL}
    for rows, selected, summary in zip(complete_inputs, (baseline, candidate), (a, b), strict=True):
        required = role == "eval" and selected[0].ablation_group.value == "G3"

        def lineage(row):
            return row.tier, row.evidence, row.cluster, row.repetition

        needed = {lineage(row) for row in selected} if required else set()
        seeds = [
            row
            for row in rows
            if row.task_role == "seed"
            and row.ablation_group == selected[0].ablation_group
            and row.cluster
            and lineage(row) in needed
        ]
        if len({(row.instance_id, row.tier, row.repetition) for row in seeds}) != len(seeds):
            raise ValueError("duplicate seed setup cells")
        observed = {lineage(row) for row in seeds}
        missing = needed - observed
        setup_complete = not missing and all(
            row.terminal_status in valid
            and row.auxiliary_metrics.get("capture", {}).get("status") == "completed"
            for row in seeds
        )
        resources = summarize(selected + seeds, k=k)
        costs = [
            row.primary_metrics.get("estimated_cost_usd")
            for row in selected + seeds
            if row.primary_metrics.get("estimated_cost_usd") is not None
        ]
        seed_costs = [
            row.primary_metrics.get("estimated_cost_usd")
            for row in seeds
            if row.primary_metrics.get("estimated_cost_usd") is not None
        ]
        # A zero lower bound from absent prices is not a dollar observation.
        if not any(row.primary_metrics.get("estimated_cost_usd") is not None for row in selected):
            summary["observed_cost_usd_lower_bound"] = None
        cost_complete = (
            setup_complete and resources["cost_complete"] and len(costs) == len(selected + seeds)
        )
        summary.update(
            seed_setup_required=required,
            seed_setup_attempts=len(seeds),
            seed_setup_lineages_required=len(needed),
            seed_setup_lineages_observed=len(observed),
            seed_setup_missing_lineages=[
                dict(tier=tier, evidence=evidence, cluster=cluster, repetition=repetition)
                for tier, evidence, cluster, repetition in sorted(missing, key=str)
            ],
            seed_setup_receipts_complete=not missing,
            seed_setup_complete=setup_complete,
            seed_setup_statuses=summarize(seeds, k=k)["statuses"],
            seed_setup_cost_usd_lower_bound=sum(seed_costs) if seed_costs else None,
            observed_cost_usd_including_seed_lower_bound=sum(costs) if costs else None,
            cost_including_seed_complete=cost_complete,
            cost_per_resolved_task_including_seed=(
                sum(costs) / summary["completed_pass"]
                if cost_complete and summary["completed_pass"]
                else None
            ),
            observed_tokens_including_seed_lower_bound=resources["observed_tokens_lower_bound"],
            token_metering_including_seed_complete=(
                setup_complete and resources["token_metering_complete"]
            ),
            observed_delivery_sec_including_seed_lower_bound=resources[
                "observed_delivery_wall_clock_sec_lower_bound"
            ],
            delivery_timing_including_seed_complete=(
                setup_complete and resources["delivery_timing_complete"]
            ),
        )
    paired = [
        (left[key], right[key])
        for key in left
        if left[key].terminal_status in valid and right[key].terminal_status in valid
    ]
    solved_pairs = [
        (x, y) for x, y in paired if x.terminal_status == y.terminal_status == TerminalStatus.PASS
    ]

    def ratio(field, pairs):
        values = [(x.primary_metrics.get(field), y.primary_metrics.get(field)) for x, y in pairs]
        if not values or any(x is None or y is None for x, y in values):
            return None
        numerator, denominator = sum(x for x, _ in values), sum(y for _, y in values)
        return numerator / denominator if denominator else None

    npu_values = [
        (x.primary_metrics.get("npu_test_runs"), y.primary_metrics.get("npu_test_runs"))
        for x, y in paired
    ]
    saved_npu_fraction = None
    if npu_values and all(x is not None and y is not None for x, y in npu_values):
        total_baseline = sum(x for x, _ in npu_values)
        if total_baseline:
            saved_npu_fraction = 1 - sum(y for _, y in npu_values) / total_baseline
    localization = []
    observed_localization = []
    for x, y in paired:
        observed = (
            x.primary_metrics.get("turns_to_first_localization"),
            y.primary_metrics.get("turns_to_first_localization"),
        )
        if all(value is not None for value in observed):
            observed_localization.append(observed)
        localization.append(
            (
                x.primary_metrics.get("turns_to_first_localization")
                or x.primary_metrics["localization_penalty_turns"],
                y.primary_metrics.get("turns_to_first_localization")
                or y.primary_metrics["localization_penalty_turns"],
            )
        )
    funnel, funnel_bases = {}, {}
    for field in (
        "total_turns",
        "npu_execution_wait_sec",
        "knowledge_query_sec",
        "llm_and_exploration_sec",
    ):
        values = [(x.primary_metrics.get(field), y.primary_metrics.get(field)) for x, y in paired]
        bases = (
            {
                row.primary_metrics.get(
                    "knowledge_measurement_basis", "legacy_native_tool_trajectory_only"
                )
                for pair in paired
                for row in pair
            }
            if field in {"knowledge_query_sec", "llm_and_exploration_sec"}
            else set()
        )
        funnel_bases[field] = dict(
            measurement_bases=sorted(bases),
            comparable=len(bases) <= 1,
            reason="knowledge_measurement_basis_mismatch" if len(bases) > 1 else None,
        )
        funnel[field] = (
            {"baseline_mean": mean(x for x, _ in values), "candidate_mean": mean(y for _, y in values)}
            if values and len(bases) <= 1 and all(x is not None and y is not None for x, y in values)
            else None
        )
    paired_lift = (
        mean(
            (y.terminal_status == TerminalStatus.PASS) - (x.terminal_status == TerminalStatus.PASS)
            for x, y in paired
        )
        if paired
        else None
    )
    declared_complete = (
        a["matrix_coverage"]["complete"]
        and b["matrix_coverage"]["complete"]
        and a["seed_setup_complete"]
        and b["seed_setup_complete"]
        if plans_available
        else None
    )
    time_observed_pairs = sum(
        x.primary_metrics.get("delivery_wall_clock_sec") is not None
        and y.primary_metrics.get("delivery_wall_clock_sec") is not None
        for x, y in solved_pairs
    )
    return dict(
        comparison_role=role,
        baseline_group=baseline[0].ablation_group.value,
        candidate_group=candidate[0].ablation_group.value,
        tier=baseline[0].tier,
        evidence=baseline[0].evidence,
        knowledge_protocols=dict(
            baseline=sorted(
                {row.environment_fingerprint.get("knowledge_mode", "unrecorded") for row in baseline}
            ),
            candidate=sorted(
                {row.environment_fingerprint.get("knowledge_mode", "unrecorded") for row in candidate}
            ),
            candidate_oracle_contexts_supplied=sum(
                bool(row.environment_fingerprint.get("oracle_context_sha256")) for row in candidate
            ),
            oracle_is_a_controlled_content_condition_not_natural_retrieval=True,
        ),
        baseline=a,
        candidate=b,
        comparison_coverage=dict(
            matched_observed_pairs=len(left),
            matched_valid_pairs=len(paired),
            excluded_invalid_pairs=len(left) - len(paired),
            mutually_solved_pairs=len(solved_pairs),
            pairs_with_recorded_delivery_times=time_observed_pairs,
            baseline=a["matrix_coverage"],
            candidate=b["matrix_coverage"],
            complete=declared_complete,
            reason=(
                "plan_inventory_unavailable"
                if not plans_available
                else "incomplete_declared_cells_or_seed_setup"
                if not declared_complete
                else None
            ),
            whole_population_reliability_complete=(
                a["reliability_coverage"]["complete_for_declared_cells"]
                and b["reliability_coverage"]["complete_for_declared_cells"]
                if plans_available
                else None
            ),
        ),
        matched_valid_pairs=len(paired),
        paired_completion_lift=paired_lift,
        wall_clock_speedup_on_mutual_success=ratio("delivery_wall_clock_sec", solved_pairs),
        mutually_solved_pairs=len(solved_pairs),
        saved_npu_test_run_fraction=saved_npu_fraction,
        # Shell-only hosts do not emit typed Read/Grep/Edit observations.
        # Preserve the required null penalty, but do not advertise asymmetric
        # observation coverage as an engineering speedup.
        localization_penalized_turns_ratio=(
            sum(x for x, _ in localization) / sum(y for _, y in localization) if localization else None
        ),
        localization_observed_pairs=len(observed_localization),
        localization_speedup=(
            sum(x for x, _ in observed_localization) / sum(y for _, y in observed_localization)
            if observed_localization and len(observed_localization) == len(paired)
            else None
        ),
        knowledge_effectiveness=dict(
            **b["knowledge_observations"]["benefit"],
            native_body_access="not_measured_by_mcp_counters",
            time_ratio_is_not_knowledge_benefit=True,
        ),
        metric_qualifications=dict(
            paired_completion_lift=dict(
                population="matched_valid_pairs",
                observed_pairs=len(paired),
                matched_observed_pairs=len(left),
                excludes_invalid_pairs=len(left) - len(paired),
                causal=False,
            ),
            wall_clock_speedup_on_mutual_success=dict(
                population="mutually_solved_pairs_with_recorded_delivery_times",
                observed_pairs=time_observed_pairs,
                mutually_solved_pairs=len(solved_pairs),
                complete_for_solved_pairs=bool(solved_pairs)
                and time_observed_pairs == len(solved_pairs),
                reason=(
                    "no_mutually_solved_pairs"
                    if not solved_pairs
                    else "delivery_timing_missing"
                    if time_observed_pairs != len(solved_pairs)
                    else "descriptive_paired_ratio_only"
                ),
                causal=False,
            ),
            saved_npu_test_run_fraction=dict(
                population="matched_valid_pairs",
                observed_pairs=sum(x is not None and y is not None for x, y in npu_values),
                definition="recognized_remote_test_command_attempts_including_failed_or_unfinished",
                is_kernel_launch_or_success_rate=False,
                causal=False,
            ),
            localization_speedup=dict(
                observed_pairs=len(observed_localization),
                matched_valid_pairs=len(paired),
                reason=(
                    "no_matched_valid_pairs"
                    if not paired
                    else "typed_localization_missing"
                    if len(observed_localization) != len(paired)
                    else None
                ),
                missing_typed_access_does_not_prove_failure=True,
            ),
            pass_pow_k=dict(
                population="eligible_observed_tasks_only",
                baseline_eligible_tasks=a["reliability_eligible_tasks"],
                baseline_observed_tasks=a["reliability_total_tasks"],
                candidate_eligible_tasks=b["reliability_eligible_tasks"],
                candidate_observed_tasks=b["reliability_total_tasks"],
                complete_declared_matrix=declared_complete,
                is_whole_matrix_reliability=bool(declared_complete)
                and a["reliability_coverage"]["complete_for_observed_tasks"]
                and b["reliability_coverage"]["complete_for_observed_tasks"],
                reason=(
                    "plan_inventory_unavailable"
                    if not plans_available
                    else "incomplete_declared_cells_or_seed_setup"
                    if not declared_complete
                    else "insufficient_valid_independent_repetitions"
                    if not (
                        a["reliability_coverage"]["complete_for_observed_tasks"]
                        and b["reliability_coverage"]["complete_for_observed_tasks"]
                    )
                    else None
                ),
            ),
        ),
        funnel=funnel,
        funnel_measurement_bases=funnel_bases,
        five_component_attribution=dict(
            baseline=summarize_five_components(baseline),
            candidate=summarize_five_components(candidate),
        ),
        amortized_k_projection=_amortized_k_projection(a, b),
    )


def _amortized_k_projection(baseline: dict, candidate: dict) -> dict:
    """Project 1-Seed-to-K-Eval amortized flywheel savings across K in {1, 3, 5, 10}."""
    base_attempts = baseline.get("attempts") or 0
    cand_attempts = candidate.get("attempts") or 0
    base_eval_tokens = (
        sum(baseline["observed_tokens_lower_bound"].values()) / base_attempts
        if base_attempts and baseline.get("token_metering_complete")
        else None
    )
    cand_eval_tokens = (
        sum(candidate["observed_tokens_lower_bound"].values()) / cand_attempts
        if cand_attempts and candidate.get("token_metering_complete")
        else None
    )
    cand_seed_tokens = (
        (
            sum(candidate["observed_tokens_including_seed_lower_bound"].values())
            - sum(candidate["observed_tokens_lower_bound"].values())
        )
        / max(1, candidate.get("seed_setup_lineages_observed", 1))
        if candidate.get("seed_setup_required")
        and candidate.get("token_metering_including_seed_complete")
        else 0.0
        if not candidate.get("seed_setup_required")
        else None
    )
    base_eval_sec = (
        baseline["observed_delivery_wall_clock_sec_lower_bound"] / base_attempts
        if base_attempts and baseline.get("delivery_timing_complete")
        else None
    )
    cand_eval_sec = (
        candidate["observed_delivery_wall_clock_sec_lower_bound"] / cand_attempts
        if cand_attempts and candidate.get("delivery_timing_complete")
        else None
    )
    cand_seed_sec = (
        (
            candidate["observed_delivery_sec_including_seed_lower_bound"]
            - candidate["observed_delivery_wall_clock_sec_lower_bound"]
        )
        / max(1, candidate.get("seed_setup_lineages_observed", 1))
        if candidate.get("seed_setup_required")
        and candidate.get("delivery_timing_including_seed_complete")
        else 0.0
        if not candidate.get("seed_setup_required")
        else None
    )
    projections = {}
    for proj_k in (1, 3, 5, 10):
        tok_ok = all(v is not None for v in (base_eval_tokens, cand_eval_tokens, cand_seed_tokens))
        sec_ok = all(v is not None for v in (base_eval_sec, cand_eval_sec, cand_seed_sec))
        base_k_tok = proj_k * base_eval_tokens if tok_ok else None
        cand_k_tok = (cand_seed_tokens + proj_k * cand_eval_tokens) if tok_ok else None
        base_k_sec = proj_k * base_eval_sec if sec_ok else None
        cand_k_sec = (cand_seed_sec + proj_k * cand_eval_sec) if sec_ok else None
        projections[str(proj_k)] = dict(
            k=proj_k,
            baseline_tokens_total=base_k_tok,
            candidate_tokens_including_seed_total=cand_k_tok,
            net_token_savings=(base_k_tok - cand_k_tok) if tok_ok else None,
            amortized_candidate_tokens_per_eval=(cand_k_tok / proj_k) if tok_ok else None,
            baseline_delivery_sec_total=base_k_sec,
            candidate_delivery_sec_including_seed_total=cand_k_sec,
            net_delivery_sec_savings=(base_k_sec - cand_k_sec) if sec_ok else None,
            amortized_candidate_delivery_sec_per_eval=(cand_k_sec / proj_k) if sec_ok else None,
        )
    break_even_k_tokens = (
        cand_seed_tokens / (base_eval_tokens - cand_eval_tokens)
        if all(v is not None for v in (base_eval_tokens, cand_eval_tokens, cand_seed_tokens))
        and (base_eval_tokens - cand_eval_tokens) > 0
        else None
    )
    break_even_k_sec = (
        cand_seed_sec / (base_eval_sec - cand_eval_sec)
        if all(v is not None for v in (base_eval_sec, cand_eval_sec, cand_seed_sec))
        and (base_eval_sec - cand_eval_sec) > 0
        else None
    )
    return dict(
        baseline_eval_tokens_mean=base_eval_tokens,
        candidate_eval_tokens_mean=cand_eval_tokens,
        candidate_seed_tokens_per_lineage=cand_seed_tokens,
        baseline_eval_delivery_sec_mean=base_eval_sec,
        candidate_eval_delivery_sec_mean=cand_eval_sec,
        candidate_seed_delivery_sec_per_lineage=cand_seed_sec,
        break_even_k_tokens=break_even_k_tokens,
        break_even_k_delivery_sec=break_even_k_sec,
        by_k=projections,
    )


def markdown(result: dict) -> str:
    def fmt(value):
        if type(value) is bool:
            return str(value).lower()
        return "unmeasured" if value is None else f"{value:.4g}"

    rows = [
        "# MindIE Bench comparison",
        "",
        f"Evidence: **{result['tier']} / {result['evidence']}**. "
        "Replay and synthetic results are not live-NPU acceptance.",
        f"Matched task role: `{result['comparison_role']}`.",
        "",
        "| Metric | Baseline | Candidate |",
        "| --- | ---: | ---: |",
    ]
    for field in (
        "attempts",
        "valid_attempts",
        "coverage",
        "pass_at_1",
        "pass_pow_k",
        "mean_wall_clock_sec",
        "mean_delivery_wall_clock_sec",
        "cost_per_resolved_task",
    ):
        rows.append(
            f"| {field} | {fmt(result['baseline'][field])} | {fmt(result['candidate'][field])} |"
        )
    if any(result[side].get("inference") for side in ("baseline", "candidate")):
        rows += ["", "## Native inference lifecycle", "",
                 "Track scores require complete declared cells. Observed scores retain partial coverage.",
                 "", "| Side / group | Track | Score | Observed | Tasks with evidence / planned |",
                 "| --- | --- | ---: | ---: | ---: |"]
        for side in ("baseline", "candidate"):
            for cohort in (result[side].get("inference") or {}).get("cohorts", []):
                for track, values in cohort["tracks"].items():
                    rows.append(
                        f"| {side} / {cohort['group']} | {track} | {fmt(values['score'])} | "
                        f"{fmt(values['observed_score'])} | {values['tasks_with_valid_evidence']} / "
                        f"{values['tasks']} |"
                    )
    rows += [
        "", "## Budget observations", "",
        "Business completion and measured performance are retained when a budget is reached or exceeded.",
        "Agent-limit stops do not imply that the stopped artifact failed verification.", "",
        "| Observation | Baseline | Candidate |", "| --- | ---: | ---: |",
    ]
    for field in (
        "agent_budget_exhausted_attempts", "completed_pass_at_agent_limit",
        "delivery_budget_observed_attempts", "delivery_budget_exceeded_attempts",
        "completed_pass_over_delivery_budget", "pass_at_1_within_delivery_budget",
    ):
        rows.append(
            f"| {field} | {fmt(result['baseline']['budget'][field])} | "
            f"{fmt(result['candidate']['budget'][field])} |"
        )
    rows += [
        "",
        "Timing means above use valid completed-pass/completed-fail attempts with recorded times; "
        "unknown and infrastructure outcomes are excluded from those means.",
        "Pass^k above covers eligible observed tasks only. "
        f"Eligible/observed tasks: baseline {result['baseline']['reliability_eligible_tasks']}/"
        f"{result['baseline']['reliability_total_tasks']}; candidate "
        f"{result['candidate']['reliability_eligible_tasks']}/"
        f"{result['candidate']['reliability_total_tasks']}. "
        "A partially eligible subset is not whole-matrix reliability.",
        "",
        "## Recorded elapsed time across all attempts",
        "",
        "These means include recorded failed/unknown/infra attempts, not successful delivery.",
        "",
        "| Measurement | Baseline | Candidate |",
        "| --- | ---: | ---: |",
    ]
    for field in (
        "mean_wall_clock_coverage",
        "mean_delivery_wall_clock_count",
        "all_attempt_recorded_wall_clock_sec_mean",
        "all_attempt_recorded_wall_clock_sec_count",
        "all_attempt_recorded_delivery_wall_clock_sec_mean",
        "all_attempt_recorded_delivery_wall_clock_sec_count",
    ):
        rows.append(
            f"| {field} | {fmt(result['baseline'][field])} | {fmt(result['candidate'][field])} |"
        )
    rows += [
        "",
        "## Coverage and missing observations",
        "",
        f"Declared matrix coverage: `{result['comparison_coverage']}`.",
        "",
        "| Observation | Baseline measured/attempts | Candidate measured/attempts |",
        "| --- | ---: | ---: |",
    ]
    for name in (
        "tokens",
        "localization",
        "remote_non_business_errors",
        "knowledge_queries",
        "knowledge_explains",
    ):
        values = [
            side["measurement_coverage"][name] for side in (result["baseline"], result["candidate"])
        ]
        rows.append(
            f"| {name} | {values[0]['observed_attempts']}/{values[0]['population_attempts']} "
            f"({values[0]['state']}) | {values[1]['observed_attempts']}/"
            f"{values[1]['population_attempts']} ({values[1]['state']}) |"
        )
    rows += [
        "",
        f"Knowledge effectiveness: `{result['knowledge_effectiveness']}`. "
        "A time ratio is not evidence of knowledge benefit. "
        "MCP count coverage does not measure native corpus-body reads.",
    ]
    rows += [
        "",
        "## Resources including required G3 seed setup",
        "",
        "| Measurement | Baseline | Candidate |",
        "| --- | ---: | ---: |",
    ]
    for field in (
        "seed_setup_required",
        "seed_setup_attempts",
        "seed_setup_lineages_required",
        "seed_setup_lineages_observed",
        "seed_setup_receipts_complete",
        "seed_setup_complete",
        "observed_cost_usd_including_seed_lower_bound",
        "cost_including_seed_complete",
        "cost_per_resolved_task_including_seed",
        "observed_delivery_sec_including_seed_lower_bound",
        "delivery_timing_including_seed_complete",
        "token_metering_including_seed_complete",
    ):
        rows.append(
            f"| {field} | {fmt(result['baseline'][field])} | {fmt(result['candidate'][field])} |"
        )
    rows += [
        "",
        "Observed token lower bounds including required seed setup: "
        f"baseline `{result['baseline']['observed_tokens_including_seed_lower_bound']}`; "
        f"candidate `{result['candidate']['observed_tokens_including_seed_lower_bound']}`.",
        "Seed coverage requires a captured seed in each selected cluster/repetition; "
        "complete run manifests or frozen receipt hashes must still detect missing individual cells. "
        "Resource lower bounds include observed failed and unknown attempts. "
        "Missing setup receipts withhold completeness, never establish zero setup cost.",
        "",
        f"All baseline terminal counts: `{result['baseline']['statuses']}`.",
        f"All candidate terminal counts: `{result['candidate']['statuses']}`.",
        "",
        f"Matched valid pairs: {result['matched_valid_pairs']}; "
        f"paired completion lift: {fmt(result['paired_completion_lift'])}.",
        f"Selected-role time speedup on mutually solved pairs (seed setup listed separately): "
        f"{fmt(result['wall_clock_speedup_on_mutual_success'])}; "
        f"pairs: {result['mutually_solved_pairs']}.",
        f"Saved NPU test-run fraction: {fmt(result['saved_npu_test_run_fraction'])}.",
        f"Localization speedup with complete typed observations: "
        f"{fmt(result['localization_speedup'])}; "
        f"observed pairs: {result['localization_observed_pairs']}/{result['matched_valid_pairs']}.",
        f"Descriptive localization-turn ratio (missing penalized by max_turns): "
        f"{fmt(result['localization_penalized_turns_ratio'])}. "
        "A missing typed observation can reflect shell-only source access; "
        "it is not proof that the agent failed to locate the target.",
        "",
        "Pass^k is a per-task plug-in estimate requiring at least k valid repetitions; "
        "it is not an observed guarantee. Cost includes all attempts; incomplete metering "
        "withholds cost per resolved task.",
        "",
        "## Effort breakdown",
        "",
        "| Measurement | Baseline mean | Candidate mean |",
        "| --- | ---: | ---: |",
    ]
    for field, value in result["funnel"].items():
        rows.append(
            f"| {field} | {fmt(value['baseline_mean']) if value else 'unmeasured'} | "
            f"{fmt(value['candidate_mean']) if value else 'unmeasured'} |"
        )
    if "amortized_k_projection" in result:
        proj = result["amortized_k_projection"]
        rows += [
            "",
            "## Amortized flywheel projection (1 Seed -> K Eval)",
            "",
            f"Break-even K (tokens): {fmt(proj['break_even_k_tokens'])}; "
            f"break-even K (delivery sec): {fmt(proj['break_even_k_delivery_sec'])}.",
            "",
            "| K | Baseline tokens | Candidate tokens (incl. 1 seed) | Net token savings | Net delivery sec savings |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for k_key, k_row in proj["by_k"].items():
            rows.append(
                f"| {k_key} | {fmt(k_row['baseline_tokens_total'])} | "
                f"{fmt(k_row['candidate_tokens_including_seed_total'])} | "
                f"{fmt(k_row['net_token_savings'])} | "
                f"{fmt(k_row['net_delivery_sec_savings'])} |"
            )
    return "\n".join(rows) + "\n"


def freeze(receipts: list[TrialReceipt], destination: Path):
    if any(r.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN} for r in receipts):
        raise ValueError("cannot freeze unresolved infrastructure or unknown attempts as a baseline")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for receipt in receipts:
        data = receipt.model_dump(mode="json")
        path = destination / receipt.trial_id / "trial_receipt.json"
        write_json(path, data, exclusive=True)
        hashes[receipt.trial_id] = digest(data)
    write_json(
        destination / "baseline_manifest.json",
        dict(schema_version="1.0", receipt_sha256=hashes),
        exclusive=True,
    )


def grouped_summaries(receipts, *, expected_cells=None, blocked_cells=None):
    groups = defaultdict(list)
    for receipt in receipts:
        key = (receipt.tier, receipt.evidence, receipt.ablation_group.value)
        groups[key + (("seed",) if receipt.task_role == "seed" else ())].append(receipt)
    result = {
        "/".join(key): dict(
            **summarize(
                rows,
                expected_cells=scope_cells(
                    expected_cells, key[0], key[2], rows[0].task_role,
                    evidence=key[1], receipts=receipts,
                ),
                blocked_cells=scope_cells(
                    blocked_cells, key[0], key[2], rows[0].task_role,
                    evidence=key[1], receipts=receipts,
                ),
            ),
            five_component_attribution=summarize_five_components(rows),
        )
        for key, rows in groups.items()
    }
    for key, rows in groups.items():
        if len(key) == 3 and key[2] == "G3" and key + ("seed",) in groups:
            seed = summarize(groups[key + ("seed",)])
            evaluation = result["/".join(key)]
            evaluation["seed_setup_attempts"] = seed["attempts"]
            evaluation["seed_setup_cost_usd_lower_bound"] = seed["observed_cost_usd_lower_bound"]
            evaluation["observed_tokens_including_seed_lower_bound"] = {
                field: value + seed["observed_tokens_lower_bound"][field]
                for field, value in evaluation["observed_tokens_lower_bound"].items()
            }
            evaluation["token_metering_including_seed_complete"] = (
                evaluation["token_metering_complete"] and seed["token_metering_complete"]
            )
            evaluation["observed_delivery_sec_including_seed_lower_bound"] = (
                evaluation["observed_delivery_wall_clock_sec_lower_bound"]
                + seed["observed_delivery_wall_clock_sec_lower_bound"]
            )
            evaluation["delivery_timing_including_seed_complete"] = (
                evaluation["delivery_timing_complete"] and seed["delivery_timing_complete"]
            )
            evaluation["cost_per_resolved_task_including_seed"] = (
                (evaluation["observed_cost_usd_lower_bound"] + seed["observed_cost_usd_lower_bound"])
                / evaluation["completed_pass"]
                if evaluation["cost_complete"] and seed["cost_complete"] and evaluation["completed_pass"]
                else None
            )
    return result
