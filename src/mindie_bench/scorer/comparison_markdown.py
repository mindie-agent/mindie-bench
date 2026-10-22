"""Markdown rendering and amortized-K projection for trial comparisons.

[Role]
Renders the paired baseline/candidate comparison report and projects 1 Seed ->
K Eval token/delivery savings without executing any workload.
[Invariants]
- Missing values render as `unmeasured`, never as zero; budget and coverage
  caveats remain in the rendered report.
- The projection is explicitly an amortized estimate, not a measured outcome.
[Timeouts & Locks]
- Pure in-memory formatting and arithmetic; no waits, subprocesses or locks.
[Side-Effect Policy]
- Returns a string/dict only; performs no file, network, or process mutation.
"""

from __future__ import annotations


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


