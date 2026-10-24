"""Compare matched trials only; incompatible frozen baselines are rejected.

[Role]
Loads and integrity-checks `trial_receipt.json` sets (`load_receipts`), builds
paired baseline/candidate comparisons over the same matrix cells (`comparison`),
renders the markdown report (`markdown`), freezes verified baselines (`freeze`),
and groups receipt summaries for the HF exporter (`grouped_summaries`).

[Invariants]
- A receipt set is compared only when trial IDs and (instance, tier, group,
  repetition) cells are unique; frozen baselines must match their SHA-256
  manifest or the comparison fails visibly.
- An interrupted attempt without a terminal receipt stays UNKNOWN and is never
  replayed; missing cells are reported, never filled with zero.

[Timeouts & Locks]
- Pure in-memory/file work; this module owns no subprocess, network call, lock
  or wall-clock deadline. Callers bound the surrounding capture/verification.

[Side-Effect Policy]
- Reads receipts; `freeze` creates a new baseline directory and refuses to
  overwrite it. This module never executes a benchmark or publishes evidence.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

from mindie_bench.evaluation.accounting import scope_cells
from mindie_bench.io import digest, read_json, write_json
from mindie_bench.models import TerminalStatus, TrialReceipt
from mindie_bench.scorer.comparison_markdown import (  # noqa: F401
    _amortized_k_projection,
    markdown,
)
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
        if len(key) == 3 and key[2] in {"G3", "G4"} and key + ("seed",) in groups:
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
            priced = [
                evaluation["observed_cost_usd_lower_bound"],
                seed["observed_cost_usd_lower_bound"],
            ]
            evaluation["cost_per_resolved_task_including_seed"] = (
                sum(priced) / evaluation["completed_pass"]
                if evaluation["cost_complete"]
                and seed["cost_complete"]
                and evaluation["completed_pass"]
                and all(value is not None for value in priced)
                else None
            )
    return result
