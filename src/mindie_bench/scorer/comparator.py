"""Compare matched trials only; incompatible frozen baselines are rejected."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

from mindie_bench.io import digest, read_json, write_json
from mindie_bench.models import TerminalStatus, TrialReceipt
from mindie_bench.scorer.metrics import summarize

CONTROL_KEYS = (
    "host_os",
    "host_arch",
    "harness_name",
    "harness_version",
    "harness_binary_sha256",
    "base_model",
    "reasoning_effort",
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


def comparison(baseline: list[TrialReceipt], candidate: list[TrialReceipt], *, k: int = 3) -> dict:
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
    a, b = summarize(baseline, k=k), summarize(candidate, k=k)
    valid = {TerminalStatus.PASS, TerminalStatus.FAIL}
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
    for x, y in paired:
        localization.append(
            (
                x.primary_metrics.get("turns_to_first_localization")
                or x.primary_metrics["localization_penalty_turns"],
                y.primary_metrics.get("turns_to_first_localization")
                or y.primary_metrics["localization_penalty_turns"],
            )
        )
    funnel = {}
    for field in (
        "total_turns",
        "npu_execution_wait_sec",
        "knowledge_query_sec",
        "llm_and_exploration_sec",
    ):
        values = [(x.primary_metrics.get(field), y.primary_metrics.get(field)) for x, y in paired]
        funnel[field] = (
            {"baseline_mean": mean(x for x, _ in values), "candidate_mean": mean(y for _, y in values)}
            if values and all(x is not None and y is not None for x, y in values)
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
    return dict(
        baseline_group=baseline[0].ablation_group.value,
        candidate_group=candidate[0].ablation_group.value,
        tier=baseline[0].tier,
        evidence=baseline[0].evidence,
        baseline=a,
        candidate=b,
        matched_valid_pairs=len(paired),
        paired_completion_lift=paired_lift,
        wall_clock_speedup_on_mutual_success=ratio("delivery_wall_clock_sec", solved_pairs),
        mutually_solved_pairs=len(solved_pairs),
        saved_npu_test_run_fraction=saved_npu_fraction,
        localization_speedup=(
            sum(x for x, _ in localization) / sum(y for _, y in localization) if localization else None
        ),
        funnel=funnel,
    )


def markdown(result: dict) -> str:
    def fmt(value):
        return "unmeasured" if value is None else f"{value:.4g}"

    rows = [
        "# MindIE Bench comparison",
        "",
        f"Evidence: **{result['tier']} / {result['evidence']}**. "
        "Replay and synthetic results are not live-NPU acceptance.",
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
    rows += [
        "",
        f"All baseline terminal counts: `{result['baseline']['statuses']}`.",
        f"All candidate terminal counts: `{result['candidate']['statuses']}`.",
        "",
        f"Matched valid pairs: {result['matched_valid_pairs']}; "
        f"paired completion lift: {fmt(result['paired_completion_lift'])}.",
        f"Time speedup on mutually solved pairs: {fmt(result['wall_clock_speedup_on_mutual_success'])}; "
        f"pairs: {result['mutually_solved_pairs']}.",
        f"Saved NPU test-run fraction: {fmt(result['saved_npu_test_run_fraction'])}.",
        f"Localization speedup (missing penalized by max_turns): {fmt(result['localization_speedup'])}.",
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


def grouped_summaries(receipts):
    groups = defaultdict(list)
    for receipt in receipts:
        key = (receipt.tier, receipt.evidence, receipt.ablation_group.value)
        groups[key + (("seed",) if receipt.task_role == "seed" else ())].append(receipt)
    result = {"/".join(key): summarize(rows) for key, rows in groups.items()}
    for key, rows in groups.items():
        if len(key) == 3 and key[2] == "G3" and key + ("seed",) in groups:
            seed = summarize(groups[key + ("seed",)])
            evaluation = result["/".join(key)]
            evaluation["seed_setup_attempts"] = seed["attempts"]
            evaluation["seed_setup_cost_usd_lower_bound"] = seed["observed_cost_usd_lower_bound"]
            evaluation["cost_per_resolved_task_including_seed"] = (
                (evaluation["observed_cost_usd_lower_bound"] + seed["observed_cost_usd_lower_bound"])
                / evaluation["completed_pass"]
                if evaluation["cost_complete"] and seed["cost_complete"] and evaluation["completed_pass"]
                else None
            )
    return result
