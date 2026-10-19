"""Collect repeatable product-value experiments for a formal release.

[Role] Bind a stable task matrix, measured receipts and component scenarios.
[Invariants] Missing cells block publication; negative results remain results.
Comparisons keep models, tasks and controls matched and include seed costs.
[Timeouts & Locks] Local file reads only; no subprocesses, retries or locks.
[Side-Effect Policy] Writes allowlisted public measurements, never native logs.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from mindie_bench.io import digest, read_json, write_json
from mindie_bench.models import TerminalStatus, TrialReceipt
from mindie_bench.scorer.comparator import CONTROL_KEYS, comparison
from mindie_bench.scorer.metrics import summarize
from mindie_bench.value_controls import require_retest_controls
from mindie_bench.value_controls import source_matches as _source_matches
from mindie_bench.value_resources import (
    TOKEN_FIELDS,
    project_component_case,
    project_setup_report,
    resource_totals,
    setup_totals,
)

VALID = {TerminalStatus.PASS, TerminalStatus.FAIL}
METRICS = {
    "reward",
    "resolved",
    "tokens",
    "total_turns",
    "total_wall_clock_sec",
    "delivery_wall_clock_sec",
    "estimated_cost_usd",
    "cost_complete",
    "npu_test_runs",
    "npu_execution_wait_sec",
    "knowledge_query_sec",
    "llm_and_exploration_sec",
    "knowledge_measurement_basis",
    "turns_to_first_localization",
    "localization_penalty_turns",
    "time_to_first_localization_sec",
    "time_to_first_target_edit_sec",
    "zero_human_intervention",
    "foreground_knowledge_queries_count",
    "foreground_knowledge_explains_count",
}


def public_receipt(row: TrialReceipt) -> dict:
    """Keep the measurement contract while removing commands, endpoints and paths."""
    source = row.environment_fingerprint
    fingerprint = {key: source[key] for key in CONTROL_KEYS if key in source}
    for key in ("provider_url", "remote_environment"):
        if key in fingerprint:
            fingerprint[key] = {"private_control_identity": digest(fingerprint[key])}
    for key in (
        "components", "component_revisions", "knowledge_mode", "skills_sha256", "corpus_view_sha256"
    ):
        if key in source:
            fingerprint[key] = source[key]
    capture = row.auxiliary_metrics.get("capture") or {}
    auxiliary = {
        "capture_wall_clock_sec": row.auxiliary_metrics.get("capture_wall_clock_sec"),
        "captured_corpus_sha256": row.auxiliary_metrics.get("captured_corpus_sha256"),
        "capture": {
            "status": capture.get("status"),
            "corpus_lint": {"status": capture.get("corpus_lint", {}).get("status")},
            "publication": {"revision": capture.get("publication", {}).get("revision")},
            "consumer": {"revision": capture.get("consumer", {}).get("revision")},
        },
        "knowledge_measurement": {
            key: value for key, value in row.auxiliary_metrics.get("knowledge_measurement", {}).items()
            if key in {"query_count", "explain_count", "successful_calls", "failed_calls",
                       "timing_complete", "basis"}
        },
    }
    return row.model_copy(
        update={
            "environment_fingerprint": fingerprint,
            "primary_metrics": {
                key: value for key, value in row.primary_metrics.items() if key in METRICS
            },
            "auxiliary_metrics": auxiliary,
            "failure_reason": "details_retained_in_private_evidence" if row.failure_reason else None,
            "errors": [{"stage": "trial", "message": "details retained in private evidence"}]
            if row.errors
            else [],
            "artifacts": {},
        }
    ).model_dump(mode="json")


def _cell(row):
    return row.instance_id, row.ablation_group.value, row.repetition


def _expected(study):
    if (
        study["repetitions"] < 1
        or not study["tasks"]
        or len(set(study["groups"])) != len(study["groups"])
    ):
        raise ValueError("a value study needs distinct groups, tasks and repetitions")
    return set(itertools.product(study["tasks"], study["groups"], range(1, study["repetitions"] + 1)))


def _selected_summary(summary):
    summary["selected_valid_success_rate"] = summary.pop("pass_at_1")
    summary["selected_cell_success_rate"] = summary.pop("all_attempt_success_rate")
    for key in list(summary):
        if key.startswith(("pass_pow_", "reliability_")):
            summary.pop(key)
    summary["population"] = "explicitly selected settled trials; historical attempts reported separately"
    return summary


def _transfer_problems(study, rows, lineages):
    contract = study.get("transfer")
    if not contract:
        return []
    by_id = {row.trial_id: row for row in rows}
    observed = {item["repetition"]: item for item in lineages}
    if len(observed) != len(lineages):
        raise ValueError("duplicate transfer lineage repetition")
    problems = []
    for rep in range(1, study["repetitions"] + 1):
        item = observed.get(rep, {})
        seed = by_id.get(item.get("seed_trial_id"))
        evaluations = item.get("evals", [])
        if (
            seed is None or seed.instance_id != contract["seed_task"]
            or seed.repetition != rep or seed.ablation_group.value != contract["group"]
            or {row["task_id"] for row in evaluations} != set(contract["eval_tasks"])
            or len(evaluations) != len(contract["eval_tasks"])
        ):
            problems.append(f"transfer_lineage_incomplete:{rep}")
            continue
        captured = seed.auxiliary_metrics.get("captured_corpus_sha256")
        if (
            not captured or captured != item.get("seed_captured_corpus_identity")
            or seed.auxiliary_metrics.get("capture", {}).get("corpus_lint", {}).get("status")
            != "passed"
        ):
            problems.append(f"transfer_seed_capture_incompatible:{rep}")
        for entry in evaluations:
            row = by_id.get(entry["trial_id"])
            if (
                row is None or row.instance_id != entry["task_id"] or row.repetition != rep
                or row.ablation_group.value != contract["group"]
                or row.environment_fingerprint.get("corpus_view_sha256") != captured
                or entry.get("initial_corpus_identity") != captured
                or type(entry.get("seed_query_hit_observed")) is not bool
                or type(entry.get("seed_explain_read_observed")) is not bool
                or (
                    entry["seed_query_hit_observed"] and (
                        not entry.get("native_query_call_ids")
                        or not row.primary_metrics.get("foreground_knowledge_queries_count", 0)
                    )
                )
                or (
                    entry["seed_explain_read_observed"] and (
                        not entry.get("native_explain_call_ids")
                        or not row.primary_metrics.get("foreground_knowledge_explains_count", 0)
                    )
                )
                or entry.get("time_order_passed") is not True
                or not isinstance(item.get("seed_capture_last_model_end_monotonic_sec"), (int, float))
                or not isinstance(entry.get("eval_first_model_start_monotonic_sec"), (int, float))
                or item["seed_capture_last_model_end_monotonic_sec"]
                >= entry["eval_first_model_start_monotonic_sec"]
            ):
                problems.append(f"transfer_reuse_unproven:{rep}:{entry['task_id']}")
    return problems


def _study(study, rows, sources, runner_patches, source_patches, excluded):
    expected = _expected(study)
    observed = Counter(_cell(row) for row in rows)
    problems = []
    for cell in sorted(expected - observed.keys()):
        problems.append(f"missing:{cell[0]}:{cell[1]}:{cell[2]}")
    for cell, count in observed.items():
        if cell not in expected or count != 1:
            problems.append(f"unexpected_or_duplicate:{cell[0]}:{cell[1]}:{cell[2]}")
    for row in rows:
        env = row.environment_fingerprint
        if (
            row.tier != study["tier"]
            or row.evidence != "verified"
            or env.get("harness_name") != study["harness"]
            or env.get("base_model") != study["model"]
            or env.get("reasoning_effort") != study.get("reasoning_effort")
            or row.task_role != study["tasks"].get(row.instance_id)
            or row.task_digest != study["task_digests"].get(row.instance_id)
        ):
            problems.append(f"identity_mismatch:{row.trial_id}")
        if row.terminal_status not in VALID or row.errors:
            problems.append(f"unsettled_trial:{row.trial_id}")
        if row.primary_metrics.get("tokens", {}).get("complete") is not True:
            problems.append(f"token_metering_incomplete:{row.trial_id}")
        if row.primary_metrics.get("delivery_wall_clock_sec") is None:
            problems.append(f"delivery_timing_incomplete:{row.trial_id}")
        group = row.ablation_group.value
        required_components = study["group_components"].get(group, [])
        if set(env.get("components", [])) != set(required_components):
            problems.append(f"ablation_components_mismatch:{row.trial_id}")
        if group in {"G2", "G3"} and not env.get("skills_sha256"):
            problems.append(f"plugin_skills_missing:{row.trial_id}")
        for component in [*required_components, study["harness"]]:
            if not _source_matches(
                component, env.get("component_revisions", {}).get(component), sources.get(component),
                group, source_patches,
            ):
                problems.append(f"source_mismatch:{row.trial_id}:{component}")
        if group == "G3":
            if row.auxiliary_metrics.get("capture", {}).get("status") != "completed":
                problems.append(f"capture_incomplete:{row.trial_id}")
    groups = {
        group: [row for row in rows if row.ablation_group.value == group] for group in study["groups"]
    }
    historical = [TrialReceipt.model_validate(item["receipt"]) for item in excluded]
    first_attempts = {row.trial_id: row.terminal_status.value for row in rows}
    for item in excluded:
        replacement = item.get("replacement_trial_id")
        if replacement:
            first_attempts[replacement] = item["receipt"]["terminal_status"]
    comparisons = []
    inventory = [
        {
            "instance_id": task,
            "tier": study["tier"],
            "group": group,
            "repetition": rep,
            "role": study["tasks"][task],
            "task_digest": study["task_digests"][task],
        }
        for task, group, rep in sorted(expected)
    ]
    for baseline, candidate in study["comparisons"]:
        if not groups[baseline] or not groups[candidate]:
            continue
        compared = [groups[baseline], groups[candidate]]
        applied_patches = []
        # A targeted infrastructure repair need not discard completed experiments.
        # Original observations remain unchanged; only reviewed runner identities
        # are treated as equivalent for the paired control check.
        for patch in runner_patches:
            before, after = patch["before"], patch["after"]
            if not patch["reason"] or before == after:
                raise ValueError("runner patch needs distinct identities and a reason")
            if any(
                row.environment_fingerprint.get("benchmark_source_sha256") == before
                for batch in compared
                for row in batch
            ):
                compared = [
                    [
                        row.model_copy(
                            update={
                                "environment_fingerprint": {
                                    **row.environment_fingerprint,
                                    "benchmark_source_sha256": after,
                                }
                            }
                        )
                        if row.environment_fingerprint.get("benchmark_source_sha256") == before
                        else row
                        for row in batch
                    ]
                    for batch in compared
                ]
                applied_patches.append(patch)
        try:
            result = comparison(
                *compared,
                expected_cells=inventory,
            )
        except ValueError as exc:
            problems.append(f"comparison_incompatible:{baseline}:{candidate}:{exc}")
        else:
            if not result["comparison_coverage"]["complete"]:
                problems.append(f"comparison_coverage_incomplete:{baseline}:{candidate}")
            result.pop("amortized_k_projection", None)
            result.pop("five_component_attribution", None)
            for side in ("baseline", "candidate"):
                result[side] = _selected_summary(result[side])
            if any(item.get("replacement_trial_id") for item in excluded):
                result["comparison_coverage"]["whole_population_reliability_complete"] = False
            result["attempt_scope"] = "selected outcomes after explicit infrastructure repairs"
            result["reviewed_runner_patches"] = applied_patches
            comparisons.append(result)
    return {
        "id": study["id"],
        "scope": study["scope"],
        "harness": study["harness"],
        "model": study["model"],
        "expected_cells": len(expected),
        "observed_cells": len(rows),
        "statuses": dict(Counter(row.terminal_status.value for row in rows)),
        "first_attempt_statuses": dict(Counter(first_attempts.values())),
        "recovered_cells": [
            {
                "trial_id": item["replacement_trial_id"],
                "original_trial_id": item["receipt"]["trial_id"],
                "original_status": item["receipt"]["terminal_status"],
                "reason": item["reason"],
            }
            for item in excluded if item.get("replacement_trial_id")
        ],
        "groups": {
            group: {**_selected_summary(summarize(items)), **resource_totals(items)}
            for group, items in groups.items()
        },
        "resources": {
            name: resource_totals(items)
            for name, items in (
                ("selected", rows), ("historical", historical),
                ("all_executed", [*rows, *historical]),
            )
        },
        "resources_by_group": {
            group: resource_totals(
                items + [row for row in historical if row.ablation_group.value == group]
            )
            for group, items in groups.items()
        },
        "comparisons": comparisons,
        "blocking_reasons": problems,
    }


def evaluate(suite: dict, evidence: dict) -> dict:
    """Recompute the gate from public observations, including in release CI."""
    if suite["schema_version"] != "mindie-value-suite/1":
        raise ValueError("unsupported product-value suite")
    if evidence["schema_version"] != "mindie-value-evidence/1":
        raise ValueError("unsupported product-value evidence")
    if evidence["suite_identity"] != digest(suite):
        raise ValueError("evidence belongs to a different workload suite")
    ids = [study["id"] for study in suite["studies"]]
    if len(ids) != len(set(ids)) or set(evidence["studies"]) != set(ids):
        raise ValueError("release must contain exactly the declared studies")
    excluded = [
        {**item, "receipt": TrialReceipt.model_validate(item["receipt"]).model_dump(mode="json")}
        for item in evidence.get("excluded_attempts", [])
    ]
    reports, problems, trial_ids = [], [], set()
    selected = {
        raw["trial_id"]: raw
        for rows in evidence["studies"].values()
        for raw in rows
    }
    replacements = set()
    for item in excluded:
        original = item["receipt"]
        identity, replacement = original["trial_id"], item.get("replacement_trial_id")
        if not item["reason"] or item["study_id"] not in ids or identity in trial_ids:
            raise ValueError("historical attempts need a study, a reason and a unique trial ID")
        if identity in selected:
            raise ValueError("a historical attempt cannot also be scored")
        trial_ids.add(identity)
        if replacement:
            candidate = selected.get(replacement)
            if candidate is None or replacement in replacements:
                raise ValueError("a retest must replace one original attempt with one selected trial")
            fields = ("instance_id", "tier", "ablation_group", "repetition", "task_digest")
            if any(original[field] != candidate[field] for field in fields):
                raise ValueError("a retest must retain the original task and cell")
            if candidate not in evidence["studies"][item["study_id"]]:
                raise ValueError("a retest must remain in the same study")
            require_retest_controls(original, candidate, evidence)
            startup = item.get("startup_observation", {})
            zero_model_startup = (
                item.get("classification") == "native_startup_infrastructure"
                and startup.get("journal_records", 0) > 0
                and startup.get("broker_identity")
                and startup.get("model_requests") == 0
                and original["primary_metrics"].get("total_turns") == 0
                and all(
                    original["primary_metrics"].get("tokens", {}).get(key) == 0
                    for key in TOKEN_FIELDS
                )
            )
            if original["terminal_status"] in {status.value for status in VALID} and not zero_model_startup:
                raise ValueError("valid task outcomes cannot be replaced by a favorable retest")
            replacements.add(replacement)
    for study in suite["studies"]:
        rows = [TrialReceipt.model_validate(raw) for raw in evidence["studies"][study["id"]]]
        for row in rows:
            if row.trial_id in trial_ids:
                raise ValueError(f"trial counted more than once: {row.trial_id}")
            trial_ids.add(row.trial_id)
        result = _study(
            study, rows, evidence["component_revisions"], evidence.get("runner_patches", []),
            evidence.get("source_patches", []),
            [item for item in excluded if item["study_id"] == study["id"]],
        )
        reports.append(result)
        transfer = evidence.get("transfer_lineages", {}).get(study["id"], [])
        result["transfer_lineages"] = transfer
        result["blocking_reasons"].extend(_transfer_problems(study, rows, transfer))
        problems.extend(f"{study['id']}:{reason}" for reason in result["blocking_reasons"])
    cases = {case["id"]: case for case in evidence["component_cases"]}
    if len(cases) != len(evidence["component_cases"]):
        raise ValueError("duplicate component value case")
    for required in suite["component_cases"]:
        case = cases.get(required["id"])
        if case is None:
            problems.append(f"component_missing:{required['id']}")
            continue
        samples = case["measurements"]
        if (
            case["status"] != "pass"
            or len(samples) != required["repetitions"]
            or {sample["repetition"] for sample in samples} != set(range(1, required["repetitions"] + 1))
            or any(sample["status"] != "pass" for sample in samples)
        ):
            problems.append(f"component_incomplete:{required['id']}")
        for sample in samples:
            for side, expected in required.get("expected_observations", {}).items():
                if any(
                    sample.get(side, {}).get(key) != value
                    for key, value in expected.items()
                ):
                    problems.append(f"component_observation_failed:{required['id']}:{sample['repetition']}")
        source = case["source_revisions"].get(required["distribution"], {})
        if (
            source.get("revision") != evidence["component_revisions"].get(required["component"])
            or source.get("editable") is not False
        ):
            problems.append(f"component_source_mismatch:{required['id']}")
    native = {item["plugin"]: item for item in evidence.get("native_plugins", [])}
    if len(native) != len(evidence.get("native_plugins", [])):
        raise ValueError("duplicate native plugin evidence")
    for plugin, required_checks in suite.get("native_plugins", {}).items():
        observed = native.get(plugin)
        if observed is None:
            problems.append(f"native_plugin_missing:{plugin}")
            continue
        if observed["status"] != "pass" or not _source_matches(
            plugin, observed["revision"], evidence["component_revisions"].get(plugin),
            "native", evidence.get("source_patches", []),
        ):
            problems.append(f"native_plugin_incompatible:{plugin}")
        checks = {check["name"]: check["status"] for check in observed["checks"]}
        if any(checks.get(name) != "pass" for name in required_checks):
            problems.append(f"native_plugin_checks_incomplete:{plugin}")
    setup_reports = [project_setup_report(item) for item in evidence.get("setup_reports", [])]
    return {
        "schema_version": "mindie-value-report/1",
        "product_version": evidence["product_version"],
        "suite": suite["id"],
        "component_revisions": evidence["component_revisions"],
        "limitations": [suite["comparability"]] + evidence.get("limitations", [])
        if "comparability" in suite else evidence.get("limitations", []),
        "studies": reports,
        "component_cases": evidence["component_cases"],
        "excluded_attempts": excluded,
        "native_plugins": list(native.values()),
        "source_patches": evidence.get("source_patches", []),
        "setup_reports": setup_reports,
        "setup_resources": setup_totals(setup_reports),
        "resources": resource_totals([
            TrialReceipt.model_validate(row)
            for row in [*selected.values(), *(item["receipt"] for item in excluded)]
        ]),
        "release_gate": not problems,
        "blocking_reasons": problems,
    }


def collect(*, suite: Path, manifest: Path, output: Path) -> dict:
    """Read explicit receipt paths; never discover and silently select a good retry."""
    protocol, inputs = read_json(suite), read_json(manifest)
    root = manifest.resolve().parent
    evidence = {
        "schema_version": "mindie-value-evidence/1",
        "public_projection": "Commands, endpoint addresses and native logs stay private; "
        "control identities and measured numeric observations are retained.",
        "suite_identity": digest(protocol),
        "product_version": inputs["product_version"],
        "component_revisions": inputs["component_revisions"],
        "runner_patches": inputs.get("runner_patches", []),
        "source_patches": inputs.get("source_patches", []),
        "limitations": inputs.get("limitations", []),
        "studies": {},
        "component_cases": [],
        "excluded_attempts": [],
        "native_plugins": [],
        "transfer_lineages": {},
        "setup_reports": [
            project_setup_report(read_json(root / path)) for path in inputs.get("setup_reports", [])
        ],
    }
    for identity, paths in inputs["studies"].items():
        evidence["studies"][identity] = [
            public_receipt(TrialReceipt.model_validate(read_json(root / path))) for path in paths
        ]
    for path in inputs["component_reports"]:
        for case in read_json(root / path)["cases"]:
            evidence["component_cases"].append(project_component_case(case))
    for item in inputs.get("excluded_attempts", []):
        startup = {}
        if item.get("classification") == "native_startup_infrastructure" and item.get("broker_path"):
            with (root / item["broker_path"]).open(encoding="utf-8") as stream:
                events = [json.loads(line) for line in stream if line.strip()]
            startup = {
                "broker_identity": digest(events),
                "journal_records": len(events),
                "model_requests": sum(row["kind"] == "model_start" for row in events),
            }
        evidence["excluded_attempts"].append(
            {
                "reason": item["reason"],
                "study_id": item["study_id"],
                "replacement_trial_id": item.get("replacement_trial_id"),
                "classification": item.get("classification"),
                "startup_observation": startup,
                "receipt": public_receipt(TrialReceipt.model_validate(read_json(root / item["path"]))),
            }
        )
    for path in inputs.get("native_plugins", []):
        record = read_json(root / path)
        evidence["native_plugins"].append({
            key: record[key] for key in ("plugin", "revision", "status", "limitations")
        } | {"checks": [
            {"name": check["name"], "status": check["status"]} for check in record["checks"]
        ]})
    for study_id, path in inputs.get("transfer_lineages", {}).items():
        evidence["transfer_lineages"][study_id] = [
            {
                **{key: item.get(key) for key in (
                    "repetition", "seed_trial_id", "seed_captured_corpus_identity",
                    "seed_capture_last_model_end_monotonic_sec",
                )},
                "evals": [
                    {key: entry.get(key) for key in (
                        "trial_id", "task_id", "initial_corpus_identity",
                        "seed_query_hit_observed", "seed_explain_read_observed",
                        "native_query_call_ids", "native_explain_call_ids", "time_order_passed",
                        "eval_first_model_start_monotonic_sec",
                    )}
                    for entry in item.get("evals", [])
                ],
            }
            for item in read_json(root / path)["lineages"]
        ]
    report = evaluate(protocol, evidence)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "value-suite.json", protocol)
    write_json(output / "value-evidence.json", evidence)
    write_json(output / "value-results.json", report)
    (output / "value-results.md").write_text(markdown(report), encoding="utf-8")
    return report


def verify_publication(*, suite: Path, evidence: Path, manifest: Path) -> dict:
    """Bind the measured product sources to the component release manifest."""
    report = evaluate(read_json(suite), read_json(evidence))
    targets = read_json(manifest)
    if report["product_version"] != targets["product_version"]:
        raise ValueError("value evidence and formal product version differ")
    for target in targets["repositories"]:
        component = target["repository"].split("/")[-1].removeprefix("mindie-agent-")
        component = {"cc": "claude-code"}.get(component, component)
        if report["component_revisions"].get(component) != target["revision"]:
            raise ValueError(f"value evidence and release source differ: {component}")
    return report


def markdown(report: dict) -> str:
    lines = [
        "# Product value benchmark",
        "",
        f"Version: {report['product_version']}. Suite: `{report['suite']}`. "
        f"Evidence: {'complete' if report['release_gate'] else 'incomplete'}.",
        "",
        "Each row compares the same tasks and model. Timing uses only pairs solved "
        "by both conditions; failures and uncompleted trials remain in the inventory. "
        "Different models are reported separately.",
        "",
        "| Study | Model | Observed / planned | Pass | Fail | Infrastructure | Unknown |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines[5:5] = [line for limitation in report["limitations"] for line in ("", limitation)]
    for study in report["studies"]:
        counts = study["statuses"]
        lines.append(
            f"| {study['id']} | {study['model']} | {study['observed_cells']} / "
            f"{study['expected_cells']} | {counts.get('completed_pass', 0)} | "
            f"{counts.get('completed_fail', 0)} | {counts.get('infra_failed', 0)} | "
            f"{counts.get('unknown', 0)} |"
        )
    for study in report["studies"]:
        lines += ["", f"## {study['id']}", "", study["scope"], ""]
        for pair in study["comparisons"]:
            cover = pair["comparison_coverage"]
            lines.append(
                f"- {pair['baseline_group']} to {pair['candidate_group']}: "
                f"{cover['matched_valid_pairs']} valid pairs; "
                f"{cover['mutually_solved_pairs']} solved by both."
            )
            ratio = pair["wall_clock_speedup_on_mutual_success"]
            lines.append(
                "  Paired evaluation delivery-time baseline/candidate ratio: "
                + (f"{ratio:.3f}x." if ratio is not None else "unmeasured.")
            )
        lines += [
            "",
            "| Condition | Selected + historical attempts | Selected solved | Total delivery seconds | "
            "Uncached input | Cached input | Output | Capture tokens | Metering complete |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for group, summary in study["groups"].items():
            totals = study["resources_by_group"][group]
            tokens = totals["observed_tokens_lower_bound"]
            duration = totals["observed_delivery_wall_clock_sec_lower_bound"]
            duration = f"{duration:.2f}" if duration is not None else "unmeasured"
            lines.append(
                f"| {group} | {totals['attempts']} | {summary['completed_pass']} | "
                f"{duration} | {tokens['input_uncached']} | {tokens['input_cached']} | "
                f"{tokens['output']} | {tokens['indexer_small_model_total']} | "
                f"{totals['token_metering_complete']} |"
            )
        lines += [
            "",
            "Totals include failed attempts and, for the transfer workflow, the seed "
            "task and capture. These workflow totals are distinct from evaluation-only "
            "paired timing. Incomplete totals are lower bounds. Token categories "
            "are kept separate; absent prices do not imply zero dollar cost.",
        ]
        if study["recovered_cells"]:
            lines += [
                "",
                "First-attempt outcomes: "
                + ", ".join(f"{key}={value}" for key, value in study["first_attempt_statuses"].items())
                + ". Paired timing uses the selected settled attempts; totals also include "
                "the original infrastructure failures and explicit recovery attempts.",
            ]
        if study["transfer_lineages"]:
            lines += [
                "",
                "| Repetition | Evaluation | Seed returned by query | Seed read through explain |",
                "| ---: | --- | --- | --- |",
            ]
            for lineage in study["transfer_lineages"]:
                for entry in lineage.get("evals", []):
                    lines.append(
                        f"| {lineage['repetition']} | {entry['task_id']} | "
                        f"{entry['seed_query_hit_observed']} | "
                        f"{entry['seed_explain_read_observed']} |"
                    )
            lines += [
                "",
                "A captured seed being available does not imply the agent used it. "
                "Missing retrieval is retained as an observed result.",
            ]
    for case in report["component_cases"]:
        lines += [
            "",
            f"## {case['component']}",
            "",
            case["scope"],
            "",
            f"Measured repetitions: {len(case['measurements'])}.",
            "",
        ]
        lines.extend(f"- {limitation}" for limitation in case["limitations"])
        lines += ["", "| Observation | Baseline samples | Candidate samples |", "| --- | --- | --- |"]
        keys = sorted(
            {
                key
                for sample in case["measurements"]
                for side in ("baseline", "candidate")
                for key in sample.get(side, {})
            }
        )
        for key in keys:
            values = [
                ", ".join(
                    str(sample.get(side, {}).get(key, "unmeasured")) for sample in case["measurements"]
                )
                for side in ("baseline", "candidate")
            ]
            lines.append(f"| {key} | {values[0]} | {values[1]} |")
    if report["setup_reports"]:
        lines += [
            "", "## Setup and diagnostic model usage", "",
            "These separately measured operations are additional execution costs. "
            "They are excluded from task success rates and paired task timing. "
            "Missing duration or price remains unmeasured.",
            "",
            "| Plugin | Operation | Seconds | Uncached input | Cached input | Output |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for setup in report["setup_reports"]:
            for row in setup["operations"]:
                tokens = row["tokens"]
                lines.append(
                    f"| {setup['plugin']} | {row['operation']} | "
                    f"{row['elapsed_sec'] if row['elapsed_sec'] is not None else 'unmeasured'} | "
                    f"{tokens['input_uncached']} | {tokens['input_cached']} | {tokens['output']} |"
                )
    if report["excluded_attempts"]:
        lines += [
            "",
            "## Excluded attempts",
            "",
            "These attempts remain in the evidence with their measured resource costs.",
        ]
        lines.extend(
            f"- `{item['receipt']['trial_id']}`: {item['reason']}"
            for item in report["excluded_attempts"]
        )
    if report["blocking_reasons"]:
        lines += ["", "## Missing or incompatible evidence", ""]
        lines.extend(f"- {reason}" for reason in report["blocking_reasons"])
    return "\n".join(lines) + "\n"
