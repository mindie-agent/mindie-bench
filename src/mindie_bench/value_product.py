"""Product-matrix publication evaluation for `mindie-value-suite/2`.

[Role] Bind public receipt projections to predeclared per-cohort product plans,
aggregate each with the existing product matrix contract and publish measured,
path-free results.
[Invariants] Product identity comes from `experiment_context.cell`; internal G
modes are never product labels. Each cohort keeps its own exact task budget.
Installed product dependencies, common instrumentation, native plugin and
benchmark runner are separate identities. Historical attempts keep original
costs/identities without being forced into current pins. Missing, infra and
unknown required cells remain visible. P3 prerequisite cost stays separate.
[Timeouts & Locks] Local file reads only; no subprocesses, retries or locks.
[Side-Effect Policy] Writes only allowlisted public result files.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from mindie_bench.io import digest, read_json, write_json
from mindie_bench.models import TrialReceipt
from mindie_bench.runner.matrix import (
    ProductGroup,
    ProductMatrixPlan,
    ProductObservation,
    ProductProfile,
    build_product_report,
    observation_from_receipt,
)
from mindie_bench.runner.matrix_contracts import PRODUCT_MATRIX_PROTOCOL
from mindie_bench.value_controls import project_component_runner, require_collector_source
from mindie_bench.value_product_compare import (
    VALID,
    _comparison,
    _functional_claim_status,
    _historical_record,
)
from mindie_bench.value_product_p3 import load_private_p3_claim, load_public_p3_claim
from mindie_bench.value_product_protocol import (
    COMPONENT_SET,
    EVIDENCE2,
    INPUTS2,
    REPORT2,
    SUITE2,
    _component_name,
    build_product_plan,
)
from mindie_bench.value_product_public import (
    public_product_receipt,
    public_runtime,
    public_runtime_manifest,
)
from mindie_bench.value_resources import (
    project_component_case,
    project_setup_report,
    resource_totals,
    setup_totals,
)


class _P3ReceiptUnavailable(ValueError):
    """A P3 receipt cannot be bound to admitted public lineage."""


def _cell_from_context(context: dict):
    from mindie_bench.runner.matrix import ProductCell

    raw = context.get("cell")
    if not isinstance(raw, dict):
        profile = context.get("profile") or {}
        raw = profile.get("cell") if isinstance(profile, dict) else None
    if not isinstance(raw, dict):
        return None
    return ProductCell(
        ProductGroup(raw["group"]),
        raw["profile_id"],
        raw["cohort"],
        raw["protocol"],
        raw["task_id"],
        raw["task_family"],
        raw["hardware_tier"],
        raw["repetition"],
        raw["order"],
    )


def _read_runtime_manifests(paths, root: Path, benchmark: dict) -> list[dict]:
    manifests = []
    for pointer in paths:
        path = root / pointer
        raw = read_json(path)
        manifests.append(
            public_runtime_manifest(raw, hashlib.sha256(path.read_bytes()).hexdigest(), benchmark)
        )
    return manifests


def _bind_runtime(
    manifest: dict,
    context: dict,
    profile,
    source: dict,
    benchmark: dict,
) -> None:
    if manifest["profile_id"] != profile.profile_id or manifest["group"] != profile.group.value:
        raise ValueError("runtime manifest belongs to another product profile")
    if manifest["native_config_sha256"] != context.get("native_config_sha256"):
        raise ValueError("native configuration differs between receipt and runtime manifest")
    if manifest["derived_inputs_sha256"] != context.get("derived_inputs_sha256"):
        raise ValueError("task inputs differ between receipt and runtime manifest")
    if manifest["conditions"] != profile.conditions.to_dict():
        raise ValueError("receipt conditions differ from the predeclared matching conditions")
    for name, value in profile.budget.task_derived_dict().items():
        if (manifest.get("budget") or {}).get(name) != value:
            raise ValueError("receipt budget differs from the predeclared cohort budget")
    if manifest["actual_runtime_sha256"] != digest(public_runtime(context.get("actual_runtime") or {})):
        raise ValueError("receipt runtime differs from its native runtime manifest")
    if manifest["runtime_modules"] != source.get("runtime_modules", {}):
        raise ValueError("installed product runtime modules differ from the declared source")
    expected_plugin = source.get("native_plugin")
    actual_plugin = manifest.get("native_plugin")
    for name in ("package_sha256", "host_entry_sha256", "contribution_enabled"):
        if (actual_plugin or {}).get(name) != (expected_plugin or {}).get(name):
            raise ValueError("installed native plugin differs from the declared source")
    if manifest.get("benchmark") != source.get("benchmark"):
        raise ValueError("benchmark runner differs from the declared source")
    if manifest["loaded_identity_sha256"] != source["loaded_identity_sha256"]:
        raise ValueError("loaded product identity differs from the declared source")
    if manifest["catalog_sha256"] != source["catalog_sha256"]:
        raise ValueError("loaded catalog differs from the declared product source")
    units = manifest["catalog"].get("units")
    if profile.group == ProductGroup.P0_BARE_HOST and units != 0:
        raise ValueError("P0 did not prove a loaded-empty catalog")
    if profile.group != ProductGroup.P0_BARE_HOST and (units is None or units <= 0):
        raise ValueError("P1/P2 must prove a loaded non-empty catalog")
    env = context.get("_receipt_component_revisions") or {}
    for component, module in source.get("runtime_modules", {}).items():
        if env.get(component) != module["revision"]:
            raise ValueError(f"receipt configured source differs from the runtime module: {component}")
    if expected_plugin and env.get("dsh") != expected_plugin.get("revision"):
        raise ValueError("receipt configured native plugin differs from the declared source")
    if env.get("mindie-bench") != benchmark["revision"]:
        raise ValueError("receipt configured benchmark runner differs from the common runner")


def _public_profile(profile: ProductProfile, source: dict) -> dict:
    payload = profile.to_dict()
    payload["experiment_profile_sha256"] = digest(profile.experiment_profile)
    payload.pop("experiment_profile", None)
    payload.pop("notes", None)
    payload["source"] = source
    return payload


def aggregate_product_matrix(plan: ProductMatrixPlan, observations) -> dict:
    """Single delegation to the existing product aggregation contract."""
    return build_product_report(plan, observations).to_dict()


def _component_evidence(
    suite: dict, evidence: dict, final_repositories: dict, problems: list
) -> None:
    revisions = {name: target["revision"] for name, target in final_repositories.items()}
    cases = {case["id"]: case for case in evidence.get("component_cases", [])}
    if len(cases) != len(evidence.get("component_cases", [])):
        raise ValueError("duplicate component value case")
    for required in suite.get("component_cases", []):
        case = cases.get(required["id"])
        if case is None:
            problems.append(f"component_missing:{required['id']}")
            continue
        samples = case["measurements"]
        if (
            case["status"] != "pass"
            or len(samples) != required["repetitions"]
            or {sample["repetition"] for sample in samples}
            != set(range(1, required["repetitions"] + 1))
            or any(sample["status"] != "pass" for sample in samples)
        ):
            problems.append(f"component_incomplete:{required['id']}")
        for sample in samples:
            for side, expected in required.get("expected_observations", {}).items():
                if any(sample.get(side, {}).get(key) != value for key, value in expected.items()):
                    problems.append(
                        f"component_observation_failed:{required['id']}:{sample['repetition']}"
                    )
        source = case["source_revisions"].get(required["distribution"], {})
        if source.get("revision") != revisions.get(required["component"]) or source.get("editable") is not False:
            problems.append(f"component_source_mismatch:{required['id']}")
    native = {item["plugin"]: item for item in evidence.get("native_plugins", [])}
    if len(native) != len(evidence.get("native_plugins", [])):
        raise ValueError("duplicate native plugin evidence")
    for plugin, required_checks in suite.get("native_plugins", {}).items():
        observed = native.get(plugin)
        if observed is None:
            problems.append(f"native_plugin_missing:{plugin}")
            continue
        if observed["status"] != "pass" or observed.get("revision") != revisions.get(plugin):
            problems.append(f"native_plugin_incompatible:{plugin}")
        checks = {check["name"]: check["status"] for check in observed["checks"]}
        if any(checks.get(name) != "pass" for name in required_checks):
            problems.append(f"native_plugin_checks_incomplete:{plugin}")
    case_ids = {case["id"] for case in evidence.get("component_cases", [])}
    observed_ids = []
    for report in evidence.get("component_runner_reports", []):
        observed_ids.extend(report["case_ids"])
        sources = [report.get("runner_source"), report.get("runner_source_after")]
        if any(
            not source
            or not (source.get("revision") or source.get("source_sha256"))
            or source.get("dirty") is True
            for source in sources
        ):
            problems.append("component_runner_identity_missing")
    if sorted(observed_ids) != sorted(case_ids):
        problems.append("component_runner_coverage_mismatch")


def _prerequisite_costs(ledger: dict) -> dict:
    roles = {
        name: ledger["roles"][name]
        for name in ("seed", "capture", "organizer", "review")
        if name in ledger.get("roles", {})
    }
    totals = {}
    for field in ("model_calls", "tokens_total", "estimated_cost_usd", "npu_runs"):
        values = [row[field] for row in roles.values() if row.get(field) is not None]
        totals[field] = sum(values) if values else None
    return {
        "roles": roles,
        "known_totals": totals,
        "unknown_usage": ledger.get("unknown_usage", []),
        "prerequisite_unknown_usage": ledger.get("prerequisite_unknown_usage", []),
        "downstream_excluded": True,
    }


def _downstream_costs(ledger: dict) -> dict:
    return {
        "role": ledger.get("roles", {}).get("downstream"),
        "declared_not_complete": True,
        "note": "downstream evaluation is measured by the P3 product receipt, not the prerequisite ledger",
    }


def evaluate_product(suite: dict, evidence: dict, *, base: Path | None = None) -> dict:
    """Recompute per-cohort gates only from predeclared plans and actual receipts."""
    if evidence.get("schema_version") != EVIDENCE2:
        raise ValueError("unsupported product-value evidence")
    public_document = public_suite(suite)
    if evidence.get("suite_identity") != digest(public_document):
        raise ValueError("evidence belongs to a different public workload suite")
    if any(key in evidence for key in ("observations", "statuses", "result_table", "groups")):
        raise ValueError("caller-supplied result tables cannot override receipt observations")
    bundle = build_product_plan(suite)
    if evidence.get("plan_sha256") != bundle["plan_sha256"]:
        raise ValueError("evidence belongs to a different product plan")
    if evidence.get("formal_baseline_sha256") != digest(bundle["formal_baseline"]):
        raise ValueError("evidence belongs to a different formal P1 baseline")
    if evidence.get("final_release_sha256") != digest(bundle["final_release"]):
        raise ValueError("evidence belongs to a different final P2 candidate")
    benchmark = bundle["benchmark"]
    if evidence.get("benchmark") != benchmark:
        raise ValueError("evidence benchmark differs from the predeclared common runner")
    if evidence.get("runner_release") != {
        key: benchmark.get(key) for key in ("revision", "version", "source_sha256")
    }:
        raise ValueError("evidence runner identity differs from the common benchmark commit")
    final_revisions = {
        name: target["revision"] for name, target in bundle["final_repositories"].items()
    }
    if evidence.get("component_revisions") not in (None, final_revisions):
        raise ValueError("evidence component revisions differ from the final P2 candidate")
    p3 = load_public_p3_claim(
        suite, evidence, bundle, bundle["cohorts"], benchmark, bundle["preparation"]
    )
    manifests = {item["sha256"]: item for item in evidence.get("runtime_manifests", [])}
    if len(manifests) != len(evidence.get("runtime_manifests", [])):
        raise ValueError("duplicate runtime manifest identity")
    selected, trial_ids = {}, set()
    problems: list[str] = []

    def bind(raw: dict) -> tuple[str, ProductGroup, str, int, ProductObservation, TrialReceipt]:
        row = TrialReceipt.model_validate(raw)
        if row.trial_id in trial_ids:
            raise ValueError(f"trial counted more than once: {row.trial_id}")
        trial_ids.add(row.trial_id)
        context = row.environment_fingerprint.get("experiment_context")
        if not isinstance(context, dict):
            raise ValueError("product receipt is missing its experiment context")
        cell = _cell_from_context(context)
        if cell is None:
            raise ValueError("product receipt is missing its product cell")
        if cell.group == ProductGroup.P3_V003_FLYWHEEL:
            if p3.get("status") != "admitted" or cell != p3["cell"]:
                raise _P3ReceiptUnavailable("P3 receipt has no admissible public lineage")
            profile, source = p3["profile"], p3["source"]
            if context.get("p3_admission", {}).get("admission_sha256") != p3["public_lineage"]["admission_sha256"]:
                raise _P3ReceiptUnavailable("P3 receipt admission differs from the public lineage")
            cohort_id = p3["comparison_cohort"]
        else:
            cohort = bundle["cohorts"].get(cell.cohort)
            if cohort is None or cohort["cells"].get(cell.key()) != cell:
                raise ValueError("receipt belongs to a different product plan cell")
            profile, source = cohort["profiles"][cell.group], cohort["sources"][cell.group]
            context_profile = context.get("profile")
            context_public_sha = (
                digest(
                    {
                        key: value
                        for key, value in context_profile.items()
                        if key not in {"experiment_profile", "source"}
                    }
                )
                if isinstance(context_profile, dict)
                else None
            )
            if context.get("profile_sha256") not in {
                digest(profile.to_dict()),
                source.get("public_profile_sha256"),
                context_public_sha,
                digest(context_profile) if isinstance(context_profile, dict) else None,
            }:
                raise ValueError("receipt belongs to a different product profile")
            cohort_id = cell.cohort
        if (cell.profile_id, cell.cohort, cell.protocol) != (
            profile.profile_id, profile.cohort, profile.protocol
        ):
            raise ValueError("receipt belongs to a different product binding")
        for key, value in (
            ("profile_id", profile.profile_id),
            ("cohort", profile.cohort),
            ("protocol", profile.protocol),
        ):
            if context.get(key) not in (None, value):
                raise ValueError("receipt belongs to a different product binding")
        if context.get("internal_capability_mode") != row.ablation_group.value:
            raise ValueError("receipt changed its internal capability mode")
        manifest = manifests.get(context.get("runtime_manifest_sha256"))
        if manifest is None:
            raise ValueError("receipt lacks its native runtime manifest")
        context = {
            **context,
            "_receipt_component_revisions": row.environment_fingerprint.get(
                "component_revisions", {}
            ),
        }
        _bind_runtime(manifest, context, profile, source, benchmark)
        task = bundle["tasks"].get(cell.task_id)
        if task is None:
            raise ValueError("receipt task is not declared by the product plan")
        if (
            row.task_digest != task["task_digest"]
            or row.tier != task["execution_tier"]
            or row.task_role != task["role"]
        ):
            raise ValueError("receipt task identity differs from the product plan")
        if (
            row.environment_fingerprint.get("base_model") != profile.conditions.model_id
            or row.environment_fingerprint.get("benchmark_version") != benchmark["version"]
            or row.environment_fingerprint.get("benchmark_source_sha256") != benchmark["source_sha256"]
        ):
            raise ValueError("receipt model or common benchmark runner differs")
        observation = replace(
            observation_from_receipt(row, cell=cell),
            total_wall_clock_sec=row.primary_metrics.get("delivery_wall_clock_sec"),
        )
        return cohort_id, cell.group, cell.task_id, cell.repetition, observation, row

    for raw in evidence.get("receipts", []):
        try:
            cohort_id, group, task_id, repetition, observation, row = bind(raw)
        except _P3ReceiptUnavailable:
            problems.append("p3_receipt_without_admitted_public_lineage")
            continue
        key = (cohort_id, (group, task_id, repetition))
        if key in selected:
            raise ValueError(f"duplicate selected product cell: {key}")
        selected[key] = (row, observation, manifests[
            row.environment_fingerprint["experiment_context"]["runtime_manifest_sha256"]
        ])
    selected_by_trial = {record[0].trial_id: (record[0], record[1]) for record in selected.values()}

    for cohort_id, cohort in bundle["cohorts"].items():
        missing = sorted(cohort["required_cells"] - {key[1] for key in selected if key[0] == cohort_id})
        problems.extend(
            f"missing_required_cell:{cohort_id}:{group}:{task}:{rep}"
            for group, task, rep in missing
        )
        for key, record in selected.items():
            if key[0] == cohort_id and key[1] in cohort["required_cells"]:
                if record[0].terminal_status.value not in VALID:
                    problems.append(f"unsettled_required_cell:{record[0].trial_id}")
        claim = _functional_claim_status(cohort, selected, cohort_id)
        if claim.get("required") and not claim["claim_ready"]:
            problems.append(f"functional_claim_failed:{cohort_id}")

    historical = [
        _historical_record(item, selected_by_trial, problems)
        for item in evidence.get("excluded_attempts", [])
    ]
    historical_ids = [record["receipt"]["trial_id"] for record in historical]
    if len(historical_ids) != len(set(historical_ids)):
        raise ValueError("a historical attempt cannot be counted twice")
    cohort_reports, all_comparisons = {}, []
    for cohort_id, cohort in bundle["cohorts"].items():
        observations = [
            record[1] for key, record in selected.items()
            if key[0] == cohort_id and key[1][0] != ProductGroup.P3_V003_FLYWHEEL
        ]
        matrix = aggregate_product_matrix(cohort["plan"], observations)
        cohort_reports[cohort_id] = {
            "definition": {
                key: cohort["definition"].get(key)
                for key in (
                    "id", "scenario", "task_ids", "repetitions", "required", "functional_claim"
                )
            },
            "matrix": matrix,
            "functional_claim": _functional_claim_status(cohort, selected, cohort_id),
            "required_cells": len(cohort["required_cells"]),
            "comparisons": [],
        }
    for spec in bundle["comparisons"]:
        cohort_id = spec["cohort"]
        cohort = bundle["cohorts"][cohort_id]
        observations = [
            record[1] for key, record in selected.items()
            if key[0] == cohort_id and key[1][0] != ProductGroup.P3_V003_FLYWHEEL
        ]
        if ProductGroup(spec["candidate"]) == ProductGroup.P3_V003_FLYWHEEL:
            observations += [
                record[1] for key, record in selected.items()
                if key[0] == p3.get("comparison_cohort") and key[1][0] == ProductGroup.P3_V003_FLYWHEEL
            ] if p3.get("status") == "admitted" else []
        result = _comparison(
            cohort["plan"], observations, cohort["profiles"], spec,
            p3 if p3.get("status") == "admitted" else p3,
        )
        cohort_reports[cohort_id]["comparisons"].append(result)
        all_comparisons.append(result)
        if result["required"] and not result["claim_ready"]:
            problems.append(f"required_comparison_incomplete:{cohort_id}:{result['id']}")
    p3_required = bool(p3.get("required"))
    p3_observation = next(
        (record[1] for key, record in selected.items()
         if key[1][0] == ProductGroup.P3_V003_FLYWHEEL),
        None,
    )
    p3_claim_ready = (
        p3.get("status") == "admitted"
        and p3_observation is not None
        and p3_observation.status.value in {"pass", "fail"}
    )
    if p3_required and not p3_claim_ready:
        problems.append("required_p3_lineage_or_observation_missing")
    _component_evidence(suite, evidence, bundle["final_repositories"], problems)
    setup_reports = [project_setup_report(item) for item in evidence.get("setup_reports", [])]
    all_receipts = [record[0] for record in selected.values()] + [
        TrialReceipt.model_validate(record["receipt"]) for record in historical
    ]
    report = {
        "schema_version": REPORT2,
        "product_version": suite["product_version"],
        "suite": suite["id"],
        "suite_identity": digest(public_document),
        "private_suite_sha256": evidence.get("private_suite_sha256"),
        "protocol": PRODUCT_MATRIX_PROTOCOL,
        "plan_sha256": bundle["plan_sha256"],
        "plan_digests": bundle["plan_digests"],
        "benchmark": benchmark,
        "runner_release": evidence.get("runner_release"),
        "collector_source": evidence.get("collector_source"),
        "formal_baseline": bundle["formal_baseline"],
        "final_release": bundle["final_release"],
        "component_revisions": final_revisions,
        "preparation": [
            {
                key: value
                for key, value in entry.items()
                if key not in {"artifact_path", "observed_artifact_path"}
            }
            for entry in bundle["preparation"].values()
        ],
        "profiles": {
            profile.profile_id: _public_profile(profile, source)
            for cohort in bundle["cohorts"].values()
            for group, profile in cohort["profiles"].items()
            for source in [cohort["sources"][group]]
        },
        "cohorts": cohort_reports,
        "comparisons": all_comparisons,
        "p3": {
            "status": p3.get("status"),
            "required": p3_required,
            "scope": p3.get("scope"),
            "claim_ready": p3_claim_ready,
            "reasons": list(p3.get("reasons", [])),
            "public_lineage": p3.get("public_lineage"),
            "prerequisite_costs": _prerequisite_costs(
                p3["public_lineage"]["admission"]["cost_ledger"]
            ) if p3.get("status") == "admitted" else None,
            "downstream_costs": _downstream_costs(
                p3["public_lineage"]["admission"]["cost_ledger"]
            ) if p3.get("status") == "admitted" else None,
            "prerequisite_costs_are_not_downstream": True,
            "public_evidence_only": True,
        },
        "resources": resource_totals(all_receipts),
        "attempts": {
            "selected": len(selected),
            "historical": len(historical),
            "all_executed": len(selected) + len(historical),
            "historical_attempts_retained": True,
            "replacement_of_valid_outcomes": False,
        },
        "historical_attempts": historical,
        "runtime_manifests": list(manifests.values()),
        "receipts": [public_product_receipt(record[0]) for record in selected.values()],
        "component_cases": evidence.get("component_cases", []),
        "native_plugins": evidence.get("native_plugins", []),
        "component_runner_reports": evidence.get("component_runner_reports", []),
        "setup_reports": setup_reports,
        "setup_resources": setup_totals(setup_reports),
        "supplementary_claims": suite.get("supplementary_claims", []),
        "limitations": [
            "Product arms are P0/P1/P2/P3 and are never translated into legacy G0..G3 labels.",
            "Installed product dependencies, common instrumentation, the native plugin and the common benchmark runner are separate identities.",
            "Different cohorts keep their own exact task budgets; no global budget is imposed.",
            "The P2 product release is an immutable candidate, not an already-published main release.",
            "Historical attempts keep their original identities and costs; they are not forced into current pins.",
            "Equal downstream budgets support conditional reuse only; prerequisite Seed/C6/review cost is separate.",
            "No lifecycle superiority is claimed without equal predeclared lifecycle allocations.",
            "Component and native compatibility evidence is not a causal single-component gain.",
            *suite.get("limitations", []),
        ],
        "release_gate": not problems,
        "blocking_reasons": problems,
    }
    if len(cohort_reports) == 1:
        report["matrix"] = next(iter(cohort_reports.values()))["matrix"]
    return report


def public_suite(suite: dict) -> dict:
    public = copy.deepcopy(suite)
    for entry in public.get("preparation", []) or []:
        entry.pop("artifact_path", None)
        entry.pop("observed_artifact_path", None)
    for cohort in public.get("cohorts", []) or []:
        for profile in cohort.get("profiles", []) or []:
            public_profile = {
                key: value
                for key, value in profile.items()
                if key not in {"experiment_profile", "source"}
            }
            profile["experiment_profile"] = "profile:" + digest(public_profile)
    p3 = public.get("p3")
    if isinstance(p3, dict):
        p3.pop("lineage_manifest", None)
        p3["public_lineage_evidence"] = "value-evidence.json:p3_public_lineage"
        profile = p3.get("profile") or {}
        if "experiment_profile" in profile:
            public_profile = {
                key: value
                for key, value in profile.items()
                if key not in {"experiment_profile", "source"}
            }
            profile["experiment_profile"] = "profile:" + digest(public_profile)
    return public


def collect_product(*, suite: Path, inputs: Path, output: Path, collector_source=None) -> dict:
    """Verify private P3 bytes once, then write self-contained public evidence."""
    protocol, source = read_json(suite), read_json(inputs)
    if source.get("schema_version") != INPUTS2:
        raise ValueError("unsupported product-value input manifest")
    if protocol.get("schema_version") != SUITE2:
        raise ValueError("collector requires mindie-value-suite/2")
    bundle = build_product_plan(protocol)
    benchmark = bundle["benchmark"]
    declared_runner = source.get("runner_release") or {}
    if declared_runner != {
        key: benchmark.get(key) for key in ("revision", "version", "source_sha256")
    }:
        raise ValueError("input runner does not match the common benchmark commit")
    collector = require_collector_source(declared_runner, collector_source)
    final_revisions = {
        name: target["revision"] for name, target in bundle["final_repositories"].items()
    }
    if source.get("component_revisions") not in (None, final_revisions):
        raise ValueError("input component revisions differ from the final P2 candidate")
    root = inputs.resolve().parent
    p3_evidence = None
    p3_private = load_private_p3_claim(
        protocol, suite.resolve().parent, bundle, benchmark, bundle["preparation"]
    )
    if p3_private.get("status") == "admitted":
        p3_evidence = p3_private["public_lineage"]
    public_document = public_suite(protocol)
    evidence = {
        "schema_version": EVIDENCE2,
        "public_projection": "Paths, endpoints, commands, credentials and native logs stay private.",
        "suite_identity": digest(public_document),
        "private_suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "plan_sha256": bundle["plan_sha256"],
        "formal_baseline_sha256": digest(bundle["formal_baseline"]),
        "final_release_sha256": digest(bundle["final_release"]),
        "benchmark": benchmark,
        "runner_release": declared_runner,
        "collector_source": collector,
        "component_revisions": final_revisions,
        "receipts": [
            public_product_receipt(TrialReceipt.model_validate(read_json(root / path)))
            for path in source.get("receipts", [])
        ],
        "excluded_attempts": [],
        "runtime_manifests": _read_runtime_manifests(
            source.get("runtime_manifests", []), root, benchmark
        ),
        "component_cases": [],
        "component_runner_reports": [],
        "native_plugins": [],
        "setup_reports": [
            project_setup_report(read_json(root / path))
            for path in source.get("setup_reports", [])
        ],
        "limitations": source.get("limitations", []),
        **({"p3_public_lineage": p3_evidence} if p3_evidence else {}),
        **({"p3_collection_failure": {
            "status": p3_private.get("status"),
            "reasons": p3_private.get("reasons", []),
        }} if p3_private.get("status") != "admitted" and protocol.get("p3") else {}),
    }
    for item in source.get("excluded_attempts", []):
        raw_bytes = (root / item["path"]).read_bytes()
        raw = json.loads(raw_bytes)
        evidence["excluded_attempts"].append({
            "reason": item["reason"],
            "replacement_trial_id": item.get("replacement_trial_id"),
            "source_receipt_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "canonical_receipt_sha256": digest(raw),
            "receipt": public_product_receipt(TrialReceipt.model_validate(raw)),
        })
    for path in source.get("component_reports", []):
        record = read_json(root / path)
        evidence["component_runner_reports"].append(project_component_runner(record))
        evidence["component_cases"].extend(
            project_component_case(case) for case in record["cases"]
        )
    for path in source.get("native_plugins", []):
        record = read_json(root / path)
        evidence["native_plugins"].append({
            key: record[key] for key in ("plugin", "revision", "status", "limitations")
        } | {"checks": [
            {"name": check["name"], "status": check["status"]} for check in record["checks"]
        ]})
    report = evaluate_product(protocol, evidence, base=suite.resolve().parent)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "value-suite.json", public_document)
    write_json(output / "value-evidence.json", evidence)
    write_report_product(report, output)
    return report


def verify_publication_product(
    *, suite: Path, evidence: Path, manifest: Path, collector_source=None
) -> dict:
    """Bind the candidate report to its ten repository targets; no circular publish demand."""
    protocol, payload = read_json(suite), read_json(evidence)
    report = evaluate_product(protocol, payload, base=suite.resolve().parent)
    report["collector_source"] = require_collector_source(report["runner_release"], collector_source)
    targets = read_json(manifest)
    final = report["final_release"]
    if report["product_version"] != targets.get("product_version") or final["product_version"] != targets["product_version"]:
        raise ValueError("value evidence and formal product version differ")
    declared = {
        _component_name(target["repository"], target.get("component")): target
        for target in targets.get("repositories", [])
    }
    expected = {
        _component_name(target["repository"], target.get("component")): target
        for target in final["repositories"]
    }
    if set(declared) != set(expected) or len(targets.get("repositories", [])) != len(COMPONENT_SET):
        raise ValueError("release manifest does not bind the final ten component targets")
    for component, target in expected.items():
        actual = declared[component]
        if (actual.get("repository"), actual.get("version"), actual.get("revision")) != (
            target.get("repository"), target.get("version"), target.get("revision")
        ):
            raise ValueError(f"value evidence and candidate release source differ: {component}")
    if report["component_revisions"] != {
        component: target["revision"] for component, target in expected.items()
    }:
        raise ValueError("value evidence and release manifest component revisions differ")
    report["release_manifest_sha256"] = digest(targets)
    return report


def write_report_product(report: dict, output: Path) -> None:
    from mindie_bench.value_product_views import homepage_markdown_product, markdown_product

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "value-results.json", report)
    (output / "value-results.md").write_text(markdown_product(report), encoding="utf-8")
    for language, filename in (("zh", "value-homepage.md"), ("en", "value-homepage.en.md")):
        path = output / filename
        if report["release_gate"]:
            path.write_text(homepage_markdown_product(report, language), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
