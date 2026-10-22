from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from mindie_bench.io import digest, write_json
from mindie_bench.value_product import (
    INPUTS2,
    SUITE2,
    build_product_plan,
    collect_product,
)
from mindie_bench.value_product_public import (
    public_admission,
    public_catalog,
)

BENCHMARK = {
    "version": "0.3.0",
    "revision": "c" * 40,
    "source_sha256": "d" * 64,
}
FORMAL = {
    "diagnostics": ("0.4.3", "1b5a0eeb79d498c39002fc1853271c17e0f26e66"),
    "knowledge": ("0.12.2", "4e328305323eafc7fd43ecc0c53eb3cc0541fdef"),
    "remote-dev": ("0.10.1", "a6649222aae4851deec5774513f8319aece09051"),
    "coordinator": ("0.5.3", "34812c2d385f3ff112be3a597d27fd3a0c15c104"),
    "npu-top": ("0.1.9", "9bbf6de590b12fe1109040e0ef86817bd2c5f3e0"),
    "codex": ("0.0.2", "f1acb69ba336f75ae2d909443ca118835b038b9c"),
    "claude-code": ("0.0.2", "78f34dd9b09e877298225fa243ea4c28978c9dd6"),
    "kimi": ("0.0.2", "0e78f0422162a61e8c56850e049f9e5fc4d290b5"),
    "dsh": ("0.0.2", "d3a1e1282356b65b51fba49072c1842918746f8f"),
    "mindie-bench": ("0.3.0", "68d90d7a689f836c61a64ab054a0282e7361fd16"),
}
REPOSITORIES = {
    "diagnostics": "mindie-agent/diagnostics",
    "knowledge": "mindie-agent/knowledge",
    "remote-dev": "mindie-agent/remote-dev",
    "coordinator": "mindie-agent/coordinator",
    "npu-top": "mindie-agent/npu-top",
    "codex": "mindie-agent/mindie-agent-codex",
    "claude-code": "mindie-agent/mindie-agent-cc",
    "kimi": "mindie-agent/mindie-agent-kimi",
    "dsh": "mindie-agent/mindie-agent-dsh",
    "mindie-bench": "mindie-agent/mindie-bench",
}
PRODUCT_DEPENDENCIES = ["knowledge", "remote-dev", "diagnostics"]
INSTRUMENTATION = ["coordinator", "npu-top"]
RUNTIME_MODULES = ["knowledge", "remote-dev", "diagnostics", "npu-top", "coordinator"]


def _revision(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:40]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def final_components() -> dict[str, tuple[str, str]]:
    return {
        "knowledge": ("0.0.3", _revision("final:knowledge")),
        "remote-dev": ("0.0.3", _revision("final:remote-dev")),
        "diagnostics": ("0.0.3", _revision("final:diagnostics")),
        "coordinator": FORMAL["coordinator"],
        "npu-top": FORMAL["npu-top"],
        "codex": ("0.0.3", _revision("final:codex")),
        "claude-code": ("0.0.3", _revision("final:claude-code")),
        "kimi": ("0.0.3", _revision("final:kimi")),
        "dsh": ("0.0.3", _revision("final:dsh")),
        "mindie-bench": ("0.3.0", BENCHMARK["revision"]),
    }


def release_identities() -> dict:
    final = final_components()
    formal_repos = [
        {
            "repository": REPOSITORIES[name],
            "component": name,
            "version": FORMAL[name][0],
            "revision": FORMAL[name][1],
        }
        for name in FORMAL
    ]
    final_repos = [
        {
            "repository": REPOSITORIES[name],
            "component": name,
            "version": final[name][0],
            "revision": final[name][1],
            "artifact_sha256": _sha("artifact:" + name),
        }
        for name in final
    ]
    final_plugin = {
        "name": "dsh",
        "version": "0.0.3",
        "revision": final["dsh"][1],
        "artifact_sha256": _sha("artifact:dsh"),
    }
    combination = {
        "product_version": "0.0.3",
        "native_plugin": final_plugin,
        "repositories": [
            {
                key: target[key]
                for key in ("repository", "component", "version", "revision", "artifact_sha256")
            }
            for target in sorted(final_repos, key=lambda item: item["component"])
        ],
    }
    combination_sha = digest(combination)
    return {
        "formal_baseline": {
            "product_version": "0.0.2",
            "revision": "d3a1e1282356b65b51fba49072c1842918746f8f",
            "artifact_sha256": "e" * 64,
            "status": "formal_release",
            "provenance": {
                "kind": "formal_release",
                "tag": "v0.0.2",
                "release_url": "https://github.com/mindie-agent/mindie-bench/releases/tag/v0.0.2",
                "published_at": "2026-10-01T00:00:00Z",
                "source_manifest_sha256": "f" * 64,
            },
            "repositories": formal_repos,
            "product_dependencies": PRODUCT_DEPENDENCIES,
            "instrumentation": INSTRUMENTATION,
            "native_plugin": {
                "name": "dsh",
                "version": "0.0.2",
                "revision": FORMAL["dsh"][1],
                "artifact_sha256": "e" * 64,
            },
        },
        "final_release": {
            "product_version": "0.0.3",
            "tag": "v0.0.3",
            "revision": final_plugin["revision"],
            "artifact_sha256": final_plugin["artifact_sha256"],
            "combination_manifest_sha256": combination_sha,
            "status": "product_release_candidate",
            "provenance": {
                "kind": "product_release_candidate",
                "tag": "v0.0.3",
                "source_manifest_sha256": combination_sha,
            },
            "repositories": final_repos,
            "product_dependencies": PRODUCT_DEPENDENCIES,
            "instrumentation": INSTRUMENTATION,
            "native_plugin": final_plugin,
        },
        "formal": FORMAL,
        "final": final,
    }


def module_identity(name: str, version: str, revision: str) -> dict:
    return {"version": version, "revision": revision}


def runtime_modules(components: dict[str, tuple[str, str]]) -> dict:
    return {
        name: module_identity(name, components[name][0], components[name][1])
        for name in RUNTIME_MODULES
    }


def plugin_identity(version: str, revision: str, seed: str) -> dict:
    return {
        "name": "dsh",
        "version": version,
        "revision": revision,
        "contribution_enabled": False,
    }


def source(
    *,
    group: str,
    modules: dict,
    plugin: dict | None,
    catalog_units: int,
    preparation_id: str | None,
    releases: dict,
) -> dict:
    if group == "P0_bare_host":
        product_version, revision, artifact = "bare-host", None, None
    elif group == "P1_v002_baseline":
        formal = releases["formal_baseline"]
        product_version, revision, artifact = (
            formal["product_version"], formal["revision"], formal["artifact_sha256"]
        )
    else:
        final = releases["final_release"]
        product_version, revision, artifact = (
            final["product_version"], final["revision"], final["artifact_sha256"]
        )
    catalog = {
        "schema": "mindie-catalog/1",
        "units": catalog_units,
        "sha256": _sha("catalog:" + group),
    }
    result = {
        "product_version": product_version,
        "product_revision": revision,
        "product_artifact_sha256": artifact,
        "runtime_modules": modules,
        "product_dependencies": PRODUCT_DEPENDENCIES if group != "P0_bare_host" else [],
        "instrumentation": INSTRUMENTATION if group != "P0_bare_host" else [],
        "native_plugin": plugin,
        "benchmark": copy.deepcopy(BENCHMARK),
        "catalog": catalog,
        "catalog_sha256": digest(public_catalog(catalog)),
    }
    if preparation_id:
        result["preparation_id"] = preparation_id
    return result


def preparation_entries(releases: dict) -> list[dict]:
    old_modules = {
        "knowledge": module_identity("knowledge", "0.12.2", FORMAL["knowledge"][1]),
        "remote-dev": module_identity("remote-dev", "0.10.1", FORMAL["remote-dev"][1]),
        "diagnostics": module_identity("diagnostics", "0.4.3", FORMAL["diagnostics"][1]),
        "npu-top": module_identity("npu-top", "0.1.9", FORMAL["npu-top"][1]),
        "coordinator": module_identity("coordinator", "0.5.2", "daf7fa38b431187c04a6dea4341df4a066100953"),
    }
    p1_modules = runtime_modules(releases["formal"])
    p2_modules = runtime_modules(releases["final"])
    return [
        {
            "id": "current-p0",
            "status": "usable",
            "scope": "Fresh P0 preparation with an empty mounted catalog.",
            "observed_runtime_modules_sha256": digest({}),
        },
        {
            "id": "current-p1",
            "status": "usable",
            "scope": "Current P1 preparation: formal product dependencies and common instrumentation.",
            "observed_runtime_modules_sha256": digest(p1_modules),
        },
        {
            "id": "current-p2",
            "status": "usable",
            "scope": "Current P2 preparation: final candidate dependencies and common instrumentation.",
            "observed_runtime_modules_sha256": digest(p2_modules),
        },
        {
            "id": "old-p1-observed",
            "status": "not_usable",
            "scope": "Historical P1 observation retained only as superseded evidence.",
            "observed_runtime_modules_sha256": digest(old_modules),
            "reasons": [
                "coordinator 0.5.2/daf7fa38 differs from formal 0.0.2 coordinator 0.5.3/34812c2",
                "bench 0.2.0/fe87041 is not the common benchmark runner",
            ],
        },
    ]


def task(
    task_id: str,
    *,
    family: str,
    digest_value: str,
    oracle: str,
    max_turns: int,
    npu_runs: int,
    timeout: float,
    verifier: float,
    wall: float,
) -> dict:
    return {
        "task_id": task_id,
        "task_family": family,
        "hardware_tier": "A2",
        "task_digest": digest_value,
        "oracle_digest": oracle,
        "execution_tier": "l3",
        "role": "eval",
        "source_budget": {
            "max_turns": max_turns,
            "max_npu_runs": npu_runs,
            "timeout_sec": timeout,
            "verifier_timeout_sec": verifier,
            "max_wall_clock_sec": wall,
            "human_interventions": 0,
        },
    }


def cohort_conditions(task_rows) -> dict:
    rows = [task_rows] if isinstance(task_rows, dict) else list(task_rows)
    return {
        "model_id": "deepseek-flash",
        "inference_profile": "temperature=0",
        "task_start_digest": "start:" + "+".join(row["task_id"] for row in rows),
        "oracle_digest": digest({row["task_id"]: row["oracle_digest"] for row in rows}),
        "input_digest": "input:" + "+".join(row["task_id"] for row in rows),
        "knowledge_cutoff": "2026-10-01",
        "tool_permission_digest": "tools-sha",
        "persistence_permission_digest": "persist-sha",
    }


def cohort_budget(task_row: dict) -> dict:
    source_budget = task_row["source_budget"]
    return {
        "total_model_calls": source_budget["max_turns"],
        "foreground_model_calls": source_budget["max_turns"],
        "npu_runs": source_budget["max_npu_runs"],
        "human_interventions": source_budget["human_interventions"],
        "max_wall_clock_sec": source_budget["max_wall_clock_sec"],
        "max_cost_usd": 2.0,
    }


def runtime_manifest(profile: dict, source_identity: dict, task_rows: list[dict], root: Path) -> dict:
    modules = source_identity["runtime_modules"]
    plugin = source_identity["native_plugin"]
    runtime = {"harness_binary_sha256": "0" * 64}
    if plugin:
        runtime["native_product"] = {
            "contribution_enabled": False,
            "runtime_identity": {"components": modules},
        }
    derived = {
        "conditions": profile["conditions"],
        "budget": {
            key: profile["budget"][key]
            for key in ("total_model_calls", "npu_runs", "human_interventions", "max_wall_clock_sec")
        },
        "budget_scope": "shared phase budget from run_trial start through completed cleanup",
        "corpora": {},
        "tasks": {
            row["task_id"]: {
                "task_root": "/private/tasks/" + row["task_id"],
                "task_sha256": row["task_digest"],
                "oracle_sha256": row["oracle_digest"],
            }
            for row in task_rows
        },
    }
    raw = {
        "profile_id": profile["profile_id"],
        "product_group": profile["group"],
        "config_sha256": _sha("config:" + profile["profile_id"]),
        "scope": "Native-contract publication fixture",
        "derived_inputs": derived,
        "runtime_snapshot": runtime,
        "native_plugin": plugin,
        "catalog": source_identity["catalog"],
    }
    path = root / f"runtime-{profile['profile_id']}.json"
    write_json(path, raw)
    return raw


def receipt(
    *,
    profile: dict,
    source_identity: dict,
    cell,
    task_row: dict,
    manifest: dict,
    delivery: float,
    status: str = "completed_pass",
    profile_document: dict | None = None,
) -> dict:
    revisions = {
        name: module["revision"] for name, module in source_identity["runtime_modules"].items()
    }
    if source_identity.get("native_plugin"):
        revisions["dsh"] = source_identity["native_plugin"]["revision"]
    revisions["mindie-bench"] = BENCHMARK["revision"]
    passed = status == "completed_pass"
    primary = {
        "reward": 1.0 if passed else 0.0,
        "resolved": passed,
        "tokens": {
            "input_uncached": 100,
            "input_cached": 10,
            "cache_creation": 0,
            "output": 20,
            "indexer_small_model_total": 0,
            "complete": True,
        },
        "cost_complete": True,
        "estimated_cost_usd": 0.1,
        "model_calls": 5,
        "total_wall_clock_sec": delivery - 10,
        "delivery_wall_clock_sec": delivery,
        "npu_test_runs": 1,
        "manual_interventions": 0,
        "zero_human_intervention": True,
    }
    if not passed:
        primary["failure_reason"] = "oracle_failed"
    return {
        "schema_version": "mindie-bench-receipt/1.0",
        "trial_id": f"{profile['cohort']}-{profile['group']}-{task_row['task_id']}",
        "instance_id": task_row["task_id"],
        "repetition": 1,
        "tier": task_row["execution_tier"],
        "ablation_group": {"P0_bare_host": "G0", "P1_v002_baseline": "G1", "P2_v003_full": "G2", "P3_v003_flywheel": "G3"}[profile["group"]],
        "knowledge_gradient": "no_hit",
        "task_role": task_row["role"],
        "task_digest": task_row["task_digest"],
        "experiment_digest": manifest["config_sha256"],
        "environment_fingerprint": {
            "host_os": "test",
            "host_arch": "arm64",
            "harness_name": "dsh",
            "harness_version": "native",
            "harness_binary_sha256": "0" * 64,
            "base_model": profile["conditions"]["model_id"],
            "reasoning_effort": "high",
            "provider_url": "http://private-service",
            "backend": "native",
            "budget": {"max_turns": task_row["source_budget"]["max_turns"]},
            "base_commit": "b" * 40,
            "npu_model": "910B",
            "npu_count": 1,
            "remote_environment": {"host": "private-host"},
            "benchmark_version": BENCHMARK["version"],
            "benchmark_source_sha256": BENCHMARK["source_sha256"],
            "component_revisions": revisions,
            "components": [name for name in source_identity["runtime_modules"]] + (["dsh"] if source_identity.get("native_plugin") else []),
            "experiment_context": {
                "schema": "mindie-bench-product-execution/1",
                "cell": cell.to_dict(),
                "profile_id": profile["profile_id"],
                "profile_group": profile["group"],
                "cohort": profile["cohort"],
                "protocol": profile["protocol"],
                "runtime_manifest": "/private/runtime.json",
                "internal_capability_mode": {"P0_bare_host": "G0", "P1_v002_baseline": "G1", "P2_v003_full": "G2", "P3_v003_flywheel": "G3"}[profile["group"]],
                "task_sha256": task_row["task_digest"],
                "execution_tier": task_row["execution_tier"],
                "actual_runtime": manifest["runtime_snapshot"],
                "max_wall_clock_sec": profile["budget"]["max_wall_clock_sec"],
                "profile": profile_document or profile,
            },
        },
        "terminal_status": status,
        "failure_reason": None if passed else "oracle_failed",
        "primary_metrics": primary,
        "auxiliary_metrics": {
            "lifecycle_budget": {
                "declared_total_sec": profile["budget"]["max_wall_clock_sec"],
                "measured_total_sec": delivery,
                "overrun_sec": 0.0,
                "overrun": False,
                "budget_exceeded": False,
            },
            "budget_exceeded": False,
            "business_terminal_status": status,
            "verifier_wall_clock_sec": 2.0,
            "capture": {
                "status": "completed",
                "corpus_lint": {"status": "passed"},
                "publication": {"revision": "publication-1"},
                "consumer": {"revision": "consumer-1"},
            },
        },
        "errors": [],
        "artifacts": {"private_trial_dir": "/private/trial"},
        "evidence": "synthetic",
    }


def make_suite(
    tmp_path: Path,
    *,
    cohorts_spec: list[dict] | None = None,
    extra_tasks: list[dict] | None = None,
) -> tuple[dict, dict, dict]:
    releases = release_identities()
    preparation = preparation_entries(releases)
    specs = cohorts_spec or [
        {
            "id": "fused-shared-budget",
            "scenario": "fixed_case_optimization",
            "task": task(
                "fused-rmsnorm-gate-dispatch-performance",
                family="performance",
                digest_value=_sha("task:fused"),
                oracle="1" * 64,
                max_turns=40,
                npu_runs=8,
                timeout=1800.0,
                verifier=600.0,
                wall=3361.65,
            ),
        }
    ]
    tasks = []
    for spec in specs:
        tasks.append(spec["task"])
        tasks.extend(spec.get("extra_tasks", []))
    tasks.extend(extra_tasks or [])
    suite = {
        "schema_version": SUITE2,
        "id": "product-value-v2-fixture",
        "product_version": "0.0.3",
        "protocol": "matched-product/1",
        "benchmark": copy.deepcopy(BENCHMARK),
        "tasks": tasks,
        "preparation": preparation,
        "formal_baseline": releases["formal_baseline"],
        "final_release": releases["final_release"],
        "cohorts": [],
        "comparisons": [],
        "supplementary_claims": [
            {"id": "codex-native", "scope": "Compatibility evidence only, not a product comparison."}
        ],
        "component_cases": [],
        "native_plugins": {},
        "limitations": ["Synthetic native-contract publication fixture."],
    }
    profile_sources = {
        "P0_bare_host": source(
            group="P0_bare_host", modules={}, plugin=None, catalog_units=0,
            preparation_id="current-p0", releases=releases,
        ),
        "P1_v002_baseline": source(
            group="P1_v002_baseline", modules=runtime_modules(releases["formal"]),
            plugin=plugin_identity("0.0.2", FORMAL["dsh"][1], "p1"),
            catalog_units=3, preparation_id="current-p1", releases=releases,
        ),
        "P2_v003_full": source(
            group="P2_v003_full", modules=runtime_modules(releases["final"]),
            plugin=plugin_identity("0.0.3", releases["final"]["dsh"][1], "p2"),
            catalog_units=3, preparation_id="current-p2", releases=releases,
        ),
    }
    manifests, rows = {}, []
    for spec in specs:
        cohort_id = spec["id"]
        cohort_tasks = [spec["task"], *spec.get("extra_tasks", [])]
        conditions = cohort_conditions(cohort_tasks)
        envelope = cohort_budget(spec["task"])
        profiles = []
        for group in ("P0_bare_host", "P1_v002_baseline", "P2_v003_full"):
            profile = {
                "profile_id": f"{cohort_id}-{group}",
                "group": group,
                "experiment_profile": f"profiles/{cohort_id}-{group}.json",
                "protocol": "matched-product/1",
                "fidelity": "matched",
                "cohort": cohort_id,
                "conditions": conditions,
                "budget": envelope,
                "source": profile_sources[group],
            }
            profiles.append(profile)
        suite["cohorts"].append({
            "id": cohort_id,
            "scenario": spec["scenario"],
            "task_ids": [row["task_id"] for row in cohort_tasks],
            "repetitions": 1,
            "required": True,
            "functional_claim": {"group": "P2_v003_full", "minimum_passes": 1},
            "profiles": profiles,
        })
        suite["comparisons"].extend([
            {
                "id": f"{cohort_id}-p0-p1", "cohort": cohort_id,
                "baseline": "P0_bare_host", "candidate": "P1_v002_baseline",
                "required": True, "scope": "Native host to formal baseline.",
            },
            {
                "id": f"{cohort_id}-p1-p2", "cohort": cohort_id,
                "baseline": "P1_v002_baseline", "candidate": "P2_v003_full",
                "required": True, "scope": "Formal baseline to final candidate.",
            },
        ])
    bundle = build_product_plan(suite)
    for spec in specs:
        cohort_id = spec["id"]
        cohort = bundle["cohorts"][cohort_id]
        for profile in cohort["profiles"].values():
            source_identity = cohort["sources"][profile.group]
            raw_manifest = runtime_manifest(
                {
                    **profile.__dict__,
                    "conditions": profile.conditions.to_dict(),
                    "budget": profile.budget.to_dict(),
                },
                source_identity,
                cohort_tasks,
                tmp_path,
            )
            manifests[profile.profile_id] = raw_manifest
        for cell in cohort["plan"].cells:
            profile = cohort["profiles"][cell.group]
            source_identity = cohort["sources"][cell.group]
            row = receipt(
                profile={
                    **profile.__dict__,
                    "conditions": profile.conditions.to_dict(),
                    "budget": profile.budget.to_dict(),
                },
                source_identity=source_identity,
                cell=cell,
                task_row=next(row for row in tasks if row["task_id"] == cell.task_id),
                manifest=manifests[profile.profile_id],
                profile_document=profile.to_dict(),
                delivery={"P0_bare_host": 100.0, "P1_v002_baseline": 110.0, "P2_v003_full": 120.0}[profile.group.value],
            )
            rows.append(row)
    receipt_paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f"receipt-{index}.json"
        write_json(path, row)
        receipt_paths.append(path.name)
    inputs = {
        "schema_version": INPUTS2,
        "product_version": "0.0.3",
        "runner_release": copy.deepcopy(BENCHMARK),
        "component_revisions": {name: value[1] for name, value in releases["final"].items()},
        "receipts": receipt_paths,
        "runtime_manifests": [f"runtime-{profile_id}.json" for profile_id in manifests],
        "component_reports": [],
        "native_plugins": [],
        "setup_reports": [],
        "excluded_attempts": [],
    }
    return suite, inputs, {"receipts": rows, "manifests": manifests, "releases": releases}


def collector() -> dict:
    return {**BENCHMARK, "dirty": False, "editable": False}


def write_evidence(tmp_path: Path, suite: dict, inputs: dict):
    suite_path, inputs_path = tmp_path / "value-suite.json", tmp_path / "inputs.json"
    write_json(suite_path, suite)
    write_json(inputs_path, inputs)
    return suite_path, inputs_path


def collected(tmp_path: Path, *, cohorts_spec=None, extra_tasks=None):
    suite, inputs, extra = make_suite(tmp_path, cohorts_spec=cohorts_spec, extra_tasks=extra_tasks)
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "out",
        collector_source=collector(),
    )
    return suite, inputs, extra, suite_path, inputs_path, report


def test_public_admission_timeline_drops_arbitrary_error_text():
    admission = {
        "status": "admitted",
        "timeline": {
            "absolute_available": True,
            "missing_absolute": [],
            "inference_performed": False,
            "seed_completed_at": "2026-01-01T00:00:00Z",
            "capture_completed_at": "2026-01-01T00:00:01Z",
            "corpus_frozen_at": "2026-01-01T00:00:02Z",
            "downstream_started_at": "2026-01-01T00:00:03Z",
            "stage_observations": [
                {
                    "stage_id": "graded_distillation",
                    "status": "completed",
                    "started_sec": 0.0,
                    "ended_sec": 1.0,
                    "elapsed_sec": 1.0,
                    "started_at": "2026-01-01T00:00:00Z",
                    "ended_at": "2026-01-01T00:00:01Z",
                    "error": {
                        "message": "/private/secret-error-token",
                        "native_log": "/Users/example/private.log",
                    },
                }
            ],
        },
    }
    projected = public_admission(admission)
    serialized = json.dumps(projected)
    assert "/private/secret-error-token" not in serialized
    assert "/Users/example/private.log" not in serialized
    stage = projected["timeline"]["stage_observations"][0]
    assert stage["stage_id"] == "graded_distillation"
    assert stage["error_sha256"] == digest(admission["timeline"]["stage_observations"][0]["error"])
