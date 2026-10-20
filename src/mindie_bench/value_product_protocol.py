"""Predeclared product-matrix cohorts, source identities and budgets.

[Role] Validate suite/2 scenarios, per-cohort P0/P1/P2 profiles, matched
conditions/budgets, formal P1 dependencies, final P2 candidate repositories,
native plugin/benchmark instrumentation and preparation evidence before any
outcome is read. Build one actual ProductMatrixPlan per cohort.
[Invariants] Product profiles never come from legacy G0..G3 values. Installed
product dependencies, common instrumentation, the native plugin and the common
benchmark runner are separate identities; ten release repositories are never
invented as ten loaded Python modules. Different cohorts keep their own exact
task budgets. P1 and P2 remain distinct; the P2 product release is a candidate,
not an already-published main-product release. A P3 claim needs a real lineage.
[Timeouts & Locks] Pure validation plus bounded read-only metadata reads.
[Side-Effect Policy] No writes, model calls, NPU execution or external requests.
"""

from __future__ import annotations

from mindie_bench.io import digest
from mindie_bench.runner.matrix import (
    BudgetEnvelope,
    MatchingConditions,
    MatrixTask,
    ProductGroup,
    ProductProfile,
    plan_product_matrix,
)
from mindie_bench.value_product_public import (
    canonical_benchmark,
    canonical_native_plugin,
    canonical_runtime_modules,
    public_catalog,
)

SUITE2 = "mindie-value-suite/2"
EVIDENCE2 = "mindie-value-evidence/2"
INPUTS2 = "mindie-value-inputs/2"
REPORT2 = "mindie-value-report/2"

CORE_GROUPS = (
    ProductGroup.P0_BARE_HOST,
    ProductGroup.P1_V002_BASELINE,
    ProductGroup.P2_V003_FULL,
)
COMPONENT_SET = {
    "knowledge",
    "remote-dev",
    "coordinator",
    "npu-top",
    "diagnostics",
    "codex",
    "claude-code",
    "kimi",
    "dsh",
    "mindie-bench",
}
RUNTIME_MODULE_SET = {"knowledge", "remote-dev", "coordinator", "npu-top", "diagnostics"}


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        char in "0123456789abcdef" for char in value
    )


def _required(value: dict, name: str):
    if name not in value:
        raise ValueError(f"product suite is missing {name}")
    return value[name]


def _component_name(repository: str, declared: str | None = None) -> str:
    name = repository.split("/")[-1].removeprefix("mindie-agent-")
    name = {"cc": "claude-code"}.get(name, name)
    if declared is not None and declared != name:
        raise ValueError("final release component name differs from its repository")
    return name


def _provenance_formal(raw: dict, version: str) -> dict:
    provenance = _required(raw, "provenance")
    if provenance.get("kind") != "formal_release":
        raise ValueError("formal source identity needs formal release provenance")
    if provenance.get("tag") != f"v{version}":
        raise ValueError("formal provenance tag differs from its product version")
    if not str(provenance.get("release_url") or "").startswith("https://"):
        raise ValueError("formal provenance needs a public release URL")
    if not provenance.get("published_at") or not _hex(provenance.get("source_manifest_sha256"), 64):
        raise ValueError("formal provenance needs publication time and source manifest hash")
    return provenance


def _provenance_candidate(raw: dict, version: str) -> dict:
    provenance = _required(raw, "provenance")
    if provenance.get("kind") != "product_release_candidate":
        raise ValueError("final P2 source must be an immutable product release candidate")
    if provenance.get("tag") != f"v{version}":
        raise ValueError("candidate provenance tag differs from its product version")
    if provenance.get("published_at"):
        raise ValueError("a pending product candidate must not claim a publication time")
    if not _hex(provenance.get("source_manifest_sha256"), 64):
        raise ValueError("candidate provenance needs a source manifest hash")
    return provenance


def _repositories(raw: dict, label: str, *, candidate: bool) -> dict:
    repositories = raw.get("repositories") or []
    if len(repositories) != len(COMPONENT_SET):
        raise ValueError(f"{label} must bind exactly ten repository targets")
    result, seen = {}, set()
    for target in repositories:
        repository = target.get("repository", "")
        if repository in seen:
            raise ValueError(f"duplicate {label} repository")
        seen.add(repository)
        component = _component_name(repository, target.get("component"))
        if component not in COMPONENT_SET or component in result:
            raise ValueError(f"unexpected or duplicate {label} component: {component}")
        if not target.get("version") or not _hex(target.get("revision"), 40):
            raise ValueError(f"{label} target is incomplete: {component}")
        if candidate and not _hex(target.get("artifact_sha256"), 64):
            raise ValueError(f"candidate target needs an artifact hash: {component}")
        result[component] = {
            "repository": repository,
            "component": component,
            "version": target["version"],
            "revision": target["revision"],
            "artifact_sha256": target.get("artifact_sha256"),
        }
    if set(result) != COMPONENT_SET:
        raise ValueError(f"{label} does not cover all ten component targets")
    return result


def _roles(raw: dict, label: str) -> tuple[list[str], list[str]]:
    dependencies = list(raw.get("product_dependencies") or [])
    instrumentation = list(raw.get("instrumentation") or [])
    if set(dependencies) | set(instrumentation) != RUNTIME_MODULE_SET:
        raise ValueError(f"{label} roles must cover the five runtime modules")
    if set(dependencies) & set(instrumentation):
        raise ValueError(f"{label} roles cannot overlap")
    return dependencies, instrumentation


def _release_identities(suite: dict) -> dict:
    formal = _required(suite, "formal_baseline")
    final = _required(suite, "final_release")
    for source, label in ((formal, "P1 formal baseline"), (final, "P2 final release")):
        for field in ("product_version", "revision", "artifact_sha256"):
            if not source.get(field):
                raise ValueError(f"{label} is missing {field}")
        if not _hex(source["revision"], 40) or not _hex(source["artifact_sha256"], 64):
            raise ValueError(f"{label} needs an exact commit and artifact hash")
    if formal.get("status") != "formal_release":
        raise ValueError("P1 formal baseline must be an already formal release")
    _provenance_formal(formal, formal["product_version"])
    if final.get("status") != "product_release_candidate":
        raise ValueError("P2 final source must be an immutable product release candidate")
    _provenance_candidate(final, final["product_version"])
    final_plugin = final.get("native_plugin") or {}
    for field in ("revision", "artifact_sha256"):
        if not _hex(final_plugin.get(field), 40 if field == "revision" else 64):
            raise ValueError(f"P2 native plugin needs an exact {field}")
    if final.get("revision") != final_plugin["revision"]:
        raise ValueError("P2 product revision must be the runnable DSH plugin revision")
    if final.get("artifact_sha256") != final_plugin["artifact_sha256"]:
        raise ValueError("P2 product artifact must be the runnable DSH plugin artifact")
    if final_plugin.get("name") != "dsh":
        raise ValueError("P2 product identity must bind the DSH native plugin")
    if not _hex(final.get("combination_manifest_sha256"), 64):
        raise ValueError("P2 candidate needs an exact combination manifest digest")
    if final["provenance"].get("source_manifest_sha256") != final["combination_manifest_sha256"]:
        raise ValueError("P2 candidate provenance does not bind its combination manifest")
    if (
        formal["product_version"] == final["product_version"]
        or formal["revision"] == final["revision"]
        or formal["artifact_sha256"] == final["artifact_sha256"]
    ):
        raise ValueError("P1 formal baseline and P2 final release must be distinct")
    formal_repos = _repositories(formal, "P1 formal baseline", candidate=False)
    final_repos = _repositories(final, "P2 final release", candidate=True)
    formal_deps, formal_inst = _roles(formal, "P1 formal baseline")
    final_deps, final_inst = _roles(final, "P2 final release")
    for source, label, repos in (
        (formal, "P1 formal baseline", formal_repos),
        (final, "P2 final release", final_repos),
    ):
        plugin = source.get("native_plugin") or {}
        if plugin.get("name") != "dsh" or not _hex(plugin.get("revision"), 40):
            raise ValueError(f"{label} needs the dsh native plugin identity")
        if set((source.get("product_dependencies") or [])) | set(
            (source.get("instrumentation") or [])
        ) != RUNTIME_MODULE_SET:
            raise ValueError(f"{label} roles are incomplete")
        for component in [*source["product_dependencies"], *source["instrumentation"]]:
            if component not in repos:
                raise ValueError(f"{label} role is not a repository target: {component}")
    if formal_repos["dsh"]["revision"] != formal["native_plugin"]["revision"]:
        raise ValueError("P1 native plugin revision differs from its release manifest")
    if final_repos["dsh"]["revision"] != final["native_plugin"]["revision"]:
        raise ValueError("P2 native plugin revision differs from its candidate manifest")
    if final_repos["dsh"]["artifact_sha256"] != final["native_plugin"]["artifact_sha256"]:
        raise ValueError("P2 native plugin artifact differs from its candidate manifest")
    if final_repos["mindie-bench"]["revision"] != suite["benchmark"]["revision"]:
        raise ValueError("P2 candidate benchmark repository differs from the common runner")
    combination = {
        "product_version": final["product_version"],
        "native_plugin": {
            key: final["native_plugin"][key]
            for key in ("name", "version", "revision", "artifact_sha256")
        },
        "repositories": [
            {
                key: target[key]
                for key in ("repository", "component", "version", "revision", "artifact_sha256")
            }
            for target in sorted(final_repos.values(), key=lambda item: item["component"])
        ],
    }
    if digest(combination) != final["combination_manifest_sha256"]:
        raise ValueError("P2 combination manifest digest does not bind the exact repositories and plugin")
    return {
        "formal_baseline": formal,
        "final_release": final,
        "formal_repositories": formal_repos,
        "final_repositories": final_repos,
        "formal_dependencies": formal_deps,
        "formal_instrumentation": formal_inst,
        "final_dependencies": final_deps,
        "final_instrumentation": final_inst,
    }


def _preparation_map(suite: dict) -> dict:
    entries = suite.get("preparation") or []
    result = {}
    for entry in entries:
        if not entry.get("id") or entry["id"] in result:
            raise ValueError("preparation entries need unique ids")
        result[entry["id"]] = entry
    return result


def _source_identity(
    raw: dict,
    group: ProductGroup,
    releases: dict,
    *,
    cohort_id: str,
    preparation: dict,
) -> dict:
    source = _required(raw, "source")
    if group == ProductGroup.P0_BARE_HOST:
        if (
            source.get("runtime_modules")
            or source.get("native_plugin")
            or source.get("product_dependencies")
            or source.get("instrumentation")
        ):
            raise ValueError("P0 must not claim installed MindIE modules or roles")
        dependencies, instrumentation = [], []
    else:
        dependencies, instrumentation = _roles(source, f"{group.value} source")
        expected_dependencies = (
            releases["formal_dependencies"]
            if group == ProductGroup.P1_V002_BASELINE
            else releases["final_dependencies"]
        )
        expected_instrumentation = (
            releases["formal_instrumentation"]
            if group == ProductGroup.P1_V002_BASELINE
            else releases["final_instrumentation"]
        )
        if dependencies != expected_dependencies or instrumentation != expected_instrumentation:
            raise ValueError(f"{group.value} source roles differ from its release identity")
    catalog = source.get("catalog") or {}
    if not _hex(source.get("catalog_sha256"), 64):
        raise ValueError(f"profile source needs a catalog hash: {group.value}")
    if source["catalog_sha256"] != digest(public_catalog(catalog)):
        raise ValueError(f"profile source catalog hash differs from its catalog: {group.value}")
    if source.get("benchmark") != canonical_benchmark(source.get("benchmark") or {}):
        raise ValueError(f"profile source benchmark identity is invalid: {group.value}")
    raw_modules = source.get("runtime_modules") or {}
    for name, module in raw_modules.items():
        if name not in RUNTIME_MODULE_SET:
            raise ValueError(f"profile source declares an unknown runtime module: {name}")
        if not module.get("version") or not _hex(module.get("revision"), 40):
            raise ValueError(f"profile runtime module identity is incomplete: {name}")
    modules = canonical_runtime_modules(raw_modules)
    source["runtime_modules"] = modules
    plugin = source.get("native_plugin")
    plugin_public = canonical_native_plugin(plugin)
    preparation_id = source.get("preparation_id")
    if group in {ProductGroup.P1_V002_BASELINE, ProductGroup.P2_V003_FULL} and not preparation_id:
        raise ValueError(f"profile {group.value} requires current usable preparation evidence")
    if preparation_id:
        entry = preparation.get(preparation_id)
        if entry is None:
            raise ValueError(f"profile source references unknown preparation evidence: {preparation_id}")
        if entry.get("status") != "usable":
            raise ValueError(
                f"profile source references unusable preparation evidence: {preparation_id}"
            )
        observed = entry.get("observed_runtime_modules_sha256")
        if observed and observed != digest(modules):
            raise ValueError(
                f"preparation evidence does not match this profile's current runtime modules: {preparation_id}"
            )
    if group == ProductGroup.P0_BARE_HOST:
        if modules or plugin_public is not None:
            raise ValueError("P0 must not claim installed MindIE runtime modules or a native plugin")
        if catalog.get("units") != 0:
            raise ValueError("P0 must prove a loaded-empty catalog")
        if source.get("product_version") != "bare-host":
            raise ValueError("P0 product identity must be bare-host")
        if source.get("product_revision") is not None or source.get("product_artifact_sha256") is not None:
            raise ValueError("P0 must not claim a product release revision or artifact")
        return source
    if set(modules) != RUNTIME_MODULE_SET:
        raise ValueError(f"{group.value} must report exactly the five installed runtime modules")
    if plugin_public is None:
        raise ValueError(f"{group.value} must report the installed native plugin")
    if not (
        plugin.get("name")
        and plugin.get("version")
        and _hex(plugin.get("revision"), 40)
        and plugin.get("contribution_enabled") is False
    ):
        raise ValueError(f"{group.value} native plugin identity is incomplete")
    if catalog.get("units") is None or catalog.get("units") <= 0:
        raise ValueError(f"profile {group.value} needs a loaded non-empty catalog")
    if source["benchmark"].get("revision") is None:
        raise ValueError(f"profile {group.value} needs the common benchmark runner")
    release = (
        releases["formal_repositories"]
        if group == ProductGroup.P1_V002_BASELINE
        else releases["final_repositories"]
    )
    product_release = (
        releases["formal_baseline"]
        if group == ProductGroup.P1_V002_BASELINE
        else releases["final_release"]
    )
    expected_product = (
        product_release["product_version"],
        product_release["revision"],
        product_release["artifact_sha256"],
    )
    if (
        source.get("product_version"),
        source.get("product_revision"),
        source.get("product_artifact_sha256"),
    ) != expected_product:
        raise ValueError(f"profile source product identity differs from its release: {group.value}")
    for component in dependencies:
        target = release[component]
        module = modules[component]
        if (module["version"], module["revision"]) != (target["version"], target["revision"]):
            raise ValueError(
                f"{group.value} product dependency differs from its release: {component}"
            )
    plugin_release = releases["formal_baseline" if group == ProductGroup.P1_V002_BASELINE else "final_release"]["native_plugin"]
    if (
        plugin.get("name"),
        plugin.get("version"),
        plugin.get("revision"),
    ) != (
        plugin_release.get("name"),
        plugin_release.get("version"),
        plugin_release.get("revision"),
    ):
        raise ValueError(f"{group.value} native plugin differs from its release identity")
    if group == ProductGroup.P2_V003_FULL:
        for component in instrumentation:
            target = release[component]
            module = modules[component]
            if (module["version"], module["revision"]) != (target["version"], target["revision"]):
                raise ValueError(
                    f"P2 instrumentation differs from its candidate repository: {component}"
                )
    return source


def _build_profile(
    raw: dict,
    cohort_id: str,
    releases: dict,
    preparation: dict,
    benchmark: dict,
) -> tuple[ProductProfile, dict]:
    if "group" not in raw:
        raise ValueError("product profile is missing group")
    group = ProductGroup(raw["group"])
    if group not in CORE_GROUPS:
        raise ValueError("core cohort accepts only P0/P1/P2 product profiles")
    if raw.get("cohort") != cohort_id:
        raise ValueError("profile cohort differs from its declared cohort")
    source = _source_identity(
        raw, group, releases, cohort_id=cohort_id, preparation=preparation
    )
    if source["benchmark"] != canonical_benchmark(benchmark):
        raise ValueError("profile source benchmark differs from the suite common runner")
    public_reference = "profile:" + digest(
        {key: value for key, value in raw.items() if key not in {"experiment_profile", "source"}}
    )
    profile = ProductProfile(
        profile_id=raw["profile_id"],
        group=group,
        experiment_profile=public_reference,
        protocol=raw["protocol"],
        fidelity=raw["fidelity"],
        cohort=cohort_id,
        conditions=MatchingConditions(**raw["conditions"]),
        budget=BudgetEnvelope(**raw["budget"]),
        preserved_capabilities=tuple(raw.get("preserved_capabilities", ())),
        notes=raw.get("notes", ""),
    )
    return profile, source


def _cohort_definitions(suite: dict, tasks: dict, repetitions: int) -> list[dict]:
    if suite.get("cohorts"):
        cohorts = suite["cohorts"]
    else:
        scenario = (suite.get("scenarios") or [{}])[0].get("id", "matched")
        cohorts = [{
            "id": suite.get("cohort", "main"),
            "scenario": scenario,
            "task_ids": list(tasks),
            "repetitions": repetitions,
            "required": True,
            "profiles": suite.get("profiles") or [],
            "functional_claim": {"group": "P2_v003_full", "minimum_passes": 1},
        }]
    if not cohorts:
        raise ValueError("product suite needs predeclared cohorts")
    result = []
    for cohort in cohorts:
        for field in ("id", "scenario", "task_ids", "profiles"):
            if not cohort.get(field):
                raise ValueError(f"cohort is missing {field}")
        if cohort.get("repetitions", repetitions) != repetitions:
            raise ValueError("cohort repetitions must match the suite repetitions")
        if not set(cohort["task_ids"]) <= set(tasks):
            raise ValueError(f"cohort uses an undeclared task: {cohort['id']}")
        result.append(cohort)
    ids = [cohort["id"] for cohort in result]
    if len(ids) != len(set(ids)):
        raise ValueError("cohort ids must be unique")
    task_cohorts = {}
    for cohort in result:
        for task_id in cohort["task_ids"]:
            if task_id in task_cohorts:
                raise ValueError("a task cannot belong to more than one cohort")
            task_cohorts[task_id] = cohort["id"]
    return result


def _validate_budget(profile: ProductProfile, tasks: list[dict]) -> None:
    for task in tasks:
        budget = task["source_budget"]
        if profile.budget.total_model_calls != budget["max_turns"]:
            raise ValueError("profile model-call budget differs from the actual task budget")
        if profile.budget.npu_runs != budget["max_npu_runs"]:
            raise ValueError("profile NPU budget differs from the actual task budget")
        if profile.budget.max_wall_clock_sec != budget["max_wall_clock_sec"]:
            raise ValueError("profile delivery budget differs from the declared scenario budget")
        if profile.budget.human_interventions != budget.get("human_interventions", 0):
            raise ValueError("profile intervention budget differs from the task budget")


def _cohort_plan(
    cohort: dict,
    tasks: dict,
    releases: dict,
    preparation: dict,
    benchmark: dict,
    comparisons: list[dict],
) -> dict:
    raw_profiles = cohort["profiles"]
    profiles, sources = {}, {}
    for raw in raw_profiles:
        profile, source = _build_profile(
            raw, cohort["id"], releases, preparation, benchmark
        )
        if profile.group in profiles:
            raise ValueError(f"cohort repeats a product group: {cohort['id']}")
        profiles[profile.group] = profile
        sources[profile.group] = source
    if set(profiles) != set(CORE_GROUPS):
        raise ValueError(f"cohort {cohort['id']} must declare P0, P1 and P2")
    cohort_tasks = [tasks[task_id] for task_id in cohort["task_ids"]]
    oracles = {task["task_id"]: task["oracle_digest"] for task in cohort_tasks}
    for profile in profiles.values():
        if profile.conditions.oracle_digest != digest(oracles):
            raise ValueError("profile matching conditions do not bind this cohort's task oracles")
        _validate_budget(profile, cohort_tasks)
    p0, p1, p2 = (sources[group] for group in CORE_GROUPS)
    for name in releases["formal_instrumentation"]:
        if p1["runtime_modules"][name] != p2["runtime_modules"][name]:
            raise ValueError(f"common instrumentation differs across product arms: {name}")
    if p0["benchmark"] != p1["benchmark"] or p1["benchmark"] != p2["benchmark"]:
        raise ValueError("the common benchmark runner differs across product arms")
    if p1["runtime_modules"] and p2["runtime_modules"]:
        changed = [
            name for name in releases["formal_dependencies"]
            if p1["runtime_modules"][name]["revision"] != p2["runtime_modules"][name]["revision"]
        ]
        if not changed:
            raise ValueError("P1 and P2 product dependencies are not distinct")
        if p1["native_plugin"] == p2["native_plugin"]:
            raise ValueError("P1 and P2 native plugin identities are not distinct")
    plan = plan_product_matrix(
        [MatrixTask(task["task_id"], task["task_family"], task["hardware_tier"])
         for task in cohort_tasks],
        [profiles[group] for group in CORE_GROUPS],
        repetitions=cohort["repetitions"],
    )
    required = bool(cohort.get("required"))
    required_cells = set()
    if required:
        for group in CORE_GROUPS:
            for task in cohort["task_ids"]:
                for repetition in range(1, cohort["repetitions"] + 1):
                    required_cells.add((group, task, repetition))
    functional_claim = cohort.get("functional_claim")
    if required:
        if not functional_claim:
            raise ValueError(f"required cohort needs a predeclared functional claim: {cohort['id']}")
        claim_group = ProductGroup(functional_claim.get("group", ""))
        if claim_group not in CORE_GROUPS:
            raise ValueError("functional claim group must be P0, P1 or P2")
        minimum = functional_claim.get("minimum_passes")
        if type(minimum) is not int or minimum < 1:
            raise ValueError("functional claim needs a positive minimum_passes")
        if minimum > len(cohort["task_ids"]) * cohort["repetitions"]:
            raise ValueError("functional claim minimum exceeds the cohort cells")
    cohort_comparisons = [
        comparison for comparison in comparisons if comparison.get("cohort") == cohort["id"]
    ]
    pairs = {
        (ProductGroup(item["baseline"]), ProductGroup(item["candidate"]))
        for item in cohort_comparisons if item.get("required")
    }
    required_pairs = {
        (ProductGroup.P0_BARE_HOST, ProductGroup.P1_V002_BASELINE),
        (ProductGroup.P1_V002_BASELINE, ProductGroup.P2_V003_FULL),
    }
    if required and not required_pairs <= pairs:
        raise ValueError(f"cohort {cohort['id']} must declare required P0->P1 and P1->P2 comparisons")
    return {
        "definition": cohort,
        "plan": plan,
        "profiles": profiles,
        "sources": sources,
        "tasks": cohort_tasks,
        "cells": {cell.key(): cell for cell in plan.cells},
        "required_cells": required_cells,
        "comparisons": cohort_comparisons,
        "functional_claim": functional_claim,
    }


def build_product_plan(suite: dict) -> dict:
    """Validate the predeclared product protocol and build one real plan per cohort."""
    if suite.get("schema_version") != SUITE2:
        raise ValueError("unsupported product-value suite")
    for field in ("id", "product_version", "protocol", "benchmark", "tasks"):
        _required(suite, field)
    benchmark = suite["benchmark"]
    if not _hex(benchmark.get("revision"), 40) or not _hex(benchmark.get("source_sha256"), 64):
        raise ValueError("benchmark identity needs an exact commit and source hash")
    if benchmark.get("version") is None:
        raise ValueError("benchmark identity needs a version")
    releases = _release_identities(suite)
    preparation = _preparation_map(suite)
    raw_tasks = suite["tasks"]
    if not raw_tasks or len({task["task_id"] for task in raw_tasks}) != len(raw_tasks):
        raise ValueError("product tasks must be non-empty and unique")
    tasks = {}
    for task in raw_tasks:
        for field in ("task_id", "task_family", "hardware_tier", "task_digest", "oracle_digest", "execution_tier", "role"):
            if not task.get(field):
                raise ValueError(f"task is missing {field}")
        if task["role"] not in {"seed", "eval"}:
            raise ValueError("task role must be seed or eval")
        if not _hex(task["oracle_digest"], 64):
            raise ValueError("task oracle identities must be SHA-256 values")
        budget = task.get("source_budget") or {}
        for field in ("max_turns", "max_npu_runs", "timeout_sec", "verifier_timeout_sec", "max_wall_clock_sec"):
            if budget.get(field) is None:
                raise ValueError(f"task source_budget is missing {field}: {task['task_id']}")
        tasks[task["task_id"]] = task
    repetitions = suite.get("repetitions", 1)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    definitions = _cohort_definitions(suite, tasks, repetitions)
    comparisons = suite.get("comparisons") or []
    if not comparisons:
        raise ValueError("product suite needs predeclared comparisons")
    cohorts, plan_digests, profile_ids = {}, {}, set()
    for definition in definitions:
        cohort = _cohort_plan(
            definition, tasks, releases, preparation, benchmark, comparisons
        )
        for profile in cohort["profiles"].values():
            if profile.profile_id in profile_ids:
                raise ValueError("profile ids must be unique across cohorts")
            profile_ids.add(profile.profile_id)
        cohorts[definition["id"]] = cohort
        plan_digests[definition["id"]] = cohort["plan"].digest()
    for comparison in comparisons:
        cohort = comparison.get("cohort")
        if cohort not in cohorts:
            raise ValueError("comparison references an undeclared cohort")
        baseline = ProductGroup(comparison["baseline"])
        candidate = ProductGroup(comparison["candidate"])
        if baseline == candidate or baseline not in cohorts[cohort]["profiles"]:
            raise ValueError("comparison needs two distinct declared product profiles")
        if candidate != ProductGroup.P3_V003_FLYWHEEL and candidate not in cohorts[cohort]["profiles"]:
            raise ValueError("comparison candidate is not declared in its cohort")
        if comparison.get("budget_scope", "downstream") not in {"downstream", "lifecycle"}:
            raise ValueError("comparison budget scope must be downstream or lifecycle")
    return {
        "cohorts": cohorts,
        "comparisons": comparisons,
        "tasks": tasks,
        "preparation": preparation,
        "plan_digests": plan_digests,
        "plan_sha256": digest(plan_digests),
        "benchmark": benchmark,
        **releases,
    }
