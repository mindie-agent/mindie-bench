"""Control equivalence for reviewed infrastructure repairs.

[Role] Check that a replacement still measures the original experiment.
[Invariants] Model, budget and workload controls cannot change during a retest.
Changed instrumentation is disclosed separately from unchanged treatment paths.
[Timeouts & Locks] Comparisons are pure; explicit CLI provenance reads allow
five seconds for each Git query. No locks or external requests.
[Side-Effect Policy] Never mutate original observations or execute a retry.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from mindie_bench.scorer.comparator import CONTROL_KEYS


def runner_source():
    """Identify the loaded distribution, including noneditable VCS installs."""
    from mindie_bench.benchmarks.value_common import git_source
    from mindie_bench.integrations.component_runtime import package_metadata
    from mindie_bench.runner.orchestrator import implementation_digest

    name = "mindie-bench"
    metadata = package_metadata(name)
    state = {"dirty": None}
    if metadata["editable"]:
        origin = json.loads(importlib.metadata.distribution(name).read_text("direct_url.json"))
        location = urlsplit(origin["url"])
        if location.scheme != "file":
            raise ValueError("editable runner provenance needs its recorded local source URL")
        source = Path(url2pathname(location.path))
        metadata = package_metadata(name, [source])
        if metadata["installation"] != "explicit_source":
            raise ValueError("loaded runner does not belong to its editable distribution")
        state = git_source(source)
    return {
        key: metadata[key] for key in ("revision", "version", "installation", "editable")
    } | state | {"source_sha256": implementation_digest()}


def runner_identity(source):
    return {key: source.get(key) for key in ("revision", "version", "source_sha256")}


def require_collector_source(expected, actual=None):
    """The CLI binds declarations to the collector it actually loaded."""
    source = runner_source() if actual is None else actual
    if (
        not expected
        or any(source.get(key) != expected.get(key) for key in ("version", "source_sha256"))
        or (source.get("revision") is not None and source["revision"] != expected.get("revision"))
        or source.get("dirty") is True
    ):
        raise ValueError("declared release runner differs from the loaded collector source")
    return source


def project_component_runner(report):
    fields = ("revision", "version", "source_sha256", "installation", "editable", "dirty")
    return {
        "case_ids": [case["id"] for case in report["cases"]],
        **{
            key: {field: value for field, value in report.get(key, {}).items() if field in fields}
            for key in ("runner_source", "runner_source_after")
        },
    }


def runner_release_problems(evidence):
    """Review historical runner identities without relabelling measured sources."""
    target = evidence.get("runner_release", {})
    if (
        not all(target.get(key) for key in ("revision", "version", "source_sha256"))
        or target.get("revision") != evidence["component_revisions"].get("mindie-bench")
    ):
        return ["release_runner_identity_missing_or_mismatched"]
    reviews = evidence.get("runner_release_reviews", [])
    for review in reviews:
        if review["kind"] not in {"trial", "component"} or not review["scope"]:
            raise ValueError("runner release reviews need a measured kind and a review scope")
        if review["release"] != target:
            raise ValueError("runner release review targets a different final source")

    def reviewed(kind, measured):
        return any(
            review["kind"] == kind and review["measured"] == measured for review in reviews
        )

    problems = []
    attempts = [
        row for rows in evidence["studies"].values() for row in rows
    ] + [item["receipt"] for item in evidence.get("excluded_attempts", [])]
    for row in attempts:
        env = row["environment_fingerprint"]
        measured = {
            "revision": env.get("component_revisions", {}).get("mindie-bench"),
            "version": env.get("benchmark_version"),
            "source_sha256": env.get("benchmark_source_sha256"),
        }
        if not measured["version"] or not measured["source_sha256"]:
            problems.append(f"measured_runner_identity_missing:{row['trial_id']}")
        elif measured != target and not reviewed("trial", measured):
            problems.append(f"unreviewed_release_runner:{row['trial_id']}")
    case_ids = {case["id"] for case in evidence["component_cases"]}
    observed = []
    for report in evidence.get("component_runner_reports", []):
        observed.extend(report["case_ids"])
        measured = {key: report[key] for key in ("runner_source", "runner_source_after")}
        sources = list(measured.values())
        if not all(source.get("revision") or source.get("source_sha256") for source in sources):
            problems.append("component_runner_identity_missing")
        elif not (
            all(runner_identity(source) == target and source.get("dirty") is not True
                for source in sources)
            or reviewed("component", measured)
        ):
            problems.append(f"unreviewed_component_runner:{','.join(report['case_ids'])}")
    if len(observed) != len(set(observed)) or set(observed) != case_ids:
        problems.append("component_runner_coverage_mismatch")
    return problems


def runner_release_markdown(report):
    target = report.get("runner_release")
    if not target:
        return []
    lines = [
        "", "## Measurement versions and scope", "",
        f"Release runner: `mindie-bench` {target['version']} (`{target['revision']}`). "
        "Task evaluations ran on the historical runner revisions listed below; later changes "
        "in `0.3.0` cover cleanup, provenance checks, and report publication without rewriting "
        "task inputs, models, budgets, or recorded attempt costs.",
    ]
    grouped: dict[tuple[str, str], list[str]] = {}
    for review in report.get("runner_release_reviews", []):
        measured = review["measured"]
        identity = (
            f"`{measured.get('version')} at {measured.get('revision') or 'revision unrecorded'}`"
            if review["kind"] == "trial" else "`component scenario sources retained in the data`"
        )
        items = grouped.setdefault((review["kind"], review["scope"]), [])
        if identity not in items:
            items.append(identity)
    for (kind, scope), identities in grouped.items():
        lines.append(f"- {kind} ({', '.join(identities)}): {scope}")
    return lines


def expected_cells(study):
    if (
        study["repetitions"] < 1
        or not study["tasks"]
        or len(set(study["groups"])) != len(study["groups"])
    ):
        raise ValueError("a value study needs distinct groups, tasks and repetitions")
    return set(itertools.product(study["tasks"], study["groups"], range(1, study["repetitions"] + 1)))


def infrastructure_markdown(patches):
    if not patches:
        return []
    return [
        "", "## Reviewed benchmark infrastructure changes", "",
        "Disclosed non-treatment instrumentation updates (raw receipt identities and "
        "original attempt costs remain preserved; component scenarios use the final release source):",
        *[
            f"- {patch['component']}: `{patch['before']}` to `{patch['after']}` "
            f"(studies: {', '.join(patch['studies'])}). {patch['scope']}"
            for patch in patches
        ],
    ]


def source_matches(component, actual, expected, group, patches):
    if actual == expected:
        return actual is not None
    return any(
        patch["component"] == component and patch["before"] == actual
        and patch["after"] == expected and group in patch["groups"] and patch["reason"]
        and patch.get("loaded_paths")
        and all(
            item["before"] and item["before"] == item["after"]
            for item in patch["loaded_paths"]
        )
        for patch in patches
    )


def validate_infrastructure_patches(suite, evidence):
    """Scope reviewed revisions to declared, non-treatment instrumentation."""
    allowed = set(suite.get("instrumentation_components", []))
    studies = {study["id"]: study for study in suite["studies"]}
    treatment = {
        component
        for study in studies.values()
        for components in study["group_components"].values()
        for component in components
    } | {study["harness"] for study in studies.values()}
    patches = evidence.get("infrastructure_patches", [])
    for patch in patches:
        component = patch["component"]
        if component not in allowed or component in treatment or component == "mindie-bench":
            raise ValueError(f"infrastructure mapping cannot change a treatment: {component}")
        if (
            not patch["before"]
            or not patch["after"]
            or patch["before"] == patch["after"]
            or patch["after"] != evidence["component_revisions"].get(component)
            or not patch["scope"]
        ):
            raise ValueError("infrastructure mapping needs exact revisions and a review scope")
        if not patch["studies"] or not set(patch["studies"]) <= studies.keys():
            raise ValueError("infrastructure mapping must name its declared studies")
    return patches


def infrastructure_source_matches(component, actual, expected, study_id, patches):
    """Accept a disclosed source change without claiming its code is unchanged."""
    return any(
        patch["component"] == component
        and patch["before"] == actual
        and patch["after"] == expected
        and study_id in patch["studies"]
        for patch in patches
    )


def reviewed_runner_controls(env, patches):
    normalized = dict(env)
    revisions = dict(env.get("component_revisions", {}))
    for patch in patches:
        if normalized.get("benchmark_source_sha256") != patch["before"]:
            continue
        if not patch["reason"] or patch["before"] == patch["after"]:
            raise ValueError("runner patch needs distinct identities and a reason")
        normalized["benchmark_source_sha256"] = patch["after"]
        for field in ("version", "revision"):
            value = normalized.get("benchmark_version") if field == "version" else revisions.get("mindie-bench")
            if patch.get(f"before_{field}") is not None and value == patch[f"before_{field}"]:
                if field == "version":
                    normalized["benchmark_version"] = patch[f"after_{field}"]
                else:
                    revisions["mindie-bench"] = patch[f"after_{field}"]
    normalized["component_revisions"] = revisions
    return normalized


def require_retest_controls(
    original, candidate, evidence, *, study_id=None, infrastructure_patches=()
):
    """Preserve experimental controls while allowing an explicit transport repair."""
    before, after = [
        reviewed_runner_controls(row["environment_fingerprint"], evidence.get("runner_patches", []))
        for row in (original, candidate)
    ]
    for field in (*CONTROL_KEYS, "components", "knowledge_mode", "skills_sha256", "corpus_view_sha256"):
        if before.get(field) != after.get(field):
            raise ValueError(f"a retest changed its experimental control: {field}")
    old, new = before["component_revisions"], after["component_revisions"]
    for component in old.keys() | new.keys():
        if old.get(component) == new.get(component):
            continue
        target = evidence["component_revisions"].get(component)
        if not all(
            source_matches(
                component, revision, target, original["ablation_group"],
                evidence.get("source_patches", []),
            )
            or infrastructure_source_matches(
                component, revision, target, study_id, infrastructure_patches
            )
            for revision in (old.get(component), new.get(component))
        ):
            raise ValueError(f"a retest changed an unreviewed component: {component}")


def retest_chain_problems(excluded, selected_ids):
    successors = {
        item["receipt"]["trial_id"]: item["replacement_trial_id"]
        for item in excluded if item.get("replacement_trial_id")
    }
    problems = []
    for original in successors:
        cursor, visited = original, set()
        while cursor in successors:
            if cursor in visited:
                raise ValueError("a retest history cannot contain a cycle")
            visited.add(cursor)
            cursor = successors[cursor]
        if cursor not in selected_ids:
            problems.append(f"retest_chain_unsettled:{original}")
    return problems


def first_attempt_statuses(rows, excluded):
    predecessors = {
        item["replacement_trial_id"]: item["receipt"]
        for item in excluded if item.get("replacement_trial_id")
    }
    statuses = {}
    for row in rows:
        cursor, status = row.trial_id, row.terminal_status.value
        while cursor in predecessors:
            original = predecessors[cursor]
            cursor, status = original["trial_id"], original["terminal_status"]
        statuses[row.trial_id] = status
    return statuses
