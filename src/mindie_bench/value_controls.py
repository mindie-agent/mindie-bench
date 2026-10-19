"""Control equivalence for reviewed infrastructure repairs.

[Role] Check that a replacement still measures the original experiment.
[Invariants] Model, budget and workload controls cannot change during a retest.
Source equivalence requires unchanged loaded paths or a declared runner repair.
[Timeouts & Locks] Pure in-memory comparisons; no waits or locks.
[Side-Effect Policy] Never mutate original observations or execute a retry.
"""

from __future__ import annotations

from mindie_bench.scorer.comparator import CONTROL_KEYS


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


def _runner_controls(env, patches):
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


def require_retest_controls(original, candidate, evidence):
    """Preserve experimental controls while allowing an explicit transport repair."""
    before, after = [
        _runner_controls(row["environment_fingerprint"], evidence.get("runner_patches", []))
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
