"""Metadata-only cross-source problem-family closure audit.

[Role] `audit_problem_closure` verifies explicit problem-family closure over
caller-supplied strong/candidate lineage metadata before manifest assignment or
body admission.

[Invariants]
- Strong edges cannot stop at a selected asset; cross-split is a hard failure.
- Candidate decisions need refs and match only a stable edge id or exact
  reason+ordered-pair key; repo/topic/session/bare-pair similarity proves nothing.
- `non_problem_association` isolates an unselected metadata-only container with
  denied body admission; it never merges selected families or crosses a split.
- Heldout rejects body exposure from implementation, seed distillation, static
  corpus, or dev evaluation; `body_exposed=False` review is recorded only.
- Readiness needs accepted independent Seed and Heldout families.
[Timeouts & Locks] Pure in-memory; no clocks or locks.
[Side-Effect Policy] No filesystem, network, log, DB, subprocess, or global mutation.
"""

import re
from typing import Any, NamedTuple

ASSIGNED_SPLITS = frozenset({"seed", "dev", "heldout"})
BODY_EXPOSURE_PURPOSES = frozenset(
    {"implementation", "seed_distillation", "static_corpus", "dev_evaluation"}
)
DECISION_RELATIONS = frozenset(
    {"same_problem", "distinct_problem", "non_problem_association", "unresolved"}
)
COARSE_ASSOCIATION_REASONS = frozenset(
    {
        "same_repo_declared_family_topic",
        "repository_path_candidate",
        "shared_baseline_history",
        "repository_contains_commit",
        "source_container_provenance",
    }
)
REQUIRED_ASSET_FIELDS = ("asset_id", "source_id", "source_locator")
BROAD_IDENTITY_PATTERN = re.compile(
    r"(public[-_]source:(repo|topic)|source[-_]repository:|repo:|repository:|topic:|broad[-_]topic:|same[-_]topic:)"
)
BROAD_IDENTITY_VALUES = frozenset(
    {"repo", "repository", "topic", "public-source", "public_source", "broad-topic"}
)
REPO_URL_PATTERN = re.compile(r"https?://[^/]+/[^/]+/[^/]+/?$")
EDGE_FIELDS = ("kind", "edge_id", "normalized_edge_id", "left", "right", "reason", "evidence")


class _Outcome(NamedTuple):
    reason: str = ""
    decision: dict | None = None
    container_asset_id: str = ""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _texts(value: Any) -> list[str]:
    values = (
        [value] if isinstance(value, str) else value if isinstance(value, (list, tuple, set)) else []
    )
    return sorted({_text(item) for item in values if _text(item)})


def _refs(raw: dict) -> list[str]:
    return _texts(raw.get("evidence_refs") or raw.get("evidence_ref"))


def _key(reason: str, left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{reason}|{first}|{second}" if reason else f"{first}|{second}"


def _edges(raw_edges: Any, kind: str, problems: list[str]) -> list[dict]:
    edges: list[dict] = []
    for index, raw in enumerate(raw_edges or []):
        left, right = _text(raw.get("left")), _text(raw.get("right"))
        if not left or not right:
            problems.append(f"malformed_{kind}_edge:{index}")
            continue
        reason = _text(raw.get("reason"))
        normalized = _key(reason, left, right)
        edges.append(
            {
                "kind": kind,
                "index": len(edges),
                "edge_id": _text(raw.get("edge_id")) or normalized,
                "normalized_edge_id": normalized,
                "left": left,
                "right": right,
                "reason": reason,
                "evidence": raw.get("evidence"),
            }
        )
    return edges


def _match(candidate_edges: list[dict], decisions: Any, problems: list[str]) -> dict[int, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    by_key: dict[str, list[dict]] = {}
    for edge in candidate_edges:
        by_id.setdefault(edge["edge_id"], []).append(edge)
        by_key.setdefault(edge["normalized_edge_id"], []).append(edge)
    matched: dict[int, list[dict]] = {}
    for raw in decisions or []:
        edge_id = _text(raw.get("edge_id") or raw.get("target_edge_id"))
        reason, left, right = _text(raw.get("reason")), _text(raw.get("left")), _text(raw.get("right"))
        key = edge_id or (_key(reason, left, right) if reason and left and right else "")
        edges = (by_id if edge_id else by_key).get(key, []) if key else []
        # One review may only settle one canonical edge; different reasons differ.
        if not edges or len({edge["normalized_edge_id"] for edge in edges}) > 1:
            continue
        for edge in edges:
            matched.setdefault(edge["index"], []).append(dict(raw))
    return matched


def _single(asset_id: str, asset_families: dict[str, list[str]]) -> str | None:
    family_ids = asset_families.get(asset_id, [])
    return family_ids[0] if len(family_ids) == 1 else None


def _broad(identity: str, repo_ids: set[str]) -> bool:
    value = identity.casefold().strip()
    if not value or value in BROAD_IDENTITY_VALUES or BROAD_IDENTITY_PATTERN.match(value):
        return True
    if value in repo_ids or REPO_URL_PATTERN.fullmatch(value):
        return True
    markers = ("#", "/pull/", "/issues/", "/commit/", "issue-", "pr-")
    return bool(re.fullmatch(r"[^/\s]+/[^/\s]+", value) and not any(m in value for m in markers))


def _candidate_outcome(
    edge: dict,
    rows: list[dict] | None,
    *,
    assets: dict[str, dict],
    selected: set[str],
    asset_families: dict[str, list[str]],
    families: dict[str, dict],
) -> _Outcome:
    left, right = edge["left"], edge["right"]
    if left not in assets or right not in assets:
        return _Outcome("unknown_candidate_endpoint")
    if not rows:
        return _Outcome("missing_candidate_decision")
    relations = {_text(raw.get("relation")).casefold() for raw in rows}
    containers = {_text(raw.get("container_asset_id")) for raw in rows}
    admissions = {_text(raw.get("container_body_admission")).casefold() for raw in rows}
    if len(relations) != 1 or "" in relations or len(containers) > 1 or len(admissions) > 1:
        return _Outcome("conflicting_candidate_decisions")
    relation = next(iter(relations))
    refs = sorted({ref for raw in rows for ref in _refs(raw)})
    decision = {
        "relation": relation,
        "refs": refs,
        "container": next(iter(containers)),
        "admission": next(iter(admissions)),
    }
    if relation not in DECISION_RELATIONS:
        return _Outcome("invalid_candidate_decision", decision)
    if relation == "unresolved":
        return _Outcome("candidate_unresolved", decision)
    if not refs:
        return _Outcome("decision_missing_evidence_refs", decision)
    left_selected, right_selected = left in selected, right in selected
    left_family = _single(left, asset_families)
    right_family = _single(right, asset_families)
    if relation == "distinct_problem":
        if not (left_selected and right_selected):
            return _Outcome("", decision)
        if left_family is None or right_family is None:
            return _Outcome("ambiguous_candidate_endpoint", decision)
        return _Outcome(
            "" if left_family != right_family else "distinct_decision_contradicts_family", decision
        )
    if relation == "non_problem_association":
        if edge["reason"] not in COARSE_ASSOCIATION_REASONS:
            return _Outcome("non_problem_association_reason_not_allowed", decision)
        if left_selected and right_selected:
            same = left_family is not None and left_family == right_family
            return _Outcome("" if same else "non_problem_association_selected_pair", decision)
        if (left_selected and left_family is None) or (right_selected and right_family is None):
            return _Outcome("ambiguous_candidate_endpoint", decision)
        container_id = decision["container"]
        container = assets.get(container_id)
        for bad, text in (
            (not container_id, "missing_container_asset"),
            (container is None, "unknown_container_asset"),
            (container_id in selected, "container_asset_selected"),
            (container_id not in {left, right}, "container_not_incident_endpoint"),
            (
                container is not None
                and _text(container.get("admission_scope")) != "metadata_only_container",
                "container_not_metadata_only",
            ),
            (decision["admission"] != "denied", "container_body_admission_not_denied"),
        ):
            if bad:
                return _Outcome(text, decision)
        family_splits = {
            families[fid]["split"]
            for fid, sel in ((left_family, left_selected), (right_family, right_selected))
            if sel
        }
        values = _texts(container.get("covered_splits") or container.get("splits"))
        container_splits = {
            value for value in values + [_text(container.get("split"))] if value in ASSIGNED_SPLITS
        }
        if len(container_splits) > 1 or not container_splits <= family_splits:
            return _Outcome("container_cross_split_conflict", decision)
        return _Outcome("", decision, container_id)
    if not left_selected or not right_selected:
        return _Outcome("family_merge_required", decision)
    if left_family is None or right_family is None:
        return _Outcome("ambiguous_candidate_endpoint", decision)
    if left_family == right_family:
        return _Outcome("", decision)
    return _Outcome(
        "family_merge_required"
        if families[left_family]["split"] == families[right_family]["split"]
        else "cross_split_conflict",
        decision,
    )


def audit_problem_closure(
    *,
    assets: list[dict],
    strong_edges: list[dict],
    candidate_edges: list[dict],
    claims: list[dict],
    decisions: list[dict],
    exposures: list[dict],
) -> dict:
    """Audit selected problem-family closure from metadata and review evidence.

    Families are never derived from repository, topic, session, or source-id
    similarity. Those signals only work through caller strong edges and explicit
    candidate decisions. A blocked family keeps the whole round unready.
    """
    assets_by_id: dict[str, dict] = {}
    problems: list[str] = []
    families: dict[str, dict] = {}
    selected: set[str] = set()
    asset_families: dict[str, list[str]] = {}
    cross_conflicts: list[dict] = []
    cross_keys: set[tuple] = set()
    unresolved_rows: list[dict] = []
    unresolved_keys: set[tuple] = set()
    uncovered: list[dict] = []
    isolated: set[str] = set()

    def reason(family_ids: list[str], text: str) -> None:
        for family_id in family_ids:
            if family := families.get(family_id):
                family["reasons"].add(text)

    def cross(kind, family_ids, splits, detail):
        ordered = sorted({fid for fid in family_ids if fid})
        for fid in ordered:
            if family := families.get(fid):
                family["reasons"].add("cross_split_conflict")
        if ordered and (kind, tuple(ordered)) not in cross_keys:
            cross_keys.add((kind, tuple(ordered)))
            cross_conflicts.append(
                {
                    "kind": kind,
                    "family_ids": ordered,
                    "splits": {fid: splits.get(fid, "") for fid in ordered},
                    "detail": detail,
                }
            )

    def check_splits(kind: str, family_ids: list[str], detail: dict) -> None:
        splits = {family_id: families[family_id]["split"] for family_id in family_ids}
        if len(set(splits.values())) > 1:
            cross(kind, family_ids, splits, detail)

    def incident(edge: dict) -> list[str]:
        family_ids: set[str] = set()
        for endpoint in (edge["left"], edge["right"]):
            if endpoint in selected:
                family_ids.update(asset_families.get(endpoint, []))
        return sorted(family_ids)

    def unresolved(edge, family_ids, text, decision):
        ordered = sorted({fid for fid in family_ids if fid})
        reason(ordered, text)
        key = (edge["kind"], edge["edge_id"], text, tuple(ordered))
        if key in unresolved_keys:
            return
        unresolved_keys.add(key)
        record = {field: edge[field] for field in EDGE_FIELDS}
        record.update(
            {
                "resolution": "unresolved",
                "blocking_reason": text,
                "family_ids": ordered,
                "decision": decision,
            }
        )
        unresolved_rows.append(record)

    for raw in assets or []:
        asset_id = _text(raw.get("asset_id"))
        if not asset_id:
            problems.append("asset_missing_asset_id")
        elif asset_id in assets_by_id:
            problems.append("duplicate_asset_id")
        else:
            assets_by_id[asset_id] = dict(raw)

    for index, raw in enumerate(claims or []):
        family_id = _text(raw.get("family_id"))
        split = _text(raw.get("split")).casefold()
        asset_ids, refs = _texts(raw.get("asset_ids")), _refs(raw)
        selected.update(asset_ids)
        if not family_id:
            problems.append(f"claim_missing_family_id:{index}")
            continue
        identity = _text(raw.get("problem_identity"))
        scope = _text(raw.get("problem_identity_scope")).casefold()
        reasons = {
            name
            for name, bad in (
                ("invalid_split", split not in ASSIGNED_SPLITS),
                ("missing_claim_assets", not asset_ids),
                ("missing_problem_identity", not identity),
                ("problem_identity_scope_not_problem", scope != "problem"),
                ("missing_claim_evidence_refs", not refs),
                ("claim_not_resolved", raw.get("resolved") is False),
            )
            if bad
        }
        unknown = [asset_id for asset_id in asset_ids if asset_id not in assets_by_id]
        if any(
            not _text(assets_by_id[asset_id].get(field))
            for asset_id in asset_ids
            if asset_id in assets_by_id
            for field in REQUIRED_ASSET_FIELDS
        ):
            reasons.add("missing_asset_metadata")
        if unknown:
            reasons.add("unknown_claim_asset")
        family = families.get(family_id)
        if family is not None:
            family["reasons"].update(reasons, {"duplicate_family_id"})
            family["covered"] = sorted(set(family["covered"]) | set(asset_ids))
            family["refs"].update(refs)
            if split and family["split"] and split != family["split"]:
                cross(
                    "duplicate_family_split",
                    [family_id],
                    {family_id: split},
                    {"splits": sorted({split, family["split"]})},
                )
            continue
        families[family_id] = {
            "family_id": family_id,
            "split": split,
            "identity": identity,
            "scope": scope,
            "refs": set(refs),
            "covered": sorted(set(asset_ids)),
            "reasons": reasons,
            "exposed": False,
            "strong": 0,
            "candidates": 0,
            "exposures": [],
        }

    if not families:
        problems.append("no_claim_families")

    for family_id, family in families.items():
        repos = set()
        for asset_id in family["covered"]:
            asset_families.setdefault(asset_id, []).append(family_id)
            if asset := assets_by_id.get(asset_id):
                repos.add(_text(asset.get("repo_identity")).casefold())
        if _broad(family["identity"], repos):
            family["reasons"].add("broad_repository_or_topic_scope")

    for asset_id, family_ids in sorted(asset_families.items()):
        if len(family_ids) < 2:
            continue
        reason(family_ids, "asset_in_multiple_families")
        check_splits("asset_in_multiple_families_cross_split", family_ids, {"asset_id": asset_id})
    identities: dict[str, list[str]] = {}
    for family_id, family in families.items():
        if family["identity"]:
            identities.setdefault(family["identity"].casefold(), []).append(family_id)
    for identity, family_ids in sorted(identities.items()):
        if len(family_ids) < 2:
            continue
        reason(family_ids, "duplicate_problem_identity")
        check_splits("problem_identity_cross_split", family_ids, {"problem_identity": identity})
    sessions: dict[str, set[str]] = {}
    for asset_id, family_ids in asset_families.items():
        if session_id := _text(assets_by_id.get(asset_id, {}).get("session_id")):
            sessions.setdefault(session_id, set()).update(family_ids)
    for session_id, family_ids in sorted(sessions.items()):
        check_splits("session_cross_split", sorted(family_ids), {"session_id": session_id})

    strong_records = _edges(strong_edges, "strong", problems)
    candidate_records = _edges(candidate_edges, "candidate", problems)
    matched = _match(candidate_records, decisions, problems)

    for edge in strong_records:
        family_ids = incident(edge)
        if not family_ids:
            continue
        for family_id in family_ids:
            families[family_id]["strong"] += 1
        left, right = edge["left"], edge["right"]
        left_selected, right_selected = left in selected, right in selected
        if not left_selected or not right_selected:
            external = right if left_selected else left
            family = families[family_ids[0]]
            family["reasons"].add("closure_incomplete")
            uncovered.append(
                {
                    "kind": "strong",
                    "edge_id": edge["edge_id"],
                    "selected_asset_id": left if left_selected else right,
                    "external_asset_id": external,
                    "external_asset_known": external in assets_by_id,
                    "family_ids": family_ids,
                    "reason": edge["reason"],
                    "evidence": edge["evidence"],
                }
            )
            unresolved(edge, family_ids, "closure_incomplete", None)
            continue
        left_family, right_family = _single(left, asset_families), _single(right, asset_families)
        if left_family is None or right_family is None:
            reason(family_ids, "ambiguous_strong_edge_family")
        elif left_family != right_family:
            splits = {family_id: families[family_id]["split"] for family_id in family_ids}
            if len(set(splits.values())) > 1:
                cross(
                    "strong_edge_cross_split",
                    family_ids,
                    splits,
                    {"left": left, "right": right, "reason": edge["reason"]},
                )
                unresolved(edge, family_ids, "cross_split_conflict", None)
            else:
                reason(family_ids, "family_merge_required")
                unresolved(edge, family_ids, "family_merge_required", None)

    for edge in candidate_records:
        family_ids = incident(edge)
        if not family_ids:
            continue
        for family_id in family_ids:
            families[family_id]["candidates"] += 1
        outcome = _candidate_outcome(
            edge,
            matched.get(edge["index"]),
            assets=assets_by_id,
            selected=selected,
            asset_families=asset_families,
            families=families,
        )
        if not outcome.reason:
            for family_id in family_ids:
                families[family_id]["refs"].update((outcome.decision or {}).get("refs", []))
            if outcome.container_asset_id:
                isolated.add(outcome.container_asset_id)
            continue
        if outcome.reason == "cross_split_conflict":
            check_splits(
                "candidate_same_problem_cross_split",
                family_ids,
                {"left": edge["left"], "right": edge["right"], "reason": edge["reason"]},
            )
        elif outcome.reason == "container_cross_split_conflict":
            splits = {family_id: families[family_id]["split"] for family_id in family_ids}
            cross(
                "container_cross_split",
                family_ids,
                splits,
                {"left": edge["left"], "right": edge["right"], "reason": edge["reason"]},
            )
        if outcome.reason == "family_merge_required":
            reason(family_ids, "family_merge_required")
        unresolved(edge, family_ids, outcome.reason, outcome.decision)

    for raw in exposures or []:
        purpose = _text(raw.get("purpose")).casefold()
        body = raw.get("body_exposed") is not False
        family_id = _text(raw.get("family_id"))
        asset_ids = _texts(raw.get("asset_ids"))
        targets = {family_id} if family_id in families else set()
        targets.update(fid for asset_id in asset_ids for fid in asset_families.get(asset_id, []))
        if not targets:
            continue
        detail = {
            "purpose": purpose,
            "body_exposed": body,
            "context_id": _text(raw.get("context_id")),
            "observed_at": _text(raw.get("observed_at")),
            "evidence_refs": _refs(raw),
            "family_id": family_id,
            "asset_ids": asset_ids,
        }
        for target in sorted(targets):
            family = families[target]
            family["exposures"].append(dict(detail))
            family["refs"].update(detail["evidence_refs"])
            if not detail["evidence_refs"]:
                family["reasons"].add("exposure_missing_evidence_ref")
            if body and family["split"] == "heldout" and purpose != "heldout_task":
                family["exposed"] = True
                text = (
                    f"{purpose}_exposure_on_heldout"
                    if purpose in BODY_EXPOSURE_PURPOSES
                    else ("body_exposure_under_non_body_purpose")
                )
                family["reasons"].add(text)

    for family in families.values():
        family["status"] = (
            "failed"
            if "cross_split_conflict" in family["reasons"]
            else "blocked"
            if family["reasons"]
            else "accepted"
        )
    accepted = [family for family in families.values() if family["status"] == "accepted"]
    accepted_ids = sorted(family["family_id"] for family in accepted)
    identities = {family["identity"].casefold() for family in accepted}
    has_seed = any(family["split"] == "seed" for family in accepted)
    has_heldout = any(family["split"] == "heldout" for family in accepted)
    incomplete = bool(uncovered)
    all_accepted = len(accepted) == len(families)
    readiness = (
        set()
        if incomplete
        else {
            name
            for name, bad in (
                ("fewer_than_two_accepted_independent_families", len(accepted) < 2),
                ("duplicate_accepted_problem_identity", len(identities) != len(accepted)),
                ("missing_accepted_seed_family", not has_seed),
                ("missing_accepted_heldout_family", not has_heldout),
                ("selected_family_not_accepted", not all_accepted),
            )
            if bad
        }
    )
    ready = (
        not problems
        and not cross_conflicts
        and not incomplete
        and all_accepted
        and len(accepted) >= 2
        and has_seed
        and has_heldout
        and len(identities) == len(accepted)
    )
    blocking = (
        set(problems)
        | readiness
        | {reason for family in families.values() for reason in family["reasons"]}
    )
    status = (
        "cross_split_conflict"
        if cross_conflicts
        else "closure_incomplete"
        if incomplete
        else ("ready_for_seed_heldout" if ready else "blocked")
    )
    reports = [
        {
            "family_id": family["family_id"],
            "split": family["split"],
            "problem_identity": family["identity"],
            "problem_identity_scope": family["scope"],
            "status": family["status"],
            "accepted": family["status"] == "accepted",
            "reasons": sorted(family["reasons"]),
            "covered_asset_ids": sorted(family["covered"]),
            "evidence_refs": sorted(family["refs"]),
            "incident_strong_edge_count": family["strong"],
            "incident_candidate_edge_count": family["candidates"],
            "exposures": family["exposures"],
            "heldout_exposed": family["exposed"],
        }
        for family in sorted(families.values(), key=lambda item: item["family_id"])
    ]
    unselected = sorted(set(assets_by_id) - selected)
    return {
        "status": status,
        "ready": ready,
        "ready_for_seed_heldout": ready,
        "accepted_claims": accepted_ids,
        "accepted_family_ids": accepted_ids,
        "families": reports,
        "selected_asset_ids": sorted(selected),
        "unselected_count": len(unselected),
        "unselected_asset_ids": unselected,
        "isolated_container_asset_ids": sorted(isolated),
        "closure_incomplete": incomplete,
        "uncovered_external_nodes": uncovered,
        "cross_split_conflicts": cross_conflicts,
        "unresolved_edges": sorted(
            unresolved_rows, key=lambda edge: (edge["kind"], edge["edge_id"], edge["blocking_reason"])
        ),
        "blocking_reasons": sorted(blocking),
        "ignored_edges": [],
    }
