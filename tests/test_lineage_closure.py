"""Contract tests for metadata-only cross-source problem closure auditing."""

from __future__ import annotations

import pytest

from mindie_bench.mining.lineage_closure import audit_problem_closure

PR_ONE = "https://github.com/example/runtime/pull/101"
PR_TWO = "https://github.com/example/runtime/pull/202"
PR_THREE = "https://github.com/example/runtime/pull/303"


def _asset(
    asset_id: str,
    *,
    session: str | None = None,
    repo: str = "example/runtime",
    revision: str = "a" * 40,
) -> dict:
    return {
        "asset_id": asset_id,
        "source_id": f"source:{asset_id}",
        "session_id": session or f"session:{asset_id}",
        "repo_identity": repo,
        "revision": revision,
        "source_locator": f"locator:{asset_id}",
    }


def _claim(
    family_id: str,
    split: str,
    asset_ids: list[str],
    problem_identity: str,
    *,
    refs: list[str] | None = None,
    scope: str = "problem",
) -> dict:
    return {
        "family_id": family_id,
        "split": split,
        "asset_ids": asset_ids,
        "problem_identity": problem_identity,
        "problem_identity_scope": scope,
        "evidence_refs": refs if refs is not None else [f"claim:{family_id}"],
    }


def _strong(left: str, right: str, reason: str = "exact_session_id") -> dict:
    return {"left": left, "right": right, "reason": reason, "evidence": {"reviewed": "strong"}}


def _candidate(
    left: str,
    right: str,
    *,
    edge_id: str | None = None,
    reason: str = "related_source_reference",
) -> dict:
    row = {"left": left, "right": right, "reason": reason, "evidence": {"claim": "candidate"}}
    if edge_id is not None:
        row["edge_id"] = edge_id
    return row


def _decision(edge_id: str, relation: str, *, refs: list[str] | None = None) -> dict:
    return {
        "edge_id": edge_id,
        "relation": relation,
        "evidence_refs": refs if refs is not None else [f"review:{edge_id}"],
        "reason": f"reviewed {relation}",
    }


def _exposure(
    *,
    family_id: str | None = None,
    asset_ids: list[str] | None = None,
    purpose: str = "implementation",
    evidence_ref: str = "exposure:review:1",
) -> dict:
    return {
        "family_id": family_id,
        "asset_ids": asset_ids or [],
        "purpose": purpose,
        "context_id": "implementation-run-7",
        "observed_at": "2026-01-02T03:04:05Z",
        "evidence_ref": evidence_ref,
    }


def _container(asset_id: str, *, split: str | None = None) -> dict:
    row = {
        "asset_id": asset_id,
        "source_id": f"source:{asset_id}",
        "source_locator": f"locator:{asset_id}",
        "admission_scope": "metadata_only_container",
    }
    if split is not None:
        row["split"] = split
    return row


def _association(edge_id: str, container_id: str, *, admission: str = "denied") -> dict:
    return {
        "edge_id": edge_id,
        "relation": "non_problem_association",
        "container_asset_id": container_id,
        "container_body_admission": admission,
        "evidence_refs": [f"review:{edge_id}"],
    }


def _families(report: dict) -> dict[str, dict]:
    return {family["family_id"]: family for family in report["families"]}


def test_same_problem_cross_source_closure_and_independent_seed_heldout_ready():
    assets = [
        _asset("transcript-1", session="session:one"),
        _asset("commit-1", session="session:one"),
        _asset("commit-2", session="session:two"),
        _asset("heldout-transcript", session="session:heldout"),
        _asset("heldout-commit", session="session:heldout"),
    ]
    claims = [
        _claim("seed-family", "seed", ["transcript-1", "commit-1", "commit-2"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout-transcript", "heldout-commit"], PR_TWO),
    ]
    strong_edges = [
        _strong("transcript-1", "commit-1"),
        _strong("heldout-transcript", "heldout-commit"),
    ]
    candidate_edges = [
        _candidate("commit-1", "commit-2", edge_id="same-problem-candidate"),
        _candidate("transcript-1", "heldout-transcript", edge_id="distinct-candidate"),
    ]
    decisions = [
        _decision("same-problem-candidate", "same_problem"),
        _decision("distinct-candidate", "distinct_problem"),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=strong_edges,
        candidate_edges=candidate_edges,
        claims=claims,
        decisions=decisions,
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is True
    assert report["status"] == "ready_for_seed_heldout"
    assert report["accepted_claims"] == ["heldout-family", "seed-family"]
    families = _families(report)
    assert families["seed-family"]["status"] == "accepted"
    assert families["seed-family"]["covered_asset_ids"] == ["commit-1", "commit-2", "transcript-1"]
    assert families["seed-family"]["incident_strong_edge_count"] == 1
    assert families["seed-family"]["incident_candidate_edge_count"] == 2
    assert families["heldout-family"]["incident_candidate_edge_count"] == 1
    assert {"claim:seed-family", "review:same-problem-candidate", "review:distinct-candidate"} <= set(
        families["seed-family"]["evidence_refs"]
    )
    assert report["unselected_count"] == 0
    assert report["cross_split_conflicts"] == []
    assert report["ignored_edges"] == []


@pytest.mark.parametrize("external_known", [True, False])
def test_strong_boundary_external_node_reports_closure_incomplete(external_known: bool):
    assets = [_asset("selected", session="session:one"), _asset("heldout", session="session:h")]
    if external_known:
        assets.append(_asset("external", session="session:one"))
    claims = [
        _claim("seed-family", "seed", ["selected"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[_strong("selected", "external")],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["closure_incomplete"] is True
    assert report["status"] == "closure_incomplete"
    seed = _families(report)["seed-family"]
    assert seed["status"] == "blocked"
    assert "closure_incomplete" in seed["reasons"]
    assert seed["incident_strong_edge_count"] == 1
    uncovered = report["uncovered_external_nodes"]
    assert uncovered[0]["external_asset_id"] == "external"
    assert uncovered[0]["external_asset_known"] is external_known
    assert "external" not in seed["covered_asset_ids"]
    assert report["unselected_count"] == (1 if external_known else 0)


def test_strong_edge_across_splits_is_hard_failure():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[_strong("seed", "heldout")],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["status"] == "cross_split_conflict"
    assert report["cross_split_conflicts"][0]["kind"] == "strong_edge_cross_split"
    assert {family["status"] for family in report["families"]} == {"failed"}


@pytest.mark.parametrize(
    ("decision_relation", "expected_reason"),
    [(None, "missing_candidate_decision"), ("unresolved", "candidate_unresolved")],
)
def test_unresolved_candidate_across_split_blocks(decision_relation, expected_reason):
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    decisions = (
        [_decision("cross-split-candidate", decision_relation)] if decision_relation is not None else []
    )

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[_candidate("seed", "heldout", edge_id="cross-split-candidate")],
        claims=claims,
        decisions=decisions,
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["status"] == "blocked"
    assert report["unresolved_edges"][0]["blocking_reason"] == expected_reason
    assert {family["status"] for family in report["families"]} == {"blocked"}
    assert {family["family_id"] for family in report["families"]} == {
        "seed-family",
        "heldout-family",
    }


def test_candidate_external_endpoints_stay_unresolved_even_when_not_selected():
    assets = [
        _asset("selected", session="session:one"),
        _asset("heldout", session="session:h"),
        _asset("known-unselected", session="session:x"),
    ]
    claims = [
        _claim("seed-family", "seed", ["selected"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    candidate_edges = [
        _candidate("selected", "unknown-external", edge_id="unknown-candidate"),
        _candidate("selected", "known-unselected", edge_id="known-unselected-candidate"),
    ]
    decisions = [_decision("unknown-candidate", "distinct_problem")]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=candidate_edges,
        claims=claims,
        decisions=decisions,
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    reasons = {edge["blocking_reason"] for edge in report["unresolved_edges"]}
    assert reasons == {"unknown_candidate_endpoint", "missing_candidate_decision"}
    seed = _families(report)["seed-family"]
    assert set(seed["reasons"]) == reasons
    assert report["unselected_count"] == 1
    assert "known-unselected" not in seed["covered_asset_ids"]


def test_explicit_distinct_metadata_releases_candidate_but_missing_refs_do_not():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    # No stable edge id: the decision must match the normalized endpoint/reason key.
    candidate_edges = [_candidate("seed", "heldout")]
    normalized_edge_id = "related_source_reference|heldout|seed"
    reviewed = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=candidate_edges,
        claims=claims,
        decisions=[_decision(normalized_edge_id, "distinct_problem")],
        exposures=[],
    )
    unreferenced = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=candidate_edges,
        claims=claims,
        decisions=[_decision(normalized_edge_id, "distinct_problem", refs=[])],
        exposures=[],
    )

    assert reviewed["ready_for_seed_heldout"] is True
    assert reviewed["unresolved_edges"] == []
    assert unreferenced["ready_for_seed_heldout"] is False
    assert unreferenced["unresolved_edges"][0]["blocking_reason"] == ("decision_missing_evidence_refs")


@pytest.mark.parametrize(
    "purpose", ["implementation", "seed_distillation", "static_corpus", "dev_evaluation"]
)
@pytest.mark.parametrize("target_by", ["family_id", "asset_ids"])
def test_body_exposure_rejects_heldout(purpose: str, target_by: str):
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    exposure = (
        _exposure(family_id="heldout-family", purpose=purpose)
        if target_by == "family_id"
        else _exposure(asset_ids=["heldout"], purpose=purpose)
    )

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[exposure],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["accepted_claims"] == ["seed-family"]
    heldout = _families(report)["heldout-family"]
    assert heldout["status"] == "blocked"
    assert heldout["heldout_exposed"] is True
    assert any("exposure_on_heldout" in reason for reason in heldout["reasons"])


@pytest.mark.parametrize(
    ("identity", "scope", "expected_reason"),
    [
        ("public-source:repo:example/runtime", "problem", "broad_repository_or_topic_scope"),
        ("topic:runtime-performance", "problem", "broad_repository_or_topic_scope"),
        (PR_ONE, "repo", "problem_identity_scope_not_problem"),
    ],
)
def test_broad_repository_or_topic_scope_cannot_pretend_problem_identity(
    identity: str, scope: str, expected_reason: str
):
    assets = [
        _asset("seed-transcript", session="session:one"),
        _asset("seed-commit", session="session:one"),
        _asset("heldout", session="session:h"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed-transcript", "seed-commit"], identity, scope=scope),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[_strong("seed-transcript", "seed-commit")],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    seed = _families(report)["seed-family"]
    assert seed["status"] == "blocked"
    assert expected_reason in seed["reasons"]
    assert report["accepted_claims"] == ["heldout-family"]


def test_same_family_source_replicas_do_not_satisfy_two_family_readiness():
    assets = [
        _asset("transcript", session="session:one"),
        _asset("commit", session="session:one"),
    ]
    claims = [_claim("only-family", "seed", ["transcript", "commit"], PR_ONE)]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[_strong("transcript", "commit")],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["accepted_claims"] == ["only-family"]
    assert "fewer_than_two_accepted_independent_families" in report["blocking_reasons"]
    assert "missing_accepted_heldout_family" in report["blocking_reasons"]
    assert report["unselected_count"] == 0


@pytest.mark.parametrize(
    ("second_split", "expected_status", "expected_reason"),
    [
        ("seed", "blocked", "family_merge_required"),
        ("dev", "cross_split_conflict", "cross_split_conflict"),
    ],
)
def test_same_problem_candidate_requires_merge_or_fails_cross_split(
    second_split: str, expected_status: str, expected_reason: str
):
    assets = [_asset("seed", session="session:one"), _asset("other", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("other-family", second_split, ["other"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[_candidate("seed", "other", edge_id="same-problem")],
        claims=claims,
        decisions=[_decision("same-problem", "same_problem")],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["status"] == expected_status
    for family in report["families"]:
        assert family["status"] == ("failed" if expected_status != "blocked" else "blocked")
        assert expected_reason in family["reasons"]


def test_sparse_commit_pr_and_task_nodes_use_claim_refs_without_session_fields():
    assets = [
        {"asset_id": "commit-1", "source_id": "source:commit-1", "source_locator": "locator:commit-1"},
        {"asset_id": "pr-1", "source_id": "source:pr-1", "source_locator": "locator:pr-1"},
        {"asset_id": "task-1", "source_id": "source:task-1", "source_locator": "locator:task-1"},
    ]
    claims = [
        _claim("seed-family", "seed", ["commit-1", "pr-1"], PR_ONE, refs=["review:seed"]),
        _claim("heldout-family", "heldout", ["task-1"], PR_TWO, refs=["review:heldout"]),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[_strong("commit-1", "pr-1")],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is True
    assert _families(report)["seed-family"]["status"] == "accepted"
    assert all("session_id" not in asset for asset in assets)


def test_decision_key_never_crosses_same_pair_reasons():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    candidate_edges = [
        _candidate("seed", "heldout", reason="related_source_reference"),
        _candidate("seed", "heldout", reason="shared_baseline_history"),
    ]

    reasoned = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=candidate_edges,
        claims=claims,
        decisions=[_decision("related_source_reference|heldout|seed", "distinct_problem")],
        exposures=[],
    )
    bare = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=candidate_edges,
        claims=claims,
        decisions=[
            {
                "left": "seed",
                "right": "heldout",
                "relation": "distinct_problem",
                "evidence_refs": ["review:bare"],
            }
        ],
        exposures=[],
    )

    assert reasoned["ready_for_seed_heldout"] is False
    assert {(edge["reason"], edge["blocking_reason"]) for edge in reasoned["unresolved_edges"]} == {
        ("shared_baseline_history", "missing_candidate_decision")
    }
    assert {edge["blocking_reason"] for edge in bare["unresolved_edges"]} == {
        "missing_candidate_decision"
    }
    assert len(bare["unresolved_edges"]) == 2


def test_metadata_only_review_is_recorded_but_not_body_exposure():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    exposure = _exposure(family_id="heldout-family", purpose="implementation")
    exposure["body_exposed"] = False

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[exposure],
    )

    heldout = _families(report)["heldout-family"]
    assert heldout["status"] == "accepted"
    assert heldout["heldout_exposed"] is False
    assert heldout["exposures"][0]["body_exposed"] is False


def test_exposure_without_evidence_blocks_even_without_body():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]
    exposure = _exposure(family_id="heldout-family", purpose="metadata_only_review", evidence_ref="")
    exposure["body_exposed"] = False

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[exposure],
    )

    heldout = _families(report)["heldout-family"]
    assert heldout["status"] == "blocked"
    assert "exposure_missing_evidence_ref" in heldout["reasons"]


def test_body_exposure_cannot_hide_under_metadata_label():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[_exposure(family_id="heldout-family", purpose="metadata_only_review")],
    )

    heldout = _families(report)["heldout-family"]
    assert heldout["status"] == "blocked"
    assert heldout["heldout_exposed"] is True
    assert "body_exposure_under_non_body_purpose" in heldout["reasons"]


def test_non_problem_association_isolates_metadata_only_container():
    assets = [
        _asset("seed", session="session:one"),
        _asset("heldout", session="session:h"),
        _container("repo-snapshot"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[
            _candidate("seed", "repo-snapshot", edge_id="repo-edge", reason="repository_contains_commit")
        ],
        claims=claims,
        decisions=[_association("repo-edge", "repo-snapshot")],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is True
    assert report["unresolved_edges"] == []
    assert report["isolated_container_asset_ids"] == ["repo-snapshot"]
    assert report["unselected_asset_ids"] == ["repo-snapshot"]
    assert "repo-snapshot" not in _families(report)["seed-family"]["covered_asset_ids"]


@pytest.mark.parametrize(
    ("scope", "admission", "expected_reason"),
    [
        ("problem_scoped_file", "denied", "container_not_metadata_only"),
        ("metadata_only_container", "allowed", "container_body_admission_not_denied"),
    ],
)
def test_non_problem_association_requires_metadata_only_denied_container(
    scope: str, admission: str, expected_reason: str
):
    container = _container("repo-snapshot")
    container["admission_scope"] = scope
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h"), container]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[
            _candidate("seed", "repo-snapshot", edge_id="repo-edge", reason="repository_contains_commit")
        ],
        claims=claims,
        decisions=[_association("repo-edge", "repo-snapshot", admission=admission)],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["unresolved_edges"][0]["blocking_reason"] == expected_reason
    assert report["isolated_container_asset_ids"] == []


def test_non_problem_association_cannot_release_two_selected_problem_families():
    assets = [
        _asset("seed", session="session:one"),
        _asset("heldout", session="session:h"),
        _container("repo-snapshot"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[
            _candidate("seed", "heldout", edge_id="repo-edge", reason="shared_baseline_history")
        ],
        claims=claims,
        decisions=[_association("repo-edge", "repo-snapshot")],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["unresolved_edges"][0]["blocking_reason"] == "non_problem_association_selected_pair"
    assert report["isolated_container_asset_ids"] == []


def test_non_problem_association_requires_coarse_reason():
    assets = [
        _asset("seed", session="session:one"),
        _asset("heldout", session="session:h"),
        _container("repo-snapshot"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[
            _candidate("seed", "repo-snapshot", edge_id="repo-edge", reason="related_source_reference")
        ],
        claims=claims,
        decisions=[_association("repo-edge", "repo-snapshot")],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["unresolved_edges"][0]["blocking_reason"] == (
        "non_problem_association_reason_not_allowed"
    )


def test_container_split_boundary_is_rejected():
    assets = [
        _asset("seed", session="session:one"),
        _asset("heldout", session="session:h"),
        _container("repo-snapshot", split="heldout"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[
            _candidate("seed", "repo-snapshot", edge_id="repo-edge", reason="repository_contains_commit")
        ],
        claims=claims,
        decisions=[_association("repo-edge", "repo-snapshot")],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["status"] == "cross_split_conflict"
    assert report["unresolved_edges"][0]["blocking_reason"] == "container_cross_split_conflict"
    assert report["isolated_container_asset_ids"] == []


def test_session_content_cannot_span_seed_and_heldout():
    assets = [
        _asset("seed", session="session:shared"),
        _asset("heldout", session="session:shared"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["status"] == "cross_split_conflict"
    assert any(item["kind"] == "session_cross_split" for item in report["cross_split_conflicts"])
    assert {family["status"] for family in report["families"]} == {"failed"}


@pytest.mark.parametrize(
    ("container_id", "expected_reason"),
    [
        ("", "missing_container_asset"),
        ("missing-snapshot", "unknown_container_asset"),
        ("heldout", "container_asset_selected"),
        ("unrelated-snapshot", "container_not_incident_endpoint"),
    ],
)
def test_non_problem_association_requires_known_unselected_container(
    container_id: str, expected_reason: str
):
    assets = [
        _asset("seed", session="session:one"),
        _asset("heldout", session="session:h"),
        _container("repo-snapshot"),
        _container("unrelated-snapshot"),
    ]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_TWO),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[
            _candidate("seed", "repo-snapshot", edge_id="repo-edge", reason="repository_contains_commit")
        ],
        claims=claims,
        decisions=[_association("repo-edge", container_id)],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["unresolved_edges"][0]["blocking_reason"] == expected_reason
    assert report["isolated_container_asset_ids"] == []


def test_duplicate_problem_identity_cannot_pad_two_families():
    assets = [_asset("seed", session="session:one"), _asset("heldout", session="session:h")]
    claims = [
        _claim("seed-family", "seed", ["seed"], PR_ONE),
        _claim("heldout-family", "heldout", ["heldout"], PR_ONE),
    ]

    report = audit_problem_closure(
        assets=assets,
        strong_edges=[],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
    )

    assert report["ready_for_seed_heldout"] is False
    assert report["accepted_claims"] == []
    assert all("duplicate_problem_identity" in family["reasons"] for family in report["families"])
