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
