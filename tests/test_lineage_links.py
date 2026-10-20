"""Business contracts for metadata-only lineage grouping and leak containment."""

from mindie_bench.mining.lineage_links import build_lineage_groups

REPO = "gitcode.com/Ascend/agent-skills"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CONTENT_SHA256 = "a" * 64


def record(asset_id: str, **overrides) -> dict:
    row = {
        "asset_id": asset_id,
        "source_namespace": "collector",
        "source_collection": "fleet",
        "source_id": f"source-{asset_id}",
        "session_id": None,
        "sha256": None,
        "size": None,
        "source_locator": f"metadata/{asset_id}",
        "record_type": "session_artifact",
        "repo_identity": None,
        "revision": None,
        "branch": None,
        "declared_family": None,
        "related_source_ids": [],
        "related_source_namespace": None,
        "split": "unassigned",
        "public_catalog": False,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    row.update(overrides)
    return row


def group_for(result: dict, asset_id: str) -> dict:
    return next(group for group in result["groups"] if asset_id in group["asset_ids"])


def test_same_session_across_machines_and_segments_unions_for_split_isolation():
    records = [
        record(
            "collector-a1",
            session_id="session-42",
            source_id="segment-1",
            source_collection="fleet",
            sha256="1" * 64,
            size=101,
        ),
        record(
            "collector-b2",
            session_id="session-42",
            source_id="segment-2",
            source_collection="offline",
            sha256="2" * 64,
            size=202,
        ),
        record(
            "collector-c3",
            session_id="session-42",
            source_id="subagent-3",
            source_collection="fleet",
            sha256="3" * 64,
            size=303,
        ),
    ]

    result = build_lineage_groups(records)

    assert len(result["groups"]) == 1
    group = group_for(result, "collector-a1")
    assert group["asset_ids"] == ["collector-a1", "collector-b2", "collector-c3"]
    assert group["session_ids"] == ["session-42"]
    assert group["source_collections"] == ["fleet", "offline"]
    assert group["grouping_basis"] == ["exact_session_id"]
    assert group["lineage_resolved"] is False
    assert group["resolution_scope"] == "unresolved_problem_lineage"
    assert "session_coupling_not_problem_lineage" in group["unresolved_reasons"]

    session_edges = [edge for edge in result["strong_edges"] if edge["reason"] == "exact_session_id"]
    assert len(session_edges) == 2, "session sharing must use a star, not all pairs"


def test_same_complete_transcript_hash_deduplicates_copies_once():
    records = [
        record(
            "hash-a",
            source_namespace="map04",
            source_collection="offline",
            source_id="artifact-a",
            sha256=CONTENT_SHA256,
            size=97,
            hash_lineage_scope="complete_session_transcript",
        ),
        record(
            "hash-b",
            source_namespace="fetcher",
            source_collection="fleet",
            source_id="copy-b",
            sha256=CONTENT_SHA256,
            size=97,
            hash_lineage_scope="complete_session_transcript",
        ),
        record(
            "hash-c",
            source_namespace="collector",
            source_collection="fleet",
            source_id="copy-c",
            sha256=CONTENT_SHA256,
            size=97,
            hash_lineage_scope="complete_session_transcript",
        ),
    ]

    result = build_lineage_groups(records)

    assert len(result["groups"]) == 1
    group = group_for(result, "hash-a")
    assert group["asset_ids"] == ["hash-a", "hash-b", "hash-c"]
    assert group["grouping_basis"] == ["positive_sha256"]
    assert len(result["strong_edges"]) == 2
    assert {edge["reason"] for edge in result["strong_edges"]} == {"positive_sha256"}


def test_identical_step_outputs_do_not_bridge_two_distinct_session_families():
    result = build_lineage_groups(
        [
            record(
                "a-transcript",
                session_id="session-a",
                sha256="1" * 64,
                size=100,
                hash_lineage_scope="complete_session_transcript",
            ),
            record("a-output", session_id="session-a", sha256="3" * 64, size=20),
            record(
                "b-transcript",
                session_id="session-b",
                sha256="2" * 64,
                size=100,
                hash_lineage_scope="complete_session_transcript",
            ),
            record("b-output", session_id="session-b", sha256="3" * 64, size=20),
        ]
    )
    assert group_for(result, "a-transcript")["asset_ids"] == ["a-output", "a-transcript"]
    assert group_for(result, "b-transcript")["asset_ids"] == ["b-output", "b-transcript"]
    duplicate = next(
        edge
        for edge in result["candidate_edges"]
        if edge["reason"] == "duplicate_content_not_resolved_problem_lineage"
    )
    assert {duplicate["left"], duplicate["right"]} == {"a-output", "b-output"}


def test_empty_file_hash_and_hash_source_id_never_union_empty_files():
    records = [
        record(
            f"empty-{index}",
            source_namespace="collector",
            source_collection=collection,
            source_id=f"sha256:{EMPTY_SHA256}",
            sha256=EMPTY_SHA256,
            size=0,
            record_type="supporting_file",
        )
        for index, collection in enumerate(("fleet", "offline", "fleet", "offline"))
    ]

    result = build_lineage_groups(records)

    assert len(result["groups"]) == 4
    assert result["strong_edges"] == []
    assert all(len(group["asset_ids"]) == 1 for group in result["groups"])


def test_same_repo_and_declared_family_stays_a_topic_candidate_not_lineage():
    records = [
        record(
            "map03-a",
            source_namespace="map03",
            source_collection="offline",
            source_id="topic-a",
            record_type="supporting_file",
            repo_identity=REPO,
            declared_family="agent-skills",
        ),
        record(
            "map03-b",
            source_namespace="map03",
            source_collection="offline",
            source_id="topic-b",
            record_type="session_artifact",
            repo_identity=REPO,
            declared_family="agent-skills",
        ),
        record(
            "map03-c",
            source_namespace="map03",
            source_collection="offline",
            source_id="topic-c",
            record_type="supporting_file",
            repo_identity="gitcode.com/Ascend/other-repo",
            declared_family="agent-skills",
        ),
    ]

    result = build_lineage_groups(records)

    assert len(result["groups"]) == 3
    assert all(group["lineage_resolved"] is False for group in result["groups"])
    assert result["strong_edges"] == []

    candidate_pairs = {
        frozenset((edge["left"], edge["right"]))
        for edge in result["candidate_edges"]
        if edge["reason"] == "same_repo_declared_family_topic"
    }
    assert frozenset(("map03-a", "map03-b")) in candidate_pairs
    assert all("map03-c" not in pair for pair in candidate_pairs), (
        "broad family names must not bridge distinct repositories"
    )


def test_public_repo_family_unions_across_revisions_and_reports_drift():
    records = [
        record(
            "repo-r1",
            source_namespace="fetcher",
            source_collection="offline",
            source_id="src/alpha.py",
            record_type="source_file",
            repo_identity=REPO,
            revision="1" * 40,
            public_catalog=True,
        ),
        record(
            "repo-r2",
            source_namespace="fetcher",
            source_collection="offline",
            source_id="src/beta.py",
            record_type="source_repository",
            repo_identity=REPO,
            revision="2" * 40,
            public_catalog=True,
        ),
    ]

    result = build_lineage_groups(records)

    assert len(result["groups"]) == 1
    group = group_for(result, "repo-r1")
    assert group["asset_ids"] == ["repo-r1", "repo-r2"]
    assert group["grouping_basis"] == ["public_source_repository"]
    assert group["revisions"] == ["1" * 40, "2" * 40]
    assert group["lineage_resolved"] is True
    assert group["resolution_scope"] == "public_source_repository"

    drift = next(conflict for conflict in result["conflicts"] if conflict["kind"] == "revision_drift")
    assert drift["group_id"] == group["group_id"]
    assert drift["details"]["revisions"] == ["1" * 40, "2" * 40]


def test_assigned_split_conflict_blocks_resolution_while_single_split_resolves():
    records = [
        record(
            "conflict-a",
            source_id="src/conflict-a.py",
            record_type="source_file",
            repo_identity="gitcode.com/Ascend/conflict-repo",
            revision="7" * 40,
            split="seed",
            public_catalog=True,
        ),
        record(
            "conflict-b",
            source_id="src/conflict-b.py",
            record_type="source_file",
            repo_identity="gitcode.com/Ascend/conflict-repo",
            revision="7" * 40,
            split="heldout",
            public_catalog=True,
        ),
        record(
            "control-a",
            source_id="src/control-a.py",
            record_type="source_file",
            repo_identity="gitcode.com/Ascend/control-repo",
            revision="8" * 40,
            split="dev",
            public_catalog=True,
        ),
        record(
            "control-b",
            source_id="src/control-b.py",
            record_type="source_file",
            repo_identity="gitcode.com/Ascend/control-repo",
            revision="8" * 40,
            split="dev",
            public_catalog=True,
        ),
    ]

    result = build_lineage_groups(records)

    conflict_group = group_for(result, "conflict-a")
    assert conflict_group["asset_ids"] == ["conflict-a", "conflict-b"]
    assert conflict_group["inherited_splits"] == ["heldout", "seed"]
    assert conflict_group["lineage_resolved"] is False
    assert conflict_group["resolution_scope"] == "unresolved_problem_lineage"
    assert "cross_split_conflict" in conflict_group["unresolved_reasons"]

    conflict = next(
        conflict for conflict in result["conflicts"] if conflict["kind"] == "cross_split_conflict"
    )
    assert conflict["group_id"] == conflict_group["group_id"]
    assert conflict["details"]["splits"] == ["heldout", "seed"]

    control_group = group_for(result, "control-a")
    assert control_group["asset_ids"] == ["control-a", "control-b"]
    assert control_group["lineage_resolved"] is True
    assert control_group["resolution_scope"] == "public_source_repository"
    assert control_group["unresolved_reasons"] == []


def test_unknown_related_reference_keeps_repo_group_unresolved():
    records = [
        record(
            "ref-a",
            source_id="src/ref-a.py",
            record_type="source_file",
            repo_identity=REPO,
            revision="3" * 40,
            public_catalog=True,
            related_source_ids=["missing-map03-topic"],
            related_source_namespace="map03",
        ),
        record(
            "ref-b",
            source_id="src/ref-b.py",
            record_type="source_file",
            repo_identity=REPO,
            revision="3" * 40,
            public_catalog=True,
        ),
    ]

    result = build_lineage_groups(records)

    assert len(result["groups"]) == 1
    group = group_for(result, "ref-a")
    assert group["asset_ids"] == ["ref-a", "ref-b"]
    assert group["lineage_resolved"] is False
    assert group["resolution_scope"] == "unresolved_problem_lineage"
    assert "unknown_related_reference" in group["unresolved_reasons"]
    assert result["unresolved_references"] == [
        {
            "asset_id": "ref-a",
            "related_source_namespace": "map03",
            "related_source_id": "missing-map03-topic",
        }
    ]


def test_single_public_catalog_source_resolves_only_its_repository_family():
    result = build_lineage_groups(
        [
            record(
                "source-only",
                source_namespace="fetcher",
                source_collection="fetcher",
                record_type="source_repository",
                repo_identity=REPO,
                revision="a" * 40,
                public_catalog=True,
            )
        ]
    )
    group = result["groups"][0]
    assert group["lineage_resolved"] is True
    assert group["resolution_scope"] == "public_source_repository"
    assert group["inherited_splits"] == []
    assert group["session_ids"] == []
