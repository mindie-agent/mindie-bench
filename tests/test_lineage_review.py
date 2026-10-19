"""Exact diff admission cannot become general body access or a split assignment."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from mindie_bench.evaluation.split_manifest import SourceAsset, build_manifest
from mindie_bench.mining.lineage_closure import audit_problem_closure
from mindie_bench.mining.lineage_review import (
    LineageReviewMaterial,
    accept_lineage_decisions,
    admit_lineage_review,
)
from mindie_bench.mining.material_dispatch import (
    MaterialPrivacyEvidence,
    ReadableMaterial,
    dispatch_materials,
)


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def inputs():
    assets = [
        dict(asset_id=i, source_id="source:" + i, source_locator="fixture://" + i)
        for i in ["file-a", "file-b", "commit-a", "commit-b", "pr-b", "repository"]
    ]
    assets[-1]["admission_scope"] = "metadata_only_container"
    strong = [
        dict(edge_id="file-a-commit", left="file-a", right="commit-a", reason="exact_firstparent_file"),
        dict(edge_id="file-b-commit", left="file-b", right="commit-b", reason="exact_firstparent_file"),
        dict(edge_id="commit-b-pr", left="commit-b", right="pr-b", reason="exact_pr_commit"),
    ]
    candidate = [
        dict(edge_id="question", left="commit-a", right="pr-b", reason="related_source_reference")
    ]
    materials, readable, evidence = [], {}, {}
    for suffix, endpoint, path in [
        ("a", "commit-a", ["file-a-commit"]),
        ("b", "pr-b", ["file-b-commit", "commit-b-pr"]),
    ]:
        ident = "file-" + suffix
        raw = (
            "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old_" + suffix + "\n+new_" + suffix + "\n"
        ).encode()
        readable[ident] = raw
        scope_ref, privacy_ref = "scope:" + ident, "privacy:" + ident
        scope = dict(
            schema_version="mindie-lineage-review-scope/1",
            material_id=ident,
            source_asset_ids=[ident],
            material_sha256=digest(raw),
            scope="problem_scoped_file",
            origin="private_source",
            endpoint_paths=[
                dict(source_asset_id=ident, endpoint_asset_id=endpoint, strong_edge_ids=path)
            ],
        )
        privacy = dict(
            material_locator="readable:" + ident,
            material_sha256=digest(raw),
            status="passed",
            completed=True,
            finding_count=0,
            channel_status="source_code",
            channel_evidence_ref="source-review:" + ident,
        )
        evidence[scope_ref], evidence[privacy_ref] = encoded(scope), encoded(privacy)
        materials.append(
            LineageReviewMaterial(
                material_id=ident,
                source_asset_ids=[ident],
                locator="readable:" + ident,
                content_sha256=digest(raw),
                scope="problem_scoped_file",
                origin="private_source",
                scope_evidence_ref=scope_ref,
                scope_evidence_sha256=digest(evidence[scope_ref]),
                privacy_proof_ref=privacy_ref,
                privacy_proof_sha256=digest(evidence[privacy_ref]),
            )
        )
    return dict(
        assets=assets,
        strong_edges=strong,
        candidate_edges=candidate,
        review_edge_ids=["question"],
        materials=materials,
        context_id="fresh-lineage-review",
        readable_bytes=readable,
        evidence_bytes=evidence,
    )


def decision(admission):
    return dict(
        edge_id="question",
        relation="distinct_problem",
        evidence=[
            dict(
                material_id=m["material_id"],
                source_asset_id=m["source_asset_ids"][0],
                content_sha256=m["content_sha256"],
                line_start=4,
                line_end=5,
            )
            for m in admission["materials"]
        ],
    )


def persisted(admission):
    raw = encoded(admission)
    exposures = [
        dict(e, admission_ref="admission:review", admission_sha256=digest(raw))
        for e in admission["exposure_records"]
    ]
    return exposures, {"admission:review": raw}


def closure(kwargs, exposures, **extra):
    return audit_problem_closure(
        assets=kwargs["assets"],
        strong_edges=kwargs["strong_edges"],
        candidate_edges=kwargs["candidate_edges"],
        claims=[
            dict(
                family_id=family,
                split=split,
                asset_ids=ids,
                problem_identity=problem,
                problem_identity_scope="problem",
                evidence_refs=["claim:" + family],
            )
            for family, split, ids, problem in [
                ("seed-family", "seed", ["file-a", "commit-a"], "https://github.com/example/r/pull/1"),
                (
                    "heldout-family",
                    "heldout",
                    ["file-b", "commit-b", "pr-b"],
                    "https://github.com/example/r/pull/2",
                ),
            ]
        ],
        decisions=[
            dict(edge_id="question", relation="distinct_problem", evidence_refs=["review:question"])
        ],
        exposures=exposures,
        **extra,
    )


def test_explicit_file_paths_cover_pr_endpoints_without_authorizing_repository_bodies():
    kwargs = inputs()
    kwargs["strong_edges"].append(
        dict(
            edge_id="whole-repository",
            left="commit-a",
            right="repository",
            reason="public_source_repository",
        )
    )
    admission = admit_lineage_review(**kwargs)
    assert admission["status"] == "admitted"
    assert {e["edge_id"] for e in admission["incident_edges"]} == {
        "file-a-commit",
        "file-b-commit",
        "commit-b-pr",
        "whole-repository",
        "question",
    }
    assert "repository" not in admission["selected_asset_ids"]
    assert {e["source_asset_ids"][0] for e in admission["exposure_records"]} == {"file-a", "file-b"}
    assert not admission["family_or_split_acceptance"] and not admission["body_read_performed"]
    accepted = accept_lineage_decisions(admission, [decision(admission)])
    assert accepted[0]["relation"] == "distinct_problem"
    scope = json.loads(kwargs["evidence_bytes"]["scope:file-a"])
    scope["endpoint_paths"][0]["strong_edge_ids"] = [
        "file-a-commit",
        "whole-repository",
        "whole-repository",
    ]
    kwargs["evidence_bytes"]["scope:file-a"] = encoded(scope)
    kwargs["materials"][0].scope_evidence_sha256 = digest(encoded(scope))
    assert admit_lineage_review(**kwargs)["status"] == "denied"


def test_changed_material_scope_privacy_or_container_never_produces_read_permission():
    original = inputs()
    variants = []
    changed = deepcopy(original)
    changed["readable_bytes"]["file-a"] += b"changed\n"
    variants.append(changed)
    changed = deepcopy(original)
    changed["evidence_bytes"]["scope:file-a"] += b" "
    variants.append(changed)
    changed = deepcopy(original)
    changed["assets"][0]["admission_scope"] = "metadata_only_container"
    variants.append(changed)
    changed = deepcopy(original)
    changed["materials"][0].source_asset_ids = ["unknown-source"]
    variants.append(changed)
    changed = deepcopy(original)
    proof = json.loads(changed["evidence_bytes"]["privacy:file-a"])
    proof["completed"] = False
    changed["evidence_bytes"]["privacy:file-a"] = encoded(proof)
    changed["materials"][0].privacy_proof_sha256 = digest(encoded(proof))
    variants.append(changed)
    changed = deepcopy(original)
    changed["materials"][0].source_asset_ids.append("commit-b")
    scope = json.loads(changed["evidence_bytes"]["scope:file-a"])
    scope["source_asset_ids"].append("commit-b")
    changed["evidence_bytes"]["scope:file-a"] = encoded(scope)
    changed["materials"][0].scope_evidence_sha256 = digest(encoded(scope))
    variants.append(changed)
    for variant in variants:
        denied = admit_lineage_review(**variant)
        assert denied["status"] == "denied" and denied["exposure_records"] == []


def test_relation_requires_bound_ranges_on_both_question_endpoints_and_no_rationale():
    admission = admit_lineage_review(**inputs())
    valid = decision(admission)
    bad = [
        dict(valid, evidence=valid["evidence"][:1]),
        dict(valid, edge_id="unreviewed"),
        dict(valid, rationale="private"),
    ]
    changed = deepcopy(valid)
    changed["evidence"][0]["content_sha256"] = "0" * 64
    bad.append(changed)
    changed = deepcopy(valid)
    changed["evidence"][0]["line_end"] = 6
    bad.append(changed)
    for value in bad:
        with pytest.raises(ValueError):
            accept_lineage_decisions(admission, [value])
    assert accept_lineage_decisions(
        admission, [dict(edge_id="question", relation="unresolved", evidence=[])]
    )


def test_only_exact_review_receipt_exempts_its_exposure_and_keeps_implementation_exposure():
    kwargs = inputs()
    admission = admit_lineage_review(**kwargs)
    exposures, receipts = persisted(admission)
    assert not closure(kwargs, exposures)["ready_for_seed_heldout"]
    assert closure(kwargs, exposures, lineage_review_receipt_bytes=receipts)["ready_for_seed_heldout"]
    changed = dict(receipts)
    changed["admission:review"] += b" "
    assert not closure(kwargs, exposures, lineage_review_receipt_bytes=changed)["ready_for_seed_heldout"]
    modified = deepcopy(exposures)
    modified[1]["material_sha256"] = "0" * 64
    assert not closure(kwargs, modified, lineage_review_receipt_bytes=receipts)["ready_for_seed_heldout"]
    modified = deepcopy(exposures)
    modified[1]["asset_ids"] = ["commit-b"]
    assert not closure(kwargs, modified, lineage_review_receipt_bytes=receipts)["ready_for_seed_heldout"]
    nonquarantined = deepcopy(admission)
    nonquarantined["permanent_quarantine"] = False
    nonquarantined["exposure_records"][1]["permanent_quarantine"] = False
    invalid_exposures, invalid_receipts = persisted(nonquarantined)
    assert not closure(kwargs, invalid_exposures, lineage_review_receipt_bytes=invalid_receipts)[
        "ready_for_seed_heldout"
    ]
    implementation = dict(
        context_id="implementation",
        purpose="implementation",
        body_exposed=True,
        asset_ids=["file-b"],
        evidence_refs=["prior-implementation-read"],
    )
    result = closure(kwargs, [*exposures, implementation], lineage_review_receipt_bytes=receipts)
    family = next(f for f in result["families"] if f["family_id"] == "heldout-family")
    assert "implementation_exposure_on_heldout" in family["reasons"]
    prior = [implementation]
    assert admit_lineage_review(**kwargs, prior_exposures=prior)["prior_exposures"] == prior


def test_quarantine_context_cannot_be_reused_for_lineage_or_existing_seed_dispatch():
    kwargs = inputs()
    admission = admit_lineage_review(**kwargs)
    exposures, _ = persisted(admission)
    assert admit_lineage_review(**kwargs, prior_exposures=exposures)["status"] == "denied"
    assert (
        admit_lineage_review(
            **kwargs, prior_exposures=[dict(context_id=kwargs["context_id"], body_exposed=False)]
        )["status"]
        == "denied"
    )
    material = kwargs["materials"][0]
    asset = SourceAsset(
        asset_id="file-a",
        task_family="one",
        source_id="source:a",
        source_locator="fixture://a",
        source_kind="commit",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lineage_resolved=True,
    )
    manifest = build_manifest([asset], {"one": "seed"})
    readable = ReadableMaterial(
        **{k: v for k, v in material.model_dump().items() if k != "scope_evidence_sha256"},
        covered_task_families=["one"],
    )
    proof = MaterialPrivacyEvidence(
        **json.loads(kwargs["evidence_bytes"][material.privacy_proof_ref]),
        proof_ref=material.privacy_proof_ref,
        proof_sha256=material.privacy_proof_sha256,
    )
    result = dispatch_materials(
        manifest,
        [readable],
        purpose="seed_distillation",
        context_id=kwargs["context_id"],
        privacy_evidence=[proof],
        observed_sha256={material.material_id: material.content_sha256},
        prior_exposures=exposures,
    )
    assert result["status"] == "denied"
    assert "consumer_context_already_exposed_to_another_split" in result["reasons"]
