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


def bind_scope(kwargs, scope, index=0):
    material = kwargs["materials"][index]
    raw = encoded(scope)
    kwargs["evidence_bytes"][material.scope_evidence_ref] = raw
    material.scope_evidence_sha256 = digest(raw)


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
            source_ranges=[dict(source_asset_ids=[ident], line_start=1, line_end=5)],
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


def test_endpoint_shortcuts_and_bridges_cannot_authorize_or_exempt_body_exposure():
    variants = []
    empty = inputs()
    scope = json.loads(empty["evidence_bytes"]["scope:file-a"])
    scope["endpoint_paths"][0]["strong_edge_ids"] = []
    bind_scope(empty, scope)
    variants.append(empty)

    endpoint = inputs()
    endpoint["materials"][0].source_asset_ids = ["commit-a"]
    scope = json.loads(endpoint["evidence_bytes"]["scope:file-a"])
    scope["source_asset_ids"] = ["commit-a"]
    scope["endpoint_paths"] = [
        dict(source_asset_id="commit-a", endpoint_asset_id="commit-a", strong_edge_ids=[])
    ]
    scope["source_ranges"][0]["source_asset_ids"] = ["commit-a"]
    bind_scope(endpoint, scope)
    variants.append(endpoint)

    bridge = inputs()
    bridge["strong_edges"].append(
        dict(edge_id="bridge", left="commit-a", right="pr-b", reason="exact_pr_commit")
    )
    scope = json.loads(bridge["evidence_bytes"]["scope:file-a"])
    scope["endpoint_paths"].append(
        dict(
            source_asset_id="file-a",
            endpoint_asset_id="pr-b",
            strong_edge_ids=["file-a-commit", "bridge"],
        )
    )
    bind_scope(bridge, scope)
    variants.append(bridge)
    for kwargs in variants:
        denied = admit_lineage_review(**kwargs)
        assert denied["status"] == "denied" and not denied["exposure_records"]

    kwargs = inputs()
    admission = admit_lineage_review(**kwargs)
    admission["materials"][0]["endpoint_paths"][0]["strong_edge_ids"] = []
    exposures, receipts = persisted(admission)
    assert not closure(kwargs, exposures, lineage_review_receipt_bytes=receipts)[
        "ready_for_seed_heldout"
    ]
    current = inputs()
    exposures, receipts = persisted(admit_lineage_review(**current))
    current["strong_edges"] = [e for e in current["strong_edges"] if e["edge_id"] != "file-b-commit"]
    assert not closure(current, exposures, lineage_review_receipt_bytes=receipts)[
        "ready_for_seed_heldout"
    ]


def test_bundle_ranges_allow_baseline_fix_aliases_but_never_relabel_one_side_as_both():
    kwargs = inputs()
    kwargs["assets"].append(
        dict(asset_id="file-a-before", source_id="source:before", source_locator="fixture://before")
    )
    for asset in kwargs["assets"]:
        if asset["asset_id"] in {"file-a", "file-a-before", "file-b"}:
            asset["path"] = "example.py"
    kwargs["strong_edges"].append(
        dict(
            edge_id="before-commit",
            left="file-a-before",
            right="commit-a",
            reason="exact_firstparent_file",
        )
    )
    material = kwargs["materials"][0]
    raw = kwargs["readable_bytes"]["file-a"] + kwargs["readable_bytes"]["file-b"]
    kwargs["readable_bytes"][material.material_id] = raw
    material.source_asset_ids = ["file-a", "file-a-before", "file-b"]
    material.scope, material.content_sha256 = "problem_scoped_bundle", digest(raw)
    scope = json.loads(kwargs["evidence_bytes"]["scope:file-a"])
    scope.update(
        source_asset_ids=material.source_asset_ids,
        scope=material.scope,
        material_sha256=material.content_sha256,
        endpoint_paths=[
            *scope["endpoint_paths"],
            dict(
                source_asset_id="file-a-before",
                endpoint_asset_id="commit-a",
                strong_edge_ids=["before-commit"],
            ),
            dict(
                source_asset_id="file-b",
                endpoint_asset_id="pr-b",
                strong_edge_ids=["file-b-commit", "commit-b-pr"],
            ),
        ],
        source_ranges=[
            dict(
                source_asset_ids=["file-a", "file-a-before"],
                line_start=1,
                line_end=5,
                file_path="example.py",
            ),
            dict(source_asset_ids=["file-b"], line_start=6, line_end=10, file_path="example.py"),
        ],
    )
    bind_scope(kwargs, scope)
    proof = json.loads(kwargs["evidence_bytes"]["privacy:file-a"])
    proof["material_sha256"] = material.content_sha256
    kwargs["evidence_bytes"]["privacy:file-a"] = encoded(proof)
    material.privacy_proof_sha256 = digest(encoded(proof))
    kwargs["materials"] = [material]
    admission = admit_lineage_review(**kwargs)
    assert admission["status"] == "admitted"
    valid = dict(
        edge_id="question",
        relation="distinct_problem",
        evidence=[
            dict(
                material_id=material.material_id,
                source_asset_id=source,
                content_sha256=material.content_sha256,
                line_start=start,
                line_end=end,
            )
            for source, start, end in [("file-a-before", 4, 5), ("file-b", 9, 10)]
        ],
    )
    assert accept_lineage_decisions(admission, [valid])
    wrong_side = deepcopy(valid)
    wrong_side["evidence"][1].update(line_start=4, line_end=5)
    with pytest.raises(ValueError, match="citation_not_bound"):
        accept_lineage_decisions(admission, [wrong_side])

    variants = []
    overlap = deepcopy(scope)
    overlap["source_ranges"][1]["line_start"] = 4
    variants.append(overlap)
    wrong_file = deepcopy(scope)
    wrong_file["source_ranges"][1]["file_path"] = "other.py"
    variants.append(wrong_file)
    uncovered = deepcopy(scope)
    uncovered["source_ranges"] = uncovered["source_ranges"][:1]
    variants.append(uncovered)
    for changed in variants:
        bind_scope(kwargs, changed)
        assert admit_lineage_review(**kwargs)["status"] == "denied"

    separate = inputs()
    duplicated = separate["materials"][1]
    duplicated.content_sha256 = separate["materials"][0].content_sha256
    separate["readable_bytes"]["file-b"] = separate["readable_bytes"]["file-a"]
    for ref, field in [
        (duplicated.scope_evidence_ref, "scope_evidence_sha256"),
        (duplicated.privacy_proof_ref, "privacy_proof_sha256"),
    ]:
        value = json.loads(separate["evidence_bytes"][ref])
        value["material_sha256"] = duplicated.content_sha256
        separate["evidence_bytes"][ref] = encoded(value)
        setattr(duplicated, field, digest(encoded(value)))
    assert admit_lineage_review(**separate)["status"] == "denied"


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
    historical = admit_lineage_review(**kwargs, prior_exposures=prior)
    assert historical["prior_exposures"] == prior
    history_exposures, history_receipts = persisted(historical)
    result = closure(kwargs, history_exposures, lineage_review_receipt_bytes=history_receipts)
    family = next(f for f in result["families"] if f["family_id"] == "heldout-family")
    assert "implementation_exposure_on_heldout" in family["reasons"]
    assert not result["ready_for_seed_heldout"]


def test_nested_review_history_requires_its_own_valid_receipt_and_preserves_prior_pollution():
    kwargs = inputs()
    first = admit_lineage_review(**kwargs)
    first_raw = encoded(first)
    first_exposures = [
        dict(e, admission_ref="admission:first", admission_sha256=digest(first_raw))
        for e in first["exposure_records"]
    ]
    kwargs["context_id"] = "second-isolated-context"
    second = admit_lineage_review(**kwargs, prior_exposures=first_exposures)
    exposures, receipts = persisted(second)
    receipts["admission:first"] = first_raw
    assert closure(kwargs, exposures, lineage_review_receipt_bytes=receipts)["ready_for_seed_heldout"]
    assert not closure(
        kwargs,
        exposures,
        lineage_review_receipt_bytes={"admission:review": receipts["admission:review"]},
    )["ready_for_seed_heldout"]

    first["prior_exposures"] = [
        dict(
            context_id="earlier-implementation",
            purpose="implementation",
            body_exposed=True,
            asset_ids=["file-b"],
            evidence_refs=["actual-prior-read"],
        )
    ]
    first_raw = encoded(first)
    for exposure in first_exposures:
        exposure["admission_sha256"] = digest(first_raw)
    second = admit_lineage_review(**kwargs, prior_exposures=first_exposures)
    exposures, receipts = persisted(second)
    receipts["admission:first"] = first_raw
    result = closure(kwargs, exposures, lineage_review_receipt_bytes=receipts)
    heldout = next(f for f in result["families"] if f["family_id"] == "heldout-family")
    assert heldout["reasons"] == ["implementation_exposure_on_heldout"]
    assert not result["ready_for_seed_heldout"]


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
