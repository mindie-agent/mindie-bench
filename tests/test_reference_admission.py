from mindie_bench.mining.material_dispatch import MaterialPrivacyEvidence
from mindie_bench.mining.reference_admission import (
    ProtectedReference,
    PublicSourceEvidence,
    ReferenceMaterial,
    ReferencePrivacyEvidence,
    admit_public_references,
)


def evidence():
    material = ReferenceMaterial(
        material_id="ascend:commit:layernorm",
        source_id="vllm-ascend",
        repo_url="https://github.com/vllm-project/vllm-ascend",
        revision="a" * 40,
        path="vllm_ascend/ops/layernorm.py",
        git_blob_oid="b" * 40,
        source_sha256="c" * 64,
        byte_start=0,
        byte_end=240,
        locator="/reference/layernorm-0-240.txt",
        content_sha256="d" * 64,
        privacy_proof_ref="/privacy/layernorm.json",
        privacy_proof_sha256="e" * 64,
    )
    source = PublicSourceEvidence(
        source_id=material.source_id,
        repo_url=material.repo_url,
        revision=material.revision,
        public_provenance_ref="/inventory/github-source.json",
        public_provenance_sha256="f" * 64,
        verified=True,
    )
    scan = MaterialPrivacyEvidence(
        proof_ref=material.privacy_proof_ref,
        proof_sha256=material.privacy_proof_sha256,
        material_locator=material.locator,
        material_sha256=material.content_sha256,
        status="passed",
        completed=True,
        finding_count=0,
        channel_status="public_source",
        channel_evidence_ref=source.public_provenance_ref,
    )
    kwargs = dict(
        context_id="fresh-public-reading-context",
        source_evidence=[source],
        privacy_evidence=[scan],
        observed_sha256={material.material_id: material.content_sha256},
        observed_git_blobs={
            material.material_id: {
                key: getattr(material, key)
                for key in ("revision", "path", "git_blob_oid", "source_sha256")
            }
        },
        protected_references=[],
    )
    return material, kwargs


def test_public_reference_admission_is_not_problem_split_or_read_completion():
    material, kwargs = evidence()
    kwargs["observed_git_blobs"][material.material_id]["git_read_completed"] = True
    result = admit_public_references([material], **kwargs)
    assert result["status"] == "admitted"
    assert result["benchmark_problem_lineage_claimed"] is False
    assert result["independent_heldout_claimed"] is False
    assert result["body_read_performed"] is False
    assert result["exposure_intents"][0]["body_exposure_state"] == "authorized_not_yet_dispatched"


def test_changed_git_object_or_readable_bytes_cannot_reuse_clean_scan():
    material, kwargs = evidence()
    kwargs["observed_git_blobs"][material.material_id]["git_blob_oid"] = "1" * 40
    kwargs["observed_sha256"][material.material_id] = "2" * 64
    result = admit_public_references([material], **kwargs)
    assert result["status"] == "denied"
    assert result["materials"][0]["reasons"] == [
        "pinned_git_blob_identity_mismatch_or_missing",
        "readable_material_current_hash_mismatch_or_missing",
    ]
    assert result["exposure_intents"] == []


def test_partial_scan_blocks_otherwise_public_pinned_source():
    material, kwargs = evidence()
    kwargs["privacy_evidence"][0].completed = False
    result = admit_public_references([material], **kwargs)
    assert result["status"] == "denied"
    assert "privacy_evidence_not_complete_clean_or_reviewed_and_bound" in result["materials"][0]["reasons"]


def test_protected_heldout_path_blocks_other_revision_and_duplicate_content_elsewhere():
    material, kwargs = evidence()
    kwargs["protected_references"] = [
        ProtectedReference(
            repo_url=material.repo_url + ".git",
            path=material.path,
            family_id="heldout-problem",
            evidence_ref="/split/frozen-manifest.json",
            source_sha256=material.source_sha256,
        )
    ]
    different_revision = material.model_copy(update={"revision": "3" * 40})
    kwargs["source_evidence"][0].revision = different_revision.revision
    kwargs["observed_git_blobs"][material.material_id]["revision"] = different_revision.revision
    result = admit_public_references([different_revision], **kwargs)
    assert result["materials"][0]["protected_family_ids"] == ["heldout-problem"]
    assert result["status"] == "denied"
    copied = material.model_copy(
        update={"repo_url": "https://example.invalid/copied", "path": "copy.py"}
    )
    result = admit_public_references([copied], **kwargs)
    assert "protected_heldout_source_material" in result["materials"][0]["reasons"]
    assert result["exposure_intents"] == []


def test_exact_public_false_positive_review_preserves_detected_count_and_source_binding():
    material, kwargs = evidence()
    proof = kwargs["privacy_evidence"][0]
    reviewed = ReferencePrivacyEvidence(
        **{**proof.model_dump(), "status": "passed_with_reviewed_exceptions", "finding_count": 2},
        reviewed_exception_count=2,
        exception_review_ref="/privacy/source-false-positive-review.json",
        exception_review_sha256="1" * 64,
        exception_source_sha256=material.source_sha256,
    )
    kwargs["privacy_evidence"] = [reviewed]
    assert admit_public_references([material], **kwargs)["status"] == "admitted"
    assert reviewed.finding_count == 2
    reviewed.reviewed_exception_count = 1
    assert admit_public_references([material], **kwargs)["status"] == "denied"
    reviewed.reviewed_exception_count = 2
    reviewed.exception_source_sha256 = "2" * 64
    assert admit_public_references([material], **kwargs)["status"] == "denied"


def test_copied_slice_hash_does_not_use_whole_source_hash():
    material, kwargs = evidence()
    kwargs["protected_references"] = [ProtectedReference(
        content_sha256=material.content_sha256, family_id="heldout-copy",
        evidence_ref="/split/heldout-materials.json",
    )]
    result = admit_public_references([material], **kwargs)
    assert result["status"] == "denied"
    assert result["materials"][0]["protected_family_ids"] == ["heldout-copy"]
    kwargs["protected_references"][0].content_sha256 = material.source_sha256
    assert admit_public_references([material], **kwargs)["status"] == "admitted"


def test_marker_replay_requires_hash_bound_whole_source_including_outside_slice():
    import hashlib

    material, kwargs = evidence()
    data = b"heldout-change-id\nclean selected body\n"
    start = data.index(b"clean")
    material.source_sha256 = hashlib.sha256(data).hexdigest()
    material.content_sha256 = hashlib.sha256(data[start:]).hexdigest()
    material.byte_start, material.byte_end = start, len(data)
    kwargs["observed_sha256"][material.material_id] = material.content_sha256
    kwargs["observed_git_blobs"][material.material_id]["source_sha256"] = material.source_sha256
    kwargs["privacy_evidence"][0].material_sha256 = material.content_sha256
    kwargs["protected_references"] = [ProtectedReference(
        exact_content_markers=["heldout-change-id"], family_id="heldout-marker",
        evidence_ref="/split/heldout-metadata.json",
    )]
    missing = admit_public_references([material], **kwargs)
    assert "protected_marker_replay_missing" in missing["materials"][0]["reasons"]
    kwargs["observed_source_bytes"] = {material.material_id: data[start:]}
    unbound = admit_public_references([material], **kwargs)
    assert "protected_source_bytes_not_bound_to_material" in unbound["materials"][0]["reasons"]
    kwargs["observed_source_bytes"][material.material_id] = data
    denied = admit_public_references([material], **kwargs)
    assert denied["materials"][0]["protected_family_ids"] == ["heldout-marker"]
    assert denied["exposure_intents"] == []
