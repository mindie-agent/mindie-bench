from datetime import datetime, timezone

from mindie_bench.evaluation.split_manifest import SourceAsset, build_manifest
from mindie_bench.mining.material_dispatch import (
    MaterialPrivacyEvidence,
    ReadableMaterial,
    dispatch_materials,
)


def inputs():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assets = [
        SourceAsset(
            asset_id=name,
            task_family=name,
            source_id=f"problem:{name}",
            source_locator=f"source://{name}",
            source_kind="commit",
            timestamp=timestamp,
            lineage_resolved=True,
        )
        for name in ("sampler", "rope")
    ]
    manifest = build_manifest(assets, {"sampler": "seed", "rope": "heldout"})
    material = ReadableMaterial(
        material_id="readable-rope",
        source_asset_ids=["rope"],
        locator="/material/redacted-rope.jsonl",
        content_sha256="b" * 64,
        covered_task_families=["rope"],
        scope_evidence_ref="scope/rope.json",
        scope="redacted_transcript",
        origin="private_transcript",
        privacy_proof_ref="scan/final.json",
        privacy_proof_sha256="c" * 64,
    )
    proof = MaterialPrivacyEvidence(
        proof_ref=material.privacy_proof_ref,
        proof_sha256=material.privacy_proof_sha256,
        material_locator=material.locator,
        material_sha256=material.content_sha256,
        status="passed",
        completed=True,
        finding_count=0,
        channel_status="public_conversation_only",
        channel_evidence_ref="capture/public-channel-selection.json",
    )
    return manifest, material, proof


def evaluate(manifest, material, proof, **extra):
    return dispatch_materials(
        manifest,
        [material],
        purpose="heldout_task",
        context_id="heldout-builder",
        privacy_evidence=[proof],
        observed_sha256={material.material_id: material.content_sha256},
        **extra,
    )


def test_exact_readable_bytes_are_admitted_without_substituting_original_source():
    manifest, material, proof = inputs()
    result = evaluate(manifest, material, proof)
    assert result["status"] == "admitted"
    assert result["materials"][0]["locator"] == material.locator
    assert result["exposure_records"][0]["material_sha256"] == "b" * 64
    assert result["body_read_performed"] is False
    proof.material_sha256 = "a" * 64
    rejected = evaluate(manifest, material, proof)
    assert rejected["status"] == "denied"
    assert "privacy_evidence_bound_to_other_bytes" in rejected["materials"][0]["reasons"]


def test_clean_scan_does_not_admit_unverified_channels_or_incomplete_files():
    manifest, material, proof = inputs()
    proof.channel_status = "unverified"
    assert evaluate(manifest, material, proof)["status"] == "denied"
    proof.channel_status = "public_conversation_only"
    proof.completed = False
    assert evaluate(manifest, material, proof)["status"] == "denied"


def test_material_spanning_splits_and_reused_seed_context_are_rejected():
    manifest, material, proof = inputs()
    material.source_asset_ids = ["rope", "sampler"]
    material.covered_task_families = ["rope", "sampler"]
    assert evaluate(manifest, material, proof)["status"] == "denied"
    material.source_asset_ids = ["rope"]
    material.covered_task_families = ["rope"]
    result = evaluate(
        manifest,
        material,
        proof,
        prior_exposures=[{"context_id": "heldout-builder", "split": "seed"}],
    )
    assert result["status"] == "denied"
    assert "consumer_context_already_exposed_to_another_split" in result["reasons"]


def test_material_hash_refresh_and_implementation_exposure_are_required():
    manifest, material, proof = inputs()
    result = dispatch_materials(
        manifest,
        [material],
        purpose="heldout_task",
        context_id="fresh-context",
        privacy_evidence=[proof],
        observed_sha256={material.material_id: "d" * 64},
        prior_exposures=[
            {"context_id": "implementation", "purpose": "implementation", "task_families": ["rope"]}
        ],
    )
    assert result["status"] == "denied"
    assert set(result["materials"][0]["reasons"]) >= {
        "readable_material_current_hash_mismatch_or_missing",
        "heldout_problem_previously_exposed",
    }
    assert result["exposure_records"] == []


def test_dev_body_exposure_blocks_heldout_but_metadata_review_does_not():
    manifest, material, proof = inputs()
    exposure = {
        "context_id": "other",
        "purpose": "dev_evaluation",
        "task_families": ["rope"],
        "body_exposed": True,
    }
    assert evaluate(manifest, material, proof, prior_exposures=[exposure])["status"] == "denied"
    exposure["body_exposed"] = False
    assert evaluate(manifest, material, proof, prior_exposures=[exposure])["status"] == "admitted"
