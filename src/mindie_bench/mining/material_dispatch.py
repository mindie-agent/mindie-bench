"""Bind readable problem materials to split, privacy and consumer evidence.

[Role] Audit metadata immediately before a separate context reads a material.
[Invariants] Original source identity never substitutes for the readable bytes;
all covered families must belong to the consumer's split. Scanner receipts bind
the exact readable hash, and Seed and Heldout never share a model context.
[Timeouts & Locks] Pure in-memory validation; the caller hashes local files.
[Side-Effect Policy] No body reads or dispatch. Return an auditable allow/deny
decision and exposure records for the caller to persist before an allowed read.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mindie_bench.evaluation.split_manifest import Purpose, SplitManifest, admit_materials
from mindie_bench.models import StrictModel

SHA = r"^[0-9a-f]{64}$"


class ReadableMaterial(StrictModel):
    material_id: str = Field(min_length=1)
    source_asset_ids: list[str] = Field(min_length=1)
    locator: str = Field(min_length=1)
    content_sha256: str = Field(pattern=SHA)
    covered_task_families: list[str] = Field(min_length=1)
    scope_evidence_ref: str = Field(min_length=1)
    scope: Literal["problem_scoped_file", "problem_scoped_bundle", "redacted_transcript"]
    origin: Literal["public_source", "public_task", "private_transcript", "private_source"]
    privacy_proof_ref: str = Field(min_length=1)
    privacy_proof_sha256: str = Field(pattern=SHA)


class MaterialPrivacyEvidence(StrictModel):
    proof_ref: str = Field(min_length=1)
    proof_sha256: str = Field(pattern=SHA)
    material_locator: str = Field(min_length=1)
    material_sha256: str = Field(pattern=SHA)
    status: Literal["passed", "failed", "unknown"]
    completed: bool
    finding_count: int | None = Field(default=None, ge=0)
    channel_status: Literal["public_source", "source_code", "public_conversation_only", "unverified"]
    channel_evidence_ref: str | None = None


def dispatch_materials(
    manifest: SplitManifest,
    materials: list[ReadableMaterial],
    *,
    purpose: Purpose,
    context_id: str,
    privacy_evidence: list[MaterialPrivacyEvidence],
    observed_sha256: dict[str, str],
    prior_exposures: list[dict] = (),
) -> dict:
    """Authorize only exact, currently observed bytes for one isolated consumer.

    Privacy and channel evidence are produced by the scanner/capture boundary,
    not inferred from filenames or a claim of public repository membership.
    ``observed_sha256`` is the caller's immediate pre-read filesystem inventory.
    The returned exposure records must be persisted before reading any bodies.
    """
    manifest.boundary_integrity()
    expected_split = {
        "seed_distillation": "seed",
        "static_corpus": "seed",
        "dev_evaluation": "dev",
        "heldout_task": "heldout",
    }[purpose]
    known = {asset.asset_id: asset for asset in manifest.assets}
    proofs = {(item.proof_ref, item.material_locator): item for item in privacy_evidence}
    global_reasons = []
    if not context_id or not materials:
        global_reasons.append("explicit_context_and_materials_required")
    if len({item.material_id for item in materials}) != len(materials):
        global_reasons.append("duplicate_material_identity")
    existing_splits = {
        item.get("split") for item in prior_exposures if item.get("context_id") == context_id
    }
    if existing_splits - {expected_split}:
        global_reasons.append("consumer_context_already_exposed_to_another_split")
    rows = []
    for material in materials:
        reasons = []
        missing = set(material.source_asset_ids) - known.keys()
        if missing:
            reasons.append("unmanifested_source_provenance")
        else:
            source_families = {known[key].task_family for key in material.source_asset_ids}
            if set(material.covered_task_families) != source_families:
                reasons.append("material_coverage_differs_from_manifested_provenance")
            if any(known[key].split != expected_split for key in material.source_asset_ids):
                reasons.append("material_contains_provenance_from_another_split")
        exposed_families = {
            family
            for item in prior_exposures
            if item.get("purpose")
            in {"implementation", "seed_distillation", "static_corpus", "dev_evaluation"}
            and item.get("body_exposed") is not False
            for family in item.get("task_families", [])
        }
        if expected_split == "heldout" and exposed_families & set(material.covered_task_families):
            reasons.append("heldout_problem_previously_exposed")
        if observed_sha256.get(material.material_id) != material.content_sha256:
            reasons.append("readable_material_current_hash_mismatch_or_missing")
        proof = proofs.get((material.privacy_proof_ref, material.locator))
        if proof is None:
            reasons.append("privacy_evidence_missing_for_readable_material")
        else:
            if (
                proof.proof_sha256 != material.privacy_proof_sha256
                or proof.material_sha256 != material.content_sha256
            ):
                reasons.append("privacy_evidence_bound_to_other_bytes")
            if proof.status != "passed" or not proof.completed or proof.finding_count != 0:
                reasons.append("privacy_scan_not_completed_cleanly")
            expected_channel = (
                "public_conversation_only"
                if material.origin == "private_transcript"
                else "source_code"
                if material.origin == "private_source"
                else "public_source"
            )
            if proof.channel_status != expected_channel or not proof.channel_evidence_ref:
                reasons.append("public_channel_or_source_evidence_missing")
            if material.origin == "private_transcript" and material.scope != "redacted_transcript":
                reasons.append("private_transcript_requires_redacted_material")
        rows.append(
            {
                "material_id": material.material_id,
                "status": "denied" if reasons or global_reasons else "admitted",
                "reasons": reasons,
                "locator": material.locator,
                "content_sha256": material.content_sha256,
                "source_asset_ids": material.source_asset_ids,
                "task_families": material.covered_task_families,
                "privacy_proof_ref": material.privacy_proof_ref,
                "privacy_proof_sha256": material.privacy_proof_sha256,
            }
        )
    allowed = not global_reasons and all(row["status"] == "admitted" for row in rows)
    admission = None
    if allowed:
        admission = admit_materials(
            manifest,
            sorted({key for item in materials for key in item.source_asset_ids}),
            purpose,
        )
    return {
        "schema_version": "mindie-material-dispatch/1",
        "status": "admitted" if allowed else "denied",
        "manifest_sha256": manifest.manifest_sha256,
        "context_id": context_id,
        "purpose": purpose,
        "split": expected_split,
        "reasons": global_reasons,
        "materials": rows,
        "source_admission": admission,
        "exposure_records": [
            {
                "context_id": context_id,
                "purpose": purpose,
                "split": expected_split,
                "task_families": item.covered_task_families,
                "material_id": item.material_id,
                "material_sha256": item.content_sha256,
                "manifest_sha256": manifest.manifest_sha256,
                "body_exposed": True,
            }
            for item in materials
        ]
        if allowed
        else [],
        "body_read_performed": False,
        "consumer_must_persist_exposure_before_read": True,
    }
