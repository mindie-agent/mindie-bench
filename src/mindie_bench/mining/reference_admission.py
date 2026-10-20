"""Admit pinned public references without inventing benchmark problem families.

[Role] Check source, readable-byte privacy and protected Heldout boundaries for
general research before an isolated model context consumes a source unit.
[Invariants] Public references are exposure records, never independent Heldout
evidence. Discovery and admission do not mean the source has been read.
[Timeouts & Locks] Pure metadata and caller-provided byte checks; Git reads and
privacy scans belong to callers. No subprocesses, waits or locks.
[Side-Effect Policy] Returns explicit admission and exposure intent; no file reads,
model calls, writes, retries or changes to existing benchmark manifests.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from mindie_bench.mining.material_dispatch import MaterialPrivacyEvidence
from mindie_bench.mining.reference_derivation import ReferenceDerivation, verify_derivation
from mindie_bench.models import StrictModel

SHA = r"^[0-9a-f]{64}$"
GIT_OID = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"


class PublicSourceEvidence(StrictModel):
    source_id: str
    repo_url: str
    revision: str = Field(pattern=GIT_OID)
    public_provenance_ref: str = Field(min_length=1)
    public_provenance_sha256: str = Field(pattern=SHA)
    verified: bool


class ReferenceMaterial(StrictModel):
    material_id: str = Field(min_length=1)
    source_id: str
    repo_url: str
    revision: str = Field(pattern=GIT_OID)
    path: str = Field(min_length=1)
    git_blob_oid: str = Field(pattern=GIT_OID)
    source_sha256: str = Field(pattern=SHA)
    byte_start: int = Field(ge=0)
    byte_end: int = Field(ge=0)
    locator: str = Field(min_length=1)
    content_sha256: str = Field(pattern=SHA)
    privacy_proof_ref: str
    privacy_proof_sha256: str = Field(pattern=SHA)
    derivation: ReferenceDerivation | None = None

    @model_validator(mode="after")
    def source_range(self):
        if self.byte_end < self.byte_start:
            raise ValueError("source byte range is reversed")
        return self


class ReferencePrivacyEvidence(MaterialPrivacyEvidence):
    """Public-reference-only review of exact detected false-positive spans.

    Detected findings stay counted. A caller verifies the review file hash and
    each bound span before supplying this receipt; private materials keep the
    ordinary clean-scan contract in material_dispatch.
    """

    status: Literal["passed", "passed_with_reviewed_exceptions", "failed", "unknown"]
    reviewed_exception_count: int = Field(default=0, ge=0)
    exception_review_ref: str | None = None
    exception_review_sha256: str | None = Field(default=None, pattern=SHA)
    exception_source_sha256: str | None = Field(default=None, pattern=SHA)


class ProtectedReference(StrictModel):
    repo_url: str = ""
    path: str | None = None
    family_id: str
    evidence_ref: str
    # A directory prefix is explicit; a filename never silently becomes a glob.
    path_prefix: bool = False
    source_sha256: str | None = Field(default=None, pattern=SHA)
    content_sha256: str | None = Field(default=None, pattern=SHA)
    exact_content_markers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_boundary(self):
        if not (self.path or self.source_sha256 or self.content_sha256 or self.exact_content_markers):
            raise ValueError("protected reference has no path, byte hash or marker")
        if any(not marker for marker in self.exact_content_markers):
            raise ValueError("protected content marker must not be empty")
        return self


def _repository(url: str) -> str:
    return url.removeprefix("https://").removeprefix("http://").removesuffix(".git").rstrip("/")


def protected_match_fields(
    protected: dict,
    *,
    repo_url: str,
    path: str,
    source_sha256: str | None = None,
    content_sha256: str | None = None,
    source_bytes: bytes | None = None,
) -> list[str]:
    """One predicate for generation and final admission, with distinct hash scopes.

    A path protects its repository across revisions. A full-source hash protects
    identical files anywhere; a content hash protects the exact selected slice.
    Markers scan the whole pinned source, including across a slice boundary.
    """
    fields = []
    protected_path = protected.get("path")
    if protected_path and _repository(protected.get("repo_url", "")) == _repository(repo_url):
        if path == protected_path or (
            protected.get("path_prefix") and path.startswith(protected_path.rstrip("/") + "/")
        ):
            fields.append("path")
    if protected.get("source_sha256") and protected["source_sha256"] == source_sha256:
        fields.append("source_sha256")
    if protected.get("content_sha256") and protected["content_sha256"] == content_sha256:
        fields.append("content_sha256")
    if source_bytes is not None and any(
        marker.encode("utf-8") in source_bytes for marker in protected.get("exact_content_markers", [])
    ):
        fields.append("exact_content_markers")
    return fields


def admit_public_references(
    materials: list[ReferenceMaterial],
    *,
    context_id: str,
    source_evidence: list[PublicSourceEvidence],
    privacy_evidence: list[MaterialPrivacyEvidence | ReferencePrivacyEvidence],
    observed_sha256: dict[str, str],
    observed_git_blobs: dict[str, dict],
    protected_references: list[ProtectedReference],
    observed_source_bytes: dict[str, bytes] | None = None,
    observed_readable_bytes: dict[str, bytes] | None = None,
    observed_derivation_evidence: dict[str, dict[str, bytes]] | None = None,
    purpose: Literal["public_reference_research"] = "public_reference_research",
) -> dict:
    """Allow only currently bound references; never create problem/split claims.

    ``observed_git_blobs`` comes from bounded reads of the pinned Git object, not
    a working-tree file. The caller persists returned exposure intent before
    sending bytes, and records actual reading/completion separately afterwards.
    """
    sources = {(row.source_id, row.revision): row for row in source_evidence}
    observed_source_bytes = observed_source_bytes or {}
    observed_readable_bytes = observed_readable_bytes or {}
    observed_derivation_evidence = observed_derivation_evidence or {}
    marker_replay_required = any(row.exact_content_markers for row in protected_references)
    scans = {(row.proof_ref, row.material_locator): row for row in privacy_evidence}
    global_reasons = []
    if purpose != "public_reference_research":
        global_reasons.append("public_reference_research_purpose_required")
    if not context_id or not materials:
        global_reasons.append("explicit_context_and_materials_required")
    if len({row.material_id for row in materials}) != len(materials):
        global_reasons.append("duplicate_material_identity")
    rows = []
    for material in materials:
        reasons = []
        source = sources.get((material.source_id, material.revision))
        if source is None or not source.verified:
            reasons.append("verified_public_source_provenance_missing")
        elif _repository(source.repo_url) != _repository(material.repo_url):
            reasons.append("public_source_repository_mismatch")
        expected_blob = {
            key: getattr(material, key) for key in ("revision", "path", "git_blob_oid", "source_sha256")
        }
        observed_blob = observed_git_blobs.get(material.material_id) or {}
        if any(observed_blob.get(key) != value for key, value in expected_blob.items()):
            reasons.append("pinned_git_blob_identity_mismatch_or_missing")
        if observed_sha256.get(material.material_id) != material.content_sha256:
            reasons.append("readable_material_current_hash_mismatch_or_missing")
        source_bytes = observed_source_bytes.get(material.material_id)
        readable_bytes = observed_readable_bytes.get(material.material_id)
        derivation_summary = None
        original_slice_sha256 = (
            material.derivation.original_slice_sha256
            if material.derivation else material.content_sha256
        )
        if source_bytes is None:
            if material.derivation:
                reasons.append("derivation_original_source_bytes_missing")
            elif marker_replay_required:
                reasons.append("protected_marker_replay_missing")
        elif (
            hashlib.sha256(source_bytes).hexdigest() != material.source_sha256
            or hashlib.sha256(source_bytes[material.byte_start:material.byte_end]).hexdigest()
            != original_slice_sha256
        ):
            reasons.append("protected_source_bytes_not_bound_to_material")
        if material.derivation:
            if readable_bytes is None:
                reasons.append("derivation_readable_bytes_missing")
            elif source_bytes is not None:
                try:
                    derivation_summary = verify_derivation(
                        derivation=material.derivation,
                        source_bytes=source_bytes, readable_bytes=readable_bytes,
                        source_sha256=material.source_sha256,
                        byte_start=material.byte_start, byte_end=material.byte_end,
                        content_sha256=material.content_sha256,
                        evidence=observed_derivation_evidence.get(material.material_id, {}),
                    )
                except ValueError as error:
                    reasons.append("derivation_not_verified:" + str(error))
        blocked_families = []
        for protected in protected_references:
            original_match = protected_match_fields(
                protected.model_dump(), repo_url=material.repo_url, path=material.path,
                source_sha256=material.source_sha256, content_sha256=original_slice_sha256,
                source_bytes=source_bytes,
            )
            derived_match = material.derivation and protected_match_fields(
                protected.model_dump(), repo_url=material.repo_url, path=material.path,
                content_sha256=material.content_sha256, source_bytes=readable_bytes,
            )
            if original_match or derived_match:
                blocked_families.append(protected.family_id)
        if blocked_families:
            reasons.append("protected_heldout_source_material")
        scan = scans.get((material.privacy_proof_ref, material.locator))
        if scan is None:
            reasons.append("privacy_evidence_missing_for_readable_material")
        else:
            clean = scan.status == "passed" and scan.finding_count == 0
            reviewed = (
                isinstance(scan, ReferencePrivacyEvidence)
                and scan.status == "passed_with_reviewed_exceptions"
                and scan.finding_count is not None
                and scan.finding_count > 0
                and scan.reviewed_exception_count == scan.finding_count
                and bool(scan.exception_review_ref)
                and bool(scan.exception_review_sha256)
                and scan.exception_source_sha256 == material.source_sha256
            )
            if (
                scan.proof_sha256 != material.privacy_proof_sha256
                or scan.material_sha256 != material.content_sha256
                or not scan.completed
                or not (clean or reviewed)
                or scan.channel_status != "public_source"
                or not scan.channel_evidence_ref
            ):
                reasons.append("privacy_evidence_not_complete_clean_or_reviewed_and_bound")
        rows.append(
            {
                "material_id": material.material_id,
                "reasons": reasons,
                "protected_family_ids": sorted(set(blocked_families)),
                "status": "denied" if reasons or global_reasons else "admitted",
                **({"derivation": derivation_summary} if material.derivation else {}),
            }
        )
    allowed = not global_reasons and all(row["status"] == "admitted" for row in rows)
    return {
        "schema_version": (
            "mindie-public-reference-admission/2"
            if any(material.derivation for material in materials)
            else "mindie-public-reference-admission/1"
        ),
        "purpose": purpose,
        "context_id": context_id,
        "status": "admitted" if allowed else "denied",
        "reasons": global_reasons,
        "materials": rows,
        "benchmark_problem_lineage_claimed": False,
        "independent_heldout_claimed": False,
        "body_read_performed": False,
        "protected_marker_replay_required": marker_replay_required,
        "exposure_intents": [
            dict(
                material.model_dump(mode="json"),
                context_id=context_id,
                purpose=purpose,
                body_exposure_state="authorized_not_yet_dispatched",
            )
            for material in materials
        ]
        if allowed
        else [],
    }
