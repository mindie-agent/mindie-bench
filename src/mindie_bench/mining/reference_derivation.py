"""Verify an explicit redacted view without changing its pinned source identity.

[Role] Bind reviewed source-span redactions to their original and readable bytes.
[Invariants] Maps partition both byte streams; unchanged bytes compare exactly.
Removed bytes remain missing evidence, including when they contain Heldout markers.
[Timeouts & Locks] Pure in-memory validation; no processes, waits or locks.
[Side-Effect Policy] Returns verified segment and loss metadata; never scans,
reads files, writes a copy, grants privacy exceptions or invokes a model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from mindie_bench.models import StrictModel

SHA = r"^[0-9a-f]{64}$"


class ReferenceDerivation(StrictModel):
    kind: Literal["redacted_derivative"]
    replacement_rule: Literal["source"]
    original_slice_sha256: str = Field(pattern=SHA)
    segment_map_ref: str = Field(min_length=1)
    segment_map_sha256: str = Field(pattern=SHA)
    transform_review_ref: str = Field(min_length=1)
    transform_review_sha256: str = Field(pattern=SHA)
    original_privacy_proof_ref: str = Field(min_length=1)
    original_privacy_proof_sha256: str = Field(pattern=SHA)


class SourceSegment(StrictModel):
    kind: Literal["unmodified", "redacted"]
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    derived_start: int = Field(ge=0)
    derived_end: int = Field(ge=0)


class SourceDerivationMap(StrictModel):
    schema_version: Literal["mindie-source-derivation-map/1"]
    source_sha256: str = Field(pattern=SHA)
    original_slice_sha256: str = Field(pattern=SHA)
    derived_sha256: str = Field(pattern=SHA)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    segments: list[SourceSegment] = Field(min_length=1)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redacted_replacement(original_span: bytes) -> bytes:
    """Keep every original CR/LF so original line numbers remain usable."""
    return (
        b"<redacted:source:" + digest(original_span)[:12].encode("ascii") + b">"
        + bytes(value for value in original_span if value in (10, 13))
    )


def verify_derivation(
    *,
    derivation: ReferenceDerivation,
    source_bytes: bytes,
    readable_bytes: bytes,
    source_sha256: str,
    byte_start: int,
    byte_end: int,
    content_sha256: str,
    evidence: dict[str, bytes],
) -> dict:
    """Reconstruct the supplied view and return exact original-byte losses.

    The caller supplies hash-bound report bytes, not an assertion that a report
    exists. Privacy scanning and exact finding-to-review mapping remain separate.
    """
    for key, expected in (
        ("segment_map", derivation.segment_map_sha256),
        ("transform_review", derivation.transform_review_sha256),
        ("original_privacy_proof", derivation.original_privacy_proof_sha256),
    ):
        if key not in evidence or digest(evidence[key]) != expected:
            raise ValueError("derivation_evidence_missing_or_changed:" + key)
    mapping = SourceDerivationMap.model_validate_json(evidence["segment_map"])
    original = source_bytes[byte_start:byte_end]
    if (
        byte_end > len(source_bytes)
        or digest(source_bytes) != source_sha256
        or digest(original) != derivation.original_slice_sha256
        or digest(readable_bytes) != content_sha256
        or mapping.source_sha256 != source_sha256
        or mapping.original_slice_sha256 != derivation.original_slice_sha256
        or mapping.derived_sha256 != content_sha256
        or (mapping.source_start, mapping.source_end) != (byte_start, byte_end)
    ):
        raise ValueError("derivation_source_or_readable_identity_mismatch")
    review = json.loads(evidence["transform_review"])
    expected_review = {
        "schema_version": "mindie-source-derivation-review/1",
        "status": "approved",
        "replacement_rule": "source",
        "source_sha256": source_sha256,
        "original_slice_sha256": derivation.original_slice_sha256,
        "derived_sha256": content_sha256,
        "segment_map_sha256": derivation.segment_map_sha256,
        "original_privacy_proof_sha256": derivation.original_privacy_proof_sha256,
    }
    if any(review.get(key) != value for key, value in expected_review.items()):
        raise ValueError("derivation_review_not_approved_or_bound")
    original_proof = json.loads(evidence["original_privacy_proof"])
    if (
        original_proof.get("source_sha256") != source_sha256
        or original_proof.get("material_sha256") != derivation.original_slice_sha256
        or original_proof.get("completed") is not True
        or original_proof.get("status") not in {"passed", "failed"}
    ):
        raise ValueError("original_slice_privacy_proof_not_complete_or_bound")
    source_cursor, derived_cursor = byte_start, 0
    unchanged, losses, affected_lines = [], [], []
    for segment in mapping.segments:
        if (
            (segment.source_start, segment.derived_start) != (source_cursor, derived_cursor)
            or segment.source_end <= segment.source_start
            or segment.source_end > byte_end
            or segment.derived_end <= segment.derived_start
            or segment.derived_end > len(readable_bytes)
        ):
            raise ValueError("derivation_segments_do_not_partition_both_streams")
        raw = source_bytes[segment.source_start:segment.source_end]
        visible = readable_bytes[segment.derived_start:segment.derived_end]
        expected = raw if segment.kind == "unmodified" else redacted_replacement(raw)
        if visible != expected:
            raise ValueError("derivation_segment_bytes_mismatch")
        target = unchanged if segment.kind == "unmodified" else losses
        target.append([segment.source_start, segment.source_end])
        if segment.kind == "redacted":
            first = source_bytes[:segment.source_start].count(b"\n") + 1
            last = source_bytes[:segment.source_end - 1].count(b"\n") + 1
            affected_lines.append([first, last])
        source_cursor, derived_cursor = segment.source_end, segment.derived_end
    if (source_cursor, derived_cursor) != (byte_end, len(readable_bytes)):
        raise ValueError("derivation_segments_do_not_cover_both_streams")
    return {
        "material_kind": "redacted_derivative",
        "original_slice_sha256": derivation.original_slice_sha256,
        "readable_sha256": content_sha256,
        "segment_map_sha256": derivation.segment_map_sha256,
        "original_bytes": byte_end - byte_start,
        "unmodified_original_ranges": unchanged,
        "withheld_original_ranges": losses,
        "withheld_original_bytes": sum(end - start for start, end in losses),
        "derived_display_bytes": len(readable_bytes),
        "affected_source_lines": affected_lines,
        "raw_source_complete": not losses,
        "segments": [segment.model_dump() for segment in mapping.segments],
    }


def original_range_for_finding(segments: list[dict], start: int, end: int) -> tuple[int, int] | None:
    """Map a finding only through contiguous byte-identical source segments.

    A finding spanning adjacent unchanged segments is valid. One crossing any
    replacement or discontinuity cannot inherit an original finding's review.
    """
    cursor, original_start, original_end = start, None, None
    for segment in segments:
        lo, hi = max(cursor, segment["derived_start"]), min(end, segment["derived_end"])
        if lo >= hi:
            continue
        if lo != cursor or segment["kind"] != "unmodified":
            return None
        source_lo = segment["source_start"] + lo - segment["derived_start"]
        source_hi = source_lo + hi - lo
        if original_end is not None and source_lo != original_end:
            return None
        if original_start is None:
            original_start = source_lo
        original_end, cursor = source_hi, hi
        if cursor == end:
            return original_start, original_end
    return None
