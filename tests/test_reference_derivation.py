import hashlib
import json

import pytest

from mindie_bench.mining.material_dispatch import MaterialPrivacyEvidence
from mindie_bench.mining.reference_admission import (
    ProtectedReference,
    PublicSourceEvidence,
    ReferenceMaterial,
    admit_public_references,
)
from mindie_bench.mining.reference_derivation import (
    ReferenceDerivation,
    original_range_for_finding,
    verify_derivation,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True).encode()


def example():
    source = (
        b"// outside selected slice\n"
        b"void cancel() {\r\n"
        b"  // operator@node.example.invalid HELDOUT_CHANGE\r\n"
        b"  // private deployment note\r\n"
        b"  abort_request();\r\n}\n"
    )
    start, end = source.index(b"void"), len(source)
    hidden_start = source.index(b"operator")
    hidden_end = source.index(b"  abort_request")
    hidden = source[hidden_start:hidden_end]
    marker = b"<redacted:source:" + sha(hidden)[:12].encode() + b">\r\n\r\n"
    readable = source[start:hidden_start] + marker + source[hidden_end:end]
    cut = hidden_start - start
    segments = [
        {"kind": "unmodified", "source_start": start, "source_end": hidden_start,
         "derived_start": 0, "derived_end": cut},
        {"kind": "redacted", "source_start": hidden_start, "source_end": hidden_end,
         "derived_start": cut, "derived_end": cut + len(marker)},
        {"kind": "unmodified", "source_start": hidden_end, "source_end": end,
         "derived_start": cut + len(marker), "derived_end": len(readable)},
    ]
    mapping = {
        "schema_version": "mindie-source-derivation-map/1",
        "source_sha256": sha(source), "original_slice_sha256": sha(source[start:end]),
        "derived_sha256": sha(readable), "source_start": start, "source_end": end,
        "segments": segments,
    }
    original_proof = encode({
        "source_sha256": sha(source), "material_sha256": sha(source[start:end]),
        "completed": True, "status": "failed", "finding_count": 1,
    })
    evidence = {"segment_map": encode(mapping), "original_privacy_proof": original_proof}
    review = {
        "schema_version": "mindie-source-derivation-review/1", "status": "approved",
        "replacement_rule": "source", "source_sha256": sha(source),
        "original_slice_sha256": sha(source[start:end]), "derived_sha256": sha(readable),
        "segment_map_sha256": sha(evidence["segment_map"]),
        "original_privacy_proof_sha256": sha(original_proof),
    }
    evidence["transform_review"] = encode(review)
    derivation = ReferenceDerivation(
        kind="redacted_derivative", replacement_rule="source",
        original_slice_sha256=sha(source[start:end]),
        **{key + "_ref": "/evidence/" + key + ".json" for key in evidence},
        **{key + "_sha256": sha(value) for key, value in evidence.items()},
    )
    return source, readable, derivation, evidence, mapping


def verify(source, readable, derivation, evidence, mapping):
    return verify_derivation(
        derivation=derivation, source_bytes=source, readable_bytes=readable,
        source_sha256=sha(source), byte_start=mapping["source_start"],
        byte_end=mapping["source_end"], content_sha256=sha(readable), evidence=evidence,
    )


def test_original_coordinates_survive_length_change_and_crlf_redaction():
    source, readable, derivation, evidence, mapping = example()
    summary = verify(source, readable, derivation, evidence, mapping)
    hidden = mapping["segments"][1]
    assert summary["withheld_original_ranges"] == [[hidden["source_start"], hidden["source_end"]]]
    assert summary["withheld_original_bytes"] == hidden["source_end"] - hidden["source_start"]
    assert summary["affected_source_lines"] == [[3, 4]]
    assert summary["raw_source_complete"] is False
    assert b"HELDOUT_CHANGE" not in readable
    assert readable.splitlines()[-2:] == source.splitlines()[-2:]
    assert bytes(c for c in readable if c in (10, 13)) == bytes(
        c for c in source[mapping["source_start"]:] if c in (10, 13)
    )


def test_unlisted_code_edit_cannot_be_laundered_by_rehashing_output_and_review():
    source, readable, derivation, evidence, mapping = example()
    changed = readable.replace(b"abort_request", b"start_request")
    mapping["derived_sha256"] = sha(changed)
    evidence["segment_map"] = encode(mapping)
    derivation.segment_map_sha256 = sha(evidence["segment_map"])
    review = json.loads(evidence["transform_review"])
    review.update(derived_sha256=sha(changed), segment_map_sha256=derivation.segment_map_sha256)
    evidence["transform_review"] = encode(review)
    derivation.transform_review_sha256 = sha(evidence["transform_review"])
    with pytest.raises(ValueError, match="derivation_segment_bytes_mismatch"):
        verify(source, changed, derivation, evidence, mapping)


def test_findings_map_across_contiguous_unchanged_segments_but_not_redactions():
    segments = [
        {"kind": "unmodified", "source_start": 100, "source_end": 105,
         "derived_start": 0, "derived_end": 5},
        {"kind": "unmodified", "source_start": 105, "source_end": 110,
         "derived_start": 5, "derived_end": 10},
        {"kind": "redacted", "source_start": 110, "source_end": 200,
         "derived_start": 10, "derived_end": 40},
        {"kind": "unmodified", "source_start": 200, "source_end": 210,
         "derived_start": 40, "derived_end": 50},
    ]
    assert original_range_for_finding(segments, 3, 8) == (103, 108)
    assert original_range_for_finding(segments, 42, 48) == (202, 208)
    assert original_range_for_finding(segments, 8, 42) is None
    assert original_range_for_finding(segments, 12, 15) is None


def test_removed_heldout_marker_still_denies_final_admission():
    source, readable, derivation, evidence, mapping = example()
    material = ReferenceMaterial(
        material_id="reference:cancel", source_id="public-source",
        repo_url="https://github.com/example/public-source", revision="a" * 40,
        path="cancel.cpp", git_blob_oid="b" * 40, source_sha256=sha(source),
        byte_start=mapping["source_start"], byte_end=mapping["source_end"],
        locator="/readable/cancel.txt", content_sha256=sha(readable),
        privacy_proof_ref="/privacy/readable.json", privacy_proof_sha256="c" * 64,
        derivation=derivation,
    )
    scan = MaterialPrivacyEvidence(
        proof_ref=material.privacy_proof_ref, proof_sha256=material.privacy_proof_sha256,
        material_locator=material.locator, material_sha256=material.content_sha256,
        status="passed", completed=True, finding_count=0,
        channel_status="public_source", channel_evidence_ref="/public/provenance.json",
    )
    kwargs = {
        "context_id": "new-isolated-context",
        "source_evidence": [PublicSourceEvidence(
            source_id=material.source_id, repo_url=material.repo_url, revision=material.revision,
            public_provenance_ref="/public/provenance.json", public_provenance_sha256="d" * 64,
            verified=True,
        )],
        "privacy_evidence": [scan],
        "observed_sha256": {material.material_id: material.content_sha256},
        "observed_git_blobs": {material.material_id: {
            key: getattr(material, key)
            for key in ("revision", "path", "git_blob_oid", "source_sha256")
        }},
        "observed_source_bytes": {material.material_id: source},
        "observed_readable_bytes": {material.material_id: readable},
        "observed_derivation_evidence": {material.material_id: evidence},
        "protected_references": [],
    }
    assert admit_public_references([material], **kwargs)["status"] == "admitted"
    kwargs["protected_references"] = [ProtectedReference(
        exact_content_markers=["HELDOUT_CHANGE"], family_id="protected-family",
        evidence_ref="/split/frozen.json",
    )]
    denied = admit_public_references([material], **kwargs)
    assert denied["status"] == "denied"
    assert denied["materials"][0]["protected_family_ids"] == ["protected-family"]
    assert denied["exposure_intents"] == []
    kwargs["protected_references"] = [ProtectedReference(
        exact_content_markers=["<redacted:source:"], family_id="derived-marker",
        evidence_ref="/split/frozen.json",
    )]
    assert admit_public_references([material], **kwargs)["materials"][0]["protected_family_ids"] == [
        "derived-marker",
    ]
    kwargs["protected_references"] = []
    scan.material_sha256 = derivation.original_slice_sha256
    assert admit_public_references([material], **kwargs)["status"] == "denied"
