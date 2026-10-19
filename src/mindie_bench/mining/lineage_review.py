"""[Role] Admit exact materials for isolated problem-lineage review before split.

[Invariants] Only explicit material bytes are readable; graph reachability never
grants body access. Review contexts remain quarantined. Relations are evidence
for the existing closure audit, never automatic family or split acceptance.
[Timeouts & Locks] Pure in-memory hashing and validation; no waits or locks.
[Side-Effect Policy] No reads, writes or dispatch. Persist returned exposure
records before reading bodies; append the admission file reference and hash.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Literal

from pydantic import Field, ValidationError

from mindie_bench.models import StrictModel

from .lineage_closure import COARSE_ASSOCIATION_REASONS, _edges
from .material_dispatch import SHA


class LineageReviewMaterial(StrictModel):
    material_id: str = Field(min_length=1)
    source_asset_ids: list[str] = Field(min_length=1)
    locator: str = Field(min_length=1)
    content_sha256: str = Field(pattern=SHA)
    scope: Literal["problem_scoped_file", "problem_scoped_bundle", "redacted_transcript"]
    origin: Literal["public_source", "public_task", "private_transcript", "private_source"]
    scope_evidence_ref: str = Field(min_length=1)
    scope_evidence_sha256: str = Field(pattern=SHA)
    privacy_proof_ref: str = Field(min_length=1)
    privacy_proof_sha256: str = Field(pattern=SHA)


class LineageCitation(StrictModel):
    material_id: str
    source_asset_id: str
    content_sha256: str = Field(pattern=SHA)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


class LineageDecision(StrictModel):
    edge_id: str
    relation: Literal["same_problem", "distinct_problem", "unresolved"]
    evidence: list[LineageCitation]


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_hash(value) -> str:
    return _hash(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    )


def _bound_json(ref, expected, evidence):
    raw = evidence.get(ref)
    if raw is None or _hash(raw) != expected:
        raise ValueError("evidence_hash_mismatch_or_missing")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("evidence_object_required")
    return value


def admit_lineage_review(
    *,
    assets: list[dict],
    strong_edges: list[dict],
    candidate_edges: list[dict],
    review_edge_ids: list[str],
    materials: list[LineageReviewMaterial],
    context_id: str,
    readable_bytes: dict[str, bytes],
    evidence_bytes: dict[str, bytes],
    prior_exposures: list[dict] = (),
) -> dict:
    """Authorize one fresh quarantined context and return every incident edge.

    Scope evidence binds each material to explicit paths of strong edges ending
    at a question endpoint. Repository/container associations cannot form such
    paths, and no path authorizes reading another asset's body.
    """
    reasons, rows = [], []
    known = {a["asset_id"]: a for a in assets}
    strong = _edges(strong_edges, "strong", reasons)
    candidates = _edges(candidate_edges, "candidate", reasons)
    strong_by_id = {e["edge_id"]: e for e in strong}
    candidate_by_id = {e["edge_id"]: e for e in candidates}
    if (
        len(known) != len(assets)
        or len(strong_by_id) != len(strong)
        or len(candidate_by_id) != len(candidates)
    ):
        reasons.append("duplicate_graph_identity")
    if not context_id or not materials or not review_edge_ids:
        reasons.append("explicit_context_materials_and_questions_required")
    if (
        len(set(review_edge_ids)) != len(review_edge_ids)
        or set(review_edge_ids) - candidate_by_id.keys()
    ):
        reasons.append("unknown_or_duplicate_review_edge")
    if len({m.material_id for m in materials}) != len(materials):
        reasons.append("duplicate_material_identity")
    if any(e.get("context_id") == context_id for e in prior_exposures):
        reasons.append("review_context_has_prior_exposure")
    questions = [candidate_by_id[e] for e in review_edge_ids if e in candidate_by_id]
    endpoints = {e[k] for e in questions for k in ("left", "right")}
    selected = set(endpoints)
    if endpoints - known.keys():
        reasons.append("unknown_question_endpoint")
    for material in materials:
        row = {**material.model_dump(), "reasons": [], "endpoint_paths": [], "line_count": 0}
        ids = set(material.source_asset_ids)
        selected.update(ids)
        try:
            if len(ids) != len(material.source_asset_ids) or ids - known.keys():
                raise ValueError("unknown_or_duplicate_source_asset")
            if any(known[i].get("admission_scope") == "metadata_only_container" for i in ids):
                raise ValueError("metadata_only_container_body_denied")
            raw = readable_bytes.get(material.material_id)
            if raw is None or _hash(raw) != material.content_sha256:
                raise ValueError("readable_material_hash_mismatch_or_missing")
            row["line_count"] = len(raw.decode("utf-8").splitlines())
            scope = _bound_json(
                material.scope_evidence_ref, material.scope_evidence_sha256, evidence_bytes
            )
            expected = {
                "schema_version": "mindie-lineage-review-scope/1",
                "material_id": material.material_id,
                "source_asset_ids": material.source_asset_ids,
                "material_sha256": material.content_sha256,
                "scope": material.scope,
                "origin": material.origin,
            }
            if any(scope.get(k) != v for k, v in expected.items()):
                raise ValueError("scope_not_bound_to_exact_material")
            paths = scope.get("endpoint_paths", [])
            if not paths:
                raise ValueError("explicit_endpoint_paths_required")
            if {p["source_asset_id"] for p in paths} != ids:
                raise ValueError("scope_path_missing_for_material_source_asset")
            for path in paths:
                current = path["source_asset_id"]
                if current not in ids or path["endpoint_asset_id"] not in endpoints:
                    raise ValueError("scope_path_outside_selected_material_or_question")
                for edge_id in path["strong_edge_ids"]:
                    edge = strong_by_id.get(edge_id)
                    if (
                        edge is None
                        or current not in (edge["left"], edge["right"])
                        or edge["reason"] in COARSE_ASSOCIATION_REASONS | {"public_source_repository"}
                    ):
                        raise ValueError("scope_path_requires_exact_strong_edges")
                    current = edge["right"] if current == edge["left"] else edge["left"]
                    if (
                        current not in known
                        or known[current].get("admission_scope") == "metadata_only_container"
                    ):
                        raise ValueError("scope_path_through_unknown_or_container_asset")
                    selected.add(current)
                if current != path["endpoint_asset_id"]:
                    raise ValueError("scope_path_does_not_reach_question_endpoint")
            row["endpoint_paths"] = [
                {
                    key: deepcopy(p[key])
                    for key in ("source_asset_id", "endpoint_asset_id", "strong_edge_ids")
                }
                for p in paths
            ]
            proof = _bound_json(
                material.privacy_proof_ref, material.privacy_proof_sha256, evidence_bytes
            )
            channel = (
                "public_conversation_only"
                if material.origin == "private_transcript"
                else "source_code"
                if material.origin == "private_source"
                else "public_source"
            )
            if (
                proof.get("material_locator") != material.locator
                or proof.get("material_sha256") != material.content_sha256
                or proof.get("status") != "passed"
                or proof.get("completed") is not True
                or proof.get("finding_count") != 0
                or proof.get("channel_status") != channel
                or not proof.get("channel_evidence_ref")
            ):
                raise ValueError("privacy_not_complete_clean_and_bound")
            if material.origin == "private_transcript" and material.scope != "redacted_transcript":
                raise ValueError("private_transcript_requires_redacted_material")
        except (ValueError, KeyError, UnicodeError) as error:
            row["reasons"].append(str(error) if type(error) is ValueError else "invalid_bound_evidence")
        rows.append(row)
    allowed = not reasons and not any(r["reasons"] for r in rows)
    for row in rows:
        row["status"] = "admitted" if allowed else "denied"
    boundary = dict(
        context_id=context_id,
        purpose="lineage_review",
        role="lineage_reviewer_only",
        split="quarantine",
        permanent_quarantine=True,
    )
    graph_hash = _json_hash(
        dict(assets=assets, strong_edges=strong_edges, candidate_edges=candidate_edges)
    )
    incident = [
        {
            **{k: e[k] for k in ("kind", "edge_id", "left", "right", "reason")},
            "edge_sha256": _json_hash(e),
        }
        for e in [*strong, *candidates]
        if selected & {e["left"], e["right"]}
    ]
    return {
        "schema_version": "mindie-lineage-review-admission/1",
        **boundary,
        "status": "admitted" if allowed else "denied",
        "reasons": reasons,
        "graph_sha256": graph_hash,
        "review_edge_ids": list(review_edge_ids),
        "selected_asset_ids": sorted(selected),
        "incident_edges": incident,
        "materials": rows,
        "prior_exposures": deepcopy(prior_exposures),
        "family_or_split_acceptance": False,
        "body_read_performed": False,
        "consumer_must_persist_exposure_before_read": True,
        "exposure_records": [
            {
                **boundary,
                "graph_sha256": graph_hash,
                "material_id": m.material_id,
                "material_sha256": m.content_sha256,
                "asset_ids": sorted(m.source_asset_ids),
                "source_asset_ids": sorted(m.source_asset_ids),
                "body_exposed": True,
                "evidence_refs": [m.scope_evidence_ref, m.privacy_proof_ref],
            }
            for m in materials
        ]
        if allowed
        else [],
    }


def accept_lineage_decisions(admission: dict, decisions: list[dict]) -> list[dict]:
    """Project only bound relations and citations; rationale stays private."""
    if admission.get("status") != "admitted" or admission.get("purpose") != "lineage_review":
        raise ValueError("lineage_review_not_admitted")
    questions = {
        e["edge_id"]: e
        for e in admission["incident_edges"]
        if e["kind"] == "candidate" and e["edge_id"] in admission["review_edge_ids"]
    }
    materials = {m["material_id"]: m for m in admission["materials"]}
    results, seen = [], set()
    for raw in decisions:
        try:
            decision = LineageDecision.model_validate(raw)
        except ValidationError:
            raise ValueError("decision_schema_invalid") from None
        edge = questions.get(decision.edge_id)
        if edge is None or decision.edge_id in seen:
            raise ValueError("decision_edge_not_admitted_or_repeated")
        seen.add(decision.edge_id)
        covered = set()
        for cite in decision.evidence:
            material = materials.get(cite.material_id)
            if (
                material is None
                or cite.source_asset_id not in material["source_asset_ids"]
                or cite.content_sha256 != material["content_sha256"]
                or not 1 <= cite.line_start <= cite.line_end <= material["line_count"]
            ):
                raise ValueError("decision_citation_not_bound_to_admitted_material")
            covered.update(
                p["endpoint_asset_id"]
                for p in material["endpoint_paths"]
                if p["source_asset_id"] == cite.source_asset_id
            )
        if decision.relation != "unresolved" and not {edge["left"], edge["right"]} <= covered:
            raise ValueError("decision_requires_both_question_endpoints")
        results.append(decision.model_dump())
    return results


def verified_lineage_review_exposure(exposure: dict, receipt_bytes: dict[str, bytes]) -> bool:
    """Recognize one exact persisted exposure without granting context reuse."""
    raw = receipt_bytes.get(exposure.get("admission_ref"))
    if raw is None or _hash(raw) != exposure.get("admission_sha256"):
        return False
    try:
        receipt = json.loads(raw)
        if not isinstance(receipt, dict):
            return False
        boundary = {
            "purpose": "lineage_review",
            "role": "lineage_reviewer_only",
            "split": "quarantine",
            "permanent_quarantine": True,
        }
        if (
            receipt.get("schema_version") != "mindie-lineage-review-admission/1"
            or receipt.get("status") != "admitted"
            or any(receipt.get(k) != v or exposure.get(k) != v for k, v in boundary.items())
            or receipt.get("context_id") != exposure.get("context_id")
        ):
            return False
        base = {k: v for k, v in exposure.items() if k not in {"admission_ref", "admission_sha256"}}
        if base not in receipt["exposure_records"]:
            return False
        return any(
            m["material_id"] == exposure.get("material_id")
            and m["status"] == "admitted"
            and m["content_sha256"] == exposure.get("material_sha256")
            and sorted(m["source_asset_ids"])
            == exposure.get("source_asset_ids")
            == exposure.get("asset_ids")
            for m in receipt["materials"]
        )
    except (ValueError, KeyError, TypeError):
        return False
