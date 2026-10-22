"""P3 lineage publication: local private verification then public CI evidence.

[Role] Verify original Seed/capture/C6/review/corpus bytes during local collect,
project a path-free self-contained lineage, and on CI recompute public
lineage/receipt/cost/identity consistency without reopening private originals.
[Invariants] Local admission always uses P3LineageBinding.prepare with artifact
verification enabled. No locator, absolute path, private corpus or native log
enters public evidence. The public projection keeps every receipt hash, cost
role and claim-readiness fact. CI never falls back to private locators.
[Timeouts & Locks] Bounded local receipt/corpus reads during collect; public
verification is pure and performs no file reads.
[Side-Effect Policy] Read-only; never starts a model, NPU or external write.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mindie_bench.io import digest, read_json
from mindie_bench.runner.matrix import (
    BudgetEnvelope,
    MatchingConditions,
    ProductCell,
    ProductGroup,
    ProductProfile,
)
from mindie_bench.runner.p3_lineage import (
    P3LineageBinding,
    P3LineageManifest,
    P3LineageRejected,
)
from mindie_bench.runner.p3_receipts import (
    P3StageObservation,
    capture_observations,
    consumer_observation,
    ledger_numeric_mismatches,
    review_observation,
    seed_observation,
    stage_order_reasons,
)
from mindie_bench.value_product_protocol import _hex, _source_identity
from mindie_bench.value_product_public import public_admission

P3_PUBLIC_SCHEMA = "mindie-product-public-p3-lineage/1"


@dataclass(frozen=True)
class P3PublicationProfile:
    """Public-evidence-only P3 profile; never used to start a new execution."""

    profile_id: str
    group: ProductGroup
    protocol: str
    fidelity: str
    cohort: str
    conditions: MatchingConditions
    budget: BudgetEnvelope


def _receipt(path: str) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _tokens(value) -> dict:
    value = value or {}
    return {
        key: value.get(key)
        for key in (
            "input_uncached",
            "input_cached",
            "cache_creation",
            "output",
            "indexer_small_model_total",
            "total",
            "complete",
        )
        if key in value
    }


def _project_seed(payload: dict) -> dict:
    primary = payload.get("primary_metrics") or {}
    auxiliary = payload.get("auxiliary_metrics") or {}
    return {
        "trial_id": payload.get("trial_id"),
        "instance_id": payload.get("instance_id"),
        "terminal_status": payload.get("terminal_status"),
        "task_digest": payload.get("task_digest"),
        "primary_metrics": {
            key: primary.get(key)
            for key in (
                "model_calls",
                "npu_test_runs",
                "remote_test_invocations",
                "cost_complete",
                "estimated_cost_usd",
            )
        } | {"tokens": _tokens(primary.get("tokens"))},
        "auxiliary_metrics": {
            "capture_model_calls": auxiliary.get("capture_model_calls"),
            "capture_provider_tokens": _tokens(auxiliary.get("capture_provider_tokens")),
        },
        "model_calls": primary.get("model_calls"),
        "npu_runs": primary.get("npu_test_runs", primary.get("remote_test_invocations")),
        "cost_complete": primary.get("cost_complete"),
        "estimated_cost_usd": primary.get("estimated_cost_usd"),
        "tokens": _tokens(primary.get("tokens")),
    }


def _project_capture(payload: dict) -> dict:
    stages = [
        {
            key: stage.get(key)
            for key in ("name", "status", "started_sec", "ended_sec", "elapsed_sec")
            if key in stage or key == "name"
        }
        for stage in payload.get("stage_observations") or []
    ]
    distill = payload.get("distill") or {}
    distill_usage = distill.get("usage") or {}
    return {
        "status": payload.get("status"),
        "task_state": payload.get("task_state"),
        "protocol": payload.get("protocol"),
        "native_harness": payload.get("native_harness"),
        "corpus_lint": {"status": (payload.get("corpus_lint") or {}).get("status")},
        "cleanup": {"npu_runs": (payload.get("cleanup") or {}).get("npu_runs")},
        "cost_complete": payload.get("cost_complete"),
        "estimated_cost_usd": payload.get("estimated_cost_usd"),
        "tokens": {
            "total": (payload.get("tokens") or {}).get("total"),
            "complete": (payload.get("tokens") or {}).get("complete"),
        },
        "distill": {
            key: distill.get(key)
            for key in ("status", "model_calls", "billing_status", "cleanup_failed")
        } | {"usage": {"estimated_cost_usd": distill_usage.get("estimated_cost_usd")}},
        "indexer_usage": {
            key: (payload.get("indexer_usage") or {}).get(key)
            for key in (
                "model_calls",
                "input_tokens",
                "cached_input_tokens",
                "cache_creation",
                "output_tokens",
                "unknown_calls",
                "npu_runs",
            )
        },
        "stage_observations": stages,
    }


def _project_consumer(payload: dict) -> dict:
    return {
        "status": payload.get("status"),
        "model_calls": payload.get("model_calls"),
        "npu_runs": payload.get("npu_runs"),
        "provider_writes": payload.get("provider_writes"),
        "remote_writes": payload.get("remote_writes"),
        "corpus_sha256": payload.get("corpus_sha256"),
        "scope": payload.get("scope"),
        "original_receipts_unchanged": payload.get("original_receipts_unchanged"),
        "raw_capture_receipt_sha256": (payload.get("raw_capture_receipt") or {}).get("sha256"),
        "raw_trial_receipt_sha256": (payload.get("raw_trial_receipt") or {}).get("sha256"),
        "raw_capture_status": payload.get("raw_capture_status"),
        "raw_trial_status": payload.get("raw_trial_status"),
    }


def _project_review(payload: dict) -> dict:
    return {
        "status": payload.get("status"),
        "verdict": payload.get("verdict"),
        "independent": payload.get("independent"),
        "organizer_receipt_sha256": payload.get("organizer_receipt_sha256"),
        "seed_family_canonical": payload.get("seed_family_canonical"),
        "heldout_family_canonical": payload.get("heldout_family_canonical"),
        "model_calls": payload.get("model_calls"),
        "npu_runs": payload.get("npu_runs"),
        "tokens": _tokens(payload.get("tokens")),
        "cost_complete": payload.get("cost_complete"),
        "estimated_cost_usd": payload.get("estimated_cost_usd"),
    }


def project_public_lineage(
    manifest: P3LineageManifest,
    binding: P3LineageBinding,
    *,
    source: dict,
    comparison_cohort: str,
) -> dict:
    """Project verified private lineage bytes into a path-free public record."""
    admission = binding.admission
    receipts = {}
    for role, locator, sha256 in (
        ("seed", manifest.seed.receipt_locator, manifest.seed.receipt_sha256),
        ("capture", manifest.capture.receipt_locator, manifest.capture.receipt_sha256),
        ("consumer", manifest.consumer.receipt_locator, manifest.consumer.receipt_sha256),
        ("review", manifest.review.receipt_locator, manifest.review.receipt_sha256),
    ):
        payload, observed = _receipt(locator)
        if observed != sha256:
            raise ValueError(f"p3 {role} receipt changed after admission")
        projector = {
            "seed": _project_seed,
            "capture": _project_capture,
            "consumer": _project_consumer,
            "review": _project_review,
        }[role]
        receipts[role] = {
            "role": role,
            "sha256": sha256,
            **projector(payload),
        }
    payload = {
        "schema": P3_PUBLIC_SCHEMA,
        "content_identity_sha256": manifest.content_identity(),
        "manifest_sha256": manifest.digest(),
        "admission_sha256": admission.get("admission_sha256"),
        "target": admission.get("target"),
        "source_sha256": digest(source),
        "comparison_cohort": comparison_cohort,
        "seed_task_id": manifest.seed.task_id,
        "seed_task_sha256": manifest.seed.task_sha256,
        "seed_trial_id": manifest.seed.trial_id,
        "downstream_task_sha256": manifest.downstream.task_sha256,
        "frozen_corpus_tree_sha256": manifest.frozen_corpus.tree_sha256,
        "public_receipts": receipts,
        "admission": public_admission(admission),
    }
    payload["public_lineage_sha256"] = digest(payload)
    return payload


def _public_claim(
    raw: dict,
    public: dict,
    profile: P3PublicationProfile,
    source: dict,
    cell: ProductCell,
    *,
    comparison_cohort: str,
) -> dict:
    return {
        "status": "admitted",
        "required": bool(raw.get("required")),
        "scope": raw.get("supplementary_scope"),
        "profile": profile,
        "source": source,
        "cell": cell,
        "public_lineage": public,
        "comparison_cohort": comparison_cohort,
    }


def load_private_p3_claim(
    suite: dict,
    base: Path | None,
    releases: dict,
    benchmark: dict,
    preparation: dict,
) -> dict:
    """Local collect: verify original bytes with artifacts enabled, then project."""
    raw = suite.get("p3")
    if not raw:
        return {"status": "missing", "required": False, "reasons": ["p3_chain_not_declared"]}
    required = bool(raw.get("required"))
    scope = raw.get("supplementary_scope")
    if not raw.get("profile") or not raw.get("lineage_manifest") or not raw.get("task"):
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": ["p3_chain_not_declared"],
        }
    declared_identity = raw.get("manifest_content_identity_sha256")
    if not _hex(declared_identity, 64):
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": ["p3_manifest_content_identity_missing"],
        }
    comparison_cohort = raw.get("comparison_cohort")
    try:
        pointer = raw["lineage_manifest"]
        if isinstance(pointer, dict):
            payload = pointer
        else:
            path = Path(pointer)
            if not path.is_absolute():
                path = (base or Path.cwd()) / path
            payload = read_json(path)
        manifest = P3LineageManifest.model_validate(payload)
        if manifest.content_identity() != declared_identity:
            raise ValueError("p3 lineage manifest differs from the declared content identity")
        relocated = manifest
        if base is not None:
            relocated = manifest.model_copy(deep=True)
            for owner, name in (
                (relocated.seed, "receipt_locator"),
                (relocated.seed, "seed_root_locator"),
                (relocated.capture, "receipt_locator"),
                (relocated.consumer, "receipt_locator"),
                (relocated.review, "receipt_locator"),
                (relocated.frozen_corpus, "locator"),
            ):
                value = getattr(owner, name)
                if value and not Path(value).is_absolute():
                    setattr(owner, name, str((base / value).resolve()))
        binding = P3LineageBinding.prepare(relocated)
        source = _source_identity(
            raw["profile"],
            ProductGroup.P2_V003_FULL,
            releases,
            cohort_id=raw["profile"]["cohort"],
            preparation=preparation,
        )
        if source["benchmark"] != benchmark:
            raise ValueError("p3 source benchmark differs from the common runner")
        profile = ProductProfile(
            profile_id=raw["profile"]["profile_id"],
            group=ProductGroup.P3_V003_FLYWHEEL,
            experiment_profile=manifest.content_identity(),
            protocol=raw["profile"]["protocol"],
            fidelity=raw["profile"].get("fidelity", "matched"),
            cohort=raw["profile"]["cohort"],
            conditions=MatchingConditions(**raw["profile"]["conditions"]),
            budget=BudgetEnvelope(**raw["profile"]["budget"]),
            p3_lineage=binding,
        )
        target = manifest.target
        task = raw["task"]
        if (
            task.get("task_digest") != manifest.downstream.task_sha256
            or task.get("oracle_digest") != manifest.downstream.oracle_sha256
        ):
            raise ValueError("p3 downstream task/oracle differs from the admitted lineage")
        if target.matrix_task_family and task.get("task_family") != target.matrix_task_family:
            raise ValueError("p3 task family differs from the admitted lineage label")
        if target.hardware_tier and task.get("hardware_tier") != target.hardware_tier:
            raise ValueError("p3 hardware tier differs from the admitted lineage label")
        cell = ProductCell(
            ProductGroup.P3_V003_FLYWHEEL,
            profile.profile_id,
            profile.cohort,
            profile.protocol,
            target.task_id,
            task["task_family"],
            task["hardware_tier"],
            raw.get("repetition", 1),
            raw.get("order", 0),
        )
        public = project_public_lineage(
            manifest, binding, source=source, comparison_cohort=comparison_cohort
        )
        return {
            **_public_claim(raw, public, profile, source, cell, comparison_cohort=comparison_cohort),
            "binding": binding,
            "manifest": manifest,
            "declared_identity": declared_identity,
        }
    except P3LineageRejected as error:
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": list(error.reasons)[:12],
        }
    except TimeoutError:
        raise
    except OSError:
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": ["p3_private_artifact_unreadable"],
        }
    except (ValueError, KeyError):
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": ["p3_private_validation_failed"],
        }


def _verify_cost_contract(
    admission: dict, ledger: dict, roles: dict, public: dict, problems: list
) -> None:
    """Recompute the declared downstream and lifecycle contract from public roles."""
    downstream = roles.get("downstream")
    if not isinstance(downstream, dict):
        problems.append("p3_public_downstream_role_missing")
        return
    if downstream.get("stage_id") != "downstream_declared":
        problems.append("p3_public_downstream_stage_id_invalid")
    if downstream.get("basis") != "declared":
        problems.append("p3_public_downstream_basis_invalid")
    if downstream.get("complete") is not False:
        problems.append("p3_public_downstream_complete_invalid")
    if downstream.get("price_basis") != "unknown":
        problems.append("p3_public_downstream_price_basis_invalid")
    if downstream.get("tokens_basis") != "unknown":
        problems.append("p3_public_downstream_tokens_basis_invalid")
    if downstream.get("estimated_cost_usd") is not None:
        problems.append("p3_public_downstream_cost_must_be_unknown")
    for field in (
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_tokens",
        "output_tokens",
        "tokens_total",
    ):
        if downstream.get(field) is not None:
            problems.append(f"p3_public_downstream_{field}_must_be_unknown")
    if downstream.get("receipt_sha256") != public.get("downstream_task_sha256"):
        problems.append("p3_public_downstream_receipt_sha_mismatch")
    for field in ("model_calls", "npu_runs"):
        value = downstream.get(field)
        if type(value) is not int or value < 0:
            problems.append(f"p3_public_downstream_{field}_invalid")

    attribution = {
        row.get("stage_id"): row
        for row in ledger.get("attribution", [])
        if isinstance(row, dict)
    }
    for role, row in roles.items():
        attributed = attribution.get(row.get("stage_id"))
        if attributed is None:
            problems.append(f"p3_public_attribution_missing:{role}")
            continue
        for field in (
            "stage_id",
            "receipt_sha256",
            "model_calls",
            "tokens_total",
            "tokens_basis",
            "npu_runs",
            "estimated_cost_usd",
            "price_basis",
            "complete",
        ):
            if attributed.get(field) != row.get(field):
                problems.append(f"p3_public_attribution_mismatch:{role}:{field}")

    unknown = sorted(
        role for role, row in roles.items() if row.get("complete") is not True
    )
    unpriced = sorted(
        role for role, row in roles.items() if row.get("estimated_cost_usd") is None
    )
    if sorted(ledger.get("unknown_usage") or []) != unknown:
        problems.append("p3_public_unknown_usage_mismatch")
    if sorted(ledger.get("unpriced_usage") or []) != unpriced:
        problems.append("p3_public_unpriced_usage_mismatch")
    prerequisite_unknown = sorted(role for role in unknown if role != "downstream")
    if sorted(ledger.get("prerequisite_unknown_usage") or []) != prerequisite_unknown:
        problems.append("p3_public_prerequisite_unknown_usage_mismatch")

    totals = {}
    for field in ("model_calls", "tokens_total", "estimated_cost_usd", "npu_runs"):
        values = [row.get(field) for row in roles.values() if row.get(field) is not None]
        totals[field] = sum(values) if values else None
    if ledger.get("known_totals") != totals:
        problems.append("p3_public_known_totals_mismatch")

    prerequisites = ("seed", "capture", "organizer", "review")
    prerequisite_complete = all(
        isinstance(roles.get(role), dict) and roles[role].get("complete") is True
        for role in prerequisites
    )
    all_complete = all(row.get("complete") is True for row in roles.values())
    if admission.get("prerequisite_cost_complete") is not prerequisite_complete:
        problems.append("p3_public_prerequisite_completeness_mismatch")
    if admission.get("lifecycle_cost_complete") is not all_complete:
        problems.append("p3_public_lifecycle_completeness_mismatch")
    claim_ready = admission.get("claim_ready") is True
    if admission.get("prerequisite_claim_ready") is not (prerequisite_complete and claim_ready):
        problems.append("p3_public_prerequisite_claim_readiness_mismatch")
    if admission.get("lifecycle_claim_ready") is not (all_complete and claim_ready):
        problems.append("p3_public_lifecycle_claim_readiness_mismatch")
    if not prerequisite_complete:
        problems.append("p3_public_prerequisite_roles_incomplete")
    if prerequisite_unknown:
        problems.append("p3_public_prerequisite_unknown_usage_present")

    expected_public_ledger = digest(
        {
            "roles": roles,
            "attribution": ledger.get("attribution"),
            "unknown_usage": ledger.get("unknown_usage"),
            "prerequisite_unknown_usage": ledger.get("prerequisite_unknown_usage"),
            "unpriced_usage": ledger.get("unpriced_usage"),
            "known_totals": ledger.get("known_totals"),
        }
    )
    if ledger.get("public_ledger_sha256") != expected_public_ledger:
        problems.append("p3_public_ledger_hash_mismatch")


def _verify_public_lineage(public: dict, declared_identity: str, target: dict) -> list[str]:
    problems = []
    if public.get("schema") != P3_PUBLIC_SCHEMA:
        problems.append("p3_public_schema_invalid")
    if public.get("content_identity_sha256") != declared_identity:
        problems.append("p3_public_content_identity_mismatch")
    if public.get("target") != target:
        problems.append("p3_public_target_mismatch")
    stored = public.get("public_lineage_sha256")
    payload = {key: value for key, value in public.items() if key != "public_lineage_sha256"}
    if stored != digest(payload):
        problems.append("p3_public_lineage_hash_mismatch")
    admission = public.get("admission") or {}
    if admission.get("status") != "admitted":
        problems.append("p3_public_admission_not_admitted")
    if admission.get("claim_ready") is not True:
        problems.append("p3_public_claim_not_ready")
    if admission.get("prerequisite_cost_complete") is not True:
        problems.append("p3_public_prerequisite_cost_incomplete")
    if admission.get("manifest_sha256") != public.get("manifest_sha256"):
        problems.append("p3_public_manifest_hash_mismatch")
    if admission.get("admission_sha256") != public.get("admission_sha256"):
        problems.append("p3_public_admission_hash_mismatch")
    seed = admission.get("seed") or {}
    heldout = admission.get("heldout") or {}
    if not seed or not heldout:
        problems.append("p3_public_canonical_families_missing")
    else:
        if seed.get("canonical_id") == heldout.get("canonical_id"):
            problems.append("p3_public_canonical_family_collision")
        if set(seed.get("content_sha256") or []) & set(heldout.get("content_sha256") or []):
            problems.append("p3_public_shared_content_across_splits")
        if seed.get("task_sha256") == heldout.get("task_sha256"):
            problems.append("p3_public_shared_task_across_splits")
        if seed.get("oracle_sha256") == heldout.get("oracle_sha256"):
            problems.append("p3_public_shared_oracle_across_splits")
    receipts = dict(public.get("public_receipts", {}))
    if set(receipts) != {"seed", "capture", "consumer", "review"}:
        problems.append("p3_public_receipt_roles_incomplete")
        return problems
    stages = {item.get("role"): item.get("sha256") for item in admission.get("stages", [])}
    for role in ("seed", "capture", "consumer", "review"):
        if receipts[role].get("sha256") != stages.get(role):
            problems.append(f"p3_public_receipt_stage_mismatch:{role}")
    if receipts["consumer"].get("raw_capture_receipt_sha256") != stages.get("capture"):
        problems.append("p3_public_consumer_raw_capture_sha_mismatch")
    if receipts["consumer"].get("raw_trial_receipt_sha256") != stages.get("seed"):
        problems.append("p3_public_consumer_raw_trial_sha_mismatch")
    if public.get("seed_task_id") != receipts["seed"].get("instance_id"):
        problems.append("p3_public_seed_task_id_mismatch")
    if public.get("seed_task_sha256") != receipts["seed"].get("task_digest"):
        problems.append("p3_public_seed_task_sha_mismatch")
    if public.get("downstream_task_sha256") != (public.get("target") or {}).get("task_sha256") and (
        public.get("downstream_task_sha256") != (admission.get("heldout") or {}).get("task_sha256")
    ):
        problems.append("p3_public_downstream_task_sha_mismatch")
    review = receipts["review"]
    if review.get("seed_family_canonical") != (admission.get("seed") or {}).get("canonical_id"):
        problems.append("p3_public_review_seed_family_mismatch")
    if review.get("heldout_family_canonical") != (admission.get("heldout") or {}).get("canonical_id"):
        problems.append("p3_public_review_heldout_family_mismatch")
    if receipts["consumer"].get("corpus_sha256") != public.get("frozen_corpus_tree_sha256"):
        problems.append("p3_public_consumer_corpus_mismatch")
    # Recompute the public cost observations from the public receipts and compare
    # every attributed numeric value with the public cost ledger.
    ledger = admission.get("cost_ledger") or {}
    seed_trial = {
        "primary_metrics": receipts["seed"].get("primary_metrics") or {},
        "auxiliary_metrics": receipts["seed"].get("auxiliary_metrics") or {},
    }
    try:
        seed_observed = seed_observation(seed_trial, receipts["seed"]["sha256"])
        capture_observed, organizer_observed = capture_observations(
            receipts["capture"], seed_trial, receipts["capture"]["sha256"]
        )
        review_observed = review_observation(receipts["review"], receipts["review"]["sha256"])
        consumer_observed = consumer_observation(
            receipts["consumer"], receipts["consumer"]["sha256"]
        )
        expected_roles = {
            "seed": seed_observed.model_dump(mode="json"),
            "capture": capture_observed.model_dump(mode="json"),
            "organizer": organizer_observed.model_dump(mode="json"),
            "review": review_observed.model_dump(mode="json"),
        }
        problems.extend(ledger_numeric_mismatches(ledger, expected_roles))
        if consumer_observed is None or consumer_observed.model_calls != 0:
            problems.append("p3_public_consumer_model_calls_invalid")
    except (TypeError, ValueError):
        problems.append("p3_public_cost_observations_invalid")
    timeline = admission.get("timeline") or {}
    try:
        stages_observed = [
            P3StageObservation.model_validate(
                {key: value for key, value in item.items() if key != "error_sha256"}
            )
            for item in timeline.get("stage_observations", [])
        ]
        problems.extend(stage_order_reasons(stages_observed))
    except (TypeError, ValueError):
        problems.append("p3_public_timeline_stages_invalid")
    public_stages = {
        item.get("name"): item
        for item in receipts["capture"].get("stage_observations", [])
        if isinstance(item, dict)
    }
    for item in timeline.get("stage_observations", []):
        name = item.get("stage_id")
        observed = public_stages.get(name)
        if observed is None:
            continue
        for field in ("status", "started_sec", "ended_sec", "elapsed_sec"):
            if observed.get(field) != item.get(field):
                problems.append(f"p3_public_timeline_stage_mismatch:{name}:{field}")
    absolute_available = timeline.get("absolute_available") is True
    missing_absolute = timeline.get("missing_absolute") or []
    if absolute_available and missing_absolute:
        problems.append("p3_public_absolute_timeline_inconsistent")
    if timeline.get("inference_performed") is not False:
        problems.append("p3_public_timeline_claims_inference")
    if absolute_available:
        times = [
            timeline.get("seed_completed_at"),
            timeline.get("capture_completed_at"),
            timeline.get("corpus_frozen_at"),
            timeline.get("downstream_started_at"),
        ]
        try:
            parsed = [datetime.fromisoformat(value) for value in times]
        except (TypeError, ValueError):
            problems.append("p3_public_absolute_timeline_invalid")
        else:
            if any(right <= left for left, right in zip(parsed, parsed[1:])):
                problems.append("p3_public_absolute_timeline_order_invalid")
    roles = ledger.get("roles") or {}
    _verify_cost_contract(admission, ledger, roles, public, problems)
    expected = {
        "seed": stages.get("seed"),
        "capture": stages.get("capture"),
        "organizer": stages.get("organizer"),
        "review": stages.get("review"),
        "downstream": public.get("downstream_task_sha256"),
    }
    for role, sha256 in expected.items():
        row = roles.get(role) or {}
        if sha256 and row.get("receipt_sha256") != sha256:
            problems.append(f"p3_public_cost_role_mismatch:{role}")
    if ledger.get("prerequisite_unknown_usage"):
        problems.append("p3_public_prerequisite_usage_unknown")
    if receipts["consumer"].get("corpus_sha256") != public.get("frozen_corpus_tree_sha256"):
        problems.append("p3_public_consumer_corpus_mismatch")
    return problems


def load_public_p3_claim(
    suite: dict,
    evidence: dict,
    releases: dict,
    cohorts: dict,
    benchmark: dict,
    preparation: dict,
) -> dict:
    """CI verify: recompute public consistency; never reopen private originals."""
    raw = suite.get("p3")
    if not raw:
        return {"status": "missing", "required": False, "reasons": ["p3_chain_not_declared"]}
    required = bool(raw.get("required"))
    scope = raw.get("supplementary_scope")
    public = evidence.get("p3_public_lineage")
    if not isinstance(public, dict):
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": ["p3_public_lineage_missing"],
        }
    declared_identity = raw.get("manifest_content_identity_sha256")
    comparison_cohort = raw.get("comparison_cohort")
    try:
        source = _source_identity(
            raw["profile"],
            ProductGroup.P2_V003_FULL,
            releases,
            cohort_id=raw["profile"]["cohort"],
            preparation=preparation,
        )
        if source["benchmark"] != benchmark:
            return {
                "status": "failed",
                "required": required,
                "scope": scope,
                "reasons": ["p3_source_benchmark_differs_from_common_runner"],
            }
        profile = P3PublicationProfile(
            profile_id=raw["profile"]["profile_id"],
            group=ProductGroup.P3_V003_FLYWHEEL,
            protocol=raw["profile"]["protocol"],
            fidelity=raw["profile"].get("fidelity", "matched"),
            cohort=raw["profile"]["cohort"],
            conditions=MatchingConditions(**raw["profile"]["conditions"]),
            budget=BudgetEnvelope(**raw["profile"]["budget"]),
        )
        task = raw["task"]
        cell = ProductCell(
            ProductGroup.P3_V003_FLYWHEEL,
            profile.profile_id,
            profile.cohort,
            profile.protocol,
            raw["profile"].get("task_id") or task["task_id"],
            task["task_family"],
            task["hardware_tier"],
            raw.get("repetition", 1),
            raw.get("order", 0),
        )
        target = {
            "profile_id": profile.profile_id,
            "product_group": "P3_v003_flywheel",
            "cohort": profile.cohort,
            "protocol": profile.protocol,
            "task_id": cell.task_id,
            "matrix_task_family": task.get("task_family"),
            "hardware_tier": task.get("hardware_tier"),
        }
        if comparison_cohort not in cohorts:
            return {
                "status": "failed",
                "required": required,
                "scope": scope,
                "reasons": ["p3_comparison_cohort_unknown"],
            }
        cohort_p2 = cohorts[comparison_cohort]["profiles"][ProductGroup.P2_V003_FULL]
        if (
            profile.conditions.to_dict() != cohort_p2.conditions.to_dict()
            or profile.budget.comparison_dict() != cohort_p2.budget.comparison_dict()
        ):
            return {
                "status": "failed",
                "required": required,
                "scope": scope,
                "reasons": ["p3_matching_conditions_or_budget_differ_from_p2"],
            }
        problems = _verify_public_lineage(public, declared_identity, target)
        if public.get("source_sha256") != digest(source):
            problems.append("p3_public_source_identity_mismatch")
        if public.get("comparison_cohort") != comparison_cohort:
            problems.append("p3_public_comparison_cohort_mismatch")
        if problems:
            return {
                "status": "failed",
                "required": required,
                "scope": scope,
                "reasons": problems[:20],
                "public_lineage": public,
            }
        return _public_claim(
            raw, public, profile, source, cell, comparison_cohort=comparison_cohort
        )
    except TimeoutError:
        raise
    except (OSError, ValueError, KeyError) as error:
        return {
            "status": "failed",
            "required": required,
            "scope": scope,
            "reasons": [f"p3_public_invalid:{error}"],
        }
