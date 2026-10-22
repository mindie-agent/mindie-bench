"""P3 lineage admission before native product execution.

[Role] Admit one P3 Seed -> capture ingestion -> in-capture C6 organization ->
publication/consumer -> freeze -> independent review -> new different-family
Heldout lineage, and expose the single entrance used by matrix planning and
ProductExecutor.

[Invariants]
- Closed-family identity is derived from exact task/oracle/instruction bytes and
  closed asset content hashes. Labels, paths and repackaging cannot create an
  independent family.
- The C6 organizer runs inside capture (`graded_distillation`). It is a stage of
  the capture receipt, never a separate post-freeze artifact. Capture ingestion
  and the nested organizer have distinct stage ids and their counts must add up
  to the capture container total.
- The existing lineage-closure audit and split manifest remain the authority for
  problem closure, split assignment and body exposure. No shadow closure state.
- Seed run, capture container, in-capture organizer, publication consumer,
  frozen corpus, independent review and downstream consumer are hash-chained in
  that order.
- Ledger model calls, input/cached/output tokens, NPU runs, completeness and
  price basis are compared with the actual receipts. Missing fields stay
  unknown; missing absolute timestamps are reported as limitations, never
  inferred from relative seconds.

[Timeouts & Locks] Pure metadata validation plus bounded local receipt/corpus
reads; no waits, subprocesses or locks.

[Side-Effect Policy] Read-only. Never starts a model, process, NPU or write.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.evaluation.split_manifest import validate_g3_timeline
from mindie_bench.io import digest, task_hash, tree_hash
from mindie_bench.mining.lineage_closure import audit_problem_closure
from mindie_bench.models import TaskSpec
from mindie_bench.runner.p3_lineage_models import *  # noqa: F401,F403
from mindie_bench.runner.p3_lineage_models import (
    _PRODUCT_GROUPS,
    P3_ADMISSION_PROTOCOL,
    SHA,
    P3LineageManifest,
    P3LineageRejected,
    P3SeedRun,
)
from mindie_bench.runner.p3_receipts import (
    CostObservation as CostObservation,
)
from mindie_bench.runner.p3_receipts import (
    budget_ledger_mismatches as budget_ledger_mismatches,
)
from mindie_bench.runner.p3_receipts import (
    capture_attribution_mismatches,
    capture_container,
    capture_observations,
    capture_stage_observations,
    consumer_observation,
    ledger_numeric_mismatches,
    review_observation,
    seed_observation,
    stage_order_reasons,
    timeline_missing,
)

_MAX_RECEIPT_BYTES = 4 * 1024 * 1024

def _hash_file(path: Path, *, max_bytes: int | None = None) -> str:
    hasher = hashlib.sha256()
    read = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            read += len(chunk)
            if max_bytes is not None and read > max_bytes:
                raise ValueError("receipt exceeds its bound")
            hasher.update(chunk)
    return hasher.hexdigest()

def _receipt(locator: str, expected_sha256: str, role: str, reasons: list[str]) -> dict | None:
    path = Path(locator)
    try:
        if path.stat().st_size > _MAX_RECEIPT_BYTES:
            reasons.append(f"{role}_receipt_too_large")
            return None
        raw = path.read_bytes()
    except TimeoutError:
        raise
    except OSError:
        reasons.append(f"{role}_receipt_unreadable")
        return None
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        reasons.append(f"{role}_receipt_sha256_mismatch")
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        reasons.append(f"{role}_receipt_not_json")
        return None
    if not isinstance(payload, dict):
        reasons.append(f"{role}_receipt_not_object")
        return None
    return payload

def _canonical(
    family: dict,
    assets_by_id: dict[str, dict],
    task_sha256: str,
    oracle_sha256: str,
    instruction_sha256: str,
    label: str,
    reasons: list[str],
) -> dict:
    content = []
    for asset_id in family.get("covered_asset_ids") or []:
        value = str((assets_by_id.get(asset_id) or {}).get("content_sha256") or "")
        if not re.fullmatch(SHA, value):
            reasons.append(f"canonical_family_requires_content_sha256:{label}:{asset_id}")
        else:
            content.append(value)
    payload = {
        "task_sha256": task_sha256,
        "oracle_sha256": oracle_sha256,
        "instruction_sha256": instruction_sha256,
        "content_sha256": sorted(set(content)),
    }
    return {
        "family_id": family.get("family_id"),
        "split": family.get("split"),
        "canonical_id": digest(payload),
        "task_sha256": task_sha256,
        "oracle_sha256": oracle_sha256,
        "instruction_sha256": instruction_sha256,
        "content_sha256": sorted(set(content)),
        "covered_asset_ids": sorted(family.get("covered_asset_ids") or []),
        "problem_identity": family.get("problem_identity"),
    }

def _check_seed_root(seed: P3SeedRun, reasons: list[str], evidence: list[dict]) -> None:
    if not seed.seed_root_locator:
        reasons.append("seed_root_locator_missing")
        return
    root = Path(seed.seed_root_locator)
    if not root.is_dir():
        reasons.append("seed_root_unreadable")
        return
    try:
        task = TaskSpec.load(root)
    except TimeoutError:
        raise
    except (OSError, ValueError) as error:
        reasons.append(f"seed_root_task_invalid:{type(error).__name__}")
        return
    if task.task.instance_id != seed.task_id:
        reasons.append("seed_root_task_id_mismatch")
    try:
        actual_task = task_hash(root)
    except TimeoutError:
        raise
    except OSError:
        actual_task = ""
    if actual_task != seed.task_sha256:
        reasons.append("seed_task_bytes_changed")
    try:
        actual_instruction = _hash_file(root / "instruction.md")
    except TimeoutError:
        raise
    except OSError:
        actual_instruction = ""
    if actual_instruction != seed.instruction_sha256:
        reasons.append("seed_instruction_bytes_changed")
    try:
        actual_oracle = freeze_oracle(root, task.oracle).sha256
    except TimeoutError:
        raise
    except (OSError, ValueError):
        actual_oracle = ""
    if actual_oracle != seed.oracle_sha256:
        reasons.append("seed_oracle_bytes_changed")
    evidence.append(
        {
            "role": "seed_root",
            "locator": str(root),
            "sha256": actual_task,
            "verified": actual_task == seed.task_sha256
            and actual_instruction == seed.instruction_sha256
            and actual_oracle == seed.oracle_sha256,
            "identity_basis": seed.seed_identity_basis,
        }
    )


def _check_structural_chain(
    manifest: P3LineageManifest,
    seed: dict,
    heldout: dict,
    reasons: list[str],
    limitations: list[str],
) -> list[dict]:
    reasons.extend(stage_order_reasons(manifest.capture.stage_observations))
    if manifest.capture.status != "completed":
        reasons.append(f"capture_status:{manifest.capture.status}")
    if manifest.capture.corpus_lint_status != "passed":
        reasons.append(f"corpus_lint:{manifest.capture.corpus_lint_status}")
    if manifest.organizer.status != "applied":
        reasons.append(f"organizer_status:{manifest.organizer.status}")
    if manifest.organizer.cleanup_failed:
        reasons.append("organizer_cleanup_failed")
    if manifest.consumer.status != "completed":
        reasons.append(f"consumer_status:{manifest.consumer.status}")
    if manifest.capture.seed_trial_id != manifest.seed.trial_id:
        reasons.append("capture_seed_trial_mismatch")
    if manifest.capture.seed_receipt_sha256 != manifest.seed.receipt_sha256:
        reasons.append("capture_seed_receipt_mismatch")
    if manifest.organizer.capture_receipt_sha256 != manifest.capture.receipt_sha256:
        reasons.append("organizer_capture_receipt_mismatch")
    if manifest.consumer.capture_receipt_sha256 != manifest.capture.receipt_sha256:
        reasons.append("consumer_capture_receipt_mismatch")
    if manifest.consumer.trial_receipt_sha256 != manifest.seed.receipt_sha256:
        reasons.append("consumer_trial_receipt_mismatch")
    if manifest.frozen_corpus.consumer_receipt_sha256 != manifest.consumer.receipt_sha256:
        reasons.append("frozen_corpus_consumer_mismatch")
    if manifest.review.consumer_receipt_sha256 != manifest.consumer.receipt_sha256:
        reasons.append("review_consumer_receipt_mismatch")
    if manifest.review.capture_receipt_sha256 != manifest.capture.receipt_sha256:
        reasons.append("review_capture_receipt_mismatch")
    if manifest.downstream.review_receipt_sha256 != manifest.review.receipt_sha256:
        reasons.append("downstream_review_receipt_mismatch")
    if not manifest.review.independent:
        reasons.append("review_is_not_independent")
    if manifest.review.actor_id == manifest.organizer.actor_id:
        reasons.append("reviewer_is_organizer")
    if seed and manifest.review.seed_family_canonical != seed["canonical_id"]:
        reasons.append("review_seed_family_mismatch")
    if heldout and manifest.review.heldout_family_canonical != heldout["canonical_id"]:
        reasons.append("review_heldout_family_mismatch")
    if manifest.review.seed_family_canonical == manifest.review.heldout_family_canonical:
        reasons.append("review_claims_one_family")
    if manifest.downstream.task_id != manifest.target.task_id:
        reasons.append("downstream_task_differs_from_target")
    used_contexts = {
        manifest.seed.context_id,
        manifest.capture.capture_id,
        manifest.consumer.consumer_id,
        manifest.review.context_id,
    }
    if manifest.downstream.context_id in used_contexts:
        reasons.append("downstream_context_is_not_new")
    for exposure in manifest.exposures:
        if exposure.get("context_id") == manifest.downstream.context_id:
            reasons.append("downstream_context_previously_exposed")
            break
    if manifest.capture.tokens_total is None:
        limitations.append("capture_tokens_total_unknown")
    if not manifest.capture.tokens_complete:
        limitations.append("capture_tokens_not_complete")
    return [
        {"stage_id": "seed_trial", "receipt_sha256": manifest.seed.receipt_sha256, "role": "seed"},
        {"stage_id": "capture_ingestion", "receipt_sha256": manifest.capture.receipt_sha256, "role": "capture"},
        {"stage_id": "graded_distillation", "receipt_sha256": manifest.capture.receipt_sha256, "role": "organizer"},
        {"stage_id": "consumer_continuation", "receipt_sha256": manifest.consumer.receipt_sha256, "role": "consumer"},
        {"stage_id": "frozen_corpus", "receipt_sha256": manifest.consumer.receipt_sha256, "role": "frozen_corpus"},
        {"stage_id": "independent_review", "receipt_sha256": manifest.review.receipt_sha256, "role": "review"},
    ]

def _check_receipts(
    manifest: P3LineageManifest,
    reasons: list[str],
    limitations: list[str],
    evidence: list[dict],
) -> dict:
    seed_receipt = _receipt(manifest.seed.receipt_locator, manifest.seed.receipt_sha256, "seed", reasons)
    capture_receipt = _receipt(
        manifest.capture.receipt_locator, manifest.capture.receipt_sha256, "capture", reasons
    )
    consumer_receipt = _receipt(
        manifest.consumer.receipt_locator, manifest.consumer.receipt_sha256, "consumer", reasons
    )
    review_receipt = _receipt(
        manifest.review.receipt_locator, manifest.review.receipt_sha256, "review", reasons
    )
    for role, locator, sha256 in (
        ("seed", manifest.seed.receipt_locator, manifest.seed.receipt_sha256),
        ("capture", manifest.capture.receipt_locator, manifest.capture.receipt_sha256),
        ("consumer", manifest.consumer.receipt_locator, manifest.consumer.receipt_sha256),
        ("review", manifest.review.receipt_locator, manifest.review.receipt_sha256),
    ):
        evidence.append({"role": role, "locator": locator, "sha256": sha256, "verified": True})
    expected = {}
    consumer_usage = None
    if seed_receipt is not None:
        for key, expected_value in (
            ("trial_id", manifest.seed.trial_id),
            ("instance_id", manifest.seed.task_id),
            ("terminal_status", manifest.seed.terminal_status),
            ("task_digest", manifest.seed.task_sha256),
        ):
            if seed_receipt.get(key) != expected_value:
                reasons.append(f"seed_receipt_field_mismatch:{key}")
        seed_usage = seed_observation(seed_receipt, manifest.seed.receipt_sha256)
        expected["seed"] = seed_usage.model_dump(mode="json")
        if not seed_usage.complete:
            limitations.append("seed_usage_incomplete_or_unpriced")
        if seed_usage.price_basis != "priced":
            limitations.append("seed_price_unknown")
    if capture_receipt is not None and seed_receipt is not None:
        receipt_stages = capture_stage_observations(capture_receipt)
        reasons.extend(stage_order_reasons(receipt_stages))
        declared_stages = {stage.stage_id: stage for stage in manifest.capture.stage_observations}
        observed_stages = {stage.stage_id: stage for stage in receipt_stages}
        for stage_id in sorted(declared_stages.keys() | observed_stages.keys()):
            declared = declared_stages.get(stage_id)
            observed = observed_stages.get(stage_id)
            if declared is None or observed is None:
                reasons.append(f"capture_receipt_stage_set_mismatch:{stage_id}")
                continue
            for field in (
                "status",
                "started_sec",
                "ended_sec",
                "elapsed_sec",
                "started_at",
                "ended_at",
            ):
                if getattr(declared, field) != getattr(observed, field):
                    reasons.append(f"capture_receipt_stage_mismatch:{stage_id}:{field}")
        raw_started = capture_receipt.get("capture_started_at")
        if raw_started is not None:
            try:
                parsed_started = datetime.fromisoformat(str(raw_started))
            except ValueError:
                reasons.append("capture_receipt_started_at_invalid")
            else:
                if manifest.capture.started_at != parsed_started:
                    reasons.append("capture_receipt_started_at_mismatch")
        if capture_receipt.get("status") != manifest.capture.status:
            reasons.append("capture_receipt_status_mismatch")
        if capture_receipt.get("task_state") != "completed":
            reasons.append(f"capture_task_state:{capture_receipt.get('task_state')}")
        if "native_harness" in capture_receipt and capture_receipt["native_harness"] != manifest.capture.native_harness:
            reasons.append("capture_receipt_harness_mismatch")
        if _mapping(capture_receipt.get("corpus_lint")).get("status") != manifest.capture.corpus_lint_status:
            reasons.append("capture_receipt_corpus_lint_mismatch")
        if _mapping(capture_receipt.get("tokens")).get("total") != manifest.capture.tokens_total:
            reasons.append("capture_receipt_tokens_total_mismatch")
        if _mapping(capture_receipt.get("tokens")).get("complete") is not manifest.capture.tokens_complete:
            reasons.append("capture_receipt_tokens_complete_mismatch")
        distill = _mapping(capture_receipt.get("distill"))
        indexer = _mapping(capture_receipt.get("indexer_usage"))
        for label, actual, expected_value in (
            ("distill_status", distill.get("status"), manifest.organizer.status),
            ("distill_model_calls", _int_or_none(distill.get("model_calls")), manifest.organizer.model_calls),
            ("distill_billing_status", distill.get("billing_status"), manifest.organizer.billing_status),
            ("distill_cleanup_failed", distill.get("cleanup_failed"), manifest.organizer.cleanup_failed),
            ("indexer_model_calls", _int_or_none(indexer.get("model_calls")), manifest.organizer.model_calls),
            ("indexer_input_tokens", _int_or_none(indexer.get("input_tokens")), manifest.organizer.input_tokens),
            ("indexer_cached_input_tokens", _int_or_none(indexer.get("cached_input_tokens")), manifest.organizer.cached_input_tokens),
            ("indexer_output_tokens", _int_or_none(indexer.get("output_tokens")), manifest.organizer.output_tokens),
            ("distill_usage_input_tokens", _int_or_none(_mapping(distill.get("usage")).get("input_tokens")), manifest.organizer.input_tokens),
            ("distill_usage_cached_input_tokens", _int_or_none(_mapping(distill.get("usage")).get("cached_input_tokens")), manifest.organizer.cached_input_tokens),
            ("distill_usage_output_tokens", _int_or_none(_mapping(distill.get("usage")).get("output_tokens")), manifest.organizer.output_tokens),
        ):
            if actual != expected_value:
                reasons.append(f"capture_receipt_field_mismatch:{label}")
        capture_usage, organizer_usage = capture_observations(
            capture_receipt, seed_receipt, manifest.capture.receipt_sha256
        )
        expected["capture"] = capture_usage.model_dump(mode="json")
        expected["organizer"] = organizer_usage.model_dump(mode="json")
        container = capture_container(capture_receipt, seed_receipt)
        container["capture_receipt_sha256"] = manifest.capture.receipt_sha256
        attribution = capture_attribution_mismatches(manifest.cost_ledger, container)
        reasons.extend(attribution["duplicate_or_lost"])
        limitations.extend(attribution["unknown"])
        if not organizer_usage.complete:
            limitations.append("organizer_usage_incomplete_or_unpriced")
        if organizer_usage.price_basis != "priced":
            limitations.append("organizer_price_unknown")
    if consumer_receipt is not None:
        if consumer_receipt.get("status") != manifest.consumer.status:
            reasons.append("consumer_receipt_status_mismatch")
        if consumer_receipt.get("model_calls") != manifest.consumer.model_calls:
            reasons.append("consumer_receipt_model_calls_mismatch")
        if consumer_receipt.get("provider_writes") != manifest.consumer.provider_writes:
            reasons.append("consumer_receipt_provider_writes_mismatch")
        if consumer_receipt.get("remote_writes") != manifest.consumer.remote_writes:
            reasons.append("consumer_receipt_remote_writes_mismatch")
        if consumer_receipt.get("corpus_sha256") != manifest.consumer.corpus_sha256:
            reasons.append("consumer_receipt_corpus_mismatch")
        if consumer_receipt.get("scope") != manifest.consumer.scope:
            reasons.append("consumer_receipt_scope_mismatch")
        if consumer_receipt.get("original_receipts_unchanged") is not True:
            reasons.append("consumer_receipt_not_immutable")
        raw_capture = _mapping(consumer_receipt.get("raw_capture_receipt")).get("sha256")
        raw_trial = _mapping(consumer_receipt.get("raw_trial_receipt")).get("sha256")
        if raw_capture != manifest.capture.receipt_sha256:
            reasons.append("consumer_raw_capture_hash_mismatch")
        if raw_trial != manifest.seed.receipt_sha256:
            reasons.append("consumer_raw_trial_hash_mismatch")
        consumer_usage = consumer_observation(consumer_receipt, manifest.consumer.receipt_sha256)
        if consumer_usage is not None and not consumer_usage.complete:
            limitations.append("consumer_usage_incomplete")
    if review_receipt is not None:
        for label, actual, expected_value in (
            ("status", review_receipt.get("status"), manifest.review.status),
            ("verdict", review_receipt.get("verdict"), manifest.review.verdict),
            ("independent", review_receipt.get("independent"), True),
        ):
            if actual != expected_value:
                reasons.append(f"review_receipt_field_mismatch:{label}")
        if review_receipt.get("organizer_receipt_sha256") != manifest.review.capture_receipt_sha256:
            reasons.append("review_receipt_field_mismatch:organizer_receipt_sha256")
        for key in ("seed_family_canonical", "heldout_family_canonical"):
            if review_receipt.get(key) != getattr(manifest.review, key):
                reasons.append(f"review_receipt_field_mismatch:{key}")
        review_usage = review_observation(review_receipt, manifest.review.receipt_sha256)
        expected["review"] = review_usage.model_dump(mode="json")
        if not review_usage.complete:
            limitations.append("review_usage_incomplete_or_unpriced")
        if review_usage.price_basis != "priced":
            limitations.append("review_price_unknown")
    if len(expected) == 4:
        reasons.extend(ledger_numeric_mismatches(manifest.cost_ledger, expected))
    downstream = manifest.cost_ledger.downstream
    if downstream.stage_id != "downstream_declared":
        reasons.append("downstream_stage_id_not_declared")
    if downstream.receipt_sha256 != manifest.downstream.task_sha256:
        reasons.append("downstream_ledger_not_bound_to_task")
    if downstream.basis != "declared":
        reasons.append("downstream_ledger_not_declared")
    if downstream.complete:
        reasons.append("downstream_usage_must_be_declared_not_complete")
    return {"expected": expected, "consumer_usage": consumer_usage}

def _int_or_none(value) -> int | None:
    return value if type(value) is int and value >= 0 else None

def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}

def evaluate_p3_lineage(
    manifest: P3LineageManifest,
    *,
    task_root: Path | None = None,
    verify_artifacts: bool = True,
    allow_incomplete_usage: bool = False,
) -> dict:
    """Return an explicit allow/deny decision without starting any native work."""
    reasons: list[str] = []
    limitations: list[str] = []
    if manifest.target.product_group not in _PRODUCT_GROUPS:
        reasons.append("p3_target_group_not_p3")
    if not manifest.shared_mechanism.strip():
        reasons.append("shared_mechanism_missing")
    assets_by_id = {}
    for asset in manifest.assets:
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id or asset_id in assets_by_id:
            reasons.append("invalid_or_duplicate_asset_id")
            continue
        assets_by_id[asset_id] = asset
    for asset in manifest.assets:
        for name in ("source_id", "source_locator"):
            if not str(asset.get(name) or "").strip():
                reasons.append(f"asset_missing_{name}")
                break
    report = audit_problem_closure(
        assets=manifest.assets,
        strong_edges=manifest.strong_edges,
        candidate_edges=manifest.candidate_edges,
        claims=manifest.claims,
        decisions=manifest.decisions,
        exposures=manifest.exposures,
    )
    if report.get("status") != "ready_for_seed_heldout" or not report.get("ready"):
        reasons.append(f"closure_status:{report.get('status')}")
        reasons.extend(f"closure:{item}" for item in report.get("blocking_reasons", [])[:12])
    families = {family["family_id"]: family for family in report.get("families", [])}
    seed_family = families.get(manifest.seed.family_id)
    heldout_family = families.get(manifest.downstream.family_id)
    if seed_family is None:
        reasons.append("seed_family_not_selected")
    elif seed_family.get("split") != "seed" or not seed_family.get("accepted"):
        reasons.append("seed_family_not_accepted_seed")
    if heldout_family is None:
        reasons.append("heldout_family_not_selected")
    elif heldout_family.get("split") != "heldout" or not heldout_family.get("accepted"):
        reasons.append("heldout_family_not_accepted_heldout")
    seed = (
        _canonical(seed_family, assets_by_id, manifest.seed.task_sha256, manifest.seed.oracle_sha256, manifest.seed.instruction_sha256, "seed", reasons)
        if seed_family is not None
        else None
    )
    heldout = (
        _canonical(heldout_family, assets_by_id, manifest.downstream.task_sha256, manifest.downstream.oracle_sha256, manifest.downstream.instruction_sha256, "heldout", reasons)
        if heldout_family is not None
        else None
    )
    if seed and heldout:
        if seed["canonical_id"] == heldout["canonical_id"]:
            reasons.append("canonical_family_collision")
        if set(seed["content_sha256"]) & set(heldout["content_sha256"]):
            reasons.append("shared_content_sha256_across_splits")
        if seed["task_sha256"] == heldout["task_sha256"]:
            reasons.append("shared_task_sha256_across_splits")
        if seed["oracle_sha256"] == heldout["oracle_sha256"]:
            reasons.append("shared_oracle_sha256_across_splits")
    split_assets = {}
    try:
        manifest.split_manifest.boundary_integrity()
    except ValueError as error:
        reasons.append(f"split_manifest_invalid:{error}")
    for asset in manifest.split_manifest.assets:
        split_assets[asset.asset_id] = asset
    for label, family in (("seed", seed_family), ("heldout", heldout_family)):
        if family is None:
            continue
        for asset_id in family.get("covered_asset_ids") or []:
            split_asset = split_assets.get(asset_id)
            if split_asset is None:
                reasons.append(f"split_manifest_missing_asset:{label}:{asset_id}")
                continue
            if split_asset.split != family.get("split"):
                reasons.append(f"split_manifest_split_mismatch:{label}:{asset_id}")
            declared = (assets_by_id.get(asset_id) or {}).get("content_sha256")
            if split_asset.content_sha256 is None:
                reasons.append(f"split_manifest_asset_missing_content_sha256:{label}:{asset_id}")
            elif split_asset.content_sha256 != declared:
                reasons.append(f"split_manifest_content_mismatch:{label}:{asset_id}")
            for field in ("source_id", "source_locator"):
                if str((assets_by_id.get(asset_id) or {}).get(field) or "") != getattr(split_asset, field):
                    reasons.append(f"split_manifest_{field}_mismatch:{label}:{asset_id}")
    for field in ("content_sha256", "source_id", "source_locator"):
        index: dict[str, set[str]] = {}
        for asset in manifest.assets:
            value = asset.get(field)
            if not value:
                continue
            for claim in manifest.claims:
                if asset.get("asset_id") in (claim.get("asset_ids") or []):
                    index.setdefault(str(value), set()).add(str(claim.get("split")))
        for value, splits in index.items():
            if len(splits) > 1:
                reasons.append(f"cross_split_{field}:{value}")
    for field in ("content_sha256", "source_id"):
        index: dict[str, set[str]] = {}
        for asset in manifest.split_manifest.assets:
            value = getattr(asset, field)
            if value:
                index.setdefault(value, set()).add(asset.split)
        for value, splits in index.items():
            if len(splits) > 1:
                reasons.append(f"split_manifest_cross_split_{field}:{value}")
    if task_root is not None:
        try:
            actual_task = task_hash(Path(task_root))
        except TimeoutError:
            raise
        except OSError:
            actual_task = ""
        if actual_task != manifest.downstream.task_sha256:
            reasons.append("downstream_task_bytes_changed")
        try:
            actual_instruction = _hash_file(Path(task_root) / "instruction.md")
        except TimeoutError:
            raise
        except OSError:
            actual_instruction = ""
        if actual_instruction != manifest.downstream.instruction_sha256:
            reasons.append("downstream_instruction_bytes_changed")
    evidence = _check_structural_chain(manifest, seed or {}, heldout or {}, reasons, limitations)
    if seed is not None:
        seed["identity_basis"] = manifest.seed.seed_identity_basis
    if manifest.seed.seed_identity_basis == "verified_bytes":
        if not manifest.seed.seed_root_locator:
            reasons.append("seed_root_locator_missing")
        elif verify_artifacts:
            _check_seed_root(manifest.seed, reasons, evidence)
    else:
        limitations.append("seed_identity_receipt_recorded_not_byte_verified")
    receipt_check = {}
    if verify_artifacts:
        receipt_check = _check_receipts(manifest, reasons, limitations, evidence)
        try:
            observed_tree = tree_hash(Path(manifest.frozen_corpus.locator))
        except TimeoutError:
            raise
        except OSError:
            observed_tree = ""
        if observed_tree != manifest.frozen_corpus.tree_sha256:
            reasons.append("frozen_corpus_bytes_changed")
        evidence.append({"role": "frozen_corpus", "locator": manifest.frozen_corpus.locator, "sha256": manifest.frozen_corpus.tree_sha256, "verified": observed_tree == manifest.frozen_corpus.tree_sha256})
    else:
        evidence.append({"role": "frozen_corpus", "locator": manifest.frozen_corpus.locator, "sha256": manifest.frozen_corpus.tree_sha256, "verified": False})
    absolute_missing = timeline_missing(
        seed_completed_at=manifest.seed.completed_at,
        capture_started_at=manifest.capture.started_at,
        organizer_completed_at=manifest.organizer.completed_at,
        consumer_completed_at=manifest.consumer.completed_at,
        corpus_frozen_at=manifest.frozen_corpus.frozen_at,
        review_completed_at=manifest.review.completed_at,
        downstream_started_at=manifest.downstream.started_at,
    )
    timeline = {
        "stage_observations": [stage.model_dump(mode="json") for stage in manifest.capture.stage_observations],
        "absolute_available": not absolute_missing,
        "missing_absolute": absolute_missing,
        "inference_performed": False,
    }
    limitations.extend(f"absolute_timeline_missing:{name}" for name in absolute_missing)
    if not absolute_missing:
        times = (
            manifest.seed.completed_at,
            manifest.capture.started_at,
            manifest.organizer.completed_at,
            manifest.consumer.completed_at,
            manifest.frozen_corpus.frozen_at,
            manifest.review.completed_at,
            manifest.downstream.started_at,
        )
        if any(right <= left for left, right in zip(times, times[1:])):
            reasons.append("absolute_timeline_order_invalid")
        try:
            timeline.update(
                validate_g3_timeline(
                    manifest.split_manifest,
                    seed_asset_ids=list(seed_family.get("covered_asset_ids") or []) if seed_family else [],
                    downstream_asset_ids=(list(heldout_family.get("covered_asset_ids") or []) if heldout_family else []),
                    seed_completed_at=manifest.seed.completed_at,
                    capture_completed_at=manifest.consumer.completed_at,
                    corpus_frozen_at=manifest.frozen_corpus.frozen_at,
                    downstream_started_at=manifest.downstream.started_at,
                )["timeline"]
            )
        except ValueError as error:
            reasons.append(f"timeline:{error}")
    ledger = manifest.cost_ledger
    if not allow_incomplete_usage:
        for role in ledger.prerequisite_unknown_usage:
            reasons.append(f"prerequisite_usage_incomplete:{role}")
    if ledger.prerequisite_unpriced_usage:
        limitations.extend(
            f"price_basis_unknown:{role}" for role in ledger.prerequisite_unpriced_usage
        )
    if receipt_check.get("consumer_usage") is not None and not receipt_check["consumer_usage"].complete:
        limitations.append("consumer_usage_incomplete")
    status = "denied" if reasons else "admitted"
    prerequisite_cost_complete = not ledger.prerequisite_unknown_usage
    lifecycle_cost_complete = (
        prerequisite_cost_complete and ledger.downstream.complete and not ledger.unknown_usage
    )
    prerequisite_claim_ready = status == "admitted" and prerequisite_cost_complete and not limitations
    lifecycle_claim_ready = status == "admitted" and lifecycle_cost_complete and not limitations
    payload = {
        "schema": P3_ADMISSION_PROTOCOL,
        "status": status,
        "reasons": reasons,
        "manifest_sha256": manifest.digest(),
        "content_identity_sha256": manifest.content_identity(),
        "split_manifest_sha256": manifest.split_manifest.manifest_sha256,
        "target": manifest.target.model_dump(mode="json"),
        "label_binding": {
            "matrix_task_family": manifest.target.matrix_task_family,
            "matrix_hardware_tier": manifest.target.hardware_tier,
            "lineage_family_id": manifest.downstream.family_id,
            "source_problem_identity": (heldout or {}).get("problem_identity"),
            "matrix_task_family_equals_lineage_family": (
                manifest.target.matrix_task_family == manifest.downstream.family_id
                if manifest.target.matrix_task_family is not None
                else None
            ),
            "note": (
                "matrix task_family is a reporting label; the lineage family and source "
                "problem identity are exposed separately and are not equated to it"
            ),
        },
        "shared_mechanism": manifest.shared_mechanism,
        "closure": {
            "status": report.get("status"),
            "ready": report.get("ready"),
            "accepted_claims": report.get("accepted_claims", []),
            "family_statuses": {family["family_id"]: family["status"] for family in report.get("families", [])},
        },
        "seed": seed,
        "heldout": heldout,
        "timeline": timeline,
        "stages": evidence,
        "cost_ledger": ledger.summary(),
        "prerequisite_cost_complete": prerequisite_cost_complete,
        "lifecycle_cost_complete": lifecycle_cost_complete,
        "unknown_usage": list(ledger.unknown_usage),
        "unpriced_usage": list(ledger.unpriced_usage),
        "limitations": sorted(set(limitations)),
        "claim_scope": (
            "claim_ready is prerequisite admission only; whole-lifecycle completion requires "
            "observed downstream usage and is reported separately"
        ),
        "claim_ready": prerequisite_claim_ready,
        "prerequisite_claim_ready": prerequisite_claim_ready,
        "lifecycle_claim_ready": lifecycle_claim_ready,
        "verification": {
            "artifact_bytes_verified": verify_artifacts,
            "artifact_count": 4 if verify_artifacts else 0,
        },
    }
    admission_sha = digest(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "verification",
                "stages",
                "limitations",
                "manifest_sha256",
                "split_manifest_sha256",
                "prerequisite_claim_ready",
                "lifecycle_claim_ready",
            }
        }
    )
    payload["admission_sha256"] = admission_sha
    return payload

def admit_p3_lineage(manifest: P3LineageManifest, *, verify_artifacts: bool = True) -> dict:
    """Fail closed unless the whole lineage is admissible."""
    result = evaluate_p3_lineage(manifest, verify_artifacts=verify_artifacts)
    if result["status"] != "admitted":
        raise P3LineageRejected(result["reasons"])
    return result
