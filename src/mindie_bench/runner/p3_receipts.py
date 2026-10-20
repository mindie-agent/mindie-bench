"""Exact native capture/seed usage attribution for P3 admission.

[Role] Parse the receipt shapes emitted by the existing native capture path and
compare declared cost-ledger rows with observed model/token/NPU/price fields.

[Invariants]
- The C6 organizer runs inside capture (`graded_distillation`); it is not a
  separate post-freeze artifact. Capture ingestion and the nested organizer may
  share the capture receipt bytes, but each has a distinct stage id and their
  attributed counts must add up to the capture container total.
- Missing `cost_complete`, token completeness, model calls, NPU runs or price
  schedule remain unknown; they are never imputed as zero or complete.
- Numeric ledger fields must match the actual receipt fields. Evidence-ref-only
  matching is insufficient.
- Legacy relative stage seconds are retained as observed; absolute timestamps
  are reported missing, never reconstructed.

[Timeouts & Locks] Pure parsing/validation; the caller reads bounded receipts.

[Side-Effect Policy] No I/O, model request, NPU execution or mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from mindie_bench.models import StrictModel

SHA = r"^[0-9a-f]{64}$"
ROLES = ("seed", "capture", "organizer", "review", "downstream")
PREREQUISITE_ROLES = ("seed", "capture", "organizer", "review")

def _int_or_none(value) -> int | None:
    return value if type(value) is int and value >= 0 else None

def _float_or_none(value) -> float | None:
    if type(value) in {int, float} and math.isfinite(value) and value >= 0:
        return float(value)
    return None

def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


class P3TimeModel(StrictModel):
    @model_validator(mode="after")
    def timezone_aware_times(self):
        for name, value in self.__dict__.items():
            if isinstance(value, datetime) and value.tzinfo is None:
                raise ValueError(f"{name} must include a timezone")
        return self


class P3StageObservation(P3TimeModel):
    """One real stage record; relative seconds and optional absolute times."""
    stage_id: str = Field(min_length=1)
    status: Literal["started", "completed", "failed"]
    started_sec: float | None = None
    ended_sec: float | None = None
    elapsed_sec: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: dict | None = None
    @model_validator(mode="after")
    def finite_seconds(self):
        for name in ("started_sec", "ended_sec", "elapsed_sec"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative or None")
        if self.started_sec is not None and self.ended_sec is not None:
            if self.ended_sec < self.started_sec:
                raise ValueError("stage end must not precede stage start")
        return self


class CostObservation(StrictModel):
    """One attributed stage; every numeric value is either observed or None."""
    basis: Literal["observed", "declared"]
    stage_id: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA)
    model_calls: int | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_tokens: int | None = None
    output_tokens: int | None = None
    tokens_total: int | None = None
    tokens_basis: Literal["complete", "lower_bound", "unknown"] = "unknown"
    npu_runs: int | None = None
    estimated_cost_usd: float | None = None
    price_basis: Literal["priced", "unpriced", "unknown"] = "unknown"
    complete: bool = False
    source_fields: tuple[str, ...] = ()
    @model_validator(mode="after")
    def visible_usage(self):
        for name in (
            "model_calls",
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "output_tokens",
            "tokens_total",
            "npu_runs",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.estimated_cost_usd is not None and (
            not math.isfinite(self.estimated_cost_usd) or self.estimated_cost_usd < 0
        ):
            raise ValueError("estimated_cost_usd must be finite and non-negative or None")
        detail = (self.input_tokens, self.cached_input_tokens, self.output_tokens)
        if self.tokens_basis == "unknown":
            if self.tokens_total is not None or any(value is not None for value in detail):
                raise ValueError("unknown token usage must not invent partial totals")
        else:
            if any(value is None for value in detail) or self.tokens_total is None:
                raise ValueError(f"{self.tokens_basis} token usage requires input/cached/output")
            if sum(detail) != self.tokens_total:
                raise ValueError("input/cached/output tokens must add up to tokens_total")
        if self.complete:
            if self.model_calls is None or self.npu_runs is None:
                raise ValueError("complete usage requires model calls and NPU runs")
            if self.tokens_basis != "complete":
                raise ValueError("complete usage requires complete token accounting")
        elif (
            self.model_calls is not None
            and self.npu_runs is not None
            and self.tokens_basis == "complete"
        ):
            raise ValueError("usage with all observed fields must be marked complete")
        if self.price_basis == "priced":
            if self.estimated_cost_usd is None:
                raise ValueError("priced usage requires an observed USD estimate")
        elif self.estimated_cost_usd is not None:
            raise ValueError("unpriced/unknown usage must not invent a USD estimate")
        if not self.stage_id.strip():
            raise ValueError("stage_id must be non-empty")
        return self


class P3CostLedger(StrictModel):
    seed: CostObservation
    capture: CostObservation
    organizer: CostObservation
    review: CostObservation
    downstream: CostObservation
    @model_validator(mode="after")
    def declared_roles(self):
        for name in PREREQUISITE_ROLES:
            if getattr(self, name).basis != "observed":
                raise ValueError(f"P3 prerequisite {name} cost must be observed")
        if self.downstream.basis != "declared":
            raise ValueError("P3 downstream cost is declared before its native attempt")
        stages = [getattr(self, name).stage_id for name in ROLES]
        if len(set(stages)) != len(stages):
            raise ValueError("each attributed cost stage must have a distinct stage_id")
        return self
    def roles(self) -> dict[str, CostObservation]:
        return {name: getattr(self, name) for name in ROLES}
    @property
    def unknown_usage(self) -> tuple[str, ...]:
        return tuple(name for name, row in self.roles().items() if not row.complete)
    @property
    def prerequisite_unknown_usage(self) -> tuple[str, ...]:
        return tuple(name for name in PREREQUISITE_ROLES if not self.roles()[name].complete)
    @property
    def unpriced_usage(self) -> tuple[str, ...]:
        return tuple(name for name, row in self.roles().items() if row.price_basis != "priced")
    @property
    def prerequisite_unpriced_usage(self) -> tuple[str, ...]:
        return tuple(name for name in PREREQUISITE_ROLES if self.roles()[name].price_basis != "priced")
    def summary(self) -> dict:
        rows = {name: row.model_dump(mode="json") for name, row in self.roles().items()}
        totals = {}
        for field in ("model_calls", "tokens_total", "estimated_cost_usd", "npu_runs"):
            values = [row[field] for row in rows.values() if row[field] is not None]
            totals[field] = sum(values) if values else None
        return {
            "roles": rows,
            "attribution": [
                {
                    "stage_id": row["stage_id"],
                    "receipt_sha256": row["receipt_sha256"],
                    "model_calls": row["model_calls"],
                    "tokens_total": row["tokens_total"],
                    "tokens_basis": row["tokens_basis"],
                    "npu_runs": row["npu_runs"],
                    "estimated_cost_usd": row["estimated_cost_usd"],
                    "price_basis": row["price_basis"],
                    "complete": row["complete"],
                }
                for row in rows.values()
            ],
            "unknown_usage": list(self.unknown_usage),
            "prerequisite_unknown_usage": list(self.prerequisite_unknown_usage),
            "unpriced_usage": list(self.unpriced_usage),
            "known_totals": totals,
            "ledger_sha256": _digest(self.model_dump(mode="json")),
        }

def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

def tokens_from_tokens(tokens: Mapping | None) -> dict:
    """Preserve complete vs lower-bound token accounting from the receipt."""
    tokens = _mapping(tokens)
    input_uncached = _int_or_none(tokens.get("input_uncached"))
    input_cached = _int_or_none(tokens.get("input_cached"))
    output = _int_or_none(tokens.get("output"))
    details = (input_uncached, input_cached, output)
    if any(value is None for value in details):
        return {
            "input_tokens": None,
            "cached_input_tokens": None,
            "cache_creation_tokens": None,
            "output_tokens": None,
            "tokens_total": None,
            "tokens_basis": "unknown",
        }
    basis = "complete" if tokens.get("complete") is True else "lower_bound"
    return {
        "input_tokens": input_uncached,
        "cached_input_tokens": input_cached,
        "cache_creation_tokens": _int_or_none(tokens.get("cache_creation")),
        "output_tokens": output,
        "tokens_total": sum(details),
        "tokens_basis": basis,
    }

def seed_observation(trial: Mapping, trial_sha256: str) -> CostObservation:
    primary = _mapping(_mapping(trial).get("primary_metrics"))
    token = tokens_from_tokens(primary.get("tokens"))
    model_calls = _int_or_none(primary.get("model_calls"))
    npu_runs = _int_or_none(primary.get("npu_test_runs"))
    cost_complete = primary.get("cost_complete") is True
    cost = _float_or_none(primary.get("estimated_cost_usd")) if cost_complete else None
    return CostObservation(
        basis="observed",
        stage_id="seed_trial",
        receipt_sha256=trial_sha256,
        model_calls=model_calls,
        npu_runs=npu_runs,
        estimated_cost_usd=cost,
        price_basis="priced" if cost is not None else "unknown",
        complete=(
            model_calls is not None and npu_runs is not None and token["tokens_basis"] == "complete"
        ),
        source_fields=(
            "primary_metrics.tokens",
            "primary_metrics.model_calls",
            "primary_metrics.npu_test_runs",
            "primary_metrics.cost_complete",
            "primary_metrics.estimated_cost_usd",
        ),
        **token,
    )


def capture_observations(
    capture: Mapping, trial: Mapping, capture_sha256: str
) -> tuple[CostObservation, CostObservation]:
    """Split the capture container total into ingestion and in-capture C6 usage.
    The capture receipt is the organizer receipt. `indexer_usage` is the C6
    organizer; the parent trial's `capture_model_calls` is the container total.
    """
    capture = _mapping(capture)
    trial = _mapping(trial)
    indexer = _mapping(capture.get("indexer_usage"))
    distill = _mapping(capture.get("distill"))
    distill_usage = _mapping(distill.get("usage"))
    capture_tokens = _mapping(capture.get("tokens"))
    auxiliary = _mapping(trial.get("auxiliary_metrics"))
    provider_tokens = _mapping(auxiliary.get("capture_provider_tokens"))
    container_calls = _int_or_none(auxiliary.get("capture_model_calls"))
    organizer_calls = _int_or_none(indexer.get("model_calls"))
    input_tokens = _int_or_none(indexer.get("input_tokens"))
    cached = _int_or_none(indexer.get("cached_input_tokens"))
    output = _int_or_none(indexer.get("output_tokens"))
    unknown_calls = _int_or_none(indexer.get("unknown_calls"))
    organizer_total = (
        sum((input_tokens, cached, output))
        if all(value is not None for value in (input_tokens, cached, output))
        else None
    )
    organizer_tokens_complete = (
        organizer_total is not None
        and unknown_calls == 0
        and capture_tokens.get("complete") is True
        and provider_tokens.get("complete") is True
    )
    organizer_npu = _int_or_none(indexer.get("npu_runs"))
    organizer_complete = (
        organizer_calls is not None and organizer_npu is not None and organizer_tokens_complete
    )
    organizer = CostObservation(
        basis="observed",
        stage_id="graded_distillation",
        receipt_sha256=capture_sha256,
        model_calls=organizer_calls,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        cache_creation_tokens=_int_or_none(indexer.get("cache_creation")),
        output_tokens=output,
        tokens_total=organizer_total,
        tokens_basis="complete" if organizer_tokens_complete else "unknown",
        npu_runs=organizer_npu,
        estimated_cost_usd=_float_or_none(distill_usage.get("estimated_cost_usd")),
        price_basis=("priced" if _float_or_none(distill_usage.get("estimated_cost_usd")) is not None else "unknown"),
        complete=organizer_complete,
        source_fields=(
            "capture.indexer_usage",
            "capture.distill.usage",
            "capture.tokens",
            "trial.auxiliary_metrics.capture_provider_tokens",
            "trial.auxiliary_metrics.capture_model_calls",
        ),
    )
    container_total = _int_or_none(capture_tokens.get("total"))
    if container_calls is not None and organizer_calls is not None:
        ingestion_calls = container_calls - organizer_calls
    else:
        ingestion_calls = None
    if container_total is not None and organizer_total is not None:
        ingestion_total = container_total - organizer_total
    else:
        ingestion_total = None
    capture_npu = _int_or_none(_mapping(capture.get("cleanup")).get("npu_runs"))
    capture_complete = (
        ingestion_calls is not None and capture_npu is not None and ingestion_total == 0
    )
    capture_cost = (
        _float_or_none(capture.get("estimated_cost_usd"))
        if capture.get("cost_complete") is True
        else None
    )
    capture_obs = CostObservation(
        basis="observed",
        stage_id="capture_ingestion",
        receipt_sha256=capture_sha256,
        model_calls=ingestion_calls if ingestion_calls is not None and ingestion_calls >= 0 else None,
        input_tokens=0 if ingestion_total == 0 else None,
        cached_input_tokens=0 if ingestion_total == 0 else None,
        cache_creation_tokens=None,
        output_tokens=0 if ingestion_total == 0 else None,
        tokens_total=ingestion_total if ingestion_total is not None and ingestion_total >= 0 else None,
        tokens_basis="complete" if ingestion_total == 0 else "unknown",
        npu_runs=capture_npu,
        estimated_cost_usd=capture_cost,
        price_basis="priced" if capture_cost is not None else "unknown",
        complete=capture_complete,
        source_fields=(
            "trial.auxiliary_metrics.capture_model_calls",
            "capture.tokens",
            "capture.cleanup",
        ),
    )
    return capture_obs, organizer


def consumer_observation(consumer: Mapping | None, consumer_sha256: str) -> CostObservation | None:
    if consumer is None:
        return None
    consumer = _mapping(consumer)
    model_calls = _int_or_none(consumer.get("model_calls"))
    npu_runs = _int_or_none(consumer.get("npu_runs"))
    zero = model_calls == 0
    return CostObservation(
        basis="observed",
        stage_id="consumer_continuation",
        receipt_sha256=consumer_sha256,
        model_calls=model_calls,
        input_tokens=0 if zero else None,
        cached_input_tokens=0 if zero else None,
        cache_creation_tokens=None,
        output_tokens=0 if zero else None,
        tokens_total=0 if zero else None,
        tokens_basis="complete" if zero else "unknown",
        npu_runs=npu_runs,
        estimated_cost_usd=None,
        price_basis="unknown",
        complete=zero and npu_runs is not None,
        source_fields=("consumer.model_calls", "consumer.npu_runs"),
    )


def review_observation(review: Mapping, review_sha256: str) -> CostObservation:
    review = _mapping(review)
    token = tokens_from_tokens(review.get("tokens"))
    model_calls = _int_or_none(review.get("model_calls"))
    npu_runs = _int_or_none(review.get("npu_runs"))
    cost = _float_or_none(review.get("estimated_cost_usd")) if review.get("cost_complete") is True else None
    return CostObservation(
        basis="observed",
        stage_id="independent_review",
        receipt_sha256=review_sha256,
        model_calls=model_calls,
        npu_runs=npu_runs,
        estimated_cost_usd=cost,
        price_basis="priced" if cost is not None else "unknown",
        complete=(model_calls is not None and npu_runs is not None and token["tokens_basis"] == "complete"),
        source_fields=("review.model_calls", "review.tokens", "review.npu_runs", "review.cost_complete"),
        **token,
    )


def capture_container(capture: Mapping, trial: Mapping) -> dict:
    capture = _mapping(capture)
    trial = _mapping(trial)
    auxiliary = _mapping(trial.get("auxiliary_metrics"))
    tokens = _mapping(capture.get("tokens"))
    provider = _mapping(auxiliary.get("capture_provider_tokens"))
    return {
        "status": capture.get("status"),
        "task_state": capture.get("task_state"),
        "protocol": capture.get("protocol"),
        "corpus_lint_status": _mapping(capture.get("corpus_lint")).get("status"),
        "cleanup_status": _mapping(capture.get("cleanup")).get("status"),
        "cleanup_errors": _mapping(capture.get("cleanup")).get("errors"),
        "distill_status": _mapping(capture.get("distill")).get("status"),
        "distill_model_calls": _int_or_none(_mapping(capture.get("distill")).get("model_calls")),
        "billing_status": _mapping(capture.get("distill")).get("billing_status"),
        "distill_cleanup_failed": _mapping(capture.get("distill")).get("cleanup_failed"),
        "model_calls": _int_or_none(auxiliary.get("capture_model_calls")),
        "tokens_total": _int_or_none(tokens.get("total")),
        "tokens_complete": tokens.get("complete") is True,
        "provider_tokens": provider,
        "provider_tokens_complete": provider.get("complete") is True,
        "capture_process": _mapping(auxiliary.get("capture_process")),
        "capture_wall_clock_sec": auxiliary.get("capture_wall_clock_sec"),
    }


CAPTURE_STAGE_IDS = frozenset(
    {
        "native_artifact_selection",
        "seed_local_publication",
        "graded_distillation",
        "local_publication_and_merge",
        "independent_consumer_readback",
    }
)


def capture_stage_observations(capture: Mapping) -> list[P3StageObservation]:
    """Read the factual stage records from the capture receipt itself."""
    rows = []
    for row in _mapping(capture).get("stage_observations") or []:
        if not isinstance(row, Mapping) or row.get("name") not in CAPTURE_STAGE_IDS:
            continue
        rows.append(
            P3StageObservation(
                stage_id=str(row.get("name")),
                status=str(row.get("status")),
                started_sec=row.get("started_sec"),
                ended_sec=row.get("ended_sec"),
                elapsed_sec=row.get("elapsed_sec"),
                started_at=row.get("started_at"),
                ended_at=row.get("ended_at"),
                error=row.get("error"),
            )
        )
    return rows


def stage_order_reasons(stages: list[P3StageObservation]) -> list[str]:
    by_name = {stage.stage_id: stage for stage in stages}
    required = (
        "native_artifact_selection",
        "seed_local_publication",
        "graded_distillation",
        "local_publication_and_merge",
        "independent_consumer_readback",
    )
    reasons = [f"capture_stage_missing:{name}" for name in required if name not in by_name]
    if reasons:
        return reasons
    ingestion_ends = (
        by_name["native_artifact_selection"].ended_sec,
        by_name["seed_local_publication"].ended_sec,
    )
    organizer = by_name["graded_distillation"]
    publication = by_name["local_publication_and_merge"]
    consumer = by_name["independent_consumer_readback"]
    if any(
        value is None
        for value in (
            *ingestion_ends,
            organizer.started_sec,
            organizer.ended_sec,
            publication.started_sec,
            publication.ended_sec,
            consumer.started_sec,
        )
    ):
        reasons.append("capture_stage_relative_time_missing")
    else:
        ingestion_end = max(ingestion_ends)
        if not ingestion_end <= organizer.started_sec:
            reasons.append("capture_ingestion_after_organizer")
        if not organizer.ended_sec <= publication.started_sec:
            reasons.append("organizer_after_publication")
        if not publication.ended_sec <= consumer.started_sec:
            reasons.append("publication_after_consumer")
    if organizer.status == "failed":
        reasons.append("organizer_stage_failed")
    if consumer.status == "failed":
        reasons.append("capture_consumer_stage_failed")
    return reasons


def timeline_missing(**times) -> list[str]:
    return [name for name, value in times.items() if value is None]


def ledger_numeric_mismatches(
    ledger: P3CostLedger | Mapping, expected: Mapping[str, Mapping]
) -> list[str]:
    """Compare each attributed ledger row with actual observed receipt values."""
    roles = ledger.roles() if isinstance(ledger, P3CostLedger) else _mapping(ledger.get("roles"))
    mismatches = []
    for role, wanted in expected.items():
        row = roles.get(role, {})
        if not isinstance(row, Mapping):
            row = row.model_dump(mode="json")
        for field in (
            "stage_id",
            "receipt_sha256",
            "model_calls",
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "output_tokens",
            "tokens_total",
            "tokens_basis",
            "npu_runs",
            "estimated_cost_usd",
            "price_basis",
            "complete",
        ):
            if row.get(field) != wanted.get(field):
                mismatches.append(f"{role}.{field} != observed.{field}")
    return mismatches


def capture_attribution_mismatches(ledger: P3CostLedger | Mapping, container: Mapping) -> dict:
    roles = ledger.roles() if isinstance(ledger, P3CostLedger) else _mapping(ledger.get("roles"))
    capture, organizer = roles.get("capture", {}), roles.get("organizer", {})
    if isinstance(capture, CostObservation):
        capture = capture.model_dump(mode="json")
    if isinstance(organizer, CostObservation):
        organizer = organizer.model_dump(mode="json")
    mismatches, unknown = [], []
    if capture.get("stage_id") == organizer.get("stage_id"):
        mismatches.append("capture_and_organizer_share_stage_id")
    for role, row in (("capture", capture), ("organizer", organizer)):
        if row.get("receipt_sha256") != container.get("capture_receipt_sha256"):
            mismatches.append(f"{role}_receipt_not_capture_container")
    if capture.get("model_calls") is None or organizer.get("model_calls") is None:
        unknown.append("capture_model_calls_unknown")
    elif container.get("model_calls") is None:
        unknown.append("capture_container_model_calls_unknown")
    elif capture["model_calls"] + organizer["model_calls"] != container["model_calls"]:
        mismatches.append("capture_model_calls_double_attributed_or_lost")
    if capture.get("tokens_basis") == "unknown" or organizer.get("tokens_basis") == "unknown":
        unknown.append("capture_tokens_unknown")
    elif container.get("tokens_total") is None:
        unknown.append("capture_container_tokens_unknown")
    elif capture["tokens_total"] + organizer["tokens_total"] != container["tokens_total"]:
        mismatches.append("capture_tokens_double_attributed_or_lost")
    if capture.get("npu_runs") is None or organizer.get("npu_runs") is None:
        unknown.append("capture_npu_runs_unknown")
    elif capture["npu_runs"] + organizer["npu_runs"] != 0:
        mismatches.append("capture_npu_runs_not_zero")
    return {"duplicate_or_lost": mismatches, "unknown": unknown}

def parse_capture_receipts(
    *,
    capture: Mapping,
    trial: Mapping,
    consumer: Mapping | None,
    capture_sha256: str,
    trial_sha256: str,
    consumer_sha256: str | None,
    capture_locator: str,
    trial_locator: str,
    consumer_locator: str | None,
    consumer_scope: str | None = None,
) -> dict:
    """Factual contract check of one retained real capture/Seed/consumer receipt."""
    capture_obs, organizer_obs = capture_observations(capture, trial, capture_sha256)
    seed_obs = seed_observation(trial, trial_sha256)
    consumer_obs = (
        consumer_observation(consumer, consumer_sha256)
        if consumer is not None and consumer_sha256 is not None
        else None
    )
    container = capture_container(capture, trial)
    container_with_sha = dict(container, capture_receipt_sha256=capture_sha256)
    stages = capture_stage_observations(capture)
    reasons = stage_order_reasons(stages)
    if capture.get("status") != "completed":
        reasons.append(f"capture_status:{capture.get('status')}")
    if _mapping(capture.get("corpus_lint")).get("status") != "passed":
        reasons.append(f"corpus_lint:{_mapping(capture.get('corpus_lint')).get('status')}")
    if _mapping(capture.get("cleanup")).get("status") != "completed":
        reasons.append("capture_cleanup_not_completed")
    if _mapping(capture.get("distill")).get("cleanup_failed") is True:
        reasons.append("capture_distill_cleanup_failed")
    if _mapping(capture.get("distill")).get("status") != "applied":
        reasons.append(f"distill_status:{_mapping(capture.get('distill')).get('status')}")
    if _mapping(capture.get("distill")).get("model_calls") != organizer_obs.model_calls:
        reasons.append("distill_model_calls_differ_from_indexer_usage")
    if _mapping(_mapping(capture.get("distill")).get("usage")) != {
        "input_tokens": organizer_obs.input_tokens,
        "cached_input_tokens": organizer_obs.cached_input_tokens,
        "output_tokens": organizer_obs.output_tokens,
    }:
        reasons.append("distill_usage_differ_from_indexer_usage")
    if not seed_obs.complete:
        reasons.append("seed_usage_incomplete")
    if not organizer_obs.complete:
        reasons.append("organizer_usage_incomplete")
    if not capture_obs.complete:
        reasons.append("capture_usage_incomplete")
    if seed_obs.price_basis != "priced":
        reasons.append("seed_price_unknown")
    if organizer_obs.price_basis != "priced":
        reasons.append("organizer_price_unknown")
    if consumer_obs is not None and not consumer_obs.complete:
        reasons.append("consumer_usage_incomplete")
    attribution = capture_attribution_mismatches(
        {
            "roles": {
                "capture": capture_obs.model_dump(mode="json"),
                "organizer": organizer_obs.model_dump(mode="json"),
            }
        },
        container_with_sha,
    )
    reasons.extend(attribution["duplicate_or_lost"])
    reasons.extend(attribution["unknown"])
    absolute = {}
    for stage in stages:
        if stage.started_at is not None:
            absolute[f"{stage.stage_id}.started_at"] = stage.started_at.isoformat()
        if stage.ended_at is not None:
            absolute[f"{stage.stage_id}.ended_at"] = stage.ended_at.isoformat()
    for name, value in (
        ("seed.completed_at", None),
        ("capture.started_at", None),
        ("capture.completed_at", None),
        ("consumer.completed_at", None),
        ("frozen_corpus.frozen_at", None),
        ("review.completed_at", None),
        ("downstream.started_at", None),
    ):
        if value is None:
            reasons.append(f"absolute_timestamp_missing:{name}")
    raw_hashes_match = None
    if consumer is not None:
        raw_capture = _mapping(_mapping(consumer).get("raw_capture_receipt")).get("sha256")
        raw_trial = _mapping(_mapping(consumer).get("raw_trial_receipt")).get("sha256")
        raw_hashes_match = raw_capture == capture_sha256 and raw_trial == trial_sha256
        if not raw_hashes_match:
            reasons.append("consumer_raw_receipt_hash_mismatch")
        if consumer.get("original_receipts_unchanged") is not True:
            reasons.append("consumer_original_receipts_not_unchanged")
        if consumer.get("raw_capture_status") != capture.get("status"):
            reasons.append("consumer_raw_capture_status_mismatch")
        if consumer.get("raw_trial_status") != _mapping(trial).get("terminal_status"):
            reasons.append("consumer_raw_trial_status_mismatch")
        if consumer_scope and "Heldout" not in consumer_scope and "heldout" not in consumer_scope:
            reasons.append("consumer_scope_is_not_heldout")
    else:
        reasons.append("consumer_receipt_missing")
    reasons.append("independent_review_missing")
    reasons.append("p3_new_family_heldout_not_established")
    reasons.append("retained_failed_receipt_not_acceptance_evidence")
    return {
        "schema": "mindie-bench-p3-real-receipt-contract/1",
        "purpose": "NONPAID metadata-only contract verification; never P3/Heldout acceptance",
        "receipts": {
            "capture": {"locator": capture_locator, "sha256": capture_sha256},
            "trial": {"locator": trial_locator, "sha256": trial_sha256},
            "consumer": (
                {"locator": consumer_locator, "sha256": consumer_sha256}
                if consumer_locator is not None and consumer_sha256 is not None
                else None
            ),
        },
        "capture": dict(container, stage_observations=[stage.model_dump(mode="json") for stage in stages]),
        "seed": seed_obs.model_dump(mode="json"),
        "organizer": organizer_obs.model_dump(mode="json"),
        "capture_ingestion": capture_obs.model_dump(mode="json"),
        "consumer": (
            consumer_obs.model_dump(mode="json") if consumer_obs is not None else None
        ),
        "attribution": {
            "capture_container_model_calls": container.get("model_calls"),
            "capture_container_tokens_total": container.get("tokens_total"),
            "capture_plus_organizer_model_calls": (
                capture_obs.model_calls + organizer_obs.model_calls
                if capture_obs.model_calls is not None and organizer_obs.model_calls is not None
                else None
            ),
            "capture_plus_organizer_tokens_total": (
                capture_obs.tokens_total + organizer_obs.tokens_total
                if capture_obs.tokens_total is not None and organizer_obs.tokens_total is not None
                else None
            ),
            "duplicate_or_lost": attribution["duplicate_or_lost"],
            "unknown": attribution["unknown"],
            "matched": not attribution["duplicate_or_lost"] and not attribution["unknown"],
        },
        "timestamps": {
            "absolute_available": bool(absolute),
            "absolute": absolute,
            "relative_stage_seconds_present": bool(stages),
            "missing_absolute_fields": [
                "seed.completed_at",
                "capture.started_at",
                "capture.completed_at",
                "consumer.completed_at",
                "frozen_corpus.frozen_at",
                "review.completed_at",
                "downstream.started_at",
            ],
            "inference_performed": False,
        },
        "receipt_immutability": {
            "consumer_original_receipts_unchanged": (
                _mapping(consumer).get("original_receipts_unchanged") if consumer else None
            ),
            "consumer_raw_receipt_hash_matches": raw_hashes_match,
            "this_worker_modified_receipts": False,
        },
        "consumer_scope": consumer_scope,
        "verdict": {
            "status": "denied",
            "p3_eligible": False,
            "acceptance_upgrade": False,
            "reasons": sorted(set(reasons)),
        },
        "model_calls": 0,
        "npu_runs": 0,
        "limitations": [
            "Legacy capture receipt has relative stage seconds but no absolute stage timestamps; missing absolute times are reported, not inferred.",
            "The capture receipt has no per-stage NPU count or price schedule; those fields remain unknown and incomplete.",
            "The consumer continuation is a zero-model local-only Dev repair for a previously exposed task; it is not a different-family Heldout.",
            "The retained capture failed (status=failed, consumer readback failed, corpus_lint=pending); it is immutable evidence and must not be upgraded.",
            "No source/oracle/transcript/body content was read; only receipt metadata.",
        ],
    }


def budget_ledger_mismatches(ledger: Mapping, declared: Mapping) -> list[str]:
    """Compare the reviewed cost ledger with the product profile's declared shares."""
    roles = ledger.get("roles", {})
    mismatches = []
    for budget_key, role, field in (
        ("seed_model_calls", "seed", "model_calls"),
        ("capture_model_calls", "capture", "model_calls"),
        ("seed_npu_runs", "seed", "npu_runs"),
        ("capture_organizer_model_calls", "organizer", "model_calls"),
        ("independent_review_model_calls", "review", "model_calls"),
    ):
        if roles.get(role, {}).get(field) != declared.get(budget_key):
            mismatches.append(f"{role}.{field} != budget.{budget_key}")
    for budget_key, field in (("total_model_calls", "model_calls"), ("npu_runs", "npu_runs")):
        if roles.get("downstream", {}).get(field) != declared.get(budget_key):
            mismatches.append(f"downstream.{field} != budget.{budget_key}")
    for budget_key, role in (
        ("seed_cost_usd", "seed"),
        ("capture_cost_usd", "capture"),
        ("capture_organizer_cost_usd", "organizer"),
        ("independent_review_cost_usd", "review"),
    ):
        row = roles.get(role, {})
        observed = row.get("estimated_cost_usd")
        expected = declared.get(budget_key)
        if observed is None:
            if row.get("price_basis") == "priced" or expected not in (0, 0.0, None):
                mismatches.append(f"{role}.estimated_cost_usd != budget.{budget_key}")
        elif observed != expected:
            mismatches.append(f"{role}.estimated_cost_usd != budget.{budget_key}")
    if sorted(ledger.get("prerequisite_unknown_usage", [])) != sorted(
        declared.get("unknown_cost_roles", [])
    ):
        mismatches.append("unknown_cost_roles != ledger.prerequisite_unknown_usage")
    return mismatches
