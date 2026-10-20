"""Metadata-only family isolation before corpus distillation or task extraction.

[Role] Freeze explicit Seed/Dev/Heldout assignments and audit provenance use.
[Invariants] Related transcripts, branches, diffs, reports and rewrites stay on
one side. Unknown lineage is unresolved; this module never reads source content.
[Timeouts & Locks] In-memory validation and bounded local JSON reads; no locks.
[Side-Effect Policy] CLI writes only the requested manifest or audit result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from mindie_bench.io import write_json
from mindie_bench.models import StrictModel

Split = Literal["seed", "dev", "heldout"]
Purpose = Literal["seed_distillation", "static_corpus", "dev_evaluation", "heldout_task"]


class SourceAsset(StrictModel):
    asset_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    # source_id identifies the original problem/session, not an entire repository.
    source_id: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    source_kind: Literal["transcript", "commit", "diff", "report", "experiment", "rewrite", "task"]
    timestamp: datetime
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    related_asset_ids: list[str] = Field(default_factory=list)
    lineage_resolved: bool

    @field_validator("timestamp")
    @classmethod
    def absolute_time(cls, value):
        if value.tzinfo is None:
            raise ValueError("source timestamps must include a timezone")
        return value


class SplitAsset(SourceAsset):
    split: Split


class SplitManifest(StrictModel):
    schema_version: Literal["mindie-bench-split/1"] = "mindie-bench-split/1"
    created_at: datetime
    seed_cutoff: datetime | None = None
    assets: list[SplitAsset]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def boundary_integrity(self):
        _validate_assets(self.assets, self.seed_cutoff)
        if self.manifest_sha256 != _digest(self.assets, self.seed_cutoff):
            raise ValueError("split manifest digest does not match its assignments and provenance")
        return self


def _digest(assets: list[SplitAsset], cutoff: datetime | None) -> str:
    payload = dict(
        seed_cutoff=cutoff.isoformat() if cutoff else None,
        assets=[
            asset.model_dump(mode="json") for asset in sorted(assets, key=lambda asset: asset.asset_id)
        ],
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_assets(assets: list[SplitAsset], cutoff: datetime | None):
    by_id = {asset.asset_id: asset for asset in assets}
    if not by_id or len(by_id) != len(assets):
        raise ValueError("split requires nonempty, unique source asset identities")
    if cutoff is not None and cutoff.tzinfo is None:
        raise ValueError("seed cutoff must include a timezone")
    families, sources, locators, hashes = {}, {}, {}, {}
    for asset in assets:
        if not asset.lineage_resolved:
            raise ValueError(f"unresolved lineage cannot be admitted for distillation: {asset.asset_id}")
        for label, value, index in (
            ("task_family", asset.task_family, families),
            ("source_id", asset.source_id, sources),
            ("source_locator", asset.source_locator, locators),
            ("content_sha256", asset.content_sha256, hashes),
        ):
            if value is None:
                continue
            if value in index and index[value] != asset.split:
                raise ValueError(f"cross-split contamination via {label}: {value}")
            index[value] = asset.split
        if asset.split == "seed" and cutoff is not None and asset.timestamp > cutoff:
            raise ValueError(f"seed asset is newer than knowledge cutoff: {asset.asset_id}")
        for related in asset.related_asset_ids:
            if related not in by_id:
                raise ValueError(f"unresolved related asset: {asset.asset_id} -> {related}")
            if by_id[related].split != asset.split:
                raise ValueError(f"related source assets cross splits: {asset.asset_id} -> {related}")


def build_manifest(
    assets: list[SourceAsset],
    family_splits: dict[str, Split],
    *,
    seed_cutoff: datetime | None = None,
    created_at: datetime | None = None,
) -> SplitManifest:
    """Apply reviewed family assignments; never infer answer-free independence.

    Inventory producers must resolve family/lineage using metadata before
    handing Seed and Heldout assets to different workers and model contexts.
    """
    families = {asset.task_family for asset in assets}
    if families != family_splits.keys():
        raise ValueError("every inventoried family needs exactly one explicit split assignment")
    assigned = [
        SplitAsset(**asset.model_dump(), split=family_splits[asset.task_family]) for asset in assets
    ]
    return SplitManifest(
        created_at=created_at or datetime.now(timezone.utc),
        seed_cutoff=seed_cutoff,
        assets=assigned,
        manifest_sha256=_digest(assigned, seed_cutoff),
    )


def admit_materials(manifest: SplitManifest, asset_ids: list[str], purpose: Purpose) -> dict:
    """Resolve all supplied provenance before allowing the content consumer to read it."""
    manifest.boundary_integrity()
    if not asset_ids:
        raise ValueError("material admission requires explicit provenance")
    allowed = {
        "seed_distillation": "seed",
        "static_corpus": "seed",
        "dev_evaluation": "dev",
        "heldout_task": "heldout",
    }[purpose]
    by_id = {asset.asset_id: asset for asset in manifest.assets}
    missing = set(asset_ids) - by_id.keys()
    if missing:
        raise ValueError("unmanifested provenance cannot be admitted: " + ", ".join(sorted(missing)))
    rejected = [identity for identity in asset_ids if by_id[identity].split != allowed]
    if rejected:
        raise ValueError(f"{purpose} rejects non-{allowed} provenance: " + ", ".join(rejected))
    return dict(
        status="admitted",
        purpose=purpose,
        split=allowed,
        manifest_sha256=manifest.manifest_sha256,
        asset_ids=sorted(set(asset_ids)),
        task_families=sorted({by_id[identity].task_family for identity in asset_ids}),
        source_locators=[by_id[identity].source_locator for identity in sorted(set(asset_ids))],
    )


def validate_g3_timeline(
    manifest: SplitManifest,
    *,
    seed_asset_ids: list[str],
    downstream_asset_ids: list[str],
    seed_completed_at: datetime,
    capture_completed_at: datetime,
    corpus_frozen_at: datetime,
    downstream_started_at: datetime,
) -> dict:
    times = (seed_completed_at, capture_completed_at, corpus_frozen_at, downstream_started_at)
    if any(value.tzinfo is None for value in times) or any(
        right <= left for left, right in zip(times, times[1:])
    ):
        raise ValueError("G3 requires timezone-aware seed -> capture -> freeze -> downstream ordering")
    if not seed_asset_ids or not downstream_asset_ids:
        raise ValueError("G3 requires seed and downstream provenance")
    seed = admit_materials(manifest, seed_asset_ids, "seed_distillation")
    downstream = admit_materials(manifest, downstream_asset_ids, "heldout_task")
    return dict(
        status="admitted",
        manifest_sha256=manifest.manifest_sha256,
        seed=seed,
        downstream=downstream,
        timeline={
            name: value.isoformat()
            for name, value in zip(
                (
                    "seed_completed_at",
                    "capture_completed_at",
                    "corpus_frozen_at",
                    "downstream_started_at",
                ),
                times,
                strict=True,
            )
        },
        claim_scope="separate problem families; shared broad domain does not establish task independence",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument(
        "--inventory", type=Path, required=True, help="JSON array of SourceAsset metadata"
    )
    build.add_argument(
        "--assignments", type=Path, required=True, help="JSON object mapping task_family to split"
    )
    build.add_argument("--seed-cutoff", type=datetime.fromisoformat)
    build.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("admit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument(
        "--purpose",
        choices=["seed_distillation", "static_corpus", "dev_evaluation", "heldout_task"],
        required=True,
    )
    audit.add_argument("--asset-id", action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        assets = [SourceAsset.model_validate(row) for row in json.loads(args.inventory.read_text())]
        manifest = build_manifest(
            assets, json.loads(args.assignments.read_text()), seed_cutoff=args.seed_cutoff
        )
        write_json(args.output, manifest.model_dump(mode="json"))
        print(
            json.dumps(
                dict(
                    status="validated",
                    manifest_sha256=manifest.manifest_sha256,
                    assets=len(manifest.assets),
                    output=str(args.output),
                )
            )
        )
    else:
        manifest = SplitManifest.model_validate_json(args.manifest.read_text())
        print(json.dumps(admit_materials(manifest, args.asset_id, args.purpose)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
