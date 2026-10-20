from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from mindie_bench.evaluation.split_manifest import (
    SourceAsset,
    SplitManifest,
    admit_materials,
    build_manifest,
    main,
    validate_g3_timeline,
)


def asset(identity, family, **changes):
    values = dict(
        asset_id=identity,
        task_family=family,
        source_id=family,
        source_locator=f"repo@commit:{identity}",
        source_kind="transcript",
        timestamp="2026-10-01T00:00:00+00:00",
        lineage_resolved=True,
    )
    values.update(changes)
    return SourceAsset(**values)


def test_family_and_source_linked_rewrites_cannot_leak_into_seed():
    with pytest.raises(ValueError, match="source_id"):
        build_manifest(
            [asset("original", "held"), asset("rewrite", "seed", source_id="held")],
            {"held": "heldout", "seed": "seed"},
        )
    with pytest.raises(ValueError, match="related source assets"):
        build_manifest(
            [asset("original", "held", related_asset_ids=["branch-copy"]), asset("branch-copy", "seed")],
            {"held": "heldout", "seed": "seed"},
        )
    with pytest.raises(ValueError, match="content_sha256"):
        build_manifest(
            [
                asset("original", "held", content_sha256="a" * 64),
                asset("renamed", "seed", content_sha256="a" * 64),
            ],
            {"held": "heldout", "seed": "seed"},
        )


def test_unresolved_lineage_and_late_seed_evidence_block_admission():
    with pytest.raises(ValueError, match="unresolved lineage"):
        build_manifest([asset("unclassified", "family", lineage_resolved=False)], {"family": "seed"})
    with pytest.raises(ValueError, match="newer than knowledge cutoff"):
        build_manifest(
            [asset("seed", "family")],
            {"family": "seed"},
            seed_cutoff=datetime(2026, 9, 30, tzinfo=timezone.utc),
        )


def test_corpus_admission_checks_every_provenance_id_without_opening_sources():
    manifest = build_manifest(
        [asset("seed-diff", "one"), asset("held-report", "two")], {"one": "seed", "two": "heldout"}
    )
    assert admit_materials(manifest, ["seed-diff"], "static_corpus")["task_families"] == ["one"]
    with pytest.raises(ValueError, match="rejects non-seed"):
        admit_materials(manifest, ["seed-diff", "held-report"], "static_corpus")
    with pytest.raises(ValueError, match="unmanifested"):
        admit_materials(manifest, ["unknown-rewrite"], "seed_distillation")
    changed = manifest.model_dump(mode="json")
    changed["assets"][1]["split"] = "seed"
    with pytest.raises(ValueError, match="digest"):
        SplitManifest.model_validate(changed)


def test_g3_requires_disjoint_sources_and_capture_before_evaluation():
    manifest = build_manifest(
        [asset("seed", "one"), asset("eval", "two")], {"one": "seed", "two": "heldout"}
    )
    times = [datetime(2026, 10, 2, hour, tzinfo=timezone.utc) for hour in range(4)]
    args = dict(
        seed_asset_ids=["seed"],
        downstream_asset_ids=["eval"],
        seed_completed_at=times[0],
        capture_completed_at=times[1],
        corpus_frozen_at=times[2],
        downstream_started_at=times[3],
    )
    assert validate_g3_timeline(manifest, **args)["status"] == "admitted"
    args["capture_completed_at"] = times[3]
    with pytest.raises(ValueError, match="ordering"):
        validate_g3_timeline(manifest, **args)
    args["capture_completed_at"] = times[0]
    with pytest.raises(ValueError, match="ordering"):
        validate_g3_timeline(manifest, **args)


def test_cli_build_and_admit_round_trip_without_source_reads(tmp_path, capsys):
    inventory = tmp_path / "inventory.json"
    assignments = tmp_path / "assignments.json"
    output = tmp_path / "benchmark-split-manifest.json"
    inventory.write_text(json.dumps([asset("seed", "one").model_dump(mode="json")]))
    assignments.write_text(json.dumps({"one": "seed"}))
    assert (
        main(
            [
                "build",
                "--inventory",
                str(inventory),
                "--assignments",
                str(assignments),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert (
        main(
            ["admit", "--manifest", str(output), "--purpose", "seed_distillation", "--asset-id", "seed"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "admitted"
