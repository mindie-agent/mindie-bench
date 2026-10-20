from __future__ import annotations

from pathlib import Path

from mindie_bench.io import digest, read_json
from mindie_bench.value_product import (
    collect_product,
    evaluate_product,
    verify_publication_product,
)

FIXTURE = Path(__file__).parent / "fixtures/product-value-v2"


def test_checked_in_minimal_suite_inputs_and_evidence_are_runnable(tmp_path):
    suite = read_json(FIXTURE / "value-suite.json")
    collector = {**suite["benchmark"], "dirty": False, "editable": False}
    collected = collect_product(
        suite=FIXTURE / "value-suite.json",
        inputs=FIXTURE / "value-inputs.json",
        output=tmp_path / "collected",
        collector_source=collector,
    )
    assert collected["release_gate"], collected["blocking_reasons"]
    evidence = read_json(FIXTURE / "value-evidence.json")
    public_suite = read_json(FIXTURE / "value-suite.json")
    assert evidence["suite_identity"] == digest(public_suite)
    assert len(evidence["private_suite_sha256"]) == 64
    recomputed = evaluate_product(suite, evidence)
    assert recomputed["release_gate"], recomputed["blocking_reasons"]
    assert recomputed["suite_identity"] == digest(public_suite)
    verified = verify_publication_product(
        suite=FIXTURE / "value-suite.json",
        evidence=FIXTURE / "value-evidence.json",
        manifest=FIXTURE / "release-manifest.json",
        collector_source=collector,
    )
    assert verified["release_gate"]
    assert verified["release_manifest_sha256"]
    assert (tmp_path / "collected/value-results.md").is_file()
    assert (tmp_path / "collected/value-homepage.en.md").is_file()
