from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_value_product import collect_product, collector, make_suite, write_evidence

from mindie_bench.value_publish import write_component_views


def component_report(tmp_path: Path):
    suite, inputs, _ = make_suite(tmp_path)
    suite_path, inputs_path = write_evidence(tmp_path, suite, inputs)
    report = collect_product(
        suite=suite_path, inputs=inputs_path, output=tmp_path / "out",
        collector_source=collector(),
    )
    assert report["release_gate"], report["blocking_reasons"]
    manifest = {
        "product_version": "0.0.3",
        "repositories": [
            {
                "repository": target["repository"],
                "version": target["version"],
                "revision": target["revision"],
            }
            for target in report["final_release"]["repositories"]
        ],
    }
    return report, manifest


def test_component_views_keep_native_compatibility_separate_from_combination_context(tmp_path):
    report, manifest = component_report(tmp_path)
    files = write_component_views(report, manifest, tmp_path / "views")
    assert len(files) == 20
    codex = json.loads((tmp_path / "views/codex/product-0.0.3-value-results.json").read_text())
    knowledge = json.loads(
        (tmp_path / "views/knowledge/product-0.0.3-value-results.json").read_text()
    )
    dsh = json.loads((tmp_path / "views/dsh/product-0.0.3-value-results.json").read_text())
    bench = json.loads(
        (tmp_path / "views/mindie-bench/product-0.0.3-value-results.json").read_text()
    )
    assert codex["schema_version"] == "mindie-component-value-report/2"
    assert codex["matrix"] is None and codex["comparisons"] == []
    assert "receipts" not in codex and "p3" not in codex
    assert "compatibility evidence only" in codex["benchmark_scope"]
    assert codex["component_revision"] == next(
        target["revision"] for target in manifest["repositories"]
        if target["repository"] == "mindie-agent/mindie-agent-codex"
    )
    assert knowledge["matrix"] is not None and knowledge["comparisons"]
    assert "cannot be attributed to this component alone" in knowledge["benchmark_scope"]
    assert dsh["comparisons"] and dsh["matrix"] is not None
    assert bench["component"] == "mindie-bench"
    for view in (codex, knowledge, dsh, bench):
        serialized = json.dumps(view)
        assert "/private/" not in serialized
        assert "private-service" not in serialized
        assert view["component_revision"] in report["component_revisions"].values()
    wrong = json.loads(json.dumps(manifest))
    wrong["repositories"][0]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="attachment source differs"):
        write_component_views(report, wrong, tmp_path / "wrong")
