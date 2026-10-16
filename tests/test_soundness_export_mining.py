import json
import sys
from pathlib import Path

import pytest
from test_metrics import receipt

from mindie_bench.hf_sync.exporter import export_hf, public_row
from mindie_bench.mining.knowledge_to_task import mine_knowledge
from mindie_bench.models import TaskSpec
from mindie_bench.runner.soundness import check_soundness
from mindie_bench.scorer.comparator import freeze, load_receipts


def test_soundness_checks_fail_before_and_pass_after_and_preserves_source(case):
    (case["task"] / "oracle/gold_patch.diff").write_text(
        "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n"
        "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    )
    proof = check_soundness(case["task"], case["config"], tier="l2", output=case["root"] / "soundness")
    assert proof["status"] == "completed"
    assert proof["before_reward"] == 0
    assert proof["after_reward"] == 1
    assert proof["model_calls"] == 0
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"
    assert TaskSpec.load(case["task"]).task.evidence == "synthetic"


def test_a_history_narrative_only_yields_a_candidate(corpus, tmp_path):
    source, commit = corpus
    output = tmp_path / "mined"
    rows = mine_knowledge(source, output, limit=2)
    assert len(rows) == 2
    assert all(row["evidence"] == "candidate" for row in rows)
    assert all(row["source_commit"] == commit for row in rows)
    assert all(row["missing_acceptance"] for row in rows)
    assert not list(output.rglob("task.toml"))


def test_baseline_tampering_is_an_error(tmp_path):
    baseline = tmp_path / "baseline"
    freeze([receipt("completed_pass")], baseline)
    path = next(baseline.rglob("trial_receipt.json"))
    value = json.loads(path.read_text())
    value["primary_metrics"]["total_wall_clock_sec"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        load_receipts(baseline)


def test_interrupted_receipts_cannot_disappear_from_score(tmp_path):
    freeze([receipt("completed_pass")], tmp_path / "baseline")
    incomplete = tmp_path / "baseline/missing"
    incomplete.mkdir()
    (incomplete / "attempt_started.json").write_text("{}")
    with pytest.raises(ValueError, match="UNKNOWN"):
        load_receipts(tmp_path / "baseline")


def test_export_never_contains_private_traces_configs_or_error_messages():
    row = receipt("unknown")
    row.artifacts["private"] = "/some/private/session.jsonl"
    row.environment_fingerprint["mcp_config"] = {"secret": "do-not-export"}
    row.errors = [{"stage": "agent", "type": "Error", "message": "private raw exception"}]
    encoded = json.dumps(public_row(row))
    assert "do-not-export" not in encoded
    assert "/some/private" not in encoded
    assert "private raw exception" not in encoded
    assert '"agent"' in encoded


def test_missing_redactor_is_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        export_hf(
            tmp_path / "input",
            tmp_path / "public",
            knowledge_python=Path("/does/not/exist"),
            gitleaks=Path("/does/not/exist"),
        )
    assert not (tmp_path / "public").exists()


def test_redactor_failure_never_publishes_partial_output(tmp_path, monkeypatch):
    source = tmp_path / "input"
    freeze([receipt("completed_pass")], source)
    scanner = tmp_path / "scanner"
    scanner.write_text("#!/bin/sh\nexit 1\n")
    scanner.chmod(0o700)
    output = tmp_path / "public"
    with pytest.raises(RuntimeError, match="redaction"):
        export_hf(source, output, knowledge_python=Path(sys.executable), gitleaks=scanner)
    assert not output.exists()
    assert not list(tmp_path.glob(".mindie-export-*"))
