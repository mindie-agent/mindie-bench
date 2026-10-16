import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def public_export_input(tmp_path):
    source = tmp_path / "input"
    originals = []
    for repetition, status in enumerate(["unknown", "infra_failed"], 1):
        row = receipt(status, identity="rmsnorm-null-description", repetition=repetition)
        row.trial_id = f"rmsnorm-null-description_G0_{repetition}_05b0191b093{repetition}"
        row.environment_fingerprint["remote_environment"] = {"kernel": "5.10.0-216.0.0"}
        directory = source / str(repetition)
        directory.mkdir(parents=True)
        (directory / "trial_receipt.json").write_text(row.model_dump_json())
        originals.append(row)
    scanner = tmp_path / "scanner"
    scanner.write_text("#!/bin/sh\nexit 0\n")
    scanner.chmod(0o700)
    return source, originals, scanner


def mock_redaction(monkeypatch, mutate):
    def run(command, **kwargs):
        if command[1:3] == ["-I", "-c"]:
            data = json.loads(json.loads(kwargs["input"])["text"])
            mutate(data["receipts"])
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"clean": json.dumps(data), "rules": []})
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mindie_bench.hf_sync.exporter.subprocess.run", run)


def test_public_trial_ids_are_stable_opaque_and_distinct(public_export_input):
    _, originals, _ = public_export_input
    rows = [public_row(row) for row in originals]
    assert [row["trial_id"] for row in rows] == [
        hashlib.sha256(row.trial_id.encode()).hexdigest() for row in originals
    ]
    assert len({row["trial_id"] for row in rows}) == len(originals)
    assert public_row(originals[0])["trial_id"] == rows[0]["trial_id"]
    assert all(row.trial_id not in json.dumps(rows) for row in originals)


@pytest.mark.parametrize(
    "damage",
    [
        "collapse_ids",
        "duplicate_public_id",
        "row_count",
        "schema_version",
        "evidence",
        "terminal_status",
        "ablation_group",
        "task_role",
        "repetition_type",
        "task_digest",
        "experiment_digest",
        "missing_nullable_field",
    ],
)
def test_redaction_damage_to_receipt_identity_never_publishes(
    tmp_path, monkeypatch, public_export_input, damage
):
    source, _, scanner = public_export_input

    def mutate(rows):
        if damage == "collapse_ids":
            for row in rows:
                row["trial_id"] = "[REDACTED_SECRET]"
        elif damage == "duplicate_public_id":
            rows[1]["trial_id"] = rows[0]["trial_id"]
        elif damage == "row_count":
            rows.pop()
        elif damage == "repetition_type":
            rows[0]["repetition"] = True
        elif damage == "missing_nullable_field":
            del rows[0]["cluster"]
        else:
            rows[0][damage] = {
                "schema_version": "2.0",
                "evidence": "verified",
                "terminal_status": "completed_pass",
                "ablation_group": "G3",
                "task_role": "seed",
                "task_digest": "[REDACTED_SECRET]",
                "experiment_digest": "[REDACTED_SECRET]",
            }[damage]

    mock_redaction(monkeypatch, mutate)
    output = tmp_path / "public"
    with pytest.raises(ValueError, match="redaction.*receipt"):
        export_hf(source, output, knowledge_python=Path(sys.executable), gitleaks=scanner)
    assert not output.exists()
    assert not list(tmp_path.glob(".mindie-export-*"))


def test_valid_public_ids_and_failure_states_survive_export_round_trip(
    tmp_path, monkeypatch, public_export_input
):
    import pyarrow.parquet as pq

    source, originals, scanner = public_export_input

    def redact(rows):
        rows.reverse()
        for row in rows:
            row["environment_fingerprint"]["remote_environment"]["kernel"] = "[REDACTED_PRIVATE_IP]"

    mock_redaction(monkeypatch, redact)
    output = tmp_path / "public"
    result = export_hf(source, output, knowledge_python=Path(sys.executable), gitleaks=scanner)
    assert result["rows"] == len(originals)
    rows = json.loads((output / "receipts.json").read_text())
    by_id = {row["trial_id"]: row for row in rows}
    expected = {hashlib.sha256(row.trial_id.encode()).hexdigest(): row for row in originals}
    assert by_id.keys() == expected.keys()
    for identity, row in by_id.items():
        original = expected[identity]
        assert row["terminal_status"] == original.terminal_status.value
        assert row["repetition"] == original.repetition
        assert row["task_role"] == original.task_role
        assert row["task_digest"] == original.task_digest
        assert row["environment_fingerprint"]["remote_environment"]["kernel"] == "[REDACTED_PRIVATE_IP]"
    assert all(row.trial_id not in (output / "receipts.json").read_text() for row in originals)
    parquet = pq.read_table(output / "trials.parquet")
    assert parquet.num_rows == len(originals)
    assert set(parquet.column("trial_id").to_pylist()) == by_id.keys()
    manifest = json.loads((output / "eval_manifest.json").read_text())
    assert manifest["attempts"] == len(originals)
    assert manifest["uploaded"] is False
    assert manifest["public_trial_id_scheme"] == "sha256-utf8-native-trial-id"
