import hashlib
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_metrics import receipt

from mindie_bench.hf_sync.exporter import export_hf, public_row
from mindie_bench.mining.knowledge_to_task import _toml_document, mine_knowledge
from mindie_bench.models import TaskSpec
from mindie_bench.runner.soundness import check_soundness
from mindie_bench.scorer.comparator import freeze, load_receipts


def test_task_serializer_roundtrips_multiple_private_oracle_probes():
    data = {
        "schema_version": "1.0",
        "oracle": {
            "protected_paths": ["golden.json"],
            "probes": [
                {"name": "first", "command": ["python", "{oracle}/verifier/first.py"]},
                {"name": "second", "command": ["python", "{oracle}/verifier/second.py"]},
            ],
        },
    }
    assert tomllib.loads(_toml_document(data)) == data


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
        if len(command) >= 3 and command[1] == "-I":
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


def test_export_hf_includes_public_tasks_and_optional_upload(
    case, tmp_path, monkeypatch, public_export_input
):
    from types import ModuleType

    import pyarrow.parquet as pq

    source, originals, scanner = public_export_input
    mock_redaction(monkeypatch, lambda rows: None)
    uploaded_calls = []
    hf_module = ModuleType("huggingface_hub")

    class FakeHfApi:
        def create_repo(self, **kwargs):
            uploaded_calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            uploaded_calls.append(("upload_folder", kwargs))

    hf_module.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
    output = tmp_path / "public-with-tasks"
    result = export_hf(
        source,
        output,
        knowledge_python=Path(sys.executable),
        gitleaks=scanner,
        tasks_root=case["task"],
        upload_repo="Ascend/mindie-bench-test",
    )
    assert result["uploaded"] is True
    assert result["tasks"] == 1
    assert result["repo_id"] == "Ascend/mindie-bench-test"
    assert len(uploaded_calls) == 2
    tasks_table = pq.read_table(output / "tasks.parquet")
    assert tasks_table.num_rows == 1
    assert tasks_table.column("instance_id").to_pylist() == ["synthetic-dtype"]
    readme = (output / "README.md").read_text()
    assert "tasks.parquet" in readme
    assert "Leaderboard Summary" in readme


def test_cli_matrix_alias_supports_multiple_tasks_flags(case, tmp_path, monkeypatch):
    import shutil

    from mindie_bench import cli
    from mindie_bench.io import write_json

    task_a = tmp_path / "task-a"
    task_b = tmp_path / "task-b"
    shutil.copytree(case["task"], task_a)
    shutil.copytree(case["task"], task_b)
    (task_b / "task.toml").write_text(
        (task_b / "task.toml").read_text().replace('instance_id = "synthetic-dtype"', 'instance_id = "synthetic-dtype-b"')
    )
    captured = {}

    def fake_run_matrix(paths, config, **kwargs):
        captured["paths"] = paths
        captured["kwargs"] = kwargs
        write_json(kwargs["output"] / "summary.json", {"cells": len(paths)})
        return []

    monkeypatch.setattr("mindie_bench.runner.orchestrator.run_matrix", fake_run_matrix)
    out_dir = tmp_path / "matrix-out"
    config_file = Path(__file__).resolve().parents[1] / "configs/example.toml"
    rc = cli.main(
        [
            "matrix",
            "--config",
            str(config_file),
            "--tasks",
            str(task_a),
            "--tasks",
            str(task_b),
            "--tier",
            "l2",
            "--groups",
            "G0,G2",
            "--repetitions",
            "1",
            "--output",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert captured["paths"] == [task_a, task_b]


def test_hardened_g4_rsi_verifiers_reject_constant_returning_dummy_modules(tmp_path):
    import os
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    for task_id, rel_mod, arms in (
        (
            "dsv4-sparse-mla-qli-fusion-rsi-zh",
            "vllm_ascend/models/deepseek_v4/sparse_mla_decode.py",
            ("full_candidate", "minus_q_fusion", "minus_view_alias", "minus_device_guard"),
        ),
        (
            "triton-spatial-norm-dim-merge-rsi-zh",
            "vllm_ascend/ops/spatial_group_norm_reduce.py",
            ("full_candidate", "minus_dim_merge", "minus_block_scaling", "minus_multibuffer_cast_guard"),
        ),
    ):
        ws = tmp_path / task_id
        ws.mkdir()
        (ws / "experiment-ledger.jsonl").write_text(" ".join(arms) + "\n")
        (ws / "PROGRESS.md").write_text("# Progress summary with at least forty characters here.\n")
        mod_path = ws / rel_mod
        mod_path.parent.mkdir(parents=True, exist_ok=True)
        mod_path.write_text(
            '_ = "fused_q_dequant_rmsnorm_rope_split"\n'
            "def select_optimal_block_size(n, d, b=3): return 4096\n"
            "def run_ablation_suite():\n"
            "    return {\n"
            '        "full_candidate": {"tensor_move_count": 0, "host_d2h_sync_count": 0, "tpot_ms": 1.0, "nested_loop_depth": 1, "block_size": 4096, "enable_multibuffer": False, "intermediate_gm_copies": 0, "correctness_ok": True, "latency_ms": 1.0},\n'
            '        "minus_q_fusion": {"tensor_move_count": 3, "tpot_ms": 2.0},\n'
            '        "minus_view_alias": {"tensor_move_count": 2, "tpot_ms": 2.0},\n'
            '        "minus_device_guard": {"host_d2h_sync_count": 1, "invalid_gm_triggered": True},\n'
            '        "minus_dim_merge": {"nested_loop_depth": 3, "latency_ms": 2.0},\n'
            '        "minus_block_scaling": {"block_size": 512, "latency_ms": 2.0},\n'
            '        "minus_multibuffer_cast_guard": {"enable_multibuffer": True, "intermediate_gm_copies": 1, "latency_ms": 2.0},\n'
            "    }\n"
        )
        verifier_sh = repo_root / "tasks/zh-vague" / task_id / "verifier/test.sh"
        proc = subprocess.run(
            ["/bin/bash", str(verifier_sh)],
            cwd=ws,
            env=dict(os.environ, MINDIE_BENCH_WORKSPACE=str(ws)),
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        payload = json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1])
        assert payload["reward"] == 0.0


# Marker strings and stubs from the 2026-10-23 audit that the former grep/AST judges scored 1.0.
STRING_ONLY_ARTIFACTS = {
    "vllm_ascend/models/deepseek_v4/model.py":
        'required_names = "use_eagle() uses_draft_model() is_last_rank _mtp_hidden_buffer"\n',
    "vllm_ascend/attention/dsa_v1.py": 'def build_for_drafting(self):\n    return "self.storage_block_size"\n',
    "vllm_ascend/worker/model_runner_v1.py": (
        "def execute_model(self):\n    self._dummy_run(1, skip_gdn_state_update=True)\n"
        "def _dummy_run(self, skip_gdn_state_update=False):\n    pass\n"
        "def _build_attention_metadata(self, skip_gdn_state_update=False):\n    pass\n"
        'required_names = "num_actual_tokens=0 max_query_len=0 zeros_like ones_like"\n'
    ),
    "vllm_ascend/_310p/model_runner_310p.py": (
        "def _dummy_run(self, skip_gdn_state_update=False):\n"
        "    missing(skip_gdn_state_update=skip_gdn_state_update)\n"
    ),
    "vllm_ascend/worker/block_table.py": 'required_names = "num_speculative_tokens - 1 self.max_num_reqs"\n',
    "vllm_ascend/spec_decode/llm_base_proposer.py":
        'required_names = "num_speculative_tokens - 1 2 * self.runner.max_num_reqs"\n',
    "vllm_ascend/attention/sfa_v1.py": (
        "def _build():\n    return \"draft_index spec_actual_seq_lengths_query[draft_index - 1] "
        'spec_actual_seq_lengths_key[draft_index - 1]"\n'
    ),
}


@pytest.mark.parametrize(
    "task_id",
    [
        "dspark-mrv2-drafter-crash-zh-vague",
        "l3-dspark-mrv2-drafter-crash-zh-vague",
        "k3-idle-dp-gdn-state-zh-vague",
        "mtp-draft-slot-mapping-capacity-zh-vague",
        "sfa-mtp-seq-lens-eval-zh-vague",
        "sfa-mtp-seq-lens-seed-zh-vague",
    ],
)
def test_behavior_verifiers_reject_string_only_artifacts(tmp_path, task_id):
    import os
    import subprocess

    task = Path(__file__).resolve().parents[1] / "tasks/zh-vague" / task_id
    for relative in tomllib.loads((task / "task.toml").read_text())["mindie_knowledge"]["target_files"]:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(STRING_ONLY_ARTIFACTS[relative])
    proc = subprocess.run(
        ["/bin/bash", str(task / "verifier/test.sh")],
        cwd=tmp_path,
        env=dict(os.environ, MINDIE_BENCH_WORKSPACE=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1, proc.stderr
    payload = json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1])
    assert payload["reward"] == 0.0
    assert "is not defined at module level" in payload["failure_reason"]
