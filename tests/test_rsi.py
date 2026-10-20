"""RSI task / Pier receipt contracts and task-balanced scoring; no GPU or model calls."""

from __future__ import annotations

import json
import subprocess
from uuid import NAMESPACE_URL, uuid5

import pytest

from mindie_bench.cli import main
from mindie_bench.integrations.rsi import catalog, execute_plan, plan_run, read_task
from mindie_bench.scorer.rsi import classify, load_trials, score_results, summarize_trials


def task(task_id="support_example", category="model_support"):
    return {
        "id": task_id, "name": f"inference-bench/{task_id}", "category": category,
        "track": "model_support" if category == "model_support" else "optimization",
        "engine": "sglang",
    }


def write_trial(root, spec, *, trial_id="attempt-1", reward=None, exception=None, metrics=None):
    """The published Pier TrialResult + verifier directory layout (synthetic observations)."""
    folder = root / trial_id
    (folder / "verifier").mkdir(parents=True)
    receipt = {
        "id": str(uuid5(NAMESPACE_URL, trial_id)),
        "trial_name": f"{spec['id']}__{trial_id}", "task_name": spec["name"],
        "trial_uri": folder.as_uri(), "task_id": {"path": f"/tasks/{spec['id']}"},
        "config": {"task": {"path": f"/tasks/{spec['id']}"},
                   "agent": {"name": "test-agent", "model_name": "test-model"},
                   "environment": {"type": "docker"}, "timeout_multiplier": 1.0},
        "task_checksum": "f" * 64, "finished_at": "2026-10-20T00:00:00Z",
        "agent_info": {"name": "test-agent", "version": "1", "model_info": {"name": "test-model"}},
        "exception_info": exception, "verifier_result": {"rewards": reward},
    }
    (folder / "result.json").write_text(json.dumps(receipt), encoding="utf-8")
    if reward is not None:
        (folder / "verifier/reward.json").write_text(json.dumps(reward), encoding="utf-8")
    if metrics is not None:
        (folder / "verifier/metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return folder


def inventory(tasks):
    return {
        "tasks": tasks, "source_commit": "a" * 40, "source_dirty": False,
        "catalog_sha256": "b" * 64, "upstream": "https://github.com/apple-aiml-research/ml-rsi-inference-bench",
    }


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "rsi checkout"
    folder = root / "tasks/support_example"
    for name in ("instruction.md", "environment/Dockerfile", "tests/test.sh", "task.toml"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (folder / "task.toml").write_text('''schema_version = "1.2"
[task]
name = "inference-bench/support_example"
[agent]
timeout_sec = 30000
[environment]
build_timeout_sec = 7200
gpus = 4
gpu_types = ["H100"]
allow_internet = false
[verifier]
timeout_sec = 55252
environment_mode = "separate"
[metadata]
repository_url = "https://github.com/sgl-project/sglang"
base_commit_hash = "b525f9e2acb1af15ecf986c0f7539fcbf229c0e9"
model_repository = "openai/gpt-oss-120b"
[metadata.inference_bench]
category = "model_support"
engine = "sglang"
score_profile = "model_support_upstream_acceptance_v60"
''', encoding="utf-8")
    pier = root / "third_party/pier/pyproject.toml"
    pier.parent.mkdir(parents=True)
    pier.write_text("", encoding="utf-8")
    for args in (
        ["init"], ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, timeout=10)
    return root


def test_catalog_and_execution_plan_preserve_upstream_boundaries(checkout, tmp_path, monkeypatch):
    data = catalog(checkout)
    entry = data["tasks"][0]
    assert entry["hardware"]["gpu_types"] == ["H100"]
    assert entry["target_revision"] == "b525f9e2acb1af15ecf986c0f7539fcbf229c0e9"
    assert entry["verifier_environment_mode"] == "separate"
    assert entry["agent_timeout_sec"] == 30000
    plan = plan_run(checkout, task_id=entry["id"], agent="codex", model="test/model",
                    output=tmp_path / "out", attempts=3)
    argv = plan["command"]
    assert argv[argv.index("--path") + 1] == str(checkout / "tasks/support_example")
    assert argv[argv.index("--n-attempts") + 1] == "3"
    assert argv[argv.index("--max-retries") + 1] == "0"
    assert argv[argv.index("--n-concurrent") + 1] == "1"
    assert not (tmp_path / "out").exists()
    calls = []
    monkeypatch.setattr("os.execvpe", lambda executable, command, env:
                        calls.append((executable, command, env["PYTHONUTF8"])))
    execute_plan(plan)
    assert calls == [("uv", argv, "1")]
    assert json.loads((tmp_path / "out/run-plan.json").read_text())["source_commit"] == data["source_commit"]
    with pytest.raises(FileExistsError):
        execute_plan(plan)


def test_catalog_reads_optimization_metadata_layout(checkout):
    path = checkout / "tasks/support_example/task.toml"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('''
[metadata.inference_bench.target]
repo = "https://github.com/vllm-project/vllm.git"
commit = "v0.20.1"
[metadata.inference_bench.model]
hf_model_id = "Qwen/Qwen3.6-27B"
''')
    entry = read_task(path.parent)
    assert entry["target_revision"] == "v0.20.1"
    assert entry["model"] == "Qwen/Qwen3.6-27B"
    with pytest.raises(ValueError, match="clean checkout"):
        plan_run(checkout, task_id=entry["id"], agent="codex", model="test", output=checkout / "out")


def test_support_partial_pass_and_text_void_precedence(tmp_path):
    spec = task()
    write_trial(tmp_path, spec, reward={"reward": 0.99})
    folder = write_trial(tmp_path, spec, trial_id="void", reward={"reward": 1})
    (folder / "verifier/reward.txt").write_text("-1", encoding="utf-8")
    rows = load_trials(tmp_path, inventory([spec]))
    assert [(r["status"], r["score"]) for r in rows] == [
        ("completed_fail", 0), ("infra_failed", None),
    ]
    report = summarize_trials(rows, [spec], expected_attempts=2)
    assert report["tracks"]["model_support"]["score"] is None
    assert report["tracks"]["model_support"]["observed_score"] == 0


@pytest.mark.parametrize("rewards,metrics,status,score", [
    ({"reward": 1, "correctness": 1, "speedup_geomean": 2}, {}, "completed_pass", 2),
    ({"reward": 0, "correctness": 1, "speedup_geomean": 0.5}, {}, "completed_fail", 0.5),
    ({"reward": 0, "correctness": 0}, {}, "completed_fail", 1),
    ({"reward": 0, "correctness": 1, "speedup_geomean": None},
     {"speedup_geomean": 90, "reason": "insufficient_data"}, "completed_fail", 1),
    ({"reward": 0, "correctness": 0, "infra_void": 1}, {}, "infra_failed", None),
    ({"reward": 0, "correctness": 1, "speedup_geomean": 1.01},
     {"reason": "no_speedup"}, "completed_fail", 1.01),
])
def test_measured_ratios_correctness_and_voids(rewards, metrics, status, score):
    result = classify(task("regression", "regression"), {"finished_at": "now"}, rewards, metrics)
    assert result["status"] == status
    assert result["score"] == score


def test_missing_receipts_and_harness_errors_remain_unknown(tmp_path):
    spec = task()
    write_trial(tmp_path, spec)
    write_trial(tmp_path, spec, trial_id="error", reward={"reward": 1},
                exception={"exception_type": "DownloadVerifierDirError"})
    rows = load_trials(tmp_path, inventory([spec]))
    assert {row["status"] for row in rows} == {"unknown"}
    assert all(row["score"] is None for row in rows)


def test_task_weighting_cohorts_and_coverage(tmp_path):
    specs = [task("a"), task("b"), task("c", "regression"), task("d", "throughput_optimization")]
    for i in range(3):
        write_trial(tmp_path, specs[0], trial_id=f"a{i}", reward={"reward": 1})
    write_trial(tmp_path, specs[1], trial_id="b", reward={"reward": 0.9})
    write_trial(tmp_path, specs[2], trial_id="c", reward={"reward": 3, "correctness": 1, "speedup_geomean": 4})
    write_trial(tmp_path, specs[3], trial_id="d", reward={"reward": 0, "correctness": 1, "speedup_geomean": 0.25})
    report = score_results(tmp_path, inventory(specs))
    tracks = report["cohorts"][0]["tracks"]
    assert tracks["model_support"]["observed_score"] == 0.5  # Not pooled 3/4.
    assert tracks["model_support"]["complete"] is False  # Extra attempts are visible.
    assert tracks["optimization"]["score"] == pytest.approx(1)
    assert report["ablation_group"] is None
    assert len(report["trials"][0]["evidence_sha256"]) == 2
    with pytest.raises(ValueError, match="outside the declared"):
        score_results(tmp_path, inventory(specs), task_ids=["a"])


def test_repeated_optimization_attempts_are_balanced_by_task(tmp_path):
    specs = [task("a", "regression"), task("b", "regression"), task("missing")]
    for i, ratio in enumerate([4, 1]):
        write_trial(tmp_path, specs[0], trial_id=f"a{i}",
                    reward={"reward": ratio - 1, "correctness": 1, "speedup_geomean": ratio})
    write_trial(tmp_path, specs[1], trial_id="b", reward={"reward": 0, "correctness": 0})
    report = score_results(tmp_path, inventory(specs))
    tracks = report["cohorts"][0]["tracks"]
    assert tracks["optimization"]["observed_score"] == pytest.approx(2 ** 0.5)
    assert tracks["optimization"]["score"] is None
    assert tracks["model_support"]["tasks_observed"] == 0


def test_conflicting_evidence_cannot_be_scored(tmp_path):
    spec = task()
    folder = write_trial(tmp_path, spec, reward={"reward": 1})
    (folder / "verifier/reward.json").write_text('{"reward": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_trials(tmp_path, inventory([spec]))


def test_text_reward_cannot_turn_a_recorded_failure_into_a_pass(tmp_path):
    spec = task()
    folder = write_trial(tmp_path, spec, reward={"reward": 0})
    (folder / "verifier/reward.txt").write_text("1", encoding="utf-8")
    with pytest.raises(ValueError, match="text reward disagree"):
        load_trials(tmp_path, inventory([spec]))


def test_duplicate_trials_and_mixed_revisions_are_not_extra_evidence(tmp_path):
    spec = task()
    first = write_trial(tmp_path, spec, reward={"reward": 1})
    second = write_trial(tmp_path, spec, trial_id="second", reward={"reward": 0})
    path = second / "result.json"
    receipt = json.loads(path.read_text())
    receipt["id"] = json.loads((first / "result.json").read_text())["id"]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate Pier trial"):
        score_results(tmp_path, inventory([spec]))
    receipt["id"] = "second"
    receipt["task_checksum"] = "e" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="mixed task revisions"):
        score_results(tmp_path, inventory([spec]))


def test_distinct_model_cohorts_do_not_share_coverage(tmp_path):
    specs = [task("a"), task("b")]
    write_trial(tmp_path, specs[0], trial_id="a", reward={"reward": 1})
    folder = write_trial(tmp_path, specs[1], trial_id="b", reward={"reward": 1})
    path = folder / "result.json"
    receipt = json.loads(path.read_text())
    receipt["agent_info"]["model_info"]["name"] = "another-model"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    report = score_results(tmp_path, inventory(specs))
    assert len(report["cohorts"]) == 2
    assert all(row["tracks"]["model_support"]["score"] is None for row in report["cohorts"])


def test_same_model_with_different_execution_budgets_is_not_pooled(tmp_path):
    spec = task()
    write_trial(tmp_path, spec, trial_id="a", reward={"reward": 1})
    folder = write_trial(tmp_path, spec, trial_id="b", reward={"reward": 0})
    path = folder / "result.json"
    receipt = json.loads(path.read_text())
    receipt["config"]["timeout_multiplier"] = 2.0
    receipt["config"]["agent"]["env"] = {"API_KEY": "synthetic-secret-do-not-export"}
    path.write_text(json.dumps(receipt), encoding="utf-8")
    report = score_results(tmp_path, inventory([spec]))
    assert len(report["cohorts"]) == 2
    assert "synthetic-secret-do-not-export" not in json.dumps(report)


def test_cli_catalog_plan_score_roundtrip(checkout, tmp_path, capsys):
    catalog_file = tmp_path / "catalog.json"
    assert main(["rsi-catalog", "--source", str(checkout), "--output", str(catalog_file)]) == 0
    capsys.readouterr()
    assert main(["rsi-run", "--source", str(checkout), "--task", "support_example",
                 "--agent", "codex", "--model", "test", "--output", str(tmp_path / "run"),
                 "--dry-run"]) == 0
    assert not (tmp_path / "run").exists()
    capsys.readouterr()
    write_trial(tmp_path / "receipts", task(), reward={"reward": 1})
    output = tmp_path / "score.json"
    assert main(["rsi-score", "--source", str(checkout), "--input", str(tmp_path / "receipts"),
                 "--output", str(output)]) == 0
    assert json.loads(output.read_text())["cohorts"][0]["tracks"]["model_support"]["score"] == 1
