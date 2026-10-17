import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.models import CorpusConfig, Group, GroupConfig, MCPServer
from mindie_bench.runner.orchestrator import check_read_boundaries, run_matrix, run_trial, runtime_paths
from mindie_bench.runner.profile_builder import build_profile
from mindie_bench.runner.remote_slot import POSTFLIGHT, WORKER, RemoteSlot
from mindie_bench.runner.sandbox import native_command
from mindie_bench.scorer.comparator import grouped_summaries


def test_broad_runtime_grants_cannot_expose_answer_or_home(tmp_path):
    with pytest.raises(ValueError, match="protected"):
        check_read_boundaries([tmp_path], [tmp_path / "oracle"])
    with pytest.raises(ValueError, match="protected"):
        check_read_boundaries([Path.home()], [Path.home() / ".codex"])
    with pytest.raises(ValueError, match="protected"):
        check_read_boundaries([tmp_path / "oracle/gold.diff"], [tmp_path / "oracle"])


def test_direct_network_cannot_bypass_github_shim(case):
    trial = case["root"] / "sandbox"
    trial.mkdir()
    profile = build_profile(trial, case["spec"], Group.G0, case["config"])
    code = (
        "import socket\n"
        "try:\n"
        " socket.create_connection(('1.1.1.1',443),timeout=1)\n"
        "except OSError:\n"
        " print('blocked')\n"
        "else:\n"
        " raise SystemExit('NETWORK_ISOLATION_BROKEN')\n"
    )
    command = native_command(
        [sys.executable, "-I", "-c", code],
        read=runtime_paths(case["config"], profile),
        write=[profile.root],
        cwd=profile.root,
        policy_path=trial / "network.sb",
    )
    result = subprocess.run(
        command, env=profile.env, cwd=profile.root, capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "blocked"


def test_untrusted_stdlib_name_cannot_run_in_pre_sandbox_owner(case):
    from conftest import git

    sentinel = case["root"] / "escaped-owner"
    (case["repo"] / "operator.py").write_text(
        f"open({str(sentinel)!r},'w').write('PRE_SANDBOX_EXECUTION')\nraise RuntimeError('shadow')\n"
    )
    git(case["repo"], "add", ".")
    git(
        case["repo"],
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "hostile stdlib name in business source",
    )
    commit = git(case["repo"], "rev-parse", "HEAD")
    manifest = case["task"] / "task.toml"
    manifest.write_text(manifest.read_text().replace(case["commit"], commit))
    output = case["root"] / "run"
    output.mkdir()
    row = run_trial(
        case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=output
    )
    assert row.terminal_status.value == "completed_pass", row.model_dump_json(indent=2)
    assert not sentinel.exists()


def test_g3_seed_eval_temporal_and_repetition_isolation(case, corpus):
    source, commit = corpus
    seed = case["task"]
    evaluation = case["root"] / "eval-task"
    shutil.copytree(seed, evaluation)
    for root, role, identity in [
        (seed, "seed", "synthetic-seed"),
        (evaluation, "eval", "synthetic-eval"),
    ]:
        manifest = root / "task.toml"
        text = manifest.read_text().replace(
            'instance_id = "synthetic-dtype"', f'instance_id = "{identity}"'
        )
        text = text.replace(
            'evidence = "synthetic"', f'evidence = "synthetic"\nrole = "{role}"\ncluster = "family"'
        )
        text = text.replace('knowledge_gradient = "analogical"', 'knowledge_gradient = "near_exact"')
        manifest.write_text(text)
    capture = case["root"] / "capture"
    capture.write_text(
        f"#!{sys.executable} -I\n"
        + """
import json,pathlib,sys
request=json.load(sys.stdin)
root=pathlib.Path(request["destination"])/"tasks"/("captured-"+request["instance_id"])
(root/"materials").mkdir(parents=True)
(root/"PACKAGE.md").write_text("# Synthetic captured advice\\n")
(root/"materials/0001.md").write_text("Synthetic locally captured public text.\\n")
print(json.dumps({"status":"completed","tokens":{"total":7,"complete":True},"estimated_cost_usd":0}))
"""
    )
    capture.chmod(0o700)
    config = case["config"]
    config.runtime_read_paths = [str(capture)]
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    config.groups[Group.G3] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(command="/usr/bin/true", component="knowledge"),
        },
        capture_command=[str(capture)],
    )
    rows = run_matrix(
        [seed, evaluation],
        config,
        tier="l2",
        groups=[Group.G3],
        repetitions=2,
        concurrency=2,
        output=case["root"] / "g3",
    )
    assert len(rows) == 4
    for row in rows:
        assert row.terminal_status.value == "completed_pass", row.model_dump_json(indent=2)
        view = Path(row.artifacts["captured_corpus"]) / "tasks"
        assert not (view / "origin").exists()
        if row.task_role == "seed":
            assert not (view / "captured-synthetic-eval").exists()
        else:
            assert (view / "captured-synthetic-seed").is_dir()
        assert row.primary_metrics["tokens"]["indexer_small_model_total"] == 7
    summaries = grouped_summaries(rows)
    assert summaries["l2/synthetic/G3"]["attempts"] == 2
    assert summaries["l2/synthetic/G3"]["seed_setup_attempts"] == 2
    assert summaries["l2/synthetic/G3/seed"]["attempts"] == 2


def test_remote_worker_scripts_compile_and_do_not_kill_unrelated_names():
    compile(WORKER, "<remote-worker>", "exec")
    compile(POSTFLIGHT, "<remote-postflight>", "exec")
    assert "pkill" not in POSTFLIGHT
    assert "MINDIE_BENCH_SLOT_ID=" in POSTFLIGHT
    assert "pidfd_send_signal" in POSTFLIGHT


def test_remote_probe_requires_exact_model_and_idle_device(case):
    from mindie_bench.models import RemoteConfig, SourceEnvironment

    config = RemoteConfig(
        host="bench-host",
        container="bench",
        repo_path="/repo",
        devices=[0],
        probe_command=["/probe-json"],
    )
    environment = SourceEnvironment(
        repo_path=str(case["repo"]), base_commit=case["commit"], npu_model="Ascend910B", npu_count=1
    )
    slot = RemoteSlot(config, environment, "test", [0], 10)
    environment_ids = {
        name: "synthetic" for name in ["image_digest", "cann", "torch_npu", "driver", "python", "kernel"]
    }
    valid = {"0": {"used_mb": 0, "pids": [], "model": "Ascend910B"}, "_environment": environment_ids}
    assert slot.validate_probe(valid) == valid
    for invalid in [
        {"0": {"used_mb": 800, "pids": [], "model": "Ascend910B"}},
        {"0": {"used_mb": 0, "pids": [123], "model": "Ascend910B"}},
        {"0": {"used_mb": 0, "pids": [], "model": "other"}},
        {"0": {"used_mb": "0", "pids": [], "model": "Ascend910B"}},
    ]:
        invalid["_environment"] = environment_ids
        with pytest.raises((ValueError, RuntimeError)):
            slot.validate_probe(invalid)
