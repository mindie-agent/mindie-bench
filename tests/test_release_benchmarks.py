import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench import release
from mindie_bench.cli import main
from mindie_bench.release import collect_release


def commit(repo, message="fixture"):
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run([
        "git", "-c", "user.name=Benchmark", "-c", "user.email=benchmark@example.invalid",
        "commit", "-qm", message,
    ], cwd=repo, check=True, capture_output=True)


@pytest.fixture(scope="session")
def benchmark_template(tmp_path_factory):
    template = tmp_path_factory.mktemp("release-template")
    repo = template / "candidate"
    repo.mkdir()
    (repo / "measure.py").write_text(
        """import json, os, sys, time
from pathlib import Path
mode = sys.argv[1]
if mode == "timeout":
    time.sleep(10)
print("private-transcript-and-secret")
payload = {"status": "pass", "metrics": [
    {"name": "latency", "unit": "us/op", "value": 2.5, "iterations": 1000}]}
if mode == "unknown":
    payload = {"status": "unknown", "reason_code": "remote_receipt_unavailable"}
if mode == "unsupported":
    payload = {"status": "unsupported", "reason_code": "hardware_unavailable"}
if mode != "missing":
    Path(os.environ["MINDIE_BENCH_RESULT_PATH"]).write_text(json.dumps(payload))
if mode == "nonzero":
    sys.exit(7)
""",
        encoding="utf-8",
    )
    runner = template / "runner"
    package = Path(release.__file__).resolve().parent
    shutil.copytree(package, runner / "src/mindie_bench", ignore=shutil.ignore_patterns("__pycache__"))
    (runner / ".gitignore").write_text("__pycache__/\n")
    for checkout in (repo, runner):
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True, capture_output=True)
        commit(checkout)
    return template


@pytest.fixture
def workspace(tmp_path, benchmark_template):
    repo = tmp_path / "repo"
    shutil.copytree(benchmark_template / "candidate", repo)
    shutil.copytree(benchmark_template / "runner", tmp_path / "runner")
    return repo


def run(tmp_path, workspace, mode="pass", *, required=True, kind="microbenchmark"):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({
        "schema_version": "mindie-release-suite/1", "suite": "component",
        "publishable": True, "cases": [{
            "id": "latency", "kind": kind, "required": required, "repetitions": 3,
            "timeout_seconds": 0.05 if mode == "timeout" else 5,
            "command": (["missing-benchmark-executable"]
                        if mode == "not_started" else ["{python}", "measure.py", mode]),
            "side_effects": "external_write" if mode in {"unknown", "not_started"} else "local",
        }],
    }))
    # Load the actual runner from a committed checkout in a fresh interpreter.
    # The candidate and runner have separate Git identities and dirty states.
    subprocess.run([
        sys.executable, "-c",
        "import sys; from pathlib import Path; "
        "from mindie_bench.release import run_release_suite; "
        "run_release_suite(repo=Path(sys.argv[1]), repository='mindie-agent/example', "
        "version='0.0.2', suite=Path(sys.argv[2]), output=Path(sys.argv[3]), "
        "python=sys.executable)",
        str(workspace), str(suite), str(tmp_path / "output"),
    ], check=True, capture_output=True, text=True, timeout=15,
        env=dict(os.environ, PYTHONPATH=str(tmp_path / "runner/src")))
    return json.loads((tmp_path / "output/benchmark-results.json").read_text())


def manifest(tmp_path, report, **overrides):
    target = {
        "repository": report["repository"], "revision": report["source"]["revision"],
        "version": report["version"], "required_cases": ["latency"],
        "platforms": [sys.platform],
    }
    if report["schema_version"] == "mindie-release-benchmark/2":
        target["runner_revision"] = subprocess.check_output(
            ["git", "-C", str(tmp_path / "runner"), "rev-parse", "HEAD"], text=True,
        ).strip()
    target.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": "mindie-release-manifest/1",
        "product_version": "0.0.2", "repositories": [target],
    }))
    return path


def test_real_process_samples_bind_source_and_exclude_private_logs(tmp_path, workspace):
    report = run(tmp_path, workspace)
    metric = report["cases"][0]["metrics"][0]
    assert report["release_gate"]
    assert metric["samples"] == [2.5, 2.5, 2.5]
    assert metric["iterations"] == [1000, 1000, 1000]
    assert len(report["source"]["revision"]) == 40
    runner_head = subprocess.check_output(
        ["git", "-C", str(tmp_path / "runner"), "rev-parse", "HEAD"], text=True,
    ).strip()
    assert report["tool"]["source"] == {"revision": runner_head, "dirty": False}
    assert report["tool"]["source_after"] == report["tool"]["source"]
    public = (tmp_path / "output" / "benchmark-results.json").read_text()
    assert "private-transcript-and-secret" not in public
    assert str(tmp_path) not in public
    assert "private-transcript-and-secret" in (
        tmp_path / "output/private/latency/1/stdout.log"
    ).read_text()
    collection = collect_release(
        manifest=manifest(tmp_path, report, runner_revision=runner_head), inputs=[tmp_path / "output"],
        output=tmp_path / "collection",
    )
    assert collection["release_gate"]


@pytest.mark.parametrize("mode,reason", [
    ("nonzero", "nonzero_exit"), ("missing", "missing_measurement"),
    ("timeout", "execution_timeout"), ("unsupported", "hardware_unavailable"),
])
def test_required_failed_or_missing_measurements_block_release(tmp_path, workspace, mode, reason):
    report = run(tmp_path, workspace, mode)
    assert not report["release_gate"]
    assert report["cases"][0]["attempts"][0]["reason_code"] == reason
    assert report["cases"][0]["repetitions_completed"] == 1
    if mode == "timeout":
        assert report["cases"][0]["status"] == "fail"


def test_unknown_external_outcome_is_preserved_without_repeat(tmp_path, workspace):
    report = run(tmp_path, workspace, "unknown")
    assert not report["release_gate"]
    assert report["cases"][0]["status"] == "unknown"
    assert report["cases"][0]["repetitions_completed"] == 1
    assert not (tmp_path / "output/private/latency/2").exists()


def test_unstarted_external_command_is_known_failure_not_unknown(tmp_path, workspace):
    report = run(tmp_path, workspace, "not_started")
    case = report["cases"][0]
    assert not report["release_gate"]
    assert case["status"] == "fail"
    assert case["attempts"][0]["reason_code"] == "command_not_started"


def test_dirty_checkout_cannot_be_presented_as_release_revision(tmp_path, workspace):
    (workspace / "measure.py").write_text((workspace / "measure.py").read_text() + "\n")
    report = run(tmp_path, workspace)
    assert not report["release_gate"]
    assert "dirty_checkout" in report["blocking_reasons"]
    assert report["cases"][0]["status"] == "pass"


def test_dirty_runner_blocks_clean_candidate_release(tmp_path, workspace):
    runner_file = tmp_path / "runner/src/mindie_bench/release.py"
    runner_file.write_text(runner_file.read_text() + "\n# Uncommitted runner edit.\n")
    report = run(tmp_path, workspace)
    assert not report["release_gate"]
    assert "dirty_runner_checkout" in report["blocking_reasons"]
    assert report["source"] == report["source_after"]
    assert not report["source"]["dirty"]
    assert report["cases"][0]["status"] == "pass"


def test_runner_commit_change_during_measurement_blocks_release(tmp_path, workspace):
    measure = workspace / "measure.py"
    measure.write_text(measure.read_text() + """
import subprocess
runner = Path(os.environ["PYTHONPATH"]).parent
changed = runner / "revision.txt"
changed.write_text(changed.read_text() + "next\\n" if changed.exists() else "first\\n")
subprocess.run(["git", "add", "revision.txt"], cwd=runner, check=True)
subprocess.run(["git", "-c", "user.name=Benchmark", "-c",
                "user.email=benchmark@example.invalid", "commit", "-qm", "changed"],
               cwd=runner, check=True)
""")
    commit(workspace, "change runner during workload")
    report = run(tmp_path, workspace)
    assert not report["release_gate"]
    assert "runner_changed_during_run" in report["blocking_reasons"]
    assert not report["tool"]["source"]["dirty"]
    assert not report["tool"]["source_after"]["dirty"]
    assert report["source"] == report["source_after"]
    assert report["cases"][0]["status"] == "pass"


def test_runner_without_git_provenance_cannot_publish(tmp_path, workspace):
    shutil.rmtree(tmp_path / "runner/.git")
    report = run(tmp_path, workspace)
    assert report["tool"]["source"] is None
    assert "runner_source_unavailable" in report["blocking_reasons"]
    assert not report["release_gate"]


def test_collection_requires_selected_runner_but_preserves_legacy_reports(tmp_path, workspace):
    report = run(tmp_path, workspace)
    expected_runner = report["tool"]["source"]["revision"]
    missing_manifest = manifest(tmp_path, report)
    missing_value = json.loads(missing_manifest.read_text())
    del missing_value["repositories"][0]["runner_revision"]
    missing_manifest.write_text(json.dumps(missing_value))
    missing = collect_release(
        manifest=missing_manifest, inputs=[tmp_path / "output"],
        output=tmp_path / "missing-runner",
    )
    assert not missing["release_gate"]
    assert missing["blocking_reasons"] == ["missing_runner_selection:mindie-agent/example"]
    result = collect_release(
        manifest=manifest(tmp_path, report, runner_revision="a" * 40),
        inputs=[tmp_path / "output"], output=tmp_path / "wrong-runner",
    )
    assert result["blocking_reasons"] == ["stale_or_missing_runner_source:mindie-agent/example"]
    report["schema_version"] = "mindie-release-benchmark/1"
    del report["tool"]["source"], report["tool"]["source_after"]
    (tmp_path / "output/benchmark-results.json").write_text(json.dumps(report))
    legacy = collect_release(
        manifest=manifest(tmp_path, report), inputs=[tmp_path / "output"],
        output=tmp_path / "legacy",
    )
    assert legacy["release_gate"]
    pinned = collect_release(
        manifest=manifest(tmp_path, report, runner_revision=expected_runner),
        inputs=[tmp_path / "output"], output=tmp_path / "legacy-pinned",
    )
    assert not pinned["release_gate"]
    assert pinned["blocking_reasons"] == ["stale_or_missing_runner_source:mindie-agent/example"]


def test_collection_blocks_stale_revision_and_missing_platform_and_case(tmp_path, workspace):
    report = run(tmp_path, workspace)
    collection = collect_release(
        manifest=manifest(
            tmp_path, report, revision="a" * 40,
            required_cases=["latency", "native-host"], platforms=[sys.platform, "another-platform"],
        ),
        inputs=[tmp_path / "output"], output=tmp_path / "collection",
    )
    assert not collection["release_gate"]
    assert len(collection["blocking_reasons"]) == 3


def test_missing_repository_evidence_blocks_cli(tmp_path, workspace, capsys):
    report = run(tmp_path, workspace)
    path = manifest(tmp_path, report, platforms=["missing"])
    code = main([
        "release-collect", "--manifest", str(path), "--input", str(tmp_path / "output"),
        "--output", str(tmp_path / "collection"),
    ])
    assert code == 1
    assert not json.loads(capsys.readouterr().out)["release_gate"]


def test_collection_rejects_success_label_on_unknown_external_outcome(tmp_path, workspace):
    report = run(tmp_path, workspace, "unknown")
    report["release_gate"] = True
    report["blocking_reasons"] = []
    (tmp_path / "output/benchmark-results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="release gate contradicts"):
        collect_release(
            manifest=manifest(tmp_path, report), inputs=[tmp_path / "output"],
            output=tmp_path / "collection",
        )


def test_collection_rejects_performance_summary_changed_from_raw_samples(tmp_path, workspace):
    report = run(tmp_path, workspace)
    report["cases"][0]["metrics"][0]["median"] = 0.001
    (tmp_path / "output/benchmark-results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metric summary disagrees"):
        collect_release(
            manifest=manifest(tmp_path, report), inputs=[tmp_path / "output"],
            output=tmp_path / "collection",
        )
