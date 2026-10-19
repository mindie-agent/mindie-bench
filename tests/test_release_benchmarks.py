import json
import shutil
import subprocess
import sys

import pytest

from mindie_bench.cli import main
from mindie_bench.release import collect_release, run_release_suite


@pytest.fixture(scope="session")
def benchmark_template(tmp_path_factory):
    repo = tmp_path_factory.mktemp("release-template")
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
    for args in [
        ["init", "-q"], ["add", "."],
        ["-c", "user.name=Benchmark", "-c", "user.email=benchmark@example.invalid",
         "commit", "-qm", "fixture"],
    ]:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


@pytest.fixture
def workspace(tmp_path, benchmark_template):
    repo = tmp_path / "repo"
    shutil.copytree(benchmark_template, repo)
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
    return run_release_suite(
        repo=workspace, repository="mindie-agent/example", version="0.0.2",
        suite=suite, output=tmp_path / "output", python=sys.executable,
    )


def manifest(tmp_path, report, **overrides):
    target = {
        "repository": report["repository"], "revision": report["source"]["revision"],
        "version": report["version"], "required_cases": ["latency"],
        "platforms": [sys.platform], **overrides,
    }
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
    public = (tmp_path / "output" / "benchmark-results.json").read_text()
    assert "private-transcript-and-secret" not in public
    assert "private-transcript-and-secret" in (
        tmp_path / "output/private/latency/1/stdout.log"
    ).read_text()
    collection = collect_release(
        manifest=manifest(tmp_path, report), inputs=[tmp_path / "output"],
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
