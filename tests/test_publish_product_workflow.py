from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/publish-product.yml").read_text())


def suite_step():
    steps = workflow()["jobs"]["publish"]["steps"]
    return next(step for step in steps if step.get("id") == "suite")


def workspace(tmp_path: Path) -> Path:
    (tmp_path / "release-tool/benchmarks").mkdir(parents=True)
    (tmp_path / "product/docs/releases/0.0.3").mkdir(parents=True)
    (tmp_path / "release-tool/benchmarks/product-value-v1.json").write_text(
        '{"schema_version": "mindie-value-suite/1", "id": "legacy", "studies": []}\n'
    )
    (tmp_path / "product/docs/releases/0.0.3/value-suite.json").write_text(
        '{"schema_version": "mindie-value-suite/2", "id": "v2", "studies": []}\n'
    )
    return tmp_path


def run_suite_step(tmp_path: Path, value_path: str | None) -> tuple[subprocess.CompletedProcess, dict]:
    root = workspace(tmp_path)
    output = tmp_path / "github-output"
    output.write_text("")
    env = {
        **os.environ,
        "GITHUB_WORKSPACE": str(root),
        "RUNNER_TEMP": str(tmp_path / "runner"),
        "GITHUB_OUTPUT": str(output),
        "VALUE_SUITE_PATH": value_path or "",
    }
    (tmp_path / "runner").mkdir()
    result = subprocess.run(
        ["bash", "-c", suite_step()["run"]],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    outputs = {}
    for line in output.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            outputs[key] = value
    return result, outputs


def test_workflow_input_is_optional_and_keeps_the_legacy_suite(tmp_path):
    inputs = workflow()[True]["workflow_call"]["inputs"]
    assert inputs["value-suite-path"]["required"] is False
    assert inputs["value-suite-path"]["default"] == ""
    result, outputs = run_suite_step(tmp_path, None)
    assert result.returncode == 0, result.stderr
    assert Path(outputs["value_suite"]) == (
        tmp_path / "release-tool/benchmarks/product-value-v1.json"
    ).resolve()
    identity = (tmp_path / "runner/product-value/value-suite-identity.json").read_text()
    assert "release-tool/benchmarks/product-value-v1.json" in identity
    assert '"selected_from_input": false' in identity
    assert outputs["value_suite_schema"] == "mindie-value-suite/1"
    assert outputs["value_suite_sha256"] == hashlib.sha256(
        (tmp_path / "release-tool/benchmarks/product-value-v1.json").read_bytes()
    ).hexdigest()


def test_workflow_uses_the_exact_selected_product_tag_path_and_rejects_escape(tmp_path):
    result, outputs = run_suite_step(tmp_path, "docs/releases/0.0.3/value-suite.json")
    assert result.returncode == 0, result.stderr
    selected = tmp_path / "product/docs/releases/0.0.3/value-suite.json"
    assert Path(outputs["value_suite"]) == selected.resolve()
    identity = (tmp_path / "runner/product-value/value-suite-identity.json").read_text()
    assert "docs/releases/0.0.3/value-suite.json" in identity
    assert '"selected_from_input": true' in identity
    assert outputs["value_suite_schema"] == "mindie-value-suite/2"
    assert outputs["value_suite_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert (tmp_path / "runner/product-value/value-suite.json").read_bytes() == selected.read_bytes()

    for bad in ("docs/releases/0.0.3/missing.json", "../outside.json", "/etc/passwd"):
        failed, _ = run_suite_step(tmp_path / bad.replace("/", "_"), bad)
        assert failed.returncode != 0
        assert "value-suite-path" in failed.stderr


def test_workflow_passes_the_selected_suite_via_env_not_shell_interpolation():
    step = suite_step()
    assert "${{ inputs.value-suite-path }}" in step["env"]["VALUE_SUITE_PATH"]
    assert "inputs.value-suite-path" not in step["run"]
    steps = workflow()["jobs"]["publish"]["steps"]
    public = next(step for step in steps if step.get("id") == "public-suite")
    assert public["env"]["VALUE_SUITE"] == "${{ steps.suite.outputs.value_suite }}"
    assert "public_suite" in public["run"]
    verify = next(step for step in steps if step.get("name") == "Require source-matched product value results")
    attach = next(
        step for step in steps if step.get("name") == "Require matching value attachments on component releases"
    )
    assert "$VALUE_SUITE" in verify["run"]
    assert "$VALUE_SUITE" in attach["run"]
    assert verify["env"]["VALUE_SUITE"] == "${{ steps.public-suite.outputs.value_suite }}"
    assert attach["env"]["VALUE_SUITE"] == "${{ steps.public-suite.outputs.value_suite }}"


@pytest.mark.parametrize(("tag", "accepted"), [("v0.1.1", True), ("v0.2.0", False)])
def test_patch_publication_keeps_milestone_value_acceptance_mandatory(tmp_path, tag, accepted):
    """Run the actual source/tag gate; patch mode cannot bypass a milestone."""
    version = tag[1:]
    docs = tmp_path / "docs/releases"
    (docs / version).mkdir(parents=True)
    (docs / (version + ".md")).write_text("Release fixture\n")
    (docs / version / "benchmark-manifest.json").write_text("{}\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True, timeout=5).strip()
    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Release test")
    git("add", "docs")
    git("commit", "-qm", "release inputs")
    git("tag", tag)
    revision = git("rev-parse", "HEAD")
    steps = workflow()["jobs"]["publish"]["steps"]
    gate = next(step for step in steps if step.get("id") == "product")
    output = tmp_path.parent / (tag + "-outputs")
    result = subprocess.run(["bash", "-c", gate["run"]], cwd=tmp_path,
        env={**os.environ, "GITHUB_REF_NAME": tag, "GITHUB_REF_TYPE": "tag",
             "GITHUB_SHA": revision, "BENCHMARK_REF": revision,
             "RELEASE_KIND": "patch", "VALUE_SUITE_PATH": "",
             "GITHUB_OUTPUT": str(output)},
        text=True, capture_output=True, timeout=10)
    assert (result.returncode == 0) is accepted, result.stderr
    if accepted:
        assert output.read_text().strip() == "version=" + version
        assert not list(tmp_path.rglob("*value-evidence*"))
    else:
        assert "milestones require product value acceptance" in result.stderr
