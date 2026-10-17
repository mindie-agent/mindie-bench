import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from mindie_bench.runner.verifier import parse_verifier

TEMPLATE = Path(__file__).resolve().parents[1] / "tasks/stale/fused-mc2-report"


def test_actual_old_and_current_complete_env_modules_demonstrate_the_version_boundary():
    result = subprocess.run(
        [sys.executable, "-I", str(TEMPLATE / "assets/environment/version_probe.py")],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    rows = json.loads(result.stdout)
    observed = {(row["version"], row["input"]): row["observed"] for row in rows}
    assert observed == {
        ("v0_23_0", "0"): 0,
        ("v0_23_0", "1"): 1,
        ("v0_27_1_contract", "0"): "unrecognized",
        ("v0_27_1_contract", "1"): "unrecognized",
    }
    recipe = json.loads((TEMPLATE / "recipe.json").read_text())
    assert recipe["task_spec"]["mindie_knowledge"]["knowledge_gradient"] == "stale"
    evidence = recipe["knowledge_evidence"][0]
    assert evidence["relation"] == "stale"
    assert (
        evidence["package_id"] not in recipe["task_spec"]["mindie_knowledge"]["same_origin_task_hashes"]
    )


@pytest.fixture
def actual_case(tmp_path):
    selected = os.environ.get("MINDIE_BENCH_STALE_PYTHON")
    repository = os.environ.get("MINDIE_BENCH_STALE_SOURCE_REPO")
    if not selected or not repository:
        pytest.skip("select the separately locked stale-task Python and exact historical source repo")
    python = Path(selected).absolute()
    recipe = json.loads((TEMPLATE / "recipe.json").read_text())
    base = recipe["task_spec"]["environment"]["l2"]["base_commit"]
    source_paths = list(recipe["source_files_sha256"])
    ancestors = {
        (parent / "__init__.py").as_posix()
        for name in source_paths
        for parent in Path(name).parents
        if parent != Path(".")
    }
    listed = set(
        subprocess.check_output(
            ["git", "-C", repository, "ls-tree", "-r", "--name-only", base, "tests"], text=True
        ).splitlines()
    )
    raw = subprocess.check_output(
        ["git", "-C", repository, "archive", base, *source_paths, *sorted(ancestors & listed)]
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(workspace, filter="data")
    for name, expected in recipe["source_files_sha256"].items():
        assert hashlib.sha256((workspace / name).read_bytes()).hexdigest() == expected
    verifier = tmp_path / "private-judge"
    shutil.copytree(TEMPLATE / "assets/verifier", verifier)
    (verifier / "runtime-binding.json").write_text(
        json.dumps(
            {
                "python": hashlib.sha256(python.resolve().read_bytes()).hexdigest(),
                "yaml": "6.0.3",
                "regex": "2026.9.29",
            }
        )
    )
    return python, workspace, verifier, recipe, repository


def grade(case):
    python, workspace, verifier, _, _ = case
    return subprocess.run(
        [str(python), "-I", str(verifier / "verify.py")],
        cwd=workspace,
        env={**os.environ, "MINDIE_BENCH_WORKSPACE": str(workspace)},
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("variant", ["base", "gold", "remove_legacy_tag_only"])
def test_real_reporting_module_needs_correct_current_semantics_not_just_removal(actual_case, variant):
    _, workspace, _, recipe, repository = actual_case
    target_name = recipe["task_spec"]["mindie_knowledge"]["target_files"][0]
    target = workspace / target_name
    if variant == "gold":
        subprocess.run(
            ["git", "apply", "--no-index", str(TEMPLATE / "assets/oracle/gold_patch.diff")],
            cwd=workspace,
            env={**os.environ, "GIT_CEILING_DIRECTORIES": str(workspace.parent)},
            capture_output=True,
            check=True,
        )
        expected = subprocess.check_output(
            [
                "git",
                "-C",
                repository,
                "show",
                recipe["task_spec"]["task"]["source"]["fix_commit"] + ":" + target_name,
            ]
        )
        assert target.read_bytes() == expected
    elif variant == "remove_legacy_tag_only":
        original = target.read_text()
        changed = original.replace('        "VLLM_ASCEND_ENABLE_FUSED_MC2": "fused_mc2",\n', "")
        assert changed != original
        target.write_text(changed)
    result = grade(actual_case)
    assert parse_verifier(result.returncode, result.stdout).reward == (1 if variant == "gold" else 0)
    if variant == "gold":
        assert json.loads(result.stdout.splitlines()[-2])["cpu_contract_cases"] == 32


def test_stale_task_runtime_identity_failure_stays_infrastructure(actual_case):
    _, _, verifier, _, _ = actual_case
    (verifier / "runtime-binding.json").write_text(
        json.dumps({"python": "0" * 64, "yaml": "6.0.3", "regex": "2026.9.29"})
    )
    result = grade(actual_case)
    assert result.returncode != 0
    assert "interpreter identity changed" in result.stderr
    with pytest.raises(ValueError, match="infrastructure"):
        parse_verifier(result.returncode, result.stdout)
