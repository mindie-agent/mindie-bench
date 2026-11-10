"""Publish a real local candidate without accepting local dependency receipts."""
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check_release_dependencies.py"


def inventory(repository):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.run([
        sys.executable, "-I", str(SCRIPT), "--inventory", "--repository", repository,
    ], capture_output=True, text=True, timeout=20, check=False, **flags)


def test_actual_local_candidate_is_excluded_before_dependency_receipt_validation():
    result = inventory("mindie-agent/mindie-bench")
    assert result.returncode == 0, result.stderr
    assert "mindie-bench" not in json.loads(result.stdout)


def test_same_local_source_cannot_masquerade_as_a_formal_dependency():
    result = inventory("mindie-agent/sim")
    assert result.returncode != 0
    assert "mindie-bench dependency has no official immutable installation receipt" in result.stderr


def test_released_consumer_process_api_reads_the_actual_local_candidate_version():
    from mindie_knowledge.component_runtime import read_versions

    assert read_versions(sys.executable, ["mindie-bench"]) == {
        "mindie-bench": metadata.version("mindie-bench")}
