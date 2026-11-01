"""Business outcomes for the repeatable component-value runner."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mindie_coordinator")
pytest.importorskip("mindie_diagnostics")
pytest.importorskip("mindie_knowledge")

from mindie_bench.benchmarks import value_coordinator, value_diagnostics


def test_competing_processes_keep_live_ports_until_release(tmp_path):
    observed = {"status": "ok", "devices": [0, 1], "free": [0, 1], "busy": {}}
    result = value_coordinator.run(tmp_path, observed, 1)
    row = result["measurements"][0]
    assert row["baseline"]["successful_listeners"] == 1
    assert row["baseline"]["bind_conflicts"] == 1
    assert row["candidate"]["successful_listeners"] == 2
    raw = json.loads(Path(row["raw"]).read_text())
    recovery = raw["candidate"]["events"][-1]
    assert recovery["busy_release"]["status"] == "orphaned_busy"
    assert recovery["recovered"]["status"] == "granted"


def test_faults_reach_http_once_with_redaction_and_revoke(tmp_path):
    result = value_diagnostics.one(str(tmp_path), 1, exercise_retry=False)
    assert result["baseline"]["contains_synthetic_secret"]
    assert result["candidate"]["http_posts"] == 1
    assert result["candidate"]["reconciled_without_duplicate"]
    assert result["candidate"]["withdrawal_prevented_requests"]
    raw = json.loads(Path(result["raw"]).read_text())
    assert raw["initial_occurrences"] == 3
    assert raw["delivery"]["status"] == "published"
    assert raw["reconcile"]["status"] == "reconciled"
    if (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("mindie_diagnostics.precision") is not None):
        assert result["candidate"]["precision_bugs_caught"] == 5
        assert result["candidate"]["precision_false_positives"] == 0
        assert raw["precision_localization"]["cases_evaluated"] == 5
