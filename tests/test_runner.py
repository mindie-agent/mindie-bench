import json
import platform
import shutil
from pathlib import Path

import pytest

from mindie_bench.models import Group
from mindie_bench.runner.orchestrator import run_matrix, run_trial
from mindie_bench.scorer.comparator import freeze, load_receipts

pytestmark = pytest.mark.skipif(
    platform.system() not in {"Darwin", "Linux"}
    or (platform.system() == "Linux" and not shutil.which("bwrap")),
    reason="requires a native sandbox; no unisolated fallback",
)


def trial(case, **kwargs):
    output = case["root"] / "run"
    output.mkdir()
    return run_trial(
        case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=output, **kwargs
    )


@pytest.mark.parametrize(
    "mode,status",
    [
        ("pass", "completed_pass"),
        ("fail", "completed_fail"),
        ("sleep", "completed_fail"),
        ("crash", "unknown"),
        ("malformed", "unknown"),
        ("replay", "completed_pass"),
    ],
)
def test_end_to_end_four_state_runner(case, mode, status):
    (case["task"] / "instruction.md").write_text(f"mode={mode}\n")
    result = trial(case)
    assert result.terminal_status.value == status, result.model_dump_json(indent=2)
    root = Path(result.artifacts["private_trial_dir"])
    assert (root / "trial_receipt.json").is_file()
    assert (root / "native_events.jsonl").is_file()
    assert not (root / "workspace").exists()
    assert (root / "final_workspace.tar.gz").is_file()
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"
    if mode == "replay":
        assert result.auxiliary_metrics["replay_test_runs"] == 1
        assert result.primary_metrics["npu_test_runs"] is None
    if mode in {"crash", "malformed"}:
        assert result.primary_metrics["resolved"] is None


def test_verifier_failure_is_infra_and_never_scored_zero(case):
    (case["task"] / "verifier/test.sh").write_text("echo 'missing environment' >&2\nexit 125\n")
    result = trial(case)
    assert result.terminal_status.value == "infra_failed", result.model_dump_json(indent=2)
    assert result.primary_metrics["reward"] is None
    assert any(error["stage"] == "verifier" for error in result.errors)


def test_wrong_harness_version_fails_before_agent(case):
    case["config"].harness.expected_version = "wrong version"
    result = trial(case)
    assert result.terminal_status.value == "infra_failed"
    assert not (Path(result.artifacts["private_trial_dir"]) / "native_events.jsonl").exists()


def test_oracle_outside_worktree_still_unreadable_by_agent(case):
    secret = case["task"] / "oracle/gold_patch.diff"
    secret.write_text("VERY_PRIVATE_GOLD_FIX\n")
    (case["task"] / "instruction.md").write_text("mode=probe\nsecret=" + str(secret) + "\nmode=pass\n")
    result = trial(case)
    assert result.terminal_status.value == "completed_pass", result.model_dump_json(indent=2)
    assert (
        "VERY_PRIVATE_GOLD_FIX"
        not in (Path(result.artifacts["private_trial_dir"]) / "native_events.jsonl").read_text()
    )


def test_missing_component_does_not_become_empty_baseline(case):
    output = case["root"] / "run"
    output.mkdir()
    result = run_trial(
        case["task"], case["config"], tier="l2", group=Group.G2, repetition=1, run_root=output
    )
    assert result.terminal_status.value == "infra_failed"
    assert "remote-dev" in result.failure_reason


def test_matrix_concurrency_receipts_and_no_accidental_rerun(case):
    output = case["root"] / "matrix"
    rows = run_matrix(
        [case["task"]],
        case["config"],
        tier="l2",
        groups=[Group.G0],
        repetitions=3,
        concurrency=3,
        output=output,
    )
    assert len(rows) == 3
    assert all(row.terminal_status.value == "completed_pass" for row in rows), [
        row.model_dump() for row in rows
    ]
    assert len(load_receipts(output)) == 3
    with pytest.raises(FileExistsError):
        run_matrix(
            [case["task"]],
            case["config"],
            tier="l2",
            groups=[Group.G0],
            repetitions=3,
            concurrency=3,
            output=output,
        )
    baseline = case["root"] / "baseline"
    freeze(rows, baseline)
    assert len(load_receipts(baseline)) == 3
    assert json.loads((baseline / "baseline_manifest.json").read_text())["receipt_sha256"]


def test_g3_cannot_claim_flywheel_without_seed(case):
    with pytest.raises(ValueError, match="seed"):
        run_matrix(
            [case["task"]],
            case["config"],
            tier="l2",
            groups=[Group.G3],
            repetitions=1,
            concurrency=1,
            output=case["root"] / "run",
        )
