import json
import platform
import shutil
from pathlib import Path

import pytest

from mindie_bench.evaluation.budget import receipt_budget
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
        ("sleep", "infra_failed"),
        ("working_sleep", "completed_fail"),
        ("repair_then_sleep", "completed_pass"),
        ("crash", "unknown"),
        ("malformed", "unknown"),
        ("replay", "completed_pass"),
    ],
)
def test_end_to_end_four_state_runner(case, mode, status):
    if mode in {"sleep", "working_sleep", "repair_then_sleep"}:
        toml = case["task"] / "task.toml"
        toml.write_text(toml.read_text().replace("timeout_sec = 3", "timeout_sec = 1"))
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
    if mode in {"crash", "malformed", "sleep"}:
        assert result.primary_metrics["resolved"] is None
    if mode == "sleep":
        assert result.auxiliary_metrics["native_process"]["classification"] == "native_startup_infrastructure"
        assert not (root / "verifier_trajectory.json").exists()
    if mode in {"working_sleep", "repair_then_sleep"}:
        assert receipt_budget(result)["agent_budget_exhausted"] is True
        assert result.primary_metrics["reward"] == (1 if mode == "repair_then_sleep" else 0)
        assert (root / "verifier_trajectory.json").is_file()


@pytest.mark.parametrize("mode,status", [
    ("pass", "completed_pass"), ("fail", "completed_fail"), ("crash", "unknown"),
    ("preflight", "infra_failed"),
])
def test_delivery_overrun_preserves_verified_outcome_and_unknown(case, monkeypatch, mode, status):
    from types import SimpleNamespace

    from mindie_bench.runner import orchestrator

    (case["task"] / "instruction.md").write_text(f"mode={mode}\n")
    clock = orchestrator.time.monotonic
    original_cleanup = orchestrator.WorktreeSlot.cleanup
    offset = 0

    def cleanup(slot):
        nonlocal offset
        original_cleanup(slot)
        offset = 100

    # Advance only the runner's clock after actual verification and cleanup.
    # Agent, gateway and subprocess deadlines retain their real clocks.
    monkeypatch.setattr(orchestrator, "time", SimpleNamespace(monotonic=lambda: clock() + offset))
    monkeypatch.setattr(orchestrator.WorktreeSlot, "cleanup", cleanup)
    if mode == "preflight":
        probe = orchestrator.probe_runtime_import

        def elapsed_preflight(*args, **kwargs):
            nonlocal offset
            result = probe(*args, **kwargs)
            offset = 100
            return result

        monkeypatch.setattr(orchestrator, "probe_runtime_import", elapsed_preflight)
        case["config"].harness.expected_version = "wrong version"
    result = trial(case, experiment_context={"max_wall_clock_sec": 50})
    assert result.terminal_status.value == status, result.model_dump_json(indent=2)
    assert result.primary_metrics["reward"] == (
        None if mode in {"crash", "preflight"} else float(mode == "pass")
    )
    budget = receipt_budget(result)
    assert budget["delivery_budget_exceeded"] is True
    assert budget["measured_delivery_sec"] >= 100
    assert budget["delivery_overrun_sec"] == budget["measured_delivery_sec"] - 50
    if mode == "preflight":
        assert budget["agent_budget_exhausted"] is None
        assert "harness version mismatch" in result.failure_reason
    assert any(error["type"] == "BudgetOverrun" for error in result.errors)
    if mode == "pass":
        assert result.failure_reason is None
    else:
        assert result.failure_reason


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
