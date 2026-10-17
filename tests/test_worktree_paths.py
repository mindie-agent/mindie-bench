from pathlib import Path

import pytest

from mindie_bench.runner.worktree_slot import WorktreeSlot


@pytest.mark.parametrize("change_cwd", [False, True])
def test_relative_worktree_paths_keep_the_callers_identity_and_cleanup(case, monkeypatch, change_cwd):
    monkeypatch.chdir(case["root"])
    slot = WorktreeSlot(case["spec"].environment["l2"], Path("owned-slot"), Path("task"))
    expected = case["root"] / "owned-slot"
    if change_cwd:
        monkeypatch.chdir(case["repo"])
    try:
        workspace = slot.prepare()
        assert workspace == expected
        assert (workspace / "module.py").read_text() == "VALUE = 1\n"
        assert not (workspace / ".git").exists()
        assert not (case["repo"] / "owned-slot").exists()
    finally:
        slot.cleanup()
    assert not expected.exists()
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"


def test_soundness_api_accepts_relative_task_and_output_paths(case, monkeypatch):
    from mindie_bench.runner.soundness import check_soundness

    (case["task"] / "oracle/gold_patch.diff").write_text(
        "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n"
        "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    )
    monkeypatch.chdir(case["root"])
    proof = check_soundness(Path("task"), case["config"], tier="l2", output=Path("soundness"))
    assert proof["before_reward"] == 0
    assert proof["after_reward"] == 1
    assert proof["errors"] == []
    assert not (case["root"] / "soundness/workspace").exists()
    assert not (case["repo"] / "soundness").exists()


def test_matrix_api_runs_relative_task_and_output_paths_in_the_callers_cwd(case, monkeypatch):
    from mindie_bench.models import Group
    from mindie_bench.runner.orchestrator import run_matrix

    monkeypatch.chdir(case["root"])
    rows = run_matrix(
        [Path("task")],
        case["config"],
        tier="l2",
        groups=[Group.G0],
        repetitions=1,
        concurrency=1,
        output=Path("relative-matrix"),
    )
    assert len(rows) == 1
    assert rows[0].terminal_status.value == "completed_pass"
    assert rows[0].errors == []
    assert not (case["repo"] / "relative-matrix").exists()
