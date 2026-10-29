import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mindie_bench.runner.worktree_slot import WorktreeSlot


@pytest.mark.tier1
def test_phase_deadline_bounds_contended_metadata_and_keeps_cleanup_recoverable(case):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "deadline-workspace", case["task"],
    )
    entered, release = threading.Event(), threading.Event()

    def hold_metadata():
        with slot.metadata_lock:
            entered.set()
            assert release.wait(2), "test failed to release worktree metadata"

    with ThreadPoolExecutor(max_workers=1) as pool:
        holder = pool.submit(hold_metadata)
        assert entered.wait(1)
        started = time.monotonic()
        slot.deadline = started + 0.03
        try:
            with pytest.raises(TimeoutError, match="worktree metadata|phase deadline"):
                with slot.metadata_edit():
                    pytest.fail("contended metadata must not be entered after the phase deadline")
            assert time.monotonic() - started < 0.5
            assert slot.created is False
        finally:
            release.set()
            holder.result(timeout=1)
    # A failed phase never permanently disables the owned slot. The next phase
    # has its own deadline, and an expired patch budget must not block removal.
    slot.deadline = time.monotonic() + 2
    workspace = slot.prepare()
    (workspace / "module.py").write_text("VALUE = 2\n")
    slot.deadline = time.monotonic()
    with pytest.raises(TimeoutError, match="phase deadline"):
        slot.snapshot_patch(workspace.parent)
    slot.deadline = time.monotonic() + 2
    slot.cleanup()
    assert not workspace.exists()
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"


@pytest.mark.tier1
def test_interrupted_git_add_retains_cleanup_of_its_real_owned_worktree(case, monkeypatch):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "interrupted-workspace", case["task"],
        deadline=time.monotonic() + 2,
    )
    original = subprocess.run

    def interrupt_after_materialization(command, **kwargs):
        result = original(command, **kwargs)
        if command[3:5] == ["worktree", "add"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return result

    monkeypatch.setattr(subprocess, "run", interrupt_after_materialization)
    with pytest.raises(subprocess.TimeoutExpired):
        slot.prepare()
    assert slot.created is True
    assert (slot.destination / "module.py").read_text() == "VALUE = 1\n"
    slot.deadline = time.monotonic() + 2
    slot.cleanup()
    assert not slot.destination.exists()
    listing = original(
        ["git", "-C", str(case["repo"]), "worktree", "list", "--porcelain"],
        check=True, capture_output=True, text=True, timeout=2,
    ).stdout
    assert str(slot.destination) not in listing
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize("same_repository", [True, False])
def test_concurrent_worktree_metadata_edits_are_scoped_to_the_source_repo(
    case, monkeypatch, same_repository
):
    environment = case["spec"].environment["l2"]
    other_environment = environment
    if not same_repository:
        other_repo = case["root"] / "other-source"
        shutil.copytree(case["repo"], other_repo)
        other_environment = environment.model_copy(update={"repo_path": str(other_repo)})
    slots = [
        WorktreeSlot(env, case["root"] / name / "workspace", case["task"])
        for name, env in (("first", environment), ("second", other_environment))
    ]
    entered, release, second_started, overlap = (threading.Event() for _ in range(4))
    run = subprocess.run

    def observe(command, *args, **kwargs):
        if command[3:5] == ["worktree", "add"]:
            if command[6] == str(slots[0].destination):
                entered.set()
                assert release.wait(5), "test did not release the first Git operation"
            else:
                overlap.set()
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", observe)

    def lifecycle(slot, *, second=False):
        if second:
            second_started.set()
        workspace = slot.prepare()
        try:
            assert (workspace / "module.py").read_text() == "VALUE = 1\n"
            assert not (workspace / ".git").exists()
            (workspace / "module.py").write_text("VALUE = 2\n")
            assert slot.snapshot_patch(workspace.parent) == ["module.py"]
        finally:
            slot.cleanup()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lifecycle, slots[0])
        try:
            assert entered.wait(5)
            second = pool.submit(lifecycle, slots[1], second=True)
            assert second_started.wait(5)
            assert overlap.wait(0.1 if same_repository else 5) is not same_repository
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert all(not slot.destination.exists() for slot in slots)
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"


def test_git_worktree_failure_preserves_its_diagnostic(case):
    destination = case["root"] / "occupied"
    destination.mkdir()
    (destination / "existing").write_text("preserve\n")
    slot = WorktreeSlot(case["spec"].environment["l2"], destination, case["task"])
    with pytest.raises(RuntimeError, match=r"git worktree add failed:.*|fatal:") as caught:
        slot.prepare()
    assert "already exists" in str(caught.value)
    assert (destination / "existing").read_text() == "preserve\n"


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
