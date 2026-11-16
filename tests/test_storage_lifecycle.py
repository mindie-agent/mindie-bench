"""Tier 0/1 storage-lifecycle regressions for the Wave 1 leak fixes.

These tests exercise the real production paths: a graded corpus install, Git
worktree admin metadata, the remote POSTFLIGHT script, and RemoteSlot teardown.
They assert that owned temporary state disappears even on failure paths.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindie_bench.runner.remote_slot as remote_slot_module
from mindie_bench.evaluation import knowledge_tiered
from mindie_bench.evaluation.knowledge_tiered import evaluate_applicability_and_distractors
from mindie_bench.models import RemoteConfig
from mindie_bench.runner import kda_build_cache
from mindie_bench.runner.host_platform import subprocess_window_kwargs
from mindie_bench.runner.remote_receipt import canonical_slot_path
from mindie_bench.runner.remote_slot import POSTFLIGHT, RemoteSlot
from mindie_bench.runner.worktree_slot import WorktreeSlot

CORPUS = Path(__file__).parent / "fixtures" / "strategy-corpora" / "knowledge-operators"


def admin_entries(slot: WorktreeSlot) -> list[Path]:
    path = slot.admin_path / "worktrees"
    return list(path.iterdir()) if path.is_dir() else []


def applicability_case(case_id: str, **overrides) -> dict:
    row = {
        "case_id": case_id,
        "level": "L1",
        "query": None,
        "conditions": {},
        "probe_ref": None,
        "expected_status": None,
        "expected_refs": [],
        "loaded_refs": [],
        "adopted_refs": [],
        "oracle_refs": [],
        "retrieval_expected": False,
    }
    row.update(overrides)
    return row


@pytest.mark.tier1
def test_owned_tiered_workspace_removed_on_success_and_failure(tmp_path, monkeypatch):
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") == "mindie-tiered-":
            path = real_mkdtemp(*args, dir=tmp_path, **kwargs)
            created.append(Path(path))
            return path
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(knowledge_tiered.tempfile, "mkdtemp", mkdtemp)

    result = evaluate_applicability_and_distractors(
        [applicability_case("no-hit")], corpus_root=CORPUS, conditions=("no_knowledge",)
    )
    assert result["conditions"]["no_knowledge"]["queries"] == 0
    assert created and not created[0].exists()

    caller_workspace = tmp_path / "caller-workspace"
    evaluate_applicability_and_distractors(
        [applicability_case("caller")],
        corpus_root=CORPUS,
        conditions=("no_knowledge",),
        workspace=caller_workspace,
    )
    assert caller_workspace.is_dir()
    assert len(created) == 1

    with pytest.raises(ValueError, match="unknown graded unit"):
        evaluate_applicability_and_distractors(
            [
                applicability_case(
                    "broken-load",
                    query="Ascend memory alignment",
                    loaded_refs=["unit-does-not-exist"],
                    retrieval_expected=True,
                )
            ],
            corpus_root=CORPUS,
            conditions=("normal_retrieval",),
        )
    assert len(created) == 2
    assert not created[1].exists()


@pytest.mark.tier1
@pytest.mark.parametrize("rogue_git", [False, True])
def test_cleanup_after_expired_deadline_removes_worktree_metadata(case, rogue_git):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "expired-workspace", case["task"]
    )
    workspace = slot.prepare()
    assert len(admin_entries(slot)) == 1
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    if rogue_git:
        (workspace / ".git").mkdir()
        (workspace / ".git/config").write_text("[core]\n", encoding="utf-8")

    slot.deadline = time.monotonic() - 1
    slot.cleanup()

    assert not workspace.exists()
    assert admin_entries(slot) == []
    listing = subprocess.run(
        [slot.git_exe, "-C", str(case["repo"]), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        **subprocess_window_kwargs(),
    ).stdout
    assert str(workspace) not in listing
    assert (case["repo"] / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.tier1
def test_discard_grace_removes_tree_and_exact_admin_after_remove_timeout(case, monkeypatch):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "remove-timeout" / "workspace", case["task"]
    )
    workspace = slot.prepare()
    admin = slot._admin_dir
    assert admin is not None and admin.is_dir()

    def timeout_remove(*arguments):
        raise TimeoutError("simulated git worktree remove timeout")

    # The remove sub-budget is bounded independently from the pure-Python
    # discard grace, so an exhausted remove deadline cannot strand the slot.
    monkeypatch.setattr(slot, "WORKTREE_REMOVE_BUDGET_S", 1.0)
    monkeypatch.setattr(slot, "worktree_command", timeout_remove)
    slot.deadline = time.monotonic() - 1
    slot.cleanup()

    assert not workspace.exists()
    assert not admin.exists()
    assert admin_entries(slot) == []


@pytest.mark.tier1
def test_fallback_cleanup_preserves_concurrent_sibling_slot_metadata_and_patch(case, monkeypatch):
    for name in ("concurrent-a", "concurrent-b"):
        (case["root"] / name).mkdir()
    slot_a = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "concurrent-a" / "workspace", case["task"]
    )
    slot_b = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "concurrent-b" / "workspace", case["task"]
    )
    workspace_a = slot_a.prepare()
    workspace_b = slot_b.prepare()
    admin_a, admin_b = slot_a._admin_dir, slot_b._admin_dir
    assert admin_a is not None and admin_b is not None
    assert len(admin_entries(slot_a)) == 2

    # Force the fallback path on A only: a rogue .git plus a failed Git remove.
    (workspace_a / ".git").mkdir()
    (workspace_a / ".git/config").write_text("[core]\n", encoding="utf-8")
    original_remove = slot_a.worktree_command

    def fail_remove(*arguments):
        if arguments[0] == "remove":
            raise RuntimeError("simulated worktree remove failure")
        return original_remove(*arguments)

    monkeypatch.setattr(slot_a, "worktree_command", fail_remove)
    slot_a.deadline = time.monotonic() - 1
    slot_a.cleanup()

    assert not workspace_a.exists()
    assert not admin_a.exists()
    assert admin_b.is_dir()
    assert admin_entries(slot_a) == [admin_b]

    # B still has its hidden .git and exact admin entry: patch extraction and
    # its own cleanup must work after A's fallback removal.
    (workspace_b / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert slot_b.snapshot_patch(case["root"] / "concurrent-b") == ["module.py"]
    slot_b.cleanup()
    assert not workspace_b.exists()
    assert not admin_b.exists()
    assert admin_entries(slot_a) == []


def run_postflight_for_slot(workspace, repo, monkeypatch, *, extra=None, run=None):
    original_path, original_run = pathlib.Path, subprocess.run

    def path(value):
        if value == "/proc":
            return SimpleNamespace(iterdir=lambda: iter(()))
        return original_path(value)

    def default_run(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        return original_run(command, **kwargs)

    config = {
        "path": str(workspace),
        "repo_path": str(repo),
        "source_projection": [],
        "probe_command": ["owned-probe"],
        "devices": [0],
        "memory_limit_mb": 100,
        **(extra or {}),
    }
    with monkeypatch.context() as patch:
        patch.setitem(
            sys.modules,
            "pathlib",
            SimpleNamespace(Path=path, PurePosixPath=pathlib.PurePosixPath),
        )
        patch.setattr(subprocess, "run", run or default_run)
        patch.setattr(sys, "argv", ["postflight", json.dumps(config)])
        exec(compile(POSTFLIGHT, "remote-postflight", "exec"), {})


@pytest.mark.tier1
def test_postflight_cleanup_preserves_concurrent_sibling_slot(case, monkeypatch):
    for name in ("postflight-a", "postflight-b"):
        (case["root"] / name).mkdir()
    slot_a = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "postflight-a" / "workspace", case["task"]
    )
    slot_b = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "postflight-b" / "workspace", case["task"]
    )
    workspace_a = slot_a.prepare()
    workspace_b = slot_b.prepare()
    admin_a, admin_b = slot_a._admin_dir, slot_b._admin_dir
    assert admin_a is not None and admin_b is not None
    assert len(admin_entries(slot_a)) == 2
    (workspace_a / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    run_postflight_for_slot(workspace_a, case["repo"], monkeypatch)

    final = json.loads(pathlib.Path(str(workspace_a) + ".final.json").read_text())
    assert final["errors"] == []
    assert not workspace_a.exists()
    assert not admin_a.exists()
    assert admin_b.is_dir()
    assert admin_entries(slot_a) == [admin_b]

    # The sibling's hidden .git is still valid after postflight removed A.
    (workspace_b / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert slot_b.snapshot_patch(case["root"] / "postflight-b") == ["module.py"]
    slot_b.cleanup()
    assert not admin_b.exists()
    assert admin_entries(slot_a) == []


@pytest.mark.tier1
def test_postflight_writes_final_artifact_before_teardown_overrun(case, monkeypatch, capsys):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "teardown-budget" / "workspace", case["task"]
    )
    workspace = slot.prepare()
    admin = slot._admin_dir
    assert admin is not None
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    original_run = subprocess.run

    def slow_remove(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        if "worktree" in command and "remove" in command:
            time.sleep(2.0)
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
        return original_run(command, **kwargs)

    run_postflight_for_slot(
        workspace,
        case["repo"],
        monkeypatch,
        extra={
            "postflight_budget_s": 1.5,
            "postflight_artifact_reserve_s": 0.1,
            "postflight_teardown_s": 0.2,
        },
        run=slow_remove,
    )

    final = json.loads(pathlib.Path(str(workspace) + ".final.json").read_text())
    output = json.loads(capsys.readouterr().out)
    # The blocking Git call overran the entire receipt deadline, not just the
    # earlier teardown deadline. The pre-teardown artifact therefore remains the
    # honest durable copy while stdout reports the exhausted reserve.
    assert final["teardown_complete"] is False
    assert not any("git worktree remove" in item for item in final["errors"])
    assert any("worktree metadata removal" in item for item in output["cleanup"]["errors"])
    assert any("artifact rewrite: TimeoutError" in item for item in output["cleanup"]["errors"])
    assert not workspace.exists()
    assert output["cleanup"]["slot_vacated"] is True
    assert output["cleanup"]["trash_deferred"] is True
    # A blocking Git call consumed the reserves before fallback retirement, so
    # exact admin cleanup is honestly deferred rather than granted unbounded time.
    assert not any("admin removal" in item for item in final["errors"])
    assert any("admin removal" in item for item in output["cleanup"]["errors"])
    assert admin.exists()
    for path in workspace.parent.glob(".trash-*"):
        shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(admin, ignore_errors=True)


@pytest.mark.tier1
def test_postflight_bounded_trash_retirement_emits_receipt_before_deadline(
    case, monkeypatch, capsys
):
    slot = WorktreeSlot(
        case["spec"].environment["l2"],
        case["root"] / "bounded-trash" / "workspace",
        case["task"],
    )
    workspace = slot.prepare()
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    for index in range(8):
        (workspace / f"payload-{index}.txt").write_text("data\n", encoding="utf-8")
    original_run, original_unlink = subprocess.run, os.unlink
    sleeper_started = []

    def fail_remove(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        if "worktree" in command and "remove" in command:
            raise subprocess.CalledProcessError(1, command, stderr=b"simulated failure")
        return original_run(command, **kwargs)

    def slow_unlink(target, *args, **kwargs):
        result = original_unlink(target, *args, **kwargs)
        if ".trash-" in str(target):
            sleeper_started.append(time.monotonic())
            time.sleep(1.0)
        return result

    monkeypatch.setattr(os, "unlink", slow_unlink)
    started = time.monotonic()
    run_postflight_for_slot(
        workspace,
        case["repo"],
        monkeypatch,
        extra={
            "postflight_budget_s": 10.0,
            "postflight_artifact_reserve_s": 5.0,
            "postflight_teardown_s": 2.0,
        },
        run=fail_remove,
    )
    elapsed = time.monotonic() - started

    final = json.loads(pathlib.Path(str(workspace) + ".final.json").read_text())
    output = json.loads(capsys.readouterr().out)
    trash = list(workspace.parent.glob(".trash-*"))
    assert sleeper_started and elapsed < 10.0
    assert not workspace.exists()
    assert trash
    assert final["slot_vacated"] is True
    assert final["slot_removed"] is False
    assert final["trash_deferred"] is True
    assert final["teardown_complete"] is True
    assert output["cleanup"] == final
    assert any("slot tree removal deferred" in item for item in final["errors"])

    monkeypatch.setattr(os, "unlink", original_unlink)
    for path in trash:
        shutil.rmtree(path, ignore_errors=True)
    if slot._admin_dir is not None:
        shutil.rmtree(slot._admin_dir, ignore_errors=True)


@pytest.mark.tier1
def test_postflight_rewrites_final_artifact_with_fast_teardown_failure(case, monkeypatch):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "teardown-status" / "workspace", case["task"]
    )
    workspace = slot.prepare()
    admin = slot._admin_dir
    assert admin is not None
    original_run = subprocess.run

    def fail_remove(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        if "worktree" in command and "remove" in command:
            raise subprocess.CalledProcessError(1, command, stderr=b"simulated failure")
        return original_run(command, **kwargs)

    run_postflight_for_slot(workspace, case["repo"], monkeypatch, run=fail_remove)

    final = json.loads(pathlib.Path(str(workspace) + ".final.json").read_text())
    assert any("worktree metadata removal" in item for item in final["errors"])
    assert not workspace.exists()
    assert not admin.exists()


@pytest.mark.tier1
def test_postflight_abandons_blocked_trash_unlink_within_receipt_budget(
    case, monkeypatch, capsys
):
    workspace = case["root"] / "blocked-unlink"
    workspace.mkdir()
    for index in range(6):
        (workspace / f"payload-{index}.txt").write_text("data\n", encoding="utf-8")
    original_unlink = os.unlink
    release = threading.Event()
    blocked = []

    def slow_unlink(target, *args, **kwargs):
        if ".trash-" in str(target):
            blocked.append(pathlib.Path(target))
            release.wait(5.0)
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", slow_unlink)
    started = time.monotonic()
    try:
        run_postflight_for_slot(
            workspace,
            case["root"] / "missing-repo",
            monkeypatch,
            extra={
                "postflight_budget_s": 1.0,
                "postflight_artifact_reserve_s": 0.1,
                "postflight_teardown_s": 0.2,
            },
        )
        elapsed = time.monotonic() - started
        final = json.loads(pathlib.Path(str(workspace) + ".final.json").read_text())
        output = json.loads(capsys.readouterr().out)
    finally:
        release.set()

    assert blocked
    assert elapsed < 1.0
    assert final["slot_vacated"] is True
    assert final["slot_removed"] is False
    assert final["trash_deferred"] is True
    assert final["teardown_complete"] is True
    assert any("slot tree removal deferred" in item for item in final["errors"])
    assert output["cleanup"] == final
    trash = list(workspace.parent.glob(".trash-*"))
    assert trash
    for path in trash:
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.tier1
def test_postflight_abandons_sixty_second_trash_unlink_under_virtual_clock(
    case, monkeypatch, capsys
):
    workspace = case["root"] / "virtual-unlink"
    workspace.mkdir()
    for index in range(3):
        (workspace / f"payload-{index}.txt").write_text("data\n", encoding="utf-8")
    real_monotonic = time.monotonic
    real_join = threading.Thread.join
    real_start = threading.Thread.start
    original_unlink = os.unlink
    clock = {"skew": 0.0}
    release = threading.Event()
    worker_entered = threading.Event()
    blocked = []

    def monotonic():
        return real_monotonic() + clock["skew"]

    def start_and_wait(self):
        real_start(self)
        if self.name == "postflight-trash-retirement":
            # The virtual clock jump below is only meaningful once the worker is
            # actually blocked inside one unlink syscall.
            assert worker_entered.wait(2.0)

    def virtual_join(self, timeout=None):
        if self.name != "postflight-trash-retirement":
            return real_join(self, timeout)
        # Simulate a 60s clock advance while the deletion worker is blocked in
        # one unlink syscall: the join must expire immediately, not wait.
        clock["skew"] = 60.0
        return real_join(self, 0)

    def slow_unlink(target, *args, **kwargs):
        if ".trash-" in str(target):
            blocked.append(target)
            worker_entered.set()
            release.wait(60.0)
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(threading.Thread, "start", start_and_wait)
    monkeypatch.setattr(threading.Thread, "join", virtual_join)
    monkeypatch.setattr(os, "unlink", slow_unlink)
    started = real_monotonic()
    try:
        run_postflight_for_slot(
            workspace,
            case["root"] / "missing-repo",
            monkeypatch,
            extra={
                "postflight_budget_s": 2.0,
                "postflight_artifact_reserve_s": 0.1,
                "postflight_teardown_s": 1.0,
            },
        )
        elapsed = real_monotonic() - started
        final = json.loads(pathlib.Path(str(workspace) + ".final.json").read_text())
        output = json.loads(capsys.readouterr().out)
    finally:
        release.set()

    assert blocked
    assert elapsed < 1.0
    # The virtual 60s jump also exhausts the receipt deadline, so the durable
    # artifact is the honest pre-teardown copy while stdout carries the final
    # deferred state and the timed-out rewrite is explicit.
    assert final["teardown_complete"] is False
    assert output["cleanup"]["teardown_complete"] is True
    assert output["cleanup"]["slot_vacated"] is True
    assert output["cleanup"]["slot_removed"] is False
    assert output["cleanup"]["trash_deferred"] is True
    assert any("artifact rewrite: TimeoutError" in item for item in output["cleanup"]["errors"])
    for path in workspace.parent.glob(".trash-*"):
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.tier1
def test_postflight_uses_receipt_reserve_and_reconciles_after_teardown_overrun(
    case, tmp_path, monkeypatch, capsys
):
    slots = tmp_path / "slots"
    slots.mkdir()
    worktree = WorktreeSlot(
        case["spec"].environment["l2"], slots / "owned-trial", case["task"]
    )
    workspace = worktree.prepare()
    admin = worktree._admin_dir
    assert admin is not None
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    original_run = subprocess.run

    def slow_success(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        if "worktree" in command and "remove" in command:
            # The removal finishes after the teardown deadline but before the
            # dedicated receipt reserve expires.
            time.sleep(2.0)
            return original_run(
                command,
                check=True,
                capture_output=True,
                timeout=30,
                **subprocess_window_kwargs(),
            )
        return original_run(command, **kwargs)

    run_postflight_for_slot(
        workspace,
        case["repo"],
        monkeypatch,
        extra={
            "postflight_budget_s": 5.0,
            "postflight_artifact_reserve_s": 0.1,
            "postflight_teardown_s": 0.2,
        },
        run=slow_success,
    )

    final = json.loads(pathlib.Path(str(workspace) + ".final.json").read_text())
    output = json.loads(capsys.readouterr().out)
    assert final["teardown_complete"] is True
    assert final["slot_vacated"] is True
    assert final["slot_removed"] is True
    assert final["trash_deferred"] is False
    assert final["errors"] == []
    assert output["cleanup"] == final
    assert not workspace.exists()
    assert not admin.exists()

    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    slot.cleanup_receipt = None
    slot.process = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 0)
    monkeypatch.setattr(
        slot,
        "_read",
        lambda timeout: (_ for _ in ()).throw(TimeoutError("postflight stdout lost")),
    )
    monkeypatch.setattr(
        remote_slot_module, "terminate_process_group", lambda *args, **kwargs: None
    )

    slot.close()

    assert slot.recovered_cleanup_receipt is True
    assert slot.cleanup_confirmed is True
    assert slot.cleanup_receipt["cleanup"]["teardown_complete"] is True
    assert slot.cleanup_receipt["cleanup"]["slot_removed"] is True
    assert slot.cleanup_receipt["cleanup"]["trash_deferred"] is False


@pytest.mark.tier1
def test_postflight_removes_worktree_when_patch_exceeds_bound(case, monkeypatch):
    slot = WorktreeSlot(
        case["spec"].environment["l2"], case["root"] / "oversize-workspace", case["task"]
    )
    workspace = slot.prepare()
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    original_path, original_run = pathlib.Path, subprocess.run

    def path(value):
        if value == "/proc":
            return SimpleNamespace(iterdir=lambda: iter(()))
        return original_path(value)

    def run(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        if "diff" in command and "--binary" in command:
            return SimpleNamespace(stdout=b"x" * (32 * 1024 * 1024 + 1))
        return original_run(command, **kwargs)

    with monkeypatch.context() as patch:
        # Replace only the script's imported module. Rebinding `pathlib.Path`
        # itself breaks Python 3.11's `Path.__new__` global lookup during
        # pytest failure formatting.
        patch.setitem(
            sys.modules,
            "pathlib",
            SimpleNamespace(Path=path, PurePosixPath=pathlib.PurePosixPath),
        )
        patch.setattr(subprocess, "run", run)
        patch.setattr(
            sys,
            "argv",
            [
                "postflight",
                json.dumps(
                    dict(
                        path=str(workspace),
                        repo_path=str(case["repo"]),
                        source_projection=[],
                        probe_command=["owned-probe"],
                        devices=[0],
                        memory_limit_mb=100,
                    )
                ),
            ],
        )
        exec(compile(POSTFLIGHT, "remote-postflight", "exec"), {})

    final = json.loads(original_path(str(workspace) + ".final.json").read_text())
    assert any("remote final patch exceeds artifact bound" in item for item in final["errors"])
    assert not workspace.exists()
    assert admin_entries(slot) == []


def remote_slot_for_local_command(case, tmp_path, monkeypatch):
    slots = tmp_path / "slots"
    slots.mkdir(exist_ok=True)
    config = RemoteConfig(
        host="fixture",
        container="fixture",
        repo_path="/fixture",
        slots_root=str(slots),
        devices=[0],
        probe_command=["probe"],
    )
    environment = case["spec"].environment["l2"].model_copy(
        update={"npu_model": "Ascend910", "npu_count": 1}
    )
    slot = RemoteSlot(config, environment, "owned-trial", [0], 1, artifact_dir=tmp_path)
    slot.created = True
    slot.cleanup_receipt = {
        "cleanup": {
            "probe": {"0": {"used_mb": 0, "pids": [], "model": "Ascend910"}},
            "errors": [],
        }
    }
    monkeypatch.setattr(slot, "validate_probe", lambda value: value)
    # Execute the container-side sweep as a local process over the tmp slots root.
    monkeypatch.setattr(slot, "command", lambda argv: [sys.executable, *argv[1:]])
    return slot


class _FakeKernel32:
    def __init__(self, handle, error, exit_code, exit_ok=True):
        self.handle = handle
        self.error = error
        self.exit_code = exit_code
        self.exit_ok = exit_ok
        self.closed = False

    def OpenProcess(self, access, inherit, pid):
        assert access == kda_build_cache.PROCESS_QUERY_LIMITED_INFORMATION
        assert inherit is False
        assert pid == 4321
        return self.handle

    def CloseHandle(self, handle):
        assert handle == self.handle
        self.closed = True
        return True

    def GetExitCodeProcess(self, handle, pointer):
        assert handle == self.handle
        pointer._obj.value = self.exit_code
        return self.exit_ok


@pytest.mark.tier1
@pytest.mark.parametrize(
    ("handle", "error", "exit_code", "expected"),
    [
        (0, kda_build_cache.ERROR_INVALID_PARAMETER, None, True),
        (0, kda_build_cache.ERROR_ACCESS_DENIED, None, False),
        (123, 0, kda_build_cache.STILL_ACTIVE, False),
        (123, 0, 0, True),
    ],
)
def test_windows_pid_exit_probe_never_calls_os_kill(
    monkeypatch, handle, error, exit_code, expected
):
    kernel32 = _FakeKernel32(handle, error, exit_code)
    monkeypatch.setattr(kda_build_cache.os, "name", "nt")
    monkeypatch.setattr(
        kda_build_cache, "_windows_kernel32", lambda: (kernel32, lambda: kernel32.error)
    )
    monkeypatch.setattr(
        kda_build_cache.os,
        "kill",
        lambda *args: pytest.fail("os.kill(pid, 0) must never run on Windows"),
    )

    assert kda_build_cache._pid_exited(4321) is expected
    assert kernel32.closed is (handle != 0)


@pytest.mark.tier1
def test_dead_pid_temporary_cleanup_uses_platform_probe(tmp_path):
    cache_root = tmp_path / "kda_build_out"
    current = cache_root / "current"
    current.mkdir(parents=True)
    dead_process = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_process.wait()
    stale = cache_root / f"stale.tmp-{dead_process.pid}"
    stale.mkdir()
    (stale / "partial").write_text("partial\n", encoding="utf-8")
    old = time.time() - 2 * kda_build_cache.STALE_TMP_AGE_SEC
    os.utime(stale, (old, old))

    kda_build_cache.prune_cache(cache_root, current)

    assert not stale.exists()
    assert not list(cache_root.glob(".trash-*"))


@pytest.mark.tier1
def test_kda_build_cache_bounds_entries_and_removes_failed_temporary(tmp_path):
    output = tmp_path / "output"
    source = output / "source"
    (source / "csrc").mkdir(parents=True)
    (output / "build-receipt.json").write_text(
        json.dumps(
            {
                "schema": "mindie-bench-kda-source-build/1",
                "status": "prepared",
                "source_files_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    helper = Path(__file__).parents[1] / "src/mindie_bench/runner/kda_build_cache.py"
    environment = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}

    def run_cache(command):
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(helper), "-c", command],
            cwd=source,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            **subprocess_window_kwargs(),
        )
        assert result.returncode == 0, result.stderr
        return result

    # A directory named like the package makes copy2 fail after the temp dir exists.
    run_cache("mkdir -p csrc/build_out/pkg.run")
    cache_root = home / "cache" / "kda_build_out"
    cache_root.mkdir(parents=True, exist_ok=True)
    assert not list(cache_root.glob("*.tmp-*"))

    dead_process = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_process.wait()
    alive_tmp = cache_root / f"alive.tmp-{os.getpid()}"
    dead_tmp = cache_root / f"dead.tmp-{dead_process.pid}"
    young_tmp = cache_root / f"young.tmp-{dead_process.pid}"
    for path in (alive_tmp, dead_tmp, young_tmp):
        path.mkdir()
        (path / "partial").write_text("partial\n", encoding="utf-8")
    old = time.time() - 2 * 60 * 60
    os.utime(alive_tmp, (old, old))
    os.utime(dead_tmp, (old, old))

    for index in range(18):
        run_cache(
            "rm -rf csrc/build_out && mkdir -p csrc/build_out && "
            f"printf pkg{index} > csrc/build_out/pkg.run"
        )

    # In-flight temporary trees are neither counted nor evicted. Only the old
    # directory whose PID has exited is atomically renamed and reclaimed.
    entries = [
        path
        for path in cache_root.iterdir()
        if path.is_dir() and ".tmp-" not in path.name
    ]
    assert len(entries) == 16
    assert alive_tmp.is_dir()
    assert young_tmp.is_dir()
    assert not dead_tmp.exists()
    assert not list(cache_root.glob(".trash-*"))


@pytest.mark.tier1
def test_remote_slot_close_recovers_persisted_receipt_after_stdout_timeout(
    case, tmp_path, monkeypatch
):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    slot.cleanup_receipt = None
    slot.process = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 0)
    payload = {
        "trial_id": "owned-trial",
        "slot_path": slot.path,
        "probe": {"0": {"used_mb": 0, "pids": [], "model": "Ascend910"}},
        "errors": [],
        "slot_vacated": True,
        "slot_removed": True,
        "trash_deferred": False,
        "teardown_complete": True,
    }
    owned = Path(slot.path + ".final.json")
    owned.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        slot,
        "_read",
        lambda timeout: (_ for _ in ()).throw(TimeoutError("postflight stdout lost")),
    )
    monkeypatch.setattr(
        remote_slot_module, "terminate_process_group", lambda *args, **kwargs: None
    )

    slot.close()

    assert slot.recovered_cleanup_receipt is True
    assert slot.cleanup_receipt == {"cleanup": payload}
    assert slot.cleanup_confirmed is True
    assert slot.cleanup_warning and "postflight stdout lost" in slot.cleanup_warning
    assert not owned.exists()
    assert json.loads((tmp_path / "remote_cleanup.json").read_text()) == slot.cleanup_receipt


@pytest.mark.tier1
def test_canonical_slot_path_accepts_windows_separator_conversion():
    # Windows `Path(...)` renders `C:\...\slots/owned-trial` with backslashes.
    mixed = r"C:\Users\runneradmin\slots/owned-trial"
    converted = r"C:\Users\runneradmin\slots\owned-trial"
    assert canonical_slot_path(mixed) == canonical_slot_path(converted)
    assert canonical_slot_path(mixed) != canonical_slot_path(
        r"C:\Users\runneradmin\slots\other-trial"
    )


@pytest.mark.tier1
def test_remote_slot_reconciles_windows_path_conversion(case, tmp_path, monkeypatch):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    assert "\\" not in slot.path
    slot.cleanup_receipt = None
    slot.process = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 0)
    payload = {
        "trial_id": "owned-trial",
        # Windows `Path(...)` renders the same slot with backslashes. The remote
        # probe runs POSIX in production, so reconciliation must compare one
        # canonical identity instead of raw separator strings.
        "slot_path": slot.path.replace("/", "\\"),
        "probe": {"0": {"used_mb": 0, "pids": [], "model": "Ascend910"}},
        "errors": [],
        "slot_vacated": True,
        "slot_removed": True,
        "trash_deferred": False,
        "teardown_complete": True,
    }
    owned = Path(slot.path + ".final.json")
    owned.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        slot,
        "_read",
        lambda timeout: (_ for _ in ()).throw(TimeoutError("postflight stdout lost")),
    )
    monkeypatch.setattr(
        remote_slot_module, "terminate_process_group", lambda *args, **kwargs: None
    )

    slot.close()

    assert slot.recovered_cleanup_receipt is True
    assert slot.cleanup_confirmed is True
    assert slot.cleanup_receipt == {"cleanup": payload}


@pytest.mark.tier1
def test_remote_slot_close_does_not_confirm_a_pre_teardown_receipt(
    case, tmp_path, monkeypatch
):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    slot.cleanup_receipt = None
    slot.process = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 0)
    payload = {
        "trial_id": "owned-trial",
        "slot_path": slot.path,
        "probe": {"0": {"used_mb": 0, "pids": [], "model": "Ascend910"}},
        "errors": [],
        "slot_vacated": True,
        "slot_removed": False,
        "trash_deferred": False,
        "teardown_complete": False,
    }
    owned = Path(slot.path + ".final.json")
    owned.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        slot,
        "_read",
        lambda timeout: (_ for _ in ()).throw(TimeoutError("postflight stdout lost")),
    )
    monkeypatch.setattr(
        remote_slot_module, "terminate_process_group", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError, match="predates teardown completion"):
        slot.close()

    assert slot.recovered_cleanup_receipt is True
    assert slot.cleanup_confirmed is False


@pytest.mark.tier1
def test_remote_slot_close_rejects_persisted_receipt_for_another_slot(
    case, tmp_path, monkeypatch
):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    slot.cleanup_receipt = None
    slot.process = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 0)
    payload = {
        "trial_id": "different-trial",
        "slot_path": "/different/slot",
        "probe": {"0": {"used_mb": 0, "pids": [], "model": "Ascend910"}},
        "errors": [],
        "slot_vacated": True,
        "slot_removed": True,
        "trash_deferred": False,
        "teardown_complete": True,
    }
    owned = Path(slot.path + ".final.json")
    owned.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        slot,
        "_read",
        lambda timeout: (_ for _ in ()).throw(TimeoutError("postflight stdout lost")),
    )
    monkeypatch.setattr(
        remote_slot_module, "terminate_process_group", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError, match="no remote cleanup receipt"):
        slot.close()

    assert slot.recovered_cleanup_receipt is False
    assert slot.cleanup_confirmed is False
    assert slot.cleanup_warning and "reconciliation" in slot.cleanup_warning
    assert slot.cleanup_warning and "exit status 4" in slot.cleanup_warning
    assert owned.exists()
    assert not (tmp_path / "remote_cleanup.json").exists()


@pytest.mark.tier1
def test_remote_slot_close_removes_final_artifact_and_sweeps_stale(
    case, tmp_path, monkeypatch
):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    slots = Path(slot.config.slots_root)
    owned = Path(slot.path + ".final.json")
    stale = slots / "stale-trial.final.json"
    recent = slots / "recent-trial.final.json"
    active = slots / "active-trial.final.json"
    for path in (owned, stale, recent, active):
        path.write_text("{}", encoding="utf-8")
    (slots / "active-trial").mkdir()
    old = time.time() - 49 * 60 * 60
    os.utime(stale, (old, old))
    os.utime(active, (old, old))

    slot.close()

    assert not owned.exists()
    assert not stale.exists()
    assert recent.exists()
    assert active.exists()
    assert json.loads((tmp_path / "remote_cleanup.json").read_text()) == slot.cleanup_receipt
    assert slot.cleanup_warning is None
    assert set(slot.removed_remote_artifacts) == {
        "owned-trial.final.json",
        "stale-trial.final.json",
    }


@pytest.mark.tier1
@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_remote_slot_keeps_remote_artifact_when_local_receipt_is_not_durable(
    case, tmp_path, monkeypatch, failure
):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    owned = Path(slot.path + ".final.json")
    owned.write_text("{}", encoding="utf-8")

    if failure == "fsync":
        monkeypatch.setattr(
            os, "fsync", lambda descriptor: (_ for _ in ()).throw(OSError("sync failed"))
        )
    else:
        monkeypatch.setattr(
            os, "replace", lambda *arguments: (_ for _ in ()).throw(OSError("replace failed"))
        )

    with pytest.raises(RuntimeError, match="local cleanup evidence"):
        slot.close()

    # The remote artifact is only deleted after the local receipt is durable.
    assert owned.exists()
    assert not (tmp_path / "remote_cleanup.json").exists()


@pytest.mark.tier1
def test_remote_slot_final_artifact_failure_is_a_cleanup_warning(
    case, tmp_path, monkeypatch
):
    slot = remote_slot_for_local_command(case, tmp_path, monkeypatch)
    owned = Path(slot.path + ".final.json")
    owned.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        slot, "command", lambda argv: [sys.executable, "-c", "raise SystemExit(3)"]
    )

    slot.close()

    assert owned.exists()
    assert "remote final artifact cleanup" in slot.cleanup_warning
    assert (tmp_path / "remote_cleanup.json").is_file()
