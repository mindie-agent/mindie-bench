"""Tier 0/1 storage-lifecycle regressions for the Wave 1 leak fixes.

These tests exercise the real production paths: a graded corpus install, Git
worktree admin metadata, the remote POSTFLIGHT script, and RemoteSlot teardown.
They assert that owned temporary state disappears even on failure paths.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.evaluation import knowledge_tiered
from mindie_bench.evaluation.knowledge_tiered import evaluate_applicability_and_distractors
from mindie_bench.models import RemoteConfig
from mindie_bench.runner.host_platform import subprocess_window_kwargs
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
    # The published artifact predates teardown; the late timeout is visible only
    # in the in-memory stdout receipt, proving the file was already complete.
    assert not any("git worktree remove" in item for item in final["errors"])
    assert any("worktree metadata removal" in item for item in output["cleanup"]["errors"])
    assert not workspace.exists()
    assert not admin.exists()


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
    slots.mkdir()
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
