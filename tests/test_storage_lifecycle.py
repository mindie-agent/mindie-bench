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
def test_cleanup_after_expired_deadline_prunes_worktree_metadata(case, rogue_git):
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
    assert not list(cache_root.glob("*.tmp-*"))

    for index in range(18):
        run_cache(
            "rm -rf csrc/build_out && mkdir -p csrc/build_out && "
            f"printf pkg{index} > csrc/build_out/pkg.run"
        )

    entries = [path for path in cache_root.iterdir() if path.is_dir()]
    assert len(entries) == 16
    assert not list(cache_root.glob("*.tmp-*"))


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
