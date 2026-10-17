import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

from mindie_bench.runner.remote_slot import POSTFLIGHT
from mindie_bench.runner.worktree_slot import WorktreeSlot


def test_local_inventory_compares_against_provided_public_inputs(case):
    public = case["task"] / "environment"
    public.mkdir()
    for name in ["unchanged.py", "deleted.py", "edited.py"]:
        (public / name).write_text("public input\n")
    slot = WorktreeSlot(case["spec"].environment["l2"], case["root"] / "local-slot", case["task"])
    workspace = slot.prepare()
    try:
        (workspace / "deleted.py").unlink()
        (workspace / "edited.py").write_text("edited input\n")
        (workspace / "module.py").write_text("VALUE = 2\n")
        assert slot.snapshot_patch(case["root"]) == ["deleted.py", "edited.py", "module.py"]
    finally:
        slot.cleanup()


def test_actual_postflight_inventories_remote_source_and_new_files(case, monkeypatch, capsys):
    slot = WorktreeSlot(case["spec"].environment["l2"], case["root"] / "remote-slot", case["task"])
    workspace = slot.prepare()
    (workspace / "module.py").write_text("VALUE = 2\n")
    unusual = "new helper\nwith space.py"
    (workspace / unusual).write_text("VALUE = 3\n")
    (workspace / ".bench-home").mkdir()
    (workspace / ".bench-home/temporary").write_text("not business source")
    original_path, original_run = pathlib.Path, subprocess.run

    def path(value):
        if value == "/proc":
            return SimpleNamespace(iterdir=lambda: iter(()))
        return original_path(value)

    def run(command, **kwargs):
        if command == ["owned-probe"]:
            return SimpleNamespace(stdout=json.dumps({"0": {"used_mb": 1, "pids": []}}))
        return original_run(command, **kwargs)

    monkeypatch.setattr(pathlib, "Path", path)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
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
    assert final["errors"] == []
    assert final["files_modified"] == ["module.py", unusual]
    assert set(final["untracked"]) == {unusual}
    assert not workspace.exists()
    assert (case["repo"] / "module.py").read_text() == "VALUE = 1\n"
    assert json.loads(capsys.readouterr().out)["cleanup"] == final
