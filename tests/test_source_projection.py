import hashlib
import subprocess

import pytest

from mindie_bench.models import SourceEnvironment
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS
from mindie_bench.runner.worktree_slot import WorktreeSlot


@pytest.fixture(autouse=True)
def neutral_git_environment(tmp_path, monkeypatch):
    # A developer's global git setup (for example an ignore rule for
    # .claude/settings.local.json) must not mask harness files in these tests.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-empty"))


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


@pytest.fixture
def hosted_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "module.py").write_text("VALUE = 1\n")
    (repo / "AGENTS.md").write_text("# project rules\n")
    (repo / ".claude/agents").mkdir(parents=True)
    (repo / ".claude/settings.json").write_text('{"permissions": {}}\n')
    (repo / ".claude/agents/reviewer.md").write_text("# reviewer\n")
    (repo / ".codex").mkdir()
    (repo / ".codex/config.toml").write_text('model = "x"\n')
    (repo / ".kimi").mkdir()
    (repo / ".kimi/config.toml").write_text('model = "x"\n')
    (repo / ".gemini").mkdir()
    (repo / ".gemini/settings.json").write_text("{}\n")
    (repo / ".agents").mkdir()
    (repo / ".agents/skills.md").write_text("# skills\n")
    (repo / ".mcp.json").write_text('{"mcpServers": {}}\n')
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Benchmark Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "source with tracked host harness configuration",
    )
    return repo, git(repo, "rev-parse", "HEAD")


def make_slot(repo, commit, destination, tmp_path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    task_root = tmp_path / "task"
    task_root.mkdir(exist_ok=True)
    environment = SourceEnvironment(repo_path=str(repo), base_commit=commit)
    return WorktreeSlot(environment, destination, task_root)


def test_snapshot_removes_host_config_but_keeps_source_and_agents_md(hosted_repo, tmp_path):
    repo, commit = hosted_repo
    slot = make_slot(repo, commit, tmp_path / "trial" / "workspace", tmp_path)
    workspace = slot.prepare()
    try:
        for name in HARNESS_CONFIG_PATHS:
            leaked = workspace / name
            assert not leaked.exists() and not leaked.is_symlink(), name
        assert not (workspace / ".git").exists()
        assert (workspace / "module.py").read_text() == "VALUE = 1\n"
        assert (workspace / "AGENTS.md").read_text() == "# project rules\n"
    finally:
        slot.cleanup()
    assert not (tmp_path / "trial" / "workspace").exists()


def test_final_patch_keeps_only_business_changes_and_source_repo_untouched(hosted_repo, tmp_path):
    repo, commit = hosted_repo
    before = dict(
        head=git(repo, "rev-parse", "HEAD"),
        worktrees=git(repo, "worktree", "list"),
        status=git(repo, "status", "--porcelain"),
        bytes=tree_bytes(repo),
    )
    slot = make_slot(repo, commit, tmp_path / "trial" / "workspace", tmp_path)
    workspace = slot.prepare()
    # Agent business changes.
    (workspace / "module.py").write_text("VALUE = 2\n")
    (workspace / "new_feature.py").write_text("X = 1\n")
    # Host client state recreated inside the snapshot is not a business change.
    (workspace / ".claude").mkdir()
    (workspace / ".claude/settings.local.json").write_text('{"noise": true}\n')
    (workspace / ".mcp.json").write_text('{"mcpServers": {"x": {}}}\n')
    output = tmp_path / "out"
    output.mkdir()
    modified = slot.snapshot_patch(output)
    try:
        patch = (output / "final_patch.diff").read_bytes()
        assert b"module.py" in patch and b"VALUE = 2" in patch
        assert b".claude" not in patch
        assert b".mcp.json" not in patch
        # Untracked business files are reported; harness paths stay out of the list.
        assert modified == ["module.py", "new_feature.py"]
        assert not (workspace / ".git").exists()
    finally:
        slot.cleanup()
    assert git(repo, "rev-parse", "HEAD") == before["head"]
    assert before["status"] == git(repo, "status", "--porcelain") == ""
    assert git(repo, "worktree", "list") == before["worktrees"]
    assert tree_bytes(repo) == before["bytes"]


def test_symlinked_destination_still_reattaches_and_cleans_up(hosted_repo, tmp_path):
    repo, commit = hosted_repo
    before = git(repo, "worktree", "list")
    link = tmp_path / "link"
    (tmp_path / "real").mkdir()
    link.symlink_to(tmp_path / "real")
    slot = make_slot(repo, commit, link / "trial" / "workspace", tmp_path)
    workspace = slot.prepare()
    (workspace / "module.py").write_text("VALUE = 2\n")
    output = tmp_path / "out"
    output.mkdir()
    assert slot.snapshot_patch(output) == ["module.py"]
    slot.cleanup()
    assert not (tmp_path / "real" / "trial" / "workspace").exists()
    assert git(repo, "worktree", "list") == before
