import platform
import shutil
import subprocess

import pytest

from mindie_bench.integrations import first_party
from mindie_bench.models import CorpusConfig, Group, GroupConfig, MCPServer
from mindie_bench.runner import orchestrator
from mindie_bench.runner.profile_builder import build_profile

pytestmark = pytest.mark.skipif(
    platform.system() not in {"Darwin", "Linux"}
    or platform.system() == "Linux"
    and not shutil.which("bwrap"),
    reason="requires a native sandbox",
)


def g3_profile(case, corpus):
    source, commit = corpus
    config = case["config"]
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    config.groups[Group.G3] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(command="/usr/bin/true", component="knowledge"),
        },
        capture_command=["/usr/bin/true"],
    )
    trial = case["root"] / "git-probe"
    trial.mkdir()
    return trial, build_profile(trial, case["spec"], Group.G3, config)


def test_actual_selected_git_publishes_and_reads_back_in_native_sandbox(case, corpus):
    trial, profile = g3_profile(case, corpus)
    selected = profile.fingerprint["capture_git_runtime"]
    assert profile.private_env["MINDIE_BENCH_GIT"] == selected["executable"]
    if platform.system() == "Darwin":
        assert selected["executable"] != "/usr/bin/git"
    observed = orchestrator.probe_capture_git(
        profile, orchestrator.runtime_paths(case["config"], profile), trial
    )
    assert observed["status"] == "completed"
    assert observed["model_calls"] == 0
    assert observed["executable"] == selected["executable"]
    producer = profile.root / "capture-git-preflight" / "producer"
    consumer = profile.root / "capture-git-preflight" / "consumer"
    assert (producer / "material.md").read_bytes() == (consumer / "material.md").read_bytes()
    assert not list((producer / ".git" / "hooks").glob("*"))
    assert (trial / "capture-git-preflight.stderr").read_text() == ""


def test_required_git_preflight_failure_precedes_workspace_remote_and_model(case, corpus, monkeypatch):
    trial, profile = g3_profile(case, corpus)
    monkeypatch.setattr(orchestrator, "build_profile", lambda *args, **kwargs: profile)
    monkeypatch.setattr(orchestrator, "probe_runtime_import", lambda *args: {})

    def failed(*args, **kwargs):
        raise RuntimeError("actual Git helper cannot execute")

    def forbidden(*args, **kwargs):
        pytest.fail("Git admission must precede the workspace, remote and model")

    monkeypatch.setattr(orchestrator, "probe_capture_git", failed)
    monkeypatch.setattr(orchestrator, "WorktreeSlot", forbidden)
    monkeypatch.setattr(orchestrator, "RemoteSlot", forbidden)
    monkeypatch.setattr(orchestrator, "Gateway", forbidden)
    output = case["root"] / "run"
    output.mkdir()
    row = orchestrator.run_trial(
        case["task"], case["config"], tier="l2", group=Group.G3, repetition=1, run_root=output
    )
    assert row.terminal_status.value == "infra_failed"
    assert "actual Git helper cannot execute" in row.failure_reason
    assert not list(output.rglob("broker.jsonl"))


def test_git_preserves_original_stderr_in_visible_exception(monkeypatch):
    monkeypatch.setenv("MINDIE_BENCH_GIT", "/explicit/git")
    original = subprocess.CalledProcessError(1, ["/explicit/git"], stderr="specific Git failure")

    def failed(*args, **kwargs):
        raise original

    monkeypatch.setattr(first_party.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        first_party.git("rev-parse", "HEAD")
    assert caught.value is original
    assert "specific Git failure" in original.__notes__[0]


def test_git_preflight_timeout_retains_original_partial_diagnostics(case, corpus, monkeypatch):
    trial, profile = g3_profile(case, corpus)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30, output=b"partial Git", stderr=b"helper stuck")

    monkeypatch.setattr(orchestrator.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="capture Git preflight timed out"):
        orchestrator.probe_capture_git(
            profile, orchestrator.runtime_paths(case["config"], profile), trial
        )
    assert (trial / "capture-git-preflight.stdout").read_bytes() == b"partial Git"
    assert (trial / "capture-git-preflight.stderr").read_bytes() == b"helper stuck"
