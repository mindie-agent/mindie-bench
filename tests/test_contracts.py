import json

import pytest
from pydantic import ValidationError

from mindie_bench.io import read_json, safe_relative, tree_hash, write_json
from mindie_bench.models import Group, GroupConfig, MCPServer, VerifierResult
from mindie_bench.runner.profile_builder import build_profile, filtered_corpus
from mindie_bench.runner.verifier import parse_verifier


@pytest.mark.parametrize("path", ["../secret", "/tmp/secret", "a/../../b", "a\\b", ""])
def test_paths_are_confined(path):
    with pytest.raises(ValueError):
        safe_relative(path)


def test_atomic_write_refuses_overwrite_in_exclusive_mode(tmp_path):
    path = tmp_path / "receipt.json"
    write_json(path, {"complete": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_json(path, {"complete": False}, exclusive=True)
    assert read_json(path) == {"complete": True}


def test_nonfinite_json_is_invalid(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"reward": NaN}')
    with pytest.raises(ValueError):
        read_json(path)
    with pytest.raises(ValueError):
        write_json(path, {"reward": float("inf")})


@pytest.mark.parametrize("reward,code", [(1.0, 0), (0.0, 1)])
def test_structured_verifier(reward, code):
    assert parse_verifier(code, json.dumps({"reward": reward})).reward == reward


@pytest.mark.parametrize(
    "output,code",
    [
        ('{"reward":0}', 0),
        ('{"reward":1}', 1),
        ("SSH failed", 255),
        ('{"reward":0.5}', 0),
        ('{"reward":1,"surprise":1}', 0),
    ],
)
def test_verifier_infrastructure_is_not_agent_failure(output, code):
    with pytest.raises((ValueError, ValidationError)):
        parse_verifier(code, output)


def test_verifier_rejects_invalid_performance():
    with pytest.raises(ValidationError):
        VerifierResult(reward=1, speedup_ratio=float("nan"))


def test_symlinks_cannot_smuggle_answers(tmp_path):
    (tmp_path / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        tree_hash(tmp_path)


def test_filter_physically_excludes_origin_and_indexes(corpus, tmp_path):
    source, _ = corpus
    destination = tmp_path / "filtered"
    fingerprint = filtered_corpus(source, destination, {"origin"})
    assert not (destination / "tasks/origin").exists()
    assert (destination / "tasks/analogy/materials/0001.md").is_file()
    assert not (destination / ".git").exists()
    assert fingerprint == tree_hash(destination)
    assert (source / "tasks/origin").exists()


def test_cached_indexes_cannot_leak_excluded_material(corpus, tmp_path):
    source, _ = corpus
    (source / "tasks/analogy/index.sqlite").write_bytes(b"origin answer")
    with pytest.raises(ValueError, match="cached indexes"):
        filtered_corpus(source, tmp_path / "filtered", {"origin"})


def test_g0_profile_is_fresh_and_drops_personal_credentials(case, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "do-not-inherit")
    monkeypatch.setenv("CODEX_THREAD_ID", "production-session")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/production/socket")
    trial = case["root"] / "trial"
    trial.mkdir()
    profile = build_profile(trial, case["spec"], Group.G0, case["config"])
    assert "GITHUB_TOKEN" not in profile.env
    assert "CODEX_THREAD_ID" not in profile.env
    assert "SSH_AUTH_SOCK" not in profile.env
    assert profile.env["HOME"].startswith(str(trial))
    assert profile.env["CODEX_HOME"].startswith(str(trial))
    assert profile.fingerprint["publication_policy"] == "local-only"
    assert (profile.root / "local_knowledge.git/HEAD").is_file()


def test_no_fake_g1_component_acceptance(case):
    trial = case["root"] / "trial"
    trial.mkdir()
    with pytest.raises(ValueError, match="remote-dev"):
        build_profile(trial, case["spec"], Group.G1, case["config"])


def test_g2_filters_and_preserves_stale_knowledge(case, corpus):
    from mindie_bench.models import CorpusConfig

    root, commit = corpus
    config = case["config"]
    config.corpora["fixture"] = CorpusConfig(path=str(root), commit=commit)
    config.groups[Group.G2] = GroupConfig(
        mcp_servers={
            "remote": MCPServer(command="/usr/bin/true", component="remote-dev"),
            "knowledge": MCPServer(command="/usr/bin/true", component="knowledge", args=["{corpus}"]),
        }
    )
    trial = case["root"] / "trial"
    trial.mkdir()
    spec = case["spec"]
    spec.mindie_knowledge.knowledge_gradient = "stale"
    profile = build_profile(trial, spec, Group.G2, config)
    assert not (profile.corpus / "tasks/origin").exists()
    assert (profile.corpus / "tasks/stale").exists()
    assert profile.fingerprint["excluded_task_hashes"] == ["origin"]


def test_corrupt_config_is_not_first_use(case):
    path = case["root"] / "invalid.toml"
    path.write_text("this is invalid TOML =")
    from mindie_bench.models import ExperimentConfig

    with pytest.raises(ValueError):
        ExperimentConfig.load(path)
