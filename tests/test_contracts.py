import hashlib
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mindie_bench.io import read_json, safe_relative, tree_hash, write_json
from mindie_bench.models import Group, GroupConfig, MCPServer
from mindie_bench.runner.profile_builder import bind_mcp_transport, build_profile, filtered_corpus
from mindie_bench.runner.verifier import parse_verifier


def test_component_python_keeps_its_virtual_environment(case, tmp_path):
    runtime = tmp_path / "component-runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(runtime)], check=True)
    python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    case["config"].groups[Group.G1] = GroupConfig(
        mcp_servers={"remote": MCPServer(command=str(python), component="remote-dev")}
    )
    destination = tmp_path / "trial"
    destination.mkdir()
    profile = build_profile(destination, case["spec"], Group.G1, case["config"])
    invocation = json.loads((profile.root / "mcp.json").read_text())["mcpServers"]["remote"]["command"]
    observed = subprocess.check_output(
        [invocation, "-I", "-c", "import sys;print(sys.prefix)"], text=True
    ).strip()
    assert observed == str(runtime)


def test_mcp_transport_explicitly_survives_native_environment_filtering(case, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-inherit")
    case["config"].groups[Group.G1] = GroupConfig(
        mcp_servers={"remote": MCPServer(command="/usr/bin/true", component="remote-dev")}
    )
    destination = case["root"] / "transport"
    destination.mkdir()
    profile = build_profile(destination, case["spec"], Group.G1, case["config"])
    profile.env["PATH"] = str(profile.root / "bin") + ":" + profile.env["PATH"]
    profile.env["MINDIE_BENCH_GATEWAY"] = "http://127.0.0.1:12345"
    bind_mcp_transport(profile)
    server = json.loads((profile.root / "mcp.json").read_text())["mcpServers"]["remote"]
    assert server["env"] == {
        "PATH": profile.env["PATH"],
        "MINDIE_BENCH_GATEWAY": "http://127.0.0.1:12345",
    }
    assert "AWS_SECRET_ACCESS_KEY" not in server["env"]


def test_explicit_model_metadata_is_frozen_and_fingerprinted(case):
    source = case["root"] / "catalog.json"
    source.write_text(json.dumps({"models": [{"slug": case["config"].harness.model}]}))
    original = source.read_bytes()
    case["config"].harness.model_catalog_path = str(source)
    destination = case["root"] / "metadata"
    destination.mkdir()
    profile = build_profile(destination, case["spec"], Group.G0, case["config"])
    source.write_text("{}")
    assert (profile.root / "model-catalog.json").read_bytes() == original
    assert profile.fingerprint["model_catalog_sha256"] == hashlib.sha256(original).hexdigest()


def test_ambiguous_native_model_metadata_is_rejected_before_a_trial(case):
    source = case["root"] / "catalog.json"
    source.write_text(json.dumps({"models": [{"slug": case["config"].harness.model}] * 2}))
    case["config"].harness.model_catalog_path = str(source)
    destination = case["root"] / "metadata"
    destination.mkdir()
    with pytest.raises(ValueError, match="explicitly selected model"):
        build_profile(destination, case["spec"], Group.G0, case["config"])


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
        ('{"reward":1,"speedup_ratio":NaN}', 0),
    ],
)
def test_verifier_infrastructure_is_not_agent_failure(output, code):
    with pytest.raises((ValueError, ValidationError)):
        parse_verifier(code, output)


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
    assert (profile.private_root / "local_knowledge.git/HEAD").is_file()
    assert "MINDIE_BENCH_LOCAL_KNOWLEDGE_REMOTE" not in profile.env


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
