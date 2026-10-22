import pytest

from mindie_bench.models import CorpusConfig, Group, GroupConfig, MCPServer
from mindie_bench.runner.profile_builder import build_profile, knowledge_prompt


def configure(case, corpus, *, mode, files=(), skills=False):
    config = case["config"].model_copy(deep=True)
    source, commit = corpus
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    skill_path = None
    if skills:
        directory = case["root"] / "skills/mindie-agent"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("# Domain guidance\nHistorical knowledge is advisory.\n")
        skill_path = str(directory.parent)
    config.groups[Group.G2] = GroupConfig(
        knowledge_mode=mode,
        oracle_context_files=list(files),
        skills_path=skill_path,
        mcp_servers={
            name: MCPServer(command="/usr/bin/true", component=name)
            for name in ["knowledge", "remote-dev"]
        },
    )
    return config


def test_explicit_condition_uses_single_user_gesture_without_internal_tool_coaching(case, corpus):
    config = configure(case, corpus, mode="explicit", skills=True)
    profile = build_profile(case["root"], case["spec"], Group.G2, config)
    assert profile.fingerprint["knowledge_mode"] == "explicit"
    prompt = knowledge_prompt(profile)
    assert prompt == "/mindie-agent\n\n"
    assert "mindie_entry" not in prompt
    assert "knowledge_query" not in prompt
    assert "knowledge_explain" not in prompt
    assert "Actual task applicability conditions" not in prompt
    assert str(profile.corpus) not in prompt
    assert profile.fingerprint["oracle_context_sha256"] is None


def test_oracle_context_is_bound_to_allowed_corpus_material(case, corpus):
    config = configure(case, corpus, mode="oracle", files=["tasks/analogy/materials/0001.md"])
    profile = build_profile(case["root"], case["spec"], Group.G2, config)
    prompt = knowledge_prompt(profile)
    assert "Controlled oracle-context condition" in prompt
    assert "Historical trace: NPU error" in prompt
    assert profile.fingerprint["oracle_context_sha256"]
    assert list(profile.fingerprint["oracle_context_files_sha256"]) == [
        "tasks/analogy/materials/0001.md"
    ]


def test_oracle_cannot_reintroduce_physically_excluded_same_origin_material(case, corpus):
    config = configure(case, corpus, mode="oracle", files=["tasks/origin/materials/0001.md"])
    with pytest.raises(FileNotFoundError):
        build_profile(case["root"], case["spec"], Group.G2, config)


def test_explicit_condition_requires_a_real_skill_and_oracle_requires_confined_paths(case, corpus):
    config = configure(case, corpus, mode="explicit")
    with pytest.raises(ValueError, match="installed domain skill"):
        build_profile(case["root"], case["spec"], Group.G2, config)
    with pytest.raises(ValueError, match="confined relative"):
        GroupConfig(knowledge_mode="oracle", oracle_context_files=["../oracle/gold_patch.diff"])
    with pytest.raises(ValueError, match="requires explicitly selected"):
        GroupConfig(knowledge_mode="oracle")
