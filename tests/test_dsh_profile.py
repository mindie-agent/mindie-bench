import stat

import yaml

from mindie_bench.models import ExperimentConfig, HarnessConfig
from mindie_bench.runner.profile_builder import DSH_REASONING_EFFORTS, build_dsh_profile


class _PatchLoader(yaml.SafeLoader):
    pass


_PatchLoader.add_constructor(
    "tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node)
)


def _patch(root):
    text = (root / "dsh-home" / "profiles" / "bench" / "cordis.patch.yml").read_text()
    rows = yaml.load(text, Loader=_PatchLoader)
    flattened = [row for row in rows if "id" in row]
    for row in rows:
        flattened.extend(item for item in (row.get("insert") or []) if "id" in item)
    return text, {row["id"]: row for row in flattened}


def _write_fake_dsh(tmp_path):
    fake = tmp_path / "dsh"
    fake.write_text(
        "#!/bin/sh\n"
        "set -e\n"
        "profile=\"$DSH_HOME/profiles/bench\"\n"
        "mkdir -p \"$profile\"\n"
        "printf '%s\\n' '{\"name\":\"dsh-profile-bench\",\"private\":true,\"dsh\":{\"profile\":{\"bundles\":[\"@deepseek-ai/dsh-base\",\"@deepseek-ai/dsh-headless\"]}}}' > \"$profile/package.json\"\n"
        "printf '%s\\n' '[]' > \"$profile/cordis.patch.yml\"\n"
        "echo '[]'\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _config(fake, **updates):
    values = dict(
        name="dsh",
        executable=str(fake),
        model="deepseek-flash",
        expected_version="0.2.0-rc.2",
        provider_url="https://api.deepseek.com/anthropic/v1",
        api_key_env="MINDIE_BENCH_RELAY_KEY",
    )
    values.update(updates)
    return ExperimentConfig(
        schema_version="1.0",
        harness=HarnessConfig(**values),
    )


def test_dsh_profile_patch_contains_model_mcp_and_skills(tmp_path):
    fake = _write_fake_dsh(tmp_path)
    config = _config(fake, reasoning_effort="low")
    root = tmp_path / "profile"
    root.mkdir()
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(root / "home")}
    mcp = {
        "remote": {
            "command": "/opt/homebrew/bin/python",
            "args": ["-m", "remote_dev.mcp.server"],
            "env": {"REMOTE_DEV_STATE_DIR": str(root / "remote")},
        }
    }
    fingerprint = build_dsh_profile(root, env, config, mcp, {})
    patch, rows = _patch(root)
    assert "api: openai-responses" in patch
    assert "serverName: \"remote\"" in patch
    model = rows["llm-pi-ai"]["config"]["providers"]["bench"]["models"][0]
    assert model["id"] == "deepseek-flash"
    assert model["reasoningEfforts"] == DSH_REASONING_EFFORTS
    assert rows["agent-default-model"]["config"]["provider"] == "bench"
    assert rows["agent-default-model"]["config"]["reasoningEffort"] == "low"
    assert rows["tools"] == {
        "id": "tools",
        "name": "@deepseek-ai/dsh-tools",
        "config": {"mode": "native"},
    }
    assert env["DSH_HOME"].endswith("dsh-home")
    assert fingerprint["profile"] == "bench"


def test_dsh_profile_adds_skill_row_and_defaults_to_max_effort(tmp_path):
    fake = _write_fake_dsh(tmp_path)
    config = _config(fake)
    root = tmp_path / "profile"
    root.mkdir()
    (root / "skills" / "mindie-agent").mkdir(parents=True)
    (root / "skills" / "mindie-agent" / "SKILL.md").write_text("---\nname: mindie-agent\n---\n")
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    build_dsh_profile(root, env, config, {}, {})
    patch, rows = _patch(root)
    assert "dsh-skill-filesystem" in patch
    assert "customSkillDirs" in patch
    assert rows["agent-default-model"]["config"]["reasoningEffort"] == "max"
    model = rows["llm-pi-ai"]["config"]["providers"]["bench"]["models"][0]
    assert model["reasoningEfforts"]["off"] is None
    assert model["reasoningEfforts"]["max"] == "max"
    assert rows["tools"]["config"] == {"mode": "native"}


def test_catalog_checkpoint_row_is_accounted_and_p0_fails_closed(tmp_path):
    import pytest

    from mindie_bench.runner.profile_builder import assert_catalog_policy

    assert_catalog_policy(
        {
            "checkpoint_row_present": True,
            "mindie_identities": {"tools": [], "skills": [], "loader_rows": []},
        },
        "P0_bare_host",
    )
    with pytest.raises(ValueError, match="P0 mounted catalog"):
        assert_catalog_policy(
            {
                "checkpoint_row_present": True,
                "mindie_identities": {"tools": ["mindie_entry"], "skills": [], "loader_rows": []},
            },
            "P0_bare_host",
        )
    with pytest.raises(ValueError, match="checkpoint row"):
        assert_catalog_policy({"checkpoint_row_present": False, "mindie_identities": {}}, "P1_v002_baseline")


def test_catalog_checkpoint_row_uses_packaged_support_file(tmp_path):
    fake = _write_fake_dsh(tmp_path)
    config = _config(fake)
    root = tmp_path / "profile"
    root.mkdir()
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    build_dsh_profile(root, env, config, {}, {}, catalog_checkpoint=True)
    patch, rows = _patch(root)
    assert "bench-catalog-checkpoint" in rows
    assert "catalog_checkpoint.mjs" in patch
    assert ".mindie003" not in patch
    assert env["MINDIE_BENCH_CATALOG_CHECKPOINT_OUT"].endswith("catalog-checkpoint.json")
