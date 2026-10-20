import stat

from mindie_bench.models import ExperimentConfig, HarnessConfig
from mindie_bench.runner.profile_builder import build_dsh_profile


def test_dsh_profile_patch_contains_model_mcp_and_skills(tmp_path):
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
    config = ExperimentConfig(
        schema_version="1.0",
        harness=HarnessConfig(
            name="dsh",
            executable=str(fake),
            model="deepseek-flash",
            expected_version="0.2.0-rc.2",
            reasoning_effort="low",
            provider_url="https://api.deepseek.com/anthropic/v1",
            api_key_env="MINDIE_BENCH_RELAY_KEY",
        ),
    )
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
    patch = (root / "dsh-home" / "profiles" / "bench" / "cordis.patch.yml").read_text()
    assert "api: openai-responses" in patch
    assert "provider: bench" in patch
    assert "serverName: \"remote\"" in patch
    assert "reasoning: \"low\"" in patch
    assert env["DSH_HOME"].endswith("dsh-home")
    assert fingerprint["profile"] == "bench"


def test_dsh_profile_adds_skill_row(tmp_path):
    fake = tmp_path / "dsh"
    fake.write_text(
        "#!/bin/sh\n"
        "profile=\"$DSH_HOME/profiles/bench\"\n"
        "mkdir -p \"$profile\"\n"
        "printf '%s\\n' '{\"name\":\"dsh-profile-bench\",\"private\":true,\"dsh\":{\"profile\":{\"bundles\":[]}}}' > \"$profile/package.json\"\n"
        "printf '%s\\n' '[]' > \"$profile/cordis.patch.yml\"\n"
        "echo '[]'\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    config = ExperimentConfig(
        schema_version="1.0",
        harness=HarnessConfig(
            name="dsh",
            executable=str(fake),
            model="deepseek-flash",
            expected_version="0.2.0-rc.2",
            provider_url="https://api.deepseek.com/anthropic/v1",
            api_key_env="MINDIE_BENCH_RELAY_KEY",
        ),
    )
    root = tmp_path / "profile"
    root.mkdir()
    (root / "skills" / "mindie-agent").mkdir(parents=True)
    (root / "skills" / "mindie-agent" / "SKILL.md").write_text("---\nname: mindie-agent\n---\n")
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    build_dsh_profile(root, env, config, {}, {})
    patch = (root / "dsh-home" / "profiles" / "bench" / "cordis.patch.yml").read_text()
    assert "dsh-skill-filesystem" in patch
    assert "customSkillDirs" in patch


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
    fake = tmp_path / "dsh"
    fake.write_text(
        "#!/bin/sh\n"
        "profile=\"$DSH_HOME/profiles/bench\"\n"
        "mkdir -p \"$profile\"\n"
        "printf '%s\\n' '{\"name\":\"dsh-profile-bench\",\"private\":true,\"dsh\":{\"profile\":{\"bundles\":[]}}}' > \"$profile/package.json\"\n"
        "printf '%s\\n' '[]' > \"$profile/cordis.patch.yml\"\n"
        "echo '[]'\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    config = ExperimentConfig(
        schema_version="1.0",
        harness=HarnessConfig(
            name="dsh",
            executable=str(fake),
            model="deepseek-flash",
            expected_version="0.2.0-rc.2",
            provider_url="https://api.deepseek.com/anthropic/v1",
            api_key_env="MINDIE_BENCH_RELAY_KEY",
        ),
    )
    root = tmp_path / "profile"
    root.mkdir()
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    build_dsh_profile(root, env, config, {}, {}, catalog_checkpoint=True)
    patch = (root / "dsh-home" / "profiles" / "bench" / "cordis.patch.yml").read_text()
    assert "bench-catalog-checkpoint" in patch
    assert "catalog_checkpoint.mjs" in patch
    assert ".mindie003" not in patch
    assert env["MINDIE_BENCH_CATALOG_CHECKPOINT_OUT"].endswith("catalog-checkpoint.json")
