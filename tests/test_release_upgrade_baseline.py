"""Resolve a released Python subproject through the actual Git and uv tools."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))
from upgrade_acceptance import baseline_install_command, baseline_requirement


def test_old_adapter_baseline_resolves_its_nested_python_project(tmp_path, monkeypatch):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def run(*args, cwd=None):
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10, check=False, **flags,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    for variable in ("USERPROFILE", "MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME"):
        home = tmp_path / variable
        home.mkdir()
        monkeypatch.setenv(variable, str(home))
    source = tmp_path / "source"
    project = "plugins/mindie-agent"
    nested = source / project
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text(
        '[project]\nname = "released-layout-fixture"\nversion = "0.1.21"\n'
        'requires-python = ">=3.11"\n',
        encoding="utf-8", newline="\n",
    )
    run("git", "init", cwd=source)
    run("git", "add", ".", cwd=source)
    run("git", "-c", "user.name=Release Fixture", "-c", "user.email=release@example.invalid",
        "commit", "-m", "Published baseline", cwd=source)
    revision = run("git", "rev-parse", "HEAD", cwd=source)
    repository = "mindie-agent/released-layout-fixture"
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{source.as_uri()}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"https://github.com/{repository}")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    requirement = baseline_requirement(repository, dict(
        package="released-layout-fixture", revision=revision, project=project,
    ))
    overrides = tmp_path / "baseline.txt"
    overrides.write_text(requirement + "\n", encoding="utf-8", newline="\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    wrong = f"released-layout-fixture @ {(tmp_path / 'unpublished').as_uri()}"
    (candidate / "pyproject.toml").write_text(
        "[tool.uv]\noverride-dependencies = " + json.dumps([wrong]) + "\n",
        encoding="utf-8", newline="\n",
    )
    command = baseline_install_command(sys.executable, overrides, [requirement])
    run(*map(str, command), "--dry-run", "--no-index", "--no-deps", cwd=candidate)
