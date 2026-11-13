"""Real native declarations produce assets accepted by the released selector."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))
from consumer_contract import load_consumer
from prepare_plugin_metadata import prepare

FIXTURES = Path(__file__).with_name("fixtures") / "plugin-release"
CONSUMER = Path(importlib.util.find_spec("mindie_knowledge").origin).parent.parent
VERSION = "0.1.27"


@pytest.fixture
def source(tmp_path, monkeypatch):
    for name in ("MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path / name))
    root = tmp_path / "source"
    root.mkdir()
    return root


def commit(source, declaration, payload):
    path = source / declaration
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    for arguments in (
        ["init"], ["config", "core.autocrlf", "false"], ["add", "."],
        ["-c", "user.name=Release Fixture", "-c", "user.email=release@example.invalid",
         "commit", "-m", "Published native declaration"],
    ):
        subprocess.run(
            ["git", "-C", str(source), *arguments], check=True, capture_output=True,
            timeout=10, **flags,
        )
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], timeout=10, **flags,
    ).decode().strip()


def candidate(source, tmp_path, harness="workbuddy", payload=None):
    load_consumer(CONSUMER)
    from mindie_knowledge.releases import REPOSITORIES
    from mindie_knowledge.releases_io import _COMPATIBILITY_PATHS
    declaration = _COMPATIBILITY_PATHS[harness]
    revision = commit(source, declaration, payload or (FIXTURES / f"{harness}.json").read_bytes())
    return dict(
        consumer=CONSUMER, repository=REPOSITORIES[harness], version=VERSION,
        revision=revision, source=source, assets=tmp_path / "assets",
    )


@pytest.mark.parametrize("harness", [
    "codex", "claude-code", "kimi-code", "dsh", "antigravity", "workbuddy", "opencode",
])
def test_each_native_layout_generates_an_asset_the_old_selector_accepts(
        source, tmp_path, harness):
    from mindie_knowledge.releases import select_release
    options = candidate(source, tmp_path, harness)
    output = prepare(**options)
    raw = output.read_bytes()
    metadata = json.loads(raw)
    declaration = json.loads((FIXTURES / f"{harness}.json").read_bytes())
    assert metadata["compatibility"] == declaration
    selected = select_release(
        [dict(id=1, tag_name="v" + VERSION, draft=False, prerelease=False, metadata=metadata)],
        repository=options["repository"], harness_name=harness,
        harness_version=declaration["supported"][0]["minimum"],
    )
    assert selected["status"] == "candidate"
    assert selected["candidate"]["metadata"]["source"]["sha"] == options["revision"]
    assert b"\r" not in raw
    assert prepare(**options).read_bytes() == raw
    # Existing repository generators may use another JSON layout; preserve valid bytes.
    raw = json.dumps(metadata).encode()
    output.write_bytes(raw)
    assert prepare(**options).read_bytes() == raw


@pytest.mark.parametrize("field", ["version", "source", "compatibility"])
def test_incorrect_existing_asset_is_rejected_without_overwriting_it(source, tmp_path, field):
    options = candidate(source, tmp_path)
    output = prepare(**options)
    metadata = json.loads(output.read_bytes())
    if field == "version":
        metadata["version"] = "0.1.26"
    elif field == "source":
        metadata["source"]["sha"] = "0" * 40
    else:
        metadata["compatibility"]["supported"] = [{
            "minimum": "0.1.0", "maximum": "0.1.0",
            "maximum_inclusive": True, "include_prerelease": True,
        }]
    original = json.dumps(metadata).encode()
    output.write_bytes(original)
    with pytest.raises(ValueError):
        prepare(**options)
    assert output.read_bytes() == original


@pytest.mark.parametrize("defect", ["legacy", "missing", "dirty", "wrong-commit"])
def test_unpublishable_declaration_fails_before_any_asset_write(source, tmp_path, defect):
    options = candidate(
        source, tmp_path,
        payload=b'{"schema":"legacy","runtime_components":{}}' if defect == "legacy" else None,
    )
    declaration = source / "harness-compatibility.json"
    if defect == "missing":
        declaration.unlink()
    elif defect == "dirty":
        declaration.write_bytes(declaration.read_bytes() + b"\n")
    elif defect == "wrong-commit":
        options["revision"] = "0" * 40
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(**options)
    assert not options["assets"].exists()


def test_components_need_no_native_declaration_or_asset(source, tmp_path):
    assert prepare(
        consumer=CONSUMER, repository="mindie-agent/mindie-bench", version="0.4.8",
        revision="0" * 40, source=source, assets=tmp_path / "assets",
    ) is None
    assert not (tmp_path / "assets").exists()
