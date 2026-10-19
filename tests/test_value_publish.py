import importlib.util
import json
from pathlib import Path

import pytest
from test_value_release import evidence

from mindie_bench.io import write_json
from mindie_bench.value_publish import write_component_views
from mindie_bench.value_release import evaluate


def prepared(tmp_path):
    suite, observations = evidence()
    target = {
        "repository": "mindie-agent/mindie-agent-codex", "version": "0.0.2",
        "revision": observations["component_revisions"]["codex"],
    }
    manifest = {"product_version": "0.0.2", "repositories": [target]}
    for name, value in (("suite", suite), ("evidence", observations), ("manifest", manifest)):
        write_json(tmp_path / f"{name}.json", value)
    return suite, observations, manifest


def test_component_attachment_preserves_negative_result_and_exact_source(tmp_path):
    suite, observations, manifest = prepared(tmp_path)
    report = evaluate(suite, observations)
    files = write_component_views(report, manifest, tmp_path / "views")
    data = json.loads(files[0].read_text())
    assert data["component_revision"] == manifest["repositories"][0]["revision"]
    assert data["studies"][0]["comparisons"][0]["wall_clock_speedup_on_mutual_success"] == 0.5
    assert "0.500x" in files[1].read_text()
    manifest["repositories"][0]["revision"] = "another-source"
    with pytest.raises(ValueError, match="attachment source differs"):
        write_component_views(report, manifest, tmp_path / "wrong")


def test_attachment_preparation_is_read_only_and_reuses_only_identical_assets(tmp_path, monkeypatch):
    suite, observations, manifest = prepared(tmp_path)
    scripts = Path(__file__).resolve().parents[1] / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("attach_value_test", scripts / "attach_product_value.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    target = manifest["repositories"][0]
    release = {"draft": False, "prerelease": False, "assets": []}
    binaries = {}

    def fetch(path, *, binary=False):
        if binary:
            return binaries[path]
        if "/git/ref/" in path:
            return {"object": {"type": "commit", "sha": target["revision"]}}
        return release

    def unexpected_write(*args, **kwargs):
        pytest.fail("read-only preparation attempted an external write")

    monkeypatch.setattr(publisher.subprocess, "run", unexpected_write)
    collector = dict(observations["runner_release"], dirty=False, editable=False)
    monkeypatch.setattr("mindie_bench.value_controls.runner_source", lambda: collector)
    options = {
        name: tmp_path / f"{name}.json" for name in ("suite", "evidence", "manifest")
    } | {"output": tmp_path / "views", "fetch": fetch}
    collector["revision"] = "different-collector"
    with pytest.raises(ValueError, match="loaded collector source"):
        publisher.attach(**options)
    collector["revision"] = observations["runner_release"]["revision"]
    planned = publisher.attach(**options)
    assert [item["status"] for item in planned] == ["ready", "ready"]
    with pytest.raises(ValueError, match="missing value attachment"):
        publisher.attach(**options, require_existing=True)
    for index, item in enumerate(planned):
        release["assets"].append({"name": item["asset"], "id": index})
        binaries[f"repos/{target['repository']}/releases/assets/{index}"] = (
            tmp_path / "views/codex" / item["asset"]
        ).read_bytes()
    reused = publisher.attach(**options, publish=True)
    assert [item["status"] for item in reused] == ["already_present", "already_present"]
    verified = publisher.attach(**options, require_existing=True)
    assert verified == reused
    release["assets"].pop()
    with pytest.raises(ValueError, match="missing value attachment"):
        publisher.attach(**options, require_existing=True)
    release["assets"].append({"name": planned[1]["asset"], "id": 1})
    binaries[f"repos/{target['repository']}/releases/assets/0"] = b"older observation"
    with pytest.raises(ValueError, match="attachment differs"):
        publisher.attach(**options, publish=True)
    with pytest.raises(ValueError, match="attachment differs"):
        publisher.attach(**options, require_existing=True)
