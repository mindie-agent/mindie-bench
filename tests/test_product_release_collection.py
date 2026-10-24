"""Formal-release transport must preserve the existing benchmark evidence gate."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import test_release_benchmarks as fixtures

benchmark_template = fixtures.benchmark_template
workspace = fixtures.workspace

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/collect_product_benchmarks.py"
spec = importlib.util.spec_from_file_location("product_release_collector", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


@pytest.fixture
def release_evidence(tmp_path, workspace):
    report = fixtures.run(tmp_path, workspace)
    revision = report["source"]["revision"]
    tag = dict(object=dict(type="commit", sha=revision))
    assets = {}
    descriptors = []
    for index, name in enumerate(collector.ASSETS):
        assets[f"repos/mindie-agent/example/releases/assets/{index}"] = (
            tmp_path / "output" / name
        ).read_bytes()
        descriptors.append(dict(id=index, name=name))
    release = dict(
        id=5, tag_name="v0.0.2", draft=False, prerelease=False,
        html_url="https://github.com/mindie-agent/example/releases/tag/v0.0.2",
        assets=descriptors,
    )
    payloads = {
        "repos/mindie-agent/example/releases/tags/v0.0.2": release,
        "repos/mindie-agent/example/git/ref/tags/v0.0.2": dict(
            object=dict(type="tag", sha="b" * 40),
        ),
        f"repos/mindie-agent/example/git/tags/{'b' * 40}": tag,
        "repos/mindie-agent/mindie-agent/git/ref/tags/v0.0.2": tag,
        **assets,
    }

    def fetch(path, *, binary=False):
        value = payloads[path]
        assert isinstance(value, bytes) if binary else isinstance(value, dict)
        return value

    kwargs = dict(
        repository="mindie-agent/mindie-agent", version="0.0.2", revision=revision,
        manifest=fixtures.manifest(tmp_path, report), output=tmp_path / "product", fetch=fetch,
    )
    return kwargs, payloads


def test_verified_assets_collect_through_the_real_runner(release_evidence):
    kwargs, payloads = release_evidence
    original_fetch = kwargs["fetch"]

    def download_updates_count(path, *, binary=False):
        if binary:
            release = payloads["repos/mindie-agent/example/releases/tags/v0.0.2"]
            for asset in release["assets"]:
                asset["download_count"] = asset.get("download_count", 0) + 1
        return original_fetch(path, binary=binary)

    kwargs["fetch"] = download_updates_count
    result = collector.collect_product(**kwargs)
    assert result["release_gate"]
    assert result["reports"][0]["source"]["revision"] == kwargs["revision"]
    output = kwargs["output"]
    provenance = json.loads((output / "product-release-provenance.json").read_text())
    assert provenance["releases"][0]["release_id"] == 5
    assert provenance["releases"][0]["revision"] == kwargs["revision"]
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert not list(output.rglob("private"))


def product_channel(kwargs):
    manifest = json.loads(kwargs["manifest"].read_text())
    path = kwargs["manifest"].parent / "channel.json"
    path.write_text(json.dumps({
        "schema": "mindie-release-channel/1",
        "current_patch_version": kwargs["version"],
        "components": {str(index): {
            "repository": item["repository"], "version": item["version"],
            "tag": "v" + item["version"], "commit": item["revision"],
            "revision": item["revision"],
        } for index, item in enumerate(manifest["repositories"])},
    }))
    return path


def test_channel_release_collects_actual_component_evidence_without_value_claims(release_evidence):
    kwargs, _ = release_evidence
    channel = product_channel(kwargs)
    result = collector.collect_product(**kwargs, channel=channel)
    assert result["release_gate"]
    output = kwargs["output"]
    assert (output / "release-channel.json").read_bytes() == channel.read_bytes()
    provenance = json.loads((output / "product-release-provenance.json").read_text())
    assert provenance["channel_sha256"] == hashlib.sha256(channel.read_bytes()).hexdigest()
    assert provenance["releases"][0]["revision"] == result["reports"][0]["source"]["revision"]
    assert not list(output.glob("*value*"))


def test_stale_update_channel_cannot_publish_with_new_component_reports(release_evidence):
    kwargs, _ = release_evidence
    channel = product_channel(kwargs)
    data = json.loads(channel.read_text())
    data["components"]["0"]["commit"] = "c" * 40
    channel.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="channel identity differs"):
        collector.collect_product(**kwargs, channel=channel)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("failure", ["draft", "tag", "asset", "checksum", "report", "platform"])
def test_missing_or_stale_publication_evidence_blocks_collection(release_evidence, failure):
    kwargs, payloads = release_evidence
    release = payloads["repos/mindie-agent/example/releases/tags/v0.0.2"]
    if failure == "draft":
        release["draft"] = True
    elif failure == "tag":
        payloads[f"repos/mindie-agent/example/git/tags/{'b' * 40}"] = dict(
            object=dict(type="commit", sha="c" * 40),
        )
    elif failure == "asset":
        release["assets"].pop()
    elif failure == "checksum":
        payloads["repos/mindie-agent/example/releases/assets/1"] = b"altered report"
    elif failure == "report":
        path = "repos/mindie-agent/example/releases/assets/0"
        report = json.loads(payloads[path])
        report["version"] = "0.0.1"
        payloads[path] = json.dumps(report).encode()
        sums = "repos/mindie-agent/example/releases/assets/2"
        lines = payloads[sums].decode().splitlines()
        lines[0] = hashlib.sha256(payloads[path]).hexdigest() + "  benchmark-results.json"
        payloads[sums] = ("\n".join(lines) + "\n").encode()
    else:
        manifest_path = kwargs["manifest"]
        value = json.loads(manifest_path.read_text())
        value["repositories"][0]["platforms"] = ["unmeasured-platform"]
        manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        collector.collect_product(**kwargs)
    assert not (kwargs["output"] / "SHA256SUMS").exists()
