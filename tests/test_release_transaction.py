"""Publication interleavings use the real released consumer and captured evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("release_transaction", SCRIPTS / "publish_release.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
FIXTURE = Path(__file__).with_name("fixtures") / "release-transaction"
REVISION = "b616a38f77ca17573e4ed22ee23c08c09adefac7"
VERSION = "0.13.2"
CONSUMER = Path(importlib.util.find_spec("mindie_knowledge").origin).parent.parent


@pytest.fixture
def bundle(tmp_path):
    for name in ("benchmark-results.json", "benchmark-results.md", "benchmark-SHA256SUMS"):
        (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())
    (tmp_path / "mindie_knowledge-0.13.2-py3-none-any.whl").write_bytes(b"wheel payload")
    (tmp_path / "mindie_knowledge-0.13.2.tar.gz").write_bytes(b"sdist payload")
    (tmp_path / "upgrade-acceptance.json").write_text(json.dumps(dict(
        schema="mindie-upgrade-acceptance/1", status="passed",
        repository="mindie-agent/knowledge", version=VERSION, revision=REVISION,
        installed_revision=REVISION, consumer_revision="ac4a89e993c89deba5a4a629e1cd89d61e57c843",
    )))
    write_sums(tmp_path)
    return tmp_path


def write_sums(root):
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(root.iterdir()) if path.name != "SHA256SUMS"))


class ReleaseAPI:
    repository = "mindie-agent/knowledge"
    prefix = "repos/" + repository

    def __init__(self):
        self.current = None
        self.payloads = {}
        self.events = []
        self.disconnect = None
        self.corrupt = None
        self.serial = 0

    def release(self, tag):
        return copy.deepcopy(self.current)

    def api(self, path):
        if self.current is not None and path == f"{self.prefix}/releases/{self.current['id']}":
            return copy.deepcopy(self.current)
        if "/releases?" in path:
            return []
        if "/git/ref/" in path:
            file = "ref.json"
        elif "/git/tags/" in path:
            file = "tag.json"
        else:
            file = "source.json"
        return json.loads((FIXTURE / file).read_bytes())

    def create(self, tag, revision, body):
        self.events.append("create-draft")
        self.current = dict(id=1, tag_name=tag, target_commitish=revision, body=body,
                            draft=True, prerelease=False, immutable=False, assets=[])
        self.drop("create")
        return copy.deepcopy(self.current)

    def upload(self, release_id, path):
        assert self.current["draft"], "upload exposed partial public assets"
        assert self.current["id"] == release_id
        self.events.append("upload:" + path.name)
        self.payloads[path.name] = path.read_bytes()
        self.serial += 1
        self.current["assets"].append(dict(
            id=self.serial, name=path.name, size=path.stat().st_size, state="uploaded"))
        self.drop("upload")

    def delete_asset(self, asset_id):
        assert self.current["draft"], "a public artifact was changed"
        row = next(row for row in self.current["assets"] if row["id"] == asset_id)
        self.events.append("delete:" + row["name"])
        del self.payloads[row["name"]]
        self.current["assets"].remove(row)
        self.drop("delete")

    def asset(self, row):
        return self.payloads[row["name"]]

    def asset_digest(self, row):
        raw = self.asset(row)
        if row["name"] == self.corrupt:
            raw += b"corrupt"
        return hashlib.sha256(raw).hexdigest()

    def publish(self, ident):
        self.events.append("publish")
        self.current.update(draft=False, immutable=True)
        self.drop("publish")
        return self.release(self.current["tag_name"])

    def create_tag(self, version, revision):
        assert self.current["draft"]
        self.events.append("create-tag")

    def require_new_version(self, version):
        sys.modules["release_transport"].GitHub.require_new_version(self, version)

    def drop(self, stage):
        if self.disconnect == stage:
            self.disconnect = None
            raise TimeoutError("response lost after the remote effect")


def invoke(api, bundle):
    return publisher.publish(api, bundle, consumer=CONSUMER, version=VERSION, revision=REVISION)


def test_complete_bundle_is_verified_with_the_real_consumer_before_it_becomes_public(bundle):
    api = ReleaseAPI()
    assert invoke(api, bundle)["status"] == "published"
    assert api.events[0] == "create-draft" and api.events[-1] == "publish"
    assert len(api.current["assets"]) == len(list(bundle.iterdir()))
    before = list(api.events)
    assert invoke(api, bundle)["status"] == "already_published"
    assert api.events == before


def test_untagged_private_draft_uses_its_created_id_when_tag_lookup_is_not_visible(bundle):
    class UntaggedDraftAPI(ReleaseAPI):
        def release(self, tag):
            if self.current and self.current["draft"]:
                return None
            return super().release(tag)

    api = UntaggedDraftAPI()
    assert invoke(api, bundle)["status"] == "published"
    assert api.events.count("create-draft") == api.events.count("publish") == 1


def test_cli_without_forwarded_key_fails_before_any_draft_transaction(bundle):
    env = dict(os.environ)
    env.pop("MINDIE_RELEASE_DEPLOY_KEY", None)
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run([
        sys.executable, str(SCRIPTS / "publish_release.py"),
        "--repository", "mindie-agent/knowledge", "--version", VERSION,
        "--revision", REVISION, "--assets", str(bundle), "--consumer", str(CONSUMER),
    ], env=env, capture_output=True, text=True, check=False, timeout=10, **flags)
    assert result.returncode != 0
    assert "main-only release deploy key is unavailable" in result.stderr


@pytest.mark.parametrize("stage", ["create", "upload", "publish"])
def test_lost_response_reconciles_the_effect_instead_of_replaying_it(bundle, stage):
    api = ReleaseAPI()
    api.disconnect = stage
    assert invoke(api, bundle)["status"] == "published"
    assert api.events.count("create-draft") == api.events.count("publish") == 1
    assert len([event for event in api.events if event.startswith("upload:")]) == len(api.payloads)


def test_a_failed_upgrade_never_creates_a_draft_or_tag(bundle):
    (bundle / "upgrade-acceptance.json").write_text('{"status":"failed"}')
    write_sums(bundle)
    api = ReleaseAPI()
    with pytest.raises(ValueError, match="old-client upgrade evidence"):
        invoke(api, bundle)
    assert not api.events and api.current is None


def test_shared_component_evidence_cannot_publish_a_native_plugin(bundle):
    api = ReleaseAPI()
    api.repository = "mindie-agent/mindie-agent-codex"
    with pytest.raises(ValueError, match="missing benchmark or real upgrade evidence"):
        invoke(api, bundle)
    assert not api.events and api.current is None


def test_an_incomplete_existing_formal_release_is_not_overwritten_or_reported_successful(bundle):
    api = ReleaseAPI()
    api.current = dict(id=1, tag_name="v" + VERSION, draft=False, prerelease=False,
                       immutable=False, assets=[])
    with pytest.raises(ValueError, match="missing benchmark or real upgrade evidence"):
        invoke(api, bundle)
    assert not api.events


def test_corrupt_uploaded_package_prevents_publication(bundle):
    api = ReleaseAPI()
    api.corrupt = "mindie_knowledge-0.13.2-py3-none-any.whl"
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        invoke(api, bundle)
    assert api.current["draft"] and "publish" not in api.events


@pytest.mark.parametrize("lost_delete_response", [False, True])
def test_retry_refreshes_only_the_owned_private_draft_and_reconciles_deletion(
        bundle, lost_delete_response):
    api = ReleaseAPI()
    api.corrupt = "mindie_knowledge-0.13.2-py3-none-any.whl"
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        invoke(api, bundle)
    path = bundle / "upgrade-acceptance.json"
    report = json.loads(path.read_bytes())
    report["retry_measurement"] = "new successful run of the same source"
    path.write_text(json.dumps(report))
    write_sums(bundle)
    api.corrupt = None
    api.disconnect = "delete" if lost_delete_response else None
    assert invoke(api, bundle)["status"] == "published"
    assert api.events.count("create-draft") == api.events.count("publish") == 1
    assert api.events.count("delete:upgrade-acceptance.json") == 1
    assert api.payloads["upgrade-acceptance.json"] == path.read_bytes()
    assert len(api.current["assets"]) == len(api.payloads) == len(list(bundle.iterdir()))


def test_missing_sdist_is_rejected_by_the_unmodified_consumer(bundle):
    (bundle / "mindie_knowledge-0.13.2.tar.gz").unlink()
    write_sums(bundle)
    api = ReleaseAPI()
    with pytest.raises(ValueError, match="artifacts are incomplete"):
        invoke(api, bundle)
    assert api.current["draft"] and "publish" not in api.events


def test_a_foreign_draft_cannot_be_adopted_by_the_publisher(bundle):
    api = ReleaseAPI()
    api.current = dict(id=1, tag_name="v" + VERSION, target_commitish="main",
                       draft=True, prerelease=False, body="operator draft", assets=[])
    with pytest.raises(ValueError, match="not owned"):
        invoke(api, bundle)
    assert not api.events


def test_an_uncreated_tag_404_does_not_hide_its_existing_draft(monkeypatch):
    transport = sys.modules["release_transport"]
    calls = []
    draft = dict(id=440168198, draft=True,
                 tag_name="mindie-release-transaction-probe-20261110",
                 target_commitish="7a9ec71f2753e7e0a038a24622f476afc26910a5")

    def respond(argv, **kwargs):
        path = argv[2]
        calls.append(path)
        if "/releases/tags/" in path:
            return subprocess.CompletedProcess(argv, 1, b'{"status":"404"}', b"Not Found")
        return subprocess.CompletedProcess(argv, 0, json.dumps([draft]).encode(), b"")

    monkeypatch.setattr(transport.subprocess, "run", respond)
    api = transport.GitHub("mindie-agent/mindie-bench")
    assert api.release(draft["tag_name"]) == draft
    assert len(calls) == 2


def test_a_network_failure_is_not_permission_to_create_another_release(monkeypatch):
    transport = sys.modules["release_transport"]
    monkeypatch.setattr(transport.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 1, b"", b"connection reset"))
    with pytest.raises(RuntimeError, match="connection reset"):
        transport.GitHub("mindie-agent/knowledge").release("v0.13.2")


def test_late_publication_of_an_older_version_cannot_replace_latest(bundle):
    api = ReleaseAPI()
    original = api.api
    api.api = lambda path: ([dict(tag_name="v0.13.3", draft=False, prerelease=False)]
                            if "/releases?" in path else original(path))
    with pytest.raises(ValueError, match="advance the highest formal version"):
        invoke(api, bundle)
    assert not api.events


def test_runner_audit_uses_the_actual_non_admin_ruleset_view():
    import release_policy

    rows = json.loads((FIXTURE / "public-rulesets.json").read_bytes())

    class PublicAPI:
        prefix = "repos/mindie-agent/mindie-bench"

        def api(self, path):
            return rows[path]

    assert release_policy.verify(PublicAPI(), workflow=True)["tag_creation"] == "restricted"
    with pytest.raises(KeyError, match="immutable-releases"):
        release_policy.verify(PublicAPI())
    create = next(row for row in rows.values()
                  if isinstance(row, dict) and row.get("name") == "mindie-release-create")
    create["enforcement"] = "disabled"
    with pytest.raises(ValueError, match="publication policy is missing or differs"):
        release_policy.verify(PublicAPI(), workflow=True)


def test_release_entry_requires_forwarding_the_environment_key(tmp_path):
    import release_policy

    workflows = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    path = workflows / "release.yml"
    declaration = (
        "on:\n  workflow_dispatch:\njobs:\n  release:\n"
        "    uses: mindie-agent/mindie-bench/.github/workflows/publish-release.yml@main\n"
    )
    path.write_text(declaration, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="explicitly forward"):
        release_policy.verify_entry(tmp_path, "mindie-agent/sim")
    path.write_text(
        declaration + "    secrets:\n"
        "      MINDIE_RELEASE_DEPLOY_KEY: ${{ secrets.MINDIE_RELEASE_DEPLOY_KEY }}\n",
        encoding="utf-8", newline="\n")
    release_policy.verify_entry(tmp_path, "mindie-agent/sim")
