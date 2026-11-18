from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.runner.mcp_probe import supports_default_capability
from mindie_bench.scorer.target_access import _validation_tail, target_access_observations

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load("build_wheels_bundle")
public_verify = _load("verify_public_release")
deps_check = _load("check_release_dependencies")
acceptance = _load("native_upgrade_acceptance")


@pytest.mark.tier0
def test_release_workflows_enable_uv_cache_and_fast_verify_public():
    for rel in (
        ".github/workflows/publish-release.yml",
        ".github/workflows/release-benchmark.yml",
        ".github/workflows/publish-product.yml",
    ):
        wf = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        uv_steps = [
            step
            for job in wf["jobs"].values()
            for step in job.get("steps", [])
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
        ]
        assert uv_steps, f"{rel} has no setup-uv step"
        for step in uv_steps:
            assert step["with"]["enable-cache"] is True
            assert "**/.github/workflows/release.yml" in step["with"]["cache-dependency-glob"]

    prod = yaml.safe_load((ROOT / ".github/workflows/publish-product.yml").read_text(encoding="utf-8"))
    checkouts = [
        step for step in prod["jobs"]["publish"]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkouts and all(step["with"].get("persist-credentials") is False for step in checkouts)

    pub = yaml.safe_load((ROOT / ".github/workflows/publish-release.yml").read_text(encoding="utf-8"))
    vp = pub["jobs"]["verify-public"]
    assert vp["timeout-minutes"] <= 15
    runs = "\n".join(step.get("run", "") for step in vp["steps"])
    assert "verify_public_release.py" in runs
    assert "upgrade_acceptance.py" not in runs
    assert "native_upgrade_acceptance.py" not in runs


@pytest.mark.tier0
def test_verify_public_release_validates_cdn_assets_and_sha256sums(tmp_path):
    rev = "d" * 40
    assets = {
        "mindie_harness-0.1.3-py3-none-any.whl": b"wheel-payload",
        "benchmark-results.json": json.dumps({"release_gate": True}).encode("utf-8"),
        "benchmark-results.md": b"# bench\n",
        "benchmark-SHA256SUMS": b"sums\n",
        "upgrade-acceptance.json": json.dumps({"status": "passed"}).encode("utf-8"),
    }
    sums_txt = "".join(
        f"{hashlib.sha256(raw).hexdigest()}  {name}\n" for name, raw in sorted(assets.items())
    ).encode("utf-8")
    seen_urls: list[str] = []

    def fake_run(argv, *, timeout=10.0):
        cmd = [str(x) for x in argv]
        if cmd[:3] == ["gh", "release", "view"]:
            return json.dumps({
                "tagName": "v0.1.3",
                "isDraft": False,
                "isPrerelease": False,
                "url": "https://github.com/mindie-agent/harness/releases/tag/v0.1.3",
                "assets": [{"name": k} for k in [*assets, "SHA256SUMS"]],
            })
        if cmd[:2] == ["gh", "api"]:
            return json.dumps({"object": {"type": "commit", "sha": rev}})
        raise AssertionError(f"unexpected command: {cmd}")

    def fake_dl_url(url: str, dest: Path, *, timeout: float):
        seen_urls.append(url)
        name = url.rsplit("/", 1)[-1]
        dest.write_bytes(sums_txt if name == "SHA256SUMS" else assets[name])

    out = tmp_path / "public-upgrade.json"
    report = public_verify.verify_public_release(
        repository="mindie-agent/harness",
        version="0.1.3",
        revision=rev,
        work_dir=tmp_path / "dl",
        output=out,
        run_fn=fake_run,
        download_url_fn=fake_dl_url,
    )
    assert report["status"] == "passed" and report["sha256sums_verified"] is True
    assert all(u.startswith("https://github.com/mindie-agent/harness/releases/download/v0.1.3/") for u in seen_urls)
    assert json.loads(out.read_text(encoding="utf-8"))["assets"] == sorted(assets)

    # Tampered asset bytes must fail closed
    assets["mindie_harness-0.1.3-py3-none-any.whl"] = b"corrupted"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        public_verify.verify_public_release(
            repository="mindie-agent/harness",
            version="0.1.3",
            revision=rev,
            work_dir=tmp_path / "dl-bad",
            output=tmp_path / "bad.json",
            run_fn=fake_run,
            download_url_fn=fake_dl_url,
        )


@pytest.mark.tier0
def test_check_release_dependencies_accepts_official_github_wheel_receipt(monkeypatch):
    sha = "e" * 64

    class FakeDist:
        version = "0.1.3"

        def read_text(self, filename):
            if filename == "RECORD":
                return ""
            assert filename == "direct_url.json"
            return json.dumps({
                "url": "https://github.com/mindie-agent/harness/releases/download/v0.1.3/mindie_harness-0.1.3-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": sha}},
            })

    def fake_dist(name):
        if name == "mindie-harness":
            return FakeDist()
        raise deps_check.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(deps_check.metadata, "distribution", fake_dist)
    inv = deps_check.installed_dependencies("mindie-agent/mindie-agent-codex")
    assert inv == {
        "mindie-harness": {
            "version": "0.1.3",
            "wheel_name": "mindie_harness-0.1.3-py3-none-any.whl",
            "archive_sha256": sha,
            "record_sha256": None,
        }
    }


@pytest.mark.tier0
def test_native_acceptance_environment_preserves_shared_uv_cache_dir(tmp_path, monkeypatch):
    shared = tmp_path / "shared-uv-cache"
    monkeypatch.setenv("UV_CACHE_DIR", str(shared))
    env = acceptance.environment(tmp_path / "profile", tmp_path / "codex")
    assert env["UV_CACHE_DIR"] == str(shared) and shared.is_dir()


@pytest.mark.tier0
def test_astra_catches_numeric_codex_poll_kill_exclusion_empty_mcp_and_wb_image(tmp_path):
    # 1. Empty MCP server catalog fails supports_default_capability for every component
    for comp in ("remote-dev", "knowledge", "coordinator", "npu-top", "native-tools"):
        assert supports_default_capability(comp, set()) is False
    assert supports_default_capability("coordinator", {"mindie_lease"}) is True

    # 2. Numeric session_id=42 on Codex remote_write_stdin confirms pending job "42"
    edit = ToolCall(
        id="c1", name="remote_exec_command", component="remote-dev",
        arguments={"cmd": "sed -i 's/x/y/' mod.py"}, turn=1, started_sec=0.0, ended_sec=1.0,
        result_observation=ToolResultObservation(
            source="codex_mcp", result_schema="remote-dev.result.v1", classification="pending",
            operation_kind="execution", job_id="42", target_id="h1", remote_status="running",
            classification_reason="explicit_remote_job_pending",
        ),
    )
    poll = ToolCall(
        id="c2", name="remote_write_stdin", component="remote-dev",
        arguments={"session_id": 42.0, "chars": ""}, turn=2, started_sec=1.0, ended_sec=2.0,
        result_observation=ToolResultObservation(
            source="codex_mcp", result_schema="remote-dev.result.v1",
            classification="business_success", operation_kind="observation", job_id="42",
            target_id="h1", remote_status="completed", exit_code=0,
            classification_reason="explicit_remote_command_exit",
        ),
    )
    obs, _ = target_access_observations(Trajectory(calls=[edit, poll]), ["mod.py"])
    assert obs[0]["call_outcome"] == "success"

    # 3. Control call (Action="kill") or different target_id cannot establish first_recorded_success_call_index
    test_cmd = ToolCall(
        id="t1", name="remote_run_command", component="remote-dev",
        arguments={"CommandLine": "pytest tests/test_npu.py"}, turn=1, started_sec=0.0, ended_sec=1.0,
        result_observation=ToolResultObservation(
            source="codex_mcp", result_schema="remote-dev.result.v1", classification="pending",
            operation_kind="execution", job_id="j9", target_id="h1", remote_status="running",
            classification_reason="explicit_remote_job_pending",
        ),
    )
    kill_cmd = ToolCall(
        id="t2", name="remote_manage_task", component="remote-dev",
        arguments={"TaskId": "j9", "Action": "kill"}, turn=2, started_sec=1.0, ended_sec=2.0,
        result_observation=ToolResultObservation(
            source="codex_mcp", result_schema="remote-dev.result.v1",
            classification="business_success", operation_kind="control", job_id="j9",
            target_id="h1", remote_status="cancelled", exit_code=0,
            classification_reason="explicit_remote_command_exit",
        ),
    )
    other_host_poll = ToolCall(
        id="t3", name="remote_manage_task", component="remote-dev",
        arguments={"TaskId": "j9", "Action": "status"}, turn=3, started_sec=2.0, ended_sec=3.0,
        result_observation=ToolResultObservation(
            source="codex_mcp", result_schema="remote-dev.result.v1",
            classification="business_success", operation_kind="observation", job_id="j9",
            target_id="h2", remote_status="completed", exit_code=0,
            classification_reason="explicit_remote_command_exit",
        ),
    )
    assert _validation_tail(Trajectory(calls=[test_cmd, kill_cmd, other_host_poll]))["first_recorded_success_call_index"] is None

    # 4. WorkBuddy remote_ViewImage image_path is extracted as read observation
    img = ToolCall(
        id="v1", name="remote_ViewImage", component="remote-dev",
        arguments={"image_path": "plots/roofline.png"}, turn=1, started_sec=0.0, ended_sec=0.5,
    )
    img_obs, _ = target_access_observations(Trajectory(calls=[img]), ["plots/roofline.png"])
    assert len(img_obs) == 1 and img_obs[0]["kind"] == "read"

    # 5. Mutable @main, <sha>-suffix, or wrong-repo ReMe URL in mindie-knowledge METADATA fails pin check
    from test_wheels_bundle_and_ux_contracts import _build, _component, _wheel
    for bad_req in (
        "reme-ai[as] @ git+https://github.com/agentscope-ai/ReMe.git@main",
        f"reme-ai[as] @ git+https://github.com/agentscope-ai/ReMe.git@{bundle.REME_REVISION}-maintenance",
        f"reme-ai[as] @ git+https://github.com/evil-fork/ReMe.git@{bundle.REME_REVISION}",
    ):
        def bad_dl(repo, ver, dest, req=bad_req):
            pkg = repo.rsplit("/", 1)[-1]
            if pkg != "mindie-knowledge":
                _component(pkg, dest)
                return
            whl = _wheel(dest / "mindie_knowledge-1.2.3-py3-none-any.whl", pkg, "1.2.3", (req,))
            (dest / "SHA256SUMS").write_text(f"{hashlib.sha256(whl.read_bytes()).hexdigest()}  {whl.name}\n")

        sub = tmp_path / hashlib.md5(bad_req.encode()).hexdigest()
        sub.mkdir()
        with pytest.raises(ValueError, match="ReMe pin drift"):
            _build(sub, download_fn=bad_dl)


@pytest.mark.tier0
def test_upgrade_acceptance_birth_component_override_prevents_uv_direct_url_conflict(tmp_path):
    upg = _load("upgrade_acceptance")
    base_toml = (
        '[project]\nname = "mindie-component-runtime"\nversion = "0.0.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["mindie-knowledge @ git+https://github.com/mindie-agent/knowledge@' + "a" * 40 + '", '
        '"mindie-harness @ file:///tmp/candidate"]\n\n'
        '[tool.uv]\npackage = false\n'
        'override-dependencies = ["mindie-knowledge @ git+https://github.com/mindie-agent/knowledge@' + "a" * 40 + '"]\n'
    )
    cand = tmp_path / "harness-cand"
    cand.mkdir()
    patched = upg._with_birth_overrides(base_toml, cand, "mindie-harness")
    import tomllib
    overrides = tomllib.loads(patched)["tool"]["uv"]["override-dependencies"]
    assert f"mindie-harness @ {cand.resolve().as_uri()}" in overrides

    adapter_dir = tmp_path / "cc-cand"
    adapter_dir.mkdir()
    rev = "b" * 40
    (adapter_dir / "pyproject.toml").write_text(
        f'[project]\nname = "mindie-agent-cc"\nversion = "0.1.30"\n'
        f'[tool.uv]\noverride-dependencies = ["mindie-harness @ git+https://github.com/mindie-agent/harness@{rev}"]\n',
        encoding="utf-8",
    )
    patched_adapter = upg._with_birth_overrides(base_toml, adapter_dir, "mindie-agent-cc")
    overrides_adapter = tomllib.loads(patched_adapter)["tool"]["uv"]["override-dependencies"]
    assert f"mindie-harness @ git+https://github.com/mindie-agent/harness@{rev}" in overrides_adapter
