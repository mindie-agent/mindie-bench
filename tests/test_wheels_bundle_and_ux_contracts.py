"""Wheel bundle determinism, trajectory remote tool classification, and native MCP acceptance tests.

[Role]
  Verify deterministic prebuilt wheel bundle assembly, cross-host remote tool observation
  scoring, and `native_upgrade_acceptance.verify()` contract checks including real
  `mcpServer/tool/call` verification on `mindie_doctor` and `knowledge_query`.
[Invariants]
  * Pure Tier 0 in-memory and `tmp_path` fixtures with zero network I/O.
[Timeouts & Locks]
  * Completes under 5s; acquires no global locks.
[Side-Effect Policy]
  * Writes only inside pytest `tmp_path`.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.runner.mcp_probe import supports_default_capability
from mindie_bench.scorer.target_access import (
    _npu_test_call,
    _validation_tail,
    remote_execution,
    target_access_observations,
)
from mindie_bench.scorer.tool_observations import operation_kind

ROOT = Path(__file__).resolve().parents[1]
def _load(name):
    module = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location(name, ROOT / f".github/scripts/{name}.py"))
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
bundle = _load("build_wheels_bundle")
acceptance = _load("native_upgrade_acceptance")
PACKAGES = bundle.SHARED_PACKAGES
REVISIONS = {name: hashlib.sha256(name.encode()).hexdigest()[:40] for name in PACKAGES}
CHANNEL = {"schema": "mindie-release-channel/1", "current_patch_version": "0.4.13", "components": {
    name: {"repository": "mindie-agent/" + name, "version": "1.2.3", "tag": "v1.2.3",
           "revision": REVISIONS[name]} for name in PACKAGES}}
def _wheel(path, name, version, requires=()):
    with zipfile.ZipFile(path, "w") as archive:
        metadata = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                    + "".join(f"Requires-Dist: {row}\n" for row in requires))
        archive.writestr(zipfile.ZipInfo(
            f"{name.replace('-', '_')}-{version}.dist-info/METADATA", (1980, 1, 1, 0, 0, 0)), metadata)
    return path
def _component(package, dest, *, name=None, version=None, reme=None, bad_sum=False):
    wheel = _wheel(dest / f"{package.replace('-', '_')}-1.2.3-py3-none-any.whl",
                   name or package, version or "1.2.3",
                   () if reme is None else
                   (f"reme-ai[as] @ git+https://github.com/agentscope-ai/ReMe.git@{reme}",))
    digest = "0" * 64 if bad_sum else hashlib.sha256(wheel.read_bytes()).hexdigest()
    (dest / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n")
    return wheel
def _download(repository, version, dest):
    package = repository.rsplit("/", 1)[-1]
    _component(package, dest, reme=bundle.REME_REVISION if package == "mindie-knowledge" else None)
def _tag(repository, version):
    return REVISIONS[repository.rsplit("/", 1)[-1]]
def _reme(repository, revision, dest):
    return _wheel(dest / "reme_ai-0.1.0-py3-none-any.whl", "reme-ai", "0.1.0")
def _build(tmp_path, name="bundle.tar.gz", mutate=None, **kwargs):
    document = json.loads(json.dumps(CHANNEL))
    if mutate:
        mutate(document)
    channel = tmp_path / "channel.json"
    channel.write_text(json.dumps(document), encoding="utf-8")
    kwargs = {"download_fn": _download, "tag_fn": _tag, "reme_builder": _reme, **kwargs}
    return bundle.build_bundle(channel, tmp_path / name, **kwargs)

@pytest.mark.tier0
def test_bundle_is_deterministic_and_matches_install_contract(tmp_path, monkeypatch):
    first, second = _build(tmp_path, "one.tar.gz"), _build(tmp_path, "two.tar.gz")
    monkeypatch.setattr(bundle, "download_component_release", _download)
    monkeypatch.setattr(bundle, "verify_tag_revision", _tag)
    monkeypatch.setattr(bundle, "build_reme_wheel", _reme)
    bundle.main(["--channel", str(tmp_path / "channel.json"), "--output", str(tmp_path / "cli.tar.gz")])
    assert first.read_bytes() == second.read_bytes() == (tmp_path / "cli.tar.gz").read_bytes()
    with tarfile.open(first) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        files = {member.name: archive.extractfile(member).read() for member in members}
    assert names == sorted(names)
    assert all(member.isfile() and member.mode == 0o644 and member.mtime == 0 for member in members)
    assert all(member.name == Path(member.name).name and not member.name.startswith("..")
               for member in members)
    wheels = [name for name in names if name.endswith(".whl")]
    assert len(wheels) == 10 and len(set(wheels)) == 10
    expected = sorted([f"{name}==1.2.3" for name in PACKAGES] + ["reme-ai[as]==0.1.0"])
    assert files["requirements.txt"].decode() == "\n".join(expected) + "\n"
    manifest = json.loads(files["bundle-manifest.json"])
    assert (manifest["schema_version"], manifest["product_version"]) == (
        "mindie-wheels-bundle/1", "0.4.13")
    assert set(manifest["components"]) == set(PACKAGES)
    assert manifest["extra_wheels"]["reme-ai"]["version"] == "0.1.0"
    assert [row["filename"] for row in manifest["wheels"]] == sorted(wheels)
    sums = dict(reversed(line.split("  ", 1)) for line in files["SHA256SUMS"].decode().splitlines())
    assert set(sums) == set(names) - {"SHA256SUMS"}
    assert all(hashlib.sha256(files[name]).hexdigest() == digest for name, digest in sums.items())

@pytest.mark.tier0
def test_bundle_rejects_tampered_or_invalid_inputs(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _build(tmp_path, "sha.tar.gz",
               download_fn=lambda repo, ver, dest: _component(
                   repo.rsplit("/", 1)[-1], dest, bad_sum=True))
    with pytest.raises(ValueError, match="tag revision"):
        _build(tmp_path, "tag.tar.gz", tag_fn=lambda repo, ver: "f" * 40)
    with pytest.raises(ValueError, match="missing shared components"):
        _build(tmp_path, "missing.tar.gz", mutate=lambda doc: doc["components"].pop("mindie-sim"))
    for name, version in (("wrong-name", None), (None, "9.9.9")):
        with pytest.raises(ValueError, match="metadata mismatch"):
            _build(tmp_path, f"{name}-{version}.tar.gz",
                   download_fn=lambda repo, ver, dest, name=name, version=version: _component(
                       repo.rsplit("/", 1)[-1], dest, name=name, version=version))
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", 1000)
    with pytest.raises(ValueError, match="size is outside"):
        _build(tmp_path, "large.tar.gz")

@pytest.mark.tier0
def test_reme_builder_strips_tokens_and_sets_source_date_epoch(tmp_path, monkeypatch):
    dest, calls = tmp_path / "reme", []
    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if "rev-parse" in argv:
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")
        if argv[0] == "uv":
            _wheel(Path(argv[argv.index("--out-dir") + 1]) / "reme_ai-0.1.0-py3-none-any.whl",
                   "reme-ai", "0.1.0")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(bundle.subprocess, "run", fake_run)
    wheel = bundle.build_reme_wheel(bundle.REME_REPO_URL, "a" * 40, dest)
    assert wheel.name == "reme_ai-0.1.0-py3-none-any.whl"
    assert all(kwargs["env"]["SOURCE_DATE_EPOCH"] == "315532800" for _, kwargs in calls)
    assert all("GH_TOKEN" not in kwargs["env"] and "GITHUB_TOKEN" not in kwargs["env"]
               for _, kwargs in calls)

def _runtime(monkeypatch, read_inventory):
    package = types.ModuleType("mindie_knowledge")
    runtime = types.ModuleType("mindie_knowledge.component_runtime")
    runtime.read_inventory = read_inventory
    package.component_runtime = runtime
    monkeypatch.setitem(sys.modules, "mindie_knowledge", package)
    monkeypatch.setitem(sys.modules, "mindie_knowledge.component_runtime", runtime)
@pytest.mark.tier0
def test_read_inventory_wheel_provenance_and_legacy_fallbacks(monkeypatch):
    _runtime(monkeypatch, lambda python, **kwargs: {"ok": kwargs})
    assert acceptance._read_inventory("python", allow_wheel_provenance=True) == {
        "ok": {"allow_wheel_provenance": True}}
    _runtime(monkeypatch, lambda python, **kwargs: (_ for _ in ()).throw(TypeError()))
    probed = []
    def probe(argv, **kwargs):
        probed.append(argv)
        return SimpleNamespace(returncode=0, stdout=b'{"mindie-sim": {"version": "1"}}')
    monkeypatch.setattr(acceptance.subprocess, "run", probe)
    assert acceptance._read_inventory(Path("python"), allow_wheel_provenance=True) == {
        "mindie-sim": {"version": "1"}}
    assert "--allow-wheel-provenance" in probed[0]
    _runtime(monkeypatch, lambda python, **kwargs: (_ for _ in ()).throw(TypeError())
             if kwargs else {"fallback": True})
    monkeypatch.setattr(acceptance.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout=b""))
    assert acceptance._read_inventory("python", allow_wheel_provenance=True) == {"fallback": True}

def _call(identity, name, *, turn=1, **arguments):
    return ToolCall(id=identity, name=name, component="remote-dev", arguments=arguments, turn=turn, started_sec=0, ended_sec=1)
@pytest.mark.tier0
def test_native_remote_file_tools_are_edits_not_reads():
    target = "vllm_ascend/ops/layernorm.py"
    calls = [
        _call("e1", "remote_Edit", file_path=target),
        _call("e2", "remote_Write", path=target),
        _call("e3", "remote_NotebookEdit", notebook_path=target),
        _call("e4", "remote_edit_file", Path=target),
        _call("r1", "remote_Read", filePath=target),
    ]
    observations, _ = target_access_observations(Trajectory(calls=calls), [target])
    assert {row["call_id"]: row["kind"] for row in observations} == {
        "e1": "edit", "e2": "edit", "e3": "edit", "e4": "edit", "r1": "read"}
    assert all(operation_kind(call) == "file" for call in calls)
@pytest.mark.tier0
def test_native_command_aliases_are_execution():
    run = _call("run", "remote_run_command", CommandLine="python -m pytest tests/test_npu.py")
    execute = _call("exec", "remote_exec_command", cmd="python -m pytest tests/test_npu.py")
    assert remote_execution(run) and remote_execution(execute) and _npu_test_call(run)
    assert operation_kind(run) == operation_kind(execute) == "execution"
@pytest.mark.tier0
@pytest.mark.parametrize(("name", "arguments", "expected"), [
    ("remote_TaskOutput", {"task_id": "job"}, "observation"),
    ("remote_write_stdin", {"session_id": "job", "chars": ""}, "observation"),
    ("remote_write_stdin", {"session_id": "job", "chars": "x"}, "control"),
    ("remote_manage_task", {"TaskId": "job", "Action": "status"}, "observation"),
    ("remote_manage_task", {"TaskId": "job", "Action": "list"}, "other"),
    ("remote_manage_task", {"TaskId": "job", "Action": "kill"}, "control"),
    ("remote_manage_task", {"TaskId": "job", "Action": "send_input"}, "control"),
])
def test_native_job_operation_kinds(name, arguments, expected):
    assert operation_kind(_call("job", name, **arguments)) == expected
@pytest.mark.tier0
def test_empty_remote_write_stdin_poll_counts_as_empty_job_poll():
    empty = _call("empty", "remote_write_stdin", session_id="job", chars="")
    nonempty = _call("nonempty", "remote_write_stdin", session_id="job", chars="x")
    assert _validation_tail(Trajectory(calls=[empty]))["total_empty_job_polls"] == 1
    assert _validation_tail(Trajectory(calls=[nonempty]))["total_empty_job_polls"] == 0
def _pending_edit():
    call = _call("submitted", "remote_exec_command",
                 cmd="sed -i 's/a/b/' vllm_ascend/ops/layernorm.py")
    call.result_observation = ToolResultObservation(
        source="codex_mcp", result_schema="remote-dev.result.v1", classification="pending",
        operation_kind="execution", job_id="job-1", target_id="host-1",
        remote_status="running", classification_reason="explicit_remote_job_pending")
    return call
def _terminal(name, arguments, operation):
    call = _call("confirmed", name, turn=2, **arguments)
    call.result_observation = ToolResultObservation(
        source="codex_mcp", result_schema="remote-dev.result.v1",
        classification="business_success", operation_kind=operation, job_id="job-1",
        target_id="host-1", remote_status="completed", exit_code=0,
        classification_reason="explicit_remote_command_exit")
    return call

@pytest.mark.tier0
@pytest.mark.parametrize(("name", "arguments", "operation", "confirms"), [
    ("remote_TaskOutput", {"task_id": "job-1"}, "observation", True),
    ("remote_manage_task", {"TaskId": "job-1", "Action": "status"}, "observation", True),
    ("remote_manage_task", {"TaskId": "job-1", "Action": "list"}, "other", False),
    ("remote_manage_task", {"TaskId": "job-1", "Action": "kill"}, "control", False),
    ("remote_manage_task", {"TaskId": "job-1", "Action": "send_input"}, "control", False),
    ("remote_manage_task", {"TaskId": "other", "Action": "status"}, "observation", False),
])
def test_manage_task_only_matching_status_confirms_command(name, arguments, operation, confirms):
    observations, _ = target_access_observations(
        Trajectory(calls=[_pending_edit(), _terminal(name, arguments, operation)]),
        ["vllm_ascend/ops/layernorm.py"])
    row = observations[0]
    assert (row["call_outcome"] == "success") is confirms
    if confirms:
        assert row["outcome_evidence"]["confirmed_exit_code"] == 0

def _verify(tmp_path, monkeypatch, *, components, managed=("sim", "profiler"),
            remote=("remote_probe", "remote_exec_command"), tool_responses=None):
    root, scripts = tmp_path / "runtime", tmp_path / "runtime/scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (root / ".codex-plugin").mkdir(exist_ok=True)
    (root / ".codex-plugin/plugin.json").write_text('{"version": "0.1.38"}', encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(
        {"runtime_scripts": str(scripts), "python": str(tmp_path / "python")}), encoding="utf-8")
    monkeypatch.setenv("MINDIE_AGENT_CONFIG", str(config_path))
    tool_calls = []
    class Hooks:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def hooks(self):
            return [{"eventName": "stop", "enabled": False},
                    *[{"eventName": str(index)} for index in range(4)]]
        def call(self, method, params):
            if method == "thread/start":
                return {"thread": {"id": "thread"}}
            if method == "mcpServerStatus/list":
                return {"data": [{"name": "mindie-remote-dev", "tools": list(remote)},
                                 {"name": "mindie-tools", "tools": list(components)}]}
            if method == "mcpServer/tool/call":
                tool_calls.append((params["server"], params["tool"], params["arguments"]))
                if tool_responses and params["tool"] in tool_responses:
                    return tool_responses[params["tool"]]
                return {"isError": False, "content": [{"type": "text", "text": f"{params['tool']} ok"}]}
            raise AssertionError(f"unexpected RPC method: {method}")
    hooks = types.ModuleType("native_hooks")
    hooks.NativeHooks = Hooks
    bindings = types.ModuleType("mindie_knowledge.component_bindings")
    bindings.managed_commands = lambda current: {
        name: (name, current["python"], "module") for name in managed}
    package = types.ModuleType("mindie_knowledge")
    package.component_bindings = bindings
    monkeypatch.setitem(sys.modules, "native_hooks", hooks)
    monkeypatch.setitem(sys.modules, "mindie_knowledge", package)
    monkeypatch.setitem(sys.modules, "mindie_knowledge.component_bindings", bindings)
    monkeypatch.setattr(acceptance, "_read_inventory", lambda python, **kwargs: {"ok": True})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        acceptance.verify(SimpleNamespace(root=tmp_path, target_host=tmp_path / "host",
                                          version="0.1.38"))
    return json.loads(buf.getvalue()), tool_calls

BASE = ["diagnostics_status", "mindie_doctor", "knowledge_query", "knowledge_explain",
        "knowledge_feedback", "knowledge_distill", "knowledge_sync"]
@pytest.mark.tier0
def test_verify_accepts_mindie_doctor_wheel_inventory_and_rebound_components(tmp_path, monkeypatch):
    report, tool_calls = _verify(tmp_path, monkeypatch, components=BASE)
    assert report["mcp_tool_calls_verified"] is True
    assert tool_calls == [
        ("mindie-tools", "mindie_doctor", {"probe_remote": False}),
        ("mindie-tools", "knowledge_query", {"query": "aclgraph", "domain": "vllm-ascend"}),
    ]


@pytest.mark.tier0
@pytest.mark.parametrize(("tool_responses", "expected_tool"), [
    ({"mindie_doctor": {"isError": True, "content": [{"type": "text", "text": "err"}]}}, "mindie_doctor"),
    ({"knowledge_query": {"isError": True, "content": [{"type": "text", "text": "knowledge_unavailable"}]}}, "knowledge_query"),
    ({"knowledge_query": {"isError": False, "content": []}}, "knowledge_query"),
])
def test_verify_rejects_failed_or_empty_mcp_tool_calls(tmp_path, monkeypatch, tool_responses, expected_tool):
    with pytest.raises(ValueError, match=f"native MCP tool call failed: {expected_tool}"):
        _verify(tmp_path, monkeypatch, components=BASE, tool_responses=tool_responses)
@pytest.mark.tier0
@pytest.mark.parametrize(("components", "managed", "remote", "message"), [
    ([name for name in BASE if name != "mindie_doctor"], ("sim", "profiler"), None, "public contract"),
    (BASE, ("sim", "profiler"), ("remote_probe", "remote_exec_command", "remote_BashOutput"),
     "public contract"),
    (BASE, ("sim", "profiler"), ("remote_probe", "remote_exec_command", "remote_KillShell"),
     "public contract"),
    (BASE, ("sim",), None, "bindings disappeared"),
])
def test_verify_rejects_contract_drift(tmp_path, monkeypatch, components, managed, remote, message):
    with pytest.raises(ValueError, match=message):
        _verify(tmp_path, monkeypatch, components=components, managed=managed,
                **({} if remote is None else {"remote": remote}))

@pytest.mark.tier0
@pytest.mark.parametrize(("host", "tools", "expected"), [
    ("codex", {"remote_exec_command"}, True), ("claude/kimi/wb", {"remote_Bash"}, True),
    ("agy", {"remote_run_command"}, True), ("remote_probe", {"remote_probe"}, False),
    ("retired", {"remote_BashOutput"}, False),
])
def test_default_remote_dev_capability_is_one_to_one(host, tools, expected):
    assert supports_default_capability("remote-dev", tools) is expected
