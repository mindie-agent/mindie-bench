"""Claude G3 identity and adapter contracts; fixture events are not native acceptance."""

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.integrations import claude_stop, first_party
from mindie_bench.models import HarnessConfig
from mindie_bench.runner.drivers.claude_code import ClaudeCodeDriver
from mindie_bench.runner.sandbox import native_command


def stop_event(tmp_path):
    return dict(
        hook_event_name="Stop",
        stop_hook_active=False,
        session_id="151d974a-c35e-42b1-8afe-396aa5e28685",
        prompt_id="native-prompt-id",
        transcript_path=str(tmp_path / "native.jsonl"),
        cwd=str(tmp_path / "workspace"),
    )


def test_claude_stop_preserves_native_prompt_identity_and_refuses_overwrite(tmp_path):
    event = stop_event(tmp_path)
    destination = tmp_path / "native-stop.json"
    command = [sys.executable, "-I", str(Path(claude_stop.__file__)), str(destination)]
    first = subprocess.run(command, input=json.dumps(event), text=True, capture_output=True)
    assert first.returncode == 0 and json.loads(first.stdout) == {}
    assert claude_stop.read_stop(destination) == event
    assert "turn_id" not in json.loads(destination.read_text())
    before = destination.read_bytes()
    repeated = subprocess.run(command, input=json.dumps(event), text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
    assert destination.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        {"hook_event_name": "SessionEnd"},
        {"stop_hook_active": True},
        {"agent_id": "child"},
        {"prompt_id": ""},
        {"session_id": ""},
        {"cwd": "relative"},
        {"transcript_path": "relative.jsonl"},
    ],
)
def test_claude_stop_cannot_invent_foreground_identity_or_accept_another_hook(tmp_path, change):
    with pytest.raises(ValueError):
        claude_stop.validate_stop({**stop_event(tmp_path), **change})


def test_claude_stop_read_is_bounded(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (claude_stop.MAX_EVENT_BYTES + 1))
    with pytest.raises(ValueError, match="evidence bound"):
        claude_stop.read_stop(path)


@pytest.mark.parametrize("group", ["G0", "G1", "G2", "G3"])
def test_only_g3_enables_the_owned_native_stop_and_session_artifact(tmp_path, group):
    root = tmp_path / "profile"
    root.mkdir()
    claude = root / "claude"
    claude.mkdir()
    (root / "settings.json").write_text(json.dumps({"disableAllHooks": True}))
    (root / "mcp.json").write_text('{"mcpServers": {}}')
    profile = SimpleNamespace(
        root=root,
        env={"CLAUDE_CONFIG_DIR": str(claude)},
        fingerprint=dict(group=group, server_components={}),
    )
    config = HarnessConfig(
        name="claude-code",
        executable="/explicit/native/claude",
        model="explicit-model",
        expected_version="native-version",
        provider_url="http://127.0.0.1:1234",
        api_key_env="OWNED_FIXTURE_KEY",
    )
    command = ClaudeCodeDriver().command(profile, config)
    assert profile.env["CLAUDE_CODE_TMPDIR"] == str(root / "tmp")
    settings = json.loads((root / "settings.json").read_text())
    if group == "G3":
        assert "--no-session-persistence" not in command
        assert settings["disableAllHooks"] is False
        assert list(settings["hooks"]) == ["Stop"]
        hook = settings["hooks"]["Stop"][0]["hooks"][0]
        assert "mindie_bench.integrations.claude_stop" in hook["command"]
        assert str(root / "native-stop.json") in hook["command"]
    else:
        assert "--no-session-persistence" in command
        assert settings["disableAllHooks"] is True and "hooks" not in settings
    assert json.loads((claude / "settings.json").read_text()) == {"model": "explicit-model"}


def test_claude_native_stream_session_is_exact_and_cannot_change():
    driver = ClaudeCodeDriver()
    driver.consume({"type": "system", "subtype": "init", "session_id": "native-session"}, 0, 0)
    assert driver.trajectory().native_session_id == "native-session"
    with pytest.raises(ValueError, match="changed"):
        driver.consume({"type": "result", "session_id": "another-session"}, 1, 1)
    with pytest.raises(ValueError, match="no native session"):
        ClaudeCodeDriver().consume({"type": "system", "subtype": "init"}, 0, 0)


def test_cc_capture_selects_real_adapter_contract_and_native_prompt_not_codex(tmp_path, monkeypatch):
    root = tmp_path / "profile"
    root.mkdir()
    profile = root / "claude"
    profile.mkdir()
    scripts = tmp_path / "cc-scripts"
    scripts.mkdir()
    (scripts / "organizer.py").write_text("# native component worker fixture\n")
    event = stop_event(tmp_path)
    stop = root / "native-stop.json"
    stop.write_text(json.dumps(event))
    calls = []
    adapter = SimpleNamespace(
        capture_source=lambda path, identity: (
            calls.append((path, identity)) or dict(project_root=event["cwd"], created_at=1)
        )
    )
    monkeypatch.setitem(sys.modules, "transcript", adapter)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
    monkeypatch.setenv("ANTHROPIC_MODEL", "foreground-model")
    monkeypatch.setenv("MINDIE_CC_CONFIG", "fixture-before")
    monkeypatch.setenv("MINDIE_CC_BIN", "fixture-before")
    monkeypatch.syspath_prepend(str(scripts))
    args = SimpleNamespace(
        claude_scripts=str(scripts), claude_bin="/explicit/native/claude", summary_model="summary-model"
    )
    native = dict(
        harness="claude-code",
        session_id=event["session_id"],
        transcript_path=event["transcript_path"],
        cwd=event["cwd"],
        stop_event_path=str(stop),
    )
    actual, source, selected, directory, worker = first_party.native_capture_inputs(args, native, root)
    assert calls == [(event["transcript_path"], event["session_id"])]
    assert actual["turn_id"] == event["prompt_id"]
    assert selected is adapter and worker == "organizer.py" and directory == scripts
    assert source["project_root"] == event["cwd"]
    assert json.loads((root / "benchmark-cc-adapter.json").read_text()) == {
        "claude_config_dir": str(profile)
    }
    assert first_party.os.environ["ANTHROPIC_MODEL"] == "summary-model"
    with pytest.raises(ValueError, match="foreground task differ"):
        first_party.native_capture_inputs(args, {**native, "session_id": "wrong"}, root)
    with pytest.raises(ValueError, match="requires an actual Codex or Claude"):
        first_party.native_capture_inputs(args, {**native, "harness": "unsupported"}, root)


def test_native_summary_loader_reads_only_the_selected_package_and_adapter(tmp_path):
    capsule, scripts = tmp_path / "capsule", tmp_path / "adapter"
    capsule.mkdir()
    package = capsule / "mindie_knowledge"
    package.mkdir()
    scripts.mkdir()
    (package / "__init__.py").write_text('IDENTITY = "selected-component"\n')
    (package / "public.py").write_text('VALUE = "component-submodule"\n')
    private = capsule / "unrelated-private-source.py"
    private.write_text('GOLD = "must not be readable"\n')
    # Seatbelt denies reads; bwrap omits the unbound sibling from its namespace.
    denied = "PermissionError" if sys.platform == "darwin" else "(PermissionError, FileNotFoundError)"
    (scripts / "organizer.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from mindie_knowledge import IDENTITY\n"
        "from mindie_knowledge.public import VALUE\n"
        f"try:\n    Path({str(private)!r}).read_text()\n"
        f"except {denied}:\n    pass\n"
        "else:\n    raise RuntimeError('sibling source was exposed')\n"
        "print(json.dumps([IDENTITY, VALUE]))\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = native_command(
        first_party.native_summary_command(package, scripts, "organizer.py"),
        read=[Path(sys.prefix), Path(sys.base_prefix), package, scripts],
        write=[workspace],
        cwd=workspace,
        policy_path=tmp_path / "summary.sb",
    )
    result = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["selected-component", "component-submodule"]


@pytest.fixture
def actual_claude_session(tmp_path, monkeypatch, request):
    """Real native hooks/transcript; local protocol fixture makes no paid model call."""
    executable = os.environ.get("MINDIE_BENCH_TEST_CLAUDE_BIN")
    scripts = os.environ.get("MINDIE_BENCH_TEST_CLAUDE_SCRIPTS")
    if not executable or not scripts:
        pytest.skip("requires an explicit native Claude binary and component scripts")
    requests = []
    reject_advice = getattr(request, "param", None) == "rejected"
    native_tool = getattr(request, "param", None) == "native_tool"

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request))
            if self.path.endswith("/count_tokens"):
                body = json.dumps({"input_tokens": 1}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            text = """## Root cause hypothesis
On this isolated CPU workspace, the repro path appears to resolve outside its workspace.
## Verification evidence
This protocol fixture has no actual business test; rerun the repro to verify any repair.
## Applicability
This example applies only to this isolated CPU repro and proves no NPU behavior.
"""
            if reject_advice:
                text += "\nNever use ACLGraph on Ascend.\n"
            message = dict(
                id="msg_protocol_fixture",
                type="message",
                role="assistant",
                model=request["model"],
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=dict(
                    input_tokens=10,
                    output_tokens=0,
                    cache_creation_input_tokens=2,
                    cache_read_input_tokens=3,
                ),
            )
            structured = next(
                (tool for tool in request.get("tools", []) if tool["name"] == "StructuredOutput"),
                None,
            )
            has_tool_result = any(
                block.get("type") == "tool_result"
                for message in request.get("messages", [])
                if isinstance(message.get("content"), list)
                for block in message["content"]
                if isinstance(block, dict)
            )
            submit_tool = structured or native_tool and not has_tool_result
            if structured:
                identifiers = structured["input_schema"]["properties"]["blocks"]["items"]["properties"][
                    "block_id"
                ]["enum"]
                metadata = dict(
                    blocks=[
                        dict(
                            block_id=identity,
                            title="Isolated CPU repro observation",
                            summary="Protocol fixture only. The repro needs independent validation.",
                        )
                        for identity in identifiers
                    ],
                    navigation=dict(
                        title="Isolated CPU repro",
                        summary="Protocol fixture with unverified repro path hypothesis.",
                    ),
                )
                start = dict(
                    type="tool_use", id="tool_summary_fixture", name="StructuredOutput", input={}
                )
                delta = dict(type="input_json_delta", partial_json=json.dumps(metadata))
            elif submit_tool:
                start = dict(type="tool_use", id="tool_native_fixture", name="Bash", input={})
                delta = dict(
                    type="input_json_delta",
                    partial_json=json.dumps(
                        dict(
                            command="printf 'native-tool-canary\\n'",
                            description="Run the owned local protocol canary",
                        )
                    ),
                )
            else:
                start = dict(type="text", text="")
                delta = dict(type="text_delta", text=text)
            events = [
                ("message_start", dict(type="message_start", message=message)),
                (
                    "content_block_start",
                    dict(type="content_block_start", index=0, content_block=start),
                ),
                (
                    "content_block_delta",
                    dict(type="content_block_delta", index=0, delta=delta),
                ),
                ("content_block_stop", dict(type="content_block_stop", index=0)),
                (
                    "message_delta",
                    dict(
                        type="message_delta",
                        delta=dict(
                            stop_reason="tool_use" if submit_tool else "end_turn", stop_sequence=None
                        ),
                        usage=dict(output_tokens=20),
                    ),
                ),
                ("message_stop", dict(type="message_stop")),
            ]
            body = "".join(f"event: {kind}\ndata: {json.dumps(value)}\n\n" for kind, value in events)
            encoded = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def close():
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    request.addfinalizer(close)
    root, workspace = tmp_path / "profile", tmp_path / "workspace"
    root.mkdir()
    workspace.mkdir()
    home, claude = root / "home", root / "claude"
    home.mkdir()
    claude.mkdir()
    env = {
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(claude),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "TMPDIR": str(tmp_path),
        "ANTHROPIC_API_KEY": "loopback-protocol-fixture",
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "MAX_THINKING_TOKENS": "0",
    }
    (root / "settings.json").write_text('{"disableAllHooks":true}')
    (root / "mcp.json").write_text('{"mcpServers":{}}')
    profile = SimpleNamespace(root=root, env=env, fingerprint=dict(group="G3", server_components={}))
    config = SimpleNamespace(executable=executable, model="claude-sonnet-4-6", reasoning_effort=None)
    driver = ClaudeCodeDriver()
    command = native_command(
        [
            *driver.command(profile, config),
            "--tools",
            "Bash" if native_tool else "",
            "--include-hook-events",
        ],
        read=[
            Path(executable).resolve(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(first_party.__file__).resolve().parents[2],
            # Native Bun/ICU maps the macOS system timezone data at startup.
            *([Path("/private/var/db/timezone")] if sys.platform == "darwin" else []),
        ],
        write=[root, workspace],
        cwd=workspace,
        policy_path=tmp_path / "native.sb",
        gateway_port=server.server_port,
    )
    result = subprocess.run(
        command,
        input="Describe this isolated repro observation briefly.",
        env=env,
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    for index, event in enumerate(events):
        driver.consume(event, index, 0)
    trajectory = driver.trajectory()
    assert trajectory.native_completed
    actual = claude_stop.read_stop(root / "native-stop.json")
    assert actual["session_id"] == trajectory.native_session_id
    assert actual["prompt_id"] and "turn_id" not in actual
    assert requests and all(path.startswith("/v1/messages") for path, _ in requests)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("MINDIE_CC_CONFIG", "fixture-before")
    monkeypatch.setenv("MINDIE_CC_BIN", "fixture-before")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fixture-before")
    monkeypatch.syspath_prepend(str(scripts))
    for name in ("transcript", "paths"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    native = dict(
        harness="claude-code",
        session_id=trajectory.native_session_id,
        cwd=str(workspace),
        transcript_path=actual["transcript_path"],
        stop_event_path=str(root / "native-stop.json"),
    )
    args = SimpleNamespace(
        claude_scripts=scripts, claude_bin=executable, summary_model="claude-haiku-4-5-20251001"
    )
    event, source, adapter, _, worker = first_party.native_capture_inputs(args, native, root)
    assert event["turn_id"] == actual["prompt_id"]
    assert source["session_id"] == actual["session_id"]
    assert source["project_root"] == str(workspace)
    assert Path(adapter.__file__).resolve() == (Path(scripts) / "transcript.py").resolve()
    assert worker == "organizer.py"
    return dict(
        root=root,
        env=env,
        native=native,
        args=args,
        requests=requests,
        trajectory=trajectory,
    )


def test_actual_claude_stop_and_component_transcript_with_loopback_provider(actual_claude_session):
    assert actual_claude_session["native"]["harness"] == "claude-code"


@pytest.mark.parametrize("actual_claude_session", ["native_tool"], indirect=True)
def test_actual_claude_native_tool_result_reaches_its_next_provider_turn(actual_claude_session):
    case = actual_claude_session
    calls = case["trajectory"].calls
    assert len(calls) == 1 and calls[0].name == "Bash" and calls[0].is_error is False
    results = [
        block
        for _, value in case["requests"]
        for message in value.get("messages", [])
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(results) == 1
    assert "native-tool-canary" in json.dumps(results[0]["content"])


def run_native_capture(tmp_path, case, python, gitleaks):
    private = tmp_path / "knowledge-private"
    corpus, remote = private / "corpus", private / "local_knowledge.git"
    (corpus / "tasks").mkdir(parents=True)
    env = dict(
        case["env"],
        MINDIE_BENCH_GIT=shutil.which("git"),
        GIT_EXEC_PATH=subprocess.check_output(["git", "--exec-path"], text=True).strip(),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        MINDIE_BENCH_KNOWLEDGE_PRIVATE_ROOT=str(private),
    )
    subprocess.run(["git", "init", "--template=", "--bare", "-q", str(remote)], env=env, check=True)
    source = Path(first_party.__file__).resolve().parents[2]
    loader = (
        f"import sys,runpy;sys.path.insert(0,{str(source)!r});"
        "runpy.run_module('mindie_bench.integrations.first_party',run_name='__main__')"
    )
    return subprocess.run(
        [
            python,
            "-B",
            "-I",
            "-c",
            loader,
            "capture",
            "--claude-scripts",
            case["args"].claude_scripts,
            "--claude-bin",
            case["args"].claude_bin,
            "--summary-model",
            case["args"].summary_model,
            "--gitleaks",
            gitleaks,
            "--query",
            "CPU repro",
        ],
        cwd=case["root"],
        env=env,
        input=json.dumps(dict(native=case["native"], destination=str(corpus), local_remote=str(remote))),
        text=True,
        capture_output=True,
        timeout=45,
    )


def capture_dependencies():
    python = os.environ.get("MINDIE_BENCH_TEST_MCP_PYTHON")
    gitleaks = shutil.which("gitleaks")
    if not python or not gitleaks:
        pytest.skip("requires the existing shared component interpreter and Gitleaks")
    return python, gitleaks


def test_actual_claude_organizer_capture_git_and_independent_consumer(tmp_path, request):
    """Native CC + actual shared components; scripted metadata is not a real model result."""
    python, gitleaks = capture_dependencies()
    case = request.getfixturevalue("actual_claude_session")
    result = run_native_capture(tmp_path, case, python, gitleaks)
    corpus = tmp_path / "knowledge-private" / "corpus"
    assert result.returncode == 0, (result.stdout, result.stderr)
    captured = json.loads(result.stdout)
    assert captured["status"] == "completed"
    assert captured["native_harness"] == "claude-code"
    assert captured["native_session_id"] == case["native"]["session_id"]
    assert captured["corpus_lint"]["status"] == "passed"
    assert captured["indexer_usage"]["model_calls"] == 1
    assert captured["indexer_usage"]["unknown_calls"] == 0
    assert captured["indexer_usage"]["cached_input_tokens"] == 3
    assert captured["tokens"]["total"] == sum(
        captured["indexer_usage"][key]
        for key in ["input_tokens", "cached_input_tokens", "output_tokens"]
    )
    assert captured["consumer"]["revision"] == captured["publication"]["revision"]
    assert captured["consumer"]["reads"]
    assert len(list((corpus / "tasks").iterdir())) == 1
    assert any(
        tool["name"] == "StructuredOutput"
        for _, value in case["requests"]
        for tool in value.get("tools", [])
    )


@pytest.mark.parametrize("actual_claude_session", ["rejected"], indirect=True)
def test_actual_claude_lint_rejection_preserves_body_and_blocks_publication(
    tmp_path, actual_claude_session
):
    python, gitleaks = capture_dependencies()
    case = actual_claude_session
    result = run_native_capture(tmp_path, case, python, gitleaks)
    assert result.returncode == 1
    failure = json.loads(result.stdout)
    assert failure["status"] == "failed"
    report = failure["corpus_lint"]
    assert report["status"] == "rejected"
    assert "unconditional_global_ban" in {item["code"] for item in report["findings"]}
    publication = case["root"] / "publication"
    assert any("Never use ACLGraph on Ascend." in p.read_text() for p in publication.rglob("*.md"))
    assert not (publication / ".git").exists()
    assert not (case["root"] / "independent-consumer").exists()
    assert not list((tmp_path / "knowledge-private" / "corpus" / "tasks").iterdir())
    assert not (case["root"] / "full-loop-receipt.json").exists()
    assert not (case["root"] / "corpus-lint.json").exists()
