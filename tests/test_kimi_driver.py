"""Kimi driver tests: real 2.2.0 stream fixtures, owned config, and the prompt adapter."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.models import ExperimentConfig, HarnessConfig
from mindie_bench.runner.drivers.kimi import KimiDriver
from mindie_bench.runner.kimi_profile import build_kimi_profile
from mindie_bench.runner.process import stream_process
from mindie_bench.runner.profile_builder import Profile
from mindie_bench.runner.sandbox import native_command

FIXTURES = Path(__file__).parent / "fixtures" / "kimi"
KIMI_BIN = Path(os.environ.get("MINDIE_BENCH_TEST_KIMI_BIN", Path.home() / ".kimi-code/bin/kimi"))


def replay(name, components=None):
    driver = KimiDriver()
    driver.server_components = components or {}
    events = [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]
    for index, event in enumerate(events):
        driver.consume(event, float(index), 0)
    return driver.trajectory()


def test_basic_tool_stream_from_real_capture():
    trajectory = replay("basic_tool.jsonl")
    assert trajectory.native_completed
    assert trajectory.native_session_id == "session_f06a8813-2cd7-4e75-8224-ea938123f369"
    assert trajectory.native_error is None
    assert trajectory.total_turns == 2
    assert [message["text"] for message in trajectory.public_messages] == ["DONE"]
    assert len(trajectory.calls) == 1
    call = trajectory.calls[0]
    assert call.name == "Bash"
    assert call.component is None
    assert call.arguments == {"command": "echo hello-kimi-probe"}
    assert call.turn == 1
    assert call.started_sec == 1.0
    assert call.ended_sec == 2.0


def test_tool_failure_has_no_structured_error_signal():
    trajectory = replay("tool_error.jsonl")
    assert trajectory.native_completed
    call = trajectory.calls[0]
    assert call.arguments == {"command": "exit 7"}
    # The 2.2.0 stream reports exit status only inside content prose; the
    # honest observation is absent, not a parsed guess.
    assert call.is_error is None
    assert call.result_observation is None
    assert trajectory.public_messages[0]["text"].startswith("The command `exit 7`")


def test_mcp_tool_stream_preserves_component_attribution():
    trajectory = replay("mcp_tool.jsonl", {"probe": "knowledge"})
    assert len(trajectory.calls) == 1
    call = trajectory.calls[0]
    assert call.name == "echo"
    assert call.component == "knowledge"
    assert call.arguments == {"text": "hi"}
    assert call.ended_sec is not None
    remote = replay("mcp_tool.jsonl", {"probe": "remote-dev"})
    assert remote.calls[0].name == "remote_echo"
    assert remote.calls[0].component == "remote-dev"


def test_retry_failure_stream_never_completes():
    trajectory = replay("retry_failure.jsonl")
    assert not trajectory.native_completed
    assert trajectory.native_session_id is None
    assert trajectory.native_error is None
    assert trajectory.total_turns == 0
    assert trajectory.calls == []


def test_tokens_stay_absent_without_native_usage_events():
    trajectory = replay("basic_tool.jsonl")
    assert not trajectory.tokens.complete
    assert trajectory.tokens.output == 0


def test_consume_rejects_missing_tool_call_id():
    driver = KimiDriver()
    with pytest.raises(ValueError, match="no ID"):
        driver.consume(
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "Bash"}}]},
            0.0,
            0,
        )


def test_consume_rejects_non_object_arguments():
    driver = KimiDriver()
    with pytest.raises(ValueError, match="not an object"):
        driver.consume(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "id": "c1", "function": {"name": "Bash", "arguments": "[1]"}}
                ],
            },
            0.0,
            0,
        )


def test_consume_rejects_session_identity_change():
    driver = KimiDriver()
    hint = {"role": "meta", "type": "session.resume_hint", "session_id": "session_a"}
    driver.consume(hint, 0.0, 0)
    with pytest.raises(ValueError, match="changed its native session identity"):
        driver.consume(dict(hint, session_id="session_b"), 1.0, 0)


def test_consume_ignores_orphan_tool_result_and_duplicate_call():
    driver = KimiDriver()
    driver.consume({"role": "tool", "tool_call_id": "orphan", "content": "x"}, 0.0, 0)
    call = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "Read", "arguments": '{"path":"a.py"}'},
            }
        ],
    }
    driver.consume(call, 1.0, 0)
    driver.consume(dict(call), 2.0, 0)
    trajectory = driver.trajectory()
    assert len(trajectory.calls) == 1
    assert trajectory.calls[0].name == "Read"


def profile_stub(tmp_path, components=None):
    root = tmp_path / "profile"
    root.mkdir()
    home = root / "kimi"
    home.mkdir()
    env = {
        "PATH": "/usr/bin:/bin",
        "KIMI_CODE_HOME": str(home),
        "MINDIE_BENCH_GATEWAY": "http://127.0.0.1:38467",
    }
    (root / "model-catalog.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "k3",
                        "max_context_size": 1048576,
                        "capabilities": ["thinking", "always_thinking", "tool_use"],
                        "support_efforts": ["low", "high", "max"],
                        "default_effort": "high",
                    }
                ]
            }
        )
    )
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "command": "/usr/bin/python3",
                        "args": ["-m", "remote_dev.mcp.server"],
                        "env": {"MINDIE_BENCH_GATEWAY": "http://127.0.0.1:38467"},
                    }
                }
            }
        )
    )
    return Profile(
        root=root,
        env=env,
        corpus=None,
        fingerprint={"server_components": components or {"remote": "remote-dev"}},
        runtime_paths=[],
    )


def harness_config(**overrides):
    values = dict(
        name="kimi",
        executable="/opt/bench/kimi-2.2.0/bin/kimi",
        model="k3",
        expected_version="2.2.0",
        reasoning_effort="high",
        provider_url="https://provider.invalid/v1",
        api_key_env="MINDIE_BENCH_TEST_KIMI_KEY",
        model_catalog_path="/declared/model-catalog.json",
    )
    values.update(overrides)
    return HarnessConfig(**values)


def test_command_writes_owned_config_and_mcp_copy(tmp_path):
    profile = profile_stub(tmp_path)
    driver = KimiDriver()
    argv = driver.command(profile, harness_config())
    config = tomllib.loads((Path(profile.env["KIMI_CODE_HOME"]) / "config.toml").read_text())
    assert config["default_model"] == "bench"
    assert config["default_permission_mode"] == "auto"
    assert config["telemetry"] is False
    assert config["builtin_product_skills"] is False
    provider = config["providers"]["benchmark"]
    assert provider["type"] == "kimi"
    assert provider["base_url"] == "http://127.0.0.1:38467/v1"
    assert provider["api_key_env"] == "MINDIE_BENCH_DUMMY_KEY"
    assert "api_key" not in provider
    model = config["models"]["bench"]
    assert model["provider"] == "benchmark"
    assert model["model"] == "k3"
    assert model["max_context_size"] == 1048576
    assert model["support_efforts"] == ["low", "high", "max"]
    assert config["thinking"] == {"enabled": True, "effort": "high"}
    assert config["loop_control"]["max_attempts_per_step"] == 1
    assert config["background"]["print_background_mode"] == "exit"
    mcp = json.loads((Path(profile.env["KIMI_CODE_HOME"]) / "mcp.json").read_text())
    server = mcp["mcpServers"]["remote"]
    assert server["startupTimeoutMs"] == 30000
    assert server["toolTimeoutMs"] == 600000
    assert server["env"]["MINDIE_BENCH_GATEWAY"] == "http://127.0.0.1:38467"
    assert argv[:4] == [sys.executable, "-I", "-m", "mindie_bench.runner.drivers.kimi"]
    assert argv[4] == "/opt/bench/kimi-2.2.0/bin/kimi"
    assert argv[5:] == [
        "-m",
        "bench",
        "--output-format",
        "stream-json",
        "--skills-dir",
        str(Path(profile.env["KIMI_CODE_HOME"]) / "no-skills"),
    ]


def test_command_uses_declared_skills_dir(tmp_path):
    profile = profile_stub(tmp_path)
    skills = profile.root / "skills"
    skills.mkdir()
    argv = KimiDriver().command(profile, harness_config())
    assert argv[-1] == str(skills)


def test_command_requires_catalog_and_valid_entry(tmp_path):
    profile = profile_stub(tmp_path)
    with pytest.raises(ValueError, match="explicit model catalog"):
        KimiDriver().command(profile, harness_config(model_catalog_path=None))
    (profile.root / "model-catalog.json").write_text(json.dumps({"models": []}))
    with pytest.raises(ValueError, match="exactly one entry"):
        KimiDriver().command(profile, harness_config())
    (profile.root / "model-catalog.json").write_text(
        json.dumps({"models": [{"slug": "k3", "max_context_size": "big"}]})
    )
    with pytest.raises(ValueError, match="max_context_size"):
        KimiDriver().command(profile, harness_config())
    (profile.root / "model-catalog.json").write_text(
        json.dumps({"models": [{"slug": "k3", "max_context_size": 8, "support_efforts": ["low"]}]})
    )
    with pytest.raises(ValueError, match="support_efforts"):
        KimiDriver().command(profile, harness_config(reasoning_effort="xhigh"))


def test_build_kimi_profile_wires_owned_environment(tmp_path):
    root = tmp_path / "profile"
    root.mkdir()
    executable = tmp_path / "kimi-dist" / "kimi"
    executable.parent.mkdir()
    executable.write_text("")
    env = {"PATH": "/usr/bin:/bin"}
    config = ExperimentConfig(
        schema_version="1.0",
        harness=HarnessConfig(
            name="kimi",
            executable=str(executable),
            model="k3",
            expected_version="2.2.0",
            provider_url="https://provider.invalid/v1",
            api_key_env="MINDIE_BENCH_TEST_KIMI_KEY",
        ),
    )
    fingerprint = build_kimi_profile(root, env, config)
    home = root / "kimi"
    assert env["KIMI_CODE_HOME"] == str(home)
    assert env["KIMI_DISABLE_TELEMETRY"] == "1"
    assert env["KIMI_CODE_NO_AUTO_UPDATE"] == "1"
    assert env["PATH"].split(os.pathsep)[0] == str(executable.parent)
    assert "KIMI_CODE_TRUST_WORKSPACE" not in env
    assert (home / "no-skills").is_dir()
    assert fingerprint == {"home": str(home)}


def fake_kimi(tmp_path):
    capture = tmp_path / "fake-argv.json"
    script = tmp_path / "fake-kimi"
    script.write_text(
        "#!/bin/sh\n"
        f"printf '%s' \"$2\" > {str(capture)!r}.prompt\n"
        f"printf '%s' \"$*\" > {str(capture)!r}.argv\n"
        'printf \'%s\\n\' \'{"role":"meta","type":"system.version","version":"2.2.0"}\'\n'
        'printf \'%s\\n\' \'{"role":"assistant","content":"ADAPTER_OK"}\'\n'
        'printf \'%s\\n\' \'{"role":"meta","type":"session.resume_hint",'
        '"session_id":"session_fake"}\'\n'
    )
    script.chmod(0o700)
    return script, capture


def test_adapter_delivers_stdin_prompt_as_dash_p(tmp_path):
    script, capture = fake_kimi(tmp_path)
    prompt = "Fix the dtype mismatch.\nLine two stays intact."
    result = subprocess.run(
        [sys.executable, "-m", "mindie_bench.runner.drivers.kimi", str(script), "--output-format",
         "stream-json"],
        input=prompt.encode(),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert Path(str(capture) + ".prompt").read_text() == prompt
    assert Path(str(capture) + ".argv").read_text() == f"-p {prompt} --output-format stream-json"
    events = [json.loads(line) for line in result.stdout.decode().splitlines()]
    assert events[1] == {"role": "assistant", "content": "ADAPTER_OK"}


def test_adapter_rejects_oversized_prompt(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mindie_bench.runner.drivers.kimi", "/bin/true"],
        input=b"x" * (1024 * 1024 + 1),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert b"adapter budget" in result.stderr


def test_adapter_fails_closed_on_missing_executable():
    result = subprocess.run(
        [sys.executable, "-m", "mindie_bench.runner.drivers.kimi", "/nonexistent/kimi"],
        input=b"hi",
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert b"failed to exec" in result.stderr


def sse(rows):
    return ("".join("data: " + json.dumps(row) + "\n\n" for row in rows) + "data: [DONE]\n\n").encode()


def chunk(delta, finish=None, usage=None):
    row = {
        "id": "chatcmpl-bench",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "k3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        row["usage"] = usage
    return row


@pytest.mark.skipif(not KIMI_BIN.exists(), reason="pinned kimi binary is not installed")
@pytest.mark.parametrize("sandboxed", [False, True])
def test_end_to_end_real_kimi_through_driver_and_stream_process(tmp_path, sandboxed):
    if sandboxed and platform.system() != "Darwin":
        pytest.skip("native FSEvents boundary is macOS-specific")
    secret = tmp_path / "private-source.txt"
    secret.write_text("PRIVATE_CORPUS_MUST_NOT_LEAK")
    tool_command = "printf 'e2e-ok' > out.txt"
    if sandboxed:
        tool_command = (
            f"if cat {shlex.quote(str(secret))} > denied.txt 2>/dev/null; "
            "then exit 19; fi; " + tool_command
        )
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            requests.append(body)
            if not any(m.get("role") == "tool" for m in body.get("messages", [])):
                rows = [
                    chunk(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_e2e",
                                    "type": "function",
                                    "index": 0,
                                    "function": {
                                        "name": "Bash",
                                        "arguments": json.dumps(
                                            {"command": tool_command}
                                        ),
                                    },
                                }
                            ],
                        }
                    ),
                    chunk({}, "tool_calls"),
                    chunk(
                        {},
                        None,
                        {
                            "prompt_tokens": 11,
                            "completion_tokens": 3,
                            "total_tokens": 14,
                            "prompt_cache_hit_tokens": 5,
                        },
                    ),
                ]
            else:
                rows = [
                    chunk({"role": "assistant", "content": "E2E_DONE"}),
                    chunk({}, "stop"),
                    chunk(
                        {}, None, {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19}
                    ),
                ]
            payload = sse(rows)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        profile = profile_stub(tmp_path)
        env = profile.env
        home_dir = profile.root / "home"
        home_dir.mkdir()
        env["HOME"] = str(home_dir)
        build_kimi_profile(
            profile.root,
            env,
            ExperimentConfig(
                schema_version="1.0",
                harness=HarnessConfig(
                    name="kimi",
                    executable=str(KIMI_BIN),
                    model="k3",
                    expected_version="2.2.0",
                    provider_url="https://provider.invalid/v1",
                    api_key_env="MINDIE_BENCH_TEST_KIMI_KEY",
                ),
            ),
        )
        env.update(
            MINDIE_BENCH_GATEWAY=f"http://127.0.0.1:{server.server_port}",
            MINDIE_BENCH_DUMMY_KEY="broker-only",
        )
        driver = KimiDriver()
        command = driver.command(profile, harness_config(executable=str(KIMI_BIN)))
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        if sandboxed:
            command = native_command(
                command,
                read=[Path(sys.prefix), Path(sys.base_prefix), KIMI_BIN.parent],
                write=[workspace, profile.root],
                cwd=workspace,
                policy_path=tmp_path / "agent.sb",
                gateway_port=server.server_port,
                directory_watchers=True,
            )
        gateway = SimpleNamespace(
            unknown=False, infrastructure_error=None, stop_reason=None, model_calls=0
        )
        result = stream_process(
            command,
            env=env,
            cwd=workspace,
            prompt="Call Bash once, then answer.",
            output=tmp_path / "native_events.jsonl",
            stderr=tmp_path / "agent.stderr",
            timeout=60,
            max_bytes=4 * 1024 * 1024,
            consume=driver.consume,
            gateway=gateway,
        )
    finally:
        server.shutdown()
    assert result.errors == []
    assert result.returncode == 0
    trajectory = driver.trajectory()
    assert trajectory.native_completed
    assert trajectory.native_session_id.startswith("session_")
    assert trajectory.total_turns == 2
    assert [m["text"] for m in trajectory.public_messages] == ["E2E_DONE"]
    assert len(trajectory.calls) == 1
    assert trajectory.calls[0].name == "Bash"
    assert trajectory.calls[0].arguments == {"command": tool_command}
    assert trajectory.calls[0].ended_sec is not None
    assert (workspace / "out.txt").read_text() == "e2e-ok"
    if sandboxed:
        assert (workspace / "denied.txt").read_text() == ""
        assert "PRIVATE_CORPUS_MUST_NOT_LEAK" not in json.dumps(requests)
    assert [request.get("model") for request in requests] == ["k3", "k3"]
    first = requests[0]
    assert first["stream"] is True
    assert first["stream_options"] == {"include_usage": True}
    assert first["thinking"]["type"] == "enabled"
    assert first["thinking"]["effort"] == "high"
    events = (tmp_path / "native_events.jsonl").read_text().splitlines()
    assert any("session.resume_hint" in line for line in events)
