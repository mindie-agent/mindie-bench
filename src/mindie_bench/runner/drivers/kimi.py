"""Kimi Code stream-json driver and stdin->argv prompt adapter (Kimi Code 2.2.0).

[Role] Adapt `kimi -p PROMPT --output-format stream-json` to the harness driver
protocol. Kimi 2.2.0 reads the prompt only from the `-p` argument value and
ignores stdin, so this module doubles as the adapter the runner spawns:
`python -I -m mindie_bench.runner.drivers.kimi EXECUTABLE [ARGS...]` reads the
prompt from stdin and replaces itself with the pinned native binary.

[Invariants]
- Real 2.2.0 stream roles, captured from the native binary: `meta`
  (`system.version`, `turn.step.retrying`, `session.resume_hint`), `assistant`
  (`content` and/or `tool_calls` with JSON-string arguments), and `tool`
  (`tool_call_id` + text `content`). No other schema is synthesized.
- `session.resume_hint` is emitted only on a clean run and carries the native
  session identity; native failure exits nonzero with the error on stderr and
  no hint, so `native_completed` stays false and the trial fails closed.
- The stream carries no usage events and no structured tool error bit. Token
  accounting stays with the metering gateway; `is_error` and
  `result_observation` remain absent (None) rather than parsed from prose.
- The owned provider uses the native `kimi` wire type so thinking/effort reach
  the gateway in the same shape as the managed K3 configuration; CLI-level
  retries are disabled (`max_attempts_per_step = 1`) because the gateway
  already enforces zero automatic retries of paid requests.

[Timeouts & Locks] The adapter bounds the stdin prompt at 1 MiB; per-argument
  exec limits (E2BIG) surface as a clean adapter exit, not a hang. No locks.

[Side-Effect Policy] `command()` writes only inside owned `KIMI_CODE_HOME` (`config.toml`,
  `mcp.json`). The adapter creates no files and execs the pinned binary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mindie_bench.models import ToolCall, Trajectory
from mindie_bench.runner.drivers import normalize_tool
from mindie_bench.scorer.tool_observations import SCHEMA

MAX_PROMPT_BYTES = 1024 * 1024
MCP_STARTUP_TIMEOUT_MS = 30_000
MCP_TOOL_TIMEOUT_MS = 600_000


class KimiDriver:
    name = "kimi"

    def __init__(self):
        self.result = Trajectory(tool_observation_schema=SCHEMA)
        self.calls: dict[str, ToolCall] = {}
        self.server_components = {}
        self.steps = 0

    def command(self, profile, config):
        self.server_components = profile.fingerprint["server_components"]
        home = Path(profile.env["KIMI_CODE_HOME"])
        if not home.is_dir():
            raise ValueError("kimi profile is missing its owned KIMI_CODE_HOME")
        if not config.model_catalog_path:
            raise ValueError("the kimi driver requires an explicit model catalog")
        catalog = json.loads((profile.root / "model-catalog.json").read_text())
        entries = [entry for entry in catalog["models"] if entry.get("slug") == config.model]
        if len(entries) != 1:
            raise ValueError("kimi model catalog must contain exactly one entry for the pinned model")
        entry = entries[0]
        context = entry.get("max_context_size")
        if type(context) is not int or context < 1:
            raise ValueError("kimi catalog entry requires a positive max_context_size")
        effort = config.reasoning_effort
        supported = entry.get("support_efforts")
        if effort and isinstance(supported, list) and supported and effort not in supported:
            raise ValueError(f"reasoning effort {effort!r} is not in the catalog support_efforts")
        lines = [
            'default_model = "bench"',
            # `kimi -p` always runs with auto permissions; state it explicitly.
            'default_permission_mode = "auto"',
            "telemetry = false",
            "builtin_product_skills = false",
            "",
            "[providers.benchmark]",
            'type = "kimi"',
            f"base_url = {json.dumps(profile.env['MINDIE_BENCH_GATEWAY'] + '/v1')}",
            'api_key_env = "MINDIE_BENCH_DUMMY_KEY"',
            "",
            "[models.bench]",
            'provider = "benchmark"',
            f"model = {json.dumps(config.model)}",
            f"max_context_size = {context}",
        ]
        if isinstance(entry.get("capabilities"), list) and entry["capabilities"]:
            lines.append(f"capabilities = {json.dumps(entry['capabilities'])}")
        if isinstance(supported, list) and supported:
            lines.append(f"support_efforts = {json.dumps(supported)}")
        if isinstance(entry.get("default_effort"), str):
            lines.append(f"default_effort = {json.dumps(entry['default_effort'])}")
        lines += [
            "",
            '[providers."managed:kimi-code"]',
            'type = "kimi"',
            f"base_url = {json.dumps(profile.env['MINDIE_BENCH_GATEWAY'] + '/v1')}",
            'api_key_env = "MINDIE_BENCH_DUMMY_KEY"',
            "",
            '[models."kimi-code/k3"]',
            'provider = "managed:kimi-code"',
            f"model = {json.dumps(config.model)}",
            f"max_context_size = {context}",
        ]
        if isinstance(entry.get("capabilities"), list) and entry["capabilities"]:
            lines.append(f"capabilities = {json.dumps(entry['capabilities'])}")
        if isinstance(supported, list) and supported:
            lines.append(f"support_efforts = {json.dumps(supported)}")
        if isinstance(entry.get("default_effort"), str):
            lines.append(f"default_effort = {json.dumps(entry['default_effort'])}")
        lines += [
            "",
            "[loop_control]",
            "max_attempts_per_step = 1",
            "",
            "[background]",
            # Print mode must end with the main turn; steering pending background
            # tasks into new turns would defeat the trial's turn budget.
            'print_background_mode = "exit"',
        ]
        if effort:
            lines += ["", "[thinking]", "enabled = true", f"effort = {json.dumps(effort)}"]
        (home / "config.toml").write_text("\n".join(lines) + "\n")
        # bind_mcp_transport has already injected PATH and the gateway URL into
        # the shared mcp.json; the owned copy adds only explicit MCP timeouts.
        servers = json.loads((profile.root / "mcp.json").read_text())["mcpServers"]
        for server in servers.values():
            server.setdefault("startupTimeoutMs", MCP_STARTUP_TIMEOUT_MS)
            server.setdefault("toolTimeoutMs", MCP_TOOL_TIMEOUT_MS)
        (home / "mcp.json").write_text(json.dumps({"mcpServers": servers}, indent=1) + "\n")
        # --skills-dir replaces auto-discovery entirely, so benchmark skills are
        # the only ones offered; an empty owned dir keeps G0/G1 deterministic.
        skills = profile.root / "skills"
        if not skills.is_dir():
            skills = home / "no-skills"
            skills.mkdir(exist_ok=True)
        return [
            sys.executable,
            "-I",
            "-m",
            "mindie_bench.runner.drivers.kimi",
            config.executable,
            "-m",
            "bench",
            "--output-format",
            "stream-json",
            "--skills-dir",
            str(skills),
        ]

    def consume(self, event, elapsed, generation):
        role = event.get("role")
        if role == "meta":
            if event.get("type") == "session.resume_hint":
                identity = event.get("session_id")
                if not isinstance(identity, str) or not identity or len(identity) > 256:
                    raise ValueError("Kimi session.resume_hint has an invalid session identity")
                if self.result.native_session_id not in {None, identity}:
                    raise ValueError("Kimi stream changed its native session identity")
                self.result.native_session_id = identity
                self.result.native_completed = True
            return
        if role == "assistant":
            self.steps += 1
            self.result.total_turns = max(self.result.total_turns, self.steps)
            text = event.get("content")
            if isinstance(text, str) and text:
                self.result.public_messages.append({"role": "assistant", "text": text})
            tool_calls = event.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("Kimi assistant tool_calls is not a list")
            for call in tool_calls:
                if not isinstance(call, dict):
                    raise ValueError("Kimi tool call is not an object")
                identity = call.get("id")
                if not isinstance(identity, str) or not identity:
                    raise ValueError("Kimi tool call has no ID")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("Kimi tool call has no function payload")
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Kimi tool call has no tool name")
                if identity in self.calls:
                    continue
                name, component = normalize_tool(name, None, self.server_components)
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Kimi tool call arguments are not an object")
                self.calls[identity] = ToolCall(
                    id=identity,
                    name=name,
                    component=component,
                    arguments=arguments,
                    turn=max(1, self.steps),
                    started_sec=elapsed,
                )
            return
        if role == "tool":
            identity = event.get("tool_call_id")
            call = self.calls.get(identity) if isinstance(identity, str) else None
            if call is not None:
                call.ended_sec = elapsed

    def trajectory(self):
        self.result.calls = list(self.calls.values())
        return self.result


def main() -> int:
    """Read the benchmark prompt from stdin and exec kimi with it as `-p`."""
    argv = sys.argv[1:]
    if not argv:
        print("kimi adapter requires the native executable path", file=sys.stderr)
        return 2
    executable, args = argv[0], argv[1:]
    data = sys.stdin.buffer.read(MAX_PROMPT_BYTES + 1)
    if len(data) > MAX_PROMPT_BYTES:
        print(f"kimi prompt exceeds the {MAX_PROMPT_BYTES}-byte adapter budget", file=sys.stderr)
        return 2
    try:
        prompt = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"kimi prompt is not valid UTF-8: {exc}", file=sys.stderr)
        return 2
    try:
        os.execv(executable, [executable, "-p", prompt, *args])
    except TimeoutError:
        raise
    except OSError as exc:
        # Includes E2BIG when the prompt exceeds the platform per-argument limit.
        print(f"kimi adapter failed to exec {executable}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
