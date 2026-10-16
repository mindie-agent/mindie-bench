"""Codex exec JSONL: completed native items and explicit usage only."""

import json
from pathlib import Path

from mindie_bench.models import Tokens, ToolCall, Trajectory
from mindie_bench.runner.drivers import normalize_tool


class CodexDriver:
    name = "codex"

    def __init__(self):
        self.result = Trajectory()
        self.calls: dict[str, ToolCall] = {}
        self.server_components = {}

    def command(self, profile, config):
        self.server_components = profile.fingerprint["server_components"]
        lines = [
            f"model = {json.dumps(config.model)}",
            'model_provider = "benchmark"',
            'approval_policy = "never"',
            "[model_providers.benchmark]",
            'name = "Benchmark model broker"',
            f"base_url = {json.dumps(profile.env['MINDIE_BENCH_GATEWAY'] + '/v1')}",
            'wire_api = "responses"',
            'env_key = "MINDIE_BENCH_DUMMY_KEY"',
            "request_max_retries = 0",
            "stream_max_retries = 0",
        ]
        if config.reasoning_effort:
            lines.insert(1, f"model_reasoning_effort = {json.dumps(config.reasoning_effort)}")
        mcp = json.loads((profile.root / "mcp.json").read_text())["mcpServers"]
        for name, server in mcp.items():
            lines += [
                f"[mcp_servers.{json.dumps(name)}]",
                f"command = {json.dumps(server['command'])}",
                f"args = {json.dumps(server['args'])}",
            ]
            if server["env"]:
                lines.append(f"[mcp_servers.{json.dumps(name)}.env]")
                lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in server["env"].items()]
        (Path(profile.env["CODEX_HOME"]) / "config.toml").write_text("\n".join(lines) + "\n")
        if (profile.root / "skills").exists():
            import shutil

            shutil.copytree(
                profile.root / "skills", Path(profile.env["CODEX_HOME"]) / "skills", dirs_exist_ok=True
            )
        return [
            config.executable,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--color",
            "never",
            "-",
        ]

    def consume(self, event, elapsed, generation):
        kind = event.get("type")
        if kind in {"error", "turn.failed"}:
            self.result.native_error = str(event.get("error", event.get("message", kind)))
        if kind == "turn.completed":
            self.result.native_completed = True
            usage = event.get("usage")
            if isinstance(usage, dict):
                total, cached, output = (
                    usage.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens")
                )
                if all(type(v) is int and v >= 0 for v in (total, cached, output)) and cached <= total:
                    self.result.tokens = Tokens(
                        input_uncached=total - cached, input_cached=cached, output=output, complete=True
                    )
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item_kind = item.get("type")
        if item_kind == "agent_message" and kind == "item.completed":
            self.result.public_messages.append({"role": "assistant", "text": item.get("text", "")})
        if item_kind not in {"command_execution", "mcp_tool_call", "file_change"}:
            return
        identity = item.get("id")
        if not isinstance(identity, str):
            raise ValueError("Codex tool item has no ID")
        if identity not in self.calls:
            tool = {"command_execution": "bash", "file_change": "Edit"}.get(
                item_kind, item.get("tool", "")
            )
            tool, component = normalize_tool(tool, item.get("server"), self.server_components)
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if item_kind == "command_execution":
                arguments = {"command": item.get("command", "")}
            if item_kind == "file_change":
                arguments = {"paths": [change["path"] for change in item.get("changes", [])]}
            self.calls[identity] = ToolCall(
                id=identity,
                name=tool,
                component=component,
                arguments=arguments,
                turn=max(1, generation),
                started_sec=elapsed if kind == "item.started" else None,
            )
        if kind == "item.completed":
            self.calls[identity].ended_sec = elapsed
            self.calls[identity].is_error = (
                item.get("status") == "failed" or item.get("exit_code", 0) != 0
            )
        self.result.total_turns = max(self.result.total_turns, generation)

    def trajectory(self):
        self.result.calls = list(self.calls.values())
        return self.result
