"""Claude print stream-json; deduplicate message IDs and correlate tool results."""

from mindie_bench.models import Tokens, ToolCall, Trajectory
from mindie_bench.runner.drivers import normalize_tool


class ClaudeCodeDriver:
    name = "claude-code"

    def __init__(self):
        self.result = Trajectory()
        self.calls = {}
        self.messages = {}
        self.server_components = {}
        self.public_blocks = set()

    def command(self, profile, config):
        self.server_components = profile.fingerprint["server_components"]
        if (profile.root / "skills").exists():
            import shutil
            from pathlib import Path

            shutil.copytree(
                profile.root / "skills",
                Path(profile.env["CLAUDE_CONFIG_DIR"]) / "skills",
                dirs_exist_ok=True,
            )
        args = [
            config.executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            config.model,
            "--strict-mcp-config",
            "--mcp-config",
            str(profile.root / "mcp.json"),
            "--setting-sources",
            "",
            "--settings",
            str(profile.root / "settings.json"),
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--no-chrome",
        ]
        if config.reasoning_effort:
            args += ["--effort", config.reasoning_effort]
        if (profile.root / "skills").exists():
            args += ["--add-dir", str(profile.root / "skills")]
        return args

    def consume(self, event, elapsed, generation):
        kind = event.get("type")
        message = event.get("message", {})
        if kind == "assistant":
            identity = message.get("id")
            if not identity:
                raise ValueError("Claude assistant message has no ID")
            self.messages[identity] = message
            turn = max(generation, len(self.messages), 1)
            self.result.total_turns = max(turn, self.result.total_turns)
            for index, block in enumerate(message.get("content", [])):
                if block.get("type") == "tool_use" and block["id"] not in self.calls:
                    name, component = normalize_tool(block["name"], None, self.server_components)
                    self.calls[block["id"]] = ToolCall(
                        id=block["id"],
                        name=name,
                        component=component,
                        arguments=block.get("input", {}),
                        turn=turn,
                        started_sec=elapsed,
                    )
                elif block.get("type") == "text":
                    public_id = (identity, event.get("uuid"), index, block["text"])
                    if public_id not in self.public_blocks:
                        self.public_blocks.add(public_id)
                        self.result.public_messages.append({"role": "assistant", "text": block["text"]})
        if kind == "user":
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    call = self.calls.get(block.get("tool_use_id"))
                    if call:
                        call.ended_sec, call.is_error = elapsed, bool(block.get("is_error", False))
        if kind == "result":
            self.result.native_completed = event.get("subtype") == "success" and not event.get(
                "is_error"
            )
            if not self.result.native_completed:
                self.result.native_error = str(event.get("errors", event.get("subtype")))
            usage = event.get("usage")
            if isinstance(usage, dict):
                self.result.tokens = self.tokens(usage, complete=True)

    @staticmethod
    def tokens(usage, *, complete=False):
        counters = [
            usage.get(k, 0)
            for k in (
                "input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
            )
        ]
        if not all(type(v) is int and v >= 0 for v in counters):
            raise ValueError("invalid Claude usage counters")
        return Tokens(
            input_uncached=counters[0] + counters[3],
            input_cached=counters[1],
            cache_creation=counters[3],
            output=counters[2],
            complete=complete,
        )

    def trajectory(self):
        self.result.calls = list(self.calls.values())
        if not self.result.tokens.complete:
            partial = [self.tokens(m.get("usage", {})) for m in self.messages.values()]
            self.result.tokens = Tokens(
                input_uncached=sum(t.input_uncached for t in partial),
                input_cached=sum(t.input_cached for t in partial),
                cache_creation=sum(t.cache_creation for t in partial),
                output=sum(t.output for t in partial),
            )
        return self.result
