"""DSH headless JSON event stream.

The stream is the headless projection of the durable session: one opening
session event, then status/text/tool_call/tool_result events, and a terminal
final event. Tool names are normalized through the same MCP naming rule as the
other drivers.
"""

from __future__ import annotations

from mindie_bench.models import Tokens, ToolCall, Trajectory
from mindie_bench.runner.drivers import normalize_tool
from mindie_bench.scorer.tool_observations import SCHEMA, observe_dsh_result

NATIVE_TOOL_NAMES = {
    "read": "Read",
    "edit": "Edit",
    "write": "Write",
    "grep": "Grep",
    "glob": "Glob",
}


class DshDriver:
    name = "dsh"

    def __init__(self):
        self.result = Trajectory(tool_observation_schema=SCHEMA)
        self.calls: dict[str, ToolCall] = {}
        self.server_components = {}
        self._turn = 1
        self._tokens = [0, 0, 0, 0]
        self._usage_seen = False
        self._public_ids = set()

    def command(self, profile, config):
        self.server_components = profile.fingerprint["server_components"]
        # headless reads the task from stdin when no positional task is given.
        return [
            config.executable,
            "--profile",
            "bench",
            "--json",
        ]

    def consume(self, event, elapsed, generation):
        kind = event.get("type")
        if kind == "session":
            identity = event.get("sessionId")
            if isinstance(identity, str) and identity:
                self.result.native_session_id = identity
            return
        if kind == "status":
            phase = event.get("phase")
            turn = event.get("turn")
            if type(turn) is int and turn > 0:
                self._turn = turn
                self.result.total_turns = max(self.result.total_turns, turn)
            if phase == "step_end":
                usage = event.get("usage")
                if isinstance(usage, dict):
                    values = [
                        usage.get("inputTokens", 0),
                        usage.get("cacheReadTokens", 0),
                        usage.get("cacheWriteTokens", 0),
                        usage.get("outputTokens", 0),
                    ]
                    if all(type(value) is int and value >= 0 for value in values):
                        self._usage_seen = True
                        for index, value in enumerate(values):
                            self._tokens[index] += value
            elif phase == "turn_end":
                reason = event.get("reason")
                if isinstance(reason, dict) and reason.get("kind") == "completed":
                    self.result.native_completed = True
                elif isinstance(reason, dict):
                    self.result.native_error = str(reason.get("error") or reason.get("kind"))
            return
        if kind == "text":
            text = event.get("text")
            identity = (self._turn, text)
            if isinstance(text, str) and text and identity not in self._public_ids:
                self._public_ids.add(identity)
                self.result.public_messages.append({"role": "assistant", "text": text})
            return
        if kind == "tool_call":
            identity = event.get("callId")
            if not isinstance(identity, str) or not identity:
                raise ValueError("DSH tool_call has no callId")
            tool = event.get("tool")
            if not isinstance(tool, str) or not tool:
                raise ValueError("DSH tool_call has no tool name")
            tool, component = normalize_tool(tool, None, self.server_components)
            if component is None and tool in NATIVE_TOOL_NAMES:
                tool = NATIVE_TOOL_NAMES[tool]
            arguments = event.get("input")
            if not isinstance(arguments, dict):
                arguments = {}
            self.calls[identity] = ToolCall(
                id=identity,
                name=tool,
                component=component,
                arguments=arguments,
                turn=max(1, self._turn),
                started_sec=elapsed,
            )
            return
        if kind == "tool_result":
            identity = event.get("callId")
            call = self.calls.get(identity) if isinstance(identity, str) else None
            if call is None:
                return
            call.ended_sec = elapsed
            call.is_error = event.get("status") != "completed"
            call.result_observation = observe_dsh_result(event, call)
            return
        if kind == "final":
            text = event.get("text")
            identity = (self._turn, text)
            if isinstance(text, str) and text and identity not in self._public_ids:
                self._public_ids.add(identity)
                self.result.public_messages.append({"role": "assistant", "text": text})

    def trajectory(self):
        self.result.calls = list(self.calls.values())
        if self._usage_seen:
            self.result.tokens = Tokens(
                input_uncached=self._tokens[0] + self._tokens[2],
                input_cached=self._tokens[1],
                cache_creation=self._tokens[2],
                output=self._tokens[3],
                complete=True,
            )
        return self.result
