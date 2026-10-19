"""Extract provider metering, including inference initiated by component tools.

[Role] Normalize Responses, Messages and Chat Completions usage.
[Invariants] Cached input is counted once; absent terminal usage remains incomplete.
[Timeouts & Locks] Incremental parsing has bounded buffers and no waits or locks.
[Side-Effect Policy] Reads supplied provider bytes only.
"""

import json

from mindie_bench.models import Tokens


class UsageMeter:
    def __init__(self):
        self.tokens = Tokens()
        self.buffer = b""
        self.json_body = bytearray()
        self.input_seen = False

    def feed(self, chunk: bytes):
        if len(self.json_body) + len(chunk) <= 4 * 1024 * 1024:
            self.json_body.extend(chunk)
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if line.startswith(b"data: "):
                self.event(line[6:])
        if len(self.buffer) > 4 * 1024 * 1024:
            self.buffer = b""

    def event(self, data):
        try:
            value = json.loads(data)
        except ValueError:
            return
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if kind == "response.completed":
            self.openai(value.get("response", {}).get("usage"), complete=True)
        elif kind == "message_start":
            self.anthropic(value.get("message", {}).get("usage", {}))
        elif kind == "message_delta":
            self.anthropic(
                value.get("usage", {}), complete=bool(value.get("delta", {}).get("stop_reason"))
            )
        elif value.get("object") in {"chat.completion", "chat.completion.chunk"}:
            self.chat_completions(value.get("usage"))
        elif kind == "message":
            self.anthropic(value.get("usage", {}), complete=bool(value.get("stop_reason")))
        elif "usage" in value:
            self.openai(value["usage"], complete=kind not in {"response.failed", "error"})

    def chat_completions(self, usage):
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details") or {}
        total = usage.get("prompt_tokens")
        cached = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))
        output = usage.get("completion_tokens")
        if all(type(v) is int and v >= 0 for v in (total, cached, output)) and cached <= total:
            self.tokens = Tokens(
                input_uncached=total - cached, input_cached=cached, output=output, complete=True
            )

    def openai(self, usage, *, complete):
        if not isinstance(usage, dict):
            return
        details = usage.get("input_tokens_details", {})
        if not isinstance(details, dict):
            return
        total, cached, output = (
            usage.get("input_tokens"),
            details.get("cached_tokens", 0),
            usage.get("output_tokens"),
        )
        if all(type(v) is int and v >= 0 for v in (total, cached, output)) and cached <= total:
            self.tokens = Tokens(
                input_uncached=total - cached, input_cached=cached, output=output, complete=complete
            )

    def anthropic(self, usage, *, complete=False):
        if not isinstance(usage, dict):
            return
        for key, field in [("output_tokens", "output"), ("cache_read_input_tokens", "input_cached")]:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                setattr(self.tokens, field, value)
        if "input_tokens" in usage:
            values = [usage.get("input_tokens"), usage.get("cache_creation_input_tokens", 0)]
            if all(type(v) is int and v >= 0 for v in values):
                self.tokens.input_uncached = sum(values)
                self.tokens.cache_creation = values[1]
                self.input_seen = True
        if complete:
            self.tokens.complete = (
                self.input_seen
                and type(usage.get("output_tokens")) is int
                and usage["output_tokens"] >= 0
            )

    def finish(self):
        if not self.tokens.complete:
            self.event(bytes(self.json_body))


def sum_usage(meters: dict, requests: int) -> Tokens:
    return Tokens(
        input_uncached=sum(m.tokens.input_uncached for m in meters.values()),
        input_cached=sum(m.tokens.input_cached for m in meters.values()),
        cache_creation=sum(m.tokens.cache_creation for m in meters.values()),
        output=sum(m.tokens.output for m in meters.values()),
        complete=len(meters) == requests and all(m.tokens.complete for m in meters.values()),
    )
