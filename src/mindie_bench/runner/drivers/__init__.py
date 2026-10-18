"""Small native JSON-stream adapters, not a replacement agent harness."""

from __future__ import annotations

from typing import Protocol

from mindie_bench.models import HarnessConfig, Trajectory
from mindie_bench.runner.profile_builder import Profile


class HarnessDriver(Protocol):
    name: str

    def command(self, profile: Profile, config: HarnessConfig) -> list[str]: ...
    def consume(self, event: dict, elapsed: float, generation: int) -> None: ...
    def trajectory(self) -> Trajectory: ...


def normalize_tool(name: str, server: str | None, components: dict) -> tuple[str, str | None]:
    if name.startswith("mcp__") and "__" in name[5:]:
        encoded_server, name = name[5:].rsplit("__", 1)
        server = server or encoded_server
    component = components.get(server)
    if component == "remote-dev" and not name.startswith("remote_"):
        name = "remote_" + name
    if component == "knowledge" and name in {"query", "explain"}:
        name = "knowledge_" + name
    return name, component


def driver_for(name: str):
    from .claude_code import ClaudeCodeDriver
    from .codex import CodexDriver
    from .dsh import DshDriver
    from .fixture import FixtureDriver

    return {
        "codex": CodexDriver,
        "claude-code": ClaudeCodeDriver,
        "dsh": DshDriver,
        "fixture": FixtureDriver,
    }[name]()
