"""DSH headless JSON event stream and native session receipt.

[Role]
Parse DSH's bounded `--json` projection into a benchmark trajectory and expose
the real session identity needed for a later `--session-id` continuation.

[Invariants]
- `native_session_id` comes only from the opening `session` event. A missing or
  malformed event leaves it unset and never fabricates a G3 session identity.
- One stream may not change its session identity; a different id is a protocol
  error rather than a silent overwrite.
- `capture_ready` requires both a real session event and a completed turn. It
  says the identity is available for a host capture adapter, not that a G3 task
  succeeded.
- The projection is bounded and is not the full Session log. This driver never
  reads transcript history, replays a session or sends external messages.

[Timeouts & Locks]
- Pure event parsing. Process, stream and timeout ownership stays with the
  orchestrator's `stream_process`.

[Side-Effect Policy]
- No I/O, subprocess, network or history read. Only the passed event objects are
  consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mindie_bench.models import Tokens, ToolCall, Trajectory
from mindie_bench.runner.drivers import normalize_tool
from mindie_bench.scorer.tool_observations import SCHEMA, observe_dsh_result

DSH_NATIVE_SESSION_SCHEMA = "mindie-bench-dsh-native-session/1"
MAX_SESSION_ID_LENGTH = 1024

NATIVE_TOOL_NAMES = {
    "read": "Read",
    "edit": "Edit",
    "write": "Write",
    "grep": "Grep",
    "glob": "Glob",
}


@dataclass(frozen=True)
class DshSessionCapture:
    """Honest DSH session-capture status derived from the actual event stream."""

    status: Literal["available", "incomplete", "unavailable"]
    session_id: str | None
    continuation_command: tuple[str, ...] | None
    native_completed: bool
    source: Literal["session_event"] | None
    reason: str | None = None
    history_read: bool = field(default=False, init=False)
    external_messages_sent: bool = field(default=False, init=False)
    limitations: tuple[str, ...] = ()

    @property
    def capture_ready(self) -> bool:
        return self.status == "available" and self.session_id is not None

    def to_dict(self) -> dict:
        return {
            "schema_version": DSH_NATIVE_SESSION_SCHEMA,
            "harness": "dsh",
            "status": self.status,
            "session_id": self.session_id,
            "capture_ready": self.capture_ready,
            "continuation_command": list(self.continuation_command or ()),
            "native_completed": self.native_completed,
            "source": self.source,
            "reason": self.reason,
            "history_read": self.history_read,
            "external_messages_sent": self.external_messages_sent,
            "limitations": list(self.limitations),
        }


def _validate_session_identity(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_SESSION_ID_LENGTH:
        raise ValueError("DSH session event has no valid native session identity")
    return value


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
        self._base_command: tuple[str, ...] | None = None
        self._session_cwd: str | None = None
        self._completed_turn: int | None = None
        self._session_reason = "no opening session event"

    def command(self, profile, config):
        self.server_components = profile.fingerprint["server_components"]
        # headless reads the task from stdin when no positional task is given.
        self._base_command = (config.executable, "--profile", "bench", "--json")
        return list(self._base_command)

    def continuation_command(self, session_id: str) -> list[str]:
        """Continue the observed session without changing the configured launcher.

        The caller may use the installed DSH launcher or a headless shim; only
        the verified `--session-id` continuation flag is appended.
        """
        identity = _validate_session_identity(session_id)
        if self._base_command is None:
            raise ValueError("DSH command must be built before requesting a continuation")
        return [*self._base_command, "--session-id", identity]

    def native_session_capture(self) -> DshSessionCapture:
        """Return session-capture facts, distinguishing missing from completed."""
        limitations = (
            "DSH --json is a bounded projection, not the full Session log; history must not be replayed",
            "this receipt is not a G3 task outcome and does not replace the orchestrator capture adapter",
        )
        identity = self.result.native_session_id
        if identity is None:
            return DshSessionCapture(
                status="unavailable",
                session_id=None,
                continuation_command=None,
                native_completed=self.result.native_completed,
                source=None,
                reason=self._session_reason,
                limitations=limitations,
            )
        try:
            continuation = self.continuation_command(identity)
        except ValueError:
            continuation = None
        ready = self.result.native_completed and self._completed_turn is not None
        return DshSessionCapture(
            status="available" if ready else "incomplete",
            session_id=identity,
            continuation_command=tuple(continuation) if continuation else None,
            native_completed=self.result.native_completed,
            source="session_event",
            reason=None if ready else "completed native turn identity is unavailable",
            limitations=limitations,
        )

    def native_capture_payload(self) -> dict:
        """Describe a completed DSH session for a host-side G3 capture adapter.

        The payload contains no transcript or history bytes: DSH's `--json`
        projection is bounded, so a caller must not mistake it for the full
        Session log. It is a capture descriptor, not a task-success claim.
        """
        capture = self.native_session_capture()
        if not capture.capture_ready:
            raise ValueError("DSH capture requires a completed native session event")
        return {
            "harness": "dsh",
            "session_id": capture.session_id,
            "turn_id": str(self._completed_turn),
            "completed": self.result.native_completed,
            "cwd": self._session_cwd,
            "continuation_command": list(capture.continuation_command or ()),
            "source": "headless_json_projection",
            "history_read": False,
            "transcript_path": None,
            "limitations": list(capture.limitations),
            "schema_version": DSH_NATIVE_SESSION_SCHEMA,
        }

    def consume(self, event, elapsed, generation):
        kind = event.get("type")
        if kind == "session":
            raw = event.get("sessionId")
            if not isinstance(raw, str) or not raw or len(raw) > MAX_SESSION_ID_LENGTH:
                # A malformed opening event leaves capture unavailable; it must
                # not fail an otherwise healthy G0-G2 trial or invent an id.
                self._session_reason = "opening session event had no valid native session identity"
                return
            if self.result.native_session_id not in {None, raw}:
                raise ValueError("DSH stream changed its native session identity")
            self.result.native_session_id = raw
            self._session_reason = None
            cwd = event.get("cwd")
            if isinstance(cwd, str) and cwd:
                if self._session_cwd not in {None, cwd}:
                    raise ValueError("DSH stream changed its native workspace identity")
                self._session_cwd = cwd
            return
        if kind == "error":
            message = event.get("message") or event.get("code") or "dsh error event"
            self.result.native_error = str(message)
            self.result.native_completed = False
            self._completed_turn = None
            return
        if kind == "status":
            phase = event.get("phase")
            turn = event.get("turn")
            if type(turn) is int and turn > 0:
                self._turn = turn
                self.result.total_turns = max(self.result.total_turns, turn)
            if phase == "turn_start":
                self.result.native_completed = False
                self._completed_turn = None
            elif phase == "step_end":
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
                    self.result.native_error = None
                    self._completed_turn = turn if type(turn) is int and turn > 0 else None
                elif isinstance(reason, dict):
                    self.result.native_completed = False
                    self._completed_turn = None
                    error = reason.get("error")
                    detail = (
                        error.get("message") or error.get("code") if isinstance(error, dict) else error
                    )
                    self.result.native_error = str(detail or reason.get("kind") or "turn_end")
                else:
                    self.result.native_completed = False
                    self._completed_turn = None
                    self.result.native_error = "turn_end without a reason object"
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
