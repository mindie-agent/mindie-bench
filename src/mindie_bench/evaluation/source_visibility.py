"""Exact initial source visibility in recorded native tool outputs.

This observes returned bytes independently of command completion. It does not
identify a semantic root cause, inspect a final workspace, or certify a fix.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

from mindie_bench.io import safe_relative
from mindie_bench.models import SourceEnvironment, Trajectory

SCHEMA = "mindie-bench-source-output-visibility/1"
MAX_BLOB_BYTES = 1024 * 1024
MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _time(value):
    if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
        raise ValueError("source-output time must be finite and nonnegative")
    return value


def _public_asset(public_root: Path, target: str) -> dict:
    root = Path(public_root)
    path = root / target
    if (
        not root.is_dir()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(
            f"initial public asset is missing or is not a confined regular file: {target!r}"
        )
    content = path.read_bytes()
    if not 0 < len(content) <= MAX_BLOB_BYTES:
        raise ValueError("initial public asset is empty or exceeds visibility bound")
    content.decode("utf-8")
    return dict(
        target=target,
        git_blob_id=None,
        source_origin="task_public_asset",
        source_bytes=len(content),
        source_sha256=_sha(content),
        content=content,
    )


def _initial_blobs(environment, targets, public_root=None):
    def git(*args):
        return subprocess.run(
            ["git", "-C", environment.repo_path, *args],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout

    if git("cat-file", "-t", environment.base_commit).strip() != b"commit":
        raise ValueError("source visibility requires an exact initial Git commit")
    blobs = []
    for target in dict.fromkeys(targets):
        safe_relative(target)
        if any(ord(char) < 32 for char in target) or target.startswith("-"):
            raise ValueError("source visibility requires an ordinary relative Git path")
        entry = git("ls-tree", "-z", environment.base_commit, "--", target).split(b"\0")
        entries = [item for item in entry if item]
        if not entries and public_root is not None:
            blobs.append(_public_asset(public_root, target))
            continue
        if len(entries) != 1:
            raise ValueError(f"initial target is not a unique Git blob: {target!r}")
        header, path = entries[0].split(b"\t", 1)
        mode, kind, identity = header.split(b" ")
        if mode not in {b"100644", b"100755"} or kind != b"blob" or path.decode() != target:
            raise ValueError(f"initial target must be a regular Git blob: {target!r}")
        content = git("cat-file", "blob", identity.decode())
        size = len(content)
        if not 0 < size <= MAX_BLOB_BYTES:
            raise ValueError("initial source blob is empty or exceeds visibility bound")
        content.decode("utf-8")
        blobs.append(
            dict(
                target=target,
                git_blob_id=identity.decode(),
                source_origin="git_blob",
                source_bytes=size,
                source_sha256=_sha(content),
                content=content,
            )
        )
    return blobs


def _text_content(content, prefix):
    if isinstance(content, str):
        yield prefix, content
    elif isinstance(content, list):
        for index, block in enumerate(content):
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                yield f"{prefix}[{index}].text", block["text"]


def _outputs(event):
    """Only native tool outputs, never assistant prose or tool input."""
    item = event.get("item")
    if event.get("type") == "item.completed" and isinstance(item, dict):
        if item.get("type") == "command_execution" and isinstance(item.get("aggregated_output"), str):
            yield item.get("id"), "item.aggregated_output", item["aggregated_output"]
        elif item.get("type") == "mcp_tool_call" and isinstance(item.get("result"), dict):
            result = item["result"]
            for key in ("structured_content", "structuredContent"):
                structured = result.get(key)
                preview = structured.get("preview") if isinstance(structured, dict) else None
                if isinstance(preview, dict) and isinstance(preview.get("stdout"), str):
                    yield item.get("id"), f"item.result.{key}.preview.stdout", preview["stdout"]
            for field, text in _text_content(result.get("content"), "item.result.content"):
                yield item.get("id"), field, text
    message = event.get("message")
    if event.get("type") == "user" and isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            for index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    for field, text in _text_content(
                        block.get("content"), f"message.content[{index}].content"
                    ):
                        yield block.get("tool_use_id"), field, text


def observe_source_visibility(
    native_events: Path,
    trajectory: Trajectory,
    environment: SourceEnvironment,
    targets: list[str],
    *,
    public_root: Path | None = None,
) -> dict:
    """Read immutable evidence locally; errors are never an empty observation."""
    blobs = _initial_blobs(environment, targets, public_root)
    calls = {call.id: call for call in trajectory.calls}
    if len(calls) != len(trajectory.calls):
        raise ValueError("source visibility cannot correlate duplicate native call IDs")
    witnesses, event_count, output_count, unmapped = [], 0, 0, 0
    digest = hashlib.sha256()
    with native_events.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line) > MAX_EVENT_LINE_BYTES:
                raise ValueError(f"native event line {line_number} exceeds visibility bound")
            digest.update(line)
            try:
                wrapper = json.loads(line)
                event = json.loads(wrapper["native_line"])
                if not isinstance(event, dict):
                    raise ValueError("native event must be an object")
                observed = _time(wrapper.get("observed_sec"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid native event line {line_number}: {exc}") from exc
            event_count += 1
            seen = set()
            for identity, field, text in _outputs(event):
                output_count += 1
                output = text.encode("utf-8")
                call = calls.get(identity)
                if call is None:
                    unmapped += 1
                for blob in blobs:
                    offset = output.find(blob["content"])
                    if offset < 0 or blob["target"] in seen:
                        continue
                    seen.add(blob["target"])
                    observed_at = (
                        observed if observed is not None else _time(call.ended_sec if call else None)
                    )
                    witnesses.append(
                        dict(
                            **{key: value for key, value in blob.items() if key != "content"},
                            native_event_line=line_number,
                            native_event_wrapper_sha256=_sha(line.rstrip(b"\r\n")),
                            native_event_json_sha256=_sha(wrapper["native_line"].encode("utf-8")),
                            call_id=identity,
                            turn=call.turn if call else None,
                            call_classification=(
                                call.result_observation.classification
                                if call and call.result_observation
                                else None
                            ),
                            output_field=field,
                            output_bytes=len(output),
                            output_sha256=_sha(output),
                            output_byte_offset=offset,
                            output_observed_sec=observed_at,
                            output_clock_basis=(
                                "native_event_observed_sec"
                                if observed is not None
                                else "correlated_call_ended_sec"
                                if observed_at is not None
                                else None
                            ),
                        )
                    )
    first = witnesses[0] if witnesses else None
    no_match = "no_exact_initial_source_blob_in_supported_tool_output"
    return dict(
        schema=SCHEMA,
        state="observed" if witnesses else "no_exact_output_match",
        source_base_commit=environment.base_commit,
        target_blobs=[{key: value for key, value in blob.items() if key != "content"} for blob in blobs],
        native_events_sha256=digest.hexdigest(),
        native_event_count=event_count,
        supported_tool_output_count=output_count,
        unmapped_tool_output_count=unmapped,
        witnesses=witnesses,
        first_verified_source_output_turn=first["turn"] if first else None,
        first_verified_source_output_sec=first["output_observed_sec"] if first else None,
        reasons=dict(
            first_verified_source_output_turn=(
                no_match
                if first is None
                else "output_call_not_in_trajectory"
                if first["turn"] is None
                else None
            ),
            first_verified_source_output_sec=(
                no_match
                if first is None
                else "output_observation_time_missing"
                if first["output_observed_sec"] is None
                else None
            ),
        ),
        scope=(
            "whole initial Git blob or declared task-public asset in actual recorded tool output; "
            "no final workspace, root-cause or fix inference"
        ),
    )


def source_evidence_metrics(
    visibility: dict | None, successful_accesses: list[dict]
) -> tuple[dict, dict]:
    """Keep success-only proxies and independent byte-visibility observations distinct."""
    if visibility is not None and visibility.get("schema") != SCHEMA:
        raise ValueError("unsupported source visibility observation schema")
    verified_turn = visibility["first_verified_source_output_turn"] if visibility else None
    verified_sec = visibility["first_verified_source_output_sec"] if visibility else None
    turns = [row["turn"] for row in successful_accesses]
    times = [row["ended_sec"] for row in successful_accesses if row["ended_sec"] is not None]
    if verified_turn is not None:
        turns.append(verified_turn)
    if verified_sec is not None:
        times.append(verified_sec)
    timing_missing = any(row["ended_sec"] is None for row in successful_accesses) or (
        visibility is not None and bool(visibility["witnesses"]) and verified_sec is None
    )
    return (
        dict(
            first_verified_source_output_turn=verified_turn,
            first_verified_source_output_sec=verified_sec,
            first_source_evidence_turn=min(turns, default=None),
            first_source_evidence_sec=min(times, default=None) if not timing_missing else None,
        ),
        dict(
            source_visibility=visibility
            or dict(
                schema=SCHEMA, state="not_measured", reason="native_output_observation_not_supplied"
            ),
            source_evidence_basis=dict(
                schema="mindie-bench-source-evidence/1",
                turn="minimum successful target access turn or exact initial-blob output turn",
                time="minimum successful target access completion or exact initial-blob output observation",
                legacy_localization_unchanged=True,
                comparison_requires_explicit_basis=True,
                reasons=dict(
                    first_verified_source_output=(
                        visibility["reasons"] if visibility else "native_output_observation_not_supplied"
                    ),
                    first_source_evidence_turn=None if turns else "no_source_evidence_with_turn",
                    first_source_evidence_sec=(
                        "source_evidence_output_or_end_time_missing"
                        if timing_missing
                        else None
                        if times
                        else "no_source_evidence_with_output_or_end_time"
                    ),
                ),
            ),
        ),
    )
