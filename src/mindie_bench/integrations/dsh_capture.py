"""Bind an observed DSH completion to the exact owned native session artifact.

[Role] Prepare post-trial capture metadata from the real headless event stream.
[Invariants] A projection is not a full transcript. Only the exact completed
session/turn/workspace and its owned v4 artifact are selected; no history scan.
[Timeouts & Locks] Local directory metadata only; no process waits or locks.
[Side-Effect Policy] Writes one native-stop.json metadata event in the trial
profile. Does not read transcript contents, start a model or claim C6 success.
"""

from __future__ import annotations

from pathlib import Path

from mindie_bench.io import write_json


def bind_dsh_capture(*, profile_root: Path, dsh_home: Path, workspace: Path, observed: dict) -> dict:
    """Bind metadata for a caller-owned capture backend, never a dead endpoint.

    G3 execution stays preflight-blocked until a capture backend survives the
    foreground headless process and its model calls can be metered.
    """
    profile_root = profile_root.resolve(strict=True)
    dsh_home = dsh_home.resolve(strict=True)
    workspace = workspace.resolve(strict=True)
    if dsh_home != profile_root / "dsh-home":
        raise ValueError("DSH capture must use this trial's owned native profile")
    if observed.get("harness") != "dsh" or observed.get("completed") is not True:
        raise ValueError("DSH capture requires an observed completed native turn")
    identity = observed["session_id"]
    if not identity or Path(identity).name != identity:
        raise ValueError("DSH native session identity must be one path segment")
    turn_id = observed["turn_id"]
    if not str(turn_id).isdigit() or int(turn_id) < 1:
        raise ValueError("DSH capture requires the actual completed native turn number")
    if not observed.get("cwd") or Path(observed["cwd"]).resolve(strict=True) != workspace:
        raise ValueError("DSH session workspace differs from the completed foreground")
    sessions = dsh_home / "sessions"
    candidates = [
        project / identity / "session.v4.jsonl.zstd"
        for project in sessions.iterdir()
        if project.is_dir() and (project / identity / "session.v4.jsonl.zstd").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError("DSH capture cannot identify exactly one owned native v4 artifact")
    transcript = candidates[0].resolve(strict=True)
    if not transcript.is_relative_to(sessions.resolve(strict=True)):
        raise ValueError("DSH transcript is outside the owned native session root")
    stop_path = profile_root / "native-stop.json"
    event = dict(
        hook_event_name="Stop",
        harness="dsh",
        session_id=identity,
        identity_kind="turn",
        turn_id=str(turn_id),
        cwd=str(workspace),
        transcript_path=str(transcript),
        event_source="observed_headless_turn_end",
    )
    write_json(stop_path, event)
    return {
        **event,
        "stop_event_path": str(stop_path),
        "history_read": False,
        "capture_backend_status": "not_configured",
        "measurement_limitations": [
            "artifact location binding does not validate transcript contents or run C6 distillation",
            "the foreground DSH summary endpoint does not survive headless exit",
        ],
    }
