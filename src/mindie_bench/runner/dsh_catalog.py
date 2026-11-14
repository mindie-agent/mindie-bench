"""[Role] Inspect the mounted DSH catalog before foreground model execution.
[Invariants] Catalog identity comes from the trial's native checkpoint, never
planned rows. P0 rejects mounted MindIE tools, skills or loader identities.
[Timeouts & Locks] The caller bounds the checkpoint wait (default 120s); owned
process-group cleanup has separate TERM 2s and KILL 5s waits. No locks.
[Side-Effect Policy] Writes only checkpoint evidence and stdout/stderr inside
the trial profile; stops its owned probe group and makes no model request.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from mindie_bench.runner.host_platform import subprocess_window_kwargs
from mindie_bench.runner.termination import terminate_process_group


def _catalog_identity(payload: dict) -> dict:
    tools = sorted(
        (str(item.get("name")), str(item.get("description") or ""))
        for item in payload.get("tools") or []
    )
    skills = sorted(
        (str(item.get("name")), str(item.get("provider") or ""))
        for item in payload.get("skills") or []
    )
    rows = sorted(
        (str(item.get("id") or ""), str(item.get("name") or ""))
        for item in payload.get("loader_rows") or []
    )
    mindie_tools = sorted(
        name
        for name, _ in tools
        if name == "mindie_entry" or name.startswith(("knowledge_", "remote_"))
    )
    mindie_skills = sorted(
        name for name, provider in skills if provider == "mindie" or name == "mindie-agent"
    )
    mindie_rows = [
        {"id": row_id, "name": name}
        for row_id, name in rows
        if "mindie-agent" in name.lower() or "mindie-agent" in row_id.lower()
    ]
    canonical = json.dumps(
        {"tools": tools, "skills": skills, "rows": rows}, sort_keys=True, separators=(",", ":")
    )
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "tool_count": len(tools),
        "tool_names": [name for name, _ in tools],
        "skill_count": len(skills),
        "skill_names": [name for name, _ in skills],
        "loader_row_count": len(rows),
        "loader_rows": [{"id": row_id, "name": name} for row_id, name in rows],
        "checkpoint_row_present": any(row_id == "bench-catalog-checkpoint" for row_id, _ in rows),
        "mindie_identities": {
            "tools": mindie_tools,
            "skills": mindie_skills,
            "loader_rows": mindie_rows,
        },
    }


def assert_catalog_policy(identity: dict, product_group: str | None) -> None:
    """Fail closed on a missing checkpoint row or a MindIE row in P0."""
    if not identity.get("checkpoint_row_present"):
        raise ValueError("mounted product catalog is missing the accounted checkpoint row")
    if product_group == "P0_bare_host":
        forbidden = identity.get("mindie_identities") or {}
        if any(forbidden.get(name) for name in ("tools", "skills", "loader_rows")):
            raise ValueError("P0 mounted catalog contains a MindIE identity")


def _stop_probe_group(process: subprocess.Popen, *, reason: str) -> None:
    """Stop the whole probe process group, never only its leader."""
    try:
        terminate_process_group(process, term_wait=2, kill_wait=5)
    except TimeoutError as exc:
        raise RuntimeError(f"catalog probe process group did not stop after {reason}") from exc


def probe_dsh_catalog(
    profile, trial_dir: Path, workspace: Path, command: list[str], *, timeout: float = 120.0
) -> dict:
    """Run the mounted catalog checkpoint through the trial's own sandbox."""
    if not profile.env.get("MINDIE_BENCH_CATALOG_CHECKPOINT_OUT"):
        raise ValueError("this profile has no accounted catalog checkpoint row")
    output = Path(profile.root) / "catalog-checkpoint.probe.json"
    if output.exists():
        output.unlink()
    env = dict(
        profile.env,
        MINDIE_BENCH_CATALOG_CHECKPOINT_OUT=str(output),
        MINDIE_BENCH_CATALOG_CHECKPOINT_EXIT="1",
    )
    stdout_path = trial_dir / "preflight-catalog.stdout"
    stderr_path = trial_dir / "preflight-catalog.stderr"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
            encoding="utf-8",
            **subprocess_window_kwargs(),
        )
        try:
            if process.stdin:
                process.stdin.write("catalog checkpoint\n")
                process.stdin.close()
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_probe_group(process, reason="timeout")
                raise RuntimeError("mounted catalog checkpoint timed out before writing")
            _stop_probe_group(process, reason="completion")
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
    if returncode != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"mounted catalog checkpoint exited {returncode}: {detail}")
    if not output.is_file():
        raise RuntimeError("mounted catalog checkpoint wrote no identity")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("schema") != "dsh-live-catalog-checkpoint/2":
        raise ValueError("mounted catalog checkpoint has an unknown schema")
    for error_field in ("tools_error", "skills_error", "loader_error"):
        if payload.get(error_field):
            raise ValueError(
                f"mounted catalog checkpoint failed {error_field}: {payload[error_field]}"
            )
    identity = _catalog_identity(payload)
    return {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "elapsed_sec": time.monotonic() - started,
        "identity": identity,
    }
