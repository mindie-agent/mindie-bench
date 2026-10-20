"""Fail-closed source and runtime prerequisite judge for the Qwen3.5 GDN task.

[Role]
Resolve the candidate `vllm-ascend` workspace, verify that the Qwen3.5 GDN
source files exist in that workspace, and report the real model/DP/TP runtime
prerequisites required before any acceptance can be attempted.

[Invariants]
- The workspace is `MINDIE_BENCH_WORKSPACE` or the private-oracle cwd; the task
  tree is never used as the candidate source.
- A reward of 1 is impossible without a real model snapshot, the candidate
  source, and an executed DP=2/TP=2 asymmetric-batch workload. This verifier
  never reads caller-authored evidence JSON, never emits prefilled TTFT/TPOT,
  and never turns a missing model into a pass.
- Missing prerequisites are `unsupported:*` with reward 0 and remain visible.

[Timeouts & Locks]
- Pure filesystem/import checks; no subprocess starts an NPU workload.

[Side-Effect Policy]
- Read-only hashing and import discovery; never writes the workspace, starts a
  server, or contacts a remote host.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

TASK_ID = "qwen35-gdn-dp2-tp2-asymmetric-aclgraph"
SCHEMA = "mindie-bench-production-prerequisite-verifier/1"
REQUIRED_SOURCE = (
    "vllm_ascend/worker/model_runner_v1.py",
    "vllm_ascend/ops/gdn.py",
)
MODEL_ENV = ("MINDIE_BENCH_QWEN35_MODEL", "MINDIE_BENCH_MODEL_PATH")
REAL_COMMAND = (
    "vllm serve <Qwen3.5-GDN-snapshot> --host 127.0.0.1 --port <P> "
    "--tensor-parallel-size 2 --data-parallel-size 2 --trust-remote-code "
    "--enforce-eager; repeat without --enforce-eager for the ACLGraph arm"
)


class WorkspaceUnavailable(RuntimeError):
    pass


class SourceMissing(RuntimeError):
    pass


class Unsupported(RuntimeError):
    pass


def resolve_workspace() -> Path:
    raw = os.environ.get("MINDIE_BENCH_WORKSPACE")
    candidate = Path(raw) if raw else Path.cwd()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkspaceUnavailable(f"workspace_unavailable: {candidate}: {error}") from error
    if not (resolved / "vllm_ascend").is_dir() or not (resolved / "pyproject.toml").is_file():
        raise WorkspaceUnavailable(
            "workspace_unavailable: expected a vllm-ascend checkout under "
            f"{resolved}; set MINDIE_BENCH_WORKSPACE"
        )
    if (resolved / "task.toml").is_file() and (resolved / "verifier").is_dir():
        raise WorkspaceUnavailable(
            f"workspace_unavailable: {resolved} is the task/oracle tree, not the candidate checkout"
        )
    return resolved


def source_identity(workspace: Path) -> dict:
    files = {}
    missing = []
    for relative in REQUIRED_SOURCE:
        path = workspace / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing:
        raise SourceMissing("source_missing: " + ", ".join(missing))
    return {"workspace": str(workspace), "files": files}


def model_snapshot() -> tuple[str, Path] | None:
    for name in MODEL_ENV:
        raw = os.environ.get(name)
        if not raw:
            continue
        path = Path(raw).resolve()
        if not path.is_dir():
            raise Unsupported(f"unsupported:model_snapshot_missing: {name}={raw}")
        if not (path / "config.json").is_file():
            raise Unsupported(
                f"unsupported:model_snapshot_incomplete: {name}={raw} lacks config.json"
            )
        return name, path
    return None


def main() -> int:
    detail = {"schema": SCHEMA, "task_id": TASK_ID, "real_command": REAL_COMMAND}
    warnings = [
        "BLOCKED / incomplete: this verifier does not execute a model workload and has no production acceptance path; the DP=2/TP=2 Qwen3.5 GDN requirement is reported only."
    ]
    try:
        workspace = resolve_workspace()
        detail["source"] = source_identity(workspace)
        snapshot = model_snapshot()
        if snapshot is None:
            raise Unsupported(
                "unsupported:model_cache_unavailable: set MINDIE_BENCH_QWEN35_MODEL "
                "or MINDIE_BENCH_MODEL_PATH to a real Qwen3.5 GDN snapshot directory"
            )
        detail["model"] = {"env": snapshot[0], "path": str(snapshot[1])}
        if importlib.util.find_spec("vllm") is None:
            raise Unsupported("unsupported:vllm_runtime_unavailable: vllm is not importable")
        raise Unsupported(
            "unsupported:task_blocked_real_workload_harness_incomplete: candidate source and model are present, "
            "but the bounded DP=2/TP=2 asymmetric-batch serving harness has not been executed; "
            "run the documented vllm command under the parent A3 lease"
        )
    except SourceMissing as error:
        detail["error"] = str(error)
        payload = {"reward": 0.0, "failure_reason": str(error), "warnings": warnings}
        code = 1
    except Unsupported as error:
        detail["error"] = str(error)
        payload = {"reward": 0.0, "failure_reason": str(error), "warnings": warnings}
        code = 1
    except WorkspaceUnavailable as error:
        detail["error"] = str(error)
        payload = {"reward": 0.0, "failure_reason": str(error), "warnings": warnings}
        code = 1
    except Exception as error:
        message = f"verifier_exception: {type(error).__name__}: {error}"
        detail["error"] = message
        payload = {"reward": 0.0, "failure_reason": message, "warnings": warnings}
        code = 1
    print(json.dumps(detail, sort_keys=True, separators=(",", ":")), file=sys.stderr)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
