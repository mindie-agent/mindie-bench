"""Fail-closed source and runtime prerequisite judge for A3 PD disaggregation.

[Role]
Resolve the candidate `vllm-ascend` workspace, verify the Mooncake KV-transfer
source files, and report the real model, 16-Die split and Mooncake runtime
prerequisites before a PD handoff acceptance can be attempted.

[Invariants]
- The candidate workspace is resolved from `MINDIE_BENCH_WORKSPACE` or the
  private-oracle cwd; the task tree is never accepted as source.
- A reward of 1 needs a real model snapshot, a 16-Die lease split into disjoint
  Prefill/Decode visibility sets, an installed Mooncake engine, and an executed
  handoff. Missing prerequisites are `unsupported:*` with reward 0.
- No prefilled first-token, TTFT, TPOT, handoff or `/dev/shm` result is emitted;
  no workspace evidence JSON is read.

[Timeouts & Locks]
- Pure filesystem/import checks; no server, transfer or NPU workload starts.

[Side-Effect Policy]
- Read-only hashing and import discovery; never writes the workspace or a
  remote host.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

TASK_ID = "a3-16die-pd-disaggregated-mooncake-handoff"
SCHEMA = "mindie-bench-production-prerequisite-verifier/1"
REQUIRED_SOURCE = (
    "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py",
    "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py",
)
MODEL_ENV = ("MINDIE_BENCH_MODEL_PATH",)
REAL_COMMAND = (
    "vllm serve <model-snapshot> --host 127.0.0.1 --port <P> "
    "--tensor-parallel-size 8 --kv-transfer-config "
    "'{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_producer\",...}' "
    "(Prefill); the Decode instance uses kv_consumer on a disjoint "
    "ASCEND_RT_VISIBLE_DEVICES half"
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
        "BLOCKED / incomplete: this verifier does not execute a PD handoff and has no production acceptance path; the two-instance Mooncake requirement is reported only."
    ]
    try:
        workspace = resolve_workspace()
        detail["source"] = source_identity(workspace)
        snapshot = model_snapshot()
        if snapshot is None:
            raise Unsupported(
                "unsupported:model_cache_unavailable: set MINDIE_BENCH_MODEL_PATH to a real model snapshot directory"
            )
        detail["model"] = {"env": snapshot[0], "path": str(snapshot[1])}
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if len([part for part in visible.split(",") if part.strip()]) < 16:
            raise Unsupported(
                "unsupported:sixteen_die_lease_required: ASCEND_RT_VISIBLE_DEVICES must expose the full 16-Die node"
            )
        if os.environ.get("MINDIE_BENCH_MOONCAKE_READY") != "1":
            raise Unsupported(
                "unsupported:mooncake_runtime_unavailable: set MINDIE_BENCH_MOONCAKE_READY=1 only after "
                "the Mooncake transfer engine and disjoint Prefill/Decode endpoints are provisioned"
            )
        if importlib.util.find_spec("mooncake") is None:
            raise Unsupported("unsupported:mooncake_engine_unavailable: mooncake is not importable")
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
            raise Unsupported("unsupported:npu_runtime_unavailable: torch/torch_npu is not importable")
        if importlib.util.find_spec("vllm") is None:
            raise Unsupported("unsupported:vllm_runtime_unavailable: vllm is not importable")
        raise Unsupported(
            "unsupported:task_blocked_real_workload_harness_incomplete: candidate source, model, 16-Die split and "
            "Mooncake runtime are present, but the bounded handoff harness has not been executed"
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
