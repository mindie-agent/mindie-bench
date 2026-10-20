"""Fail-closed source and runtime prerequisite judge for the Qwen3-MoE task.

[Role]
Resolve the candidate `vllm-ascend` workspace, verify the MoE/EPLB source files
that implement MC2 Dispatch/Combine, and report the real 8-Die model/runtime
prerequisites before acceptance.

[Invariants]
- The candidate workspace is resolved from `MINDIE_BENCH_WORKSPACE` or the
  private-oracle cwd; the task tree is never accepted as source.
- A real run needs a Qwen3-MoE snapshot, eight leased Dies, `torch_npu`, and an
  executed TP=2/EP=4 or TP=2/DP=4 workload. Missing prerequisites stay
  `unsupported:*` with reward 0; no prefilled skew, latency, EPLB or workspace
  metric is emitted.
- No evidence JSON in the workspace is read or trusted.

[Timeouts & Locks]
- Pure filesystem/import checks; no NPU workload or server is started here.

[Side-Effect Policy]
- Read-only hashing and import discovery; never writes the workspace or remote
  hosts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

TASK_ID = "qwen3-moe-tp2-ep4-mc2-eplb-skew"
SCHEMA = "mindie-bench-production-prerequisite-verifier/1"
REQUIRED_SOURCE = (
    "vllm_ascend/ops/fused_moe/moe_comm_method.py",
    "vllm_ascend/eplb/core/eplb_worker.py",
)
MODEL_ENV = ("MINDIE_BENCH_QWEN3_MOE_MODEL", "MINDIE_BENCH_MODEL_PATH")
REAL_COMMAND = (
    "vllm serve <Qwen3-MoE-snapshot> --host 127.0.0.1 --port <P> "
    "--tensor-parallel-size 2 --data-parallel-size 4 --enable-expert-parallel "
    "--trust-remote-code --additional-config '{\"dynamic_eplb\": true}'"
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
        "BLOCKED / incomplete: this verifier does not execute a model workload and has no production acceptance path; the skewed-expert TP=2/EP=4 requirement is reported only."
    ]
    try:
        workspace = resolve_workspace()
        detail["source"] = source_identity(workspace)
        snapshot = model_snapshot()
        if snapshot is None:
            raise Unsupported(
                "unsupported:model_cache_unavailable: set MINDIE_BENCH_QWEN3_MOE_MODEL "
                "or MINDIE_BENCH_MODEL_PATH to a real Qwen3-MoE snapshot directory"
            )
        detail["model"] = {"env": snapshot[0], "path": str(snapshot[1])}
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if len([part for part in visible.split(",") if part.strip()]) < 8:
            raise Unsupported(
                "unsupported:eight_die_lease_required: ASCEND_RT_VISIBLE_DEVICES must expose eight leased Dies"
            )
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
            raise Unsupported("unsupported:npu_runtime_unavailable: torch/torch_npu is not importable")
        if importlib.util.find_spec("vllm") is None:
            raise Unsupported("unsupported:vllm_runtime_unavailable: vllm is not importable")
        raise Unsupported(
            "unsupported:task_blocked_real_workload_harness_incomplete: candidate source, model, eight-Die lease "
            "and runtime are present, but the bounded skewed-expert serving harness has not been executed"
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
