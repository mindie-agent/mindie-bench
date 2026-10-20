"""Fail-closed source and runtime prerequisite judge for the two-node A3 canary.

[Role]
Resolve the candidate `vllm-ascend` workspace, verify the HCCL/interface source
files, and report the blocked two-node RoCE prerequisite. The standard L3
private oracle runs inside a `--unshare-net` slot, so it must not pretend that a
single-node sandbox is a two-node acceptance.

[Invariants]
- The candidate workspace is resolved from `MINDIE_BENCH_WORKSPACE` or the
  private-oracle cwd; the task tree is never accepted as source.
- A reward of 1 requires two reachable hosts, non-overlapping Dies, explicit
  HCCL_IF_IP/HCCL_SOCKET_IFNAME bindings on both ends, and an executed
  cross-node collective. This verifier emits no synthetic skew, timeout or
  cleanup evidence.
- A missing peer/network lease is `unsupported:*` with reward 0 and stays
  visible in the trial receipt.

[Timeouts & Locks]
- Pure filesystem/import checks; no SSH, no collective, no NPU workload.

[Side-Effect Policy]
- Read-only hashing and import discovery; never contacts .156/.165 or writes a
  remote host.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

TASK_ID = "a3-2node-roce-hccl-dp4-ep8-canary"
SCHEMA = "mindie-bench-production-prerequisite-verifier/1"
REQUIRED_SOURCE = (
    "vllm_ascend/distributed/parallel_state.py",
    "vllm_ascend/utils.py",
)
PEER_ENV = ("MINDIE_BENCH_PEER_HOST", "MINDIE_BENCH_HCCL_PEER")
REAL_COMMAND = (
    "external two-host coordinator: torchrun --nnodes=2 --nproc_per_node=8 "
    "--rdzv_endpoint=<host>:<port> <DP=4 x EP=8 alltoall canary> with "
    "HCCL_IF_IP/HCCL_SOCKET_IFNAME bound on both hosts"
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


def peer() -> str | None:
    for name in PEER_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


def main() -> int:
    detail = {"schema": SCHEMA, "task_id": TASK_ID, "real_command": REAL_COMMAND}
    warnings = [
        "BLOCKED / incomplete: the standard L3 private oracle is a single-host network namespace; there is no cross-node RoCE acceptance path in this verifier."
    ]
    try:
        workspace = resolve_workspace()
        detail["source"] = source_identity(workspace)
        remote_peer = peer()
        if remote_peer is None:
            raise Unsupported(
                "unsupported:multi_node_coordinator_required: no MINDIE_BENCH_PEER_HOST/"
                "MINDIE_BENCH_HCCL_PEER is configured for the second A3 host"
            )
        detail["peer"] = remote_peer
        if os.environ.get("MINDIE_BENCH_MULTINODE_DIRECT") != "1":
            raise Unsupported(
                "unsupported:multi_node_network_isolated: the production L3 slot uses bwrap "
                "--unshare-net, so cross-node HCCL cannot reach the peer; run this task under an "
                "external two-host coordinator with MINDIE_BENCH_MULTINODE_DIRECT=1"
            )
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
            raise Unsupported("unsupported:npu_runtime_unavailable: torch/torch_npu is not importable")
        if importlib.util.find_spec("vllm") is None:
            raise Unsupported("unsupported:vllm_runtime_unavailable: vllm is not importable")
        raise Unsupported(
            "unsupported:task_blocked_real_workload_harness_incomplete: candidate source and a direct multi-node "
            "lease are present, but the bounded cross-node alltoall/cleanup harness has not been executed"
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
