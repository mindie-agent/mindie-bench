# [Side-Effect Policy] Writes only explicitly owned trial artifacts; no network or unowned mutation.

"""Execute the bounded A3 dual-Die TP=2 MatmulAllReduce workload.

[Role]
Run the candidate placement guard inside each real HCCL rank launch, then run
the real `torch_npu.npu_mm_all_reduce_base` operator on the candidate sibling
pair. Run a clearly labelled explicit negative-control HCCL arm on a
cross-package pair under the same four-Die lease; that arm is never reported as
a candidate production path. Emit raw per-iteration samples only.

[Invariants]
- The candidate guard is loaded from `MINDIE_BENCH_WORKSPACE` in the child that
  is about to create the HCCL group. `enforce_visible_tp_placement()` runs
  before `torch`/`torch_npu`/HCCL initialization and its returned actual pair
  must equal the requested pair.
- The negative-control child also executes the guard before HCCL, but only to
  observe the expected rejection. It then runs an explicitly labelled
  diagnostic HCCL arm with `production_path=False`; a negative-control result
  can never be promoted to a candidate pass.
- Every measured iteration contributes to `relative_errors`; the receipt's
  `relative_error` is the maximum over all of them, not the last sample.
- A successful measurement requires an observed process-group destruction on
  every rank. Cleanup failure changes the rank status to `failed` and is
  propagated to the receipt; it cannot be hidden in a `finally` pass.
- No result is synthesized when the NPU runtime, HCCL group, operator, or
  candidate guard is unavailable; those are explicit `unsupported`/`failed`.

[Timeouts & Locks]
- One pair: `pair_timeout_sec` (210 s) enforced by the parent; HCCL connection
  gets its runtime-required 120 s, execution 60 s, plus drain grace. The verifier owns the
  outer four-Die device lease and no Python lock.

[Side-Effect Policy]
- Runs only inside the leased `/dev/davinci*` nodes. Writes config JSON to the
  private verifier temporary directory; leaves no files in the workspace and
  does not edit candidate source.
"""

from __future__ import annotations

import contextlib
import datetime
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "mindie-bench-tp2-allreduce-workload/1"
TASK_ID = "a3-dual-die-tp2-allreduce-placement"
CANDIDATE_ROLE = "candidate"
NEGATIVE_CONTROL_ROLE = "negative_control"


def _last_json(text: str) -> dict[str, Any] | None:
    for line in reversed([row.strip() for row in text.splitlines() if row.strip()]):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_guard(workspace: Path):
    path = workspace / "vllm_ascend/distributed/placement.py"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location("mindie_candidate_placement", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate guard: {path}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _cross_rejected(error: BaseException) -> bool:
    return (
        type(error).__name__ == "CrossPackagePlacementError"
        or "cross_package" in str(error).casefold()
        or "cross-package" in str(error).casefold()
    )


def enforce_pair_in_rank(
    workspace: Path,
    pair: list[int],
    role: str,
    *,
    production_rank: int | None = None,
    lease_devices: list[int] | None = None,
) -> dict[str, Any]:
    """Run the candidate guard in this rank before any torch/HCCL initialization."""
    if role not in {CANDIDATE_ROLE, NEGATIVE_CONTROL_ROLE}:
        raise ValueError(f"unknown workload role: {role!r}")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(value) for value in pair)
    result: dict[str, Any] = {
        "source_present": False,
        "role": role,
        "requested_phy_ids": list(pair),
        "observed": None,
        "actual_phy_ids": None,
        "error": None,
    }
    module = _load_guard(workspace)
    result["source_present"] = True
    try:
        enforced = module.enforce_visible_tp_placement()
    except Exception as error:
        if role == NEGATIVE_CONTROL_ROLE and _cross_rejected(error):
            result.update(
                observed="rejected",
                error=f"{type(error).__name__}: {error}",
            )
            return result
        raise RuntimeError(f"candidate_guard_failed:{role}:{type(error).__name__}: {error}") from error
    if role == NEGATIVE_CONTROL_ROLE:
        raise RuntimeError("negative_control_guard_accepted_cross_package")
    if not isinstance(enforced, dict):
        raise RuntimeError(f"candidate_guard_returned_{type(enforced).__name__}, expected dict")
    actual = enforced.get("device_phy_ids") or enforced.get("visible_phy_ids")
    if not isinstance(actual, (list, tuple)) or sorted(int(value) for value in actual) != sorted(pair):
        raise RuntimeError(
            f"candidate_guard_actual_pair_mismatch: requested={pair} returned={actual!r}"
        )
    if production_rank is not None and lease_devices is not None:
        validate_rank = getattr(module, "validate_rank_tp2_placement", None)
        if not callable(validate_rank):
            raise RuntimeError("candidate_guard_missing_validate_rank_tp2_placement")
        ranked = validate_rank(production_rank, list(lease_devices))
        ranked_pair = ranked.get("device_phy_ids") if isinstance(ranked, dict) else None
        if not isinstance(ranked_pair, (list, tuple)) or sorted(
            int(value) for value in ranked_pair
        ) != sorted(pair):
            raise RuntimeError(
                "candidate_guard_production_rank_mismatch: "
                f"rank={production_rank} requested={pair} returned={ranked_pair!r}"
            )
    result.update(
        observed="accepted",
        actual_phy_ids=[int(value) for value in actual],
        packages=enforced.get("packages"),
        sibling_pairs=enforced.get("sibling_pairs"),
    )
    return result


def _inputs(rank: int, world: int, m: int, k: int, n: int):
    import torch

    generator = torch.Generator().manual_seed(20260601)
    full = torch.randn((world, m, k), generator=generator).to(torch.bfloat16)
    x2 = torch.randn((k, n), generator=generator).to(torch.bfloat16)
    x1 = full[rank].contiguous()
    expected = full.float().sum(dim=0).matmul(x2.float())
    return x1, x2, expected


def run_child(config: dict[str, Any], rank: int) -> int:
    role = str(config["role"])
    pair = [int(value) for value in config["pair"]]
    output: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "nonce": config["nonce"],
        "arm": config["arm"],
        "role": role,
        "production_path": role == CANDIDATE_ROLE,
        "negative_control": role == NEGATIVE_CONTROL_ROLE,
        "rank": rank,
        "device_phy_ids": pair,
        "guard": None,
        "cleanup": {"required": False, "status": "not_applicable", "error": None},
    }
    torch_module = None
    group = False
    try:
        output["guard"] = enforce_pair_in_rank(
            Path(config["workspace"]),
            pair,
            role,
            production_rank=config.get("production_rank"),
            lease_devices=[int(value) for value in config.get("lease_devices", ())],
        )
        import torch
        import torch_npu

        torch_module = torch
        torch.npu.set_device(rank)
        torch.distributed.init_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:{int(config['master_port'])}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=190),
        )
        group = True
        from torch.distributed.distributed_c10d import _get_default_group

        default_group = _get_default_group()
        hcom = default_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        x1, x2, expected = _inputs(rank, 2, int(config["m"]), int(config["k"]), int(config["n"]))
        x1_npu = x1.to("npu")
        x2_npu = x2.to("npu")
        for _ in range(int(config["warmup"])):
            output_tensor = torch_npu.npu_mm_all_reduce_base(
                x1_npu, x2_npu, hcom, reduce_op="sum"
            )
            torch.npu.synchronize()
        if output_tensor is None:
            raise RuntimeError("npu_mm_all_reduce_base returned None")
        latencies: list[float] = []
        relative_errors: list[float] = []
        for _ in range(int(config["samples"])):
            torch.npu.synchronize()
            started = time.perf_counter_ns()
            output_tensor = torch_npu.npu_mm_all_reduce_base(
                x1_npu, x2_npu, hcom, reduce_op="sum"
            )
            torch.npu.synchronize()
            latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
            actual = output_tensor.cpu().float()
            denominator = max(float(expected.abs().max()), 1.0e-6)
            relative_errors.append(float((actual - expected).abs().max()) / denominator)
        output.update(
            status="measured",
            hcom=hcom,
            device=str(torch.npu.get_device_name(0)),
            torch_version=str(torch.__version__),
            torch_npu_version=str(torch_npu.__version__),
            latency_ms=latencies,
            relative_errors=relative_errors,
            relative_error=max(relative_errors),
        )
    except Exception as error:
        output.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        cleanup = {"required": group, "status": "not_applicable", "error": None}
        if group:
            try:
                if torch_module is None:
                    raise RuntimeError("torch module unavailable during process-group cleanup")
                torch_module.distributed.destroy_process_group()
                if torch_module.distributed.is_initialized():
                    raise RuntimeError("process group still initialized after destroy")
                cleanup.update(status="ok")
            except Exception as error:
                cleanup.update(status="failed", error=f"{type(error).__name__}: {error}")
        output["cleanup"] = cleanup
        if cleanup["status"] == "failed":
            if output.get("status") == "measured":
                output["status"] = "failed"
                output["error"] = f"cleanup_failed: {cleanup['error']}"
            else:
                output["error"] = f"{output.get('error')}; cleanup_failed: {cleanup['error']}"
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0 if output.get("status") == "measured" else 1


def combine_rank_receipts(
    outputs: list[dict[str, Any]], pair: list[int], role: str
) -> dict[str, Any]:
    """Combine two rank receipts without hiding cleanup, role or sample failures."""
    errors: list[str] = []
    expected_production = role == CANDIDATE_ROLE
    expected_negative = role == NEGATIVE_CONTROL_ROLE
    if len(outputs) != 2:
        errors.append(f"expected two rank receipts, received {len(outputs)}")
    for row in outputs:
        rank = row.get("rank")
        if row.get("role") != role:
            errors.append(f"rank={rank} role={row.get('role')!r}, expected {role!r}")
        if row.get("production_path") is not expected_production:
            errors.append(f"rank={rank} production_path mismatch")
        if row.get("negative_control") is not expected_negative:
            errors.append(f"rank={rank} negative_control mismatch")
        cleanup = row.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("status") != "ok":
            errors.append(f"rank={rank} cleanup not observed: {cleanup!r}")
        guard = row.get("guard")
        expected_observed = "accepted" if expected_production else "rejected"
        if not isinstance(guard, dict) or guard.get("observed") != expected_observed:
            errors.append(f"rank={rank} guard did not observe {expected_observed}")
        elif expected_production and sorted(guard.get("actual_phy_ids") or []) != sorted(pair):
            errors.append(f"rank={rank} guard actual pair mismatch")
        if row.get("status") != "measured":
            errors.append(f"rank={rank} status={row.get('status')!r}")
        if len(row.get("relative_errors", [])) != len(row.get("latency_ms", [])):
            errors.append(f"rank={rank} did not measure error for every iteration")
    if errors:
        return {
            "status": "failed",
            "role": role,
            "production_path": expected_production,
            "negative_control": expected_negative,
            "device_phy_ids": pair,
            "errors": errors,
            "rank_outputs": outputs,
        }
    first, second = outputs
    if len(first["latency_ms"]) != len(second["latency_ms"]):
        return {
            "status": "failed",
            "role": role,
            "production_path": expected_production,
            "negative_control": expected_negative,
            "device_phy_ids": pair,
            "errors": ["rank sample count mismatch"],
            "rank_outputs": outputs,
        }
    relative_errors = [
        float(value)
        for row in outputs
        for value in row.get("relative_errors", [])
    ]
    if not relative_errors:
        return {
            "status": "failed",
            "role": role,
            "production_path": expected_production,
            "negative_control": expected_negative,
            "device_phy_ids": pair,
            "errors": ["no measured relative-error samples"],
            "rank_outputs": outputs,
        }
    return {
        "status": "measured",
        "role": role,
        "production_path": expected_production,
        "negative_control": expected_negative,
        "device_phy_ids": pair,
        "latency_ms": [
            max(float(left), float(right))
            for left, right in zip(first["latency_ms"], second["latency_ms"], strict=True)
        ],
        "relative_errors": relative_errors,
        "relative_error": max(relative_errors),
        "guard": first.get("guard"),
        "cleanup": {
            "status": "ok",
            "rank_statuses": [row.get("cleanup", {}).get("status") for row in outputs],
        },
        "hcom": first.get("hcom"),
        "device": first.get("device"),
        "torch_version": first.get("torch_version"),
        "torch_npu_version": first.get("torch_npu_version"),
    }



def child_command(config_path: Path, rank: int) -> list[str]:
    """Build the real child argv; rank must be present for run_child dispatch."""
    return [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "--child",
        str(config_path),
        str(rank),
    ]


def run_pair(config: dict[str, Any], role: str) -> dict[str, Any]:
    pair_key = "candidate_pair" if role == CANDIDATE_ROLE else "negative_control_pair"
    pair = [int(value) for value in config[pair_key]]
    lease = [int(value) for value in config["lease_devices"]]
    production_rank = None
    if role == CANDIDATE_ROLE and pair[0] in lease:
        production_rank = lease.index(pair[0])
    timeout = float(config["pair_timeout_sec"])
    port = _free_port()
    children: list[subprocess.Popen] = []
    with tempfile.TemporaryDirectory(prefix=f"mindie-tp2-{role}-") as temporary:
        for rank in range(2):
            child_config = dict(
                config,
                arm=role,
                role=role,
                pair=pair,
                rank=rank,
                production_rank=production_rank,
                master_port=port,
            )
            path = Path(temporary) / f"rank-{rank}.json"
            path.write_text(json.dumps(child_config, sort_keys=True), encoding="utf-8")
            env = dict(os.environ)
            env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(value) for value in pair)
            # The private oracle runs with loopback only; a stale host
            # HCCL_IF_IP would not exist here. HCCL uses this interface for
            # process-group bootstrap while the fused op uses sibling HCCS.
            env.update(
                HCCL_IF_IP="127.0.0.1",
                HCCL_SOCKET_IFNAME="lo",
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                HCCL_CONNECT_TIMEOUT="120",
                HCCL_EXEC_TIMEOUT="60",
            )
            children.append(
                subprocess.Popen(
                    child_command(path, rank),
                    cwd=str(config["workspace"]),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        outputs: list[dict[str, Any]] = []
        process_errors: list[str] = []
        deadline = time.monotonic() + timeout
        for child in children:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                stdout, stderr = child.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                process_errors.append(f"rank timeout after {timeout}s")
                for process in children:
                    if process.poll() is None:
                        process.kill()
                for process in children:
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process_errors.append(f"rank cleanup unconfirmed: pid={process.pid}")
                return {
                    "status": "failed",
                    "role": role,
                    "production_path": role == CANDIDATE_ROLE,
                    "negative_control": role == NEGATIVE_CONTROL_ROLE,
                    "device_phy_ids": pair,
                    "errors": process_errors,
                    "rank_outputs": outputs,
                }
            parsed = _last_json(stdout)
            if parsed is not None:
                outputs.append(parsed)
            if child.returncode != 0:
                process_errors.append(
                    f"rank={parsed.get('rank') if parsed else '?'} exit={child.returncode}: "
                    + (stderr.strip() or (parsed or {}).get("error", ""))[-800:]
                )
    combined = combine_rank_receipts(outputs, pair, role)
    if process_errors:
        combined.setdefault("errors", []).extend(process_errors)
        if combined["status"] == "measured":
            combined["status"] = "failed"
    return combined


def _shm_state() -> dict[str, Any]:
    root = Path("/dev/shm")
    if not root.is_dir():
        return {"available": False, "residuals": []}
    return {
        "available": True,
        "residuals": sorted(path.name for path in root.glob("hccl*") if path.exists()),
    }


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "--child":
        config = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        return run_child(config, int(argv[2]))
    if len(argv) != 1:
        print(json.dumps({"status": "failed", "error": "usage: tp2_workload.py CONFIG"}))
        return 1
    config = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("task_id") != TASK_ID:
        print(json.dumps({"status": "failed", "error": "workload identity mismatch"}))
        return 1
    try:
        import importlib.util as _importlib

        if _importlib.find_spec("torch") is None or _importlib.find_spec("torch_npu") is None:
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "task_id": TASK_ID,
                        "nonce": config["nonce"],
                        "status": "unsupported",
                        "reason": "unsupported:npu_runtime_unavailable: torch/torch_npu is not importable",
                    }
                )
            )
            return 3
        workspace = Path(config["workspace"]).resolve(strict=True)
        lease = [int(value) for value in config["lease_devices"]]
        candidate_pair = [int(value) for value in config["candidate_pair"]]
        negative_pair = [int(value) for value in config["negative_control_pair"]]
        if len(set(lease)) != 4 or not (
            set(candidate_pair) | set(negative_pair)
        ).issubset(set(lease)):
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "task_id": TASK_ID,
                        "nonce": config["nonce"],
                        "status": "failed",
                        "error": "the explicit negative control requires the full four-Die lease and contained pairs",
                    }
                )
            )
            return 1
        before = _shm_state()
        if not before["available"] or before["residuals"]:
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "task_id": TASK_ID,
                        "nonce": config["nonce"],
                        "status": "failed",
                        "error": "residual_or_unavailable_dev_shm: " + ",".join(before["residuals"]),
                    }
                )
            )
            return 1
        candidate = run_pair(config, CANDIDATE_ROLE)
        negative = run_pair(config, NEGATIVE_CONTROL_ROLE)
        time.sleep(0.5)
        after = _shm_state()
        status = "measured"
        if candidate["status"] != "measured" or negative["status"] != "measured":
            status = "failed"
        receipt = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "nonce": config["nonce"],
            "status": status,
            "workspace": str(workspace),
            "lease_devices": lease,
            "candidate_run": {
                "pair": candidate_pair,
                "production_path": True,
            },
            "negative_control": {
                "label": "explicit_negative_control",
                "pair": negative_pair,
                "production_path": False,
                "inside_full_four_die_lease": True,
            },
            "measurements": {
                "candidate": candidate,
                "negative_control": negative,
            },
            "shm": {"available": after["available"], "residuals": after["residuals"]},
            "errors": [*candidate.get("errors", []), *negative.get("errors", [])],
        }
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0 if status == "measured" else 1
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "task_id": TASK_ID,
                    "nonce": config.get("nonce"),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
