"""Judge a real A3 dual-Die TP=2 MatmulAllReduce placement run.

[Role]
Resolve the candidate `vllm-ascend` workspace (never the task fixture), probe the
real leased A3 Dies with read-only `npu-smi`, execute one candidate two-rank
HCCL arm and one explicitly labelled cross-package negative control, and
recompute every gate from the workload's raw per-iteration samples.

[Invariants]
- `MINDIE_BENCH_WORKSPACE` or the private-oracle cwd must be a real
  `vllm-ascend` checkout; falling back to the task tree is forbidden.
- The unpatched source must fail before any workload starts because
  `vllm_ascend/distributed/placement.py` is absent. The gold patch must add the
  guard that the workload actually imports.
- A reward of 1 requires real measured samples from the candidate sibling-Die
  pair and an explicitly labelled cross-package negative control, the candidate
  guard executing inside each candidate rank before HCCL, zero
  `/dev/shm/hccl*` residue, and observed cleanup on every measured rank.
- Missing NPU/runtime/topology prerequisites are reported as `unsupported:*`
  with reward 0. A failed or partial run is never converted into a pass.

[Timeouts & Locks]
- `npu-smi` probe: 15 s. Workload: 300 s outer, with a 90 s per-pair child
  deadline and a 60 s HCCL process-group timeout. The verifier owns no NPU lock;
  the remote slot lease owns the device flock.

[Side-Effect Policy]
- Read-only `npu-smi`; subprocesses run only inside the leased device nodes.
  Writes are confined to a `TemporaryDirectory` for workload config/children.
  Never edits the candidate workspace or a task directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TASK_ID = "a3-dual-die-tp2-allreduce-placement"
VERIFIER_SCHEMA = "mindie-bench-production-topology-verifier/2"
WORKLOAD_SCHEMA = "mindie-bench-tp2-allreduce-workload/1"
REQUIRED_SOURCE = (
    "vllm_ascend/distributed/parallel_state.py",
    "vllm_ascend/distributed/placement.py",
)
MAX_WORKLOAD_BYTES = 1024 * 1024
WORKLOAD_TIMEOUT_SEC = 300.0
MIN_SAMPLES = 8
MAX_RELATIVE_ERROR = 2.0e-2
MAX_CANDIDATE_TO_CONTROL_RATIO = 2.0

_CHIP_WITH_PHY = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+(?:\.\d+)?)\s+"
    r"(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
_PARENT = re.compile(r"^\|\s*(\d+)\s+([A-Za-z0-9][^|]*?)\s*\|\s*([A-Za-z]+)\s*\|")
_PROCESS = re.compile(r"^\|\s*(\d+)(?:\s+(\d+))?\s*\|\s*(\d+)\s*\|")
_NO_PROCESS = re.compile(r"No running processes found in NPU\s+(\d+)")


class WorkspaceUnavailable(RuntimeError):
    """The caller did not provide a real candidate checkout."""


class SourceMissing(RuntimeError):
    """The candidate checkout lacks a file the task must repair."""


class Unsupported(RuntimeError):
    """A real runtime prerequisite is absent; no acceptance is possible."""


class WorkloadFailed(RuntimeError):
    """The workload executed but failed or reported an invalid receipt."""


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _parse_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{field} must contain nonnegative integers")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def resolve_workspace(environ: dict[str, str] | None = None) -> Path:
    """Resolve the candidate checkout; never fall back to the oracle/task root."""
    env = os.environ if environ is None else environ
    raw = env.get("MINDIE_BENCH_WORKSPACE")
    candidate = Path(raw) if raw else Path.cwd()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkspaceUnavailable(
            f"workspace_unavailable: cannot resolve candidate workspace {candidate}: {error}"
        ) from error
    if not (resolved / "vllm_ascend").is_dir() or not (resolved / "pyproject.toml").is_file():
        raise WorkspaceUnavailable(
            "workspace_unavailable: expected a vllm-ascend checkout with "
            f"vllm_ascend/ and pyproject.toml under {resolved}; set MINDIE_BENCH_WORKSPACE"
        )
    if (resolved / "task.toml").is_file() and (resolved / "verifier").is_dir():
        raise WorkspaceUnavailable(
            f"workspace_unavailable: {resolved} is the task/oracle tree, not the candidate checkout"
        )
    return resolved


def source_identity(workspace: Path) -> dict[str, Any]:
    """Hash the candidate files consumed by this task and fail if a guard is absent."""
    files: dict[str, str] = {}
    missing = []
    for relative in REQUIRED_SOURCE:
        path = workspace / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing:
        raise SourceMissing(
            "source_missing: candidate workspace lacks " + ", ".join(missing)
        )
    parallel_state = workspace / "vllm_ascend/distributed/parallel_state.py"
    try:
        parallel_state_text = parallel_state.read_text(encoding="utf-8")
        tree = ast.parse(parallel_state_text)
    except (OSError, SyntaxError) as error:
        raise SourceMissing(
            f"source_missing: cannot parse {parallel_state}: {type(error).__name__}: {error}"
        ) from error
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected = {"init_ascend_model_parallel", "get_mc2_group"}
    if not expected.issubset(functions):
        raise SourceMissing(
            "source_missing: parallel_state.py lacks " + ", ".join(sorted(expected - functions))
        )
    if "validate_rank_tp2_placement" not in parallel_state_text:
        raise SourceMissing(
            "source_missing: parallel_state.py does not integrate validate_rank_tp2_placement"
        )
    return {
        "workspace": str(workspace),
        "files": files,
        "candidate_guard_sha256": files["vllm_ascend/distributed/placement.py"],
    }


def parse_npu_smi(text: str) -> dict[int, dict[str, Any]]:
    """Parse the A3 `Chip Phy-ID` table and the process table (read-only)."""
    dies: dict[int, dict[str, Any]] = {}
    current_npu = 0
    parent_health = "OK"
    saw_parent = False
    for line in text.splitlines():
        row = _CHIP_WITH_PHY.match(line)
        if row:
            chip_id, phy_id = int(row.group(1)), int(row.group(2))
            npu_id = current_npu if saw_parent else chip_id
            dies[phy_id] = {
                "phy_id": phy_id,
                "npu_id": npu_id,
                "chip_id": chip_id,
                "health": parent_health,
                "busy": parent_health != "OK",
                "busy_reason": None if parent_health == "OK" else f"health:{parent_health}",
            }
            continue
        parent = _PARENT.match(line)
        if parent:
            saw_parent = True
            current_npu = int(parent.group(1))
            parent_health = parent.group(3)
    process_header = next(
        (
            index
            for index, line in enumerate(text.splitlines())
            if "Process id" in line and "Process name" in line
        ),
        None,
    )
    busy_npus: set[int] = set()
    if process_header is not None:
        for line in text.splitlines()[process_header + 1 :]:
            match = _NO_PROCESS.search(line)
            if match:
                continue
            row = _PROCESS.match(line)
            if row:
                busy_npus.add(int(row.group(1)))
    for die in dies.values():
        if die["npu_id"] in busy_npus and not die["busy"]:
            die["busy"] = True
            die["busy_reason"] = "npu-process"
    return dies


def probe_topology() -> dict[int, dict[str, Any]]:
    executable = shutil.which("npu-smi")
    if executable is None:
        raise Unsupported("unsupported:npu_smi_unavailable: npu-smi is not installed")
    try:
        completed = subprocess.run(
            [executable, "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Unsupported(f"unsupported:npu_smi_unavailable: {error}") from error
    if completed.returncode != 0:
        raise Unsupported(
            "unsupported:npu_smi_unavailable: "
            + (completed.stderr.strip() or completed.stdout.strip())[-400:]
        )
    dies = parse_npu_smi(completed.stdout)
    if not dies:
        raise Unsupported("unsupported:npu_smi_no_chip_rows")
    return dies


def visible_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not raw.strip():
        raise Unsupported(
            "unsupported:visible_devices_unset: the remote slot must export "
            "ASCEND_RT_VISIBLE_DEVICES for the leased A3 Dies"
        )
    try:
        values = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise Unsupported(f"unsupported:visible_devices_invalid: {raw!r}") from error
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise Unsupported(f"unsupported:visible_devices_invalid: {raw!r}")
    missing = [value for value in values if not Path(f"/dev/davinci{value}").exists()]
    if missing:
        raise Unsupported(
            "unsupported:device_node_unavailable: /dev/davinci"
            + ",/dev/davinci".join(str(value) for value in missing)
        )
    return values


def select_pairs(
    visible: list[int], dies: dict[int, dict[str, Any]]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select the candidate sibling pair and the explicit negative-control pair."""
    if len(visible) < 4:
        raise Unsupported(
            "unsupported:cross_package_control_requires_4_dies: "
            f"ASCEND_RT_VISIBLE_DEVICES exposes {len(visible)} Dies"
        )
    by_package: dict[int, list[int]] = {}
    for phy in visible:
        die = dies.get(phy)
        if die is None:
            raise Unsupported(f"unsupported:visible_die_not_in_npu_smi:{phy}")
        if die["busy"] or die["health"] != "OK":
            raise Unsupported(
                f"unsupported:leased_die_busy:{phy}:{die.get('busy_reason') or die['health']}"
            )
        by_package.setdefault(phy // 2, []).append(phy)
    sibling: tuple[int, int] | None = None
    sibling_package: int | None = None
    for package in sorted(by_package):
        ids = sorted(by_package[package])
        if len(ids) >= 2:
            sibling = (ids[0], ids[1])
            sibling_package = package
            break
    if sibling is None or sibling_package is None:
        raise Unsupported(
            "unsupported:sibling_pair_unavailable: visible Dies expose no full A3 package"
        )
    candidate_slot = visible.index(sibling[0])
    if (
        candidate_slot % 2
        or candidate_slot + 1 >= len(visible)
        or visible[candidate_slot + 1] != sibling[1]
    ):
        raise Unsupported(
            "unsupported:lease_visible_order_not_tp2_aligned: the selected sibling pair "
            "must occupy consecutive even two-Die TP slots in ASCEND_RT_VISIBLE_DEVICES"
        )
    # The explicit negative control deliberately reuses one sibling Die (for
    # example (0,1) vs (1,2)). The two runs are sequential, so this reproduces
    # the production straddling placement without requiring a third package.
    cross: tuple[int, int] | None = None
    for package in sorted(by_package):
        if package == sibling_package:
            continue
        for phy in sorted(by_package[package]):
            if phy not in sibling:
                cross = (sibling[1], phy)
                break
        if cross is not None:
            break
    if cross is None:
        raise Unsupported(
            "unsupported:cross_package_control_unavailable: no second package is visible"
        )
    return sibling, cross


def shm_entries() -> list[str]:
    root = Path("/dev/shm")
    if not root.is_dir():
        raise Unsupported("unsupported:dev_shm_unavailable: /dev/shm is not mounted")
    return sorted(path.name for path in root.glob("hccl*") if path.exists())


def _read_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed([row.strip() for row in stdout.splitlines() if row.strip()]):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise WorkloadFailed("workload produced no JSON receipt")


def run_workload(
    workspace: Path,
    lease_devices: tuple[int, ...],
    candidate_pair: tuple[int, int],
    negative_control_pair: tuple[int, int],
    *,
    timeout: float = WORKLOAD_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Execute the frozen workload against the candidate workspace and collect raw JSON."""
    workload = Path(__file__).resolve().with_name("tp2_workload.py")
    if not workload.is_file():
        raise WorkloadFailed(f"workload_missing:{workload}")
    nonce = secrets.token_hex(24)
    config = {
        "schema": WORKLOAD_SCHEMA,
        "task_id": TASK_ID,
        "nonce": nonce,
        "workspace": str(workspace),
        "lease_devices": list(lease_devices),
        "candidate_pair": list(candidate_pair),
        "negative_control_pair": list(negative_control_pair),
        "m": 256,
        "k": 256,
        "n": 256,
        "warmup": 3,
        "samples": 12,
        "pair_timeout_sec": 90,
    }
    with tempfile.TemporaryDirectory(prefix="mindie-tp2-") as temporary:
        config_path = Path(temporary) / "workload.json"
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        env = dict(
            os.environ,
            MINDIE_BENCH_WORKSPACE=str(workspace),
            PYTHONDONTWRITEBYTECODE="1",
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(workload), str(config_path)],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise WorkloadFailed(f"workload timeout after {timeout}s") from error
    if completed.stderr:
        sys.stderr.write(completed.stderr[-8000:])
    receipt = _read_last_json(completed.stdout)
    if receipt.get("schema") != WORKLOAD_SCHEMA:
        raise WorkloadFailed("workload schema mismatch")
    if receipt.get("task_id") != TASK_ID:
        raise WorkloadFailed("workload task identity mismatch")
    if receipt.get("nonce") != nonce:
        raise WorkloadFailed("workload nonce mismatch")
    recorded_workspace = receipt.get("workspace")
    if recorded_workspace and Path(str(recorded_workspace)).resolve() != workspace.resolve():
        raise WorkloadFailed("workload workspace identity mismatch")
    if completed.returncode == 3 or receipt.get("status") == "unsupported":
        raise Unsupported(str(receipt.get("reason") or "unsupported:workload"))
    if completed.returncode != 0 or receipt.get("status") != "measured":
        detail = receipt.get("error") or f"exit={completed.returncode}"
        raise WorkloadFailed(f"workload_failed: {detail}")
    receipt["_stdout_sha256"] = hashlib.sha256(completed.stdout.encode()).hexdigest()
    return receipt


def evaluate_receipt(
    receipt: dict[str, Any],
    lease_devices: tuple[int, ...],
    candidate_pair: tuple[int, int],
    negative_control_pair: tuple[int, int],
) -> dict[str, Any]:
    """Recompute all gates from raw samples; never trust a declared aggregate."""
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    lease = receipt.get("lease_devices")
    if not isinstance(lease, list) or sorted(lease) != sorted(lease_devices):
        failures.append("lease_devices_mismatch")
    expected_pairs = {
        "candidate": list(candidate_pair),
        "negative_control": list(negative_control_pair),
    }
    candidate_run = receipt.get("candidate_run")
    if not isinstance(candidate_run, dict) or candidate_run.get("production_path") is not True:
        failures.append("missing_candidate_production_path")
    elif candidate_run.get("pair") != list(candidate_pair):
        failures.append("candidate_pair_mismatch")
    negative = receipt.get("negative_control")
    if not isinstance(negative, dict):
        failures.append("missing_negative_control_label")
    else:
        if negative.get("label") != "explicit_negative_control":
            failures.append("negative_control_not_explicitly_labelled")
        if negative.get("production_path") is not False:
            failures.append("negative_control_mislabeled_as_production")
        if negative.get("pair") != list(negative_control_pair):
            failures.append("negative_control_pair_mismatch")
        if negative.get("inside_full_four_die_lease") is not True:
            failures.append("negative_control_not_inside_four_die_lease")
    latencies: dict[str, list[float]] = {}
    for arm in ("candidate", "negative_control"):
        row = receipt.get("measurements", {}).get(arm)
        if not isinstance(row, dict):
            failures.append(f"missing_measurement:{arm}")
            continue
        if row.get("role") != arm:
            failures.append(f"measurement_role_mismatch:{arm}")
        if row.get("device_phy_ids") != expected_pairs[arm]:
            failures.append(f"measurement_device_mismatch:{arm}")
        if arm == "candidate":
            if row.get("production_path") is not True:
                failures.append("candidate_measurement_not_production_path")
            if row.get("negative_control") is not False:
                failures.append("candidate_measurement_mislabeled_negative_control")
        else:
            if row.get("production_path") is not False:
                failures.append("negative_control_mislabeled_as_production")
            if row.get("negative_control") is not True:
                failures.append("negative_control_measurement_not_labelled")
        cleanup = row.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("status") != "ok":
            failures.append(f"cleanup_not_observed:{arm}:{cleanup!r}")
        guard = row.get("guard")
        expected_observed = "accepted" if arm == "candidate" else "rejected"
        if not isinstance(guard, dict) or guard.get("observed") != expected_observed:
            failures.append(f"guard_not_observed:{arm}:{expected_observed}")
        elif arm == "candidate" and sorted(guard.get("actual_phy_ids") or []) != sorted(
            candidate_pair
        ):
            failures.append("candidate_guard_actual_pair_mismatch")
        samples = row.get("latency_ms")
        if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
            failures.append(f"insufficient_samples:{arm}")
            continue
        if not all(_finite_positive(value) for value in samples):
            failures.append(f"invalid_latency_sample:{arm}")
            continue
        latencies[arm] = [float(value) for value in samples]
        relative_errors = row.get("relative_errors")
        if not isinstance(relative_errors, list) or len(relative_errors) < MIN_SAMPLES:
            failures.append(f"missing_all_iteration_errors:{arm}")
            continue
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in relative_errors
        ):
            failures.append(f"invalid_relative_error_sample:{arm}")
            continue
        max_error = max(float(value) for value in relative_errors)
        declared_error = row.get("relative_error")
        if (
            not isinstance(declared_error, (int, float))
            or isinstance(declared_error, bool)
            or not math.isfinite(float(declared_error))
            or abs(float(declared_error) - max_error) > 1.0e-12
        ):
            failures.append(f"relative_error_not_max_of_all_iterations:{arm}")
        if max_error > MAX_RELATIVE_ERROR:
            failures.append(f"relative_error_exceeded:{arm}:{max_error}")
        metrics[f"{arm}_median_ms"] = statistics.median(latencies[arm])
        metrics[f"{arm}_samples"] = len(latencies[arm])
        metrics[f"{arm}_error_samples"] = len(relative_errors)
        metrics[f"{arm}_max_relative_error"] = max_error
    if "candidate" in latencies and "negative_control" in latencies:
        ratio = metrics["candidate_median_ms"] / metrics["negative_control_median_ms"]
        metrics["candidate_to_negative_control_latency_ratio"] = ratio
        if ratio > MAX_CANDIDATE_TO_CONTROL_RATIO:
            failures.append(f"candidate_latency_degraded:{ratio:.3f}")
    else:
        metrics["candidate_to_negative_control_latency_ratio"] = None
    shm = receipt.get("shm")
    if not isinstance(shm, dict) or shm.get("available") is not True:
        failures.append("shm_measurement_unavailable")
    elif shm.get("residuals"):
        failures.append("residual_hccl_shm:" + ",".join(shm["residuals"]))
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "samples": latencies,
    }


def _summary(reward: float, failure_reason: str | None, warnings: list[str]) -> dict[str, Any]:
    return {
        "reward": float(reward),
        "failure_reason": failure_reason,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments not in ([], ["--self-test"]):
        print(json.dumps({"error": "usage: verify.py [--self-test]"}, sort_keys=True))
        return 2
    detail: dict[str, Any] = {"schema": VERIFIER_SCHEMA, "task_id": TASK_ID}
    warnings: list[str] = [
        "This verifier executes one bounded candidate two-rank HCCL workload and one explicitly labelled cross-package negative control; it does not establish model-serving acceptance.",
        "The cross-package arm is diagnostic only and is never a candidate production path.",
    ]
    try:
        workspace = resolve_workspace()
        identity = source_identity(workspace)
        detail["source"] = identity
        dies = probe_topology()
        visible = visible_devices()
        candidate_pair, negative_control_pair = select_pairs(visible, dies)
        detail["topology"] = {
            "visible_devices": visible,
            "candidate_pair": list(candidate_pair),
            "negative_control_pair": list(negative_control_pair),
            "negative_control_label": "explicit_negative_control",
            "dies": [dies[phy] for phy in visible],
        }
        before = shm_entries()
        if before:
            raise WorkloadFailed(
                "residual_hccl_shm_before_run:" + ",".join(before)
            )
        receipt = run_workload(
            workspace,
            tuple(visible),
            candidate_pair,
            negative_control_pair,
        )
        detail["workload"] = receipt
        evaluation = evaluate_receipt(
            receipt,
            tuple(visible),
            candidate_pair,
            negative_control_pair,
        )
        detail["evaluation"] = evaluation
        if not evaluation["passed"]:
            raise WorkloadFailed("gate_failed: " + "; ".join(evaluation["failures"]))
        after = shm_entries()
        if after:
            raise WorkloadFailed("residual_hccl_shm_after_run:" + ",".join(after))
        detail["shm_after"] = after
        payload = _summary(1.0, None, warnings)
        code = 0
    except SourceMissing as error:
        detail["error"] = {"type": type(error).__name__, "message": str(error)}
        payload = _summary(0.0, str(error), warnings)
        code = 1
    except Unsupported as error:
        detail["error"] = {"type": type(error).__name__, "message": str(error)}
        payload = _summary(0.0, str(error), warnings)
        code = 1
    except (WorkspaceUnavailable, WorkloadFailed) as error:
        detail["error"] = {"type": type(error).__name__, "message": str(error)}
        payload = _summary(0.0, str(error), warnings)
        code = 1
    except Exception as error:  # visible infrastructure failure, never a pass
        detail["error"] = {"type": type(error).__name__, "message": str(error)}
        payload = _summary(0.0, f"verifier_exception: {type(error).__name__}: {error}", warnings)
        code = 1
    print(json.dumps(detail, sort_keys=True, separators=(",", ":")), file=sys.stderr)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
