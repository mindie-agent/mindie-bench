"""Fail-closed production multi-Die verifier contracts.

These tests exercise the verifier protocol and negative paths only: workspace
resolution, candidate-source identity, gate recomputation from raw samples,
private-oracle bundling, and the gold guard. They do **not** run an NPU
workload and cannot establish hardware acceptance. The parent reviewer owns
the live A3 execution of the dual-Die task.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks/production-multi-die"
DUAL = TASKS_ROOT / "a3-dual-die-tp2-allreduce-placement"
SOURCE_REPO = REPO_ROOT / "repositories/vllm-ascend-idle-dp"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifier_modules() -> list[Path]:
    return sorted(TASKS_ROOT.glob("*/verifier/*.py"))


def _fake_bin(tmp_path: Path) -> Path:
    """PATH with a python3 link but a deliberately failing npu-smi."""
    directory = tmp_path / "fake-bin"
    directory.mkdir()
    (directory / "python3").symlink_to(Path(sys.executable).resolve())
    npu_smi = directory / "npu-smi"
    npu_smi.write_text("#!/bin/sh\necho 'protocol test: npu-smi disabled' >&2\nexit 9\n")
    npu_smi.chmod(0o755)
    return directory


def _real_source(relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(SOURCE_REPO), "show", f"HEAD:{relative}"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return completed.stdout


def _workspace(tmp_path: Path, required: tuple[str, ...]) -> Path:
    workspace = tmp_path / "workspace"
    for relative in required:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("parallel_state.py") and SOURCE_REPO.is_dir():
            content = _real_source(relative)
        else:
            content = "# protocol-test candidate source\n"
        path.write_text(content)
    (workspace / "pyproject.toml").write_text("[project]\nname = 'vllm-ascend'\n")
    return workspace


def _run_verifier(task: Path, workspace: Path | None, fake_bin: Path) -> tuple[subprocess.CompletedProcess, dict]:
    env = dict(os.environ)
    env["PATH"] = str(fake_bin)
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    if workspace is not None:
        env["MINDIE_BENCH_WORKSPACE"] = str(workspace)
    else:
        env.pop("MINDIE_BENCH_WORKSPACE", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(task / "verifier/verify.py")],
        cwd=task,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, (task.name, completed.stdout, completed.stderr)
    return completed, json.loads(lines[-1])


def _apply_dual_patch(workspace: Path) -> None:
    subprocess.run(
        ["git", "apply", "--no-index", str(DUAL / "oracle/gold_patch.diff")],
        cwd=workspace,
        check=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.tier1
def test_all_five_verifiers_never_read_the_task_fixture_or_prefilled_success(tmp_path):
    forbidden = (
        "topology_case",
        "BROKEN",
        "FIXED",
        "after_reward",
        "before_reward",
        '"kl_divergence": 0.0001',
        '"top1_agreement": 0.9999',
        '"ttft_p50_ms": 120.0',
        '"goodput_tokens_per_sec": 500.0',
    )
    for path in sorted(item for item in TASKS_ROOT.iterdir() if item.is_dir()):
        assert not (path / "environment/topology_case.py").exists(), path.name
        assert not (path / "soundness_receipt.json").exists(), path.name
        assert not (path / "MANIFEST.sha256").exists(), path.name
        for verifier in path.glob("verifier/*.py"):
            text = verifier.read_text()
            for token in forbidden:
                assert token not in text, (path.name, verifier.name, token)
        assert any(
            "MINDIE_BENCH_WORKSPACE" in verifier.read_text()
            for verifier in path.glob("verifier/*.py")
        ), path.name


@pytest.mark.tier1
def test_verifiers_fail_closed_without_a_candidate_workspace(tmp_path):
    fake_bin = _fake_bin(tmp_path)
    for path in sorted(item for item in TASKS_ROOT.iterdir() if item.is_dir()):
        completed, payload = _run_verifier(path, None, fake_bin)
        assert completed.returncode == 1, (path.name, completed.stdout, completed.stderr)
        assert payload["reward"] == 0.0
        assert "workspace_unavailable" in payload["failure_reason"]
        assert payload["warnings"]


@pytest.mark.tier1
def test_dual_die_unpatched_and_patched_workspaces_are_distinguished(tmp_path):
    workspace = _workspace(
        tmp_path,
        ("vllm_ascend/distributed/parallel_state.py",),
    )
    fake_bin = _fake_bin(tmp_path)
    before, before_payload = _run_verifier(DUAL, workspace, fake_bin)
    assert before.returncode == 1
    assert before_payload["reward"] == 0.0
    assert before_payload["failure_reason"].startswith("source_missing:")

    _apply_dual_patch(workspace)
    after, after_payload = _run_verifier(DUAL, workspace, fake_bin)
    assert after.returncode == 1
    assert after_payload["reward"] == 0.0
    assert after_payload["failure_reason"].startswith("unsupported:")
    detail = json.loads(
        [line for line in after.stderr.splitlines() if line.strip()][-1]
    )
    assert detail["source"]["candidate_guard_sha256"]
    assert detail["source"]["files"]["vllm_ascend/distributed/placement.py"]


@pytest.mark.tier1
def test_dual_die_placement_file_without_production_integration_is_rejected(tmp_path):
    if not SOURCE_REPO.is_dir():
        pytest.skip(f"pinned candidate repository unavailable: {SOURCE_REPO}")
    workspace = _workspace(
        tmp_path,
        ("vllm_ascend/distributed/parallel_state.py",),
    )
    shutil.copy2(
        DUAL / "oracle/placement_guard.py",
        workspace / "vllm_ascend/distributed/placement.py",
    )
    fake_bin = _fake_bin(tmp_path)
    completed, payload = _run_verifier(DUAL, workspace, fake_bin)
    assert completed.returncode == 1
    assert payload["reward"] == 0.0
    assert "does not integrate validate_rank_tp2_placement" in payload["failure_reason"]


@pytest.mark.tier1
def test_dual_die_gold_patch_integrates_and_executes_the_candidate_guard(tmp_path):
    if not SOURCE_REPO.is_dir():
        pytest.skip(f"pinned candidate repository unavailable: {SOURCE_REPO}")
    workspace = _workspace(
        tmp_path,
        ("vllm_ascend/distributed/parallel_state.py",),
    )
    _apply_dual_patch(workspace)
    applied = workspace / "vllm_ascend/distributed/placement.py"
    assert applied.read_text() == (DUAL / "oracle/placement_guard.py").read_text()
    parallel_state = (workspace / "vllm_ascend/distributed/parallel_state.py").read_text()
    assert "_enforce_tp2_a3_placement" in parallel_state
    assert "global_tp_size == 2" in parallel_state
    assert "tensor_parallel_size != 2" in parallel_state
    assert "AscendDeviceType.A3" in parallel_state
    assert "validate_rank_tp2_placement" in parallel_state
    assert "nnodes" in parallel_state
    assert "assigned_physical_gpu_ids" in parallel_state
    assert "refusing to skip the placement guard" in parallel_state
    assert "cannot resolve visible Dies" in parallel_state
    assert "except Exception:\n        return" not in parallel_state

    guard = _load_module(applied, "candidate_guard")
    assert guard.validate_tp_placement([0, 1])["sibling_pairs"] == [[0, 1]]
    with pytest.raises(guard.CrossPackagePlacementError, match="cross_package"):
        guard.validate_tp_placement([1, 2])
    with pytest.raises(ValueError):
        guard.validate_tp_placement([0, 0])
    with pytest.raises(ValueError):
        guard.validate_tp_placement([])
    assert guard.enforce_visible_tp_placement("0,1")["packages"] == [0]
    with pytest.raises(guard.CrossPackagePlacementError):
        guard.enforce_visible_tp_placement("1,2")
    assert guard.validate_rank_tp2_placement(0, [0, 1, 2, 3])["device_phy_ids"] == [0, 1]
    assert guard.validate_rank_tp2_placement(2, [0, 1, 2, 3])["device_phy_ids"] == [2, 3]
    with pytest.raises(guard.CrossPackagePlacementError):
        guard.validate_rank_tp2_placement(0, [0, 2])
    with pytest.raises(ValueError, match="needs visible slots"):
        guard.validate_rank_tp2_placement(4, [0, 1, 2, 3])

    workload = _load_module(DUAL / "verifier/tp2_workload.py", "dual_die_workload")
    source = (DUAL / "verifier/tp2_workload.py").read_text()
    run_child_source = source[source.index("def run_child(") :]
    assert run_child_source.index("enforce_pair_in_rank(") < run_child_source.index(
        "import torch\n"
    )
    previous = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    try:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
        candidate = workload.enforce_pair_in_rank(
            workspace,
            [0, 1],
            "candidate",
            production_rank=0,
            lease_devices=[0, 1, 2, 3],
        )
        assert candidate["observed"] == "accepted"
        assert candidate["actual_phy_ids"] == [0, 1]
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1,2"
        negative = workload.enforce_pair_in_rank(workspace, [1, 2], "negative_control")
        assert negative["observed"] == "rejected"
    finally:
        if previous is None:
            os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        else:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = previous


def _valid_receipt() -> dict:
    samples = [1.0 + index * 0.01 for index in range(12)]
    errors = [0.001 + index * 0.00001 for index in range(24)]
    control_errors = [0.002 + index * 0.00001 for index in range(24)]
    return {
        "schema": "mindie-bench-tp2-allreduce-workload/1",
        "task_id": "a3-dual-die-tp2-allreduce-placement",
        "nonce": "protocol-nonce",
        "status": "measured",
        "lease_devices": [0, 1, 2, 3],
        "candidate_run": {"pair": [0, 1], "production_path": True},
        "negative_control": {
            "label": "explicit_negative_control",
            "pair": [1, 2],
            "production_path": False,
            "inside_full_four_die_lease": True,
        },
        "measurements": {
            "candidate": {
                "role": "candidate",
                "production_path": True,
                "negative_control": False,
                "device_phy_ids": [0, 1],
                "latency_ms": samples,
                "relative_errors": errors,
                "relative_error": max(errors),
                "guard": {"observed": "accepted", "actual_phy_ids": [0, 1]},
                "cleanup": {"status": "ok"},
            },
            "negative_control": {
                "role": "negative_control",
                "production_path": False,
                "negative_control": True,
                "device_phy_ids": [1, 2],
                "latency_ms": [value * 1.5 for value in samples],
                "relative_errors": control_errors,
                "relative_error": max(control_errors),
                "guard": {"observed": "rejected"},
                "cleanup": {"status": "ok"},
            },
        },
        "shm": {"available": True, "residuals": []},
    }


@pytest.mark.tier0
def test_dual_die_judge_recomputes_gates_from_all_iterations_and_cleanup():
    """Protocol-only arithmetic test; no fabricated receipt can be accepted."""
    judge = _load_module(DUAL / "verifier/production_judge.py", "dual_die_judge")
    receipt = _valid_receipt()
    result = judge.evaluate_receipt(receipt, (0, 1, 2, 3), (0, 1), (1, 2))
    assert result["passed"], result["failures"]
    assert result["metrics"]["candidate_samples"] == 12
    assert result["metrics"]["candidate_error_samples"] == 24
    assert result["metrics"]["candidate_to_negative_control_latency_ratio"] < 1.0

    missing = _valid_receipt()
    missing["measurements"]["candidate"]["latency_ms"] = [1.0, 1.1]
    assert "insufficient_samples:candidate" in judge.evaluate_receipt(
        missing, (0, 1, 2, 3), (0, 1), (1, 2)
    )["failures"]

    error = _valid_receipt()
    error["measurements"]["candidate"]["relative_errors"][-1] = 0.5
    error["measurements"]["candidate"]["relative_error"] = 0.5
    assert any(
        row.startswith("relative_error_exceeded:candidate")
        for row in judge.evaluate_receipt(error, (0, 1, 2, 3), (0, 1), (1, 2))["failures"]
    )

    last_sample = _valid_receipt()
    last_sample["measurements"]["candidate"]["relative_error"] = 0.001
    assert "relative_error_not_max_of_all_iterations:candidate" in judge.evaluate_receipt(
        last_sample, (0, 1, 2, 3), (0, 1), (1, 2)
    )["failures"]

    cleanup = _valid_receipt()
    cleanup["measurements"]["candidate"]["cleanup"] = {
        "status": "failed",
        "error": "destroy_process_group timeout",
    }
    assert any(
        row.startswith("cleanup_not_observed:candidate")
        for row in judge.evaluate_receipt(cleanup, (0, 1, 2, 3), (0, 1), (1, 2))["failures"]
    )

    mislabeled = _valid_receipt()
    mislabeled["measurements"]["negative_control"]["production_path"] = True
    assert "negative_control_mislabeled_as_production" in judge.evaluate_receipt(
        mislabeled, (0, 1, 2, 3), (0, 1), (1, 2)
    )["failures"]

    guard = _valid_receipt()
    guard["measurements"]["candidate"]["guard"] = {"observed": "rejected"}
    assert "guard_not_observed:candidate:accepted" in judge.evaluate_receipt(
        guard, (0, 1, 2, 3), (0, 1), (1, 2)
    )["failures"]

    dirty = _valid_receipt()
    dirty["shm"]["residuals"] = ["hccl_abc"]
    assert any(
        row.startswith("residual_hccl_shm")
        for row in judge.evaluate_receipt(dirty, (0, 1, 2, 3), (0, 1), (1, 2))["failures"]
    )

    slow = _valid_receipt()
    slow["measurements"]["candidate"]["latency_ms"] = [4.0] * 12
    slow["measurements"]["negative_control"]["latency_ms"] = [1.0] * 12
    assert any(
        row.startswith("candidate_latency_degraded")
        for row in judge.evaluate_receipt(slow, (0, 1, 2, 3), (0, 1), (1, 2))["failures"]
    )


@pytest.mark.tier0
def test_dual_die_workload_combines_all_iterations_and_preserves_cleanup_failures():
    workload = _load_module(DUAL / "verifier/tp2_workload.py", "dual_die_workload_combine")
    base = {
        "role": "candidate",
        "production_path": True,
        "negative_control": False,
        "device_phy_ids": [0, 1],
        "guard": {"observed": "accepted", "actual_phy_ids": [0, 1]},
        "cleanup": {"status": "ok"},
        "latency_ms": [1.0, 2.0],
    }
    first = dict(
        base,
        rank=0,
        status="measured",
        relative_errors=[0.001, 0.010],
        relative_error=0.010,
        hcom="hcom",
    )
    second = dict(
        base,
        rank=1,
        status="measured",
        relative_errors=[0.020, 0.030],
        relative_error=0.030,
    )
    combined = workload.combine_rank_receipts([first, second], [0, 1], "candidate")
    assert combined["status"] == "measured", combined.get("errors")
    assert combined["relative_error"] == 0.030
    assert combined["relative_errors"] == [0.001, 0.010, 0.020, 0.030]
    assert combined["cleanup"]["status"] == "ok"

    failed_cleanup = dict(second, cleanup={"status": "failed", "error": "destroy timeout"})
    failed = workload.combine_rank_receipts([first, failed_cleanup], [0, 1], "candidate")
    assert failed["status"] == "failed"
    assert any("cleanup not observed" in row for row in failed["errors"])

    wrong_role = dict(second, role="negative_control", production_path=False, negative_control=True)
    wrong = workload.combine_rank_receipts([first, wrong_role], [0, 1], "candidate")
    assert wrong["status"] == "failed"
    assert any("role=" in row for row in wrong["errors"])


@pytest.mark.tier0
def test_dual_die_judge_parses_a3_phy_ids_and_selects_real_pairs():
    judge = _load_module(DUAL / "verifier/production_judge.py", "dual_die_judge_pairs")
    capture = """+------------------------------------------------------------------+
| npu-smi 26.0.rc1                 Version: 26.0.rc1                 |
+---------------------------+---------------+----------------------+
| NPU   Name                | Health        | Power(W) Temp(C)     |
| Chip  Phy-ID              | Bus-Id        | AICore(%) HBM-Usage  |
+===========================+===============+======================+
| 0     Ascend910           | OK            | 163.9 51 0 / 0       |
| 0     0                   | 0000:9D:00.0  | 0 0 / 0 3122 / 65536 |
+------------------------------------------------------------------+
| 0     Ascend910           | OK            | - 51 0 / 0           |
| 1     1                   | 0000:9F:00.0  | 0 0 / 0 2892 / 65536 |
+===========================+===============+======================+
| 1     Ascend910           | OK            | 173.3 52 0 / 0       |
| 0     2                   | 0000:99:00.0  | 0 0 / 0 3118 / 65536 |
+------------------------------------------------------------------+
| 1     Ascend910           | OK            | - 52 0 / 0           |
| 1     3                   | 0000:9B:00.0  | 0 0 / 0 2889 / 65536 |
+===========================+===============+======================+
| NPU     Chip              | Process id    | Process name         |
+===========================+===============+======================+
| No running processes found in NPU 0                              |
+===========================+===============+======================+
"""
    dies = judge.parse_npu_smi(capture)
    assert sorted(dies) == [0, 1, 2, 3]
    assert dies[2]["npu_id"] == 1 and dies[2]["chip_id"] == 0
    assert not any(row["busy"] for row in dies.values())
    assert judge.select_pairs([0, 1, 2, 3], dies) == ((0, 1), (1, 2))
    with pytest.raises(judge.Unsupported, match="cross_package_control_requires_4_dies"):
        judge.select_pairs([0, 1], dies)
    with pytest.raises(judge.Unsupported, match="lease_visible_order_not_tp2_aligned"):
        judge.select_pairs([1, 0, 2, 3], dies)
    with pytest.raises(judge.Unsupported, match="lease_visible_order_not_tp2_aligned"):
        judge.select_pairs([0, 2, 1, 3], dies)
    dies[2]["busy"] = True
    with pytest.raises(judge.Unsupported, match="leased_die_busy"):
        judge.select_pairs([0, 1, 2, 3], dies)


@pytest.mark.tier1
def test_private_oracle_bundle_runs_without_environment_fixture(tmp_path):
    """Mirror the remote bundle: only verifier/, oracle/ and task.toml arrive."""
    oracle_root = tmp_path / "oracle"
    shutil.copytree(
        DUAL / "verifier",
        oracle_root / "verifier",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(DUAL / "oracle", oracle_root / "oracle")
    shutil.copy2(DUAL / "task.toml", oracle_root / "task.toml")
    assert not (oracle_root / "environment").exists()
    workspace = _workspace(
        tmp_path,
        ("vllm_ascend/distributed/parallel_state.py",),
    )
    _apply_dual_patch(workspace)
    fake_bin = _fake_bin(tmp_path)
    env = dict(os.environ)
    env.update(
        PATH=str(fake_bin),
        MINDIE_BENCH_ORACLE_ROOT=str(oracle_root),
        MINDIE_BENCH_WORKSPACE=str(workspace),
    )
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    completed = subprocess.run(
        ["/bin/bash", str(oracle_root / "verifier/test.sh")],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1, (completed.stdout, completed.stderr)
    payload = json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )
    assert payload["reward"] == 0.0
    assert payload["failure_reason"].startswith("unsupported:npu_smi_unavailable")


@pytest.mark.tier1
@pytest.mark.parametrize(
    "task_name",
    [
        "qwen35-gdn-dp2-tp2-asymmetric-aclgraph",
        "qwen3-moe-tp2-ep4-mc2-eplb-skew",
        "a3-16die-pd-disaggregated-mooncake-handoff",
        "a3-2node-roce-hccl-dp4-ep8-canary",
    ],
)
def test_unsupported_private_bundles_fail_closed_without_model_or_peer(tmp_path, task_name):
    task = TASKS_ROOT / task_name
    oracle_root = tmp_path / "oracle"
    shutil.copytree(
        task / "verifier",
        oracle_root / "verifier",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(task / "task.toml", oracle_root / "task.toml")
    assert not (oracle_root / "environment").exists()
    target_files = tuple(
        tomllib.loads((task / "task.toml").read_text())["mindie_knowledge"]["target_files"]
    )
    workspace = _workspace(tmp_path, target_files)
    fake_bin = _fake_bin(tmp_path)
    env = dict(os.environ)
    env.update(
        PATH=str(fake_bin),
        MINDIE_BENCH_ORACLE_ROOT=str(oracle_root),
        MINDIE_BENCH_WORKSPACE=str(workspace),
    )
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    for name in (
        "MINDIE_BENCH_MODEL_PATH",
        "MINDIE_BENCH_QWEN35_MODEL",
        "MINDIE_BENCH_QWEN3_MOE_MODEL",
        "MINDIE_BENCH_MOONCAKE_READY",
        "MINDIE_BENCH_PEER_HOST",
        "MINDIE_BENCH_HCCL_PEER",
        "MINDIE_BENCH_MULTINODE_DIRECT",
    ):
        env.pop(name, None)
    completed = subprocess.run(
        ["/bin/bash", str(oracle_root / "verifier/test.sh")],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1, (completed.stdout, completed.stderr)
    payload = json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )
    assert payload["reward"] == 0.0
    assert payload["failure_reason"].startswith("unsupported:")


@pytest.mark.tier1
def test_task_test_sh_is_executable_and_verifiers_emit_structured_rewards():
    for path in sorted(item for item in TASKS_ROOT.iterdir() if item.is_dir()):
        assert os.access(path / "verifier/test.sh", os.X_OK)
        text = (path / "verifier/test.sh").read_text()
        assert "MINDIE_BENCH_ORACLE_ROOT" in text
        assert "verifier/verify.py" in text


@pytest.mark.tier1
def test_tp2_child_command_passes_rank_and_reaches_guard_before_torch(tmp_path):
    """Real subprocess boundary: child argv carries rank and dispatches run_child.

    A sentinel guard raises before the child imports torch, so this test cannot
    start an NPU/HCCL run and does not fabricate a measured result.
    """
    workload_path = DUAL / "verifier/tp2_workload.py"
    workload = _load_module(workload_path, "dual_die_workload_child")
    workspace = tmp_path / "workspace"
    (workspace / "vllm_ascend/distributed").mkdir(parents=True)
    (workspace / "vllm_ascend/distributed/placement.py").write_text(
        "class CrossPackagePlacementError(RuntimeError):\n    pass\n\n"
        "def enforce_visible_tp_placement():\n"
        "    raise RuntimeError('argument-boundary-reached-guard')\n"
    )
    config = {
        "schema": workload.SCHEMA,
        "task_id": workload.TASK_ID,
        "nonce": "argument-boundary-nonce",
        "workspace": str(workspace),
        "lease_devices": [0, 1, 2, 3],
        "candidate_pair": [0, 1],
        "negative_control_pair": [1, 2],
        "pair": [0, 1],
        "role": "candidate",
        "arm": "candidate",
        "rank": 1,
        "m": 8,
        "k": 8,
        "n": 8,
        "warmup": 0,
        "samples": 1,
        "pair_timeout_sec": 5,
    }
    config_path = tmp_path / "child.json"
    config_path.write_text(json.dumps(config))
    command = workload.child_command(config_path, 1)
    assert command[-2:] == [str(config_path), "1"]
    env = dict(os.environ)
    env["PATH"] = str(_fake_bin(tmp_path))
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 1, (completed.stdout, completed.stderr)
    payload = json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )
    assert payload["status"] == "failed"
    assert payload["rank"] == 1
    assert "argument-boundary-reached-guard" in payload["error"]
    assert "usage:" not in payload["error"]

