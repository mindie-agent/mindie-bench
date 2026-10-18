from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path

import pytest

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.models import OracleContract, RemoteConfig, SourceEnvironment
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.verifier import remote_verifier

LIVE_HOST = os.environ.get("MINDIE_BENCH_LIVE_A2_HOST", "root@173.125.1.2")
LIVE_CONTAINER = os.environ.get(
    "MINDIE_BENCH_LIVE_A2_CONTAINER", "mindie-bench-a2-core-20261017"
)
LIVE_IMAGE_DIGEST = (
    "sha256:d85730f56df057d890f9f1ded8f9d42b55e746e1cd1e9e3bb6a8adf36eeb4a16"
)


def _container_exec(command: str, *, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            LIVE_HOST,
            shlex.join(["docker", "exec", "-i", LIVE_CONTAINER, "/bin/bash", "-lc", command]),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _select_idle_a2_device() -> int:
    probe = _container_exec(
        shlex.join(
            [
                "python3",
                "-I",
                "/bench/ascend_probe.py",
                "--devices",
                "0,1,2,3,4,5,6,7",
                "--image-digest",
                LIVE_IMAGE_DIGEST,
            ]
        )
    )
    if probe.returncode != 0:
        pytest.skip(f"live A2 container {LIVE_CONTAINER} on {LIVE_HOST} unavailable: {probe.stderr.strip()}")
    payload = json.loads(probe.stdout)
    for device in range(7, -1, -1):
        row = payload.get(str(device), {})
        if not row.get("pids") and row.get("used_mb", 999999) < 4000 and row.get("model") == "910B4":
            return device
    pytest.skip(f"no idle 910B4 device found on {LIVE_HOST}")


@pytest.mark.live_npu
def test_live_a2_remote_slot_workspace_projection_verifier_inventory_and_timeout_isolation(
    tmp_path: Path,
) -> None:
    device = _select_idle_a2_device()
    sandbox_root = f"/tmp/mindie-bench-live-test-{uuid.uuid4().hex[:12]}"
    repo_path = f"{sandbox_root}/repo"
    slots_root = f"{sandbox_root}/slots"

    setup_script = f"""
set -eu
mkdir -p {shlex.quote(repo_path)} {shlex.quote(slots_root)}
git -C {shlex.quote(repo_path)} init -q
cat > {shlex.quote(repo_path)}/module.py <<'EOF'
VALUE = 1
EOF
git -C {shlex.quote(repo_path)} add module.py
git -C {shlex.quote(repo_path)} -c user.name='Live Test' -c user.email='live@example.invalid' commit -qm 'init'
git -C {shlex.quote(repo_path)} rev-parse HEAD
"""
    init_res = _container_exec(setup_script)
    assert init_res.returncode == 0, init_res.stderr
    base_commit = init_res.stdout.strip().splitlines()[-1]

    remote_config = RemoteConfig(
        host=LIVE_HOST,
        container=LIVE_CONTAINER,
        repo_path=repo_path,
        slots_root=slots_root,
        devices=[device],
        memory_limit_mb=4000,
        probe_command=[
            "python3",
            "-I",
            "/bench/ascend_probe.py",
            "--devices",
            str(device),
            "--image-digest",
            LIVE_IMAGE_DIGEST,
        ],
        runtime_read_paths=["/usr/local/Ascend"],
    )
    source_env = SourceEnvironment(
        repo_path=repo_path,
        base_commit=base_commit,
        npu_model="910B4",
        npu_count=1,
    )

    artifact_dir = tmp_path / "slot-artifacts"
    artifact_dir.mkdir()
    slot = RemoteSlot(
        remote_config,
        source_env,
        trial_id="live-slot-trial",
        devices=[device],
        timeout=30.0,
        artifact_dir=artifact_dir,
    )
    try:
        slot.prepare()
        assert slot.created is True
        assert slot.preflight is not None
        assert slot.preflight[str(device)]["model"] == "910B4"

        public_dir = tmp_path / "public"
        public_dir.mkdir()
        (public_dir / "notes.txt").write_text("projected-public-asset\n")
        slot.upload_public(public_dir)

        exec_res = slot.execute(
            "python3 -c 'import os, pathlib; "
            "assert pathlib.Path(\"notes.txt\").read_text() == \"projected-public-asset\\n\"; "
            "pathlib.Path(\"module.py\").write_text(\"VALUE = 2\\n\"); "
            "print(os.environ[\"ASCEND_RT_VISIBLE_DEVICES\"])'"
        )
        assert exec_res["exit_code"] == 0
        assert base64.b64decode(exec_res["stdout_b64"]).decode().strip() == str(device)

        # 1. Passing remote_verifier check with real inventory() on the remote NPU container
        pass_task = tmp_path / "pass-task"
        (pass_task / "verifier").mkdir(parents=True)
        (pass_task / "oracle").mkdir()
        (pass_task / "oracle/golden.py").write_text("EXPECTED = 2\n")
        (pass_task / "verifier/test.sh").write_text(
            '#!/bin/bash\n'
            'if grep -qx "VALUE = 2" module.py; then\n'
            '  echo \'{"reward": 1.0}\'\n'
            '  exit 0\n'
            'fi\n'
            'echo \'{"reward": 0.0}\'\n'
            'exit 1\n'
        )
        pass_trial = tmp_path / "pass-trial"
        pass_trial.mkdir()
        contract = OracleContract()
        verdict, record = remote_verifier(
            pass_task,
            slot,
            pass_trial,
            10.0,
            freeze_oracle(pass_task, contract),
            contract,
            None,
        )
        assert verdict.reward == 1.0
        assert verdict.timed_out is False
        assert verdict.oracle_status == "intact"
        assert record["oracle"]["private_bundle"]["before"] == record["oracle"]["private_bundle"]["after"]
        assert slot.operation_unknown is False

        # 2. Timeout remote_verifier check (`sleep 2` with `timeout=0.5`) verifying no UnknownExternalEffects pollution
        timeout_task = tmp_path / "timeout-task"
        (timeout_task / "verifier").mkdir(parents=True)
        (timeout_task / "oracle").mkdir()
        (timeout_task / "oracle/golden.py").write_text("EXPECTED = 2\n")
        (timeout_task / "verifier/test.sh").write_text("#!/bin/bash\nsleep 2\necho '{\"reward\": 1.0}'\n")
        timeout_trial = tmp_path / "timeout-trial"
        timeout_trial.mkdir()
        timeout_verdict, timeout_record = remote_verifier(
            timeout_task,
            slot,
            timeout_trial,
            0.5,
            freeze_oracle(timeout_task, contract),
            contract,
            None,
        )
        assert timeout_verdict.reward == 0.0
        assert timeout_verdict.timed_out is True
        assert timeout_verdict.oracle_status == "intact"
        assert timeout_record.get("timed_out") is True
        assert (
            timeout_record["oracle"]["private_bundle"]["before"]
            == timeout_record["oracle"]["private_bundle"]["after"]
        )
        assert slot.operation_unknown is False
    finally:
        try:
            if slot.process is not None and not slot.closed:
                slot.close()
        finally:
            _container_exec(f"rm -rf {shlex.quote(sandbox_root)}")

    assert slot.modified_files == ["module.py"]
    assert slot.postflight is not None
