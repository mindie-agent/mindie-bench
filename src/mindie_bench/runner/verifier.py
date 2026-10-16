"""The judge runs after agent termination and must distinguish failure from outage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mindie_bench.models import VerifierResult
from mindie_bench.runner.sandbox import native_command


def parse_verifier(returncode: int, stdout: str) -> VerifierResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if lines:
        try:
            raw = json.loads(lines[-1])
        except ValueError:
            raw = None
        if isinstance(raw, dict):
            value = VerifierResult.model_validate(raw)
            if (returncode == 0) != (value.reward == 1.0):
                raise ValueError("verifier reward contradicts exit status")
            return value
    if returncode == 0:
        return VerifierResult(reward=1.0)
    raise ValueError("nonzero verifier without structured reward is infrastructure failure")


def local_verifier(
    task_root: Path, workspace: Path, profile, trial_dir: Path, timeout: float, read_paths: list[Path]
) -> tuple[VerifierResult, dict]:
    trajectory_path = trial_dir / "verifier_trajectory.json"
    judge_reads = [trajectory_path] if trajectory_path.exists() else []
    command = native_command(
        ["/bin/bash", str(task_root / "verifier/test.sh")],
        read=[workspace, task_root / "verifier", *read_paths, *judge_reads],
        write=[profile.root / "tmp"],
        cwd=workspace,
        policy_path=trial_dir / "verifier.sb",
    )
    env = dict(
        profile.env, MINDIE_BENCH_WORKSPACE=str(workspace), MINDIE_BENCH_TRAJECTORY=str(trajectory_path)
    )
    result = subprocess.run(
        command, cwd=workspace, env=env, timeout=timeout, capture_output=True, text=True
    )
    (trial_dir / "verifier.stdout").write_text(result.stdout)
    (trial_dir / "verifier.stderr").write_text(result.stderr)
    value = parse_verifier(result.returncode, result.stdout)
    return value, dict(returncode=result.returncode, result=value.model_dump())
