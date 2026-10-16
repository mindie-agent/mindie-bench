"""Zero-model fail-before / pass-after validation against the external judge."""

from __future__ import annotations

import base64
import re
import subprocess
import uuid
from pathlib import Path

from mindie_bench.io import task_hash, write_json
from mindie_bench.models import Group, TaskSpec
from mindie_bench.runner.orchestrator import runtime_paths
from mindie_bench.runner.profile_builder import build_profile
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.verifier import local_verifier, parse_verifier
from mindie_bench.runner.worktree_slot import WorktreeSlot


def check_soundness(task_root: Path, config, *, tier: str, output: Path, promote: bool = False):
    task = TaskSpec.load(task_root)
    environment = task.environment[tier]
    patch = task_root / "oracle/gold_patch.diff"
    if not patch.is_file():
        raise ValueError("soundness requires an external gold_patch.diff")
    if (task_root / "soundness_receipt.json").exists():
        raise FileExistsError(
            "a soundness proof already exists; preserve it and select a new task revision"
        )
    if promote and task.task.evidence != "candidate":
        raise ValueError("only candidate tasks can be promoted; synthetic fixtures remain synthetic")
    output.mkdir(parents=True, exist_ok=False)
    profile = build_profile(output, task, Group.G0, config)
    slot = WorktreeSlot(environment, output / "workspace", task_root)
    remote = None
    errors = []
    proof = dict(
        schema_version="1.0",
        tier=tier,
        status="failed",
        model_calls=0,
        before_reward=None,
        after_reward=None,
        source_commit=environment.base_commit,
    )
    try:
        workspace = slot.prepare()
        if tier == "l3":
            if not config.remote or environment.npu_count < 1:
                raise ValueError("L3 soundness requires a configured NPU worker")
            remote = RemoteSlot(
                config.remote,
                environment,
                "soundness-" + uuid.uuid4().hex,
                config.remote.devices[: environment.npu_count],
                task.budget.verifier_timeout_sec,
                artifact_dir=output,
            ).prepare()
            remote.upload_public(task_root / "environment")
        paths = runtime_paths(config, profile)

        def verify():
            if remote:
                result = remote.verify((task_root / "verifier/test.sh").read_bytes())
                return parse_verifier(
                    result["exit_code"], base64.b64decode(result["stdout_b64"]).decode()
                )
            return local_verifier(
                task_root, workspace, profile, output, task.budget.verifier_timeout_sec, paths
            )[0]

        before = verify()
        for name in ["stdout", "stderr"]:
            path = output / f"verifier.{name}"
            if path.exists():
                path.rename(output / f"verifier_before.{name}")
        proof["before_reward"] = before.reward
        if before.reward != 0.0:
            raise ValueError("base source already passes; this is not a valid fail-to-pass task")
        if remote:
            applied = remote.execute(
                "git apply --no-index -", base64.b64encode(patch.read_bytes()).decode()
            )
            if applied["exit_code"] != 0:
                raise RuntimeError("gold patch did not apply in the remote slot")
        else:
            subprocess.run(
                ["git", "apply", "--no-index", str(patch.resolve())],
                cwd=workspace,
                env=dict(profile.env, GIT_CEILING_DIRECTORIES=str(workspace.parent)),
                check=True,
                capture_output=True,
                timeout=20,
            )
        after = verify()
        for name in ["stdout", "stderr"]:
            path = output / f"verifier.{name}"
            if path.exists():
                path.rename(output / f"verifier_after.{name}")
        proof["after_reward"] = after.reward
        if after.reward != 1.0:
            raise ValueError("gold source did not pass the external verifier")
        proof["status"] = "completed"
        if remote:
            proof["remote_preflight"] = remote.preflight
    except Exception as exc:
        errors.append(dict(stage="soundness", type=type(exc).__name__, message=str(exc)))
    finally:
        if remote:
            try:
                remote.close()
                proof["remote_postflight"] = remote.postflight
            except Exception as exc:
                errors.append(dict(stage="remote_cleanup", type=type(exc).__name__, message=str(exc)))
        try:
            slot.cleanup()
        except Exception as exc:
            errors.append(dict(stage="worktree_cleanup", type=type(exc).__name__, message=str(exc)))
        proof["errors"] = errors
        if errors:
            proof["status"] = "failed"
        write_json(output / "soundness_attempt.json", proof, exclusive=True)
    if errors:
        raise RuntimeError("soundness failed; inspect its preserved soundness_attempt.json")
    if promote:
        manifest = task_root / "task.toml"
        text, count = re.subn(
            r'(?m)^evidence\s*=\s*"candidate"\s*$', 'evidence = "verified"', manifest.read_text()
        )
        if count != 1:
            raise ValueError(
                "promotion requires an explicit, unique task.evidence=candidate declaration"
            )
        manifest.write_text(text)
    proof["task_sha256"] = task_hash(task_root)
    # A second request must use a new evidence destination; it never replaces
    # a previous proof or silently retries a hardware/model operation.
    write_json(task_root / "soundness_receipt.json", proof, exclusive=True)
    return proof
