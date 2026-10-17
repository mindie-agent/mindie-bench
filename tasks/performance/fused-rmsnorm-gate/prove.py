"""Zero-model source/gold proof through the existing five-component RemoteSlot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "src"))

from mindie_bench.evaluation.oracle_freezer import check_oracle, freeze_oracle
from mindie_bench.io import task_hash, write_json
from mindie_bench.models import RemoteConfig, TaskSpec
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.verifier import remote_verifier


def prove(output: Path):
    output = output.resolve()
    task = TaskSpec.load(ROOT)
    frozen = freeze_oracle(ROOT, task.oracle)
    config = json.loads(
        (ROOT.parents[2] / "reports/v0.2.0-runtime-canary/slot-1/config.json").read_text()
    )
    config["repo_path"] = "/bench/source-fusedgate-history-20261017.git"
    config["slots_root"] = "/bench/slots-performance-fusedgate-20261017"
    # The installed backend already contains the repair; never expose its Python
    # sources to a future foreground candidate. Reuse vLLM and torch dependencies.
    config["runtime_read_paths"] = [
        "/usr/local/Ascend",
        "/usr/local/python3.12.13",
        "/usr/local/dcmi",
        "/vllm-workspace/vllm",
        "/vllm-workspace/vllm-ascend/vllm_ascend/_build_info.py",
        "/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so",
        "/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so",
    ]
    config["component_runtime"]["diagnostics_root"] = str(output / "diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    # Development proof reruns replace their own local stderr after cleanup.
    stderr_path = output / "remote.stderr"
    if stderr_path.exists():
        stderr_path.unlink()
    write_json(output / "remote_config.json", config)
    remote = RemoteSlot(
        RemoteConfig.model_validate(config),
        task.environment["l3"],
        "fusedgate-proof-" + uuid.uuid4().hex[:12],
        [12],
        task.budget.verifier_timeout_sec,
        artifact_dir=output,
    )
    proof = {
        "schema_version": "1.0",
        "tier": "l3",
        "model_calls": 0,
        "source_commit": task.environment["l3"].base_commit,
        "status": "failed",
        "before_reward": None,
        "after_reward": None,
        "oracle_manifest": frozen.to_dict(),
        "errors": [],
    }
    try:
        remote.prepare()
        remote.upload_public(ROOT / "environment")
        for stage in ["source", "gold"]:
            stage_dir = output / stage
            stage_dir.mkdir(exist_ok=True)
            value, record = remote_verifier(
                ROOT,
                remote,
                stage_dir,
                task.budget.verifier_timeout_sec,
                frozen,
                task.oracle,
                task.performance,
            )
            write_json(stage_dir / "verdict.json", record)
            if value.oracle_status == "oracle_tampered":
                raise RuntimeError(f"{stage} oracle control failed: {value.failure_reason}")
            proof["before_reward" if stage == "source" else "after_reward"] = value.reward
            if stage == "source":
                if value.reward != 0.0:
                    raise RuntimeError("historical source passes; this is not a valid performance task")
                applied = remote.execute(
                    "git apply --no-index -",
                    base64.b64encode((ROOT / "oracle/gold_patch.diff").read_bytes()).decode(),
                )
                write_json(output / "gold_apply.json", applied)
                if applied["exit_code"] != 0:
                    raise RuntimeError("actual historical gold patch did not apply")
        if proof["after_reward"] != 1.0:
            raise RuntimeError("gold does not satisfy the actual performance/transfer gate")
        proof["status"] = "completed"
    except Exception as error:
        proof["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        if remote.created or remote.process is not None or remote.component_runtime is not None:
            try:
                remote.close()
            except Exception as error:
                proof["errors"].append(f"cleanup {type(error).__name__}: {error}")
                proof["status"] = "failed"
        proof["preflight"] = remote.preflight
        proof["postflight"] = remote.postflight
        proof["remote_cleanup"] = remote.cleanup_receipt
        proof["operation_unknown"] = remote.operation_unknown
        proof["task_sha256"] = task_hash(ROOT)
        check_oracle(frozen, task.oracle)
        write_json(output / "proof.json", proof)
        print(
            json.dumps(
                {
                    key: proof[key]
                    for key in ["status", "before_reward", "after_reward", "errors", "operation_unknown"]
                },
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prove(parser.parse_args().output)
