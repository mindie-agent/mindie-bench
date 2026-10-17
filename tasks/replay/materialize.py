"""Materialize one source-bound Ascend diagnostic replay with a CPU tensor judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(source_repo: Path, runtime_manifest: Path, output: Path) -> Path:
    templates = Path(__file__).resolve().parent
    template = templates / "sampler-topk"
    source = json.loads((template / "source-manifest.json").read_text())
    runtime = json.loads(runtime_manifest.read_text())
    if runtime.get("schema") != "mindie-bench-cpu-task-runtime/1" or runtime["status"] != "completed":
        raise ValueError("an independently prepared, completed locked CPU runtime is required")
    python = Path(runtime["interpreter"]).absolute()
    if any(character in str(python) for character in "'\n\r"):
        raise ValueError("unsupported CPU interpreter path quoting")
    if sha(python.resolve(strict=True)) != runtime["python_binary_sha256"]:
        raise ValueError("CPU runtime interpreter hash changed")
    if sha(templates.parent / "cpu/runtime/uv.lock") != runtime["uv_lock_sha256"]:
        raise ValueError("CPU runtime lock differs from this recipe")
    source_repo = source_repo.resolve(strict=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source_repo), *args], timeout=30)

    for key in ["base_commit", "fix_commit"]:
        if git("rev-parse", source[key] + "^{commit}").decode().strip() != source[key]:
            raise ValueError("source commit did not resolve exactly")
    original = git("show", source["base_commit"] + ":" + source["source_path"])
    if (
        hashlib.sha256(original).hexdigest() != source["source_sha256"]
        or original != (template / "source_snapshot/sampler.py.txt").read_bytes()
    ):
        raise ValueError("complete historical source blob identity changed")
    gold = git("show", source["fix_commit"] + ":" + source["source_path"])
    if hashlib.sha256(gold).hexdigest() != source["gold_source_sha256"]:
        raise ValueError("historical gold blob identity changed")
    for relative, expected in [
        ("oracle/gold_patch.diff", source["gold_patch_sha256"]),
        ("environment/history-error.stderr", source["history_stderr_sha256"]),
    ]:
        if sha(template / relative) != expected:
            raise ValueError("historical replay input identity changed: " + relative)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = output / "replay-sampler-topk"
    task.mkdir()
    for name in ["environment", "verifier", "oracle"]:
        shutil.copytree(template / name, task / name)
    shutil.copy2(
        template / "environment/replay_cpu_boundary.py", task / "verifier/replay_cpu_boundary.py"
    )
    binding = {
        "python": runtime["python_binary_sha256"],
        "torch": runtime["observed"]["torch"],
        "runtime_manifest_sha256": sha(runtime_manifest),
        "uv_lock_sha256": runtime["uv_lock_sha256"],
    }
    (task / "verifier/runtime-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    (task / "verifier/test.sh").write_text(
        "#!/bin/bash\nset -eu\nexec '" + str(python) + '\' -I "$(dirname "$0")/verify.py"\n'
    )
    text = (template / "instruction.md").read_text().replace("@CPU_PYTHON@", str(python))
    (task / "instruction.md").write_text(text)
    (task / "task.toml").write_text(f'''schema_version = "1.0"
[task]
instance_id = "replay-sampler-topk"
domain_category = "operator_shape_contract"
difficulty = "medium"
evidence = "candidate"
[task.source]
repository = "{source["repository"]}"
fix_commit = "{source["fix_commit"]}"
validation_scope = "complete historical module on CPU; historical real NPU diagnostic replay; no new hardware or HCCL acceptance"
[mindie_knowledge]
corpus_id = "replay-empty"
knowledge_gradient = "no_hit"
target_files = ["{source["source_path"]}"]
[budget]
max_turns = 35
max_npu_runs = 12
timeout_sec = 600
verifier_timeout_sec = 90
[environment.l2]
repo_path = {json.dumps(str(source_repo))}
base_commit = "{source["base_commit"]}"
npu_count = 0
replay_commands = ["npu-replay-sampler"]
''')
    report = {
        "schema": "mindie-bench-ascend-replay-materialization/1",
        "task_root": str(task),
        "source_manifest_sha256": sha(template / "source-manifest.json"),
        "source": source,
        "runtime_binding": binding,
        "proofs_inherited": False,
        "model_calls": 0,
        "npu_calls": 0,
        "knowledge_requirement": "Freeze and mount an explicitly empty replay-empty scope for no_hit.",
    }
    (output / "materialization.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(materialize(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
