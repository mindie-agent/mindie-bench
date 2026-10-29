# [Role] Reproducible zero-model behavioral controls for RoPE and offline component tasks.
# [Invariants] Runs actual verifier subprocesses and preserves each input, output and SHA.
# [Timeouts & Locks] Each subprocess has a 40s bound; remote caller reserves the sum plus cleanup.
# [Side-Effect Policy] Writes only requested report/workspace directories and task soundness receipts.
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

COMPONENTS = {
    "profiler-msprof-straggler-and-hbm-fix-zh": "reconcile_profile_and_hbm",
    "sim-a3-ub-fusion-and-pipeline-dispatch-zh": "evaluate_a3_simulation_suite",
    "diagnostics-cann-plog-cascade-root-cause-fix-zh": "reconcile_plog_and_outbox",
    "npu-top-and-coordinator-multi-die-lease-repair-zh": "reconcile_multi_die_and_leases",
}
STAGES = {
    "base": 0.0,
    "negative": 0.0,
    "gold": 1.0,
    "gold_helper_function": 1.0,
    "gold_helper_class": 1.0,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root, exclude=()):
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in exclude and "__pycache__" not in p.parts
    }
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def helper_variant(source, entry, *, class_helper):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry)
    function.name = "_soundness_original_implementation"
    if class_helper:
        wrapper = f"""class SoundnessHelper:
    @staticmethod
    def invoke(*args, **kwargs):
        return _soundness_original_implementation(*args, **kwargs)

def {entry}(*args, **kwargs):
    return SoundnessHelper.invoke(*args, **kwargs)
"""
    else:
        wrapper = f"""def _soundness_helper(*args, **kwargs):
    return _soundness_original_implementation(*args, **kwargs)

def {entry}(*args, **kwargs):
    return _soundness_helper(*args, **kwargs)
"""
    tree.body.extend(ast.parse(wrapper).body)
    return ast.unparse(tree) + "\n"


def prepare(repo, output, *, npu, python, tasks=None):
    names = (
        [f"rope-rotary-dim-{d}-zh-vague" for d in (32, 64)]
        if npu
        else [
            *[f"cpu-rope-rotary-dim-{d}-zh-vague" for d in (32, 64)],
            *COMPONENTS,
        ]
    )
    if tasks:
        names = [name for name in names if name in tasks]
        if set(names) != set(tasks):
            raise ValueError("requested task does not belong to the selected CPU/NPU scope")
    jobs = []
    for name in names:
        task = repo / "tasks/zh-vague" / name
        spec = tomllib.loads((task / "task.toml").read_text())
        tier = "l3" if npu else "l2"
        environment = spec["environment"][tier]
        source_repo = Path(environment["repo_path"])
        commit = environment["base_commit"]
        target = Path(spec["mindie_knowledge"]["target_files"][0])
        entry = COMPONENTS.get(name, "set_cos_and_sin")
        task_output = output / name
        task_output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task / "verifier", task_output / "verifier", dirs_exist_ok=True)
        files = [str(target)]
        if name in COMPONENTS:
            files.extend(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(source_repo),
                        "ls-tree",
                        "-r",
                        "--name-only",
                        commit,
                        "vllm_ascend_diag/fixtures",
                    ],
                    text=True,
                    timeout=10,
                ).splitlines()
            )
        base = task_output / "base"
        for relative in files:
            dest = base / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(
                subprocess.check_output(
                    ["git", "-C", str(source_repo), "show", f"{commit}:{relative}"], timeout=10
                )
            )
        gold = task_output / "gold"
        shutil.copytree(base, gold, dirs_exist_ok=True)
        subprocess.run(
            ["git", "apply", "--no-index", str(task / "oracle/gold_patch.diff")],
            cwd=gold,
            check=True,
            capture_output=True,
            timeout=10,
        )
        for stage in ("negative", "gold_helper_function", "gold_helper_class"):
            shutil.copytree(gold, task_output / stage, dirs_exist_ok=True)
        gold_source = (gold / target).read_text()
        for stage in ("gold_helper_function", "gold_helper_class"):
            (task_output / stage / target).write_text(
                helper_variant(gold_source, entry, class_helper=stage.endswith("class"))
            )
        negative = task_output / "negative" / target
        if name in COMPONENTS:
            # Replay the complete default gold output as a constant. Perturbed inputs must reject it.
            capture = subprocess.run(
                [
                    python,
                    "-I",
                    "-B",
                    "-c",
                    "import importlib.util,json,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(getattr(m,sys.argv[2])()))",
                    str(gold / target),
                    entry,
                ],
                capture_output=True,
                text=True,
                timeout=40,
            )
            if capture.returncode:
                raise RuntimeError(capture.stderr)
            constant = json.loads(capture.stdout)
            negative.write_text(f"def {entry}(*args, **kwargs):\n    return {constant!r}\n")
        else:
            negative.write_text(
                "def set_cos_and_sin(*args, **kwargs):\n    return None\n\ndef update_cos_sin(*args, **kwargs):\n    return None\n\ndef get_cos_and_sin_slice():\n    return None, None\n"
            )
        job = {
            "task": name,
            "source_commit": commit,
            "tier": tier,
            "target": str(target),
            "task_sha256": tree_hash(task, ("soundness_receipt.json",)),
            "verifier_sha256": tree_hash(task / "verifier"),
            "verifier_files": {
                p.name: digest(p) for p in sorted((task / "verifier").iterdir()) if p.is_file()
            },
            "scope": "npu_tensor_correctness"
            if npu
            else "cpu_tensor_correctness"
            if name.startswith("cpu-")
            else "controlled_offline_fixture",
        }
        jobs.append(job)
    write_json(output / "jobs.json", jobs)
    shutil.copy2(Path(__file__), output / "run_soundness.py")
    return jobs


def run_prepared(output, python):
    jobs = json.loads((output / "jobs.json").read_text())
    results = []
    for job in jobs:
        directory = output / job["task"]
        runs = {}
        for stage, expected in STAGES.items():
            workspace = directory / stage
            env = dict(os.environ, MINDIE_BENCH_WORKSPACE=str(workspace), MINDIE_BENCH_PYTHON=python)
            command = ["bash", str(directory / "verifier/test.sh")]
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    command, cwd=workspace, env=env, capture_output=True, text=True, timeout=40
                )
                stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = (
                    (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                )
                stderr = (
                    (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                )
                stderr += "\nVerifier subprocess exceeded 40s budget.\n"
                code = None
            (directory / f"{stage}.stdout").write_text(stdout)
            (directory / f"{stage}.stderr").write_text(stderr)
            verdict = None
            for line in reversed(stdout.splitlines()):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and "reward" in item:
                    verdict = item
                    break
            reward = verdict.get("reward") if verdict else None
            runs[stage] = {
                "command": command,
                "python_executable": python,
                "expected_reward": expected,
                "reward": reward,
                "exit_code": code,
                "passed": reward == expected and code == (0 if expected == 1.0 else 1),
                "duration_sec": time.monotonic() - start,
                "verdict": verdict,
                "source_sha256": digest(workspace / job["target"]),
                "workspace_sha256": tree_hash(workspace),
                "stdout_sha256": digest(directory / f"{stage}.stdout"),
                "stderr_sha256": digest(directory / f"{stage}.stderr"),
            }
            print(
                json.dumps(
                    {
                        "task": job["task"],
                        "stage": stage,
                        "reward": reward,
                        "passed": runs[stage]["passed"],
                    }
                ),
                flush=True,
            )
        receipt = dict(
            job,
            schema_version="2.0",
            status="completed" if all(r["passed"] for r in runs.values()) else "failed",
            model_calls=0,
            before_reward=runs["base"]["reward"],
            negative_reward=runs["negative"]["reward"],
            after_reward=runs["gold"]["reward"],
            controls=runs,
            measured_npu_speedup=None,
            speedup_ratio=None,
            tpot_ms=None,
            errors=[
                {"stage": s, "reason": r["verdict"] or "verifier did not emit a verdict"}
                for s, r in runs.items()
                if not r["passed"]
            ],
        )
        write_json(directory / "soundness.json", receipt)
        results.append(receipt)
    write_json(output / "results.json", results)
    return results


def publish(repo, output):
    results = json.loads((output / "results.json").read_text())
    for receipt in results:
        task = repo / "tasks/zh-vague" / receipt["task"]
        if tree_hash(task, ("soundness_receipt.json",)) != receipt["task_sha256"]:
            raise RuntimeError(f"task changed after soundness: {task}")
        receipt["evidence_directory"] = str(output / receipt["task"])
        write_json(task / "soundness_receipt.json", receipt)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--task", action="append")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.publish_only:
        results = publish(args.repo, output)
    else:
        if not args.run_prepared:
            prepare(args.repo, output, npu=args.npu, python=args.python, tasks=args.task)
        if args.prepare_only:
            return
        results = run_prepared(output, args.python)
        if not args.run_prepared:
            publish(args.repo, output)
    raise SystemExit(0 if all(r["status"] == "completed" for r in results) else 1)


if __name__ == "__main__":
    main()
