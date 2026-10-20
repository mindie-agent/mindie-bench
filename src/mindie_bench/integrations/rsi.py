"""RSI-Inference task discovery and native Pier execution handoff.

[Role] Expose the upstream GPU benchmark through MindIE Bench without vendoring it.
[Invariants] Preserve upstream tasks, budgets, hardware and verifier isolation.
[Timeouts & Locks] Git probes have a 10s bound; Pier owns execution and cleanup budgets.
[Side-Effect Policy] Discovery is read-only; execution explicitly replaces this process.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

from mindie_bench.io import digest, write_json

UPSTREAM = "https://github.com/apple-aiml-research/ml-rsi-inference-bench"
CATEGORIES = {"model_support", "throughput_optimization", "regression"}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, timeout=10,
    ).stdout.strip()


def read_task(path: Path) -> dict:
    """Read both upstream metadata layouts, without treating GPU tasks as L3 NPU tasks."""
    raw = (path / "task.toml").read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    meta = data["metadata"]["inference_bench"]
    if data["schema_version"] != "1.2" or meta["category"] not in CATEGORIES:
        raise ValueError(f"unsupported RSI task contract: {path}")
    for required in ("instruction.md", "environment/Dockerfile", "tests/test.sh"):
        if not (path / required).is_file():
            raise ValueError(f"missing upstream task asset: {path / required}")
    env, verifier = data["environment"], data["verifier"]
    target = meta.get("target", {})
    return {
        "id": path.name,
        "name": data["task"]["name"],
        "category": meta["category"],
        "track": "model_support" if meta["category"] == "model_support" else "optimization",
        "engine": meta["engine"],
        "score_profile": meta["score_profile"],
        "task_toml_sha256": hashlib.sha256(raw).hexdigest(),
        "target_repository": target.get("repo", data["metadata"].get("repository_url")),
        "target_revision": target.get("commit", data["metadata"].get("base_commit_hash")),
        "model": meta.get("model", {}).get("hf_model_id", data["metadata"].get("model_repository")),
        "hardware": {key: env.get(key) for key in (
            "gpus", "gpu_types", "cpus", "memory_mb", "storage_mb", "shm_size_mb"
        )},
        "allow_internet": env.get("allow_internet", True),
        "agent_timeout_sec": data["agent"]["timeout_sec"],
        "verifier_timeout_sec": verifier["timeout_sec"],
        "build_timeout_sec": env["build_timeout_sec"],
        "verifier_environment_mode": verifier.get("environment_mode", "shared"),
        "verifier_environment": verifier.get("environment"),
        "workloads": meta.get("workloads"),
        "scoring": meta.get("scoring"),
        "correctness": meta.get("correctness"),
    }


def catalog(root: Path) -> dict:
    root = root.resolve()
    if not (root / "third_party/pier/pyproject.toml").is_file():
        raise ValueError("--source must be an RSI checkout containing its bundled Pier runtime")
    tasks = [read_task(path.parent) for path in sorted((root / "tasks").glob("*/task.toml"))]
    if not tasks or len({item["name"] for item in tasks}) != len(tasks):
        raise ValueError("RSI catalog is empty or contains duplicate task names")
    return {
        "schema": "mindie-rsi-catalog/1",
        "upstream": UPSTREAM,
        "source_root": str(root),
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=normal")),
        "catalog_sha256": digest(tasks),
        "categories": dict(Counter(item["category"] for item in tasks)),
        "tasks": tasks,
    }


def plan_run(
    source: Path, *, task_id: str, agent: str, model: str, output: Path,
    backend: str = "docker", attempts: int = 1, env_file: Path | None = None,
    uv: str = "uv",
) -> dict:
    """Prepare a single-task job; serial attempts avoid unallocated GPU oversubscription."""
    inventory = catalog(source)
    if inventory["source_dirty"]:
        raise ValueError("RSI execution requires a clean checkout to bind the run to its source commit")
    selected = [item for item in inventory["tasks"] if item["id"] == task_id]
    if not selected:
        raise ValueError(f"unknown RSI task: {task_id}")
    if backend not in {"docker", "modal"} or attempts < 1:
        raise ValueError("RSI execution requires docker/modal and at least one attempt")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse an RSI run directory: {output}")
    command = [
        uv, "run", "--locked", "--directory", inventory["source_root"], "pier", "run",
        "--path", str(Path(inventory["source_root"]) / "tasks" / task_id),
        "--env", backend, "--agent", agent, "--model", model,
        "--jobs-dir", str(output / "jobs"), "--job-name", "rsi",
        "--n-attempts", str(attempts), "--n-concurrent", "1", "--max-retries", "0",
    ]
    if env_file is not None:
        if not env_file.is_file():
            raise FileNotFoundError(env_file)
        command.extend(["--env-file", str(env_file.resolve())])
    return {
        "schema": "mindie-rsi-run/1", "source_commit": inventory["source_commit"],
        "upstream": UPSTREAM, "catalog_sha256": inventory["catalog_sha256"],
        "task": selected[0], "attempts": attempts, "agent": agent, "model": model,
        "backend": backend, "output": str(output), "command": command,
        "execution": "upstream_pier", "ablation_group": None,
    }


def execute_plan(plan: dict) -> None:
    """Hand off lifecycle ownership instead of wrapping Pier in a second runner."""
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run-plan.json", plan, exclusive=True)
    # No credential values are copied into the manifest. Pier consumes the
    # selected environment and retains its own timeout / container cleanup logic.
    # Upstream reads task files with Python's default encoding. UTF-8 keeps
    # real task text portable on Windows hosts using a legacy ANSI code page.
    os.execvpe(plan["command"][0], plan["command"], {**os.environ, "PYTHONUTF8": "1"})
