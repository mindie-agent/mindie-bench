"""Task-balanced scoring of original RSI/Pier trial receipts.

[Role] Import external RSI results and report its two lifecycle tracks.
[Invariants] Partial support is failure; invalid runs are not neutral speedups.
[Timeouts & Locks] Local file reads only; no subprocesses, locks or network.
[Side-Effect Policy] Read original receipts; never relabel MindIE's native trials.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from mindie_bench.io import digest, read_json

EXECUTION_FIELDS = (
    "agent", "environment", "verifier", "timeout_multiplier", "agent_timeout_multiplier",
    "verifier_timeout_multiplier", "agent_setup_timeout_multiplier", "environment_build_timeout_multiplier",
)


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def classify(task: dict, result: dict, rewards: dict | None, metrics: dict) -> dict:
    """Interpret the released verifier contract, never derive speedup as 1 + reward."""
    outcome = {"status": "unknown", "score": None, "reason": None}
    if metrics.get("infra_void") or (rewards and rewards.get("infra_void")):
        return {**outcome, "status": "infra_failed", "reason": "infra_void"}
    if rewards and rewards.get("preflight_gpu_mismatch_allowed"):
        return {**outcome, "reason": "nonstandard_hardware"}
    if rewards is not None and rewards.get("reward") == -1:
        return {**outcome, "status": "infra_failed", "reason": "upstream_void_sentinel"}
    # Pier's errors include teardown, transport and agent errors: without a
    # known terminal boundary none is evidence of a completed candidate failure.
    if result.get("exception_info"):
        return {**outcome, "reason": "pier_exception", "exception": result["exception_info"]}
    if not result.get("finished_at") or rewards is None or "reward" not in rewards:
        return {**outcome, "reason": "missing_terminal_reward"}
    reward = _number(rewards["reward"], "reward")
    if reward < 0:
        raise ValueError(f"unrecognized negative reward: {reward}")
    if task["category"] == "model_support":
        if reward > 1:
            raise ValueError("model-support reward must be a fraction in [0, 1]")
        return {
            "status": "completed_pass" if reward == 1 else "completed_fail",
            "score": float(reward == 1), "reason": None, "test_pass_fraction": reward,
        }
    correctness = _number(rewards.get("correctness"), "correctness")
    if correctness not in {0, 1}:
        raise ValueError("optimization correctness must be 0 or 1")
    if correctness == 0:
        if reward > 0:
            raise ValueError("positive reward contradicts failed correctness")
        return {"status": "completed_fail", "score": 1.0, "reason": "correctness_failed"}
    ratio = rewards.get("speedup_geomean")
    if ratio is None:
        # Failed acceptance/guard/no-patch cases have zero reward and no scored
        # ratio. Partial diagnostic ratios in metrics.json must never be scored.
        if reward > 0:
            raise ValueError("positive reward is missing its measured speedup_geomean")
        return {"status": "completed_fail", "score": 1.0, "reason": metrics.get("reason")}
    ratio = _number(ratio, "speedup_geomean")
    if ratio <= 0:
        raise ValueError("measured speedup must be positive")
    return {
        "status": "completed_pass" if reward > 0 else "completed_fail",
        "score": ratio, "reason": metrics.get("reason"),
    }


def load_trials(root: Path, inventory: dict) -> list[dict]:
    """Read actual Pier result.json files, including failures without verifier output."""
    tasks = {item["name"]: item for item in inventory["tasks"]}
    rows, seen = [], set()
    # Pier's result_path is singular; unrelated nested artifacts must not be
    # mistaken for trial receipts. A caller may also select one result directly.
    paths = [root] if root.is_file() else sorted(root.rglob("result.json"))
    for path in paths:
        result = read_json(path)
        if "trial_name" not in result or "task_name" not in result:
            continue  # Pier also writes a job-level result.json.
        if result["task_name"] not in tasks:
            raise ValueError(f"result references a task outside the RSI catalog: {path}")
        if result.get("step_results"):
            raise ValueError(f"multi-step Pier results are not part of the released RSI contract: {path}")
        task = tasks[result["task_name"]]
        identity = result["id"]
        if identity in seen:
            raise ValueError(f"duplicate Pier trial ID: {identity}")
        seen.add(identity)
        verifier = path.parent / "verifier"
        reward_path = verifier / "reward.json"
        reward_text = verifier / "reward.txt"
        metrics_path = verifier / "metrics.json"
        rewards = (result.get("verifier_result") or {}).get("rewards")
        evidence_paths = [path]
        # Pier gives reward.txt precedence when both are present (void sentinels).
        if reward_text.is_file():
            text_rewards = {"reward": float(reward_text.read_text(encoding="utf-8").strip())}
            if text_rewards["reward"] != -1 and rewards is not None and rewards != text_rewards:
                raise ValueError(f"Pier result and verifier text reward disagree: {path}")
            rewards = text_rewards
            evidence_paths.append(reward_text)
        elif reward_path.is_file():
            raw_rewards = read_json(reward_path)
            if rewards is not None and rewards != raw_rewards:
                raise ValueError(f"Pier result and verifier reward disagree: {path}")
            rewards = raw_rewards
            evidence_paths.append(reward_path)
        metrics = read_json(metrics_path) if metrics_path.is_file() else {}
        if metrics_path.is_file():
            evidence_paths.append(metrics_path)
        agent = result["agent_info"]
        config = result["config"]
        rows.append({
            "id": identity, "task_id": task["id"], "category": task["category"],
            "track": task["track"], "engine": task["engine"],
            "agent": agent["name"], "agent_version": agent.get("version"),
            "model": (agent.get("model_info") or {}).get("name"),
            # Only the digest is exported; config may contain credentials. Job
            # identity/output paths are excluded so repetitions remain comparable.
            "execution_config_sha256": digest({key: config.get(key) for key in EXECUTION_FIELDS}),
            "task_checksum": result["task_checksum"],
            "source_result": str(path.resolve()),
            "evidence_sha256": {
                str(p.relative_to(path.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in evidence_paths
            },
            "rewards": rewards, "usage": result.get("agent_result"),
            **classify(task, result, rewards, metrics),
        })
    if not rows:
        raise ValueError(f"no Pier trial results found: {root}")
    return rows


def summarize_trials(rows: list[dict], tasks: list[dict], *, expected_attempts: int = 1) -> dict:
    """Equal task weight, then equal valid attempt weight in each task.

    Infrastructure/unknown outcomes retain coverage counts. Official scores are
    withheld until every selected task has exactly its planned valid attempts.
    """
    if expected_attempts < 1:
        raise ValueError("expected_attempts must be positive")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_scores = []
    for task in tasks:
        attempts = by_task[task["id"]]
        checksums = {row["task_checksum"] for row in attempts}
        if len(checksums) > 1:
            raise ValueError(f"mixed task revisions: {task['id']}")
        values = [row["score"] for row in attempts if row["score"] is not None]
        aggregate = statistics.mean if task["track"] == "model_support" else statistics.geometric_mean
        task_scores.append({
            "task_id": task["id"], "category": task["category"], "track": task["track"],
            "attempts": len(attempts), "valid_attempts": len(values),
            "expected_attempts": expected_attempts,
            "statuses": dict(Counter(row["status"] for row in attempts)),
            "score": aggregate(values) if values else None,
            "complete": len(values) == expected_attempts and len(attempts) == expected_attempts,
        })
    tracks = {}
    for track in ("model_support", "optimization"):
        selected = [row for row in task_scores if row["track"] == track]
        values = [row["score"] for row in selected if row["score"] is not None]
        complete = bool(selected) and all(row["complete"] for row in selected)
        aggregate = statistics.mean if track == "model_support" else statistics.geometric_mean
        observed = aggregate(values) if values else None
        tracks[track] = {
            "metric": "pass@1" if track == "model_support" else "speedup_geomean",
            "score": observed if complete else None,
            "observed_score": observed, "complete": complete,
            "tasks_planned": len(selected), "tasks_observed": len(values),
        }
    return {"tracks": tracks, "tasks": task_scores, "statuses": dict(Counter(r["status"] for r in rows))}


def score_results(root: Path, inventory: dict, *, task_ids: list[str] | None = None,
                  expected_attempts: int = 1) -> dict:
    tasks = inventory["tasks"]
    if task_ids:
        unknown = set(task_ids) - {item["id"] for item in tasks}
        if unknown:
            raise ValueError(f"unknown selected tasks: {sorted(unknown)}")
        tasks = [item for item in tasks if item["id"] in task_ids]
    rows = load_trials(root, inventory)
    if set(row["task_id"] for row in rows) - {item["id"] for item in tasks}:
        raise ValueError("input contains tasks outside the declared scoring selection")
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[(row["agent"], row["agent_version"], row["model"], row["execution_config_sha256"])].append(row)
    return {
        "schema": "mindie-rsi-score/1", "upstream": inventory["upstream"],
        "catalog_source_commit": inventory["source_commit"],
        "catalog_sha256": inventory["catalog_sha256"],
        "catalog_source_dirty": inventory["source_dirty"],
        "scope": "selected_tasks" if task_ids else "full_catalog",
        "execution": "external_pier", "ablation_group": None,
        "provenance": "local receipts; catalog revision does not attest the original execution",
        "cohorts": [
            {"agent": key[0], "agent_version": key[1], "model": key[2], "execution_config_sha256": key[3],
             **summarize_trials(values, tasks, expected_attempts=expected_attempts)}
            for key, values in sorted(cohorts.items(), key=lambda item: str(item[0]))
        ],
        "trials": rows,
    }
