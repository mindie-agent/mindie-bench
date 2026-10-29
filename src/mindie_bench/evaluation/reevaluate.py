"""Rejudge archived candidates without invoking a model or rewriting receipts.

[Role] Restore saved local/remote candidates and run the current independent oracle.
[Invariants] Historical status and resource meters are immutable; replay is a separate
CPU behavior observation. Remote candidates come from remote artifacts, never local mirrors.
[Timeouts & Locks] Git operations have 30s budgets; judges use one common timeout per
run plus 7s owned-process cleanup. Candidates are restored and evaluated sequentially.
[Side-Effect Policy] Reads original artifacts; writes only the selected new report directory
and disposable workspaces. No network, model requests or device performance claims.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mindie_bench.runner.termination import terminate_process_group
from mindie_bench.runner.verifier import parse_verifier

AFFECTED_SCOPES = {
    "ast_only_oracle", "ast_and_focused_cpu_function", "g4_tensor_or_proxy_ablation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    # Remote artifacts have a 32 MiB raw bound; refuse an unbounded JSON load.
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(f"JSON artifact exceeds 50 MiB: {path}")
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verifier_identity(task: Path) -> dict:
    files = {p.relative_to(task).as_posix(): sha256(p)
             for p in sorted((task / "verifier").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    return {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def restore_candidate(row: dict, receipt: dict, task: Path, workspace: Path) -> dict:
    """Restore exactly the judged side; a local L3 source mirror is not authoritative."""
    trial = Path(row["receipt_path"]).parent
    if row["tier"] == "l3":
        cleanup_path, patch_path = trial / "remote_cleanup.json", trial / "remote_final.patch"
        cleanup = read_json(cleanup_path)["cleanup"]
        if not {"patch_b64", "untracked", "files_modified"} <= cleanup.keys():
            raise ValueError("remote snapshot lacks its completed file inventory")
        patch_bytes = base64.b64decode(cleanup["patch_b64"], validate=True)
        if patch_path.read_bytes() != patch_bytes:
            raise ValueError("remote patch disagrees with the saved cleanup artifact")
        spec = tomllib.loads((task / "task.toml").read_text())
        environment = spec["environment"][row["tier"]]
        base = receipt["environment_fingerprint"]["base_commit"]
        with tempfile.TemporaryFile() as archive:
            subprocess.run(["git", "-C", environment["repo_path"], "archive", base],
                           stdout=archive, stderr=subprocess.PIPE, check=True, timeout=30)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as source:
                source.extractall(workspace, filter="data")
        if patch_bytes:
            subprocess.run(["git", "apply", "--no-index", "--binary", str(patch_path)],
                           cwd=workspace, check=True, capture_output=True, timeout=30)
        for name, encoded in cleanup["untracked"].items():
            target = workspace / name
            if not target.resolve().is_relative_to(workspace.resolve()):
                raise ValueError(f"remote artifact escapes workspace: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(encoded, validate=True))
        return {"kind": "remote_patch_and_untracked", "base_commit": base,
                "patch_sha256": sha256(patch_path), "cleanup_sha256": sha256(cleanup_path),
                "untracked_files": sorted(cleanup["untracked"]),
                "snapshot_errors": cleanup.get("errors", [])}
    archive = trial / "final_workspace.tar.gz"
    with tarfile.open(archive) as source:
        source.extractall(workspace, filter="data")
    # Worktree metadata points outside the artifact and is not candidate source.
    (workspace / ".git").unlink(missing_ok=True)
    return {"kind": "local_workspace_archive", "archive_sha256": sha256(archive)}


def run_verifier(task: Path, workspace: Path, output: Path, python: Path, timeout: float) -> dict:
    before = verifier_identity(task)
    env = dict(os.environ, MINDIE_BENCH_WORKSPACE=str(workspace),
               MINDIE_BENCH_PYTHON=str(python), MINDIE_BENCH_CPU_PYTHON=str(python),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    timed_out = False
    with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(["bash", str(task / "verifier/test.sh")], cwd=workspace,
                                   env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
    result = {"status": "not_evaluated", "reward": None, "failure_category": None,
              "failure_reason": None, "returncode": process.returncode,
              "elapsed_sec": time.monotonic() - started, "timeout_sec": timeout,
              "verifier": before, "python_executable": str(python),
              "measurement_scope": "cpu_behavior_only",
              "native_kernel_bound": None, "measured_npu_speedup": None,
              "speedup_ratio": None, "tpot_ms": None, "npu_card_seconds": None}
    if before != verifier_identity(task):
        return dict(result, failure_category="verifier_changed_during_run",
                    failure_reason="current oracle bytes changed during re-evaluation")
    if timed_out:
        return dict(result, failure_category="verifier_timeout", failure_reason="judge timed out")
    payload = None
    # Process output is streamed to disk; only JSON verdict lines enter the report.
    with (output / "stdout.txt").open(errors="replace") as stream:
        for line in stream:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "reward" in candidate:
                payload = candidate
    if payload is None or payload["reward"] not in (0.0, 1.0):
        return dict(result, failure_category="invalid_verifier_output",
                    failure_reason="no binary reward in verifier output")
    try:
        validated = parse_verifier(process.returncode, json.dumps(payload))
    except ValueError as exc:
        return dict(result, failure_category="invalid_verifier_output",
                    failure_reason=str(exc), verdict=payload)
    reward = validated.reward
    reason = payload.get("failure_reason")
    # A loader limitation must remain visible instead of being called a business failure.
    error_type = payload.get("error_type", payload.get("failure_type"))
    category = None if reward == 1.0 else "behavior_assertion_failed"
    incomplete = False
    if payload.get("verification_status") in {"unmeasured", "unsupported"}:
        category, incomplete = "unsupported_cpu_execution", True
    elif reward == 0.0 and error_type in {"ImportError", "ModuleNotFoundError"}:
        category, incomplete = "dependency_unavailable", True
    elif reward == 0.0 and (error_type == "NameError"
                           or (isinstance(reason, str) and "is not defined" in reason)):
        category, incomplete = "oracle_loading_error", True
    return dict(result, status="not_evaluated" if incomplete else "completed",
                reward=None if incomplete else reward,
                failure_category=category, failure_reason=reason, verdict=payload)


def comparisons(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["selected_for_reevaluation"]:
            groups[(row["task_id"], row["harness"])].append(row)
    result = []
    for (task, harness), values in sorted(groups.items()):
        budgets = [v["original_budget"] for v in values]
        identities = [v["original_verifier_sha256"] for v in values]
        environments = [v["original_controls"] for v in values]
        result.append({
            "task_id": task, "harness": harness,
            "trials": [{"trial_id": v["trial_id"], "group": v["group"],
                        "original_status": v["original_status"],
                        "rejudged_reward": v["reevaluation"]["reward"]} for v in values],
            "original_verifier_equal": len(set(identities)) == 1,
            "original_budget_equal": all(x == budgets[0] for x in budgets),
            "original_controls_equal": all(x == environments[0] for x in environments),
            "common_new_verifier_bundle_sha256": values[0]["new_verifier_bundle_sha256"]
                if len({v["new_verifier_bundle_sha256"] for v in values}) == 1 else None,
            "causal_performance_comparison": "not_established_by_zero_model_replay",
            "speedup_ratio": None,
        })
    return result


def reevaluate(inventory: Path, tasks: Path, output: Path, python: Path,
               timeout: float = 45.0, include_tasks: tuple[str, ...] = (),
               task_pythons: dict[str, Path] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    evaluator_sha256 = sha256(Path(__file__))
    audit = read_json(inventory)
    task_pythons = task_pythons or {}
    rows = []
    for item in audit["trials"]:
        if not item["primary"]:
            continue
        source = Path(item["receipt_path"])
        receipt_hash = sha256(source)
        receipt = read_json(source)
        task = tasks / item["task_id"]
        selected = item["oracle_scope"] in AFFECTED_SCOPES or item["task_id"] in include_tasks
        identity = verifier_identity(task)
        original_files = receipt["auxiliary_metrics"]["oracle_manifest"]["files"]
        fingerprint = receipt["environment_fingerprint"]
        row = {"trial_id": item["trial_id"], "task_id": item["task_id"],
               "harness": item["harness"], "group": item["group"], "tier": item["tier"],
               "original_scope": item["oracle_scope"], "selected_for_reevaluation": selected,
               "selection_reason": ("audited_weak_or_proxy_oracle"
                   if item["oracle_scope"] in AFFECTED_SCOPES else "additional_revised_oracle"
                   if selected else "outside_replay_scope"),
               "receipt_path": str(source), "receipt_sha256": receipt_hash,
               "original_status": receipt["terminal_status"],
               "original_resolved": receipt["primary_metrics"].get("resolved"),
               "original_reward": receipt["primary_metrics"].get("reward"),
               "original_verifier_sha256": original_files.get("verifier/test.sh"),
               "original_verifier_files": original_files,
               "new_verifier_sha256": identity["files"]["verifier/test.sh"],
               "new_verifier_bundle_sha256": identity["sha256"],
               "new_verifier_files": identity["files"],
               "original_budget": fingerprint.get("budget"),
               "original_controls": {key: fingerprint.get(key) for key in (
                   "base_commit", "base_model", "reasoning_effort", "benchmark_source_sha256",
                   "harness_version", "remote_environment")},
               "reevaluation": {"status": "not_selected", "reward": None,
                                "failure_category": None, "failure_reason": None}}
        if selected:
            trial_output = output / item["trial_id"]
            trial_output.mkdir()
            try:
                if receipt_hash != item["receipt_sha256"]:
                    raise ValueError("receipt hash differs from audited source")
                with tempfile.TemporaryDirectory(prefix="mindie-reevaluate-") as directory:
                    workspace = Path(directory)
                    row["candidate_source"] = restore_candidate(item, receipt, task, workspace)
                    row["reevaluation"] = run_verifier(
                        task, workspace, trial_output, task_pythons.get(item["task_id"], python), timeout)
            except Exception as exc:
                row["reevaluation"] = {"status": "not_evaluated", "reward": None,
                                       "failure_category": "artifact_or_execution_error",
                                       "failure_reason": f"{type(exc).__name__}: {exc}"}
            print(item["trial_id"], row["reevaluation"]["reward"],
                  row["reevaluation"]["failure_category"], flush=True)
        row["original_receipt_unchanged"] = receipt_hash == sha256(source)
        if not row["original_receipt_unchanged"]:
            raise RuntimeError(f"original receipt changed during replay: {source}")
        if selected:
            write_json(trial_output / "result.json", row)
        rows.append(row)
    selected = [r for r in rows if r["selected_for_reevaluation"]]
    result = {
        "schema_version": "mindie-bench-reevaluation/1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory_path": str(inventory), "inventory_sha256": sha256(inventory),
        "model_calls": 0, "scope": "saved candidate CPU behavior; no new agent or NPU experiment",
        "evaluator_source_sha256": evaluator_sha256,
        "python": str(python), "task_python_overrides": {k: str(v) for k, v in task_pythons.items()},
        "verifier_timeout_sec": timeout,
        "summary": {"primary_trials": len(rows), "selected_trials": len(selected),
                    "original_statuses": dict(Counter(r["original_status"] for r in rows)),
                    "selected_original_statuses": dict(Counter(r["original_status"] for r in selected)),
                    "rejudged_rewards": dict(Counter(str(r["reevaluation"]["reward"]) for r in selected)),
                    "failure_categories": dict(Counter(r["reevaluation"]["failure_category"]
                        for r in selected if r["reevaluation"]["failure_category"])),
                    "original_receipts_unchanged": all(r["original_receipt_unchanged"] for r in rows)},
        "comparisons": comparisons(rows), "trials": rows,
        "limitations": [
            "Original unknown and infrastructure outcomes are not rewritten by a later functional pass.",
            "Original time/token/cost meters describe historical execution, not execution under this oracle.",
            "Equal replay oracle does not repair different historical budgets, environments or runner versions.",
            "CPU/shim correctness and ablation contracts do not establish native binding or NPU speedup.",
        ],
    }
    write_json(output / "reevaluation.json", result)
    (output / "README.md").write_text(markdown(result))
    return result


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = ["# Saved candidate re-evaluation", "",
             f"Inventory: {summary['primary_trials']} primary trials; "
             f"re-evaluated: {summary['selected_trials']}; model calls: {report['model_calls']}.", "",
             f"Rejudged rewards: `{json.dumps(summary['rejudged_rewards'], sort_keys=True)}`.", "",
             *[f"- {item}" for item in report["limitations"]], "",
             "| Trial | Original status | New reward | Failure category |",
             "| --- | --- | ---: | --- |"]
    for row in report["trials"]:
        if row["selected_for_reevaluation"]:
            verdict = row["reevaluation"]
            reward = verdict["reward"] if verdict["reward"] is not None else "unmeasured"
            lines.append(f"| {row['trial_id']} | {row['original_status']} | {reward} | "
                         f"{verdict['failure_category'] or ''} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--include-task", action="append", default=[])
    parser.add_argument("--task-python", action="append", default=[], metavar="TASK=PYTHON")
    args = parser.parse_args()
    reevaluate(args.inventory.resolve(), args.tasks.resolve(), args.output.resolve(),
               args.python.absolute(), args.timeout, tuple(args.include_task),
               {name: Path(path).absolute() for name, path in
                (binding.split("=", 1) for binding in args.task_python)})


if __name__ == "__main__":
    main()
