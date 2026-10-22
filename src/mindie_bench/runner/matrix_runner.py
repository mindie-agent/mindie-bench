"""Finite ablation matrix orchestration.

[Role]
Admits, orders, and accounts independent G0..G2 cells and sequential G3
lineages while preserving every completed cell receipt.
[Invariants]
- A G3 seed must pass its oracle and corpus lint before dependent evals start.
- Canary failures block remaining cells; independent admitted cells still finish.
- Blocked cells stay explicit in summary/accounting instead of disappearing.
[Timeouts & Locks]
- Cell concurrency is caller-bounded; NPU devices are leased under a condition
  variable; `threading.Lock` only guards receipt/blocked-cell appends.
[Side-Effect Policy]
- Creates the run root exclusively and writes only matrix-level manifests,
  summaries, accounting, warm corpora, and delegated trial directories.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mindie_bench.evaluation.budget import has_non_budget_errors
from mindie_bench.io import read_json, task_hash, tree_hash, write_json
from mindie_bench.models import ExperimentConfig, Group, TaskSpec, TerminalStatus, TrialReceipt
from mindie_bench.runner.profile_builder import filtered_corpus
from mindie_bench.scorer.comparator import grouped_summaries


def run_matrix(
    task_roots: list[Path],
    config: ExperimentConfig,
    *,
    tier: str,
    groups: list[Group],
    repetitions: int,
    concurrency: int,
    output: Path,
    canary_first: bool = False,
) -> list[TrialReceipt]:
    from mindie_bench.runner import orchestrator as _orchestrator

    as_completed = _orchestrator.as_completed
    if concurrency < 1 or repetitions < 1 or not groups or len(set(groups)) != len(groups):
        raise ValueError("positive concurrency/repetitions and unique groups are required")
    task_roots = [path.resolve(strict=True) for path in task_roots]
    output = output.resolve()
    specs = {path: TaskSpec.load(path) for path in task_roots}
    if not specs or len(specs) != len(task_roots):
        raise ValueError("matrix task paths must be nonempty and unique")
    if len({spec.task.instance_id for spec in specs.values()}) != len(specs):
        raise ValueError("matrix task instance IDs must be unique")
    lineages = {}
    if Group.G3 in groups:
        if not any(spec.task.role == "seed" for spec in specs.values()):
            raise ValueError(
                "G3 requires seed tasks followed by eval tasks, not an unproven warm snapshot"
            )
        for path, spec in specs.items():
            cluster, corpus_id = spec.task.cluster, spec.mindie_knowledge.corpus_id
            if not cluster or corpus_id not in config.corpora:
                raise ValueError("G3 requires explicit transfer clusters and corpus IDs")
            lineages.setdefault((cluster, corpus_id), []).append(path)
        if not lineages or any(
            not any(specs[path].task.role == "seed" for path in paths) for paths in lineages.values()
        ):
            raise ValueError("G3 requires seed tasks in each lineage, not an unproven warm snapshot")
        for paths in lineages.values():
            paths.sort(key=lambda path: specs[path].task.role != "seed")
    if tier == "l3":
        if not config.remote:
            raise ValueError("L3 requires explicit remote configuration")
        for spec in specs.values():
            count = spec.environment[tier].npu_count
            if count < 1 or count > len(config.remote.devices):
                raise ValueError("each L3 task must fit the explicitly configured NPU device pool")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    task_digests = {path: task_hash(path) for path in task_roots}
    cells = [
        dict(
            instance_id=spec.task.instance_id,
            tier=tier,
            group=group.value,
            repetition=repetition,
            role=spec.task.role,
            task_digest=task_digests[path],
            cluster=spec.task.cluster,
            corpus_id=spec.mindie_knowledge.corpus_id,
            inference_category=spec.inference.category if spec.inference else None,
            evidence=spec.task.evidence,
        )
        for path, spec in specs.items()
        for group in groups
        for repetition in range(1, repetitions + 1)
    ]
    write_json(
        output / "eval_manifest.json",
        dict(
            schema_version="1.0",
            created_unix=time.time(),
            experiment=config.model_dump(mode="json"),
            tasks=[dict(path=str(p), sha256=tree_hash(p)) for p in task_roots],
            tier=tier,
            groups=[g.value for g in groups],
            repetitions=repetitions,
            automatic_retries=0,
            cells=cells,
        ),
        exclusive=True,
    )
    receipts = []
    blocked_cells = []
    receipt_lock = threading.Lock()
    device_condition = threading.Condition()
    cancelled = threading.Event()
    available_devices = set(config.remote.devices) if tier == "l3" else set()
    excluded = {
        identity for spec in specs.values() for identity in spec.mindie_knowledge.same_origin_task_hashes
    }

    def execute_cell(path, group, repetition, *, corpus_override=None):
        from mindie_bench.runner.orchestrator import run_trial

        devices = []
        with device_condition:
            if tier == "l3":
                count = specs[path].environment[tier].npu_count
                device_condition.wait_for(lambda: cancelled.is_set() or len(available_devices) >= count)
            if cancelled.is_set():
                raise RuntimeError("matrix cancelled before this trial was admitted")
            if tier == "l3":
                devices = sorted(available_devices)[:count]
                available_devices.difference_update(devices)
        try:
            receipt = run_trial(
                path,
                config,
                tier=tier,
                group=group,
                repetition=repetition,
                run_root=output,
                devices=devices or None,
                corpus_override=corpus_override,
                cancelled=cancelled,
            )
            # Retain completed cells even when a later capture in this lineage fails.
            with receipt_lock:
                receipts.append(receipt)
            return receipt
        finally:
            with device_condition:
                available_devices.update(devices)
                device_condition.notify_all()

    def execute_lineage(key, paths, repetition):
        cluster, corpus_id = key
        warm = output / "warm-initial" / str(repetition) / cluster / corpus_id
        warm.parent.mkdir(parents=True, exist_ok=True)
        filtered_corpus(Path(config.corpora[corpus_id].path), warm, excluded)
        for index, path in enumerate(paths):
            receipt = execute_cell(path, Group.G3, repetition, corpus_override=warm)
            if specs[path].task.role == "seed":
                if (
                    receipt.terminal_status not in {TerminalStatus.PASS, TerminalStatus.FAIL}
                    or receipt.auxiliary_metrics.get("capture", {}).get("status") != "completed"
                    or receipt.auxiliary_metrics.get("capture", {}).get("corpus_lint", {}).get("status")
                    != "passed"
                ):
                    blocked = [
                        dict(
                            **cell,
                            reason="seed_prerequisite_not_completed",
                            prerequisite_trial_id=receipt.trial_id,
                            prerequisite_status=receipt.terminal_status.value,
                        )
                        for cell in cells
                        if cell["group"] == "G3"
                        and cell["repetition"] == repetition
                        and cell["instance_id"]
                        in {specs[remaining].task.instance_id for remaining in paths[index + 1 :]}
                    ]
                    with receipt_lock:
                        blocked_cells.extend(blocked)
                    raise RuntimeError(
                        "G3 seed prerequisites failed; dependent evals were not started "
                        f"(cluster={cluster}, corpus={corpus_id}, repetition={repetition}, "
                        f"status={receipt.terminal_status.value}, "
                        f"capture={receipt.auxiliary_metrics.get('capture', {}).get('status')})"
                    )
                # Failed engineering attempts are still valid reference experience.
                # Held-out evaluations never become input to later evaluations.
                warm = Path(receipt.artifacts["captured_corpus"])
                if tree_hash(warm) != receipt.auxiliary_metrics["captured_corpus_sha256"]:
                    raise ValueError("captured seed corpus changed before its held-out evaluation")

    jobs = [
        (path, group, repetition)
        for path in task_roots
        for group in groups
        for repetition in range(1, repetitions + 1)
        if group != Group.G3
    ]
    lineage_jobs = [
        (key, paths, repetition)
        for repetition in range(1, repetitions + 1)
        for key, paths in lineages.items()
    ]
    matrix_error = None
    try:
        if canary_first and len(cells) > 1:
            if jobs:
                canary = execute_cell(*jobs.pop(0))
                if (
                    canary.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
                    or has_non_budget_errors(canary)
                ):
                    raise RuntimeError(
                        f"canary gate failed on {canary.trial_id} "
                        f"(status={canary.terminal_status.value}); remaining cells were not admitted"
                    )
            elif lineage_jobs:
                execute_lineage(*lineage_jobs.pop(0))
                bad = next(
                    (
                        row
                        for row in receipts
                        if row.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
                        or has_non_budget_errors(row)
                    ),
                    None,
                )
                if bad is not None:
                    raise RuntimeError(
                        f"canary gate failed on {bad.trial_id} "
                        f"(status={bad.terminal_status.value}); remaining cells were not admitted"
                    )
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            try:
                # Interleave independent lineages and cells so neither class waits
                # behind the complete queue of the other before it can start.
                for index in range(max(len(jobs), len(lineage_jobs))):
                    if index < len(jobs):
                        futures.append(pool.submit(execute_cell, *jobs[index]))
                    if index < len(lineage_jobs):
                        futures.append(pool.submit(execute_lineage, *lineage_jobs[index]))
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        # A failed lineage stops its dependents. Independent
                        # admitted trials can still finish their paid work.
                        if matrix_error is None:
                            matrix_error = error
                        else:
                            matrix_error.add_note(
                                f"another independent matrix job failed: {type(error).__name__}: {error}"
                            )
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                with device_condition:
                    device_condition.notify_all()
                raise
    except BaseException as error:
        matrix_error = error
    try:
        write_json(output / "summary.json", grouped_summaries(
            receipts, expected_cells=cells, blocked_cells=blocked_cells,
        ), exclusive=True)
        from mindie_bench.evaluation.accounting import account_cells

        manifest_path = output / "eval_manifest.json"
        accounting = account_cells(read_json(manifest_path), receipts, blocked_cells=blocked_cells)
        write_json(
            output / "matrix_accounting.json",
            dict(
                schema="mindie-bench-matrix-accounting/1",
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                receipt_sha256={
                    row.trial_id: hashlib.sha256(
                        (Path(row.artifacts["private_trial_dir"]) / "trial_receipt.json").read_bytes()
                    ).hexdigest()
                    for row in receipts
                },
                **accounting,
            ),
            exclusive=True,
        )
    except Exception as error:
        if matrix_error is not None:
            matrix_error.add_note(f"matrix reporting write also failed: {error}")
            raise matrix_error from error
        raise
    if matrix_error is not None:
        raise matrix_error
    return receipts
