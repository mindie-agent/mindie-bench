"""Lifecycle scores from native MindIE receipts and the original matrix plan.

[Role] Add task-balanced support and optimization tracks to native summaries.
[Invariants] Preserve G0-G3, evidence/role/cohort boundaries and invalid outcomes.
[Timeouts & Locks] Pure computation; no filesystem, subprocesses or locks.
[Side-Effect Policy] Recompute from evidence without editing historical receipts.
"""

from collections import Counter, defaultdict
from statistics import geometric_mean, mean

from mindie_bench.evaluation.accounting import cell_key, receipt_cell
from mindie_bench.evaluation.inference import evaluate_inference
from mindie_bench.inference_models import InferenceContract, InferenceEvidence
from mindie_bench.io import digest
from mindie_bench.models import TerminalStatus

VALID = {TerminalStatus.PASS, TerminalStatus.FAIL}


def _observation(receipt, contract):
    if receipt.terminal_status not in VALID:
        return None
    raw = receipt.primary_metrics.get("performance_metrics", {}).get("inference_evidence")
    if raw is None:
        if receipt.terminal_status == TerminalStatus.PASS:
            raise ValueError("successful inference receipt has no judge evidence")
        return None
    result = evaluate_inference(contract, InferenceEvidence.model_validate(raw))
    if receipt.terminal_status == TerminalStatus.PASS and not result["accepted"]:
        raise ValueError("successful native receipt contradicts inference gates")
    if receipt.terminal_status == TerminalStatus.FAIL and result["accepted"]:
        # Other native gates (ACLGraph, workspace, final process state) still
        # reject the candidate. A fast but rejected patch is never a scored gain.
        result = dict(result, score=0.0 if contract.category == "model_support" else 1.0,
                      accepted=False, failed_gates=["native_acceptance_failed"])
    return result


def summarize_inference(receipts, *, expected_cells=None):
    selected = [row for row in receipts if row.environment_fingerprint.get("inference_contract")]
    declared = [cell for cell in (expected_cells or []) if cell.get("inference_category")]
    if not selected and not declared:
        return None
    identities = [cell_key(receipt_cell(row)) for row in selected]
    if len(set(identities)) != len(identities) or len({r.trial_id for r in selected}) != len(selected):
        raise ValueError("duplicate native inference trials or matrix cells")
    # Different model/config/evidence/role populations must not silently merge.
    cohorts = defaultdict(list)
    for row in selected:
        fingerprint = row.environment_fingerprint
        key = (row.tier, row.evidence, row.ablation_group.value, row.task_role,
               row.experiment_digest, digest({name: fingerprint.get(name) for name in (
                   "harness_name", "harness_version", "base_model", "reasoning_effort",
                   "backend", "benchmark_source_sha256",
               )}))
        cohorts[key].append(row)
    reports = []
    for key, rows in sorted(cohorts.items()):
        planned = [cell for cell in declared if (
            cell["tier"], cell["group"], cell.get("role", "eval")
        ) == (key[0], key[2], key[3])]
        expected = {cell_key(cell): cell for cell in planned}
        if len(expected) != len(planned):
            raise ValueError("duplicate planned inference cells")
        by_task = defaultdict(list)
        categories = {cell["instance_id"]: cell["inference_category"] for cell in planned}
        for row in rows:
            contract = InferenceContract.model_validate(row.environment_fingerprint["inference_contract"])
            if row.instance_id in categories and categories[row.instance_id] != contract.category:
                raise ValueError("inference lifecycle category differs from its declared plan")
            categories[row.instance_id] = contract.category
            by_task[row.instance_id].append((row, contract, _observation(row, contract)))
        tasks = []
        for task_id, category in sorted(categories.items()):
            attempts = by_task[task_id]
            if len({(row.task_digest, digest(contract.model_dump(mode="json")))
                    for row, contract, _ in attempts}) > 1:
                raise ValueError("mixed inference task revisions or contracts")
            expected_task = {k: cell for k, cell in expected.items() if cell["instance_id"] == task_id}
            observed = {cell_key(receipt_cell(row)) for row, _, _ in attempts}
            for row, _, _ in attempts:
                cell = expected_task.get(cell_key(receipt_cell(row)))
                if cell and cell.get("task_digest") != row.task_digest:
                    raise ValueError("inference task digest differs from its declared plan")
            values = [outcome["score"] for _, _, outcome in attempts if outcome is not None]
            aggregate = mean if category == "model_support" else geometric_mean
            tasks.append(dict(
                instance_id=task_id, category=category, attempts=len(attempts), valid_attempts=len(values),
                planned_attempts=len(expected_task) if expected_cells is not None else None,
                statuses=dict(Counter(row.terminal_status.value for row, _, _ in attempts)),
                trials=[dict(trial_id=row.trial_id, status=row.terminal_status.value, result=outcome)
                        for row, _, outcome in attempts],
                observed_score=aggregate(values) if values else None,
                complete=bool(expected_task) and observed == expected_task.keys() and len(values) == len(attempts),
            ))
        tracks = {}
        for name in ("model_support", "optimization"):
            members = [task for task in tasks if (task["category"] == "model_support") == (name == "model_support")]
            values = [task["observed_score"] for task in members if task["observed_score"] is not None]
            aggregate = mean if name == "model_support" else geometric_mean
            observed_score = aggregate(values) if values else None
            complete = bool(members) and all(task["complete"] for task in members)
            tracks[name] = dict(
                metric="task_balanced_pass_at_1" if name == "model_support" else "task_balanced_speedup",
                score=observed_score if complete else None, observed_score=observed_score,
                complete=complete, tasks=len(members), tasks_with_valid_evidence=len(values),
            )
        reports.append(dict(
            tier=key[0], evidence=key[1], group=key[2], role=key[3], experiment_digest=key[4],
            harness_controls_sha256=key[5], tasks=tasks, tracks=tracks,
        ))
    return dict(
        schema_version="mindie-bench-inference-summary/1", cohorts=reports,
        planned_cells=len(declared) if expected_cells is not None else None,
        observed_receipts=len(selected),
        scope="declared_inference_cells" if declared else "observed_tasks_only",
        estimator="equal_task_weight; arithmetic_support; geometric_speedup; paired_median_within_workload",
    )
