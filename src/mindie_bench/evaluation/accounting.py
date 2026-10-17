"""Read incomplete experiments without making them eligible frozen baselines."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from mindie_bench.io import read_json
from mindie_bench.models import TrialReceipt


def cell_key(cell: dict) -> tuple[str, str, str, int]:
    identity, tier, group, repetition = (
        cell["instance_id"],
        cell["tier"],
        cell["group"],
        cell["repetition"],
    )
    if (
        not isinstance(identity, str)
        or not identity
        or tier not in {"l1", "l2", "l3"}
        or group not in {"G0", "G1", "G2", "G3"}
        or type(repetition) is not int
        or repetition < 1
    ):
        raise ValueError("invalid declared experiment cell")
    return identity, tier, group, repetition


def receipt_cell(receipt: TrialReceipt) -> dict:
    return dict(
        instance_id=receipt.instance_id,
        tier=receipt.tier,
        group=receipt.ablation_group.value,
        repetition=receipt.repetition,
        role=receipt.task_role,
        task_digest=receipt.task_digest,
    )


def account_cells(
    manifest: dict, receipts: list[TrialReceipt], *, blocked_cells: list[dict] | None = None
) -> dict:
    declared = manifest.get("cells")
    expected_count = len(manifest["tasks"]) * len(manifest["groups"]) * manifest["repetitions"]
    by_key = {cell_key(receipt_cell(row)): row for row in receipts}
    if len(by_key) != len(receipts) or len({row.trial_id for row in receipts}) != len(receipts):
        raise ValueError("duplicate trial identities or experiment cells")
    if len(receipts) > expected_count:
        raise ValueError("more terminal receipts than declared experiment cells")
    blocked = blocked_cells or []
    blocked_keys = {cell_key(cell) for cell in blocked}
    if len(blocked_keys) != len(blocked):
        raise ValueError("duplicate blocked experiment cells")
    if blocked_keys & by_key.keys():
        raise ValueError("a blocked cell already has a terminal receipt")
    missing = None
    if declared is not None:
        expected = {cell_key(cell): cell for cell in declared}
        if len(expected) != len(declared) or len(expected) != expected_count:
            raise ValueError("explicit cells disagree with the declared experiment matrix")
        if by_key.keys() - expected.keys() or blocked_keys - expected.keys():
            raise ValueError("observed or blocked cell is outside the declared experiment matrix")
        for key, row in by_key.items():
            for field, actual in (("role", row.task_role), ("task_digest", row.task_digest)):
                if field in expected[key] and expected[key][field] != actual:
                    raise ValueError(f"receipt {field} differs from the declared cell")
        by_trial = {row.trial_id: row for row in receipts}
        for cell in blocked:
            key = cell_key(cell)
            original = expected[key]
            for field in ("role", "task_digest", "cluster", "corpus_id"):
                if field in cell and cell[field] != original.get(field):
                    raise ValueError("blocked identity differs from its declared cell")
            seed = by_trial.get(cell.get("prerequisite_trial_id"))
            if (
                seed is None
                or seed.task_role != "seed"
                or seed.ablation_group.value != "G3"
                or cell["group"] != "G3"
                or seed.repetition != cell["repetition"]
                or seed.tier != cell["tier"]
            ):
                raise ValueError("blocked cell requires an actual seed prerequisite receipt")
            seed_cell = expected[cell_key(receipt_cell(seed))]
            if any(
                not original.get(field) or original[field] != seed_cell.get(field)
                for field in ("cluster", "corpus_id")
            ):
                raise ValueError("blocked prerequisite must belong to the same declared lineage")
            prerequisite_failed = (
                seed.terminal_status.value not in {"completed_pass", "completed_fail"}
                or seed.auxiliary_metrics.get("capture", {}).get("status") != "completed"
            )
            if not prerequisite_failed or cell.get("reason") != "seed_prerequisite_not_completed":
                raise ValueError("blocked prerequisite was completed or its reason is invalid")
            if (
                "prerequisite_status" in cell
                and cell["prerequisite_status"] != seed.terminal_status.value
            ):
                raise ValueError("blocked prerequisite status differs from its receipt")
        missing = [expected[key] for key in expected.keys() - by_key.keys() - blocked_keys]
        missing.sort(key=cell_key)
    elif blocked:
        raise ValueError("blocked identities require an explicit original cell inventory")
    missing_count = expected_count - len(receipts) - len(blocked)
    if missing_count < 0:
        raise ValueError("terminal and blocked counts exceed the declared matrix")
    return dict(
        planned_cells=expected_count,
        terminal_receipts=len(receipts),
        raw_statuses=dict(Counter(row.terminal_status.value for row in receipts)),
        blocked_cells=blocked,
        blocked_count=len(blocked) if declared is not None else None,
        missing_cells=missing,
        missing_count=missing_count,
        missing_classification=(
            "explicit_inventory" if declared is not None else "not_adjudicated_from_original_inventory"
        ),
        terminal_accounting_complete=len(receipts) == expected_count,
        all_cells_accounted_for=missing_count == 0,
        all_planned_cells_passed=(
            len(receipts) == expected_count
            and expected_count > 0
            and all(row.terminal_status.value == "completed_pass" for row in receipts)
        ),
    )


def audit_report(root: Path) -> dict:
    """Inspect only direct trial children, retaining missing/unknown evidence."""
    from mindie_bench.scorer.comparator import grouped_summaries

    root = root.resolve(strict=True)
    manifest_path = root / "eval_manifest.json"
    manifest = read_json(manifest_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipts, receipt_hashes, interrupted = [], {}, []
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            raise ValueError("experiment report cannot follow a linked trial directory")
        if not child.is_dir():
            continue
        path, marker = child / "trial_receipt.json", child / "attempt_started.json"
        if path.is_symlink() or marker.is_symlink():
            raise ValueError("experiment evidence cannot be a symlink")
        if path.is_file():
            receipt = TrialReceipt.model_validate(read_json(path))
            receipts.append(receipt)
            receipt_hashes[receipt.trial_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif marker.is_file():
            interrupted.append(
                dict(
                    directory=child.name,
                    marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
                    recorded=read_json(marker),
                    outcome="unknown_no_terminal_receipt",
                )
            )
    blocked = []
    accounting_path = root / "matrix_accounting.json"
    if accounting_path.exists():
        saved = read_json(accounting_path)
        if saved["manifest_sha256"] != manifest_hash:
            raise ValueError("matrix accounting is bound to a different manifest")
        if saved["receipt_sha256"] != receipt_hashes:
            raise ValueError("matrix accounting receipt hashes changed")
        blocked = saved["blocked_cells"]
    accounting = account_cells(manifest, receipts, blocked_cells=blocked)
    accounting["unsettled_attempt_markers"] = len(interrupted)
    accounting["complete_result_eligible"] = (
        accounting["terminal_accounting_complete"] and not interrupted
    )
    if interrupted:
        accounting["terminal_accounting_complete"] = False
        accounting["all_planned_cells_passed"] = False
    return dict(
        schema="mindie-bench-experiment-audit/1",
        role="partial_readback_only_not_a_frozen_baseline",
        manifest_sha256=manifest_hash,
        receipt_sha256=receipt_hashes,
        accounting=accounting,
        interrupted_attempts=interrupted,
        summaries=grouped_summaries(receipts),
        warnings=[
            "Legacy manifests without explicit cells cannot identify why absent cells did not run.",
            "Raw receipt classifications are preserved; external adjudications remain separate.",
            "Recorded elapsed attempt time is not time to a successful delivery.",
        ],
    )
