"""[Role] Bind new product cells to native trial execution and observation.
[Invariants] Product profiles, config bytes and runtime evidence are bound
before execution. The explicit G mode is only an internal capability setting;
no previously executed component receipt can enter through this executor.
[Timeouts & Locks] Existing run_trial owns model, process, verifier and locks.
[Side-Effect Policy] Starts only explicitly requested cells and writes their
observation beside the original trial receipt. No automatic retries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mindie_bench.io import write_json
from mindie_bench.models import ExperimentConfig, Group, TrialReceipt
from mindie_bench.runner.matrix import observation_from_receipt
from mindie_bench.runner.matrix_contracts import (
    ProductCell,
    ProductGroup,
    ProductObservation,
    ProductProfile,
)
from mindie_bench.runner.orchestrator import run_trial
from mindie_bench.runner.p3_lineage import P3LineageRejected, budget_ledger_mismatches
from mindie_bench.runner.product_inputs import runtime_snapshot


@dataclass(frozen=True)
class ProductExecutionBinding:
    """One independently prepared runtime manifest and native configuration."""

    profile: ProductProfile
    config: ExperimentConfig
    capability_mode: Group
    runtime_manifest: Path
    execution_tier: str
    manifest: dict = field(init=False, repr=False)
    runtime: dict = field(init=False, repr=False)

    def __post_init__(self):
        manifest = json.loads(self.runtime_manifest.read_text())
        runtime = runtime_snapshot(self.config, self.capability_mode)
        if runtime != manifest["runtime_snapshot"]:
            raise ValueError("installed product runtime differs from its prepared source identity")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "runtime", runtime)

    def context(self, cell: ProductCell, task_root: Path) -> dict:
        profile = self.profile
        if (cell.group, cell.profile_id, cell.cohort, cell.protocol) != (
            profile.group,
            profile.profile_id,
            profile.cohort,
            profile.protocol,
        ):
            raise ValueError("product cell differs from its independently bound profile")
        manifest = self.manifest
        if self.config.harness.model != profile.conditions.model_id:
            raise ValueError("product configuration model differs from the bound profile")
        if manifest["profile_id"] != profile.profile_id or manifest["product_group"] != profile.group:
            raise ValueError("runtime manifest belongs to another product profile")
        if profile.experiment_profile != str(self.runtime_manifest):
            raise ValueError("product profile does not reference this runtime manifest")
        derived = manifest["derived_inputs"]
        derived_task = derived["tasks"].get(cell.task_id)
        if derived_task is None:
            raise ValueError("product cell and task identity differ")
        derived_budget = derived["budget"]
        if profile.conditions.to_dict() != derived["conditions"] or any(
            derived_budget.get(name) != value
            for name, value in profile.budget.task_derived_dict().items()
        ):
            raise ValueError(
                "product matching conditions or matched downstream budget differ from the "
                "actual native inputs"
            )
        p3 = None
        if profile.group == ProductGroup.P3_V003_FLYWHEEL:
            if profile.p3_lineage is None:
                raise ValueError("P3 product execution requires a closed-family lineage admission")
            target = profile.p3_lineage.manifest.target
            if target.matrix_task_family is not None and target.matrix_task_family != cell.task_family:
                raise ValueError("P3 lineage matrix task_family label conflicts with the product cell")
            if target.hardware_tier is not None and target.hardware_tier != cell.hardware_tier:
                raise ValueError("P3 lineage hardware tier conflicts with the product cell")
            try:
                p3 = profile.p3_lineage.admit(
                    profile_id=profile.profile_id,
                    product_group=profile.group.value,
                    cohort=profile.cohort,
                    protocol=profile.protocol,
                    task_id=cell.task_id,
                    task_root=task_root,
                    verify_artifacts=True,
                )
            except P3LineageRejected as error:
                raise ValueError("P3 lineage admission failed before native execution") from error
            mismatches = budget_ledger_mismatches(
                p3["cost_ledger"], profile.budget.ledger_dict()
            )
            if mismatches:
                raise ValueError(
                    "P3 declared prerequisite budget differs from the cost ledger: "
                    + "; ".join(mismatches)
                )
            if (
                p3["heldout"]["task_sha256"] != derived_task["task_sha256"]
                or p3["heldout"]["oracle_sha256"] != derived_task["oracle_sha256"]
            ):
                raise ValueError("P3 Heldout task or oracle differs from the actual native inputs")
        elif profile.p3_lineage is not None:
            raise ValueError("only the P3 product group may carry a P3 lineage admission")
        runtime = self.runtime
        return {
            "schema": "mindie-bench-product-execution/1",
            "cell": cell.to_dict(),
            "profile": profile.to_dict(),
            "runtime_manifest": str(self.runtime_manifest),
            "internal_capability_mode": self.capability_mode.value,
            "task_sha256": derived_task["task_sha256"],
            "oracle_sha256": derived_task["oracle_sha256"],
            "execution_tier": self.execution_tier,
            "actual_runtime": runtime,
            "max_wall_clock_sec": profile.budget.max_wall_clock_sec,
            **({"p3_admission": p3} if p3 is not None else {}),
        }


class ProductExecutor:
    """Native executor for the already planned product cells.

    Runtime manifests are reviewable preparation artifacts, not an alternate
    attempt state machine. run_trial remains the sole attempt/receipt owner.
    """

    def __init__(
        self,
        bindings: dict[str, ProductExecutionBinding],
        tasks: dict[str, Path],
        run_root: Path,
        *,
        devices: list[int] | None = None,
        execute_trial: Callable[..., TrialReceipt] = run_trial,
    ):
        self.bindings = bindings
        self.tasks = tasks
        self.run_root = run_root
        self.devices = devices
        self.execute_trial = execute_trial

    def __call__(self, cell: ProductCell) -> ProductObservation:
        binding = self.bindings[cell.profile_id]
        task_root = self.tasks[cell.task_id]
        context = binding.context(cell, task_root)
        receipt = self.execute_trial(
            task_root,
            binding.config,
            tier=binding.execution_tier,
            group=binding.capability_mode,
            repetition=cell.repetition,
            run_root=self.run_root,
            devices=self.devices,
            experiment_context=context,
            expected_task_digest=context["task_sha256"],
            expected_oracle_sha256=context["oracle_sha256"],
        )
        if receipt.environment_fingerprint.get("experiment_context") != context:
            raise ValueError("trial did not retain its pre-execution product binding")
        if receipt.task_digest != context["task_sha256"]:
            raise ValueError("trial used a different task")
        observation = observation_from_receipt(receipt, cell=cell)
        trial_dir = Path(receipt.artifacts["private_trial_dir"])
        write_json(
            trial_dir / "product_observation.json",
            {
                "schema": "mindie-bench-product-observation/1",
                "experiment_context": context,
                "trial_receipt": str(trial_dir / "trial_receipt.json"),
                "observation": observation.to_dict(),
            },
            exclusive=True,
        )
        return observation
