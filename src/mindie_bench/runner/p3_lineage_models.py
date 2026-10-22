"""Pydantic models for P3 lineage admission and bindings.

[Role]
Defines the exact Seed -> capture -> organizer -> consumer -> freeze -> review ->
downstream metadata chain, admission rejection, and the prepared binding.
[Invariants]
- Every stage is hash-chained to its predecessor; locators are separate from
  logical content identity.
- A binding rejects a manifest from another product group or a mismatched
  admission digest.
[Timeouts & Locks]
- Pure model construction and digesting; no waits, subprocesses or locks.
[Side-Effect Policy]
- No I/O; `P3LineageBinding.prepare` delegates artifact reads to the lineage
  admission function without writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from mindie_bench.evaluation.split_manifest import SplitManifest
from mindie_bench.io import digest
from mindie_bench.runner.p3_receipts import P3CostLedger, P3StageObservation, P3TimeModel
from mindie_bench.runner.protocol_json import JsonRecord

SHA = r"^[0-9a-f]{64}$"
P3_LINEAGE_PROTOCOL = "mindie-bench-p3-lineage/1"
P3_ADMISSION_PROTOCOL = "mindie-bench-p3-admission/1"
_PRODUCT_GROUPS = ("P3_v003_flywheel",)
class P3LineageRejected(ValueError):
    """A P3 cell must not start: the closed-family lineage is not admissible."""
    def __init__(self, reasons: list[str] | tuple[str, ...]):
        self.reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__("P3 lineage rejected: " + ", ".join(self.reasons))
class P3Target(P3TimeModel):
    """The exact product column and downstream task this lineage may admit."""
    profile_id: str = Field(min_length=1)
    product_group: Literal["P3_v003_flywheel"]
    cohort: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    matrix_task_family: str | None = None
    hardware_tier: str | None = None
class P3SeedRun(P3TimeModel):
    trial_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    task_sha256: str = Field(pattern=SHA)
    oracle_sha256: str = Field(pattern=SHA)
    instruction_sha256: str = Field(pattern=SHA)
    terminal_status: Literal["completed_pass", "completed_fail"]
    receipt_locator: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA)
    seed_root_locator: str | None = None
    seed_identity_basis: Literal["verified_bytes", "receipt_recorded"] = "receipt_recorded"
    context_id: str = Field(min_length=1)
    completed_at: datetime | None = None

class P3NativeCapture(P3TimeModel):
    """The real capture container receipt; organizer is a stage inside it."""
    capture_id: str = Field(min_length=1)
    seed_trial_id: str = Field(min_length=1)
    seed_receipt_sha256: str = Field(pattern=SHA)
    native_harness: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    receipt_locator: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA)
    stage_observations: list[P3StageObservation]
    tokens_total: int | None = Field(default=None, ge=0)
    tokens_complete: bool = False
    corpus_lint_status: Literal["passed", "pending", "rejected", "failed"] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

class P3Organizer(P3TimeModel):
    """Projection of the in-capture C6 `graded_distillation` stage."""
    stage_id: Literal["graded_distillation"] = "graded_distillation"
    capture_receipt_sha256: str = Field(pattern=SHA)
    actor_id: str = Field(min_length=1)
    status: Literal["applied", "completed", "failed"]
    model_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    billing_status: str = Field(min_length=1)
    cleanup_failed: bool
    completed_at: datetime | None = None

class P3Consumer(P3TimeModel):
    """Publication/consumer stage inside the capture chain, not the P3 Heldout."""
    consumer_id: str = Field(min_length=1)
    receipt_locator: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA)
    capture_receipt_sha256: str = Field(pattern=SHA)
    trial_receipt_sha256: str = Field(pattern=SHA)
    status: Literal["completed", "failed"]
    model_calls: int | None = Field(default=None, ge=0)
    provider_writes: int | None = Field(default=None, ge=0)
    remote_writes: int | None = Field(default=None, ge=0)
    corpus_sha256: str | None = Field(default=None, pattern=SHA)
    scope: str = Field(min_length=1)
    completed_at: datetime | None = None

class P3FrozenCorpus(P3TimeModel):
    locator: str = Field(min_length=1)
    tree_sha256: str = Field(pattern=SHA)
    unit_count: int = Field(ge=0)
    consumer_receipt_sha256: str = Field(pattern=SHA)
    frozen_at: datetime | None = None

class P3Review(P3TimeModel):
    review_id: str = Field(min_length=1)
    consumer_receipt_sha256: str = Field(pattern=SHA)
    capture_receipt_sha256: str = Field(pattern=SHA)
    status: Literal["accepted"]
    verdict: Literal["distinct_closed_family"]
    independent: bool
    receipt_locator: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA)
    actor_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    seed_family_canonical: str = Field(pattern=SHA)
    heldout_family_canonical: str = Field(pattern=SHA)
    completed_at: datetime | None = None

class P3Downstream(P3TimeModel):
    task_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    task_sha256: str = Field(pattern=SHA)
    oracle_sha256: str = Field(pattern=SHA)
    instruction_sha256: str = Field(pattern=SHA)
    context_id: str = Field(min_length=1)
    review_receipt_sha256: str = Field(pattern=SHA)
    started_at: datetime | None = None

class P3LineageManifest(P3TimeModel):
    """Reviewed metadata for one real Seed -> capture -> C6 -> consumer -> freeze chain."""
    schema_version: Literal["mindie-bench-p3-lineage/1"] = P3_LINEAGE_PROTOCOL
    created_at: datetime
    target: P3Target
    shared_mechanism: str = Field(min_length=1)
    split_manifest: SplitManifest
    assets: list[dict]
    strong_edges: list[dict]
    candidate_edges: list[dict]
    claims: list[dict]
    decisions: list[dict]
    exposures: list[dict] = Field(default_factory=list)
    seed: P3SeedRun
    capture: P3NativeCapture
    organizer: P3Organizer
    consumer: P3Consumer
    frozen_corpus: P3FrozenCorpus
    review: P3Review
    downstream: P3Downstream
    cost_ledger: P3CostLedger
    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))

    def content_identity(self) -> str:
        """Locator-free serialization of the same logical lineage content.

        Machine-specific locator fields are removed at every level; receipt/corpus
        SHA-256 values, source IDs, split assignments, stage ids/times, canonical
        family ids and the cost ledger remain. Locators are retained separately in
        the manifest digest and in admission artifact evidence.
        """
        def strip(value):
            if isinstance(value, dict):
                return {
                    key: strip(item)
                    for key, item in value.items()
                    if key
                    not in {
                        "locator",
                        "receipt_locator",
                        "seed_root_locator",
                        "source_locator",
                        "created_at",
                        "manifest_sha256",
                    }
                }
            if isinstance(value, list):
                return [strip(item) for item in value]
            return value

        return digest(strip(self.model_dump(mode="json")))

@dataclass(frozen=True)
class P3LineageBinding(JsonRecord):
    """Prepared admission plus the exact manifest identity that produced it."""
    manifest: P3LineageManifest
    admission: dict
    def __post_init__(self):
        if self.manifest.target.product_group not in _PRODUCT_GROUPS:
            raise ValueError("P3 lineage target must be the P3 product group")
        if self.admission.get("manifest_sha256") != self.manifest.digest():
            raise ValueError("P3 lineage admission does not belong to this manifest")
    @classmethod
    def prepare(cls, manifest: P3LineageManifest) -> P3LineageBinding:
        """Read and verify every bound artifact once, then freeze the admission."""
        from mindie_bench.runner.p3_lineage import admit_p3_lineage

        return cls(manifest=manifest, admission=admit_p3_lineage(manifest))
    def to_dict(self) -> dict:
        return self.admission
    def admit(
        self,
        *,
        profile_id: str,
        product_group: str,
        cohort: str,
        protocol: str,
        task_id: str,
        task_root: Path | None = None,
        verify_artifacts: bool = True,
    ) -> dict:
        """Re-run the same entrance; the executor uses fresh artifact bytes."""
        if self.manifest.digest() != self.admission.get("manifest_sha256"):
            raise P3LineageRejected(["p3_manifest_evidence_changed_since_admission"])
        target = self.manifest.target
        if (profile_id, product_group, cohort, protocol, task_id) != (
            target.profile_id,
            target.product_group,
            target.cohort,
            target.protocol,
            target.task_id,
        ):
            raise P3LineageRejected(["p3_target_mismatch"])
        from mindie_bench.runner.p3_lineage import evaluate_p3_lineage

        current = evaluate_p3_lineage(
            self.manifest, task_root=task_root, verify_artifacts=verify_artifacts
        )
        if current["status"] != "admitted":
            raise P3LineageRejected(current["reasons"])
        if current["admission_sha256"] != self.admission["admission_sha256"]:
            raise P3LineageRejected(["p3_lineage_changed_after_admission"])
        return dict(self.admission)

