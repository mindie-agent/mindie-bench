"""P3 real-receipt lineage contracts and the real product entrance.

Synthetic cases use the exact native capture/Seed/consumer schema (including the
in-capture C6 `graded_distillation` stage) with small values. They are contract
tests, not P3/Heldout acceptance evidence. One test reads the authorized legacy
receipt metadata only, and skips if it is absent.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.evaluation.split_manifest import SourceAsset, build_manifest
from mindie_bench.io import digest, task_hash, tree_hash, write_json
from mindie_bench.models import Group, TaskSpec, TerminalStatus, TrialReceipt
from mindie_bench.runner.matrix import (
    BudgetEnvelope,
    MatchingConditions,
    MatrixTask,
    ProductGroup,
    ProductObservation,
    ProductProfile,
    build_product_report,
    execute_product_matrix,
    plan_product_matrix,
)
from mindie_bench.runner.p3_lineage import (
    CostObservation,
    P3Consumer,
    P3CostLedger,
    P3Downstream,
    P3FrozenCorpus,
    P3LineageBinding,
    P3LineageManifest,
    P3LineageRejected,
    P3NativeCapture,
    P3Organizer,
    P3Review,
    P3SeedRun,
    P3StageObservation,
    P3Target,
    budget_ledger_mismatches,
)
from mindie_bench.runner.p3_receipts import parse_capture_receipts
from mindie_bench.runner.product_executor import ProductExecutionBinding, ProductExecutor
from mindie_bench.runner.product_inputs import derive_inputs, runtime_snapshot

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
HELDOUT_TASK = "c" * 64
HELDOUT_ORACLE = "d" * 64
HELDOUT_INSTRUCTION = "f" * 64
REAL_BASE = Path("/Volumes/data/temp/mindie-003-execution/wave4/g3-native-dev-v1")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(task_sha256: str, oracle_sha256: str, instruction_sha256: str, content: list[str]) -> str:
    return digest(
        {
            "task_sha256": task_sha256,
            "oracle_sha256": oracle_sha256,
            "instruction_sha256": instruction_sha256,
            "content_sha256": sorted(set(content)),
        }
    )


def _stage(stage_id: str, started: float, ended: float, *, absolute: bool) -> P3StageObservation:
    return P3StageObservation(
        stage_id=stage_id,
        status="completed",
        started_sec=started,
        ended_sec=ended,
        elapsed_sec=ended - started,
        started_at=BASE + timedelta(seconds=started) if absolute else None,
        ended_at=BASE + timedelta(seconds=ended) if absolute else None,
    )


def _capture_stage(name: str, started: float, ended: float, absolute: bool) -> dict:
    row = {"name": name, "status": "completed", "started_sec": started, "ended_sec": ended, "elapsed_sec": ended - started}
    if absolute:
        row["started_at"] = (BASE + timedelta(seconds=started)).isoformat()
        row["ended_at"] = (BASE + timedelta(seconds=ended)).isoformat()
    return row


def _build_seed_task(root: Path) -> tuple[Path, str, str, str]:
    seed_root = root / "seed-task"
    (seed_root / "verifier").mkdir(parents=True, exist_ok=True)
    (seed_root / "oracle").mkdir(parents=True, exist_ok=True)
    (seed_root / "instruction.md").write_text("seed instruction\n", encoding="utf-8")
    (seed_root / "verifier/test.sh").write_text("exit 0\n", encoding="utf-8")
    (seed_root / "oracle/replay_rules.json").write_text("{}\n", encoding="utf-8")
    (seed_root / "task.toml").write_text(
        'schema_version = "1.0"\n'
        '[task]\n'
        'instance_id = "seed-task"\n'
        'domain_category = "graph_and_kernel_debugging"\n'
        'evidence = "synthetic"\n'
        '[mindie_knowledge]\n'
        'knowledge_gradient = "analogical"\n'
        '[environment.l1]\n'
        'repo_path = "/tmp"\n'
        f'base_commit = "{"a" * 40}"\n'
        '[oracle]\n'
        'protected_paths = ["oracle"]\n',
        encoding="utf-8",
    )
    spec = TaskSpec.load(seed_root)
    task_sha = task_hash(seed_root)
    oracle_sha = freeze_oracle(seed_root, spec.oracle).sha256
    instruction_sha = sha_bytes((seed_root / "instruction.md").read_bytes())
    return seed_root, task_sha, oracle_sha, instruction_sha


def build_lineage(
    root: Path,
    *,
    same_problem: bool = False,
    absolute: bool = True,
    seed_cost_complete: bool = True,
    ledger_capture_calls: int = 0,
    bad_stage_order: bool = False,
    profile_id: str = "p3-profile",
    cohort: str = "main",
    protocol: str = "matched-product/1",
    downstream_task_id: str = "heldout-task",
    downstream_task_sha256: str = HELDOUT_TASK,
    downstream_oracle_sha256: str = HELDOUT_ORACLE,
    downstream_instruction_sha256: str = HELDOUT_INSTRUCTION,
    downstream_model_calls: int = 25,
    downstream_npu_runs: int = 4,
    matrix_task_family: str | None = "heldout-family",
    hardware_tier: str | None = "A2",
) -> tuple[Path, P3LineageManifest]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seed_root, seed_task, seed_oracle, seed_instruction = _build_seed_task(root)
    heldout_task = seed_task if same_problem else downstream_task_sha256
    heldout_oracle = seed_oracle if same_problem else downstream_oracle_sha256
    heldout_instruction = seed_instruction if same_problem else downstream_instruction_sha256

    def receipt(name: str, payload: dict) -> tuple[Path, str]:
        path = root / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    seed_content = sha_bytes(b"seed transcript\n")
    heldout_content = sha_bytes(b"heldout transcript\n")
    seed_commit = sha_bytes(b"seed commit\n")
    heldout_commit = sha_bytes(b"heldout commit\n")
    assets = [
        {"asset_id": "seed-transcript", "source_id": "source:seed", "source_locator": "locator:seed-transcript", "session_id": "session:seed", "content_sha256": seed_content},
        {"asset_id": "seed-commit", "source_id": "source:seed-commit", "source_locator": "locator:seed-commit", "session_id": "session:seed", "content_sha256": seed_commit},
        {"asset_id": "heldout-transcript", "source_id": "source:heldout", "source_locator": "locator:heldout-transcript", "session_id": "session:heldout", "content_sha256": heldout_content},
        {"asset_id": "heldout-commit", "source_id": "source:heldout-commit", "source_locator": "locator:heldout-commit", "session_id": "session:heldout", "content_sha256": heldout_commit},
    ]
    claims = [
        {"family_id": "seed-family", "split": "seed", "asset_ids": ["seed-transcript", "seed-commit"], "problem_identity": "problem:seed", "problem_identity_scope": "problem", "evidence_refs": ["claim:seed"]},
        {"family_id": "heldout-family", "split": "heldout", "asset_ids": ["heldout-transcript", "heldout-commit"], "problem_identity": "problem:heldout", "problem_identity_scope": "problem", "evidence_refs": ["claim:heldout"]},
    ]
    split_manifest = build_manifest(
        [
            SourceAsset(asset_id="seed-transcript", task_family="seed-family", source_id="source:seed", source_locator="locator:seed-transcript", source_kind="transcript", timestamp=BASE, content_sha256=seed_content, lineage_resolved=True),
            SourceAsset(asset_id="seed-commit", task_family="seed-family", source_id="source:seed-commit", source_locator="locator:seed-commit", source_kind="commit", timestamp=BASE, content_sha256=seed_commit, lineage_resolved=True),
            SourceAsset(asset_id="heldout-transcript", task_family="heldout-family", source_id="source:heldout", source_locator="locator:heldout-transcript", source_kind="transcript", timestamp=BASE, content_sha256=heldout_content, lineage_resolved=True),
            SourceAsset(asset_id="heldout-commit", task_family="heldout-family", source_id="source:heldout-commit", source_locator="locator:heldout-commit", source_kind="commit", timestamp=BASE, content_sha256=heldout_commit, lineage_resolved=True),
        ],
        {"seed-family": "seed", "heldout-family": "heldout"},
        created_at=BASE,
    )
    corpus = root / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "unit-1.md").write_text("captured publication unit\n", encoding="utf-8")
    corpus_sha = tree_hash(corpus)

    seed_receipt, seed_receipt_sha = receipt(
        "seed.json",
        {
            "trial_id": "seed-trial",
            "instance_id": "seed-task",
            "terminal_status": "completed_pass",
            "task_digest": seed_task,
            "primary_metrics": {
                "model_calls": 10,
                "npu_test_runs": 2,
                "cost_complete": seed_cost_complete,
                "estimated_cost_usd": 1.0 if seed_cost_complete else None,
                "tokens": {"input_uncached": 1000, "input_cached": 0, "cache_creation": 0, "output": 500, "complete": True},
            },
            "auxiliary_metrics": {
                "capture_model_calls": 1,
                "capture_provider_tokens": {"input_uncached": 41, "input_cached": 0, "output": 108, "complete": True},
            },
        },
    )
    capture_receipt, capture_sha = receipt(
        "capture.json",
        {
            "status": "completed",
            "task_state": "completed",
            "protocol": "graded-c6",
            "native_harness": "dsh",
            "corpus_lint": {"status": "passed"},
            "cleanup": {"status": "completed", "errors": [], "npu_runs": 0},
            "cost_complete": True,
            "estimated_cost_usd": 0.0,
            "stage_observations": [
                _capture_stage("native_artifact_selection", 0.0, 0.1, absolute),
                _capture_stage("seed_local_publication", 0.1, 0.2, absolute),
                _capture_stage("graded_distillation", 0.3, 1.0, absolute),
                _capture_stage("local_publication_and_merge", 1.1, 1.2, absolute),
                _capture_stage("independent_consumer_readback", 1.3, 1.4, absolute),
            ],
            "indexer_usage": {"model_calls": 1, "input_tokens": 41, "cached_input_tokens": 0, "output_tokens": 108, "unknown_calls": 0, "npu_runs": 0},
            "distill": {"status": "applied", "model_calls": 1, "billing_status": "reported", "cleanup_failed": False, "usage": {"input_tokens": 41, "cached_input_tokens": 0, "output_tokens": 108, "estimated_cost_usd": 0.0}},
            "tokens": {"total": 149, "complete": True},
        },
    )
    seed_family_canonical = canonical(seed_task, seed_oracle, seed_instruction, [seed_content, seed_commit])
    heldout_family_canonical = canonical(heldout_task, heldout_oracle, heldout_instruction, [heldout_content, heldout_commit])
    consumer_receipt, consumer_sha = receipt(
        "consumer.json",
        {
            "status": "completed",
            "model_calls": 0,
            "npu_runs": 0,
            "provider_writes": 0,
            "remote_writes": 0,
            "corpus_sha256": corpus_sha,
            "scope": "P3 publication consumer for the independently reviewed family",
            "original_receipts_unchanged": True,
            "raw_capture_receipt": {"sha256": capture_sha},
            "raw_trial_receipt": {"sha256": seed_receipt_sha},
        },
    )
    review_receipt, review_sha = receipt(
        "review.json",
        {
            "status": "accepted",
            "verdict": "distinct_closed_family",
            "independent": True,
            "organizer_receipt_sha256": capture_sha,
            "seed_family_canonical": seed_family_canonical,
            "heldout_family_canonical": heldout_family_canonical,
            "model_calls": 0,
            "npu_runs": 0,
            "tokens": {"input_uncached": 0, "input_cached": 0, "cache_creation": 0, "output": 0, "complete": True},
            "cost_complete": True,
            "estimated_cost_usd": 0.0,
        },
    )
    ledger = P3CostLedger(
        seed=CostObservation(basis="observed", stage_id="seed_trial", receipt_sha256=seed_receipt_sha, model_calls=10, input_tokens=1000, cached_input_tokens=0, cache_creation_tokens=0, output_tokens=500, tokens_total=1500, tokens_basis="complete", npu_runs=2, estimated_cost_usd=1.0, price_basis="priced", complete=True),
        capture=CostObservation(basis="observed", stage_id="capture_ingestion", receipt_sha256=capture_sha, model_calls=ledger_capture_calls, input_tokens=0, cached_input_tokens=0, cache_creation_tokens=None, output_tokens=0, tokens_total=0, tokens_basis="complete", npu_runs=0, estimated_cost_usd=0.0, price_basis="priced", complete=True),
        organizer=CostObservation(basis="observed", stage_id="graded_distillation", receipt_sha256=capture_sha, model_calls=1, input_tokens=41, cached_input_tokens=0, cache_creation_tokens=None, output_tokens=108, tokens_total=149, tokens_basis="complete", npu_runs=0, estimated_cost_usd=0.0, price_basis="priced", complete=True),
        review=CostObservation(basis="observed", stage_id="independent_review", receipt_sha256=review_sha, model_calls=0, input_tokens=0, cached_input_tokens=0, cache_creation_tokens=0, output_tokens=0, tokens_total=0, tokens_basis="complete", npu_runs=0, estimated_cost_usd=0.0, price_basis="priced", complete=True),
        downstream=CostObservation(basis="declared", stage_id="downstream_declared", receipt_sha256=heldout_task, model_calls=downstream_model_calls, tokens_total=None, tokens_basis="unknown", npu_runs=downstream_npu_runs, estimated_cost_usd=None, price_basis="unknown", complete=False),
    )
    manifest = P3LineageManifest(
        created_at=BASE - timedelta(seconds=20),
        target=P3Target(profile_id=profile_id, product_group="P3_v003_flywheel", cohort=cohort, protocol=protocol, task_id=downstream_task_id, matrix_task_family=matrix_task_family, hardware_tier=hardware_tier),
        shared_mechanism="rope position repair",
        split_manifest=split_manifest,
        assets=assets,
        strong_edges=[
            {"left": "seed-transcript", "right": "seed-commit", "reason": "exact_session_id"},
            {"left": "heldout-transcript", "right": "heldout-commit", "reason": "exact_session_id"},
        ],
        candidate_edges=[],
        claims=claims,
        decisions=[],
        exposures=[],
        seed=P3SeedRun(trial_id="seed-trial", task_id="seed-task", family_id="seed-family", task_sha256=seed_task, oracle_sha256=seed_oracle, instruction_sha256=seed_instruction, terminal_status="completed_pass", receipt_locator=str(seed_receipt), receipt_sha256=seed_receipt_sha, seed_root_locator=str(seed_root), seed_identity_basis="verified_bytes", context_id="seed-ctx", completed_at=BASE - timedelta(seconds=10) if absolute else None),
        capture=P3NativeCapture(capture_id="capture-1", seed_trial_id="seed-trial", seed_receipt_sha256=seed_receipt_sha, native_harness="dsh", status="completed", receipt_locator=str(capture_receipt), receipt_sha256=capture_sha, stage_observations=[
            _stage("native_artifact_selection", 0.0, 0.1, absolute=absolute),
            _stage("seed_local_publication", 0.1, 0.2, absolute=absolute),
            _stage("graded_distillation", 1.25, 1.5, absolute=absolute) if bad_stage_order else _stage("graded_distillation", 0.3, 1.0, absolute=absolute),
            _stage("local_publication_and_merge", 1.1, 1.2, absolute=absolute),
            _stage("independent_consumer_readback", 1.3, 1.4, absolute=absolute),
        ], tokens_total=149, tokens_complete=True, corpus_lint_status="passed", started_at=BASE if absolute else None, completed_at=BASE + timedelta(seconds=2) if absolute else None),
        organizer=P3Organizer(capture_receipt_sha256=capture_sha, actor_id="c6-organizer", status="applied", model_calls=1, input_tokens=41, cached_input_tokens=0, cache_creation_tokens=None, output_tokens=108, billing_status="reported", cleanup_failed=False, completed_at=BASE + timedelta(seconds=1) if absolute else None),
        consumer=P3Consumer(consumer_id="consumer-1", receipt_locator=str(consumer_receipt), receipt_sha256=consumer_sha, capture_receipt_sha256=capture_sha, trial_receipt_sha256=seed_receipt_sha, status="completed", model_calls=0, provider_writes=0, remote_writes=0, corpus_sha256=corpus_sha, scope="P3 publication consumer for the independently reviewed family", completed_at=BASE + timedelta(seconds=2) if absolute else None),
        frozen_corpus=P3FrozenCorpus(locator=str(corpus), tree_sha256=corpus_sha, unit_count=1, consumer_receipt_sha256=consumer_sha, frozen_at=BASE + timedelta(seconds=3) if absolute else None),
        review=P3Review(review_id="review-1", consumer_receipt_sha256=consumer_sha, capture_receipt_sha256=capture_sha, status="accepted", verdict="distinct_closed_family", independent=True, receipt_locator=str(review_receipt), receipt_sha256=review_sha, actor_id="independent-reviewer", context_id="review-ctx", seed_family_canonical=seed_family_canonical, heldout_family_canonical=heldout_family_canonical, completed_at=BASE + timedelta(seconds=4) if absolute else None),
        downstream=P3Downstream(task_id=downstream_task_id, family_id="heldout-family", task_sha256=heldout_task, oracle_sha256=heldout_oracle, instruction_sha256=heldout_instruction, context_id="heldout-ctx", review_receipt_sha256=review_sha, started_at=BASE + timedelta(seconds=5) if absolute else None),
        cost_ledger=ledger,
    )
    return root, manifest


def p3_budget() -> BudgetEnvelope:
    return BudgetEnvelope(
        total_model_calls=25,
        foreground_model_calls=24,
        organizer_model_calls=0,
        review_model_calls=0,
        npu_runs=4,
        max_wall_clock_sec=3600.0,
        max_cost_usd=2.0,
        lifecycle_model_calls=36,
        lifecycle_cost_usd=3.0,
        seed_model_calls=10,
        capture_model_calls=0,
        capture_organizer_model_calls=1,
        independent_review_model_calls=0,
        seed_npu_runs=2,
        seed_cost_usd=1.0,
        capture_cost_usd=0.0,
        capture_organizer_cost_usd=0.0,
        independent_review_cost_usd=0.0,
    )


def p3_conditions() -> MatchingConditions:
    return MatchingConditions(model_id="model-a", inference_profile="temperature=0", task_start_digest="start-sha", oracle_digest="oracle-sha", input_digest="input-sha", knowledge_cutoff="2026-10-01", tool_permission_digest="tools-sha", persistence_permission_digest="persist-sha")


def p3_profile(binding: P3LineageBinding, *, budget: BudgetEnvelope | None = None) -> ProductProfile:
    return ProductProfile(profile_id="p3-profile", group=ProductGroup.P3_V003_FLYWHEEL, experiment_profile="experiments/p3.json", protocol="matched-product/1", fidelity="matched", cohort="main", conditions=p3_conditions(), budget=budget or p3_budget(), p3_lineage=binding)


def _relocate(manifest: P3LineageManifest, root: Path) -> P3LineageManifest:
    moved = manifest.model_copy(deep=True)
    moved.seed.receipt_locator = str(root / "seed.json")
    moved.seed.seed_root_locator = str(root / "seed-task")
    moved.capture.receipt_locator = str(root / "capture.json")
    moved.consumer.receipt_locator = str(root / "consumer.json")
    moved.review.receipt_locator = str(root / "review.json")
    moved.frozen_corpus.locator = str(root / "corpus")
    return moved


def test_p3_admission_binds_real_capture_stage_schema_and_numeric_ledger(tmp_path):
    _, manifest = build_lineage(tmp_path)
    binding = P3LineageBinding.prepare(manifest)
    admission = binding.admission
    assert admission["status"] == "admitted"
    assert admission["claim_ready"] is True
    assert admission["prerequisite_cost_complete"] is True
    assert admission["lifecycle_cost_complete"] is False
    assert admission["prerequisite_claim_ready"] is True
    assert admission["lifecycle_claim_ready"] is False
    assert "prerequisite admission" in admission["claim_scope"]
    assert admission["seed"]["canonical_id"] != admission["heldout"]["canonical_id"]
    assert admission["cost_ledger"]["roles"]["seed"]["model_calls"] == 10
    assert admission["cost_ledger"]["roles"]["capture"]["model_calls"] == 0
    assert admission["cost_ledger"]["roles"]["organizer"]["model_calls"] == 1
    assert admission["cost_ledger"]["roles"]["organizer"]["tokens_total"] == 149
    assert admission["cost_ledger"]["roles"]["organizer"]["input_tokens"] == 41
    assert admission["cost_ledger"]["roles"]["organizer"]["output_tokens"] == 108
    stages = {item["stage_id"]: item for item in admission["stages"] if "stage_id" in item}
    assert stages["capture_ingestion"]["receipt_sha256"] == stages["graded_distillation"]["receipt_sha256"]
    assert stages["capture_ingestion"]["stage_id"] != stages["graded_distillation"]["stage_id"]
    assert admission["timeline"]["absolute_available"] is True
    assert admission["timeline"]["missing_absolute"] == []
    assert budget_ledger_mismatches(admission["cost_ledger"], p3_budget().ledger_dict()) == []


def test_p3_numeric_ledger_tamper_is_denied(tmp_path):
    _, manifest = build_lineage(tmp_path)
    manifest.cost_ledger.seed.model_calls = 11
    with pytest.raises(P3LineageRejected, match="seed.model_calls != observed.model_calls"):
        P3LineageBinding.prepare(manifest)


def test_p3_missing_completeness_cannot_be_marked_complete(tmp_path):
    _, manifest = build_lineage(tmp_path / "missing", seed_cost_complete=False)
    with pytest.raises(P3LineageRejected, match="seed.complete|seed.price_basis|seed_usage"):
        P3LineageBinding.prepare(manifest)


def test_p3_double_attribution_is_detected(tmp_path):
    _, manifest = build_lineage(tmp_path, ledger_capture_calls=1)
    with pytest.raises(P3LineageRejected, match="capture_model_calls_double_attributed_or_lost|capture.model_calls"):
        P3LineageBinding.prepare(manifest)


def test_p3_missing_absolute_timestamps_are_reported_not_inferred(tmp_path):
    _, manifest = build_lineage(tmp_path, absolute=False)
    admission = P3LineageBinding.prepare(manifest).admission
    assert admission["status"] == "admitted"
    assert admission["claim_ready"] is False
    assert admission["timeline"]["absolute_available"] is False
    assert "seed_completed_at" in admission["timeline"]["missing_absolute"]
    assert "capture_started_at" in admission["timeline"]["missing_absolute"]
    assert any(item.startswith("absolute_timeline_missing:") for item in admission["limitations"])


def test_renamed_same_task_bytes_cannot_create_an_independent_family(tmp_path):
    _, manifest = build_lineage(tmp_path, same_problem=True)
    with pytest.raises(P3LineageRejected, match="shared_task_sha256_across_splits"):
        P3LineageBinding.prepare(manifest)


def test_p3_profile_requires_admission_and_plan_binds_exact_downstream_task(tmp_path):
    with pytest.raises(ValueError, match="closed-family lineage"):
        ProductProfile(profile_id="label-only", group=ProductGroup.P3_V003_FLYWHEEL, experiment_profile="experiments/p3.json", protocol="matched-product/1", fidelity="matched", cohort="main", conditions=p3_conditions(), budget=p3_budget())
    _, manifest = build_lineage(tmp_path / "bound")
    binding = P3LineageBinding.prepare(manifest)
    plan = plan_product_matrix([MatrixTask("heldout-task", "heldout-family", "A2")], [p3_profile(binding)])
    assert len(plan.cells) == 1
    with pytest.raises(ValueError, match="exactly the planned downstream Heldout task"):
        plan_product_matrix([MatrixTask("other-task", "heldout-family", "A2")], [p3_profile(binding)])


def test_product_report_exposes_ledger_budget_contract_and_overrun(tmp_path):
    _, manifest = build_lineage(tmp_path)
    binding = P3LineageBinding.prepare(manifest)
    plan = plan_product_matrix([MatrixTask("heldout-task", "heldout-family", "A2")], [p3_profile(binding)])
    report = build_product_report(plan, [ProductObservation(ProductGroup.P3_V003_FLYWHEEL, "heldout-task", 1, "pass", model_calls=30, npu_runs=5, estimated_cost_usd=5.0, total_wall_clock_sec=100.0)])
    group = report.groups["P3_v003_flywheel"]
    assert group.p3_lineage["cost_ledger"]["roles"]["seed"]["model_calls"] == 10
    assert group.p3_lineage["cost_ledger"]["roles"]["organizer"]["tokens_total"] == 149
    assert {(item["field"], item["observed"]) for item in group.budget_overruns} == {("model_calls", 30), ("npu_runs", 5), ("estimated_cost_usd", 5.0)}
    assert group.budget_contract["conditional_reuse_only"] is True
    assert group.budget_contract["lifecycle_predeclared"] is True
    assert group.budget_contract["p2_waste_padding"] is False
    assert group.budget_contract["seed_organizer_overhead"]["seed_model_calls"] == 10


def test_p2_report_has_no_p3_seed_padding(tmp_path):
    p2 = ProductProfile(profile_id="p2-profile", group=ProductGroup.P2_V003_FULL, experiment_profile="experiments/p2.json", protocol="matched-product/1", fidelity="matched", cohort="main", conditions=p3_conditions(), budget=BudgetEnvelope(total_model_calls=25, foreground_model_calls=24, organizer_model_calls=1, review_model_calls=0, npu_runs=4, max_wall_clock_sec=3600.0, max_cost_usd=2.0))
    plan = plan_product_matrix([MatrixTask("heldout-task", "heldout-family", "A2")], [p2])
    group = build_product_report(plan, []).groups["P2_v003_full"]
    assert group.budget_contract["conditional_reuse_only"] is True
    assert group.budget_contract["lifecycle_predeclared"] is False
    assert group.budget_contract["seed_organizer_overhead"]["seed_model_calls"] == 0
    assert group.budget_contract["matched_downstream"]["total_model_calls"] == 25


def test_capture_stage_order_is_bound_to_the_receipt(tmp_path):
    _, manifest = build_lineage(tmp_path, bad_stage_order=True)
    with pytest.raises(P3LineageRejected, match="organizer_after_publication|capture_receipt_stage_mismatch"):
        P3LineageBinding.prepare(manifest)


def test_lifecycle_budget_requires_predeclared_seed_organizer_allocation():
    with pytest.raises(ValueError, match="lifecycle_model_calls"):
        BudgetEnvelope(total_model_calls=25, seed_model_calls=10, capture_model_calls=0, lifecycle_model_calls=34, lifecycle_cost_usd=3.0)
    with pytest.raises(ValueError, match="lifecycle_cost_usd"):
        BudgetEnvelope(total_model_calls=25, seed_model_calls=10, capture_model_calls=0, seed_cost_usd=1.0, max_cost_usd=2.0, lifecycle_model_calls=35, lifecycle_cost_usd=2.5)


def test_execute_product_matrix_rechecks_changed_capture_before_caller_executor(tmp_path):
    _, manifest = build_lineage(tmp_path)
    binding = P3LineageBinding.prepare(manifest)
    plan = plan_product_matrix(
        [MatrixTask("heldout-task", "heldout-family", "A2")], [p3_profile(binding)]
    )
    Path(manifest.capture.receipt_locator).write_text("{}", encoding="utf-8")
    calls = []
    with pytest.raises(ValueError, match="P3 lineage admission failed"):
        execute_product_matrix(plan, lambda cell: calls.append(cell))
    assert calls == []


def test_seed_root_bytes_are_reverified_and_changed_bytes_denied(tmp_path):
    root, manifest = build_lineage(tmp_path)
    binding = P3LineageBinding.prepare(manifest)
    (root / "seed-task/instruction.md").write_text("changed seed\n", encoding="utf-8")
    with pytest.raises(P3LineageRejected, match="seed_instruction_bytes_changed|seed_task_bytes_changed"):
        binding.admit(
            profile_id="p3-profile",
            product_group="P3_v003_flywheel",
            cohort="main",
            protocol="matched-product/1",
            task_id="heldout-task",
            verify_artifacts=True,
        )


def test_receipt_recorded_seed_identity_is_an_explicit_limitation(tmp_path):
    _, manifest = build_lineage(tmp_path / "recorded")
    manifest.seed.seed_identity_basis = "receipt_recorded"
    manifest.seed.seed_root_locator = None
    admission = P3LineageBinding.prepare(manifest).admission
    assert admission["status"] == "admitted"
    assert admission["seed"]["identity_basis"] == "receipt_recorded"
    assert "seed_identity_receipt_recorded_not_byte_verified" in admission["limitations"]
    assert admission["claim_ready"] is False


def test_content_identity_survives_relocation_and_changed_bytes_still_deny(tmp_path):
    root, manifest = build_lineage(tmp_path / "origin")
    binding = P3LineageBinding.prepare(manifest)
    moved_root = tmp_path / "moved"
    shutil.copytree(root, moved_root)
    moved = _relocate(manifest, moved_root)
    assert moved.content_identity() == manifest.content_identity()
    assert moved.digest() != manifest.digest()
    moved_binding = P3LineageBinding.prepare(moved)
    assert (
        moved_binding.admission["admission_sha256"]
        == binding.admission["admission_sha256"]
    )
    assert (
        moved_binding.admission["content_identity_sha256"]
        == binding.admission["content_identity_sha256"]
    )
    assert moved_binding.admission["manifest_sha256"] != binding.admission["manifest_sha256"]
    (moved_root / "capture.json").write_text("{}", encoding="utf-8")
    with pytest.raises(P3LineageRejected, match="capture_receipt_sha256_mismatch"):
        P3LineageBinding.prepare(moved)


def test_p3_label_binding_rejects_conflicts_and_keeps_family_labels_distinct(tmp_path):
    _, manifest = build_lineage(tmp_path, matrix_task_family="matrix-bucket")
    binding = P3LineageBinding.prepare(manifest)
    plan_product_matrix(
        [MatrixTask("heldout-task", "matrix-bucket", "A2")], [p3_profile(binding)]
    )
    with pytest.raises(ValueError, match="task_family label conflicts"):
        plan_product_matrix(
            [MatrixTask("heldout-task", "other-bucket", "A2")], [p3_profile(binding)]
        )
    with pytest.raises(ValueError, match="hardware tier conflicts"):
        plan_product_matrix(
            [MatrixTask("heldout-task", "matrix-bucket", "A3")], [p3_profile(binding)]
        )
    labels = binding.admission["label_binding"]
    assert labels["matrix_task_family"] == "matrix-bucket"
    assert labels["lineage_family_id"] == "heldout-family"
    assert labels["matrix_task_family_equals_lineage_family"] is False


def test_budget_keeps_downstream_allowances_and_accounts_prerequisite_lifecycle():
    with pytest.raises(ValueError, match="fit the total"):
        BudgetEnvelope(total_model_calls=5, organizer_model_calls=10)
    envelope = BudgetEnvelope(
        total_model_calls=25,
        foreground_model_calls=10,
        organizer_model_calls=10,
        review_model_calls=5,
        capture_organizer_model_calls=2,
        independent_review_model_calls=1,
        seed_model_calls=10,
        capture_model_calls=3,
        max_cost_usd=2.0,
        seed_cost_usd=1.0,
        capture_cost_usd=0.5,
        capture_organizer_cost_usd=0.25,
        independent_review_cost_usd=0.25,
        lifecycle_model_calls=41,
        lifecycle_cost_usd=4.0,
    )
    assert envelope.lifecycle_model_calls == 25 + 10 + 3 + 2 + 1
    with pytest.raises(ValueError, match="lifecycle_model_calls"):
        BudgetEnvelope(
            total_model_calls=25,
            foreground_model_calls=10,
            organizer_model_calls=10,
            review_model_calls=5,
            capture_organizer_model_calls=2,
            independent_review_model_calls=1,
            seed_model_calls=10,
            capture_model_calls=3,
            lifecycle_model_calls=40,
            lifecycle_cost_usd=4.0,
        )


def test_real_legacy_capture_receipts_report_missing_timestamps_and_deny_p3():
    capture_path = REAL_BASE / "trials/dsh-g3-dev-rope-64_G3_2_21d368801b24/profile/full-loop-receipt.json"
    trial_path = REAL_BASE / "trials/dsh-g3-dev-rope-64_G3_2_21d368801b24/trial_receipt.json"
    consumer_path = REAL_BASE / "seed-v2-consumer-continuation-v3/receipt.json"
    if not (capture_path.is_file() and trial_path.is_file() and consumer_path.is_file()):
        pytest.skip("authorized legacy receipt metadata is not present")
    capture = json.loads(capture_path.read_text())
    trial = json.loads(trial_path.read_text())
    consumer = json.loads(consumer_path.read_text())
    result = parse_capture_receipts(
        capture=capture,
        trial=trial,
        consumer=consumer,
        capture_sha256=sha_bytes(capture_path.read_bytes()),
        trial_sha256=sha_bytes(trial_path.read_bytes()),
        consumer_sha256=sha_bytes(consumer_path.read_bytes()),
        capture_locator=str(capture_path),
        trial_locator=str(trial_path),
        consumer_locator=str(consumer_path),
        consumer_scope=consumer.get("scope"),
    )
    assert result["verdict"]["status"] == "denied"
    assert result["verdict"]["p3_eligible"] is False
    assert result["verdict"]["acceptance_upgrade"] is False
    assert "capture_status:failed" in result["verdict"]["reasons"]
    assert "corpus_lint:pending" in result["verdict"]["reasons"]
    assert "absolute_timestamp_missing:capture.started_at" in result["verdict"]["reasons"]
    assert "independent_review_missing" in result["verdict"]["reasons"]
    assert result["organizer"]["model_calls"] == 1
    assert result["organizer"]["input_tokens"] == 4121
    assert result["organizer"]["output_tokens"] == 10804
    assert result["organizer"]["tokens_total"] == 14925
    assert result["attribution"]["capture_plus_organizer_model_calls"] == 1
    assert result["attribution"]["capture_plus_organizer_tokens_total"] == 14925
    assert result["attribution"]["duplicate_or_lost"] == []
    assert result["attribution"]["unknown"] == ["capture_npu_runs_unknown"]
    assert result["timestamps"]["absolute_available"] is False
    assert result["timestamps"]["inference_performed"] is False
    assert result["receipt_immutability"]["consumer_raw_receipt_hash_matches"] is True
    assert result["consumer"]["model_calls"] == 0


def _executor_case(case, tmp_path: Path):
    task_id = case["spec"].task.instance_id
    derived = derive_inputs({task_id: case["task"]}, case["config"], "l2")
    instruction_sha = sha_bytes((case["task"] / "instruction.md").read_bytes())
    _, manifest = build_lineage(
        tmp_path / "lineage",
        profile_id="p3-native-entrance",
        cohort="synthetic",
        protocol="native-product-contract-test/1",
        downstream_task_id=task_id,
        downstream_task_sha256=derived["tasks"][task_id]["task_sha256"],
        downstream_oracle_sha256=derived["tasks"][task_id]["oracle_sha256"],
        downstream_instruction_sha256=instruction_sha,
        downstream_model_calls=derived["budget"]["total_model_calls"],
        downstream_npu_runs=derived["budget"]["npu_runs"],
        hardware_tier="cpu",
    )
    binding = P3LineageBinding.prepare(manifest)
    runtime_manifest = tmp_path / "product-runtime.json"
    write_json(runtime_manifest, {"profile_id": "p3-native-entrance", "product_group": ProductGroup.P3_V003_FLYWHEEL, "config_sha256": digest(case["config"].model_dump(mode="json")), "scope": "Zero-model product entrance contract test only", "derived_inputs": derived, "runtime_snapshot": runtime_snapshot(case["config"], Group.G3)})
    budget_values = dict(derived["budget"])
    budget_values.update(foreground_model_calls=24, organizer_model_calls=0, review_model_calls=0, seed_model_calls=10, capture_model_calls=0, capture_organizer_model_calls=1, independent_review_model_calls=0, seed_npu_runs=2, seed_cost_usd=1.0, capture_cost_usd=0.0, capture_organizer_cost_usd=0.0, independent_review_cost_usd=0.0, max_cost_usd=2.0, lifecycle_model_calls=derived["budget"]["total_model_calls"] + 11, lifecycle_cost_usd=3.0)
    profile = ProductProfile(profile_id="p3-native-entrance", group=ProductGroup.P3_V003_FLYWHEEL, experiment_profile=str(runtime_manifest), protocol="native-product-contract-test/1", fidelity="matched", cohort="synthetic", conditions=MatchingConditions(**derived["conditions"]), budget=BudgetEnvelope(**budget_values), p3_lineage=binding)
    execution_binding = ProductExecutionBinding(profile, case["config"], Group.G3, runtime_manifest, hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(), "l2")
    plan = plan_product_matrix([MatrixTask(task_id, "heldout-family", "cpu")], [profile])
    run_root = tmp_path / "runs"
    run_root.mkdir()
    return execution_binding, plan, task_id, run_root


def _fake_trial(tmp_path: Path, task_id: str, calls: list[dict]):
    def execute(task_root, config, **kwargs):
        context = kwargs["experiment_context"]
        calls.append(context)
        trial_dir = tmp_path / "zero-model-trial"
        trial_dir.mkdir(exist_ok=True)
        return TrialReceipt(trial_id="trial-p3-zero-model", instance_id=task_id, repetition=kwargs["repetition"], tier=kwargs["tier"], ablation_group=kwargs["group"], knowledge_gradient="no_hit", task_digest=context["task_sha256"], experiment_digest=context["native_config_sha256"], environment_fingerprint={"experiment_context": json.loads(json.dumps(context))}, terminal_status=TerminalStatus.PASS, primary_metrics={"reward": 1.0, "resolved": True, "delivery_wall_clock_sec": 1.0}, evidence="synthetic", artifacts={"private_trial_dir": str(trial_dir)})
    return execute


def test_zero_model_product_entrance_admits_bound_p3_lineage(case, tmp_path):
    execution_binding, plan, task_id, run_root = _executor_case(case, tmp_path)
    calls: list[dict] = []
    executor = ProductExecutor({execution_binding.profile.profile_id: execution_binding}, {task_id: case["task"]}, run_root, execute_trial=_fake_trial(tmp_path, task_id, calls))
    observation = executor(plan.cells[0])
    assert observation.status.value == "pass"
    assert len(calls) == 1
    assert calls[0]["p3_admission"]["claim_ready"] is True
    assert calls[0]["p3_admission"]["lifecycle_claim_ready"] is False
    trial_dir = tmp_path / "zero-model-trial"
    observed = json.loads((trial_dir / "product_observation.json").read_text())
    assert observed["experiment_context"]["p3_admission"]["admission_sha256"] == calls[0]["p3_admission"]["admission_sha256"]


def test_zero_model_product_entrance_denies_changed_capture_before_trial(case, tmp_path):
    execution_binding, plan, task_id, run_root = _executor_case(case, tmp_path)
    Path(execution_binding.profile.p3_lineage.manifest.capture.receipt_locator).write_text("{}", encoding="utf-8")
    calls: list[dict] = []
    executor = ProductExecutor({execution_binding.profile.profile_id: execution_binding}, {task_id: case["task"]}, run_root, execute_trial=_fake_trial(tmp_path, task_id, calls))
    with pytest.raises(ValueError, match="P3 lineage admission failed before native execution"):
        executor(plan.cells[0])
    assert calls == []
    assert list(run_root.iterdir()) == []


def test_zero_model_product_entrance_denies_double_attributed_ledger_before_trial(case, tmp_path):
    execution_binding, plan, task_id, run_root = _executor_case(case, tmp_path)
    execution_binding.profile.p3_lineage.manifest.cost_ledger.capture.model_calls = 1
    calls: list[dict] = []
    executor = ProductExecutor({execution_binding.profile.profile_id: execution_binding}, {task_id: case["task"]}, run_root, execute_trial=_fake_trial(tmp_path, task_id, calls))
    with pytest.raises(ValueError, match="P3 lineage admission failed before native execution"):
        executor(plan.cells[0])
    assert calls == []
    assert list(run_root.iterdir()) == []


def test_task_hash_binds_the_real_downstream_bytes(tmp_path):
    _, manifest = build_lineage(tmp_path)
    binding = P3LineageBinding.prepare(manifest)
    task_root = tmp_path / "heldout-task"
    (task_root / "verifier").mkdir(parents=True)
    (task_root / "instruction.md").write_text("mode=pass\n", encoding="utf-8")
    (task_root / "task.toml").write_text('schema_version = "1.0"\n', encoding="utf-8")
    (task_root / "verifier/test.sh").write_text("exit 0\n", encoding="utf-8")
    assert task_hash(task_root) != binding.manifest.downstream.task_sha256
    with pytest.raises(P3LineageRejected, match="downstream_task_bytes_changed"):
        binding.admit(profile_id="p3-profile", product_group="P3_v003_flywheel", cohort="main", protocol="matched-product/1", task_id="heldout-task", task_root=task_root, verify_artifacts=True)
