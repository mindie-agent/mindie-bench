"""High-signal tiered-knowledge contracts over the real corpora.

Tier 0 tests are pure in-memory decisions. Tier 1 tests create isolated SQLite
stores, copy real graded corpora, and run the production organizer.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest
import yaml

from mindie_bench.evaluation.knowledge_tiered import (
    ConsultedKnowledgeRef,
    bridge_run_receipt_to_feedback,
    evaluate_l4_heldout_generalization,
    run_sequential_learning_episodes,
)
from mindie_bench.evaluation.knowledge_tiered_support import (
    GradedUnit,
    Store,
    summarize_usage_events,
)
from mindie_bench.models import ExperimentConfig, HarnessConfig
from mindie_bench.runner.profile_builder import DSH_REASONING_EFFORTS, build_dsh_profile

OPERATORS = Path(os.environ.get("MINDIE_BENCH_OPERATORS_CORPUS", "/Volumes/data/code/knowledge-operators"))
TRANSCRIPT = Path("/Users/maoxx241/code/mindie-transcript-20261004/content")
ALIGNMENT = "unit-ascend-910b-a3-memory-alignment-and-tiling-axioms"
ROOFLINE = "unit-ascend-operator-roofline-and-four-view-measurement"
REDUCTION_SOP = "unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop"
BOTTLENECK = "unit-operator-memory-vs-compute-bottleneck-convergence-workflow"
TRITON_CONTRACT = "unit-triton-ascend-constexpr-and-specialization-contract"
TRITON_SOP = "unit-triton-dynamic-scalar-specialization-sop"
RMSNORM_L4 = "unit-residual-free-rmsnorm-fastpath"
ACLGRAPH_SOP = "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop"
IDLE_DP = "unit-vllm-ascend-idle-dp-gdn-state"
GRAPH_INVARIANTS = "unit-vllm-ascend-npu-runtime-and-graph-invariants"
PARITY_GATES = "unit-vllm-ascend-verification-and-state-parity-gates"
DP_INVARIANTS = "unit-vllm-ascend-hccl-and-dp-collective-invariants"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"real corpus is unavailable: {path}")
    return path


@pytest.fixture
def operator_store(tmp_path):
    store = Store(tmp_path / "operators", "operators")
    store.graded.install_root(_require(OPERATORS))
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def transcript_store(tmp_path):
    store = Store(tmp_path / "transcript", "vllm-ascend")
    store.graded.install_root(_require(TRANSCRIPT))
    try:
        yield store
    finally:
        store.close()


def _revision(store, identity: str) -> int:
    return store.graded._load().units[identity].revision


def _ref(store, identity: str, revision: int | None = None) -> str:
    return store.graded.ref(identity, _revision(store, identity) if revision is None else revision)


def _case(case_id, *, query, conditions, probe, expected_status, expected_refs=None,
          loaded_refs=None, adopted_refs=None, oracle_refs=None, level="L1",
          retrieval_expected=True, applies_to=None):
    return {
        "case_id": case_id,
        "level": level,
        "query": query,
        "conditions": conditions,
        "probe_ref": probe,
        "expected_status": expected_status,
        "expected_refs": expected_refs or [],
        "loaded_refs": loaded_refs or [],
        "adopted_refs": adopted_refs or [],
        "oracle_refs": oracle_refs or [],
        "retrieval_expected": retrieval_expected,
        **({"applies_to": applies_to} if applies_to else {}),
    }


def _distractor(identity, *, conditions, observed, expected, title, body, keywords):
    return {
        "frontmatter": {
            "schema": "mindie-graded-unit/1",
            "id": identity,
            "revision": 1,
            "tier": "L2",
            "category": "sop",
            "evidence_strength": "single_observation",
            "domain_path": "distractors",
            "title": title,
            "inherits_from": [],
            "conditions": conditions,
            "triggers": {"error_signatures": [], "keywords": keywords},
            "provenance": {
                "source_type": "reviewed_distractor",
                "source_locator": "tiered-fixture@abcdef1:distractor.md",
                "evidence_refs": ["distractor-review"],
                "supersedes": [],
            },
        },
        "body": body,
        "observed": observed,
        "expected_status": expected,
        "label": "near_miss_or_stale",
    }


def _material(session: str, body: str, *, operation: str = "create", identity: str = "unit-tiered-sequence-test"):
    digest = hashlib.sha256(session.encode()).hexdigest()
    content = f"Verified session {session} evidence: the measured tiered sequence passed.\n"
    locator = "tiered-fixture@abcdef1:tiered-sequence.md"
    reference = f"run-{session}"
    frontmatter = {
        "schema": "mindie-graded-unit/1",
        "id": identity,
        "revision": 1,
        "tier": "L2",
        "category": "sop",
        "evidence_strength": "single_observation",
        "domain_path": "tiered/sequence",
        "title": "Tiered sequence decode SOP",
        "inherits_from": [],
        "conditions": {
            "soc": ["Ascend910B4"],
            "dtype": "bfloat16",
            "workload_phase": "decode",
        },
        "triggers": {"error_signatures": [], "keywords": ["tiered", "sequence", "decode", "sop"]},
        "provenance": {
            "source_type": "session_distill",
            "source_locator": locator,
            "source_digest": hashlib.sha256(content.encode()).hexdigest(),
            "evidence_refs": [reference],
            "supersedes": [],
        },
    }
    unit = GradedUnit.from_parts(frontmatter, body)
    evidence = {
        "kind": "authorized_transcript",
        "outcome": "verified",
        "source_locator": locator,
        "content": content,
        "evidence_refs": [reference],
        "repetitions": 1,
    }
    decision = {
        "schema_version": "mindie-distill-decision/1",
        "session_digest": digest,
        "evidence_inputs": ["authorized_transcript"],
        "action": "propose_graded_units",
        "drop_reason": None,
        "units": [
            {
                "target_unit_id": unit.id,
                "target_tier": "L2",
                "category": "sop",
                "evidence_strength": "single_observation",
                "domain_path": "tiered/sequence",
                "operation": operation,
                "frontmatter": unit.metadata,
                "markdown_body": unit.body,
            }
        ],
    }
    return digest, evidence, decision


# --------------------------------------------------------------- L1 and L2
@pytest.mark.tier0
def test_l4_guard_system_regression_scope_fallback_and_final_combination():
    baseline = {
        "fast": {"correct": True, "duration_ms": 100.0},
        "guard": {"correct": True, "duration_ms": 100.0},
    }
    candidate = {
        "fast": {"correct": True, "duration_ms": 80.0, "fallback": True},
        "guard": {"correct": True, "duration_ms": 120.0},
    }
    rejected = evaluate_l4_heldout_generalization(
        candidate_id="candidate-guard",
        baseline_results=baseline,
        candidate_results=candidate,
        guard_workloads=[{"input_id": "guard", "tolerance": 0.1}],
        candidate_artifact="guard-candidate",
    )
    assert rejected["guard_workloads"]["passed"] is False
    assert rejected["guard_failures"][0]["reason"] == "guard_regression"
    assert rejected["verdict"] == "rejected_guard_regression"
    assert rejected["accepted"] is False
    tolerated = evaluate_l4_heldout_generalization(
        candidate_id="candidate-guard",
        baseline_results=baseline,
        candidate_results=candidate,
        guard_workloads=[{"input_id": "guard", "tolerance": 0.25}],
        candidate_artifact="guard-candidate",
    )
    assert tolerated["guard_workloads"]["passed"] is True
    assert tolerated["accepted"] is True

    scope_bad = evaluate_l4_heldout_generalization(
        candidate_id="candidate-scope",
        baseline_results=baseline,
        candidate_results={
            "fast": {"correct": True, "duration_ms": 80.0},
            "guard": {"correct": True, "duration_ms": 100.0},
        },
        out_of_scope=["fast"],
    )
    assert scope_bad["l4_01_applicability"]["violations"][0]["reason"] == (
        "optimized_path_applied_outside_scope"
    )
    assert scope_bad["verdict"] == "rejected_out_of_scope_application"
    scope_good = evaluate_l4_heldout_generalization(
        candidate_id="candidate-scope",
        baseline_results=baseline,
        candidate_results=candidate,
        out_of_scope=["fast"],
    )
    assert scope_good["l4_01_applicability"]["violations"] == []
    assert scope_good["l4_01_applicability"]["correct_rate"] == 1.0

    ablations = {
        "run_id": "run-full",
        "conditions": {"soc": "Ascend910B4"},
        "full": {
            "run_id": "run-full", "speedup_ratio": 1.25,
            "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
        },
        "factors": ["A", "B"],
        "leave_one_out": {
            "A": {
                "run_id": "run-without-a", "speedup_ratio": 1.05,
                "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
            },
            "B": {
                "run_id": "run-without-b", "speedup_ratio": 1.245,
                "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
            },
        },
        "final_combination": {
            "run_id": "run-final", "speedup_ratio": 1.23, "verified": True,
            "compile_ok": True, "oracle_passed": True, "measurement_passed": True,
        },
    }
    verified = evaluate_l4_heldout_generalization(
        candidate_id="candidate-ablation",
        baseline_results={"case": {"correct": True, "duration_ms": 100.0}},
        candidate_results={"case": {"correct": True, "duration_ms": 80.0}},
        ablation=ablations,
    )
    assert verified["ablation"]["accepted"] is True
    assert verified["ablation"]["final_combination"]["schema"] == (
        "mindie-exploration-evaluation/1"
    )
    classes = {
        row["factor_id"]: row["classification"]
        for row in verified["ablation"]["leave_one_out"]
    }
    assert classes == {"A": "main_effect", "B": "within_noise_preserve"}
    reused = {
        **ablations,
        "final_combination": {**ablations["final_combination"], "reuses_full_candidate": True},
    }
    unverified = evaluate_l4_heldout_generalization(
        candidate_id="candidate-ablation",
        baseline_results={"case": {"correct": True, "duration_ms": 100.0}},
        candidate_results={"case": {"correct": True, "duration_ms": 80.0}},
        ablation=reused,
    )
    assert unverified["ablation"]["independent_final_validation"] is False
    assert unverified["ablation"]["accepted"] is False
    assert unverified["verdict"] == "rejected_ablation"


# --------------------------------------------------------------- X-01..X-05
@pytest.mark.tier1
def test_x01_to_x03_shared_premises_compaction_revision_and_domain_boundary(
    operator_store, transcript_store
):
    conditions = {
        "soc": "Ascend910B4", "dtype": "bfloat16", "op_family": "reduction",
        "workload": "fixed_shape", "reduction_axis": "last",
    }
    full = operator_store.graded.explain(
        refs=[_ref(operator_store, BOTTLENECK)], conditions=conditions, token_budget=16000
    )
    assert full["unresolved"] == []
    assert [row["id"] for row in full["requires"]] == [ALIGNMENT, ROOFLINE, REDUCTION_SOP]
    assert len({row["id"] for row in full["requires"]}) == len(full["requires"])
    assert full["usage"]["has_more"] is False
    known = [f"mindie://operators/units/{row['id']}@{row['revision']}" for row in full["requires"]]
    deduplicated = operator_store.graded.explain(
        refs=[_ref(operator_store, BOTTLENECK)], conditions=conditions,
        known_refs=known, token_budget=16000,
    )
    assert [row["id"] for row in deduplicated["units"]] == [BOTTLENECK]
    assert deduplicated["requires"] == []
    tight = operator_store.graded.explain(
        refs=[_ref(operator_store, BOTTLENECK)], conditions=conditions, token_budget=1000
    )
    assert tight["units"] == []
    assert tight["unresolved"][0]["status"] == "budget_excluded"
    assert tight["usage"]["has_more"] is True

    parent = _ref(operator_store, ALIGNMENT)
    full_parent = operator_store.graded.explain(
        refs=[parent], conditions=conditions, token_budget=16000
    )
    assert [row["id"] for row in full_parent["units"]] == [ALIGNMENT]
    body_retained = operator_store.graded.explain(
        refs=[parent], conditions=conditions, known_refs=[parent], token_budget=16000
    )
    assert body_retained["units"] == []
    bare_ref = operator_store.graded.explain(
        refs=[parent], conditions=conditions, known_refs=[ALIGNMENT], token_budget=16000
    )
    assert [row["id"] for row in bare_ref["units"]] == [ALIGNMENT]
    old_parent = operator_store.graded.explain(
        refs=[parent], conditions=conditions,
        known_refs=[operator_store.graded.ref(ALIGNMENT, 1)], token_budget=16000,
    )
    assert [row["id"] for row in old_parent["units"]] == [ALIGNMENT]

    current = operator_store.graded._load().units[ALIGNMENT]
    revised = GradedUnit.from_parts(
        {**current.metadata, "revision": current.revision},
        current.body + "\nRevised parent rule: keep the 32-byte tail invariant.\n",
        current.path,
    )
    operator_store.graded.install([revised], admitted_ids=[ALIGNMENT])
    new_revision = _revision(operator_store, ALIGNMENT)
    assert new_revision == current.revision + 1
    with pytest.raises(ValueError, match="replaced"):
        operator_store.graded.explain(
            operator_store.graded.ref(ALIGNMENT, current.revision)
        )
    old_batch = operator_store.graded.explain(
        refs=[operator_store.graded.ref(ALIGNMENT, current.revision)],
        conditions=conditions, token_budget=16000,
    )
    assert any(row["status"] == "removed_or_superseded" for row in old_batch["unresolved"])
    child = operator_store.graded.explain(
        refs=[_ref(operator_store, BOTTLENECK)], conditions=conditions, token_budget=16000
    )
    parent_row = next(row for row in child["requires"] if row["id"] == ALIGNMENT)
    assert parent_row["revision"] == new_revision
    cross_domain = transcript_store.graded.explain(
        refs=[f"mindie://operators/units/{ALIGNMENT}@{new_revision}"],
        conditions={}, token_budget=4096,
    )
    assert any(row["status"] == "not_found" for row in cross_domain["unresolved"])


@pytest.mark.tier1
def test_receipt_bridge_deduplicates_and_never_refutes_infra_failures(
    operator_store, monkeypatch
):
    identity = RMSNORM_L4
    revision = _revision(operator_store, identity)
    receipt = {
        "run_id": "run-bridge-pass",
        "task_id": "task-bridge",
        "harness": "dsh",
        "commit_sha": "a" * 40,
        "run_manifest_verified": True,
        "consulted_knowledge": [
            ConsultedKnowledgeRef(ref=_ref(operator_store, identity), applied=True)
        ],
        "verification": {"status": "pass", "verified": True},
        "primary_metrics": {"speedup_ratio": 1.2},
    }
    bridged = bridge_run_receipt_to_feedback(
        receipt, store=operator_store, root_hash="b" * 64
    )
    assert bridged["primary"]["pass"] is True and bridged["is_error"] is False
    assert [event["event_type"] for event in bridged["events"]] == [
        "discovered", "loaded", "applied", "evaluated",
    ]
    assert all(event["duplicate"] is False for event in bridged["events"])
    summary = summarize_usage_events(operator_store, identity, revision)
    assert summary["discovered"] == 1 and summary["loaded"] == 1
    assert summary["applied"] == 1 and summary["supported"] == 1

    replay = bridge_run_receipt_to_feedback(
        receipt, store=operator_store, root_hash="b" * 64
    )
    assert all(event["duplicate"] is True for event in replay["events"])
    assert summarize_usage_events(operator_store, identity, revision) == summary

    for run_id, status in (
        ("run-bridge-infra", "infra_failed"),
        ("run-bridge-invalid", "measurement_invalid"),
    ):
        result = bridge_run_receipt_to_feedback(
            {
                "run_id": run_id,
                "consulted_knowledge": [
                    {
                        "ref": _ref(operator_store, identity),
                        "applied": True,
                        "evaluation_outcome": "refuted",
                    }
                ],
                "verification": {"status": status},
            },
            store=operator_store,
            root_hash="b" * 64,
        )
        assert result["primary"]["outcome"] == "infra_failed"
        evaluated = next(event for event in result["events"] if event["event_type"] == "evaluated")
        assert evaluated["outcome"] == "inconclusive"
    summary = summarize_usage_events(operator_store, identity, revision)
    assert summary["refuted"] == 0 and summary["inconclusive"] == 2

    def fail_accounting(*args, **kwargs):
        raise TimeoutError("local accounting budget exhausted")

    monkeypatch.setattr(
        "mindie_bench.evaluation.knowledge_tiered.record_usage_event", fail_accounting
    )
    protected = bridge_run_receipt_to_feedback(
        {**receipt, "run_id": "run-bridge-timeout"},
        store=operator_store,
        root_hash="b" * 64,
    )
    assert protected["primary"]["pass"] is True
    assert protected["is_error"] is False
    assert protected["accounting_error"]["stage"] == "feedback"
    assert protected["accounting_error"]["timeout"] is True
    assert protected["diagnostics"]


@pytest.mark.tier1
def test_x04_to_x06_sequential_a_to_e_learning_growth_rare_recall_and_idempotence(operator_store):
    identity = "unit-tiered-sequence-test"
    digest_a, evidence_a, decision_a = _material(
        "a", "Use the verified tiered sequence SOP with a 32-byte aligned tail.\n"
    )
    episode_a = {
        "session": "A", "run_id": "run-a", "task_id": "task-a", "harness": "dsh",
        "commit_sha": "a" * 40, "verification": {"status": "pass", "verified": True},
        "session_digest": digest_a, "evidence_inputs": [evidence_a],
        "proposal": decision_a, "consulted_knowledge": [],
    }
    empty = Store(operator_store.root / "empty", "operators")
    try:
        with pytest.raises(ValueError, match="seed the graded repository"):
            run_sequential_learning_episodes(
                [episode_a], store=empty, root_hash="c" * 64
            )
    finally:
        empty.close()
    revision_1 = 1

    digest_c, evidence_c, decision_c = _material(
        "c",
        "Use the verified tiered sequence SOP with a 32-byte aligned tail.\n"
        "Boundary: this fastpath is decode-only and must not be used for prefill.\n",
        operation="merge_into_existing",
    )
    episodes = [
        episode_a,
        {
            "session": "B", "run_id": "run-b", "task_id": "task-b",
            "verification": {"status": "pass", "verified": True},
            "consulted_knowledge": [
                {"ref": _ref(operator_store, identity, revision_1), "applied": True}
            ],
        },
        {
            "session": "C", "run_id": "run-c", "task_id": "task-c",
            "verification": {"status": "failed", "verified": True},
            "session_digest": digest_c, "evidence_inputs": [evidence_c],
            "proposal": decision_c,
            "consulted_knowledge": [
                {"ref": _ref(operator_store, identity, revision_1), "applied": True}
            ],
        },
        {
            "session": "D", "run_id": "run-d", "task_id": "task-d",
            "verification": {"status": "pass", "verified": True},
            "consulted_knowledge": [
                {"ref": _ref(operator_store, identity, 2), "applied": True}
            ],
        },
        {
            "session": "E", "run_id": "run-e", "task_id": "task-e",
            "verification": {"status": "pass", "verified": True},
            "consulted_knowledge": [
                {"ref": _ref(operator_store, identity, 2), "applied": True}
            ],
        },
        {
            "session": "F", "run_id": "run-d", "task_id": "task-d",
            "verification": {"status": "pass", "verified": True},
            "consulted_knowledge": [
                {"ref": _ref(operator_store, identity, 2), "applied": True}
            ],
        },
    ]
    initial_units = len(operator_store.graded._load().units)
    result = run_sequential_learning_episodes(
        episodes,
        store=operator_store,
        root_hash="c" * 64,
        rare_condition_probes=[
            {
                "probe_id": "rare-decode", "expected_unit_id": identity,
                "query": "tiered sequence decode SOP",
                "conditions": {
                    "soc": "Ascend910B4", "dtype": "bfloat16", "workload_phase": "decode",
                },
            }
        ],
    )
    assert result["accounting_errors"] == []
    assert result["duplicate_runs"] == 1
    assert next(row for row in result["episodes"] if row["session"] == "F")["replayed"] is True
    assert result["gate"]["duplicate_events_do_not_add_evidence"] is True
    revisions = [
        row["revision"] for row in result["version_chain"] if row["unit_id"] == identity
    ]
    assert revisions == [1, 2]
    assert result["growth"]["initial_units"] == initial_units
    assert result["growth"]["final_units"] == initial_units + 1
    assert result["growth"]["delta_active_units"] == 1
    assert result["growth"]["g_units"] == 0.5
    assert result["growth"]["g_body"] > 0
    assert result["rare_condition_recall"]["recall_rate"] == 1.0
    probe = result["rare_condition_recall"]["probes"][0]
    assert probe["recalled"] is True
    assert probe["evidence"] == "repository.query+explain"
    assert identity in probe["retrieved_refs"] and identity in probe["loaded_refs"]

    first_revision = summarize_usage_events(operator_store, identity, 1)
    second_revision = summarize_usage_events(operator_store, identity, 2)
    assert first_revision["applied"] == 2
    assert first_revision["evaluated"] == 2
    assert first_revision["supported"] == 1 and first_revision["refuted"] == 1
    assert second_revision["applied"] == 2
    assert second_revision["evaluated"] == 2 and second_revision["supported"] == 2


@pytest.mark.tier1
def test_sequential_distillation_timeout_preserves_verified_primary(operator_store):
    initial_units = len(operator_store.graded._load().units)
    digest, evidence, decision = _material(
        "timeout", "Use the verified tiered sequence SOP with a 32-byte aligned tail.\n"
    )

    def timeout_distiller(**kwargs):
        raise TimeoutError("distillation budget exhausted")

    result = run_sequential_learning_episodes(
        [
            {
                "session": "A", "run_id": "run-timeout", "task_id": "task-timeout",
                "verification": {"status": "pass", "verified": True},
                "session_digest": digest, "evidence_inputs": [evidence],
                "proposal": decision,
            }
        ],
        store=operator_store,
        root_hash="d" * 64,
        distiller=timeout_distiller,
    )
    episode = result["episodes"][0]
    assert episode["primary"]["pass"] is True
    assert episode["is_error"] is False
    assert episode["distillation"]["status"] == "accounting_error"
    assert episode["distillation"]["timeout"] is True
    assert episode["accounting_error"]["stage"] == "distillation"
    assert episode["diagnostics"]
    assert result["accounting_errors"][0]["stage"] == "distillation"
    assert result["gate"]["primary_preserved_on_accounting_error"] is True
    assert result["growth"]["final_units"] == initial_units


# --------------------------------------------------------------- DSH host
class _PatchLoader(yaml.SafeLoader):
    pass


_PatchLoader.add_constructor(
    "tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node)
)


@pytest.mark.tier1
def test_dsh_profiles_force_standard_tools_and_max_reasoning_effort(tmp_path):
    fake = tmp_path / "dsh"
    fake.write_text(
        "#!/bin/sh\nset -e\n"
        "profile=\"$DSH_HOME/profiles/bench\"\n"
        "mkdir -p \"$profile\"\n"
        "printf '%s\\n' '{\"name\":\"dsh-profile-bench\",\"private\":true,"
        "\"dsh\":{\"profile\":{\"bundles\":[]}}}' > \"$profile/package.json\"\n"
        "printf '%s\\n' '[]' > \"$profile/cordis.patch.yml\"\n"
        "echo '[]'\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    config = ExperimentConfig(
        schema_version="1.0",
        harness=HarnessConfig(
            name="dsh", executable=str(fake), model="deepseek-flash",
            expected_version="0.2.0-rc.2",
            provider_url="https://api.deepseek.com/anthropic/v1",
            api_key_env="MINDIE_BENCH_RELAY_KEY",
        ),
    )
    root = tmp_path / "profile"
    root.mkdir()
    build_dsh_profile(root, {"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}, config, {}, {})
    patch = yaml.load(
        (root / "dsh-home" / "profiles" / "bench" / "cordis.patch.yml").read_text(),
        Loader=_PatchLoader,
    )
    rows = {row["id"]: row for row in patch if "id" in row}
    model = rows["llm-pi-ai"]["config"]["providers"]["bench"]["models"][0]
    assert model["reasoningEfforts"] == DSH_REASONING_EFFORTS
    assert rows["agent-default-model"]["config"]["reasoningEffort"] == "max"
    assert rows["tools"]["name"] == "@deepseek-ai/dsh-tools"
    assert rows["tools"]["config"] == {"mode": "native"}
