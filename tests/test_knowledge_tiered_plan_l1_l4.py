# ruff: noqa: F811, I001
"""High-signal tiered-knowledge contracts over the real corpora.

Tier 0 tests are pure in-memory decisions. Tier 1 tests create isolated SQLite
stores, copy real graded corpora, and run the production organizer.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from mindie_bench.evaluation.knowledge_tiered import (
    evaluate_applicability_and_distractors,
    evaluate_l4_heldout_generalization,
)
from mindie_bench.evaluation.knowledge_tiered_support import (
    assess_stagnation,
    build_exploration_context,
)

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


from test_knowledge_tiered_plan import operator_store, transcript_store  # noqa: E402,F401
from test_knowledge_tiered_plan import (  # noqa: E402
    _case,
    _distractor,
    _ref,
    _require,
)


@pytest.mark.tier1
def test_l1_condition_boundaries_missing_facts_and_natural_trigger(tmp_path):
    _require(OPERATORS)
    conditions = ("no_knowledge", "oracle", "normal_retrieval")
    cases = [
        _case(
            "L1-01-matched", query="Ascend 910B memory alignment tiling reduction UB",
            conditions={"soc": "Ascend910B4", "dtype": "bfloat16"},
            probe=ALIGNMENT, expected_status="matched", expected_refs=[ALIGNMENT],
            loaded_refs=[ALIGNMENT], adopted_refs=[ALIGNMENT], oracle_refs=[ALIGNMENT],
        ),
        _case(
            "L1-01-other-soc", query="Ascend 910B memory alignment tiling reduction UB",
            conditions={"soc": "Ascend910C", "dtype": "bfloat16"},
            probe=ALIGNMENT, expected_status="mismatch",
        ),
        _case(
            "L1-01-unknown-soc", query="Ascend 910B memory alignment tiling reduction UB",
            conditions={"dtype": "bfloat16"}, probe=ALIGNMENT,
            expected_status="condition_unknown",
        ),
        _case(
            "L1-02-current-triton", query="Triton constexpr specialization contract",
            conditions={"kernel_language": "triton"}, probe=TRITON_CONTRACT,
            expected_status="matched", expected_refs=[TRITON_CONTRACT],
            loaded_refs=[TRITON_CONTRACT], adopted_refs=[TRITON_CONTRACT],
            oracle_refs=[TRITON_CONTRACT], level="L1",
        ),
        _case(
            "L1-02-other-stack", query="Triton constexpr specialization contract",
            conditions={"kernel_language": "cuda"}, probe=TRITON_CONTRACT,
            expected_status="mismatch",
        ),
        _case(
            "L1-03-measured-boundary", query="operator roofline four view measurement",
            conditions={"soc": "Ascend910B4", "dtype": "float16"}, probe=ROOFLINE,
            expected_status="matched", expected_refs=[ROOFLINE],
            loaded_refs=[ROOFLINE], adopted_refs=[ROOFLINE], oracle_refs=[ROOFLINE],
        ),
        _case(
            "L1-03-uncovered-dtype", query="operator roofline four view measurement",
            conditions={"soc": "Ascend910B4", "dtype": "float64"}, probe=ROOFLINE,
            expected_status="mismatch",
        ),
        _case(
            "L1-03-missing-dtype", query="operator roofline four view measurement",
            conditions={"soc": "Ascend910B4"}, probe=ROOFLINE,
            expected_status="condition_unknown",
        ),
        _case(
            "L1-04-no-hit", query=None, conditions={}, probe=None,
            expected_status=None, retrieval_expected=False,
        ),
        _case(
            "L1-04-natural-trigger", query="unrelated quantum banana",
            conditions={}, probe=None, expected_status=None,
            retrieval_expected=False,
        ),
    ]
    result = evaluate_applicability_and_distractors(
        cases, corpus_root=OPERATORS, workspace=tmp_path, conditions=conditions
    )
    assert result["source_corpus_mutated"] is False
    for name in conditions:
        metrics = result["conditions"][name]["metrics"]
        assert metrics["applicability_accuracy"] == 1.0
        assert result["conditions"][name]["natural_trigger_zero_query_is_not_failure"] is True
    normal = result["conditions"]["normal_retrieval"]
    no_hit = next(row for row in normal["cases"] if row["case_id"] == "L1-04-no-hit")
    natural = next(row for row in normal["cases"] if row["case_id"] == "L1-04-natural-trigger")
    assert no_hit["query_calls"] == 0 and no_hit["passed"] is True
    assert natural["query_calls"] == 0 and natural["passed"] is True
    assert no_hit["natural_trigger"] is True and natural["natural_trigger"] is True
    assert normal["metrics"]["recall_at_k"] == 1.0
    assert normal["metrics"]["premise_completeness_rate"] == 1.0
    assert result["conditions"]["no_knowledge"]["queries"] == 0
    assert result["conditions"]["oracle"]["queries"] == 0
    tampered = {
        row["case_id"]: row["applicability"] for row in normal["cases"]
    }
    assert tampered["L1-01-other-soc"] == "mismatch"
    assert tampered["L1-01-unknown-soc"] == "condition_unknown"


@pytest.mark.tier1
def test_l2_complete_sop_loading_and_premise_failure_stop(tmp_path):
    _require(OPERATORS)
    _require(TRANSCRIPT)
    operators = [
        _case(
            "L2-01-reduction-sop", query="vector reduction FP32 accumulator UB alignment SOP",
            conditions={
                "soc": "Ascend910B4", "dtype": "bfloat16", "op_family": "reduction",
                "reduction_axis": "last",
            },
            probe=REDUCTION_SOP, expected_status="matched", expected_refs=[REDUCTION_SOP],
            loaded_refs=[REDUCTION_SOP, ALIGNMENT, ROOFLINE], adopted_refs=[REDUCTION_SOP],
            oracle_refs=[REDUCTION_SOP], level="L2",
        ),
        _case(
            "L2-03-triton-recompile", query="Triton dynamic scalar specialization recompilation",
            conditions={"kernel_language": "triton"}, probe=TRITON_SOP,
            expected_status="matched", expected_refs=[TRITON_SOP],
            loaded_refs=[TRITON_SOP, TRITON_CONTRACT], adopted_refs=[TRITON_SOP],
            oracle_refs=[TRITON_SOP], level="L2",
        ),
        _case(
            "L2-04-missing-premise", query="vector reduction FP32 accumulator UB alignment SOP",
            conditions={"dtype": "bfloat16", "op_family": "reduction", "reduction_axis": "last"},
            probe=REDUCTION_SOP, expected_status="condition_unknown", level="L2",
        ),
    ]
    result = evaluate_applicability_and_distractors(
        operators, corpus_root=OPERATORS, workspace=tmp_path / "operators",
        conditions=("normal_retrieval",),
    )
    normal = result["conditions"]["normal_retrieval"]
    assert normal["metrics"]["applicability_accuracy"] == 1.0
    assert normal["metrics"]["premise_completeness_rate"] == 1.0
    missing = next(row for row in normal["cases"] if row["case_id"] == "L2-04-missing-premise")
    assert missing["applicability"] == "condition_unknown"
    assert missing["adopted_refs"] == [] and missing["passed"] is True

    transcript_case = _case(
        "L2-02-aclgraph", query="ACLGraph capture invalidation dynamic shape stream sync",
        conditions={
            "graph_mode": "aclgraph", "npu_model": "Ascend910B4",
            "repository": "vllm-ascend",
        },
        probe=ACLGRAPH_SOP, expected_status="matched", expected_refs=[ACLGRAPH_SOP],
        loaded_refs=[ACLGRAPH_SOP, GRAPH_INVARIANTS, PARITY_GATES],
        adopted_refs=[ACLGRAPH_SOP], oracle_refs=[ACLGRAPH_SOP], level="L2",
    )
    transcript = evaluate_applicability_and_distractors(
        [transcript_case], corpus_root=TRANSCRIPT, workspace=tmp_path / "transcript",
        conditions=("normal_retrieval",),
    )["conditions"]["normal_retrieval"]
    assert transcript["passed_cases"] == 1
    row = transcript["cases"][0]
    assert ACLGRAPH_SOP in row["retrieved_refs"]
    assert row["applicability"] == "matched"
    assert transcript["metrics"]["premise_completeness_rate"] == 1.0


# --------------------------------------------------------------- distractors
@pytest.mark.tier1
def test_distractors_are_injected_in_a_copy_and_wrong_adoption_is_counted(tmp_path):
    _require(OPERATORS)
    matched_conditions = {
        "activation": "sigmoid", "dtype": "bfloat16", "soc": "Ascend910B4",
        "target_file": "vllm_ascend/ops/layernorm.py",
    }
    unknown = _distractor(
        "unit-distractor-unknown-premise",
        conditions={**matched_conditions, "cann": ">=8,<9"},
        observed=matched_conditions,
        expected="condition_unknown",
        title="Near-miss residual RMSNorm fastpath",
        body="Apply the residual-free RMSNorm fastpath on Ascend910B4.\n",
        keywords=["residual", "rmsnorm", "fastpath", "layernorm"],
    )
    stale = _distractor(
        "unit-distractor-stale-soc",
        conditions={**matched_conditions, "soc": "Ascend910C"},
        observed=matched_conditions,
        expected="mismatch",
        title="Stale residual RMSNorm fastpath for Ascend910C",
        body="Old fastpath for another SoC.\n",
        keywords=["residual", "rmsnorm", "fastpath"],
    )
    cases = [
        _case(
            "L2-01-real-fastpath", query="residual free rmsnorm fastpath Ascend910B bfloat16",
            conditions=matched_conditions, probe=RMSNORM_L4, expected_status="matched",
            expected_refs=[RMSNORM_L4],
            loaded_refs=[RMSNORM_L4, ALIGNMENT, ROOFLINE], adopted_refs=[RMSNORM_L4],
            level="L2", applies_to=["normal_retrieval", "with_distractors"],
        ),
        _case(
            "L1-04-unknown-distractor", query="residual free rmsnorm fastpath Ascend910B bfloat16",
            conditions=matched_conditions, probe="unit-distractor-unknown-premise",
            expected_status="condition_unknown", expected_refs=[RMSNORM_L4],
            adopted_refs=["unit-distractor-unknown-premise"], level="L1",
            applies_to=["with_distractors"],
        ),
    ]
    result = evaluate_applicability_and_distractors(
        cases, corpus_root=OPERATORS, workspace=tmp_path, distractors=[unknown, stale],
        conditions=("normal_retrieval", "with_distractors"),
    )
    assert result["source_corpus_mutated"] is False
    assert result["annotation_errors"] == []
    assert not (OPERATORS / "distractors").exists()
    normal = result["conditions"]["normal_retrieval"]
    assert normal["metrics"]["wrong_adoption_rate"] is None
    disrupted = result["conditions"]["with_distractors"]
    assert disrupted["metrics"]["wrong_adoption_rate"] == 0.5
    assert disrupted["metric_counts"]["wrong_adoption_rate"] == {
        "value": 0.5, "numerator": 1, "denominator": 2, "missing": 0,
    }
    assert disrupted["metrics"]["applicability_accuracy"] == 1.0
    assert disrupted["metrics"]["premise_completeness_rate"] == 1.0
    unknown_row = next(
        row for row in disrupted["cases"] if row["case_id"] == "L1-04-unknown-distractor"
    )
    assert unknown_row["passed"] is False
    assert unknown_row["applicability"] == "condition_unknown"
    assert "wrong_adoption:unit-distractor-unknown-premise" in unknown_row["errors"]
    assert "unit-distractor-unknown-premise" in unknown_row["retrieved_refs"]
    real_row = next(row for row in disrupted["cases"] if row["case_id"] == "L2-01-real-fastpath")
    assert real_row["applicability"] == "matched" and real_row["passed"] is True


# --------------------------------------------------------------- L3 contracts
@pytest.mark.tier1
def test_l3_bottleneck_triage_and_idle_dp_state_are_condition_bound(
    operator_store, transcript_store
):
    op_conditions = {
        "soc": "Ascend910B4", "dtype": "bfloat16", "op_family": "reduction",
        "workload": "fixed_shape", "reduction_axis": "last",
    }
    for family in ("reduction", "norm", "matmul"):
        conditions = {**op_conditions, "op_family": family}
        found = operator_store.graded.query(
            "operator bottleneck convergence memory compute host launch",
            conditions=conditions, include_unknown=True, navigation_only=True, limit=5,
        )
        assert BOTTLENECK in {
            row["id"] for row in [*found["cards"], *found["foundation_cards"]]
        }
    closure = operator_store.graded.explain(
        refs=[_ref(operator_store, BOTTLENECK)], conditions=op_conditions,
        token_budget=16000,
    )
    assert closure["unresolved"] == []
    assert all(row["ready_to_apply"] for row in closure["units"])
    mismatch = operator_store.graded.explain(
        refs=[_ref(operator_store, BOTTLENECK)],
        conditions={**op_conditions, "op_family": "conv"}, token_budget=16000,
    )
    assert any(row["status"] == "condition_mismatch" for row in mismatch["unresolved"])

    idle_conditions = {
        "dp": 2,
        "idle_rank_state": "no_scheduled_requests",
        "peer_rank_state": "active_requests",
        "affected_state": ["conv_state", "ssm_state"],
        "model": "Kimi K3",
        "npu_count": 2,
        "npu_model": "Ascend910B4",
        "repository": "vllm-ascend-idle-dp",
    }
    found = transcript_store.graded.query(
        "idle DP GDN recurrent state conv_state ssm_state",
        conditions=idle_conditions, include_unknown=True, navigation_only=True, limit=5,
    )
    assert IDLE_DP in {
        row["id"] for row in [*found["cards"], *found["foundation_cards"]]
    }
    idle = transcript_store.graded.explain(
        refs=[_ref(transcript_store, IDLE_DP)], conditions=idle_conditions,
        token_budget=16000,
    )
    assert idle["unresolved"] == []
    assert {DP_INVARIANTS, PARITY_GATES} <= {row["id"] for row in idle["requires"]}
    wrong_state = transcript_store.graded.explain(
        refs=[_ref(transcript_store, IDLE_DP)],
        conditions={**idle_conditions, "affected_state": ["ssm_state"]},
        token_budget=16000,
    )
    assert any(row["status"] == "condition_mismatch" for row in wrong_state["unresolved"])
    missing_peer = transcript_store.graded.explain(
        refs=[_ref(transcript_store, IDLE_DP)],
        conditions={key: value for key, value in idle_conditions.items() if key != "peer_rank_state"},
        token_budget=16000,
    )
    assert any(row["status"] == "condition_unknown" for row in missing_peer["unresolved"])


@pytest.mark.tier0
def test_l3_stagnation_repairs_and_native_subcontext_drops_l3_guidance():
    repeated = assess_stagnation(
        [
            {"run_id": "run-1", "verdict": "unsupported"},
            {"run_id": "run-1", "verdict": "unsupported"},
            {"run_id": "run-2", "verdict": "supported"},
        ]
    )
    assert repeated["status"] == "progressing" and repeated["stalled_count"] == 0
    repair = assess_stagnation(
        [
            {"run_id": "run-3", "outcome": "infra_failed"},
            {"run_id": "run-4", "verdict": "inconclusive"},
        ]
    )
    assert repair["status"] == "repair_execution_conditions" and repair["stalled"] is False
    stalled = assess_stagnation(
        [
            {"run_id": "run-5", "verdict": "unsupported"},
            {"run_id": "run-6", "verdict": "unsupported"},
        ]
    )
    assert stalled["stalled"] is True and stalled["stalled_count"] == 2
    reset = assess_stagnation(
        [
            {"run_id": "run-7", "verdict": "unsupported"},
            {"run_id": "run-8", "new_observation": True},
            {"run_id": "run-9", "verdict": "unsupported"},
        ]
    )
    assert reset["status"] == "monitoring" and reset["stalled_count"] == 1

    old_l3 = {
        "tier": "L3", "id": "unit-l3-old-guidance",
        "workflow": "retry the stale tiling forever",
        "inherited_rules": ["old-l3-premise-that-must-not-mount"],
    }
    context = build_exploration_context(
        goal="recover decode throughput",
        retrieved_units=[
            old_l3,
            {"tier": "L1", "id": ALIGNMENT, "content": "Preserve dtype and alignment."},
            {"tier": "L2", "id": REDUCTION_SOP, "content": "Synchronize before timing."},
            {"tier": "L4", "id": RMSNORM_L4, "content": "Use the residual-free fastpath."},
        ],
        bottleneck_summary={"bottleneck": "vector-bound"},
        ledger_entries=[
            {
                "run_id": "run-old", "verdict": "refuted",
                "mutation_summary": "old tiling", "reason": "slower",
                "conditions": {"soc": "Ascend910B4"}, "axes": {"evidence_validity": "valid"},
            }
        ],
        conditions={"soc": "Ascend910B4", "dtype": "bfloat16"},
        native_subcontext=True,
    )
    assert context["native_subcontext"] is True
    assert context["stripped_l3_ids"] == ["unit-l3-old-guidance"]
    serialized = json.dumps(context)
    assert "retry the stale tiling forever" not in serialized
    assert "old-l3-premise-that-must-not-mount" not in serialized
    assert context["mounted_units"] and all(row["tier"] != "L3" for row in context["mounted_units"])
    assert context["conditional_avoid"][0]["run_id"] == "run-old"
    other = build_exploration_context(
        goal="recover decode throughput", retrieved_units=[old_l3],
        bottleneck_summary={},
        ledger_entries=[
            {
                "run_id": "run-old", "verdict": "refuted", "mutation_summary": "old tiling",
                "conditions": {"soc": "Ascend910B4"}, "axes": {"evidence_validity": "valid"},
            }
        ],
        conditions={"soc": "Ascend910C"}, native_subcontext=True,
    )
    assert other["conditional_avoid"] == []


# --------------------------------------------------------------- L4 frozen
@pytest.mark.tier0
def test_l4_freezes_and_classifies_every_heldout_input():
    baseline = {
        "both": {"correct": True, "duration_ms": 100.0},
        "retained": {"correct": True, "duration_ms": 50.0},
        "regressed": {"correct": True, "duration_ms": 100.0},
        "both-fail": {"correct": False},
        "improved": {"correct": False},
        "no-timing": {"correct": True},
        "broken-result": {"correct": True},
    }
    candidate = {
        "both": {"correct": True, "duration_ms": 80.0},
        "retained": {"correct": True, "duration_ms": 50.0},
        "regressed": {"correct": False},
        "both-fail": {"correct": False},
        "improved": {"correct": True},
        "no-timing": {"correct": True},
    }
    result = evaluate_l4_heldout_generalization(
        candidate_id="candidate-rmsnorm",
        baseline_results=baseline,
        candidate_results=candidate,
        candidate_artifact="frozen-rmsnorm-candidate",
    )
    assert result["candidate_freeze"]["frozen"] is True
    assert result["candidate_freeze"]["heldout_input_ids"] == sorted(
        {*baseline, *candidate}
    )
    assert set(result["buckets"]["both_pass"]) == {"both", "retained", "no-timing"}
    assert result["buckets"]["opt_regression"] == ["regressed"]
    assert result["buckets"]["both_fail"] == ["both-fail"]
    assert result["buckets"]["opt_improvement"] == ["improved"]
    assert result["buckets"]["evaluation_error"] == ["broken-result"]
    assert result["counts"] == {
        "both_pass": 3, "opt_regression": 1, "both_fail": 1,
        "opt_improvement": 1, "evaluation_error": 1,
    }
    assert result["correctness_retention_rate"] == 0.75
    assert result["correctness_retention"]["numerator"] == 3
    assert result["correctness_retention"]["denominator"] == 4
    assert result["speedup"]["comparable_inputs"] == ["both", "retained"]
    assert result["speedup"]["excluded_inputs"] == ["no-timing"]
    assert result["speedup"]["value"] == pytest.approx(math.sqrt(100 / 80), rel=1e-6)
    broken = evaluate_l4_heldout_generalization(
        candidate_id="candidate-broken",
        baseline_results={"a": {"correct": False}},
        candidate_results={"a": {"correct": False}},
    )
    assert broken["baseline_broken"] is True
    assert broken["correctness_retention_rate"] is None
    assert broken["verdict"] == "baseline_broken"


