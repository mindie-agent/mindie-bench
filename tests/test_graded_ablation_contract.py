"""C4 producer to C6 consumer contract; synthetic data, no hardware/model claim."""

from __future__ import annotations

import hashlib

import pytest
from test_ablation import run as measured_run

from mindie_bench.evaluation.ablation import AblationFactor, AblationGate

WORKLOAD = [
    {
        "input_shape": "synthetic-case",
        "workload_case_sha256": hashlib.sha256(b"contract-fixture-case").hexdigest(),
        "samples_per_shape": 4,
    }
]


def _receipt(pruning):
    def run(identity, factors, duration):
        return measured_run(
            identity,
            factors,
            duration,
            conditions={
                "soc": "Ascend910B",
                "cann": "8.1",
                "workload_scope": "synthetic",
                "workload": WORKLOAD,
            },
            evidence_refs=[f"synthetic/{identity}"],
        )

    return AblationGate(min_speedup=1.05).evaluate(
        task_id="synthetic-c4-c6-contract",
        factors=[AblationFactor(factor_id="A"), AblationFactor(factor_id="B")],
        baseline=run("baseline", [], 100),
        full_candidate=run("full", ["A", "B"], 90),
        ablations=[run("without-A", ["B"], 90), run("without-B", ["A"], 90)],
        pruning_runs=[run("prune-A", ["B"], 90)] if pruning else [],
    )


@pytest.mark.parametrize(
    ("pruning", "cann", "accepted"),
    [(True, ">=8,<9", True), (False, ">=8,<9", False), (True, ">=9", False)],
)
def test_c4_receipt_is_consumed_by_real_c6_l4_admission(pruning, cann, accepted):
    distill = pytest.importorskip("mindie_knowledge.graded.distill")
    receipt = _receipt(pruning)
    content = receipt.model_dump_json()
    locator = "mindie-bench-contract@abcdef1:c4-synthetic.json"
    digest = hashlib.sha256(content.encode()).hexdigest()
    refs = ["synthetic/c4-receipt"]
    # This is a pure contract fixture at the post-redaction boundary. No
    # scanner or organizer is invoked, and no unit is written to a corpus.
    request = {
        "schema_version": "mindie-distill-request/1",
        "session_digest": "a" * 64,
        "task_state": "verified_milestone",
        "evidence_inputs": ["ablation_receipt"],
        "sanitized_inputs": [
            dict(
                kind="ablation_receipt",
                outcome="verified",
                source_locator=locator,
                source_digest=digest,
                content=content,
                evidence_refs=refs,
                repetitions=1,
            )
        ],
        "redaction_rules": [],
        "redaction_completed": True,
        "model_calls": 0,
    }
    request["request_digest"] = distill._sha(request)
    frontmatter = {
        "schema": "mindie-graded-unit/1",
        "id": "unit-synthetic-c4",
        "tier": "L4",
        "category": "mutator",
        "evidence_strength": "single_observation",
        "domain_path": "synthetic/contract",
        "title": "Synthetic C4 contract",
        "inherits_from": [],
        "conditions": {
            "soc": ["Ascend910B"],
            "cann": cann,
            "workload_scope": "synthetic",
            "workload": {"exact_set": WORKLOAD},
        },
        "triggers": {"error_signatures": [], "keywords": ["synthetic"]},
        "provenance": {
            "source_type": "session_distill",
            "source_locator": locator,
            "source_digest": digest,
            "evidence_refs": refs,
            "supersedes": [],
        },
    }
    decision = {
        "schema_version": "mindie-distill-decision/1",
        "session_digest": request["session_digest"],
        "evidence_inputs": request["evidence_inputs"],
        "action": "propose_graded_units",
        "drop_reason": None,
        "units": [
            {
                "target_unit_id": frontmatter["id"],
                "target_tier": "L4",
                "category": "mutator",
                "evidence_strength": "single_observation",
                "domain_path": "synthetic/contract",
                "operation": "create",
                "frontmatter": frontmatter,
                "markdown_body": "Synthetic compatibility fixture; no measured hardware result.",
            }
        ],
    }
    if accepted:
        assert distill.validate_decision(decision, request) == decision
    else:
        with pytest.raises(ValueError, match="baseline-bound verified final-combination"):
            distill.validate_decision(decision, request)
