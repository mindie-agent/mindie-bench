import copy
import json
from dataclasses import replace

import pytest

from mindie_bench.evaluation.coverage import (
    MetricEvidence,
    MetricTarget,
    assess_coverage,
    default_targets,
)
from mindie_bench.models import TaskSpec


def task(identity="task", *, family="a", gradient="near_exact", npu_count=1, evidence="verified"):
    return TaskSpec.model_validate(
        {
            "schema_version": "1.0",
            "task": {
                "instance_id": identity,
                "domain_category": "operator_correctness",
                "difficulty": "easy",
                "evidence": evidence,
                "cluster": "one-cluster",
                "source": {"repository": "upstream/source", "fix_commit": family * 40},
            },
            "mindie_knowledge": {"knowledge_gradient": gradient},
            "environment": {
                "l3" if npu_count else "l2": {
                    "repo_path": "/not-read",
                    "base_commit": "b" * 40,
                    "npu_count": npu_count,
                }
            },
        }
    )


def assess(tasks, target, evidence=None, *, groups=("G0", "G2"), repetitions=3):
    return assess_coverage(
        tasks,
        groups=groups,
        repetitions=repetitions,
        targets=[target],
        evidence={target.metric: evidence} if evidence is not None else None,
    ).metrics[0]


def complete(target, *, ids=("task",), samples=None, **kwargs):
    samples = len(ids) if samples is None else samples
    return MetricEvidence(
        observed_samples=samples,
        valid_samples=samples,
        observed_task_ids=ids,
        field_samples={key: samples for key in target.required_fields},
        protocol_samples={key: 1 for key in target.required_protocols},
        **kwargs,
    )


def test_planned_repetitions_and_verified_manifest_are_not_observations():
    result = assess([task()], MetricTarget("completion", "Completion", min_tasks=1))
    assert result.status == "not_measured"
    assert result.valid_samples == 0
    assert result.declared_tasks == 1
    assert result.observed_tasks == 0
    assert "planned tasks/repetitions are not measurements" in " ".join(result.reasons)


def test_repeated_triggers_and_clusters_do_not_create_independent_families():
    tasks = [task("seed"), task("eval")]
    target = MetricTarget("diversity", "Independent defects", min_families=2)
    result = assess(tasks, target, complete(target, ids=("seed", "eval"), samples=100))
    assert result.status == "insufficient_sample"
    assert result.declared_families == result.observed_families == 1
    assert result.observed_tasks == 2
    tasks[1].task.source["fix_commit"] = "c" * 40
    assert assess(tasks, target, complete(target, ids=("seed", "eval"))).status == "ready"


def test_source_identity_missing_is_not_a_success_or_a_report_only_fix():
    target = MetricTarget("metric", "Question", required_fields=("source_identity", "outcome"))
    original = MetricEvidence(
        observed_samples=2,
        valid_samples=2,
        observed_task_ids=("task",),
        field_samples={"outcome": 2, "source_identity": 1},
    )
    result = assess([task()], target, original)
    assert result.status == "not_measured"
    assert result.missing_field_samples == {"source_identity": 1}
    assert original.field_samples == {"outcome": 2, "source_identity": 1}


def test_actual_repetitions_required_even_when_plan_requests_three():
    target = MetricTarget("reliability", "Repeated outcomes", min_repetitions=3)
    evidence = complete(target, samples=3, task_repetitions={"task": (1,)})
    assert assess([task()], target, evidence).status == "insufficient_sample"
    evidence = replace(evidence, task_repetitions={"task": (1, 2, 3)})
    assert assess([task()], target, evidence).status == "ready"


def test_baseline_ceiling_blocks_positive_completion_lift_but_not_descriptive_completion():
    target = MetricTarget("lift", "Positive lift", require_baseline_headroom=True)
    evidence = complete(target, samples=9, baseline_successes=9, baseline_samples=9)
    result = assess([task()], target, evidence)
    assert result.status == "insufficient_sample"
    assert "ceiling" in " ".join(result.reasons)
    assert assess([task()], replace(target, require_baseline_headroom=False), evidence).status == "ready"
    assert assess([task()], target, replace(evidence, baseline_successes=8)).status == "ready"


def test_missing_oracle_cannot_be_inferred_from_g0_g3_group_names():
    target = next(item for item in default_targets() if item.metric == "knowledge_attribution")
    evidence = complete(target)
    evidence = replace(evidence, protocol_samples={"baseline": 1, "dynamic_explicit": 1})
    result = assess([task()], target, evidence)
    assert result.status == "not_measured"
    assert result.missing_protocols == ("static", "dynamic_natural", "oracle")


def test_explicit_activation_is_a_legal_ready_protocol_without_natural_query():
    target = MetricTarget(
        "explicit_path",
        "The user explicitly requests domain knowledge",
        required_protocols=("dynamic_explicit",),
        required_fields=("exact_delivered_knowledge",),
    )
    result = assess([task()], target, complete(target))
    assert result.status == "ready"
    assert "natural" not in result.missing_protocols


def test_direct_body_reads_do_not_supply_ranked_query_or_semantic_diagnosis():
    targets = {target.metric: target for target in default_targets()}
    raw = MetricEvidence(
        observed_samples=1,
        valid_samples=1,
        observed_task_ids=("task",),
        field_samples={"exact_body_read": 1, "both_sides_source_access": 1},
    )
    retrieval = assess([task()], targets["retrieval_quality"], raw)
    assert retrieval.status == "not_measured"
    assert "foreground_query" in retrieval.missing_field_samples
    root_cause = assess([task()], targets["root_cause_speedup"], raw)
    assert root_cause.status == "not_measured"
    assert "diagnosis_judgement" in root_cause.missing_field_samples
    assert "root_cause_oracle" in root_cause.missing_field_samples


def test_all_knowledge_gradients_need_observed_tasks_and_no_hit_is_distinct():
    target = MetricTarget(
        "transfer",
        "All knowledge conditions",
        required_gradients=("near_exact", "analogical", "compositional", "stale"),
    )
    tasks = [task(), task("analogy", family="b", gradient="analogical")]
    result = assess(tasks, target, complete(target))
    assert result.status == "insufficient_sample"
    assert result.missing_gradients == ("analogical", "compositional", "stale")
    tasks += [task("empty", family="c", gradient="no_hit")]
    result = assess(tasks, target, complete(target, ids=("task", "analogy", "empty")))
    assert result.missing_gradients == ("compositional", "stale")


def test_npu_only_metric_is_inapplicable_to_real_cpu_tasks():
    target = next(item for item in default_targets() if item.metric == "npu_attempt_savings")
    result = assess([task(npu_count=0)], target)
    assert result.status == "not_applicable"
    assert "no applicable tier/NPU" in " ".join(result.reasons)


def test_selected_group_does_not_create_missing_contrast():
    target = MetricTarget("contrast", "Compare", required_groups=("G0", "G2"))
    assert assess([task()], target, groups=("G3",)).status == "not_applicable"


def test_unknown_is_visible_and_never_treated_as_valid_or_retried():
    target = MetricTarget("effect", "Complete controlled sample", require_complete_observation=True)
    evidence = MetricEvidence(observed_samples=2, valid_samples=1, unknown_samples=1)
    result = assess([], target, evidence)
    assert result.status == "insufficient_sample"
    assert result.unknown_samples == 1
    assert result.valid_samples == 1
    assert "invalid/unknown" in " ".join(result.reasons)


def test_manifest_candidate_does_not_establish_verified_benefit():
    target = MetricTarget("completion", "Completion", require_verified_tasks=True)
    result = assess([task(evidence="candidate")], target, complete(target))
    assert result.status == "insufficient_sample"


def test_default_report_json_is_pure_and_does_not_claim_runtime_or_effect_acceptance():
    tasks = [task()]
    before = copy.deepcopy(tasks[0].model_dump())
    report = assess_coverage(tasks, groups=("G0", "G1", "G2", "G3"), repetitions=3)
    result = json.loads(json.dumps(report.to_dict()))
    assert result["execution_readiness"] == "not_checked"
    assert result["metrics"]["l0_latency"]["status"] == "not_measured"
    assert "positive benefit" in result["interpretation"]
    assert tasks[0].model_dump() == before


@pytest.mark.parametrize(
    "evidence",
    [
        MetricEvidence(observed_samples=1, valid_samples=2),
        MetricEvidence(observed_samples=1, valid_samples=1, unknown_samples=1),
        MetricEvidence(valid_samples=True),
        MetricEvidence(observed_samples=1, valid_samples=1, field_samples={"key": 2}),
        MetricEvidence(observed_samples=1, valid_samples=1, observed_task_ids=("other",)),
        MetricEvidence(observed_samples=1, valid_samples=1, baseline_samples=1),
        MetricEvidence(observed_samples=1, valid_samples=1, baseline_samples=1, baseline_successes=2),
        MetricEvidence(
            observed_samples=1,
            valid_samples=1,
            observed_task_ids=("task",),
            task_repetitions={"task": (1, 1)},
        ),
    ],
)
def test_malformed_evidence_is_an_error(evidence):
    with pytest.raises(ValueError):
        assess([task()], MetricTarget("metric", "Question"), evidence)


def test_duplicate_task_identity_or_metric_evidence_does_not_silently_merge():
    with pytest.raises(ValueError, match="task identities"):
        assess([task(), task()], MetricTarget("metric", "Question"))
    with pytest.raises(ValueError, match="unconfigured metric"):
        assess_coverage(
            [task()],
            groups=["G0"],
            repetitions=1,
            targets=[],
            evidence={"unconfigured": MetricEvidence()},
        )
