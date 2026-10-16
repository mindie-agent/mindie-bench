import pytest

from mindie_bench.models import Tokens, ToolCall, Trajectory, TrialReceipt
from mindie_bench.scorer.comparator import CONTROL_KEYS, comparison
from mindie_bench.scorer.metrics import (
    first_localization,
    npu_test_runs,
    retrieval_metrics,
    routing_metrics,
    summarize,
    trajectory_metrics,
)


def receipt(status, *, identity="task", repetition=1, group="G0", **metrics):
    passed = status == "completed_pass"
    valid = status in {"completed_pass", "completed_fail"}
    return TrialReceipt(
        trial_id=f"{identity}-{group}-{repetition}",
        instance_id=identity,
        repetition=repetition,
        tier="l2",
        ablation_group=group,
        knowledge_gradient="analogical",
        task_digest="same-task",
        experiment_digest="experiment",
        environment_fingerprint={key: None for key in CONTROL_KEYS},
        terminal_status=status,
        failure_reason=None if passed else "explicit failure",
        primary_metrics=dict(
            resolved=passed if valid else None,
            reward=float(passed) if valid else None,
            total_wall_clock_sec=10,
            localization_penalty_turns=25,
            estimated_cost_usd=0.1,
            cost_complete=True,
            **metrics,
        ),
        evidence="synthetic",
    )


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "vllm serve model",
        "python -m unittest",
        "python3 tests/test_layer.py",
        "msprof --application=x",
        "torchrun x.py",
    ],
)
def test_exact_npu_test_definition(command):
    trajectory = Trajectory(
        calls=[
            ToolCall(id="1", name="mcp__remote__remote_bash", arguments={"command": command}, turn=1),
            ToolCall(id="2", name="Bash", arguments={"command": "ssh bench-slot " + command}, turn=2),
            ToolCall(id="3", name="Bash", arguments={"command": command}, turn=3),
        ]
    )
    assert npu_test_runs(trajectory) == 2


def test_filename_mentions_do_not_claim_localization():
    trajectory = Trajectory(
        calls=[
            ToolCall(id="1", name="Bash", arguments={"command": "echo target.py"}, turn=1),
            ToolCall(id="2", name="Read", arguments={"file_path": "prefix-target.py"}, turn=2),
            ToolCall(id="3", name="Grep", arguments={"matched_files": ["/workspace/target.py"]}, turn=4),
        ]
    )
    assert first_localization(trajectory, ["target.py"]) == 4
    assert first_localization(trajectory, ["other.py"]) is None


def test_nonfinite_and_missing_timing_is_not_zero(case):
    trajectory = Trajectory(
        calls=[
            ToolCall(id="1", name="remote_bash", arguments={"command": "pytest"}, turn=1, ended_sec=5)
        ]
    )
    primary, auxiliary = trajectory_metrics(
        trajectory, case["spec"], case["config"].harness, 7, tier="l2"
    )
    assert primary["npu_execution_wait_sec"] is None
    assert primary["llm_and_exploration_sec"] is None
    assert primary["npu_test_runs"] is None
    assert auxiliary["tool_timing_complete"] is False


def test_overlap_never_produces_negative_llm_time(case):
    trajectory = Trajectory(
        calls=[
            ToolCall(id=str(i), name="remote_bash", turn=1, started_sec=0, ended_sec=10)
            for i in range(2)
        ]
    )
    primary, _ = trajectory_metrics(trajectory, case["spec"], case["config"].harness, 12, tier="l3")
    assert primary["npu_execution_wait_sec"] == 20
    assert primary["llm_and_exploration_sec"] is None


def test_infra_and_unknown_visible_but_excluded_from_valid_pass_denominator():
    rows = [
        receipt(status, identity=str(i))
        for i, status in enumerate(["completed_pass", "completed_fail", "infra_failed", "unknown"])
    ]
    result = summarize(rows)
    assert result["attempts"] == 4
    assert result["valid_attempts"] == 2
    assert result["pass_at_1"] == 0.5
    assert result["all_attempt_success_rate"] == 0.25
    assert result["coverage"] == 0.5
    assert result["pass_pow_k"] is None
    assert result["cost_per_resolved_task"] == pytest.approx(0.4)


def test_reliability_is_task_stratified_and_needs_repeats():
    rows = [receipt("completed_pass", identity="easy", repetition=i) for i in [1, 2, 3]]
    rows += [receipt("completed_fail", identity="hard", repetition=i) for i in [1, 2, 3]]
    result = summarize(rows)
    assert result["pass_at_1"] == 0.5
    assert result["pass_pow_k"] == 0.5
    assert result["reliability_eligible_tasks"] == 2


def test_unknown_metering_withholds_resolved_cost():
    row = receipt("unknown")
    row.primary_metrics["cost_complete"] = False
    result = summarize([row, receipt("completed_pass", identity="solved")])
    assert result["cost_per_resolved_task"] is None
    assert result["observed_cost_usd_lower_bound"] == pytest.approx(0.2)


def test_cached_tokens_are_not_double_charged(case):
    trajectory = Trajectory(tokens=Tokens(input_uncached=10, input_cached=20, output=3, complete=True))
    harness = case["config"].harness.model_copy(
        update={"input_usd_per_million": 2, "cached_usd_per_million": 0.5, "output_usd_per_million": 8}
    )
    primary, _ = trajectory_metrics(trajectory, case["spec"], harness, 1, tier="l2")
    assert primary["estimated_cost_usd"] == pytest.approx(54 / 1_000_000)


def test_cache_writes_need_an_explicit_price(case):
    trajectory = Trajectory(tokens=Tokens(input_uncached=20, cache_creation=10, output=5, complete=True))
    harness = case["config"].harness.model_copy(
        update={"input_usd_per_million": 2, "output_usd_per_million": 8}
    )
    primary, _ = trajectory_metrics(trajectory, case["spec"], harness, 1, tier="l2")
    assert primary["estimated_cost_usd"] is None
    harness.cache_creation_usd_per_million = 2.5
    primary, _ = trajectory_metrics(trajectory, case["spec"], harness, 1, tier="l2")
    assert primary["estimated_cost_usd"] == pytest.approx(85 / 1_000_000)


def test_retrieval_and_negative_routing_guardrails():
    result = retrieval_metrics(["wrong", "gold"], {"gold"}, k=3)
    assert result["recall_at_k"] == 1
    assert result["reciprocal_rank"] == 0.5
    assert 0 < result["ndcg_at_k"] < 1
    result = routing_metrics([(True, True), (False, False), (False, True)])
    assert result["precision"] == 0.5
    assert result["recall"] == 1
    assert result["zero_false_trigger_rate"] == 0.5


def test_comparison_requires_exact_controls_and_same_samples():
    left, right = receipt("completed_pass"), receipt("completed_pass", group="G2")
    assert comparison([left], [right])["paired_completion_lift"] == 0
    right.environment_fingerprint["harness_version"] = "different-patch-release"
    with pytest.raises(ValueError, match="harness_version"):
        comparison([left], [right])


def test_no_survivorship_time_speedup_or_mixed_tier():
    left, right = receipt("completed_fail"), receipt("completed_pass", group="G2")
    result = comparison([left], [right])
    assert result["wall_clock_speedup_on_mutual_success"] is None
    right.tier = "l3"
    with pytest.raises(ValueError):
        comparison([left], [right])
