import json

import pytest

from mindie_bench.cli import main
from mindie_bench.models import TrialReceipt
from mindie_bench.scorer.comparator import CONTROL_KEYS, comparison, markdown


def row(group, role="eval", *, repetition=1, status="completed_pass", cost=0.1, tokens=10, seconds=20):
    return TrialReceipt(
        trial_id=f"{group}-{role}-{repetition}",
        instance_id=role,
        task_role=role,
        cluster="transfer",
        repetition=repetition,
        tier="l3",
        ablation_group=group,
        knowledge_gradient="analogical",
        task_digest=f"{role}-task",
        experiment_digest="experiment",
        environment_fingerprint={key: None for key in CONTROL_KEYS},
        terminal_status=status,
        failure_reason=None if status == "completed_pass" else "explicit failure",
        primary_metrics=dict(
            resolved=(status == "completed_pass") if status.startswith("completed") else None,
            reward=float(status == "completed_pass") if status.startswith("completed") else None,
            total_wall_clock_sec=seconds,
            delivery_wall_clock_sec=seconds,
            localization_penalty_turns=40,
            estimated_cost_usd=cost,
            cost_complete=cost is not None,
            tokens=dict(
                input_uncached=tokens,
                input_cached=0,
                output=0,
                indexer_small_model_total=1,
                complete=True,
            ),
        ),
        auxiliary_metrics={"capture": {"status": "completed"}} if role == "seed" else {},
        evidence="synthetic",
    )


def test_required_failed_g3_seed_costs_are_included_without_changing_eval_scores():
    baseline = [
        row("G0", cost=0.2, tokens=10, seconds=30),
        row("G0", "seed", cost=99, tokens=999, seconds=999),
    ]
    candidate = [
        row("G3", cost=0.1, tokens=5, seconds=10),
        row("G3", "seed", status="completed_fail", cost=0.4, tokens=20, seconds=40),
    ]
    result = comparison(baseline, candidate)
    assert result["comparison_role"] == "eval"
    assert result["matched_valid_pairs"] == 1
    assert result["paired_completion_lift"] == 0
    left, right = result["baseline"], result["candidate"]
    assert left["attempts"] == right["attempts"] == 1
    assert left["pass_at_1"] == right["pass_at_1"] == 1
    assert not left["seed_setup_required"]
    assert left["seed_setup_attempts"] == 0
    assert left["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 10
    assert left["observed_delivery_sec_including_seed_lower_bound"] == 30
    assert right["seed_setup_required"] and right["seed_setup_complete"]
    assert right["seed_setup_attempts"] == 1
    assert right["seed_setup_statuses"]["completed_fail"] == 1
    assert right["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 25
    assert right["observed_tokens_including_seed_lower_bound"]["indexer_small_model_total"] == 2
    assert right["observed_delivery_sec_including_seed_lower_bound"] == 50
    assert right["cost_per_resolved_task"] == pytest.approx(0.1)
    assert right["cost_per_resolved_task_including_seed"] == pytest.approx(0.5)
    assert "Resources including required G3 seed setup" in markdown(result)


@pytest.mark.parametrize("absent", ["all", "repetition", "cluster"])
def test_missing_required_seed_receipt_withholds_total_resource_completeness(absent):
    seeds = [] if absent == "all" else [row("G3", "seed")]
    if absent == "repetition":
        seeds[0].repetition = 2
    elif absent == "cluster":
        seeds[0].cluster = "other"
    result = comparison([row("G0")], [row("G3"), *seeds])["candidate"]
    assert result["attempts"] == result["valid_attempts"] == 1
    assert result["seed_setup_required"]
    assert result["seed_setup_attempts"] == 0
    assert result["seed_setup_lineages_required"] == 1
    assert result["seed_setup_lineages_observed"] == 0
    assert result["seed_setup_missing_lineages"][0]["cluster"] == "transfer"
    assert not result["seed_setup_receipts_complete"]
    assert not result["seed_setup_complete"]
    assert not result["cost_including_seed_complete"]
    assert not result["token_metering_including_seed_complete"]
    assert not result["delivery_timing_including_seed_complete"]
    assert result["seed_setup_cost_usd_lower_bound"] is None
    assert result["cost_per_resolved_task_including_seed"] is None
    assert result["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 10


def test_unknown_seed_stays_in_observed_resources_without_becoming_a_resolved_eval():
    result = comparison(
        [row("G0")],
        [row("G3"), row("G3", "seed", status="unknown", tokens=90, seconds=80)],
    )
    candidate = result["candidate"]
    assert result["matched_valid_pairs"] == 1
    assert candidate["statuses"]["unknown"] == 0
    assert candidate["seed_setup_statuses"]["unknown"] == 1
    assert candidate["seed_setup_receipts_complete"]
    assert not candidate["seed_setup_complete"]
    assert candidate["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 100
    assert candidate["observed_delivery_sec_including_seed_lower_bound"] == 100
    assert not candidate["token_metering_including_seed_complete"]
    assert candidate["cost_per_resolved_task_including_seed"] is None


def test_unpriced_routes_keep_comparison_dollars_null_including_seed_setup():
    result = comparison(
        [row("G0", cost=None)],
        [row("G3", cost=None), row("G3", "seed", cost=None)],
    )
    for summary in [result["baseline"], result["candidate"]]:
        assert summary["observed_cost_usd_lower_bound"] is None
        assert summary["seed_setup_cost_usd_lower_bound"] is None
        assert summary["observed_cost_usd_including_seed_lower_bound"] is None
        assert summary["cost_per_resolved_task"] is None
        assert summary["cost_per_resolved_task_including_seed"] is None
        assert not summary["cost_including_seed_complete"]
        assert summary["token_metering_including_seed_complete"]


def test_seed_role_keeps_existing_matching_and_does_not_double_count_setup():
    baseline, candidate = [row("G0"), row("G0", "seed")], [row("G3"), row("G3", "seed")]
    result = comparison(baseline, candidate, role="seed")
    assert result["comparison_role"] == "seed"
    assert result["candidate"]["attempts"] == 1
    assert not result["candidate"]["seed_setup_required"]
    assert result["candidate"]["seed_setup_attempts"] == 0
    assert result["candidate"]["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 10
    candidate[1].environment_fingerprint["base_model"] = "changed"
    with pytest.raises(ValueError, match="base_model"):
        comparison(baseline, candidate, role="seed")


def test_missing_one_repetition_is_incomplete():
    baseline = [row("G0"), row("G0", repetition=2)]
    candidate = [row("G3"), row("G3", repetition=2), row("G3", "seed", tokens=50)]
    result = comparison(baseline, candidate)["candidate"]
    assert result["seed_setup_lineages_required"] == 2
    assert result["seed_setup_lineages_observed"] == 1
    assert result["seed_setup_missing_lineages"][0]["repetition"] == 2
    assert result["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 70
    assert not result["seed_setup_complete"]


def test_one_seed_shared_by_two_evaluations_is_charged_once():
    left, right = row("G0"), row("G3")
    extra_left, extra_right = row("G0"), row("G3")
    for extra in [extra_left, extra_right]:
        extra.instance_id = "second-eval"
        extra.trial_id += "-second"
    result = comparison([left, extra_left], [right, extra_right, row("G3", "seed", tokens=50)])[
        "candidate"
    ]
    assert result["attempts"] == 2
    assert result["seed_setup_attempts"] == 1
    assert result["seed_setup_lineages_required"] == 1
    assert result["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 70
    assert result["cost_per_resolved_task_including_seed"] == pytest.approx(0.15)


def test_duplicate_seed_receipts_cannot_inflate_resource_totals():
    seed = row("G3", "seed")
    with pytest.raises(ValueError, match="duplicate seed setup"):
        comparison([row("G0")], [row("G3"), seed, seed])


def test_compare_cli_preserves_full_receipts_until_resource_accounting(tmp_path, capsys):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for root, group in [(baseline, "G0"), (candidate, "G3")]:
        for receipt in [row(group, cost=None), row(group, "seed", cost=None, tokens=70)]:
            path = root / receipt.trial_id / "trial_receipt.json"
            path.parent.mkdir(parents=True)
            path.write_text(receipt.model_dump_json())
    output = tmp_path / "comparison"
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
            ]
        )
        == 0
    ), capsys.readouterr()
    result = json.loads((output / "comparison.json").read_text())
    assert result["baseline"]["seed_setup_attempts"] == 0
    assert result["candidate"]["seed_setup_attempts"] == 1
    assert result["candidate"]["observed_tokens_including_seed_lower_bound"]["input_uncached"] == 80
    assert result["candidate"]["cost_per_resolved_task_including_seed"] is None
