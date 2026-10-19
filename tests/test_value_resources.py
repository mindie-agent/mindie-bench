import copy
import json
from pathlib import Path

import pytest

from mindie_bench.io import read_json
from mindie_bench.models import TrialReceipt
from mindie_bench.value_resources import project_component_case, resource_totals


def receipt():
    return TrialReceipt.model_validate(
        read_json(Path(__file__).parent / "fixtures/value-dp2-g0-receipt.json")
    )


def test_historical_unknown_keeps_costs_separate_and_in_all_execution_totals():
    selected = receipt()
    original = selected.model_dump(mode="json")
    original.update(
        trial_id="original-provider-ssl-unknown",
        terminal_status="unknown",
        failure_reason="provider transport did not settle",
    )
    original["primary_metrics"].update(
        reward=None,
        resolved=None,
        delivery_wall_clock_sec=140.93433387600817,
        total_wall_clock_sec=120.0,
        estimated_cost_usd=1.25,
        cost_complete=False,
        tokens={
            "input_uncached": 99372,
            "input_cached": 1270144,
            "cache_creation": 0,
            "output": 18472,
            "indexer_small_model_total": 0,
            "complete": False,
        },
    )
    history = TrialReceipt.model_validate(original)
    chosen = resource_totals([selected])
    excluded = resource_totals([history])
    combined = resource_totals([selected, history])
    assert chosen["attempts"] == excluded["attempts"] == 1
    assert combined["statuses"] == {"completed_pass": 1, "unknown": 1}
    assert combined["observed_tokens_lower_bound"]["input_uncached"] == 116973 + 99372
    assert combined["observed_tokens_lower_bound"]["input_cached"] == 2475008 + 1270144
    assert combined["observed_delivery_wall_clock_sec_lower_bound"] == pytest.approx(
        573.1825801659725 + 140.93433387600817
    )
    assert combined["delivery_timing_complete"]
    assert not combined["token_metering_complete"]
    assert chosen["observed_cost_usd_lower_bound"] is None
    assert combined["observed_cost_usd_lower_bound"] == 1.25
    assert combined["priced_attempts"] == 1
    assert not combined["cost_complete"]


def test_missing_measurements_stay_missing_and_seed_capture_is_not_counted_twice():
    seed = receipt().model_dump(mode="json")
    seed.update(trial_id="captured-seed", task_role="seed", ablation_group="G3")
    seed["primary_metrics"].update(
        delivery_wall_clock_sec=25.0,
        total_wall_clock_sec=20.0,
        estimated_cost_usd=0.0,
        cost_complete=True,
    )
    seed["primary_metrics"]["tokens"]["indexer_small_model_total"] = 321
    seed["auxiliary_metrics"].update(capture_wall_clock_sec=4.0, capture={"status": "completed"})
    evaluation = copy.deepcopy(seed)
    evaluation.update(trial_id="later-evaluation", task_role="eval")
    evaluation["primary_metrics"].update(
        delivery_wall_clock_sec=None,
        total_wall_clock_sec=None,
        tokens={},
        estimated_cost_usd=None,
        cost_complete=False,
    )
    seed_row, eval_row = [TrialReceipt.model_validate(row) for row in (seed, evaluation)]
    known = resource_totals([seed_row])
    assert known["observed_cost_usd_lower_bound"] == 0.0
    assert known["cost_complete"]
    missing = resource_totals([eval_row])
    assert missing["observed_tokens_lower_bound"]["input_uncached"] is None
    assert missing["observed_delivery_wall_clock_sec_lower_bound"] is None
    assert not missing["token_metering_complete"]
    combined = resource_totals([seed_row, eval_row])
    assert combined["task_roles"] == {"seed": 1, "eval": 1}
    assert combined["observed_delivery_wall_clock_sec_lower_bound"] == 25.0
    assert combined["observed_tokens_lower_bound"]["indexer_small_model_total"] == 321
    assert combined["token_observed_attempts"]["indexer_small_model_total"] == 1
    assert not combined["delivery_timing_complete"]
    empty = resource_totals([])
    assert empty["observed_cost_usd_lower_bound"] is None
    assert not empty["delivery_timing_complete"]


def test_actual_diagnostics_schema_projects_only_public_observations():
    case = {
        "id": "diagnostics-fault-delivery",
        "component": "diagnostics",
        "status": "pass",
        "scope": "real logging, linked consent, ingestion, outbox and reporter with local HTTP",
        "baseline": "Three real exception tracebacks retained as ordinary local raw logs.",
        "candidate": "One sanitized issue payload, remote-marker deduplication and recovery.",
        "repetitions": 1,
        "limitations": ["The HTTP receiver is a local fixture, not GitHub availability."],
        "raw": "/private/fixture/case.json",
        "measurements": [{
            "repetition": 1,
            "status": "pass",
            "raw": "/private/fixture/raw.json",
            "report": "/private/fixture/delivered-report.md",
            "baseline": {
                "failure_occurrences": 3,
                "raw_tracebacks": 3,
                "contains_synthetic_secret": True,
                "automatic_delivery": False,
                "traceback": "/private/fixture/project/module.py",
            },
            "candidate": {
                "failure_occurrences": 3,
                "queue_items": 1,
                "http_posts": 1,
                "delivered_reports": 1,
                "reconciled_without_duplicate": True,
                "private_fixture_removed": True,
                "useful_fault_fields_retained": True,
                "withdrawal_prevented_requests": True,
                "production_retry_wait_sec": 59.99844813346863,
                "failure_to_delivery_sec": 60.58435137500055,
                "unknown_numeric_field": 987654,
                "endpoint": "http://private-fixture",
                "extra": {"path": "/private/fixture"},
            },
        }],
        "source_revisions": {
            "mindie-diagnostics": {
                "version": "0.4.3",
                "revision": "1b5a0eeb79d498c39002fc1853271c17e0f26e66",
                "editable": False,
                "path": "/private/fixture/site-packages",
            },
        },
    }
    projected = project_component_case(case)
    encoded = json.dumps(projected, allow_nan=False)
    assert "/private" not in encoded
    assert "private-fixture" not in encoded
    assert "unknown_numeric_field" not in encoded
    observation = projected["measurements"][0]
    assert observation["baseline"]["raw_tracebacks"] == 3
    assert observation["candidate"]["http_posts"] == 1
    assert observation["candidate"]["reconciled_without_duplicate"] is True
    assert observation["candidate"]["failure_to_delivery_sec"] == 60.58435137500055
    assert projected["source_revisions"]["mindie-diagnostics"] == {
        "version": "0.4.3",
        "revision": "1b5a0eeb79d498c39002fc1853271c17e0f26e66",
        "editable": False,
    }
    assert case["measurements"][0]["report"] == "/private/fixture/delivered-report.md"
