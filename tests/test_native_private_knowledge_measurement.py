import pytest

from mindie_bench.models import TerminalStatus, Trajectory, TrialReceipt
from mindie_bench.runner.drivers.codex import CodexDriver
from mindie_bench.runner.trial_finalize import score_trial_trajectory
from mindie_bench.scorer.components import summarize_five_components
from mindie_bench.scorer.metrics import summarize


@pytest.mark.parametrize("native_product,named_calls", [(True, False), (True, True), (False, True)])
def test_private_cli_coverage_survives_scoring_and_aggregation(
    case, tmp_path, native_product, named_calls
):
    driver = CodexDriver()
    item = {
        "id": "item_28",
        "type": "command_execution",
        "command": (
            "python -I /plugin/scripts/knowledge_cli.py knowledge_query <<'JSON'\n"
            '{"query":"missing invariant","domain":"vllm-ascend"}\nJSON'
        ),
        "status": "in_progress",
    }
    driver.consume({"type": "item.started", "item": item}, 1, 1)
    driver.consume(
        {
            "type": "item.completed",
            "item": dict(item, status="completed", exit_code=0),
        },
        2,
        1,
    )
    if named_calls:
        for index, operation in enumerate(("knowledge_query", "knowledge_explain")):
            named = dict(
                id=f"named_{index}",
                type="mcp_tool_call",
                server="knowledge",
                tool=operation,
                arguments={"query": "invariant"} if index == 0 else {"ref": "rule@abc123"},
            )
            driver.consume({"type": "item.started", "item": named}, 2 + index / 4, 1)
            driver.consume(
                {"type": "item.completed", "item": dict(named, status="completed")},
                2 + (index + 1) / 4,
                1,
            )
    driver.consume(
        dict(type="turn.completed", usage=dict(
            input_tokens=30, cached_input_tokens=20, output_tokens=3
        )),
        3,
        1,
    )
    config = case["config"].model_copy(
        update={
            "backend": "native",
            "harness": case["config"].harness.model_copy(update={
                "name": "codex",
                "input_usd_per_million": 2,
                "cached_usd_per_million": 0.5,
                "output_usd_per_million": 8,
            }),
        }
    )
    errors = []
    fingerprint = {"native_product": {"harness": "codex"}} if native_product else {}
    result = score_trial_trajectory(
        driver=driver,
        gateway=None,
        result=None,
        trajectory=Trajectory(),
        config=config,
        environment=None,
        task=case["spec"],
        trial_dir=tmp_path,
        task_root=case["task"],
        wall_clock=3,
        tier="l2",
        fingerprint=fingerprint,
        errors=errors,
        primary={},
        auxiliary={},
        primary_verdict=TerminalStatus.PASS,
        status=TerminalStatus.PASS,
        reason=None,
    )
    assert errors == []
    primary, auxiliary = result["primary"], result["auxiliary"]
    assert primary["estimated_cost_usd"] == pytest.approx(54 / 1_000_000)
    assert primary["cost_complete"] is True
    assert primary["tokens"]["input_uncached"] == 10
    timing = auxiliary["tool_timing"]
    assert timing["native_knowledge"]["observed_union_sec"] == (0.5 if named_calls else 0)
    if native_product:
        assert primary["foreground_knowledge_queries_count"] is None
        assert primary["foreground_knowledge_explains_count"] is None
        assert primary["knowledge_query_sec"] is None
        assert primary["llm_and_exploration_sec"] is None
        assert auxiliary["knowledge_queries_count"] is None
        assert auxiliary["knowledge_explains_count"] is None
        assert auxiliary["knowledge_measurement"]["observed_named_query_count_lower_bound"] == int(
            named_calls
        )
        assert timing["knowledge"]["calls"] is None
        assert timing["knowledge"]["observed_union_sec"] is None
        assert timing["combined"]["observed_union_sec"] is None
        assert timing["knowledge"]["complete"] is False
        assert auxiliary["tool_timing_complete"] is False
    else:
        assert primary["foreground_knowledge_queries_count"] == 1
        assert primary["foreground_knowledge_explains_count"] == 1
        assert primary["knowledge_query_sec"] == 0.5
        assert primary["llm_and_exploration_sec"] == 2.5
    receipt = TrialReceipt(
        trial_id="private-cli",
        instance_id="task",
        repetition=1,
        tier="l2",
        ablation_group="G2",
        knowledge_gradient="analogical",
        task_digest="task",
        experiment_digest="experiment",
        environment_fingerprint=fingerprint,
        terminal_status="completed_pass",
        primary_metrics=dict(primary, resolved=True, reward=1.0),
        auxiliary_metrics=auxiliary,
        evidence="synthetic",
    )
    summary = summarize([receipt])
    observed = summary["knowledge_observations"]
    components = summarize_five_components([receipt])["knowledge"]
    assert summary["cost_per_resolved_task"] == pytest.approx(54 / 1_000_000)
    assert observed["observed_query_count_lower_bound"] == int(named_calls)
    assert observed["observed_explain_count_lower_bound"] == int(named_calls)
    assert observed["attempts_with_observed_queries"] == int(named_calls)
    if native_product:
        assert observed["state"] == "not_measured"
        assert observed["query_count"] is None
        assert observed["explain_count"] is None
        assert observed["query_count_coverage"]["missing_attempts"] == 1
        assert components["queries_total"] is None
        assert components["explains_total"] is None
        assert components["mean_knowledge_query_sec"] is None
    else:
        assert observed["query_count"] == components["queries_total"] == 1
        assert observed["explain_count"] == components["explains_total"] == 1
        unknown = receipt.model_copy(deep=True)
        unknown.primary_metrics.update(
            foreground_knowledge_queries_count=None,
            foreground_knowledge_explains_count=None,
        )
        mixed = summarize_five_components([receipt, unknown])["knowledge"]
        assert mixed["queries_total"] is None
        assert mixed["explains_total"] is None
        assert mixed["query_count_observed_attempts"] == 1
