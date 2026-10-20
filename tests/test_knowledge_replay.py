from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_bench.runner.knowledge_replay import (
    RoutingIndex,
    compare_loading_configs,
    run_growth_replay,
)

OPERATORS = Path("/Volumes/data/code/knowledge-operators")
TRANSCRIPT = Path("/Users/maoxx241/code/mindie-transcript-20261004/content")
METRIC_KEYS = {
    "total_loaded_tokens",
    "duplicate_parent_tokens",
    "truncated_procedural_units",
    "ready_to_apply_rate",
    "compaction_reread_tokens",
    "net_token_savings_c_vs_b",
}


def _body(tokens: int) -> str:
    """Produce an ASCII body whose production UTF-8/4 estimate is exactly tokens."""
    pattern = "procedure detail with a measured condition. "
    body = (pattern * ((tokens * 4) // len(pattern) + 1))[: tokens * 4 - 1]
    return body[:-1] + "x" if body[-1].isspace() else body


def _rewrite_body(path: Path, tokens: int) -> None:
    frontmatter = path.read_text(encoding="utf-8").split("\n---\n", 1)[0]
    path.write_text(frontmatter + "\n---\n" + _body(tokens), encoding="utf-8")


def _write_corpus(root: Path) -> Path:
    rows = []
    units = [
        ("foundation", "_common", "L1", "axiom", 400, {"dtype": "bfloat16"}, [], ["dtype"]),
        ("measurement", "_common", "L1", "measurement", 350, {"dtype": "bfloat16"}, [], ["dtype"]),
        ("workflow", "norm/rmsnorm", "L3", "workflow", 900,
         {"dtype": "bfloat16", "soc": "Ascend910B4", "dp": ">1"}, ["foundation", "measurement"],
         ["rmsnorm", "workflow", "triage"]),
        ("mutator", "norm/rmsnorm", "L4", "mutator", 1500,
         {"dtype": "bfloat16", "soc": "Ascend910B4"}, ["foundation", "measurement"],
         ["rmsnorm", "mutator", "fusion"]),
    ]
    for identity, domain, tier, category, tokens, conditions, parents, keywords in units:
        path = root / domain / f"unit-{identity}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        strength = "hardware_spec" if tier == "L1" else (
            "reproduced_empirical" if tier == "L4" else "single_observation")
        path.write_text(
            "---\n"
            f"schema: mindie-graded-unit/1\nid: unit-{identity}\nrevision: 2\n"
            f"tier: {tier}\ncategory: {category}\ndomain_path: {domain}\n"
            f"evidence_strength: {strength}\n"
            f"title: '{identity} replay unit'\nconditions: {json.dumps(conditions)}\n"
            f"inherits_from: {json.dumps([f'unit-{parent}' for parent in parents])}\n"
            "provenance:\n  source_type: replay_fixture\n"
            "  source_locator: tests/test_knowledge_replay.py\n"
            "  evidence_refs:\n  - tests/test_knowledge_replay.py\n"
            f"triggers: {json.dumps({'error_signatures': [], 'keywords': keywords})}\n---\n"
            + _body(tokens),
            encoding="utf-8",
        )
        rows.append({
            "id": f"unit-{identity}", "revision": 2, "tier": tier, "category": category,
            "domain_path": domain, "title": f"{identity} replay unit",
            "evidence_strength": strength, "conditions": conditions,
            "inherits_from": [f"unit-{parent}" for parent in parents],
            "triggers": {"error_signatures": [], "keywords": keywords},
            "token_estimate": tokens, "path": path.relative_to(root).as_posix(),
        })
    manifest = root / "index" / "routing-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema_version": "mindie-routing-manifest/1",
                                    "units": rows}), encoding="utf-8")
    return root


@pytest.fixture()
def corpus(tmp_path):
    return _write_corpus(tmp_path / "corpus")


def _tasks():
    conditions = {"dtype": "bfloat16", "soc": "Ascend910B4", "dp": 2}
    return [
        {
            "task_id": "operator-perf", "category": "operator_perf",
            "query": "rmsnorm mutator fusion", "conditions": conditions,
            "required_refs": ["unit-mutator"],
            "turns": [{"query": "rmsnorm mutator fusion"}, {"compacted": True}],
        },
        {
            "task_id": "diagnose", "category": "diagnose",
            "query": "rmsnorm workflow triage", "conditions": conditions,
            "required_refs": ["unit-workflow"],
        },
        {
            "task_id": "zero-day", "category": "0day",
            "query": "foundation dtype", "conditions": conditions,
            "required_refs": ["unit-foundation"],
        },
        {
            "task_id": "model-perf", "category": "model_perf",
            "query": "rmsnorm workflow triage", "conditions": conditions,
            "required_refs": ["unit-workflow"],
        },
    ]


def test_three_configs_measure_legacy_dump_against_progressive_closure(corpus):
    result = compare_loading_configs(_tasks(), corpus_root=corpus)
    a, b, c = (result["configs"][name] for name in ("A", "B", "C"))

    assert a["name"] == "no_domain_knowledge"
    assert a["total_loaded_tokens"] == 0
    assert a["ready_to_apply_rate"] == 0.0
    assert b["name"] == "legacy_query_dump"
    assert b["duplicate_parent_tokens"] > 0
    assert b["truncated_procedural_units"] > 0
    assert c["name"] == "progressive_navigation_and_closure"
    assert c["duplicate_parent_tokens"] == 0
    assert c["truncated_procedural_units"] == 0
    assert c["total_loaded_tokens"] < b["total_loaded_tokens"]
    assert c["ready_to_apply_rate"] > b["ready_to_apply_rate"]
    assert c["compaction_reread_tokens"] < b["compaction_reread_tokens"]
    assert result["net_token_savings_c_vs_b"] == (
        b["total_loaded_tokens"] - c["total_loaded_tokens"])
    assert c["net_token_savings_c_vs_b"] == result["net_token_savings_c_vs_b"]
    assert b["net_token_savings_c_vs_b"] == 0


def test_per_category_metrics_keep_every_required_field(corpus):
    result = compare_loading_configs(_tasks(), corpus_root=corpus)
    assert set(result["categories"]) == {"0day", "diagnose", "operator_perf", "model_perf"}
    for category in result["categories"]:
        for name in ("A", "B", "C"):
            metrics = result["categories"][category][name]
            assert METRIC_KEYS <= set(metrics)
            assert result["configs"][name]["categories"][category] == metrics
    operator = result["categories"]["operator_perf"]
    assert operator["C"]["net_token_savings_c_vs_b"] > 0
    assert operator["B"]["truncated_procedural_units"] > 0
    assert operator["C"]["truncated_procedural_units"] == 0
    assert result["configs"]["A"]["categories"]["0day"]["net_token_savings_c_vs_b"] == 0


def test_navigation_cards_never_inline_a_body_and_closure_deduplicates_known_parents(corpus):
    index = RoutingIndex(corpus)
    cards = index.query("rmsnorm mutator fusion", {"dtype": "bfloat16", "soc": "Ascend910B4"})
    assert cards and all("body" not in card for card in cards)
    assert cards[0]["unit_id"] == "unit-mutator"
    assert cards[0]["ready_to_apply"] is True

    full = index.explain_closure(["unit-mutator"],
                                 conditions={"dtype": "bfloat16", "soc": "Ascend910B4"},
                                 token_budget=4096)
    assert [unit["unit_id"] for unit in full["units"]] == [
        "unit-foundation", "unit-measurement", "unit-mutator"]
    assert not full["unresolved"]
    deduplicated = index.explain_closure(
        ["unit-mutator"], known_refs=["unit-foundation", "unit-measurement"],
        conditions={"dtype": "bfloat16", "soc": "Ascend910B4"}, token_budget=4096,
    )
    assert [unit["unit_id"] for unit in deduplicated["units"]] == ["unit-mutator"]

    excluded = index.explain_closure(
        ["unit-foundation", "unit-measurement", "unit-mutator"], token_budget=1200)
    assert [unit["unit_id"] for unit in excluded["units"]] == [
        "unit-foundation", "unit-measurement"]
    excluded_mutator = next(
        item for item in excluded["unresolved"] if item["reason"] == "budget_excluded")
    assert excluded_mutator["ref"] == "unit-mutator@2"
    assert excluded_mutator["required_tokens"] > 0
    assert excluded["usage"]["complete"] is False


def test_unknown_conditions_are_discoverable_but_never_ready_to_apply(corpus):
    index = RoutingIndex(corpus)
    cards = index.query("rmsnorm mutator fusion", {"soc": "Ascend910B4"})
    card = next(card for card in cards if card["unit_id"] == "unit-mutator")
    assert card["match_status"] == "condition_unknown"
    assert card["ready_to_apply"] is False
    assert card["missing_conditions"] == ["dtype"]

    tasks = [{**_tasks()[0], "conditions": {"soc": "Ascend910B4"}}]
    result = compare_loading_configs(tasks, corpus_root=corpus)
    for name in ("B", "C"):
        assert result["configs"][name]["ready_to_apply_rate"] == 0.0


def _growth_events():
    return [
        {"event_id": "e1", "action": "create", "category": "new_mechanism",
         "body_tokens": 500, "read_tokens": 100, "rare_conditions": ["strided-gate"]},
        {"event_id": "e2", "action": "merge_into_existing", "category": "repeat",
         "body_delta": 80, "read_tokens": 40},
        {"event_id": "e3", "action": "noop", "category": "repeat", "read_tokens": 20},
        {"event_id": "e4", "action": "branch", "category": "version_contradiction",
         "body_tokens": 300, "read_tokens": 50, "rare_conditions": ["fp16-fallback"]},
        {"event_id": "e5", "action": "supersede", "category": "version_contradiction",
         "body_delta": -50, "read_tokens": 30},
        {"event_id": "e6", "action": "consolidate", "category": "rare_condition",
         "merged_units": 1, "reclaimed_body_tokens": 200, "read_tokens": 10},
    ]


def test_growth_replay_reports_a_monotonic_trajectory_and_final_ratios():
    result = run_growth_replay(_growth_events())
    trajectory = result["trajectory"]
    assert len(trajectory) == 6
    assert [row["active_units"] for row in trajectory] == [1, 1, 1, 2, 2, 1]
    assert [row["delta_u_active"] for row in trajectory] == [1, 0, 0, 1, 0, -1]
    assert result["milestones"] == 6
    assert result["creations"] == 2
    assert result["revisions"] == 4
    assert result["active_units"] == 1
    assert result["body_tokens"] == 630
    assert result["g_units"] == pytest.approx(1 / 6)
    assert result["g_body"] == pytest.approx(105.0)
    assert result["g_read"] == pytest.approx(250 / 6)
    assert result["revision_to_creation_ratio"] == pytest.approx(2.0)
    assert result["rare_condition_recall_rate"] == 1.0
    assert result["final"]["g_units"] == result["g_units"]
    assert result["rare_conditions"] == {
        "known": ["strided-gate", "fp16-fallback"],
        "retained": ["fp16-fallback", "strided-gate"],
    }
    assert set(result["categories"]) == {
        "new_mechanism", "repeated_success", "version_contradiction", "rare_condition"}
    assert result["categories"]["repeated_success"]["delta_u_active"] == 0


def test_rare_condition_drop_is_visible_in_recall():
    result = run_growth_replay([
        {"action": "create", "category": "new_mechanism", "body_tokens": 100,
         "rare_conditions": ["keep-me", "drop-me"]},
        {"action": "consolidate", "category": "rare_condition", "merged_units": 0,
         "dropped_rare_conditions": ["drop-me"]},
    ])
    assert result["rare_condition_recall_rate"] == 0.5
    assert result["rare_conditions"]["retained"] == ["keep-me"]


def test_repeated_success_updates_do_not_grow_active_units_and_missing_milestones_are_uncomputable():
    result = run_growth_replay([
        {"action": "create", "body_tokens": 100, "verified": False},
        {"action": "merge_into_existing", "category": "repeat", "body_delta": 10,
         "verified": False},
        {"action": "noop", "category": "repeat", "verified": False},
    ])
    assert result["active_units"] == 1
    assert result["milestones"] == 0
    assert result["g_units"] is None
    assert result["g_body"] is None
    assert result["g_read"] is None
    assert result["revision_to_creation_ratio"] == 1.0


@pytest.mark.skipif(not (OPERATORS / "index" / "routing-manifest.json").is_file(),
                    reason="operators knowledge corpus is not mounted")
def test_real_operator_corpus_legacy_dump_loses_to_progressive_closure():
    conditions = {
        "soc": "Ascend910B4", "dtype": "bfloat16", "activation": "sigmoid",
        "target_file": "vllm_ascend/ops/layernorm.py",
    }
    tasks = [
        {"task_id": "rmsnorm", "category": "operator_perf",
         "query": "rmsnorm residual fastpath ablation", "conditions": conditions,
         "required_refs": ["unit-residual-free-rmsnorm-fastpath"],
         "turns": [{"query": "rmsnorm residual fastpath ablation"}, {"compacted": True}]},
        {"task_id": "triton", "category": "diagnose",
         "query": "triton dynamic length recompilation",
         "conditions": {**conditions, "kernel_language": "triton"},
         "required_refs": ["unit-triton-dynamic-scalar-specialization-sop"]},
    ]
    result = compare_loading_configs(tasks, corpus_root=OPERATORS)
    b, c = result["configs"]["B"], result["configs"]["C"]
    assert b["duplicate_parent_tokens"] > 0
    assert b["truncated_procedural_units"] > 0
    assert c["truncated_procedural_units"] == 0
    assert c["total_loaded_tokens"] < b["total_loaded_tokens"]
    assert c["ready_to_apply_rate"] >= b["ready_to_apply_rate"]
    assert c["compaction_reread_tokens"] < b["compaction_reread_tokens"]
    assert result["net_token_savings_c_vs_b"] > 0


@pytest.mark.skipif(not (TRANSCRIPT / "index" / "routing-manifest.json").is_file(),
                    reason="vllm-ascend transcript corpus is not mounted")
def test_real_transcript_corpus_replays_0day_and_model_performance_tasks():
    base = {"model_family": "qwen3_5", "npu_model": "Ascend910B4", "repository": "vllm-ascend"}
    tasks = [
        {"task_id": "hybrid-bringup", "category": "0day",
         "query": "hybrid model bringup gated deltanet", "conditions": base,
         "required_refs": ["unit-vllm-ascend-0day-hybrid-model-bringup-hermes-workflow"]},
        {"task_id": "idle-dp-state", "category": "model_perf",
         "query": "hybrid recurrent dp state idle", "conditions": {
             **base, "affected_state": ["conv_state", "ssm_state"], "dp": 2,
             "idle_rank_state": "no_scheduled_requests", "model": "Kimi K3",
             "npu_count": 2, "peer_rank_state": "active_requests"},
         "required_refs": ["unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow"]},
    ]
    result = compare_loading_configs(
        tasks, corpus_root=TRANSCRIPT, nav_budget=1024, body_budget=8192)
    b, c = result["configs"]["B"], result["configs"]["C"]
    assert b["duplicate_parent_tokens"] > 0
    assert c["truncated_procedural_units"] == 0
    assert c["ready_to_apply_rate"] == 1.0
    assert c["total_loaded_tokens"] < b["total_loaded_tokens"]
    assert result["categories"]["0day"]["C"]["net_token_savings_c_vs_b"] > 0


@pytest.mark.skipif(not (OPERATORS / "index" / "routing-manifest.json").is_file(),
                    reason="operators knowledge corpus is not mounted")
def test_real_corpus_revisions_drive_growth_without_duplicating_active_units():
    manifest = json.loads((OPERATORS / "index" / "routing-manifest.json").read_text())
    units = manifest["units"]
    events = []
    for row in units:
        events.append({"event_id": f"create-{row['id']}", "action": "create",
                       "category": "new_mechanism", "body_tokens": row["token_estimate"],
                       "read_tokens": 10, "rare_conditions": [row["id"]]})
        for revision in range(2, int(row["revision"]) + 1):
            events.append({"event_id": f"revise-{row['id']}-{revision}",
                           "action": "merge_into_existing", "category": "repeated_success",
                           "body_delta": 10, "read_tokens": 5})
    events.append({"event_id": "consolidate", "action": "consolidate",
                   "category": "rare_condition", "merged_units": 0})
    result = run_growth_replay(events)
    assert result["active_units"] == len(units)
    assert result["delta_u_active"] == len(units)
    assert result["rare_condition_recall_rate"] == 1.0
    assert result["trajectory"][-1]["active_units"] == len(units)
    expected_revisions = sum(max(0, int(row["revision"]) - 1) for row in units) + 1
    assert result["revisions"] == expected_revisions
    assert result["revision_to_creation_ratio"] == pytest.approx(
        expected_revisions / len(units))
    assert result["body_tokens"] == sum(row["token_estimate"] for row in units) + (
        10 * sum(max(0, int(row["revision"]) - 1) for row in units))


def test_growth_event_aliases_infer_the_four_section_13_5_categories():
    result = run_growth_replay([
        {"action": "merge", "body_delta": 10},
        {"action": "create", "body_tokens": 10},
        {"action": "branch", "body_tokens": 10},
        {"action": "consolidate", "merged_units": 0, "rare_conditions": ["rare"]},
    ])
    assert set(result["categories"]) == {
        "repeated_success", "new_mechanism", "version_contradiction", "rare_condition"}
    assert result["categories"]["repeated_success"]["delta_u_active"] == 0
    assert result["categories"]["new_mechanism"]["delta_u_active"] == 1
    assert result["categories"]["version_contradiction"]["delta_u_active"] == 1
    assert result["rare_condition_recall_rate"] == 1.0


def test_budget_excluded_premise_prevents_ready_to_apply(tmp_path):
    root = _write_corpus(tmp_path / "budget-corpus")
    # A premise too large for the body budget must block the whole closure, even
    # when the requested target itself is small enough to load.
    _rewrite_body(root / "_common" / "unit-foundation.md", 1200)
    _rewrite_body(root / "norm" / "rmsnorm" / "unit-mutator.md", 50)
    task = {
        "task_id": "narrow-body-budget", "category": "operator_perf",
        "query": "rmsnorm mutator fusion",
        "conditions": {"dtype": "bfloat16", "soc": "Ascend910B4", "dp": 2},
        "required_refs": ["unit-mutator"],
    }
    result = compare_loading_configs([task], corpus_root=root, body_budget=1200)
    c = result["configs"]["C"]
    assert c["total_loaded_tokens"] > 0
    assert c["ready_to_apply_rate"] == 0.0
