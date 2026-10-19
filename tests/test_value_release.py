import copy
from pathlib import Path

import pytest

from mindie_bench.io import digest, read_json
from mindie_bench.models import TrialReceipt
from mindie_bench.value_release import evaluate, markdown, public_receipt


def evidence():
    # Sanitized real final-component DP2 receipt; only experiment conditions change.
    original = read_json(Path(__file__).parent / "fixtures/value-dp2-g0-receipt.json")
    candidate = copy.deepcopy(original)
    candidate.update(trial_id="candidate", ablation_group="G2")
    candidate["environment_fingerprint"].update(
        components=["knowledge", "remote-dev"], skills_sha256="released-skills"
    )
    candidate["primary_metrics"]["delivery_wall_clock_sec"] *= 2
    suite = {
        "schema_version": "mindie-value-suite/1",
        "id": "release-test",
        "studies": [
            {
                "id": "dp2",
                "scope": "paired DP2",
                "harness": "codex",
                "model": original["environment_fingerprint"]["base_model"],
                "reasoning_effort": "xhigh",
                "tier": "l3",
                "repetitions": 1,
                "tasks": {original["instance_id"]: "eval"},
                "task_digests": {original["instance_id"]: original["task_digest"]},
                "groups": ["G0", "G2"],
                "group_components": {"G0": [], "G2": ["knowledge", "remote-dev"]},
                "comparisons": [["G0", "G2"]],
            }
        ],
        "component_cases": [],
    }
    inputs = {
        "schema_version": "mindie-value-evidence/1",
        "suite_identity": digest(suite),
        "product_version": "0.0.2",
        "component_revisions": original["environment_fingerprint"]["component_revisions"].copy(),
        "studies": {"dp2": [original, candidate]},
        "component_cases": [],
    }
    return suite, inputs


def test_negative_result_is_published_but_missing_or_mismatched_trial_blocks():
    suite, inputs = evidence()
    report = evaluate(suite, inputs)
    assert report["release_gate"]
    assert report["studies"][0]["comparisons"][0]["wall_clock_speedup_on_mutual_success"] == 0.5
    assert "0.500x" in markdown(report)
    inputs["studies"]["dp2"][1]["environment_fingerprint"]["base_model"] = "different-model"
    assert not evaluate(suite, inputs)["release_gate"]
    inputs["studies"]["dp2"].pop()
    incomplete = evaluate(suite, inputs)
    assert not incomplete["release_gate"]
    assert incomplete["studies"][0]["observed_cells"] == 1


def test_actual_task_failure_remains_in_inventory_and_all_attempt_costs():
    suite, inputs = evidence()
    candidate = inputs["studies"]["dp2"][1]
    candidate.update(terminal_status="completed_fail", failure_reason="oracle_failed")
    candidate["primary_metrics"].update(reward=0.0, resolved=False)
    report = evaluate(suite, inputs)
    assert report["release_gate"]
    study = report["studies"][0]
    assert study["statuses"] == {"completed_pass": 1, "completed_fail": 1}
    assert study["comparisons"][0]["wall_clock_speedup_on_mutual_success"] is None
    assert study["groups"]["G2"]["observed_delivery_wall_clock_sec_lower_bound"] > 0
    assert study["groups"]["G2"]["observed_tokens_lower_bound"]["input_uncached"] > 0
    assert study["groups"]["G2"]["cost_per_resolved_task"] is None


def test_seed_is_in_total_cost_and_missing_capture_blocks():
    suite, inputs = evidence()
    study = suite["studies"][0]
    study["groups"] = ["G1", "G3"]
    study["group_components"] = {
        "G1": ["remote-dev"], "G3": ["knowledge", "remote-dev"]
    }
    study["comparisons"] = [["G1", "G3"]]
    study["tasks"]["earlier-task"] = "seed"
    study["task_digests"]["earlier-task"] = "seed-content"
    for row, group in zip(inputs["studies"]["dp2"], study["groups"], strict=True):
        row.update(ablation_group=group, cluster="transfer")
        row["environment_fingerprint"]["components"] = study["group_components"][group]
        row["auxiliary_metrics"]["capture"] = {"status": "completed"}
    for group in study["groups"]:
        row = copy.deepcopy(inputs["studies"]["dp2"][0])
        row.update(
            trial_id=f"seed-{group}",
            instance_id="earlier-task",
            task_role="seed",
            task_digest="seed-content",
            ablation_group=group,
        )
        row["auxiliary_metrics"]["capture"] = {"status": "completed"}
        row["environment_fingerprint"]["components"] = study["group_components"][group]
        row["environment_fingerprint"]["skills_sha256"] = "released-skills"
        inputs["studies"]["dp2"].append(row)
    inputs["suite_identity"] = digest(suite)
    result = evaluate(suite, inputs)
    assert result["release_gate"]
    assert result["studies"][0]["groups"]["G3"]["attempts"] == 2
    pair = result["studies"][0]["comparisons"][0]
    assert pair["candidate"]["seed_setup_attempts"] == 1
    inputs["studies"]["dp2"][-1]["auxiliary_metrics"]["capture"]["status"] = "unknown"
    assert not evaluate(suite, inputs)["release_gate"]


def test_source_mismatch_and_duplicate_receipts_cannot_publish():
    suite, inputs = evidence()
    row = inputs["studies"]["dp2"][1]
    row["environment_fingerprint"].update(
        components=["knowledge"],
        component_revisions={"knowledge": "old-source"},
    )
    inputs["component_revisions"] = {"knowledge": "final-source"}
    assert not evaluate(suite, inputs)["release_gate"]
    inputs["studies"]["dp2"].append(row)
    with pytest.raises(ValueError, match="counted more than once"):
        evaluate(suite, inputs)


def test_targeted_runner_repair_preserves_existing_results_without_relaxing_model_controls():
    suite, inputs = evidence()
    baseline, candidate = inputs["studies"]["dp2"]
    before = baseline["environment_fingerprint"]["benchmark_source_sha256"]
    candidate["environment_fingerprint"]["benchmark_source_sha256"] = "patched-runner"
    assert not evaluate(suite, inputs)["release_gate"]
    inputs["runner_patches"] = [
        {
            "before": before,
            "after": "patched-runner",
            "reason": "Only transport error reporting changed; normal execution is unchanged.",
        }
    ]
    assert evaluate(suite, inputs)["release_gate"]
    assert baseline["environment_fingerprint"]["benchmark_source_sha256"] == before
    candidate["environment_fingerprint"]["base_model"] = "another-model"
    assert not evaluate(suite, inputs)["release_gate"]


def test_public_projection_retains_measurements_without_native_paths_or_messages():
    _, inputs = evidence()
    raw = inputs["studies"]["dp2"][0]
    raw["environment_fingerprint"].update(
        provider_url="http://private-service",
        remote_environment={"host": "private-host"},
        mcp_config={"token": "must-not-export"},
    )
    raw["artifacts"] = {"native": "/private/user/transcript"}
    raw["errors"] = [{"stage": "tool", "message": "private-command"}]
    projected = public_receipt(TrialReceipt.model_validate(raw))
    assert projected["primary_metrics"] == raw["primary_metrics"]
    assert projected["artifacts"] == {}
    assert "must-not-export" not in str(projected)
    assert "private-service" not in str(projected)
    assert "private-command" not in str(projected)


def test_single_cell_recovery_preserves_first_attempt_outcome_and_total_cost():
    suite, inputs = evidence()
    old = copy.deepcopy(inputs["studies"]["dp2"][1])
    old.update(
        trial_id="transport-failure", terminal_status="unknown", failure_reason="provider TLS failure"
    )
    old["primary_metrics"].update(resolved=None, reward=None)
    old["primary_metrics"]["delivery_wall_clock_sec"] = 10
    old["primary_metrics"]["tokens"]["complete"] = False
    inputs["excluded_attempts"] = [{
        "study_id": "dp2", "reason": "provider TLS failure",
        "replacement_trial_id": "candidate", "receipt": old,
    }]
    report = evaluate(suite, inputs)
    assert report["release_gate"]
    study = report["studies"][0]
    assert study["first_attempt_statuses"] == {"completed_pass": 1, "unknown": 1}
    assert study["resources"]["all_executed"]["attempts"] == 3
    assert not study["resources"]["all_executed"]["token_metering_complete"]
    assert (
        study["resources"]["all_executed"]["observed_delivery_wall_clock_sec_lower_bound"]
        == study["resources"]["selected"]["observed_delivery_wall_clock_sec_lower_bound"] + 10
    )
    assert "original infrastructure failures" in markdown(report)
    old["terminal_status"] = "completed_fail"
    old["primary_metrics"].update(resolved=False, reward=0.0)
    with pytest.raises(ValueError, match="valid task outcomes"):
        evaluate(suite, inputs)
    old["primary_metrics"]["total_turns"] = 0
    old["primary_metrics"]["tokens"] = {
        "input_uncached": 0, "input_cached": 0, "cache_creation": 0,
        "output": 0, "indexer_small_model_total": 0, "complete": False,
    }
    item = inputs["excluded_attempts"][0]
    item["classification"] = "native_startup_infrastructure"
    item["startup_observation"] = {
        "journal_records": 8, "model_requests": 0, "broker_identity": "recorded-startup-journal"
    }
    assert evaluate(suite, inputs)["release_gate"]
    item["startup_observation"]["model_requests"] = 1
    with pytest.raises(ValueError, match="valid task outcomes"):
        evaluate(suite, inputs)


def test_plugin_sources_skills_ablation_and_metering_are_required():
    suite, inputs = evidence()
    candidate = inputs["studies"]["dp2"][1]
    candidate["environment_fingerprint"].update(components=[], skills_sha256=None)
    candidate["environment_fingerprint"]["component_revisions"]["codex"] = "old"
    candidate["primary_metrics"]["tokens"]["complete"] = False
    report = evaluate(suite, inputs)
    assert not report["release_gate"]
    reasons = report["blocking_reasons"]
    assert "dp2:ablation_components_mismatch:candidate" in reasons
    assert "dp2:plugin_skills_missing:candidate" in reasons
    assert "dp2:source_mismatch:candidate:codex" in reasons
    assert "dp2:token_metering_incomplete:candidate" in reasons


def test_retest_cannot_change_model_budget_or_unreviewed_runtime():
    suite, inputs = evidence()
    candidate = inputs["studies"]["dp2"][1]
    original = copy.deepcopy(candidate)
    original.update(trial_id="original", terminal_status="unknown", failure_reason="transport failure")
    original["primary_metrics"].update(reward=None, resolved=None)
    inputs["excluded_attempts"] = [{
        "study_id": "dp2", "reason": "transport repair",
        "replacement_trial_id": candidate["trial_id"], "receipt": original,
    }]
    assert evaluate(suite, inputs)["release_gate"]
    for field, changed in (
        ("base_model", "different-model"), ("reasoning_effort", "low"),
        ("budget", {"max_turns": 1}), ("harness_binary_sha256", "other-native"),
        ("corpus_view_sha256", "other-knowledge"),
    ):
        saved = copy.deepcopy(original["environment_fingerprint"])
        original["environment_fingerprint"][field] = changed
        with pytest.raises(ValueError, match="experimental control"):
            evaluate(suite, inputs)
        original["environment_fingerprint"] = saved
    original["environment_fingerprint"]["component_revisions"]["knowledge"] = "other-runtime"
    with pytest.raises(ValueError, match="unreviewed component"):
        evaluate(suite, inputs)
    original["environment_fingerprint"] = copy.deepcopy(candidate["environment_fingerprint"])
    original["environment_fingerprint"].update(
        benchmark_source_sha256="prior-runner", benchmark_version="old-version"
    )
    original["environment_fingerprint"]["component_revisions"]["mindie-bench"] = "prior-source"
    inputs["runner_patches"] = [{
        "before": "prior-runner",
        "after": candidate["environment_fingerprint"]["benchmark_source_sha256"],
        "before_version": "old-version",
        "after_version": candidate["environment_fingerprint"]["benchmark_version"],
        "before_revision": "prior-source",
        "after_revision": candidate["environment_fingerprint"]["component_revisions"]["mindie-bench"],
        "reason": "Reviewed startup transport repair.",
    }]
    assert evaluate(suite, inputs)["release_gate"]


def test_paid_setup_is_accounted_separately_without_inventing_missing_cost():
    suite, inputs = evidence()
    task_totals = evaluate(suite, inputs)["resources"]
    inputs["setup_reports"] = [{
        "schema_version": "mindie-setup-costs/1", "plugin": "codex",
        "scope": "native summary setup", "limitations": [],
        "totals": {"cost_usd": 0, "elapsed_sec": 99},
        "operations": [{
            "id": "summary-probe", "operation": "summary setup", "model": "luna",
            "reasoning_effort": "max", "status": "pass", "elapsed_sec": None,
            "tokens": {"input_uncached": 10501, "input_cached": 0, "output": 202,
                       "cache_creation": 0, "indexer_small_model_total": 0},
            "token_metering_complete": True, "cost_usd": None,
            "measurement_basis": "native usage", "private_path": "/private/probe",
        }],
    }]
    report = evaluate(suite, inputs)
    assert report["resources"] == task_totals
    totals = report["setup_resources"]
    assert totals["observed_tokens_lower_bound"]["input_uncached"] == 10501
    assert totals["observed_elapsed_sec_lower_bound"] is None
    assert totals["observed_cost_usd_lower_bound"] is None
    assert not totals["cost_complete"]
    assert "/private/probe" not in str(report["setup_reports"])
    assert "summary setup" in markdown(report)
    inputs["setup_reports"].append(copy.deepcopy(inputs["setup_reports"][0]))
    with pytest.raises(ValueError, match="setup operation"):
        evaluate(suite, inputs)


def test_source_revision_equivalence_requires_unchanged_loaded_files():
    suite, inputs = evidence()
    original = inputs["studies"]["dp2"][0]["environment_fingerprint"]["component_revisions"]["codex"]
    inputs["component_revisions"]["codex"] = "merged-source"
    patch = {
        "component": "codex", "before": original, "after": "merged-source",
        "groups": ["G0", "G2"], "reason": "Only the unused summary provider changed.",
    }
    inputs["source_patches"] = [patch]
    assert not evaluate(suite, inputs)["release_gate"]
    patch["loaded_paths"] = [{"path": "skills", "before": "same-tree", "after": "same-tree"}]
    assert evaluate(suite, inputs)["release_gate"]
    patch["loaded_paths"][0]["after"] = "different-tree"
    assert not evaluate(suite, inputs)["release_gate"]


def test_native_plugin_lifecycle_evidence_is_independent_of_task_comparisons():
    suite, inputs = evidence()
    suite["native_plugins"] = {"codex": ["tools registered", "native capture"]}
    inputs["suite_identity"] = digest(suite)
    assert not evaluate(suite, inputs)["release_gate"]
    inputs["native_plugins"] = [{
        "plugin": "codex", "revision": inputs["component_revisions"]["codex"],
        "status": "pass", "limitations": ["Native acceptance, not model productivity."],
        "checks": [{"name": name, "status": "pass"} for name in suite["native_plugins"]["codex"]],
    }]
    assert evaluate(suite, inputs)["release_gate"]
    inputs["native_plugins"][0]["checks"][-1]["status"] = "not_run"
    assert not evaluate(suite, inputs)["release_gate"]


def test_transfer_requires_the_same_captured_seed_and_actual_later_reads():
    suite, inputs = evidence()
    study = suite["studies"][0]
    study.update(groups=["G3"], comparisons=[], group_components={"G3": ["knowledge", "remote-dev"]})
    evaluation = inputs["studies"]["dp2"][1]
    evaluation["ablation_group"] = "G3"
    evaluation["environment_fingerprint"]["corpus_view_sha256"] = "captured-seed"
    evaluation["primary_metrics"].update(
        foreground_knowledge_queries_count=1, foreground_knowledge_explains_count=1
    )
    evaluation["auxiliary_metrics"]["capture"] = {"status": "completed"}
    seed = copy.deepcopy(evaluation)
    seed.update(trial_id="seed", instance_id="seed-task", task_digest="seed-content", task_role="seed")
    seed["auxiliary_metrics"].update(
        captured_corpus_sha256="captured-seed",
        capture={"status": "completed", "corpus_lint": {"status": "passed"}},
    )
    study["tasks"]["seed-task"] = "seed"
    study["task_digests"]["seed-task"] = "seed-content"
    study["transfer"] = {
        "group": "G3", "seed_task": "seed-task", "eval_tasks": [evaluation["instance_id"]]
    }
    inputs["suite_identity"] = digest(suite)
    inputs["studies"]["dp2"] = [seed, evaluation]
    entry = {
        "trial_id": "candidate", "task_id": evaluation["instance_id"],
        "initial_corpus_identity": "captured-seed", "seed_query_hit_observed": True,
        "seed_explain_read_observed": True, "native_query_call_ids": ["actual-query"],
        "native_explain_call_ids": ["actual-explain"], "time_order_passed": True,
        "eval_first_model_start_monotonic_sec": 200,
    }
    inputs["transfer_lineages"] = {"dp2": [{
        "repetition": 1, "seed_trial_id": "seed", "seed_captured_corpus_identity": "captured-seed",
        "seed_capture_last_model_end_monotonic_sec": 100, "evals": [entry],
    }]}
    assert evaluate(suite, inputs)["release_gate"]
    entry["seed_query_hit_observed"] = False
    entry["seed_explain_read_observed"] = False
    entry["native_query_call_ids"] = []
    entry["native_explain_call_ids"] = []
    assert evaluate(suite, inputs)["release_gate"]
    assert "| False | False |" in markdown(evaluate(suite, inputs))
    evaluation["environment_fingerprint"]["corpus_view_sha256"] = "unrelated-corpus"
    assert not evaluate(suite, inputs)["release_gate"]
    evaluation["environment_fingerprint"]["corpus_view_sha256"] = "captured-seed"
    entry["eval_first_model_start_monotonic_sec"] = 50
    assert not evaluate(suite, inputs)["release_gate"]
