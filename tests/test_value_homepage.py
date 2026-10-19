import copy

import pytest
from test_value_release import evidence

from mindie_bench.io import digest
from mindie_bench.value_homepage import BEGIN_MARKER, END_MARKER, homepage_markdown
from mindie_bench.value_release import evaluate, write_report


def table_rows(markdown, prefix):
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith(f"| {prefix}")
    ]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_separate_models_keep_actual_selected_counts_and_negative_time_ratio(language):
    suite, observations = evidence()
    other = copy.deepcopy(suite["studies"][0])
    other.update(id="other-model", model="another-model")
    suite["studies"].append(other)
    rows = copy.deepcopy(observations["studies"]["dp2"])
    for row in rows:
        row["trial_id"] += "-other"
        row["environment_fingerprint"]["base_model"] = other["model"]
    rows[1]["primary_metrics"]["delivery_wall_clock_sec"] /= 8
    observations["studies"]["other-model"] = rows
    observations["suite_identity"] = digest(suite)
    report = evaluate(suite, observations)
    before = copy.deepcopy(report)

    rendered = homepage_markdown(report, language)

    assert rendered.startswith(BEGIN_MARKER + "\n")
    assert rendered.endswith(END_MARKER + "\n")
    assert report == before
    first = table_rows(rendered, "dp2 /")[0]
    second = table_rows(rendered, "other-model /")[0]
    assert first[1:] == ["1", "G0 → G2", "1/1", "1/1", "0.500x", "1"]
    assert second[1:] == ["1", "G0 → G2", "1/1", "1/1", "4.000x", "1"]
    assert "home/codex/gpt-6.1-sol (xhigh)" in first[0]
    assert "another-model (xhigh)" in second[0]
    assert "docs/releases/0.0.2/value-results.md" in rendered


@pytest.mark.parametrize("language,missing", [("zh", "未测"), ("en", "not measured")])
def test_real_task_failure_remains_in_denominator_without_a_success_timing_claim(language, missing):
    suite, observations = evidence()
    candidate = observations["studies"]["dp2"][1]
    candidate.update(terminal_status="completed_fail", failure_reason="oracle_failed")
    candidate["primary_metrics"].update(reward=0.0, resolved=False)
    report = evaluate(suite, observations)

    assert report["release_gate"]
    row = table_rows(homepage_markdown(report, language), "dp2 /")[0]
    assert row[3:] == ["1/1", "0/1", missing, "0"]


def test_missing_trial_cannot_be_promoted_to_homepage_results(tmp_path):
    suite, observations = evidence()
    write_report(evaluate(suite, observations), tmp_path)
    assert (tmp_path / "value-homepage.md").is_file()
    assert (tmp_path / "value-homepage.en.md").is_file()
    observations["studies"]["dp2"].pop()
    incomplete = evaluate(suite, observations)
    assert not incomplete["release_gate"]
    with pytest.raises(ValueError, match="passed release gate"):
        homepage_markdown(incomplete)
    write_report(incomplete, tmp_path)
    assert not list(tmp_path.glob("value-homepage*"))
    assert (tmp_path / "value-results.json").is_file()


def transfer_report():
    suite, observations = evidence()
    template = observations["studies"]["dp2"][0]
    study = suite["studies"][0]
    study.update(
        id="transfer",
        tier=template["tier"],
        repetitions=3,
        tasks={"seed": "seed", "first-eval": "eval", "second-eval": "eval"},
        task_digests={"seed": "seed-content", "first-eval": "first-content", "second-eval": "second-content"},
        groups=["G1", "G3"],
        group_components={"G1": ["remote-dev"], "G3": ["knowledge", "remote-dev"]},
        comparisons=[["G1", "G3"]],
        transfer={"group": "G3", "seed_task": "seed", "eval_tasks": ["first-eval", "second-eval"]},
    )
    rows, lineages = [], []
    for repetition in range(1, 4):
        lineage = {
            "repetition": repetition,
            "seed_trial_id": f"seed-G3-{repetition}",
            "seed_captured_corpus_identity": "captured-seed",
            "seed_capture_last_model_end_monotonic_sec": 100,
            "evals": [],
        }
        for group in study["groups"]:
            for index, (task, role) in enumerate(study["tasks"].items(), 1):
                row = copy.deepcopy(template)
                row.update(
                    trial_id=f"{task}-{group}-{repetition}", instance_id=task, cluster="transfer",
                    task_role=role, task_digest=study["task_digests"][task],
                    repetition=repetition, ablation_group=group,
                )
                row["environment_fingerprint"].update(
                    components=study["group_components"][group],
                    skills_sha256="released-skills",
                    corpus_view_sha256="captured-seed",
                )
                # Delivery already includes the measured capture time.
                row["primary_metrics"].update(
                    total_wall_clock_sec=index * 10,
                    delivery_wall_clock_sec=index * 10 + (5 if group == "G3" else 0),
                )
                if group == "G3":
                    row["auxiliary_metrics"].update(
                        capture_wall_clock_sec=5,
                        captured_corpus_sha256="captured-seed",
                        capture={"status": "completed", "corpus_lint": {"status": "passed"}},
                    )
                    if role == "eval":
                        hit = not (repetition == 3 and task == "first-eval")
                        row["primary_metrics"].update(
                            foreground_knowledge_queries_count=int(hit),
                            foreground_knowledge_explains_count=int(hit),
                        )
                        lineage["evals"].append({
                            "trial_id": row["trial_id"], "task_id": task,
                            "initial_corpus_identity": "captured-seed",
                            "seed_query_hit_observed": hit, "seed_explain_read_observed": hit,
                            "native_query_call_ids": ["query"] if hit else [],
                            "native_explain_call_ids": ["explain"] if hit else [],
                            "time_order_passed": True,
                            "eval_first_model_start_monotonic_sec": 200,
                        })
                rows.append(row)
        lineages.append(lineage)
    original = copy.deepcopy(next(row for row in rows if row["trial_id"] == "first-eval-G3-1"))
    original.update(trial_id="original-unknown", terminal_status="unknown", failure_reason="stream EOF")
    original["primary_metrics"].update(reward=None, resolved=None, delivery_wall_clock_sec=7)
    original["primary_metrics"]["tokens"]["complete"] = False
    observations.update(
        studies={"transfer": rows},
        transfer_lineages={"transfer": lineages},
        excluded_attempts=[{
            "study_id": "transfer", "reason": "Single transport recovery, original cost retained",
            "replacement_trial_id": "first-eval-G3-1", "receipt": original,
        }],
        suite_identity=digest(suite),
    )
    return evaluate(suite, observations)


@pytest.mark.parametrize("language", ["zh", "en"])
def test_transfer_preserves_missed_seed_and_counts_capture_and_retest_once(language):
    report = transfer_report()
    assert report["release_gate"], report["blocking_reasons"]
    rendered = homepage_markdown(report, language)
    pair = table_rows(rendered, "transfer /")[0]
    assert pair[1:5] == ["3", "G1 → G3", "6/6", "6/6"]
    assert pair[5:] == ["0.833x", "6"]
    assert table_rows(rendered, "G1 |") == [["G1", "180.0", "180.0"]]
    assert table_rows(rendered, "G3 |") == [["G3", "225.0", "232.0"]]
    assert rendered.count("5/6") == 2
    assert "docs/releases/0.0.2/value-results.md" in rendered
    assert report["studies"][0]["resources"]["historical"]["attempts"] == 1


def component_report():
    suite, observations = evidence()
    scenarios = [
        (
            "coordinator-contention", "coordinator", "mindie-coordinator",
            {"successful_listeners": 1, "bind_conflicts": 1, "duplicate_device_selections": 1},
            {"successful_listeners": 2, "bind_conflicts": 0, "duplicate_device_selections": 0},
        ),
        (
            "npu-top-physical-owner-query", "npu-top", "npu-top",
            {"human_answer_accuracy": None, "human_completion_time_sec": None},
            {"golden_physical_devices_correct": 24, "golden_process_assignments_correct": 16,
             "golden_process_assignments_expected": 16, "live_container_owner_correct": True},
        ),
        (
            "diagnostics-fault-delivery", "diagnostics", "mindie-diagnostics",
            {"failure_occurrences": 3, "raw_tracebacks": 3},
            {"failure_occurrences": 3, "delivered_reports": 1, "http_posts": 1,
             "useful_fault_fields_retained": True, "private_fixture_removed": True,
             "withdrawal_prevented_requests": True, "failure_to_delivery_sec": 60.58435137500055},
        ),
    ]
    for case_id, component, distribution, baseline, candidate in scenarios:
        suite["component_cases"].append({
            "id": case_id, "repetitions": 3, "distribution": distribution, "component": component,
        })
        observations["component_cases"].append({
            "id": case_id, "component": component, "status": "pass",
            "measurements": [
                {"repetition": rep, "status": "pass", "baseline": baseline, "candidate": candidate}
                for rep in range(1, 4)
            ],
            "source_revisions": {
                distribution: {"revision": observations["component_revisions"][component], "editable": False}
            },
        })
    observations["suite_identity"] = digest(suite)
    observations["component_runner_reports"] = [{
        "case_ids": [case["id"] for case in observations["component_cases"]],
        "runner_source": observations["runner_release"],
        "runner_source_after": observations["runner_release"],
    }]
    return evaluate(suite, observations)


@pytest.mark.parametrize("language", ["zh", "en"])
def test_component_numbers_keep_replay_counts_separate_from_unmeasured_human_results(language):
    report = component_report()
    assert report["release_gate"], report["blocking_reasons"]
    rendered = homepage_markdown(report, language)
    lines = {line.split(":", 1)[0].split("：", 1)[0]: line for line in rendered.splitlines() if line.startswith("- ")}
    coordinator = lines["- coordinator"]
    assert "3 → 6" in coordinator
    assert coordinator.count("3 → 0") == 2
    npu = lines["- npu-top"]
    assert "24" in npu and "16/16" in npu and "3/3" in npu
    assert npu.count("未测" if language == "zh" else "not measured") == 2
    diagnostics = lines["- diagnostics"]
    assert "9" in diagnostics and "60.58" in diagnostics
    assert diagnostics.count("3/3") == 3
