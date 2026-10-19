import copy
import json

import pytest
from test_value_release import evidence, review_runners

from mindie_bench.io import digest, read_json, write_json
from mindie_bench.value_controls import (
    project_component_runner,
    require_collector_source,
    runner_identity,
)
from mindie_bench.value_release import collect, evaluate, markdown

COORDINATOR_REPAIR = "9adfe5ea4ded4847d0d7e045958e184fa3ab65d4"


def repaired_infrastructure():
    suite, observations = evidence()
    suite["instrumentation_components"] = ["coordinator"]
    candidate = observations["studies"]["dp2"][1]
    original = copy.deepcopy(candidate)
    original.update(
        trial_id="original-lease-cleanup-unknown",
        terminal_status="unknown",
        failure_reason="Owned completion was mixed with foreign occupancy",
    )
    original["primary_metrics"].update(resolved=None, reward=None)
    original["primary_metrics"]["tokens"]["complete"] = False
    env = candidate["environment_fingerprint"]
    coordinator_before = env["component_revisions"]["coordinator"]
    runner_before = {
        "source": env["benchmark_source_sha256"],
        "version": env["benchmark_version"],
        "revision": env["component_revisions"]["mindie-bench"],
    }
    env["component_revisions"]["coordinator"] = COORDINATOR_REPAIR
    env["component_revisions"]["mindie-bench"] = "runner-owned-cleanup-repair"
    env["benchmark_version"] = "0.3.0"
    env["benchmark_source_sha256"] = "runner-owned-cleanup-source"
    observations["component_revisions"]["coordinator"] = COORDINATOR_REPAIR
    observations["infrastructure_patches"] = [{
        "component": "coordinator",
        "before": coordinator_before,
        "after": COORDINATOR_REPAIR,
        "studies": ["dp2"],
        "scope": "Owned lease completion accounting; task, model, budget and treatment unchanged.",
    }]
    observations["runner_patches"] = [{
        "before": runner_before["source"],
        "after": env["benchmark_source_sha256"],
        "before_version": runner_before["version"],
        "after_version": env["benchmark_version"],
        "before_revision": runner_before["revision"],
        "after_revision": env["component_revisions"]["mindie-bench"],
        "reason": "Owned cleanup and foreign occupancy are recorded separately.",
    }]
    observations["excluded_attempts"] = [{
        "study_id": "dp2",
        "reason": "One retest after repairing the lease-accounting boundary.",
        "replacement_trial_id": candidate["trial_id"],
        "receipt": original,
    }]
    observations["suite_identity"] = digest(suite)
    review_runners(observations)
    return suite, observations


def test_reviewed_instrumentation_and_runner_version_changes_preserve_raw_evidence(tmp_path):
    suite, observations = repaired_infrastructure()
    before = copy.deepcopy(observations)
    report = evaluate(suite, observations)
    assert report["release_gate"], report["blocking_reasons"]
    assert observations == before
    pair = report["studies"][0]["comparisons"][0]
    assert pair["wall_clock_speedup_on_mutual_success"] == 0.5
    assert pair["reviewed_infrastructure_patches"] == observations["infrastructure_patches"]
    assert report["studies"][0]["first_attempt_statuses"] == {"completed_pass": 1, "unknown": 1}
    assert report["resources"]["attempts"] == 3
    assert not report["resources"]["token_metering_complete"]
    patch = observations["infrastructure_patches"][0]
    rendered = markdown(report)
    assert all(patch[key] in rendered for key in ("before", "after", "scope"))

    manifest = {
        "product_version": "0.0.2",
        "component_revisions": observations["component_revisions"],
        "infrastructure_patches": observations["infrastructure_patches"],
        "runner_patches": observations["runner_patches"],
        "runner_release": observations["runner_release"],
        "runner_release_reviews": observations["runner_release_reviews"],
        "component_reports": [],
        "studies": {"dp2": []},
        "excluded_attempts": [],
    }
    for index, receipt in enumerate(observations["studies"]["dp2"]):
        path = f"selected-{index}.json"
        write_json(tmp_path / path, receipt)
        manifest["studies"]["dp2"].append(path)
    original = observations["excluded_attempts"][0]
    write_json(tmp_path / "original.json", original["receipt"])
    manifest["excluded_attempts"].append({
        key: original[key] for key in ("study_id", "reason", "replacement_trial_id")
    } | {"path": "original.json"})
    write_json(tmp_path / "suite.json", suite)
    write_json(tmp_path / "inputs.json", manifest)
    result = collect(
        suite=tmp_path / "suite.json", manifest=tmp_path / "inputs.json", output=tmp_path / "out",
        collector_source=observations["runner_release"],
    )
    assert result["release_gate"]
    projected = read_json(tmp_path / "out/value-evidence.json")
    assert projected["infrastructure_patches"] == observations["infrastructure_patches"]
    old = projected["excluded_attempts"][0]["receipt"]["environment_fingerprint"]
    new = projected["studies"]["dp2"][1]["environment_fingerprint"]
    assert old["component_revisions"]["coordinator"] != new["component_revisions"]["coordinator"]
    assert old["benchmark_version"] == "0.2.0"
    assert new["benchmark_version"] == "0.3.0"
    assert read_json(tmp_path / "original.json") == original["receipt"]


def test_unreviewed_revision_or_out_of_scope_retest_remains_incompatible():
    suite, observations = repaired_infrastructure()
    observations["infrastructure_patches"] = []
    with pytest.raises(ValueError, match="unreviewed component: coordinator"):
        evaluate(suite, observations)

    suite, observations = repaired_infrastructure()
    candidate = observations["studies"]["dp2"][1]
    candidate["environment_fingerprint"]["component_revisions"]["coordinator"] = "unreviewed-third-revision"
    with pytest.raises(ValueError, match="unreviewed component: coordinator"):
        evaluate(suite, observations)

    suite, observations = repaired_infrastructure()
    other = copy.deepcopy(suite["studies"][0])
    other["id"] = "other-study"
    suite["studies"].append(other)
    observations["studies"]["other-study"] = []
    observations["infrastructure_patches"][0]["studies"] = ["other-study"]
    observations["suite_identity"] = digest(suite)
    with pytest.raises(ValueError, match="unreviewed component: coordinator"):
        evaluate(suite, observations)


@pytest.mark.parametrize("component", ["knowledge", "remote-dev", "codex"])
def test_treatment_or_host_sources_cannot_be_relabelled_as_infrastructure(component):
    suite, observations = repaired_infrastructure()
    suite["instrumentation_components"].append(component)
    patch = observations["infrastructure_patches"][0]
    patch.update(
        component=component,
        before=observations["component_revisions"][component],
        after="changed-treatment",
    )
    observations["component_revisions"][component] = patch["after"]
    observations["suite_identity"] = digest(suite)
    with pytest.raises(ValueError, match="cannot change a treatment"):
        evaluate(suite, observations)


def test_infrastructure_mapping_does_not_certify_old_component_scenario_results():
    suite, observations = repaired_infrastructure()
    suite["component_cases"] = [{
        "id": "coordinator-contention", "component": "coordinator",
        "distribution": "mindie-coordinator", "repetitions": 1,
    }]
    observations["component_cases"] = [{
        "id": "coordinator-contention", "component": "coordinator", "status": "pass",
        "measurements": [{"repetition": 1, "status": "pass", "baseline": {}, "candidate": {}}],
        "source_revisions": {"mindie-coordinator": {
            "revision": observations["infrastructure_patches"][0]["before"], "editable": False,
        }},
    }]
    observations["suite_identity"] = digest(suite)
    report = evaluate(suite, observations)
    assert not report["release_gate"]
    assert "component_source_mismatch:coordinator-contention" in report["blocking_reasons"]


def test_mapping_is_not_needed_for_a_study_without_coordinator_provenance():
    suite, observations = evidence()
    suite["instrumentation_components"] = ["coordinator"]
    for row in observations["studies"]["dp2"]:
        row["environment_fingerprint"]["component_revisions"].pop("coordinator")
    observations["component_revisions"]["coordinator"] = COORDINATOR_REPAIR
    observations["suite_identity"] = digest(suite)
    assert evaluate(suite, observations)["release_gate"]


def test_final_runner_update_needs_exact_measured_review_without_rewriting_receipts():
    suite, observations = evidence()
    rows = observations["studies"]["dp2"]
    for row in rows:
        row["environment_fingerprint"]["component_revisions"].pop("mindie-bench")
    original = copy.deepcopy(rows)
    observations["component_revisions"]["mindie-bench"] = "final-release"
    assert not evaluate(suite, observations)["release_gate"]
    observations["runner_release"] = {
        "revision": "final-release", "version": "0.3.0", "source_sha256": "final-source",
    }
    assert not evaluate(suite, observations)["release_gate"]
    review_runners(observations)
    report = evaluate(suite, observations)
    assert report["release_gate"], report["blocking_reasons"]
    assert rows == original
    assert report["runner_release_reviews"][0]["measured"]["revision"] is None
    assert report["runner_release_reviews"] == observations["runner_release_reviews"]
    for field, changed in (
        ("benchmark_source_sha256", "another-loaded-package"),
        ("benchmark_version", "another-version"),
    ):
        env = rows[0]["environment_fingerprint"]
        saved = env[field]
        env[field] = changed
        assert not evaluate(suite, observations)["release_gate"]
        env[field] = saved
    observations["runner_release"]["revision"] = "unreviewed-next-release"
    observations["component_revisions"]["mindie-bench"] = "unreviewed-next-release"
    with pytest.raises(ValueError, match="different final source"):
        evaluate(suite, observations)


def test_historical_runner_is_reviewed_even_when_not_selected_for_pairing():
    suite, observations = evidence()
    old = copy.deepcopy(observations["studies"]["dp2"][0])
    old.update(
        trial_id="startup-history", terminal_status="infra_failed",
        failure_reason="Setup failed before the task started",
    )
    old["primary_metrics"].update(resolved=None, reward=None)
    old["environment_fingerprint"]["benchmark_source_sha256"] = "historical-runner"
    observations["excluded_attempts"] = [{
        "study_id": "dp2", "reason": "Setup failed before the task started", "receipt": old,
    }]
    report = evaluate(suite, observations)
    assert not report["release_gate"]
    assert "unreviewed_release_runner:startup-history" in report["blocking_reasons"]
    review_runners(observations)
    report = evaluate(suite, observations)
    assert report["release_gate"], report["blocking_reasons"]
    assert report["resources"]["attempts"] == 3


def test_component_runner_projection_preserves_dirty_source_and_requires_scoped_review(tmp_path):
    suite, observations = evidence()
    source = observations["component_revisions"]["coordinator"]
    suite["component_cases"] = [{
        "id": "coordinator-contention", "component": "coordinator",
        "distribution": "mindie-coordinator", "repetitions": 1,
    }]
    case = {
        "id": "coordinator-contention", "component": "coordinator", "status": "pass",
        "scope": "Owned process contention", "baseline": "no queue", "candidate": "queue",
        "repetitions": 1, "limitations": [],
        "measurements": [{"repetition": 1, "status": "pass", "baseline": {}, "candidate": {}}],
        "source_revisions": {"mindie-coordinator": {
            "revision": source, "version": "0.5.3", "editable": False,
        }},
    }
    raw = {
        "runner_source": {"revision": "historical-before", "dirty": True},
        "runner_source_after": {"revision": "historical-after", "dirty": False},
        "environment": {"executable": "/private/runner/bin/python"},
        "cases": [case],
    }
    observations["component_cases"] = [case]
    observations["component_runner_reports"] = [project_component_runner(raw)]
    observations["suite_identity"] = digest(suite)
    assert not evaluate(suite, observations)["release_gate"]
    observations["runner_release_reviews"] = [{
        "kind": "component",
        "measured": {key: raw[key] for key in ("runner_source", "runner_source_after")},
        "release": copy.deepcopy(observations["runner_release"]),
        "scope": "Historical scenario preserved; later cleanup and publication changes only.",
    }]
    assert evaluate(suite, observations)["release_gate"]
    manifest = {
        key: observations[key] for key in (
            "product_version", "component_revisions", "runner_release", "runner_release_reviews",
        )
    } | {"component_reports": ["component.json"], "studies": {"dp2": []}}
    for row in observations["studies"]["dp2"]:
        name = f"{row['trial_id']}.json"
        write_json(tmp_path / name, row)
        manifest["studies"]["dp2"].append(name)
    for name, content in (("component", raw), ("suite", suite), ("manifest", manifest)):
        write_json(tmp_path / f"{name}.json", content)
    result = collect(
        suite=tmp_path / "suite.json", manifest=tmp_path / "manifest.json",
        output=tmp_path / "output", collector_source=observations["runner_release"],
    )
    assert result["release_gate"]
    public = read_json(tmp_path / "output/value-evidence.json")
    assert public["component_runner_reports"] == observations["component_runner_reports"]
    assert public["component_runner_reports"][0]["runner_source"]["dirty"] is True
    assert "/private/" not in json.dumps(public)
    public["component_runner_reports"][0]["runner_source"]["dirty"] = False
    assert not evaluate(suite, public)["release_gate"]


def test_publication_collector_must_match_loaded_revision_version_and_content():
    _, observations = evidence()
    expected = observations["runner_release"]
    require_collector_source(expected, expected | {"editable": False, "dirty": None})
    wheel = expected | {"revision": None, "editable": False, "installation": "installed", "dirty": None}
    assert require_collector_source(expected, wheel) == wheel
    for field in ("revision", "version", "source_sha256"):
        with pytest.raises(ValueError, match="loaded collector"):
            require_collector_source(expected, expected | {field: "different"})
    with pytest.raises(ValueError, match="loaded collector"):
        require_collector_source(expected, expected | {"dirty": True})
    assert runner_identity(expected | {"metadata_path": "/private/path"}) == expected


def test_cli_without_release_manifest_rejects_another_loaded_collector(tmp_path, capsys):
    from mindie_bench.cli import main

    suite, observations = evidence()
    observations["component_revisions"]["mindie-bench"] = "another-release"
    observations["runner_release"] = {
        "revision": "another-release", "version": "0.3.0", "source_sha256": "another-source",
    }
    review_runners(observations)
    assert evaluate(suite, observations)["release_gate"]
    write_json(tmp_path / "suite.json", suite)
    write_json(tmp_path / "evidence.json", observations)
    result = main([
        "value-verify", "--suite", str(tmp_path / "suite.json"),
        "--evidence", str(tmp_path / "evidence.json"), "--output", str(tmp_path / "public"),
    ])
    assert result == 2
    assert "loaded collector source" in capsys.readouterr().err
    assert not (tmp_path / "public/value-results.json").exists()
