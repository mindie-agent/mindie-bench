import hashlib

import pytest

from mindie_bench.evaluation.accounting import account_cells, audit_report
from mindie_bench.evaluation.timing import StageRecorder
from mindie_bench.io import read_json, write_json
from mindie_bench.models import TrialReceipt


def receipt(identity="seed", status="unknown"):
    return TrialReceipt(
        trial_id=identity + "-attempt",
        instance_id=identity,
        repetition=1,
        tier="l2",
        ablation_group="G3",
        knowledge_gradient="near_exact",
        task_role="seed" if identity == "seed" else "eval",
        task_digest="task-" + identity,
        experiment_digest="experiment",
        environment_fingerprint={},
        terminal_status=status,
        primary_metrics=dict(reward=None, resolved=None),
        failure_reason="lost original acknowledgement",
        evidence="synthetic",
    )


def plan():
    return dict(
        tasks=[{"path": "seed"}, {"path": "eval"}],
        groups=["G3"],
        repetitions=1,
        cells=[
            dict(
                instance_id=identity,
                tier="l2",
                group="G3",
                repetition=1,
                role=identity,
                task_digest="task-" + identity,
                cluster="family",
                corpus_id="ascend",
            )
            for identity in ["seed", "eval"]
        ],
    )


def test_unknown_seed_blocks_eval_without_fabricating_a_receipt():
    manifest = plan()
    blocked = dict(
        manifest["cells"][1],
        prerequisite_trial_id="seed-attempt",
        reason="seed_prerequisite_not_completed",
    )
    report = account_cells(manifest, [receipt()], blocked_cells=[blocked])
    assert report["planned_cells"] == 2
    assert report["terminal_receipts"] == 1
    assert report["raw_statuses"] == {"unknown": 1}
    assert report["blocked_count"] == 1
    assert report["all_cells_accounted_for"] is True
    assert report["all_planned_cells_passed"] is False
    assert report["terminal_accounting_complete"] is False


def test_legacy_absence_does_not_become_a_blocked_cell():
    manifest = plan()
    del manifest["cells"]
    report = account_cells(manifest, [receipt()])
    assert report["missing_count"] == 1
    assert report["blocked_count"] is None
    assert report["missing_cells"] is None
    with pytest.raises(ValueError, match="explicit original"):
        account_cells(manifest, [receipt()], blocked_cells=[plan()["cells"][1]])


def test_blocked_cells_require_real_failed_prerequisites_in_the_same_lineage():
    manifest = plan()
    forged = [
        dict(cell, reason="seed_prerequisite_not_completed", prerequisite_trial_id="invented")
        for cell in manifest["cells"]
    ]
    with pytest.raises(ValueError, match="actual seed"):
        account_cells(manifest, [], blocked_cells=forged)
    manifest["cells"][1]["cluster"] = "different-family"
    blocked = dict(
        manifest["cells"][1],
        reason="seed_prerequisite_not_completed",
        prerequisite_trial_id="seed-attempt",
    )
    with pytest.raises(ValueError, match="same declared lineage"):
        account_cells(manifest, [receipt()], blocked_cells=[blocked])


def test_declared_identity_and_blocked_overlap_are_checked():
    manifest = plan()
    wrong = receipt().model_copy(update={"task_digest": "changed"})
    with pytest.raises(ValueError, match="task_digest"):
        account_cells(manifest, [wrong])
    with pytest.raises(ValueError, match="already has"):
        account_cells(manifest, [receipt()], blocked_cells=[manifest["cells"][0]])


def test_partial_audit_is_read_only_and_does_not_recurse_into_corpus(tmp_path):
    write_json(tmp_path / "eval_manifest.json", plan())
    raw = receipt().model_dump(mode="json")
    path = tmp_path / "seed-attempt/trial_receipt.json"
    write_json(path, raw)
    write_json(tmp_path / "warm-initial/tasks/hidden/trial_receipt.json", raw)
    interrupted = tmp_path / "eval-attempt/attempt_started.json"
    write_json(interrupted, {"trial_id": "eval-attempt"})
    before = path.read_bytes()
    report = audit_report(tmp_path)
    assert report["accounting"]["terminal_receipts"] == 1
    assert report["accounting"]["missing_count"] == 1
    assert len(report["interrupted_attempts"]) == 1
    assert path.read_bytes() == before
    assert report["receipt_sha256"]["seed-attempt"] == hashlib.sha256(before).hexdigest()


def test_partial_audit_checks_bound_manifest_and_receipt_hashes(tmp_path):
    manifest_path = tmp_path / "eval_manifest.json"
    write_json(manifest_path, plan())
    write_json(tmp_path / "seed-attempt/trial_receipt.json", receipt().model_dump(mode="json"))
    write_json(
        tmp_path / "matrix_accounting.json",
        {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "receipt_sha256": {"seed-attempt": "wrong"},
            "blocked_cells": [],
        },
    )
    with pytest.raises(ValueError, match="receipt hashes changed"):
        audit_report(tmp_path)


def test_extra_unsettled_attempt_prevents_a_clean_complete_report(tmp_path):
    manifest = plan()
    manifest["tasks"] = manifest["tasks"][:1]
    manifest["cells"] = manifest["cells"][:1]
    write_json(tmp_path / "eval_manifest.json", manifest)
    raw = receipt().model_dump(mode="json")
    raw.update(terminal_status="completed_pass", failure_reason=None)
    raw["primary_metrics"].update(reward=1.0, resolved=True)
    write_json(tmp_path / "seed-attempt/trial_receipt.json", raw)
    write_json(tmp_path / "unsettled/attempt_started.json", {"trial_id": "unsettled"})
    report = audit_report(tmp_path)
    assert report["accounting"]["unsettled_attempt_markers"] == 1
    assert report["accounting"]["complete_result_eligible"] is False
    assert report["accounting"]["all_planned_cells_passed"] is False
    assert report["accounting"]["terminal_accounting_complete"] is False


def test_stage_failure_is_recorded_and_preserved_when_evidence_write_also_fails(tmp_path, monkeypatch):
    from mindie_bench.evaluation import timing

    recorder = StageRecorder(tmp_path / "stages.json")
    with pytest.raises(RuntimeError, match="original index failure") as caught:
        with recorder.measure("index"):
            monkeypatch.setattr(
                timing, "write_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
            )
            raise RuntimeError("original index failure")
    assert any("disk full" in note for note in caught.value.__notes__)
    assert recorder.stages[0]["status"] == "failed"


def test_stage_write_failure_cannot_hide_behind_a_previously_handled_exception(tmp_path, monkeypatch):
    from mindie_bench.evaluation import timing

    recorder = StageRecorder(tmp_path / "stages.json")
    monkeypatch.setattr(
        timing, "write_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
    )
    try:
        raise RuntimeError("already handled")
    except RuntimeError:
        with pytest.raises(OSError, match="disk full"):
            with recorder.measure("completed"):
                pass


def test_stage_times_are_separate_observations(tmp_path):
    clock = iter([0.0, 1.0, 3.5, 5.0, 6.0])
    path = tmp_path / "stages.json"
    recorder = StageRecorder(path, clock=lambda: next(clock))
    with recorder.measure("index_due_wait"):
        pass
    with recorder.measure("local_git_publish"):
        pass
    report = read_json(path)
    assert [row["elapsed_sec"] for row in report["stages"]] == [2.5, 1.0]
