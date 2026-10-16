import json
import shutil
import threading
from pathlib import Path

import pytest

from mindie_bench.io import write_json
from mindie_bench.models import CorpusConfig, Group, RemoteConfig, TaskSpec, TerminalStatus, TrialReceipt
from mindie_bench.runner import orchestrator
from mindie_bench.scorer.comparator import load_receipts


def make_task(case, identity, role, *, cluster="family", corpus_id="fixture", tier="l2"):
    root = case["root"] / identity
    shutil.copytree(case["task"], root)
    manifest = root / "task.toml"
    text = manifest.read_text().replace('instance_id = "synthetic-dtype"', f'instance_id = "{identity}"')
    text = text.replace(
        'evidence = "synthetic"', f'evidence = "synthetic"\nrole = "{role}"\ncluster = "{cluster}"'
    )
    text = text.replace('corpus_id = "fixture"', f'corpus_id = "{corpus_id}"')
    if tier == "l3":
        text = text.replace("[environment.l2]", "[environment.l3]")
        text += '\nnpu_count = 1\nnpu_model = "Ascend910"\n'
    manifest.write_text(text)
    return root


def configure(case, corpus, *, tier="l2"):
    source, commit = corpus
    config = case["config"].model_copy(deep=True)
    config.corpora["fixture"] = CorpusConfig(path=str(source), commit=commit)
    config.corpora["other"] = CorpusConfig(path=str(source), commit=commit)
    if tier == "l3":
        config.remote = RemoteConfig(
            host="bench-host",
            container="bench",
            repo_path="/repo",
            devices=[8, 9],
            probe_command=["/probe-json"],
        )
    return config


def completed_trial(path, arguments, *, status=TerminalStatus.PASS, capture="completed"):
    spec = TaskSpec.load(path)
    identity = spec.task.instance_id
    trial_id = f"{identity}-{arguments['group'].value}-{arguments['repetition']}"
    directory = arguments["run_root"] / trial_id
    directory.mkdir()
    view = directory / "profile/corpus"
    previous = arguments.get("corpus_override")
    if previous:
        shutil.copytree(previous, view)
    else:
        view.mkdir(parents=True)
    (view / f"captured-{identity}-{arguments['repetition']}").write_text("captured\n")
    outcome = (
        dict(reward=1.0, resolved=True)
        if status == TerminalStatus.PASS
        else dict(reward=0.0, resolved=False)
        if status == TerminalStatus.FAIL
        else dict(reward=None, resolved=None)
    )
    receipt = TrialReceipt(
        trial_id=trial_id,
        instance_id=identity,
        repetition=arguments["repetition"],
        tier=arguments["tier"],
        ablation_group=arguments["group"],
        knowledge_gradient=spec.mindie_knowledge.knowledge_gradient,
        task_role=spec.task.role,
        cluster=spec.task.cluster,
        task_digest="synthetic-task",
        experiment_digest="synthetic-experiment",
        environment_fingerprint={},
        terminal_status=status,
        failure_reason=None if status == TerminalStatus.PASS else "uncertain capture effects",
        primary_metrics=outcome,
        auxiliary_metrics={"capture": {"status": capture}},
        artifacts={"private_trial_dir": str(directory)},
        evidence="synthetic",
    )
    write_json(directory / "trial_receipt.json", receipt.model_dump(mode="json"), exclusive=True)
    return receipt


def test_ordinary_cells_overlap_g3_and_eval_waits_for_capture(case, corpus, monkeypatch):
    seed = make_task(case, "seed", "seed")
    evaluation = make_task(case, "eval", "eval")
    rendezvous = threading.Barrier(2)
    captured = threading.Event()
    seed_view = None

    def run_trial(path, config, **arguments):
        nonlocal seed_view
        if path == seed:
            rendezvous.wait(timeout=5)
        if arguments["group"] == Group.G3 and path == evaluation:
            assert captured.is_set()
            assert arguments["corpus_override"] == seed_view
            assert (seed_view / "captured-seed-1").is_file()
        receipt = completed_trial(path, arguments)
        if arguments["group"] == Group.G3 and path == seed:
            seed_view = Path(receipt.artifacts["private_trial_dir"]) / "profile/corpus"
            captured.set()
        return receipt

    monkeypatch.setattr(orchestrator, "run_trial", run_trial)
    output = case["root"] / "overlap"
    rows = orchestrator.run_matrix(
        [seed, evaluation],
        configure(case, corpus),
        tier="l2",
        groups=[Group.G0, Group.G3],
        repetitions=1,
        concurrency=2,
        output=output,
    )
    assert len(rows) == 4
    assert len(load_receipts(output)) == 4
    assert json.loads((output / "summary.json").read_text())["l2/synthetic/G3"]["attempts"] == 1


@pytest.mark.parametrize("independent_axis", ["repetition", "cluster", "corpus"])
def test_g3_lineages_run_concurrently_without_sharing_warm_views(
    case, corpus, monkeypatch, independent_axis
):
    seed = make_task(case, "seed", "seed")
    evaluation = make_task(case, "eval", "eval")
    roots = [seed, evaluation]
    repetitions = 2
    if independent_axis != "repetition":
        second = (
            dict(cluster="other-family") if independent_axis == "cluster" else dict(corpus_id="other")
        )
        roots += [
            make_task(case, "seed-other", "seed", **second),
            make_task(case, "eval-other", "eval", **second),
        ]
        repetitions = 1
    seeds = {}
    for path in roots:
        spec = TaskSpec.load(path)
        if spec.task.role == "seed":
            seeds[(spec.task.cluster, spec.mindie_knowledge.corpus_id)] = spec.task.instance_id
    rendezvous = threading.Barrier(2)
    lock = threading.Lock()
    initial_views = []

    def run_trial(path, config, **arguments):
        spec = TaskSpec.load(path)
        view = arguments["corpus_override"]
        if spec.task.role == "seed":
            assert not list(view.glob("captured-*"))
            assert not (view / "tasks/origin").exists()
            with lock:
                initial_views.append(view)
            rendezvous.wait(timeout=5)
        else:
            identity = seeds[(spec.task.cluster, spec.mindie_knowledge.corpus_id)]
            assert {item.name for item in view.glob("captured-*")} == {
                f"captured-{identity}-{arguments['repetition']}"
            }
        return completed_trial(path, arguments)

    monkeypatch.setattr(orchestrator, "run_trial", run_trial)
    rows = orchestrator.run_matrix(
        roots,
        configure(case, corpus),
        tier="l2",
        groups=[Group.G3],
        repetitions=repetitions,
        concurrency=2,
        output=case["root"] / "independent",
    )
    assert len(rows) == 4
    assert len(set(initial_views)) == 2


def test_l3_cells_and_g3_share_device_allocator_and_cancellation(case, corpus, monkeypatch):
    seed = make_task(case, "seed", "seed", tier="l3")
    evaluation = make_task(case, "eval", "eval", tier="l3")
    rendezvous = threading.Barrier(2)
    lock = threading.Lock()
    active = set()
    observed_devices = set()
    cancellation_events = set()

    def run_trial(path, config, **arguments):
        devices = arguments["devices"]
        assert len(devices) == 1
        with lock:
            assert active.isdisjoint(devices)
            active.update(devices)
            observed_devices.update(devices)
            cancellation_events.add(arguments["cancelled"])
        try:
            if path == seed:
                rendezvous.wait(timeout=5)
            return completed_trial(path, arguments)
        finally:
            with lock:
                active.difference_update(devices)

    monkeypatch.setattr(orchestrator, "run_trial", run_trial)
    rows = orchestrator.run_matrix(
        [seed, evaluation],
        configure(case, corpus, tier="l3"),
        tier="l3",
        groups=[Group.G0, Group.G3],
        repetitions=1,
        concurrency=3,
        output=case["root"] / "devices",
    )
    assert len(rows) == 4
    assert observed_devices == {8, 9}
    assert len(cancellation_events) == 1
    assert None not in cancellation_events
    assert not active


@pytest.mark.parametrize(
    "status,capture",
    [
        (TerminalStatus.UNKNOWN, "unknown"),
        (TerminalStatus.PASS, "failed"),
        (TerminalStatus.INFRA, "completed"),
    ],
)
def test_g3_failed_capture_preserves_receipt_and_partial_summary_without_retry(
    case, corpus, monkeypatch, status, capture
):
    seed = make_task(case, "seed", "seed")
    evaluation = make_task(case, "eval", "eval")
    calls = []

    def run_trial(path, config, **arguments):
        calls.append(path)
        assert path == seed
        return completed_trial(path, arguments, status=status, capture=capture)

    monkeypatch.setattr(orchestrator, "run_trial", run_trial)
    output = case["root"] / "failed-capture"
    with pytest.raises(RuntimeError, match="G3 seed prerequisites failed") as failed:
        orchestrator.run_matrix(
            [seed, evaluation],
            configure(case, corpus),
            tier="l2",
            groups=[Group.G3],
            repetitions=1,
            concurrency=2,
            output=output,
        )
    assert f"status={status.value}, capture={capture}" in str(failed.value)
    assert calls == [seed]
    receipt_path = output / "seed-G3-1/trial_receipt.json"
    assert json.loads(receipt_path.read_text())["terminal_status"] == status.value
    summary = json.loads((output / "summary.json").read_text())
    assert summary["l2/synthetic/G3/seed"]["statuses"][status.value] == 1
    with pytest.raises(ValueError, match="incomplete experiment matrix"):
        load_receipts(output)


def test_g3_requires_a_seed_in_every_lineage_before_starting(case, corpus, monkeypatch):
    seed = make_task(case, "seed", "seed")
    evaluation = make_task(case, "eval", "eval", cluster="unseeded")

    def unexpected_trial(*args, **kwargs):
        pytest.fail("an invalid G3 lineage started a trial")

    monkeypatch.setattr(orchestrator, "run_trial", unexpected_trial)
    with pytest.raises(ValueError, match="seed tasks in each lineage"):
        orchestrator.run_matrix(
            [seed, evaluation],
            configure(case, corpus),
            tier="l2",
            groups=[Group.G3],
            repetitions=1,
            concurrency=2,
            output=case["root"] / "invalid-lineage",
        )


def test_failed_experience_is_captured_without_eval_to_eval_leakage(case, corpus, monkeypatch):
    seed = make_task(case, "seed", "seed")
    first = make_task(case, "eval-one", "eval")
    second = make_task(case, "eval-two", "eval")
    statuses = [TerminalStatus.FAIL, TerminalStatus.FAIL, TerminalStatus.PASS]
    calls = []

    def run_trial(path, config, **arguments):
        calls.append(path)
        if path != seed:
            view = arguments["corpus_override"]
            assert {p.name for p in view.glob("captured-*")} == {"captured-seed-1"}
        return completed_trial(path, arguments, status=statuses[len(calls) - 1])

    monkeypatch.setattr(orchestrator, "run_trial", run_trial)
    output = case["root"] / "failed-engineering"
    rows = orchestrator.run_matrix(
        [seed, first, second],
        configure(case, corpus),
        tier="l2",
        groups=[Group.G3],
        repetitions=1,
        concurrency=3,
        output=output,
    )
    assert calls == [seed, first, second]
    assert [r.terminal_status for r in rows] == statuses
    assert len(load_receipts(output)) == 3


def test_failed_lineage_does_not_cancel_independent_admitted_work(case, corpus, monkeypatch):
    seed = make_task(case, "seed", "seed")
    handled = threading.Event()

    def run_trial(path, config, **arguments):
        if arguments["group"] == Group.G0:
            assert handled.wait(5)
            assert not arguments["cancelled"].is_set()
            return completed_trial(path, arguments)
        return completed_trial(path, arguments, capture="failed")

    def lineage_first(futures):
        # The shared queue is explicitly interleaved G0 then G3.
        assert len(futures) == 2
        try:
            futures[1].result(timeout=5)
        except RuntimeError:
            pass
        yield futures[1]
        handled.set()
        yield futures[0]

    monkeypatch.setattr(orchestrator, "run_trial", run_trial)
    monkeypatch.setattr(orchestrator, "as_completed", lineage_first)
    output = case["root"] / "independent-failure"
    with pytest.raises(RuntimeError, match="G3 seed prerequisites failed"):
        orchestrator.run_matrix(
            [seed],
            configure(case, corpus),
            tier="l2",
            groups=[Group.G0, Group.G3],
            repetitions=1,
            concurrency=2,
            output=output,
        )
    ordinary = json.loads((output / "seed-G0-1/trial_receipt.json").read_text())
    assert ordinary["terminal_status"] == "completed_pass"
