"""Real local strategy context -> sandbox edit -> independent Python oracle."""

from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.evaluation.strategy_selection import (
    PILOT_CASES,
    StrategyContinuationExecutor,
    StrategyContinuationTask,
    StrategySelectionBudget,
    load_combined_routing_index,
    plan_strategy_selection_cells,
    run_strategy_selection_cell,
    run_strategy_selection_pilot,
)
from mindie_bench.models import SourceEnvironment

CORPORA = Path(__file__).parent / "fixtures" / "strategy-corpora"
ROOTS = [CORPORA / "knowledge-vllm-ascend", CORPORA / "knowledge-operators"]


@pytest.fixture(scope="module")
def source_and_oracle(tmp_path_factory):
    root = tmp_path_factory.mktemp("strategy-continuation")
    source = root / "source"
    source.mkdir()
    (source / "candidate.py").write_text("def compute(value):\n    return value * value\n")
    for arguments in (
        ["init", "-q"], ["add", "candidate.py"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(source), *arguments], check=True, timeout=5)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True, timeout=5,
    ).stdout.strip()
    task = root / "task"
    (task / "verifier").mkdir(parents=True)
    (task / "verifier/verify.py").write_text(
        "import importlib.util, json, os\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('candidate', "
        "Path(os.environ['MINDIE_BENCH_WORKSPACE']) / 'candidate.py')"
        "\nmodule = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "passed = all(module.compute(x) == x ** 2 for x in (-17, -3, 0, 1, 4, 21))\n"
        "print(json.dumps({'reward': float(passed), 'validation_scope': 'cpu_behavior'}))\n"
        "raise SystemExit(0 if passed else 1)\n"
    )
    (task / "verifier/test.sh").write_text(
        f"exec {shlex.quote(sys.executable)} -I -B "
        '"$MINDIE_BENCH_ORACLE_ROOT/verifier/verify.py"\n'
    )
    agent = root / "agent.py"
    agent.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "context = json.load(sys.stdin)\n"
        "assert not {'private_judge', 'reference_solution', 'known_bad_solution'} & context.keys()\n"
        "selected = context['selected_strategy_ref'] != 'default'\n"
        "if selected:\n"
        "    units = context['strategy_context']['units']\n"
        "    assert units and all(unit['content'] for unit in units)\n"
        "    assert context['strategy_context']['requires']\n"
        "    expression = 'value + value' if context['case_id'].endswith('regression') else 'value ** 2'\n"
        "    Path('candidate.py').write_text('def compute(value):\\n    return ' + expression + '\\n')\n"
        "print(json.dumps({'verifier_passed': True, 'model_calls': 999}))\n"
        "if context['case_id'].endswith('crash'):\n"
        "    raise SystemExit(3)\n"
    )
    return task, SourceEnvironment(repo_path=str(source), base_commit=commit), agent


def cases():
    original = PILOT_CASES["S1-F1-trigger"]
    return [
        dataclasses.replace(original, case_id="local-pass"),
        dataclasses.replace(original, case_id="local-regression"),
    ]


def selector(view):
    return {
        "selected_strategy_ref": view["candidate_navigation_cards"][0]["ref"],
        "usage": {"model_calls": 0},
    }


@pytest.mark.tier1
def test_selector_only_preserves_missing_continuation_metrics():
    report = run_strategy_selection_pilot(
        corpus_roots=ROOTS, cases=cases(), repetitions=1, selector_fn=selector,
    )
    assert report["continuation"] == "not_executed"
    for row in report["cells"]:
        assert row["terminal_status"] == "not_executed"
        assert row["continuation_completed"] is None
        assert row["continuation_passed"] is None
        assert row["continuation_regressed"] is None
    for arm in ("B", "C"):
        metrics = report["continuation_metrics"][arm]
        assert metrics["selected_and_completed_rate"]["value"] is None
        assert metrics["regressed_rate"]["value"] is None
        assert metrics["selected_and_completed_rate"]["missing"] == 2


@pytest.mark.tier1
def test_sandbox_continuation_runs_real_oracle_and_pairs_actual_regressions(
    tmp_path, source_and_oracle,
):
    task, source, agent = source_and_oracle
    executor = StrategyContinuationExecutor(
        {case.case_id: StrategyContinuationTask(task, source) for case in cases()},
        command=[sys.executable, "-I", "-B", str(agent)],
        read_paths=[agent], run_root=tmp_path / "runs", verifier_timeout_sec=5,
    )
    report = run_strategy_selection_pilot(
        corpus_roots=ROOTS, cases=cases(), repetitions=1,
        selector_fn=selector, executor_fn=executor,
    )
    assert report["continuation"] == "executed"
    assert report["continuation_verdict_basis"] == "independent_local_verifier"
    assert report["decision_gate"] != "continue_expanding_families"
    workspaces = []
    for row in report["cells"]:
        evidence = row["continuation_evidence"]
        assert evidence, row
        workspaces.append(evidence["workspace"])
        assert row["continuation_completed"] is True, evidence
        expected = row["cell"]["arm"] == "A" or row["cell"]["case_id"] == "local-pass"
        assert row["continuation_passed"] is expected, evidence
        assert row["judgment"]["task_succeeded"] is expected
        assert evidence["verifier_record"]["result"]["reward"] == float(expected)
        assert evidence["verifier_record"]["oracle"]["frozen_before_foreground"] is True
        assert row["executor_model_calls"] is None  # Agent stdout cannot self-certify usage/verdict.
        assert row["prompt_tokens"] is None and row["npu_runs"] is None
        trial = Path(evidence["trial_dir"])
        assert (trial / "verifier.stdout").is_file()
        assert (trial / "final_patch.diff").is_file()
        assert not Path(evidence["workspace"]).exists()
        if row["cell"]["arm"] != "A":
            assert row["continuation_regressed"] is not expected
            context = json.loads((trial / "agent_context.json").read_text())
            assert context["strategy_context"]["units"]
    assert len(workspaces) == len(set(workspaces)) == 6
    assert (Path(source.repo_path) / "candidate.py").read_text() == (
        "def compute(value):\n    return value * value\n"
    )
    for arm in ("B", "C"):
        assert report["arms"][arm]["selected_and_completed_rate"]["value"] == 0.5
        assert report["arms"][arm]["regressed_rate"]["value"] == 0.5
        assert report["arms"][arm]["false_adoption_rate"]["value"] is None
        assert report["arms"][arm]["false_adoption_rate"]["missing"] == 2
    assert report["comparisons"]["C_vs_B"]["selection_reasonable_delta"] is None


@pytest.mark.tier1
@pytest.mark.parametrize("crash", [False, True])
def test_real_verdict_survives_bad_selection_label_and_agent_exit(
    tmp_path, source_and_oracle, crash,
):
    task, source, agent = source_and_oracle
    original = PILOT_CASES["S1-F1-trigger"]
    case = dataclasses.replace(
        original, case_id="local-crash" if crash else "local-pass",
        private_judge={**original.private_judge, "acceptable_strategies": (),
                       "forbidden_strategies": original.candidate_ids},
    )
    executor = StrategyContinuationExecutor(
        {case.case_id: StrategyContinuationTask(task, source)},
        command=[sys.executable, "-I", "-B", str(agent)],
        read_paths=[agent], run_root=tmp_path / "runs", verifier_timeout_sec=5,
    )
    cell = next(
        cell for cell in plan_strategy_selection_cells([case], repetitions=1)[0]
        if cell.arm == "B"
    )
    result = run_strategy_selection_cell(
        cell, case, routing_index=load_combined_routing_index(ROOTS),
        selector_fn=selector, executor_fn=executor,
    )
    assert result["selection"]["false_adoption"] is True
    assert result["continuation_completed"] is (not crash)
    assert result["continuation_passed"] is True
    assert result["terminal_status"] == "pass"
    assert result["judgment"]["task_succeeded"] is True
    assert result["judgment"]["applied_in_actions"] is None
    assert result["judgment"]["false_adoption"] is None
    assert result["continuation_regressed"] is None  # No paired default execution.
    assert result["continuation_evidence"]["agent_returncode"] == (3 if crash else 0)


@pytest.mark.tier1
@pytest.mark.parametrize(
    ("filename", "error_field"),
    [("verifier_record.json", "verifier_record_error"), ("continuation.json", "receipt_error")],
)
def test_real_pass_survives_local_evidence_write_failure(
    tmp_path, source_and_oracle, monkeypatch, filename, error_field,
):
    from mindie_bench.runner import strategy_continuation

    task, source, agent = source_and_oracle
    case = cases()[0]
    original_write = strategy_continuation.write_json

    def write(path, value, **kwargs):
        if path.name == filename:
            raise OSError("disk full after real verifier returned")
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(strategy_continuation, "write_json", write)
    executor = StrategyContinuationExecutor(
        {case.case_id: StrategyContinuationTask(task, source)},
        command=[sys.executable, "-I", "-B", str(agent)],
        read_paths=[agent], run_root=tmp_path / "runs", verifier_timeout_sec=5,
    )
    result = executor(
        case=case, arm="A", selector_decision=None,
        routing_index=load_combined_routing_index(ROOTS),
        remaining_model_calls=1, remaining_window_sec=5,
    )
    assert result.passed is True and result.completed is True
    assert result.terminal_status == "pass"
    assert "disk full" in result.evidence[error_field]
    assert result.evidence["verifier_record"]["result"]["reward"] == 1


@pytest.mark.tier1
def test_verifier_time_uses_its_separate_allowance(tmp_path, source_and_oracle):
    task, source, agent = source_and_oracle
    case = cases()[0]
    executor = StrategyContinuationExecutor(
        {case.case_id: StrategyContinuationTask(task, source)},
        command=[sys.executable, "-I", "-B", str(agent)],
        read_paths=[agent], run_root=tmp_path / "runs", verifier_timeout_sec=5,
    )
    budget = StrategySelectionBudget(
        business_window_sec=10, selector_timeout_sec=1,
        verifier_timeout_sec=5, cleanup_grace_sec=5, outer_timeout_sec=20,
    )
    cell = next(cell for cell in plan_strategy_selection_cells([case], repetitions=1)[0] if cell.arm == "A")
    ticks = iter([0.0, 12.0])
    result = run_strategy_selection_cell(
        cell, case, routing_index=load_combined_routing_index(ROOTS),
        executor_fn=executor, budget=budget, clock=lambda: next(ticks),
    )
    assert result["continuation_passed"] is True
    assert result["wall_clock_sec"] == 12.0
    assert result["budget_compliant"] is None  # Unknown model spend, no observed time overrun.
    evidence = result["continuation_evidence"]
    assert evidence["verifier_timeout_sec"] <= 5
    assert evidence["verifier_drain_grace_sec"] == 5
    assert evidence["outer_budget_exceeded"] is False
