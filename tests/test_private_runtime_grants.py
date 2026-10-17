from mindie_bench.models import Group
from mindie_bench.runner.orchestrator import run_trial, runtime_paths
from mindie_bench.runner.profile_builder import build_profile


def test_knowledge_only_runtime_is_not_a_foreground_read_capability(case):
    backend = case["root"] / "backend-library"
    backend.mkdir()
    case["config"].knowledge_runtime_read_paths = [str(backend)]
    trial = case["root"] / "capabilities"
    trial.mkdir()
    profile = build_profile(trial, case["spec"], Group.G0, case["config"])
    assert backend not in runtime_paths(case["config"], profile)


def test_foreground_cannot_regrant_a_knowledge_only_library(case):
    backend = case["root"] / "backend-library"
    backend.mkdir()
    case["config"].knowledge_runtime_read_paths = [str(backend)]
    case["config"].runtime_read_paths = [str(backend)]
    output = case["root"] / "attempt"
    output.mkdir()
    row = run_trial(
        case["task"], case["config"], tier="l2", group=Group.G0, repetition=1, run_root=output
    )
    assert row.terminal_status.value == "infra_failed"
    assert "runtime read grant exposes protected" in row.failure_reason
    assert not list(output.glob("*/native_events.jsonl"))
