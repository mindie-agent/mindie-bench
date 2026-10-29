"""Run bounded component-value tasks and retain their raw evidence.

[Role] One repeatable entry point for allocation, observation and fault-report delivery.
[Invariants] Scenario-specific outcomes are not collapsed into a generic speed score.
[Timeouts & Locks] Child operations are bounded by their scenario deadlines.
[Side-Effect Policy] Uses caller-owned output, loopback HTTP and optional read-only SSH.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import multiprocessing
from pathlib import Path

from mindie_bench.value_controls import runner_source

from .value_common import environment, write_json

BASE_SCENARIOS = ("coordinator", "npu-top", "diagnostics")
OPTIONAL_SCENARIOS = (
    ("profiler", "mindie_profiler", ("analyze", "compare", "memory_profile")),
    ("sim", "mindie_sim", ("prediction.operator_analysis", "prediction.graph_analysis")),
)


def _module_available(module: str, features: tuple[str, ...]) -> bool:
    try:
        return importlib.util.find_spec(module) is not None and all(
            importlib.util.find_spec(f"{module}.{feature}") is not None
            for feature in features
        )
    except (ImportError, ModuleNotFoundError):
        return False


def _optional_scenarios() -> list[str]:
    return [name for name, module, features in OPTIONAL_SCENARIOS
            if _module_available(module, features)]


def scenario_choices() -> list[str]:
    return [*BASE_SCENARIOS, *(name for name, _module, _features in OPTIONAL_SCENARIOS)]


def default_scenarios() -> list[str]:
    """Keep the historical default while including optional components only when installed."""
    return [*BASE_SCENARIOS, *_optional_scenarios()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--npu-fixtures", type=Path)
    parser.add_argument("--live-host")
    parser.add_argument("--container")
    parser.add_argument("--scenarios", nargs="+", choices=scenario_choices(),
                        default=None)
    args = parser.parse_args(argv)
    scenarios = args.scenarios if args.scenarios is not None else default_scenarios()
    if not 1 <= args.repetitions <= 5:
        parser.error("repetitions must be between one and five")
    if any(name in scenarios for name in ("coordinator", "npu-top")) and args.npu_fixtures is None:
        parser.error("--npu-fixtures is required by coordinator and npu-top scenarios")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    initial_source = runner_source()
    cases = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        diagnostics = None
        if "diagnostics" in scenarios:
            from .value_diagnostics import run as diagnostics_run

            diagnostics = executor.submit(diagnostics_run, output, args.repetitions)
        if any(name in scenarios for name in ("coordinator", "npu-top")):
            from .value_npu import run as npu_run

            npu, observed = npu_run(
                output, args.npu_fixtures, args.repetitions, args.live_host, args.container
            )
            if "npu-top" in scenarios:
                cases.append(npu)
            if "coordinator" in scenarios:
                from .value_coordinator import run as coordinator_run

                if observed is None:
                    from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_info

                    observed = parse_npu_smi_info(
                        (args.npu_fixtures / "a2-173-125-1-2.txt").read_text()
                    )
                cases.append(coordinator_run(output, observed, args.repetitions))
        if diagnostics:
            cases.append(diagnostics.result(timeout=120))
        if "profiler" in scenarios:
            from .value_profiler import run as profiler_run

            cases.append(profiler_run(output, args.repetitions))
        if "sim" in scenarios:
            from .value_sim import run as sim_run

            cases.append(sim_run(output, args.repetitions))
    result = {
        "schema_version": "mindie-component-value/1",
        "environment": environment(), "runner_source": initial_source,
        "runner_source_after": runner_source(),
        "cases": cases,
    }
    write_json(output / "component-value-results.json", result)
    print(output / "component-value-results.json")
    return int(any(case["status"] != "pass" for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())
