"""Run bounded component-value tasks and retain their raw evidence.

[Role] One repeatable entry point for allocation, observation and fault-report delivery.
[Invariants] Scenario-specific outcomes are not collapsed into a generic speed score.
[Timeouts & Locks] Child operations are bounded by their scenario deadlines.
[Side-Effect Policy] Uses caller-owned output, loopback HTTP and optional read-only SSH.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path

from .value_common import environment, git_source, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--npu-fixtures", type=Path, required=True)
    parser.add_argument("--live-host")
    parser.add_argument("--container")
    parser.add_argument("--scenarios", nargs="+", choices=["coordinator", "npu-top", "diagnostics"],
                        default=["coordinator", "npu-top", "diagnostics"])
    args = parser.parse_args(argv)
    if not 1 <= args.repetitions <= 5:
        parser.error("repetitions must be between one and five")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    source_root = Path(__file__).resolve().parents[3]
    initial_source = git_source(source_root)
    cases = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        diagnostics = None
        if "diagnostics" in args.scenarios:
            from .value_diagnostics import run as diagnostics_run

            diagnostics = executor.submit(diagnostics_run, output, args.repetitions)
        if any(name in args.scenarios for name in ("coordinator", "npu-top")):
            from .value_npu import run as npu_run

            npu, observed = npu_run(
                output, args.npu_fixtures, args.repetitions, args.live_host, args.container
            )
            if "npu-top" in args.scenarios:
                cases.append(npu)
            if "coordinator" in args.scenarios:
                from .value_coordinator import run as coordinator_run

                if observed is None:
                    from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_info

                    observed = parse_npu_smi_info(
                        (args.npu_fixtures / "a2-173-125-1-2.txt").read_text()
                    )
                cases.append(coordinator_run(output, observed, args.repetitions))
        if diagnostics:
            cases.append(diagnostics.result(timeout=120))
    result = {
        "schema_version": "mindie-component-value/1",
        "environment": environment(), "runner_source": initial_source,
        "runner_source_after": git_source(source_root),
        "cases": cases,
    }
    write_json(output / "component-value-results.json", result)
    print(output / "component-value-results.json")
    return int(any(case["status"] != "pass" for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())
