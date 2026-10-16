"""Small operator interface; no daemon, activation hook, or publication side effect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mindie_bench.io import write_json
from mindie_bench.models import ExperimentConfig, Group, TaskSpec, TerminalStatus
from mindie_bench.scorer.comparator import (
    comparison,
    freeze,
    grouped_summaries,
    load_receipts,
    markdown,
)


def parser():
    root = argparse.ArgumentParser(prog="mindie-bench")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="validate task and experiment contracts without a model call"
    )
    validate.add_argument("--tasks", type=Path, required=True)
    validate.add_argument("--config", type=Path)
    soundness = commands.add_parser("soundness", help="prove base FAIL and gold PASS without a model")
    soundness.add_argument("--task", type=Path, required=True)
    soundness.add_argument("--config", type=Path, required=True)
    soundness.add_argument("--tier", choices=["l2", "l3"], required=True)
    soundness.add_argument("--output", type=Path, required=True)
    soundness.add_argument("--promote", action="store_true")
    demo = commands.add_parser(
        "demo", help="run a fully synthetic end-to-end example with zero model calls"
    )
    demo.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="run a finite matrix once; never resume unknown attempts")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--tier", choices=["l1", "l2", "l3"], required=True)
    run.add_argument("--groups", default="G0")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    slices = commands.add_parser("score-slices", help="score observed routing or retrieval JSONL labels")
    slices.add_argument("--input", type=Path, required=True)
    slices.add_argument("--kind", choices=["routing", "retrieval"], required=True)
    slices.add_argument("--k", type=int, default=3)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--role", choices=["seed", "eval"], default="eval")
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--input", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    mine = commands.add_parser("mine", help="generate unverified source-grounded candidates")
    mine.add_argument("--corpus", type=Path, required=True)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--limit", type=int, default=12)
    export = commands.add_parser(
        "export-hf", help="redact, scan, and stage Parquet locally; never upload"
    )
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--knowledge-python", type=Path, required=True)
    export.add_argument("--gitleaks", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            from mindie_bench.synthetic import demo

            result = demo(args.output.resolve())
        elif args.command == "soundness":
            from mindie_bench.runner.soundness import check_soundness

            result = check_soundness(
                args.task.resolve(),
                ExperimentConfig.load(args.config),
                tier=args.tier,
                output=args.output.resolve(),
                promote=args.promote,
            )
        elif args.command in {"run", "validate"}:
            paths = (
                [args.tasks]
                if (args.tasks / "task.toml").is_file()
                else sorted(path.parent for path in args.tasks.rglob("task.toml"))
            )
            if not paths:
                raise ValueError("no tasks found")
            specs = [TaskSpec.load(path) for path in paths]
            if len({task.task.instance_id for task in specs}) != len(specs):
                raise ValueError("duplicate task instance IDs")
            config = ExperimentConfig.load(args.config) if args.config else None
            if args.command == "validate":
                result = dict(
                    validated_tasks=len(specs), experiment_validated=bool(config), model_calls=0
                )
            else:
                from mindie_bench.runner.orchestrator import run_matrix

                groups = [Group(value.strip()) for value in args.groups.split(",")]
                rows = run_matrix(
                    paths,
                    config,
                    tier=args.tier,
                    groups=groups,
                    repetitions=args.repetitions,
                    concurrency=args.concurrency,
                    output=args.output.resolve(),
                )
                print(json.dumps(grouped_summaries(rows), indent=2))
                return (
                    2
                    if any(
                        row.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
                        or row.errors
                        for row in rows
                    )
                    else 0
                )
        elif args.command == "score":
            result = grouped_summaries(load_receipts(args.input))
        elif args.command == "score-slices":
            from mindie_bench.scorer.slices import score_slices

            result = score_slices(args.input, kind=args.kind, k=args.k)
        elif args.command == "compare":
            result = comparison(
                [row for row in load_receipts(args.baseline) if row.task_role == args.role],
                [row for row in load_receipts(args.candidate) if row.task_role == args.role],
            )
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "comparison.json", result, exclusive=True)
            (args.output / "comparison.md").write_text(markdown(result))
        elif args.command == "freeze":
            rows = load_receipts(args.input)
            freeze(rows, args.output)
            result = dict(frozen=len(rows), output=str(args.output))
        elif args.command == "mine":
            from mindie_bench.mining.knowledge_to_task import mine_knowledge

            result = dict(
                candidates=len(mine_knowledge(args.corpus, args.output, limit=args.limit)),
                evidence="candidate",
                output=str(args.output),
            )
        else:
            from mindie_bench.hf_sync.exporter import export_hf

            result = export_hf(
                args.input, args.output, knowledge_python=args.knowledge_python, gitleaks=args.gitleaks
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(dict(status="failed", error=type(exc).__name__, message=str(exc))),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
