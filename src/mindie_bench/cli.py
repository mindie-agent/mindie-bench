"""Small operator interface.

[Role] Dispatch finite benchmark runs, comparisons and release evidence checks.
[Invariants] Missing evidence and unknown effects remain visible to the caller.
[Timeouts & Locks] Execution budgets belong to the selected runner operation.
[Side-Effect Policy] Local reports by default; uploads require explicit CLI options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mindie_bench.evaluation.budget import has_non_budget_errors
from mindie_bench.io import read_json, write_json
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
    validate.add_argument("--tasks", type=Path, action="append", required=True)
    validate.add_argument("--config", type=Path)
    for name, description in (
        ("ablation-evaluate", "evaluate measured C4 conditional ablations and sequential pruning"),
        ("gate-check", "check candidate identity, task correctness and applicable measurement gates"),
    ):
        evidence = commands.add_parser(name, help=description)
        evidence.add_argument("--input", type=Path, required=True)
        evidence.add_argument("--output", type=Path, required=True)
    coverage = commands.add_parser(
        "coverage", help="check whether a study can answer its declared metric questions"
    )
    coverage.add_argument("--tasks", type=Path, action="append", required=True)
    coverage.add_argument("--groups", default="G0,G1,G2,G3")
    coverage.add_argument("--repetitions", type=int, default=3)
    audit = commands.add_parser(
        "audit-report", help="read an incomplete experiment without freezing or replaying it"
    )
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    retrieval = commands.add_parser(
        "component-retrieval",
        help="execute frozen source-bound queries in the selected first-party component runtime",
    )
    retrieval.add_argument("--corpus", type=Path, required=True)
    retrieval.add_argument("--labels", type=Path, required=True)
    retrieval.add_argument("--output", type=Path, required=True)
    retrieval.add_argument("--repetitions", type=int, default=3)
    retrieval.add_argument("--k", type=int, default=3)
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
    run = commands.add_parser(
        "run",
        aliases=["matrix"],
        help="run a finite matrix once; never resume unknown attempts",
    )
    run.add_argument("--tasks", type=Path, action="append", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--tier", choices=["l1", "l2", "l3"], required=True)
    run.add_argument("--groups", default="G0")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument(
        "--canary-first",
        action="store_true",
        help="validate one initial cell before admitting the rest of a multi-cell matrix",
    )
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
    mine = commands.add_parser(
        "mine", help="discover candidates or compile a hashed source-bound executable L2 recipe"
    )
    mine.add_argument("--corpus", type=Path, required=True)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--limit", type=int, default=12)
    mine.add_argument("--recipe", type=Path, help="compile an explicit L2 recipe instead of discovery")
    export = commands.add_parser(
        "export-hf", help="redact, scan, and stage Parquet locally; optionally upload to Hugging Face"
    )
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--knowledge-python", type=Path, required=True)
    export.add_argument("--gitleaks", type=Path, required=True)
    export.add_argument("--tasks-root", "--tasks", dest="tasks_root", type=Path, default=None)
    export.add_argument("--upload-repo", dest="upload_repo", type=str, default=None)
    export.add_argument("--private", dest="private", action="store_true", default=True)
    export.add_argument("--public", dest="private", action="store_false")
    release = commands.add_parser("release-run", help="measure a repository release candidate")
    release.add_argument("--repo", type=Path, required=True)
    release.add_argument("--repository", required=True)
    release.add_argument("--version", required=True)
    release.add_argument("--suite", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--python", default=sys.executable)
    collect = commands.add_parser(
        "release-collect", help="gate product publication on exact component benchmark evidence"
    )
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--input", type=Path, action="append", required=True)
    collect.add_argument("--output", type=Path, required=True)
    value = commands.add_parser(
        "value-collect", help="collect stable product-value tasks and component scenarios"
    )
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    verify_value = commands.add_parser(
        "value-verify", help="recompute a release gate from published value observations"
    )
    verify_value.add_argument("--suite", type=Path, required=True)
    verify_value.add_argument("--evidence", type=Path, required=True)
    verify_value.add_argument("--release-manifest", type=Path)
    verify_value.add_argument("--output", type=Path)
    replay = commands.add_parser(
        "mechanism-replay",
        help="prepare isolated learner prompt views or evaluate M1-01..M6-01 mechanism decisions",
    )
    replay.add_argument("--case", required=True)
    replay.add_argument("--decision", type=Path)
    replay.add_argument("--prompt-only", action="store_true")
    replay.add_argument("--output", type=Path, required=True)
    qual_gate = commands.add_parser(
        "qualification-gate",
        help="execute zero-burden G0..G3 qualification gate checks from a JSON specification",
    )
    qual_gate.add_argument("--gate", choices=["G0", "G1", "G2", "G3"], required=True)
    qual_gate.add_argument("--input", type=Path, required=True)
    qual_gate.add_argument("--output", type=Path, required=True)
    strategy_pilot = commands.add_parser(
        "strategy-selection-pilot",
        help="run the Section 4.7 A/B/C conditional strategy selection continuation pilot",
    )
    strategy_pilot.add_argument("--corpus", type=Path, action="append", required=True)
    strategy_pilot.add_argument("--repetitions", type=int, default=2)
    strategy_pilot.add_argument("--selector-mode", choices=["reference", "claude"], default="reference")
    strategy_pilot.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in {"ablation-evaluate", "gate-check"}:
            from mindie_bench.evaluation.ablation import AblationRequest
            from mindie_bench.evaluation.gates import GateRequest

            request_type = AblationRequest if args.command == "ablation-evaluate" else GateRequest
            result = (
                request_type.model_validate_json(args.input.read_text())
                .evaluate()
                .model_dump(mode="json")
            )
            write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return {"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]]
        elif args.command == "mechanism-replay":
            from mindie_bench.evaluation.mechanism_replay import (
                prepare_learner_prompt_view,
                run_host_mechanism_replay,
            )

            if args.prompt_only or not args.decision:
                result = prepare_learner_prompt_view(args.case)
                write_json(args.output, result)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return 0
            result = run_host_mechanism_replay(
                args.case, decision_payload=read_json(args.decision)
            )
            write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return {"completed": 0, "failed": 1, "unknown": 2}[result["state"]]
        elif args.command == "qualification-gate":
            from mindie_bench.runner.qualification_gate import (
                G0Spec,
                QualificationGateRunner,
            )

            payload = read_json(args.input)
            runner = QualificationGateRunner()
            if args.gate == "G0":
                gate_result = runner.verify_g0_artifact_and_contract(G0Spec(**payload))
            else:
                raise ValueError(f"unsupported direct JSON gate dispatch: {args.gate}")
            result = gate_result.to_dict()
            write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return {"completed": 0, "failed": 1, "unknown": 2}[result["state"]]
        elif args.command == "strategy-selection-pilot":
            from mindie_bench.evaluation.strategy_selection import (
                claude_cli_strategy_selector,
                condition_aware_selector,
                run_strategy_selection_pilot,
            )

            selector = (
                claude_cli_strategy_selector
                if args.selector_mode == "claude"
                else condition_aware_selector
            )
            result = run_strategy_selection_pilot(
                corpus_roots=args.corpus,
                repetitions=args.repetitions,
                selector_fn=selector,
            )
            write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["readiness"]["ready"] else 1
        elif args.command == "value-collect":
            from mindie_bench.value_release import collect

            result = collect(suite=args.suite, manifest=args.manifest, output=args.output)
        elif args.command == "value-verify":
            from mindie_bench.value_release import (
                evaluate,
                verify_publication,
                write_report,
            )

            result = (
                verify_publication(
                    suite=args.suite, evidence=args.evidence, manifest=args.release_manifest
                )
                if args.release_manifest
                else evaluate(read_json(args.suite), read_json(args.evidence))
            )
            if not args.release_manifest:
                from mindie_bench.value_controls import require_collector_source

                result["collector_source"] = require_collector_source(result["runner_release"])
            if args.output:
                write_report(result, args.output)
                if args.release_manifest:
                    from mindie_bench.value_publish import write_component_views

                    write_component_views(
                        result, read_json(args.release_manifest), args.output / "components"
                    )
        elif args.command == "release-run":
            from mindie_bench.release import run_release_suite

            result = run_release_suite(
                repo=args.repo,
                repository=args.repository,
                version=args.version,
                suite=args.suite,
                output=args.output,
                python=args.python,
            )
        elif args.command == "release-collect":
            from mindie_bench.release import collect_release

            result = collect_release(manifest=args.manifest, inputs=args.input, output=args.output)
        elif args.command == "demo":
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
        elif args.command == "audit-report":
            from mindie_bench.evaluation.accounting import audit_report

            result = audit_report(args.input)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "report.json", result, exclusive=True)
        elif args.command == "component-retrieval":
            from mindie_bench.evaluation.retrieval import evaluate_retrieval

            result = evaluate_retrieval(
                args.corpus,
                args.labels,
                args.output,
                repetitions=args.repetitions,
                k=args.k,
            )
        elif args.command in {"run", "matrix", "validate", "coverage"}:
            task_args = args.tasks if isinstance(args.tasks, list) else [args.tasks]
            paths = []
            for task_arg in task_args:
                if (task_arg / "task.toml").is_file():
                    paths.append(task_arg)
                else:
                    paths.extend(sorted(path.parent for path in task_arg.rglob("task.toml")))
            if not paths:
                raise ValueError("no tasks found")
            specs = [TaskSpec.load(path) for path in paths]
            if len({task.task.instance_id for task in specs}) != len(specs):
                raise ValueError("duplicate task instance IDs")
            config_path = getattr(args, "config", None)
            config = ExperimentConfig.load(config_path) if config_path else None
            if args.command == "coverage":
                from mindie_bench.evaluation.coverage import assess_coverage, default_targets

                result = assess_coverage(
                    specs,
                    groups=[Group(value.strip()) for value in args.groups.split(",")],
                    repetitions=args.repetitions,
                    targets=default_targets(),
                ).to_dict()
            elif args.command == "validate":
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
                    canary_first=args.canary_first,
                )
                print(json.dumps(read_json(args.output / "summary.json"), indent=2))
                return (
                    2
                    if any(
                        row.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
                        or has_non_budget_errors(row)
                        for row in rows
                    )
                    else 0
                )
        elif args.command == "score":
            if (args.input / "eval_manifest.json").is_file():
                from mindie_bench.evaluation.accounting import audit_report

                result = audit_report(args.input)["summaries"]
            else:
                result = grouped_summaries(load_receipts(args.input))
        elif args.command == "score-slices":
            from mindie_bench.scorer.slices import score_slices

            result = score_slices(args.input, kind=args.kind, k=args.k)
        elif args.command == "compare":
            plans = [
                read_json(path / "eval_manifest.json").get("cells")
                if (path / "eval_manifest.json").is_file() else None
                for path in (args.baseline, args.candidate)
            ]
            result = comparison(
                load_receipts(args.baseline),
                load_receipts(args.candidate),
                role=args.role,
                baseline_cells=plans[0],
                candidate_cells=plans[1],
            )
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "comparison.json", result, exclusive=True)
            (args.output / "comparison.md").write_text(markdown(result))
        elif args.command == "freeze":
            rows = load_receipts(args.input)
            freeze(rows, args.output)
            result = dict(frozen=len(rows), output=str(args.output))
        elif args.command == "mine":
            from mindie_bench.mining.knowledge_to_task import compile_recipe, mine_knowledge

            if args.recipe:
                result = compile_recipe(
                    args.corpus.resolve(), args.recipe.resolve(), args.output.resolve()
                )
            else:
                result = dict(
                    candidates=len(mine_knowledge(args.corpus, args.output, limit=args.limit)),
                    evidence="candidate",
                    output=str(args.output),
                )
        else:
            from mindie_bench.hf_sync.exporter import export_hf

            result = export_hf(
                args.input,
                args.output,
                knowledge_python=args.knowledge_python,
                gitleaks=args.gitleaks,
                tasks_root=args.tasks_root,
                upload_repo=args.upload_repo,
                private=args.private,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result.get("release_gate") is False else 0
    except Exception as exc:
        print(
            json.dumps(
                dict(
                    status="failed",
                    error=type(exc).__name__,
                    message=str(exc),
                    notes=getattr(exc, "__notes__", []),
                )
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
