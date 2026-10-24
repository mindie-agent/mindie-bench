"""Prepare component-specific value attachments from a verified product report.

[Role] Keep every component release tied to its measured product combination.
[Invariants] Joint ablations are not attributed to a single component. Failed
attempts and recovery costs remain visible in every selected study.
[Timeouts & Locks] Local serialization only; no locks or network requests.
[Side-Effect Policy] Writes public JSON and Markdown under the chosen output.
"""

from __future__ import annotations

from pathlib import Path

from mindie_bench.io import write_json
from mindie_bench.value_release import markdown
from mindie_bench.value_resources import setup_totals


def component_name(repository: str) -> str:
    name = repository.split("/")[-1].removeprefix("mindie-agent-")
    return {"cc": "claude-code"}.get(name, name)


def component_view(report: dict, target: dict) -> dict:
    if report.get("schema_version") == "mindie-value-report/2":
        from mindie_bench.value_product_views import component_view_product

        return component_view_product(report, target)
    component = component_name(target["repository"])
    if report["component_revisions"].get(component) != target["revision"]:
        raise ValueError(f"component value attachment source differs: {component}")
    if component in {"codex", "claude-code", "kimi", "dsh"}:
        studies = [row for row in report["studies"] if row["harness"] == component]
        scope = (
            "Native host tasks with controlled MCP and shipped Skills. "
            "The task comparisons measure the configured MindIE combination; "
            "full plugin loading is verified by the separate native checks."
        )
    elif component in {"knowledge", "remote-dev"}:
        studies = report["studies"]
        scope = (
            "G0/G1 compares the addition of remote-dev; G1/G2 compares the addition "
            "of knowledge retrieval, shipped Skills and explicit retrieval guidance together. "
            "G2/G3 measures capture and later reuse, including "
            "the seed task. G0/G2 measures their joint effect and cannot attribute "
            "a gain to one component."
        )
    elif component == "mindie-bench":
        studies = report["studies"]
        scope = (
            "Benchmark execution and measurement coverage across the product suite. "
            "These task results measure the evaluated product, not productivity "
            "caused by the measurement tool itself."
        )
    else:
        studies = []
        scope = "Repeated component scenarios; no model-productivity claim."
    study_ids = {row["id"] for row in studies}
    harnesses = {row["harness"] for row in studies}
    setup_reports = [
        row for row in report.get("setup_reports", []) if row["plugin"] in harnesses
    ]
    cases = [
        row for row in report["component_cases"]
        if component == "mindie-bench" or row["component"] == component
    ]
    plugins = [row for row in report["native_plugins"] if row["plugin"] == component]
    view = {
        **report,
        "schema_version": "mindie-component-value-report/1",
        "repository": target["repository"],
        "component": component,
        "component_version": target["version"],
        "component_revision": target["revision"],
        "benchmark_scope": scope,
        "studies": studies,
        "setup_reports": setup_reports,
        "setup_resources": setup_totals(setup_reports),
        "component_cases": cases,
        "excluded_attempts": [
            row for row in report["excluded_attempts"] if row["study_id"] in study_ids
        ],
        "native_plugins": plugins,
        # An attachment states the scope of the evidence it carries, not the whole suite's.
        "gate_scope": {
            "agent_studies": len(studies),
            "selected_agent_trials": sum(study["observed_cells"] for study in studies),
            "component_cases": len(cases),
            "native_plugins": len(plugins),
        },
        "limitations": [scope, *report["limitations"]],
    }
    # Product-wide resource totals do not describe an individual component.
    view.pop("resources", None)
    return view


def write_component_views(report: dict, targets: dict, output: Path) -> list[Path]:
    written = []
    for target in targets["repositories"]:
        view = component_view(report, target)
        directory = output / component_name(target["repository"])
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"product-{report['product_version']}-value-results"
        data, prose = directory / f"{stem}.json", directory / f"{stem}.md"
        write_json(data, view)
        if report.get("schema_version") == "mindie-value-report/2":
            from mindie_bench.value_product_views import component_markdown_product

            prose.write_text(component_markdown_product(view), encoding="utf-8")
        else:
            prose.write_text(
                f"# {target['repository']} {target['version']}\n\n"
                + markdown(view).removeprefix("# Product value benchmark\n"),
                encoding="utf-8",
            )
        written.extend((data, prose))
    return written
