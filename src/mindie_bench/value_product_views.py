"""Public rendering and component views for product-matrix reports.

[Role] Derive Markdown, homepage blocks and component attachments only from the
measured product report.
[Invariants] P0/P1/P2/P3 labels are never translated into legacy G labels.
Failed/infra/unknown/missing counts stay visible; incomplete token/cost totals
render as lower bounds or "not measured", never zero. Native Codex/CC/Kimi
evidence is compatibility-only; knowledge/remote/core views are combination
context, never causal single-component gains.
[Timeouts & Locks] Pure in-memory formatting; no waits, subprocesses or locks.
[Side-Effect Policy] Returns structures and Markdown only; no I/O here.
"""

from __future__ import annotations

from mindie_bench.value_homepage import BEGIN_MARKER, END_MARKER


def _text(value) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _number(value, *, digits: int = 1) -> str:
    if value is None:
        return "not measured"
    return f"{value:.{digits}f}"


def _status(report: dict) -> str:
    return "complete" if report["release_gate"] else "incomplete"


def _budget_overrun_counts(comparison: dict) -> str:
    budgets = comparison.get("budget_observations", {})
    return "/".join(
        _number(budgets.get(side, {}).get("exceeded_attempts"), digits=0)
        for side in ("baseline", "candidate")
    )


def _group_row(group: dict) -> str:
    denominator = group["denominator"]
    tokens = group["tokens_total"]
    cost = group["cost_usd_total"]
    return (
        f"| {group['group']} | {_text(group['profile_id'])} | "
        f"{denominator['planned']} | {denominator['observed']} | {denominator['missing']} | "
        f"{denominator['pass']} | {denominator['fail']} | {denominator['infra']} | "
        f"{denominator['unknown']} | {tokens['complete']} | {cost['complete']} |"
    )


def _comparison_row(comparison: dict) -> str:
    denominator = comparison.get("denominator", {})
    return (
        f"| {_text(comparison['id'])} | {_text(comparison.get('cohort', ''))} | "
        f"{comparison['baseline_group']} -> "
        f"{comparison['candidate_group']} | {denominator['observed']}/{denominator['planned']} | "
        f"{denominator['valid']} | {denominator['missing']} | {denominator['unsettled']} | "
        f"{_number(comparison['wall_clock_speedup_on_mutual_success'], digits=3)} | "
        f"{comparison['measurement_status']} | {comparison['claim_ready']} |"
    )


def _component_case_lines(cases: list[dict]) -> list[str]:
    if not cases:
        return []
    lines = ["", "## Component functional scenarios", "",
             "Repeated component scenarios are functional evidence inside the measured combination; "
             "they are not causal single-component productivity gains.", "",
             "| Scenario | Component | Repetitions | Status |", "| --- | --- | ---: | --- |"]
    for case in cases:
        lines.append(
            f"| {_text(case['id'])} | {_text(case['component'])} | "
            f"{len(case['measurements'])} | {case['status']} |"
        )
    return lines


def _native_lines(plugins: list[dict]) -> list[str]:
    if not plugins:
        return []
    lines = ["", "## Native compatibility evidence", "",
             "Native Codex, Claude Code and Kimi checks are compatibility evidence only. "
             "The matched product comparison is carried by DSH.", "",
             "| Plugin | Revision | Status | Checks |", "| --- | --- | --- | ---: |"]
    for plugin in plugins:
        passed = sum(check["status"] == "pass" for check in plugin["checks"])
        lines.append(
            f"| {_text(plugin['plugin'])} | `{plugin['revision']}` | {plugin['status']} | "
            f"{passed}/{len(plugin['checks'])} |"
        )
    return lines


def _runner_lines(reports: list[dict]) -> list[str]:
    if not reports:
        return []
    lines = ["", "## Accepted component runner identities", "",
             "Component and native compatibility reports retain the runner they actually used. "
             "They are not relabelled with the new common product benchmark commit.", ""]
    for report in reports:
        source = report.get("runner_source") or {}
        after = report.get("runner_source_after") or {}
        lines.append(
            f"- Cases `{', '.join(report['case_ids'])}`: accepted runner "
            f"`{source.get('revision') or source.get('source_sha256') or 'unrecorded'}` "
            f"(after: `{after.get('revision') or after.get('source_sha256') or 'unrecorded'}`)."
        )
    return lines


def markdown_product(report: dict) -> str:
    lines = [
        "# Product value benchmark",
        "",
        f"Version: {report['product_version']}. Suite: `{report['suite']}`. "
        f"Evidence: {_status(report)}.",
        "",
        "Product arms are **P0** bare host, **P1** formal v0.0.2 baseline, **P2** final "
        "v0.0.3 release and **P3** conditional flywheel. They are never translated into "
        "legacy G0..G3 component-increment labels. A receipt's internal G capability mode "
        "is an execution setting, not its product identity.",
        "",
        f"Common product benchmark commit: `{report['benchmark']['revision']}` "
        f"(`{report['benchmark']['version']}`, source `{report['benchmark']['source_sha256']}`). "
        "Every new P0/P1/P2/P3 arm uses this common runner. Component compatibility reports "
        "keep their own accepted runner identities.",
        "",
        f"P1 formal baseline: `{report['formal_baseline']['product_version']}` at "
        f"`{report['formal_baseline']['revision']}` (artifact "
        f"`{report['formal_baseline']['artifact_sha256']}`).",
        f"P2 final release: `{report['final_release']['product_version']}` at "
        f"`{report['final_release']['revision']}` (artifact "
        f"`{report['final_release']['artifact_sha256']}`), binding "
        f"{len(report['final_release']['repositories'])} repository targets.",
        "",
        "| Cohort | Product arm | Profile | Planned | Observed | Missing | Pass | Fail | Infra | Unknown | "
        "Tokens complete | Cost complete |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cohort_id, cohort in report["cohorts"].items():
        for group in cohort["matrix"]["groups"].values():
            denominator = group["denominator"]
            tokens = group["tokens_total"]
            cost = group["cost_usd_total"]
            lines.append(
                f"| {_text(cohort_id)} | {group['group']} | {_text(group['profile_id'])} | "
                f"{denominator['planned']} | {denominator['observed']} | {denominator['missing']} | "
                f"{denominator['pass']} | {denominator['fail']} | {denominator['infra']} | "
                f"{denominator['unknown']} | {tokens['complete']} | {cost['complete']} |"
            )
    lines += [
        "",
        "## Matched comparisons",
        "",
        "Delivery time includes preparation, verifier and cleanup. Speedup uses only mutually "
        "solved pairs; failed, infrastructure, unknown and missing cells remain in their denominators.",
        "",
        "| Comparison | Cohort | Arms | Observed / planned | Valid pairs | Missing | Unsettled | "
        "Delivery ratio (baseline/candidate) | Status | Claim ready |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines += [_comparison_row(comparison) for comparison in report["comparisons"]]
    lines += [
        "", "## Budget observations", "",
        "Verified completions and delivery ratios include over-budget results. Matched allocations "
        "do not imply actual resource compliance; unmeasured resources cannot establish compliance.", "",
        "| Comparison | Arm | Fully measured / observed | Over-budget attempts | Successful overruns |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for comparison in report["comparisons"]:
        for side, budget in comparison.get("budget_observations", {}).items():
            lines.append(
                f"| {_text(comparison['id'])} | {comparison[f'{side}_group']} | "
                f"{budget['fully_measured_attempts']}/{budget['observed_attempts']} | "
                f"{budget['exceeded_attempts']} | {budget['completed_pass_over_budget']} |"
            )
    resources = report["resources"]
    lines += [
        "",
        "## Resource accounting",
        "",
        f"Selected attempts: {report['attempts']['selected']}; retained historical attempts: "
        f"{report['attempts']['historical']}; all executed: {report['attempts']['all_executed']}. "
        "Historical attempts keep their full measured costs and never replace a valid outcome.",
        "",
        f"- Delivery lower bound: {_number(resources['observed_delivery_wall_clock_sec_lower_bound'], digits=2)} s "
        f"across {resources['delivery_observed_attempts']}/{resources['attempts']} attempts "
        f"(complete={resources['delivery_timing_complete']}).",
        f"- Cost lower bound: {_number(resources['observed_cost_usd_lower_bound'], digits=4)} USD "
        f"across {resources['priced_attempts']}/{resources['attempts']} attempts "
        f"(complete={resources['cost_complete']}).",
        f"- Token lower bound: `{resources['observed_tokens_lower_bound']}` "
        f"(complete={resources['token_metering_complete']}).",
        "",
        "Unknown token or cost values are null and incomplete, never zero.",
        "",
        "## P3 conditional flywheel",
        "",
        f"Status: {report['p3']['status']}. Required: {report['p3']['required']}. "
        f"Claim ready: {report['p3']['claim_ready']}. "
        f"Scope: {report['p3']['scope'] or 'conditional reuse only'}.",
        "",
        "Prerequisite Seed/capture/C6 organizer/independent review costs are reported separately "
        "from downstream P3 evaluation cost and are counted once. They are never added to P2. "
        "Equal downstream budgets prove conditional reuse only; lifecycle superiority requires "
        "equal predeclared lifecycle allocations for both arms.",
    ]
    if report["p3"]["reasons"]:
        lines += ["", "P3 blockers or limitations:"] + [
            f"- {reason}" for reason in report["p3"]["reasons"]
        ]
    costs = report["p3"].get("prerequisite_costs") or {}
    if costs:
        totals = costs.get("known_totals", {})
        lines += [
            "",
            f"P3 prerequisite known totals (downstream excluded): model calls "
            f"{totals.get('model_calls')}, tokens {totals.get('tokens_total')}, "
            f"NPU runs {totals.get('npu_runs')}, cost USD {totals.get('estimated_cost_usd')}.",
        ]
    downstream = report["p3"].get("downstream_costs") or {}
    role = downstream.get("role") or {}
    if role:
        lines += [
            f"P3 downstream declared accounting (measured separately by the P3 receipt): "
            f"model calls {role.get('model_calls')}, tokens {role.get('tokens_total')}, "
            f"NPU runs {role.get('npu_runs')}, cost USD {role.get('estimated_cost_usd')}.",
        ]
    lines += _component_case_lines(report["component_cases"])
    lines += _native_lines(report["native_plugins"])
    lines += _runner_lines(report["component_runner_reports"])
    if report.get("supplementary_claims"):
        lines += ["", "## Explicitly scoped supplementary claims", ""]
        lines += [
            f"- {item.get('id')}: {item.get('scope') or 'scoped limitation only'}"
            for item in report["supplementary_claims"]
        ]
    if report["historical_attempts"]:
        lines += ["", "## Historical attempts retained", ""]
        lines += [
            f"- `{item['receipt']['trial_id']}`: {item['reason']}" for item in report["historical_attempts"]
        ]
    if report["blocking_reasons"]:
        lines += ["", "## Missing or incompatible evidence", ""]
        lines += [f"- {reason}" for reason in report["blocking_reasons"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    return "\n".join(lines) + "\n"


def homepage_markdown_product(report: dict, language: str = "zh") -> str:
    if report["release_gate"] is not True:
        raise ValueError("homepage results require a passed release gate")
    if language not in {"zh", "en"}:
        raise ValueError("homepage language must be zh or en")
    zh = language == "zh"
    lines = [
        BEGIN_MARKER,
        f"### {_text(report['product_version'])} " + ("实测结果" if zh else "measured results"),
        "",
        (
            "固定任务集上的产品矩阵实测（P0/P1/P2/P3 产品臂，绝不回译为旧 G0..G3 标签；"
            "失败、基础设施失败、未知与缺失均保留在分母中）："
            if zh else
            "Measured product matrix on the fixed task set (P0/P1/P2/P3 product arms, never "
            "translated back into legacy G0..G3 labels; failures, infrastructure failures, "
            "unknown outcomes and missing cells remain in every denominator):"
        ),
        "",
        "| " + ("场景组" if zh else "Cohort") + " | "
        + ("对照" if zh else "Comparison") + " | "
        + ("观察/计划" if zh else "Observed / planned") + " | "
        + ("双方解出" if zh else "Mutually solved") + " | "
        + ("交付耗时比（基线/候选）" if zh else "Delivery ratio (baseline/candidate)") + " | "
        + ("超预算次数（基线/候选）" if zh else "Over-budget attempts (baseline/candidate)") + " |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for comparison in report["comparisons"]:
        denominator = comparison.get("denominator", {})
        overruns = _budget_overrun_counts(comparison)
        lines.append(
            f"| {_text(comparison.get('cohort', ''))} | "
            f"{comparison['baseline_group']} → {comparison['candidate_group']} | "
            f"{denominator.get('observed', 0)}/{denominator.get('planned', 0)} | "
            f"{denominator.get('mutually_solved', 0)} | "
            f"{_number(comparison.get('wall_clock_speedup_on_mutual_success'), digits=3)} | {overruns} |"
        )
    lines += [
        "",
        (
            "**条件**：P0 原生宿主；P1 正式 v0.0.2 基线；P2 最终 v0.0.3 完整发布；"
            "P3 条件性经验复用（Seed/C6/评审成本单独计账，下游预算相同仅支持条件复用）。"
            if zh else
            "**Conditions**: P0 bare host; P1 formal v0.0.2 baseline; P2 final v0.0.3 "
            "complete release; P3 conditional experience reuse (Seed/C6/review cost is "
            "separately accounted; equal downstream budgets support conditional reuse only)."
        ),
        "",
        (
            "组件与原生兼容证据不构成单组件因果收益；Codex/Claude Code/Kimi 仅为兼容性证据，"
            "匹配的产品比较由 DSH 承载。成功和耗时比保留已验收的超预算结果；零次已知超限不代表预算计量完整。"
            "未测得的 token 或成本为 null 且不完整，绝不记为零。"
            if zh else
            "Component and native compatibility evidence is not a causal single-component gain; "
            "Codex/Claude Code/Kimi are compatibility-only and DSH carries the matched product "
            "comparison. Success and delivery ratios retain verified over-budget results; zero known "
            "overruns does not imply complete metering. Unmeasured token or cost values are null "
            "and incomplete, never zero."
        ),
    ]
    if report["p3"]["status"] != "admitted" or not report["p3"]["claim_ready"]:
        lines += [
            "",
            (
                "P3 真实闭环链未完成：该行仅作为明确限定的补充声明，不构成实测成功。"
                if zh else
                "The real P3 closed-family chain is not complete: this row is an explicitly "
                "scoped limitation, not a measured success."
            ),
        ]
    lines += [
        "",
        (
            f"[完整评测报告、成本记录与限制](docs/releases/{report['product_version']}/value-results.md)"
            if zh else
            f"[Full report, cost records and limitations](docs/releases/{report['product_version']}/value-results.md)"
        ),
        END_MARKER,
    ]
    return "\n".join(lines) + "\n"


def component_view_product(report: dict, target: dict) -> dict:
    from mindie_bench.value_publish import component_name

    component = component_name(target["repository"])
    if report["component_revisions"].get(component) != target["revision"]:
        raise ValueError(f"component value attachment source differs: {component}")
    if component in {"codex", "claude-code", "kimi"}:
        scope = (
            "Native compatibility evidence only. This plugin does not carry the matched "
            "product comparison; no single-component productivity or causal gain is claimed."
        )
        comparisons, matrix = [], None
    elif component == "dsh":
        scope = (
            "DSH carries the matched P0/P1/P2/P3 product comparison. The results measure the "
            "configured product combination, not a causal single-component gain."
        )
        comparisons, matrix = report["comparisons"], report.get("matrix")
    elif component in {"knowledge", "remote-dev", "coordinator", "npu-top", "diagnostics"}:
        scope = (
            "Combination context and component functional evidence. Product arms are joint "
            "configurations, so these results cannot be attributed to this component alone."
        )
        comparisons, matrix = report["comparisons"], report.get("matrix")
    elif component == "mindie-bench":
        scope = (
            "Product measurement and coverage evidence. These task results measure the "
            "evaluated product, not productivity caused by the measurement tool itself."
        )
        comparisons, matrix = report["comparisons"], report.get("matrix")
    else:
        scope = "No product or component claim."
        comparisons, matrix = [], None
    cases = [
        case for case in report["component_cases"]
        if component == "mindie-bench" or case["component"] == component
    ]
    if component in {"codex", "claude-code", "kimi"}:
        base = {
            key: report[key]
            for key in (
                "schema_version",
                "product_version",
                "suite",
                "suite_identity",
                "plan_sha256",
                "plan_digests",
                "benchmark",
                "formal_baseline",
                "final_release",
                "component_revisions",
                "component_runner_reports",
                "limitations",
            )
        }
    else:
        base = report
    view = {
        **base,
        "schema_version": "mindie-component-value-report/2",
        "repository": target["repository"],
        "component": component,
        "component_version": target["version"],
        "component_revision": target["revision"],
        "benchmark_scope": scope,
        "matrix": matrix,
        "comparisons": comparisons,
        "component_cases": cases,
        "native_plugins": [
            plugin for plugin in report["native_plugins"] if plugin["plugin"] == component
        ],
        "limitations": [scope, *report["limitations"]],
    }
    return view


def component_markdown_product(view: dict) -> str:
    lines = [
        f"# {view['repository']} {view['component_version']}",
        "",
        f"Component: `{view['component']}` at `{view['component_revision']}`. "
        f"Product version: {view['product_version']}.",
        "",
        view["benchmark_scope"],
        "",
        "## Product context",
        "",
        "| Cohort | Product arm | Planned | Observed | Missing | Pass | Fail | Infra | Unknown |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    matrices = (
        {cohort_id: cohort["matrix"] for cohort_id, cohort in view["cohorts"].items()}
        if view.get("cohorts") else ({"": view["matrix"]} if view.get("matrix") else {})
    )
    for cohort_id, matrix in matrices.items():
        for group in matrix["groups"].values():
            denominator = group["denominator"]
            lines.append(
                f"| {_text(cohort_id)} | {group['group']} | {denominator['planned']} | "
                f"{denominator['observed']} | {denominator['missing']} | "
                f"{denominator['pass']} | {denominator['fail']} | "
                f"{denominator['infra']} | {denominator['unknown']} |"
            )
    if view["comparisons"]:
        lines += [
            "",
            "## Measured comparisons",
            "",
            "| Cohort | Comparison | Observed / planned | Mutually solved | Delivery ratio | "
            "Over-budget attempts (baseline/candidate) | Status |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
        for comparison in view["comparisons"]:
            denominator = comparison.get("denominator", {})
            lines.append(
                f"| {_text(comparison.get('cohort', ''))} | "
                f"{comparison['baseline_group']} -> {comparison['candidate_group']} | "
                f"{denominator.get('observed', 0)}/{denominator.get('planned', 0)} | "
                f"{denominator.get('mutually_solved', 0)} | "
                f"{_number(comparison.get('wall_clock_speedup_on_mutual_success'), digits=3)} | "
                f"{_budget_overrun_counts(comparison)} | "
                f"{comparison['measurement_status']} |"
            )
    lines += _component_case_lines(view["component_cases"])
    lines += _native_lines(view["native_plugins"])
    lines += _runner_lines(view["component_runner_reports"])
    if view.get("blocking_reasons"):
        lines += ["", "## Missing or incompatible evidence", ""]
        lines += [f"- {reason}" for reason in view["blocking_reasons"]]
    lines += ["", "## Limitations", ""] + [f"- {limitation}" for limitation in view["limitations"]]
    return "\n".join(lines) + "\n"
