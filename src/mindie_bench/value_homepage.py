"""Render a compact release-results block for product homepages.

[Role] Present verified task comparisons and component observations in zh/en.
[Invariants] Never average models or turn functional checks into productivity.
Seed/capture costs and historical attempts stay distinct from paired eval time.
[Timeouts & Locks] Pure in-memory formatting; no waits, subprocesses or locks.
[Side-Effect Policy] Returns Markdown only; an incomplete release cannot render.
"""

from __future__ import annotations

BEGIN_MARKER = "<!-- mindie-value-results:start -->"
END_MARKER = "<!-- mindie-value-results:end -->"


def _text(value) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _number(value, language: str, *, digits: int = 1) -> str:
    if value is None:
        return "未测" if language == "zh" else "not measured"
    return f"{value:.{digits}f}"


def _duration(resources: dict, language: str) -> str:
    value = resources["observed_delivery_wall_clock_sec_lower_bound"]
    rendered = _number(value, language)
    if value is not None and not resources["delivery_timing_complete"]:
        rendered = ("至少 " if language == "zh" else "at least ") + rendered
    return rendered


def _span(values, language: str, *, digits: int = 0) -> str:
    values = list(values)
    if not values or any(value is None for value in values):
        return "未测" if language == "zh" else "not measured"
    low, high = min(values), max(values)
    return f"{low:.{digits}f}" if low == high else f"{low:.{digits}f}-{high:.{digits}f}"


def _component_line(case: dict, language: str) -> str:
    samples = case["measurements"]
    baseline = [sample["baseline"] for sample in samples]
    candidate = [sample["candidate"] for sample in samples]
    count = len(samples)
    zh = language == "zh"

    def total(rows, key):
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            return "未测" if zh else "not measured"
        return str(sum(values))

    def observed(rows, key):
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            return "未测" if zh else "not measured"
        return f"{sum(value is True for value in values)}/{len(values)}"

    if case["id"] == "coordinator-contention":
        listeners = f"{total(baseline, 'successful_listeners')} → {total(candidate, 'successful_listeners')}"
        conflicts = f"{total(baseline, 'bind_conflicts')} → {total(candidate, 'bind_conflicts')}"
        duplicates = (
            f"{total(baseline, 'duplicate_device_selections')} → "
            f"{total(candidate, 'duplicate_device_selections')}"
        )
        return (
            f"- coordinator：{count} 轮双 Agent 并发抢占实测，成功建立监听数 {listeners}，"
            f"端口冲突 {conflicts}，重复选卡 {duplicates}（对比无协调的“先检查再绑定”基线）。"
            if zh else
            f"- coordinator: {count} concurrent two-agent contention runs; successful listeners {listeners}, "
            f"port conflicts {conflicts}, duplicate device selections {duplicates} "
            "(compared against an uncoordinated check-then-act baseline)."
        )
    if case["id"] == "npu-top-physical-owner-query":
        devices = _span((row.get("golden_physical_devices_correct") for row in candidate), language)
        correct = _span((row.get("golden_process_assignments_correct") for row in candidate), language)
        expected = _span((row.get("golden_process_assignments_expected") for row in candidate), language)
        owners = observed(candidate, "live_container_owner_correct")
        human_accuracy = _span((row.get("human_answer_accuracy") for row in baseline), language, digits=3)
        human_time = _span((row.get("human_completion_time_sec") for row in baseline), language, digits=2)
        return (
            f"- npu-top：{count} 轮硬件快照回放，每轮准确识别物理卡 {devices} 张、进程归属 {correct}/{expected}；"
            f"实机容器归属探测 {owners} 通过。人工排查准确率：{human_accuracy}，耗时（秒）：{human_time}"
            "（快照回放验证解析确定性，不等同于独立硬件样本）。"
            if zh else
            f"- npu-top: {count} hardware-capture replay runs; {devices} physical devices and {correct}/{expected} "
            f"process assignments identified per run; live container-owner checks passed {owners}. "
            f"Human accuracy: {human_accuracy}; task time (s): {human_time} "
            "(capture replays verify parser determinism rather than independent hardware samples)."
        )
    if case["id"] == "diagnostics-fault-delivery":
        failures = total(candidate, "failure_occurrences")
        reports = total(candidate, "delivered_reports")
        posts = total(candidate, "http_posts")
        fields = observed(candidate, "useful_fault_fields_retained")
        privacy = observed(candidate, "private_fixture_removed")
        withdrawal = observed(candidate, "withdrawal_prevented_requests")
        duration = _span((row.get("failure_to_delivery_sec") for row in candidate), language, digits=2)
        return (
            f"- diagnostics：{count} 轮本地 HTTP 上报场景中，{failures} 次报错自动去重形成 {reports} 条报告、"
            f"{posts} 次 POST；关键字段保留 {fields}、隐私脱敏 {privacy}、关闭开关后立即停发 {withdrawal}。"
            f"从报错到送达耗时 {duration} 秒（含约 60 秒生产重试等待；本地测试不代表 GitHub 线上可用性或人工排障耗时）。"
            if zh else
            f"- diagnostics: {count} local HTTP reporting runs deduplicated {failures} faults into "
            f"{reports} reports and {posts} POSTs; key fault fields retained {fields}, privacy redaction {privacy}, "
            f"requests stopped after consent withdrawal {withdrawal}. Delivery took {duration} s "
            "(including the ~60 s production retry wait; local HTTP does not measure GitHub availability or human triage time)."
        )
    if case["id"] == "profiler-rank-link-memory-attribution":
        aligned = observed(candidate, "profile_steps_aligned")
        memory_met = observed(candidate, "memory_target_met")
        mutations = [row.get("db_mutation") for row in candidate]
        immutable = (
            ("未测" if zh else "not measured")
            if any(value is None for value in mutations)
            else f"{sum(value is False for value in mutations)}/{len(mutations)}"
        )
        analyze_ms = _span((row.get("analyze_latency_ms") for row in candidate), language, digits=2)
        return (
            f"- profiler：{count} 轮确定性 SQLite 采集归因回放，Step 窗口对齐 {aligned}，"
            f"慢卡（Attention）与慢链路（Comm）归因分离，显存归因达标 {memory_met}，"
            f"只读零库文件变更 {immutable}，分析耗时 {analyze_ms} ms（夹具回放验证归因准确性，不等同于新增实机采样）。"
            if zh else
            f"- profiler: {count} deterministic SQLite attribution runs; step alignment {aligned}, "
            f"slow-rank (Attention) vs slow-link (Comm) separated, memory attribution target met {memory_met}, "
            f"read-only DB immutability {immutable}, analyze latency {analyze_ms} ms "
            "(fixture replay verifies attribution accuracy rather than new hardware captures)."
        )
    if case["id"] == "sim-a3-graph-and-boundary-screening":
        deterministic = observed(candidate, "qwen_case_deterministic")
        overflow_rejected = observed(candidate, "subgraph_overflow_rejected")
        tpot_err = _span(
            (
                abs(row["tpot_error_pct"]) if isinstance(row.get("tpot_error_pct"), (int, float)) else None
                for row in candidate
            ),
            language,
            digits=2,
        )
        saving_us = _span(
            (row.get("subgraph_critical_path_saving_us") for row in candidate),
            language,
            digits=1,
        )
        return (
            f"- sim：{count} 轮 A3 物理标定图预测与算子/子图边界筛查，Qwen2.5-7B 图预测确定性 {deterministic}"
            f"（TPOT 相对误差 {tpot_err}%），UB 溢出融合拦截 {overflow_rejected}，"
            f"可行融合关键路径收益 {saving_us} us（基于公开标定锚点与算术夹具，不含客户端固定调度底噪）。"
            if zh else
            f"- sim: {count} A3 physical-calibration graph prediction and operator/subgraph screening runs; "
            f"Qwen2.5-7B graph prediction determinism {deterministic} (TPOT relative error {tpot_err}%), "
            f"UB overflow fusion rejected {overflow_rejected}, feasible fusion critical-path saving {saving_us} us "
            "(anchored on published calibration fixtures without client scheduler floor)."
        )
    raise ValueError(f"unsupported homepage component case: {case['id']}")


def _transfer_lines(study: dict, language: str) -> list[str]:
    lineages = study["transfer_lineages"]
    if not lineages:
        return []
    zh = language == "zh"
    entries = [entry for lineage in lineages for entry in lineage["evals"]]
    queries = sum(entry["seed_query_hit_observed"] is True for entry in entries)
    explains = sum(entry["seed_explain_read_observed"] is True for entry in entries)
    evaluations = len(lineages[0]["evals"])
    label = f"{_text(study['id'])} / {_text(study['model'])}"
    lines = [
        "",
        (
            f"**跨任务经验复用：{label}** — 共 {len(lineages)} 轮，每轮先完成 1 个前置任务并沉淀经验（`seed` + `capture`），"
            f"再执行 {evaluations} 个不同的后续任务（`eval`）。"
            if zh else
            f"**Cross-task experience reuse: {label}** — {len(lineages)} runs, each executing 1 seed task "
            f"with experience capture followed by {evaluations} distinct eval tasks."
        ),
        (
            f"后续任务中检索（`query`）命中前置经验 {queries}/{len(entries)} 次、展开读取（`explain`）{explains}/{len(entries)} 次"
            "（未命中轮次原样保留）。下表为包含前置任务与经验沉淀在内的全流程总耗时；知识检索与全流程耗时预计在 `0.0.3` 优化："
            if zh else
            f"Later eval tasks hit the captured seed via `query` in {queries}/{len(entries)} runs and read it via "
            f"`explain` in {explains}/{len(entries)} runs (misses are retained). Workflow totals below include the "
            "seed task and capture cost; retrieval overhead and end-to-end workflow latency are scheduled for optimization in `0.0.3`:"
        ),
        "",
        (
            "| 条件 | 选定正式轮次全流程耗时（秒） | 全部已执行（含历史重跑，秒） |"
            if zh else
            "| Condition | Selected workflow total (s) | All executed, including historical attempts (s) |"
        ),
        "| --- | ---: | ---: |",
    ]
    for group, selected in study["groups"].items():
        all_executed = study["resources_by_group"][group]
        lines.append(
            f"| {_text(group)} | {_duration(selected, language)} | {_duration(all_executed, language)} |"
        )
    return lines


def homepage_markdown(report: dict, language: str = "zh") -> str:
    """Render only a report whose release gate has already passed."""
    if report.get("schema_version") == "mindie-value-report/2":
        from mindie_bench.value_product_views import homepage_markdown_product

        return homepage_markdown_product(report, language)
    if report["release_gate"] is not True:
        raise ValueError("homepage results require a passed release gate")
    if language not in {"zh", "en"}:
        raise ValueError("homepage language must be zh or en")
    zh = language == "zh"
    version = _text(report["product_version"])
    link = f"docs/releases/{report['product_version']}/value-results.md"
    lines = [
        BEGIN_MARKER,
        f"### {version} " + ("实测结果" if zh else "measured results"),
        "",
        (
            "固定任务集上的小样本实测结果（按宿主、模型与对照条件分别列出，不跨模型求平均）："
            "`remote-dev`（G0 → G1）在 `quarot` 上实现 `3/3` 解出与 `1.025x` 交付提速，在 `dp2` 上与原生基线基本持平（`0.963x`）；"
            "`coordinator` 在双 Agent 并发抢占下实现零冲突协调；"
            "`knowledge` + Skills（G2/G3）提升了 `dp2` 与 `dsh` 的解题完成数并跑通跨任务经验复用，检索开销预计在 `0.0.3` 重点优化。"
            if zh else
            "Small-sample results on the fixed task suite (reported separately by host, model and comparison, never averaged across models): "
            "`remote-dev` (G0 → G1) solved `3/3` `quarot` runs with a `1.025x` speedup and matched baseline on `dp2` (`0.963x`); "
            "`coordinator` eliminated port and device conflicts under concurrent contention; "
            "`knowledge` + Skills (G2/G3) increased solved tasks on `dp2` and `dsh` and validated cross-task reuse, with retrieval latency targeted for optimization in `0.0.3`."
        ),
        "",
        (
            "**实验条件**：**G0** 原生宿主；**G1** 接入 `remote-dev` 远端执行；"
            "**G2** 在 G1 上同时启用 `knowledge` 检索、随包 Skills 与显式检索提示；"
            "**G3** 在 G2 上将前置任务经验沉淀入库，供后续任务检索复用。"
            if zh else
            "**Conditions**: **G0** native host; **G1** adds `remote-dev` remote execution; "
            "**G2** adds `knowledge` retrieval, shipped Skills and explicit retrieval guidance on top of G1; "
            "**G3** captures an earlier task's experience on top of G2 for later tasks."
        ),
        "",
        (
            "| 任务集 / 宿主 / 模型 | 每任务重复数 | 对照 | 基线通过/尝试 | 启用后通过/尝试 | "
            "耗时比（基线/启用后） | 双方均通过的配对数 |"
            if zh else
            "| Study / host / model | Repeats per task | Pair | Baseline solved/attempts | "
            "Candidate solved/attempts | Delivery-time ratio (baseline/candidate) | Mutually solved pairs |"
        ),
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for study in report["studies"]:
        effort = study.get("reasoning_effort")
        model = _text(study["model"]) + (f" ({_text(effort)})" if effort else "")
        label = f"{_text(study['id'])} / {_text(study['harness'])} / {model}"
        repeats = study.get("repetitions")
        repeats = str(repeats) if repeats is not None else ("未记录" if zh else "not recorded")
        for pair in study["comparisons"]:
            baseline, candidate = pair["baseline"], pair["candidate"]
            ratio = pair["wall_clock_speedup_on_mutual_success"]
            rendered = _number(ratio, language, digits=3) + ("x" if ratio is not None else "")
            lines.append(
                f"| {label} | {repeats} | {_text(pair['baseline_group'])} → "
                f"{_text(pair['candidate_group'])} | "
                f"{baseline['completed_pass']}/{baseline['attempts']} | "
                f"{candidate['completed_pass']}/{candidate['attempts']} | {rendered} | "
                f"{pair['comparison_coverage']['mutually_solved_pairs']} |"
            )
    lines += [
        "",
        (
            "注：通过/尝试统计选定的正式实验轮次（失败计入分母）。耗时比仅统计同一轮双方均通过的配对，"
            "`> 1.0x` 表示启用后更快，`< 1.0x` 表示因增加检索或工具交互而更慢。"
            "通过结果和耗时比保留已验收的超预算完成；完整报告单列预算观察，未计量不视为合规。"
            if zh else
            "Note: Solved/attempts counts the selected settled runs (with failures in the denominator). "
            "Timing ratios use only pairs solved by both sides: `> 1.0x` means the candidate was faster; "
            "`< 1.0x` means slower. Verified over-budget completions remain included; the full report "
            "shows budget observations separately, and missing meters do not establish compliance."
        ),
    ]
    for study in report["studies"]:
        lines.extend(_transfer_lines(study, language))
    if report["component_cases"]:
        lines += ["", "**组件实测场景**" if zh else "**Component scenarios**", ""]
        lines.extend(_component_line(case, language) for case in report["component_cases"])
    lines += [
        "",
        (
            f"[完整评测报告、条件定义与成本记录]({link})保留全部原始失败、状态未知（`unknown`）记录、历史尝试与单点重测链路；"
            "环境准备与诊断探针的模型调用单独记账，不混入任务评测。"
            if zh else
            f"[Full benchmark report, condition definitions and cost records]({link}) retain original failures, "
            "unknown outcomes, historical attempts and targeted retests. Setup and diagnostic "
            "model usage is accounted for separately from task evaluations."
        ),
        END_MARKER,
    ]
    return "\n".join(lines) + "\n"
