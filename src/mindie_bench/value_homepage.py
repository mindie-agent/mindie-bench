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
            f"- coordinator：{count} 轮竞争，成功监听数 {listeners}，端口冲突 {conflicts}，"
            f"重复设备选择 {duplicates}。基线是先检查再操作，不是另一种原子调度器。"
            if zh else
            f"- coordinator: {count} contention runs; successful listeners {listeners}, "
            f"port conflicts {conflicts}, duplicate device selections {duplicates}. "
            "The baseline uses check-then-act, not another atomic scheduler."
        )
    if case["id"] == "npu-top-physical-owner-query":
        devices = _span((row.get("golden_physical_devices_correct") for row in candidate), language)
        correct = _span((row.get("golden_process_assignments_correct") for row in candidate), language)
        expected = _span((row.get("golden_process_assignments_expected") for row in candidate), language)
        owners = observed(candidate, "live_container_owner_correct")
        human_accuracy = _span((row.get("human_answer_accuracy") for row in baseline), language, digits=3)
        human_time = _span((row.get("human_completion_time_sec") for row in baseline), language, digits=2)
        return (
            f"- npu-top：{count} 轮回放，每轮正确识别设备 {devices} 个、进程归属 {correct}/{expected}；"
            f"现场容器归属检查 {owners} 通过。人工定位准确率：{human_accuracy}，耗时（秒）：{human_time}。"
            "重复回放不是独立硬件样本。"
            if zh else
            f"- npu-top: {count} replay runs; {devices} devices and {correct}/{expected} process "
            f"assignments correct per run; live container-owner checks passed {owners}. "
            f"Human accuracy: {human_accuracy}; task time (s): {human_time}. "
            "Replays are not independent hardware samples."
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
            f"- diagnostics：{count} 轮本地 HTTP 场景中，{failures} 次故障形成 {reports} 条报告、"
            f"{posts} 次 POST；证据保留 {fields}、脱敏 {privacy}、撤回后停止请求 {withdrawal}。"
            f"故障到交付耗时 {duration} 秒，包含生产重试等待；不代表 GitHub 可用性或人工排障收益。"
            if zh else
            f"- diagnostics: {count} local HTTP runs turned {failures} fault occurrences into "
            f"{reports} reports and {posts} POSTs; evidence retained {fields}, redaction {privacy}, "
            f"requests stopped after withdrawal {withdrawal}. Delivery took {duration} s, including "
            "the production retry wait; GitHub availability and human triage gains were not measured."
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
            f"**知识复用：{label}** — {len(lineages)} 轮，每轮 1 个 seed + {evaluations} 个不同 eval。"
            if zh else
            f"**Knowledge transfer: {label}** — {len(lineages)} runs, each with 1 seed + "
            f"{evaluations} distinct eval tasks."
        ),
        (
            f"实际 query 命中 seed {queries}/{len(entries)}，explain 读取 {explains}/{len(entries)}；"
            "未命中也保留在结果中。以下为全流程累计交付时间，已包含 seed 与 capture，不是仅 eval 的时间。"
            if zh else
            f"Query returned the seed in {queries}/{len(entries)} evals; explain read it in "
            f"{explains}/{len(entries)}. Misses remain in the results. Workflow totals below "
            "include seed and capture, unlike the eval-only paired timing above."
        ),
        "",
        (
            "| 条件 | 选定全流程（秒） | 全部已执行，含历史尝试（秒） |"
            if zh else
            "| Condition | Selected workflow (s) | All executed, including historical attempts (s) |"
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
            "固定任务集的小样本结果，按宿主、模型和对照分开列出，不跨模型求平均，也不代表普遍提效。"
            if zh else
            "Small-sample results on fixed tasks, reported separately by host, model and comparison. "
            "They are not averaged across models and do not establish a general productivity gain."
        ),
        "",
        (
            "G0：原生宿主；G1：加入 remote-dev；G2：再加入知识检索、随包 Skills 和显式检索提示；"
            "G3：在 G2 上采集前一任务的经验，供后续任务使用。"
            if zh else
            "G0: native host. G1: add remote-dev. G2: also add knowledge retrieval, shipped Skills "
            "and explicit retrieval guidance. G3: capture an earlier task's experience for later tasks."
        ),
        "",
        (
            "| 任务集 / 宿主 / 模型 | 每任务重复数 | 对照 | 基线解题数/尝试数 | 候选解题数/尝试数 | "
            "交付时间比（基线/候选） | 双方均解决的配对数 |"
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
            "解题数与尝试数使用选定的已结束 eval；失败仍计入分母。时间比仅使用双方均解决的任务，"
            "大于 1 表示候选在这些配对中更快，小于 1 表示更慢。"
            if zh else
            "Solved/attempts uses the selected settled evals, with failures in the denominator. "
            "Timing includes only tasks solved by both sides: above 1 means the candidate was faster "
            "on those pairs; below 1 means slower."
        ),
    ]
    for study in report["studies"]:
        lines.extend(_transfer_lines(study, language))
    if report["component_cases"]:
        lines += ["", "**组件场景**" if zh else "**Component scenarios**", ""]
        lines.extend(_component_line(case, language) for case in report["component_cases"])
    lines += [
        "",
        (
            f"[完整结果、条件定义与成本记录]({link})保留原始失败、未知终态、历史尝试和单点重测；"
            "准备与诊断模型调用单独记账，不混入任务价值。"
            if zh else
            f"[Full results, condition definitions and costs]({link}) retain original failures, "
            "unknown outcomes, historical attempts and targeted retests. Setup and diagnostic "
            "model usage is accounted for separately from task value."
        ),
        END_MARKER,
    ]
    return "\n".join(lines) + "\n"
