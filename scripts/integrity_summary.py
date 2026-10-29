"""Assemble an integrity readout from current soundness and replay evidence.

[Role] Bind current verifier bytes, subprocess logs, and historical replays before reporting.
[Invariants] TaskSpec checks task identity; this summary additionally enforces real three-state evidence.
[Timeouts & Locks] Reads bounded local evidence only; no subprocesses, devices, models or locks.
[Side-Effect Policy] Writes summary.json and README.md only under the requested output.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mindie_bench.evaluation.reevaluate import read_json, sha256, verifier_identity, write_json
from mindie_bench.io import task_hash
from mindie_bench.models import TaskSpec
from mindie_bench.runner.verifier import parse_verifier


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def evidence_path(value: str, task: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else task.parents[2] / path


def verify_control(task: Path, receipt: dict, name: str, item: dict) -> dict:
    label = f"{task.name}/{name}"
    expected = 1.0 if name.startswith("gold") else 0.0
    require(item.get("expected_reward", expected) == expected, f"{label}: conflicting expectation")
    require(item["reward"] == expected, f"{label}: control reward failed")
    require(item["exit_code"] == (0 if expected else 1), f"{label}: inconsistent subprocess exit")
    paths, hashes = {}, {}
    for stream in ("stdout", "stderr"):
        supplied = item.get(f"{stream}_path", item.get(stream))
        path = evidence_path(supplied, task) if supplied else Path(receipt["evidence_directory"]) / f"{name}.{stream}"
        hashes[stream] = sha256(path)
        require(item.get(f"{stream}_sha256", hashes[stream]) == hashes[stream], f"{label}: {stream} hash changed")
        paths[stream] = path
    require(paths["stdout"].stat().st_size <= 50 * 1024 * 1024, f"{label}: oversized stdout")
    last = ""
    with paths["stdout"].open() as stream:
        for line in stream:
            if line.strip():
                last = line
    payload = json.loads(last)
    require(isinstance(payload, dict) and "reward" in payload, f"{label}: missing structured verdict")
    require(payload == item["verdict"], f"{label}: saved verdict differs from actual stdout")
    validated = parse_verifier(item["exit_code"], last)
    require(validated.reward == expected, f"{label}: actual stdout contradicts control expectation")
    return {"reward": validated.reward, "expected_reward": expected, "exit_code": item["exit_code"],
            "stdout_sha256": hashes["stdout"], "stderr_sha256": hashes["stderr"]}


def soundness_entry(task: Path) -> dict:
    TaskSpec.load(task)
    path = task / "soundness_receipt.json"
    receipt, identity = read_json(path), verifier_identity(task)
    recorded = receipt.get("verifier_files", receipt.get("verifier_sha256"))
    require(isinstance(recorded, dict), f"{task.name}: missing verifier file identities")
    files = {name if name.startswith("verifier/") else f"verifier/{name}": value for name, value in recorded.items()}
    require(files == identity["files"], f"{task.name}: stale soundness verifier files")
    controls = {name: verify_control(task, receipt, name, item)
                for name, item in receipt.get("cases", receipt.get("controls", {})).items()}
    base = controls.get("base", controls.get("baseline"))
    negative = controls.get("negative_control", controls.get("negative"))
    gold = controls.get("gold")
    helpers = {name: item["reward"] for name, item in controls.items() if "helper" in name}
    require(bool(base and negative and gold and helpers), f"{task.name}: incomplete three-state/helper controls")
    require(receipt["before_reward"] == base["reward"] == 0, f"{task.name}: baseline summary mismatch")
    require(receipt.get("negative_control_reward", receipt.get("negative_reward")) == negative["reward"] == 0,
            f"{task.name}: negative summary mismatch")
    require(receipt["after_reward"] == gold["reward"] == 1 and all(v == 1 for v in helpers.values()),
            f"{task.name}: gold/helper summary mismatch")
    require(receipt["model_calls"] == 0, f"{task.name}: controls contain model calls")
    return {"task_id": task.name, "receipt_path": str(path), "receipt_sha256": sha256(path),
            "task_sha256": task_hash(task), "verifier_bundle_sha256": identity["sha256"],
            "status": receipt["status"], "three_state_and_helpers_passed": True,
            "scope": receipt.get("validation_scope", receipt.get("execution_scope", receipt.get("scope"))),
            "before_reward": base["reward"], "negative_reward": negative["reward"],
            "after_reward": gold["reward"], "helper_rewards": helpers,
            "subprocess_controls": controls, "model_calls": receipt["model_calls"],
            "execution": receipt.get("execution")}


def verify_replay(replay: dict, tasks: Path) -> None:
    require(replay["model_calls"] == 0, "replay contains model calls")
    for row in replay["trials"]:
        label = row["trial_id"]
        require(row["original_receipt_unchanged"] and sha256(Path(row["receipt_path"])) == row["receipt_sha256"],
                f"{label}: original receipt changed")
        if not row["selected_for_reevaluation"]:
            continue
        identity = verifier_identity(tasks / row["task_id"])
        require(row["new_verifier_bundle_sha256"] == identity["sha256"]
                and row["new_verifier_files"] == identity["files"]
                and row["new_verifier_sha256"] == identity["files"]["verifier/test.sh"],
                f"{label}: stale replay verifier identity")
        actual = row["reevaluation"].get("verifier")
        if actual is not None:
            require(actual == identity, f"{label}: executed verifier differs from recorded/current verifier")
        else:
            require(row["reevaluation"]["status"] == "not_evaluated", f"{label}: missing executed verifier identity")


def assemble(tasks: Path, replay_path: Path, report_check_path: Path, output: Path) -> dict:
    replay, report_check = read_json(replay_path), read_json(report_check_path)
    soundness = [soundness_entry(task) for task in sorted(tasks.iterdir()) if task.is_dir()]
    verify_replay(replay, tasks)
    transitions = Counter((row["original_status"], row["reevaluation"]["reward"])
                          for row in replay["trials"] if row["selected_for_reevaluation"])
    result = {
        "schema_version": "mindie-bench-integrity-summary/1",
        "evidence_gate": "current_task_and_verifier_hashes; control_expectations_and_exits; actual_logs; current_replay_oracle",
        "reevaluation_path": str(replay_path), "reevaluation_sha256": sha256(replay_path),
        "report_input_check_path": str(report_check_path), "report_input_check_sha256": sha256(report_check_path),
        "tasks_reviewed": len(soundness), "all_three_state_and_helpers_passed": bool(soundness),
        "soundness_subprocesses": sum(len(row["subprocess_controls"]) for row in soundness),
        "model_calls": replay["model_calls"] + sum(row["model_calls"] for row in soundness),
        "historical_summary": replay["summary"],
        "transitions": [{"original_status": old, "new_reward": reward, "trials": count}
                        for (old, reward), count in transitions.items()],
        "soundness": soundness, "report_generation": report_check,
        "hardware_speedup_measured": any(
            row["reevaluation"]["status"] == "completed"
            and row["reevaluation"].get("verdict", {}).get("measured_npu_speedup") is not None
            for row in replay["trials"]),
        "limitations": replay["limitations"],
        "supporting_evidence": [
            {"label": label, "path": str(path), "sha256": sha256(path)}
            for label, path in (
                ("DSpark 构造器与失败路径复核", output / "dspark-constructor-review/receipt.json"),
                ("G4 行为与性能边界", output / "g4-soundness-v3/README.md"),
            ) if path.is_file()
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    (output / "README.md").write_text(markdown(result))
    return result


def markdown(result: dict) -> str:
    summary, report_check = result["historical_summary"], result["report_generation"]
    lines = ["# 裁判加固与历史产物重判", "",
             f"已复核 {result['tasks_reviewed']} 题；最终有效控制共 {result['soundness_subprocesses']} 次真实子进程，"
             "包含原始基线、负对照、Gold、顶层 helper／辅助类变体。",
             "汇总已核对当前任务与裁判 SHA、控制退出码、实际日志，以及重判实际使用的裁判身份。",
             f"零模型重判覆盖 {summary['selected_trials']} / {summary['primary_trials']} 条主记录；"
             f"结果为 {summary['rejudged_rewards'].get('1.0', 0)} 条通过、"
             f"{summary['rejudged_rewards'].get('0.0', 0)} 条行为失败、"
             f"{summary['rejudged_rewards'].get('None', 0)} 条未完成验证。", "",
             f"[逐 trial 对账 JSON]({result['reevaluation_path']}) · [输入驱动报告检查]({result['report_input_check_path']})", "",
             "CPU 行为、受控夹具和 NPU 张量正确性按下表各自声明。"
             f"硬件提速实测证据：{'已记录，范围见实际 verdict' if result['hardware_speedup_measured'] else 'unmeasured'}。"
             "历史预算和裁判不一致不能靠零模型重判修复为因果性能实验。", "",
             "| 原始状态 | 新裁判 reward | 条数 |", "| --- | --- | ---: |"]
    for transition in result["transitions"]:
        value = transition["new_reward"]
        lines.append(f"| {transition['original_status']} | {value if value is not None else 'unmeasured'} | {transition['trials']} |")
    lines += ["", "新 reward 只描述保存产物在当前验证边界内的结果；原始 receipt、unknown、墙钟和用量均未改写。", "",
              "| 任务 | 验证范围 | 基线 / 负对照 / Gold | Helper |", "| --- | --- | --- | --- |"]
    for row in result["soundness"]:
        lines.append(f"| {row['task_id']} | {row['scope']} | {row['before_reward']} / {row['negative_reward']} / {row['after_reward']} | "
                     f"{', '.join(str(v) for v in row['helper_rewards'].values())} |")
    lines += ["", "`value_release.py` / `value_publish.py` 使用当前输入重建了 "
              f"{report_check['component_views']} 份本地组件视图；历史输入和空输入分别为 "
              f"{report_check['current_studies']} / {report_check['empty_studies']} 个 Agent study，"
              f"组件范围随输入从 {report_check['current_scope']['component_cases']} 变为 "
              f"{report_check['empty_scope']['component_cases']}。未修改旧发布材料。", "",
              "首轮和中间重判仅作为裁判诊断证据保留；本页链接的逐 trial JSON 是最终对账结果。"]
    if result["supporting_evidence"]:
        lines += ["", "失败边界复核：" + " · ".join(
            f"[{item['label']}]({item['path']})" for item in result["supporting_evidence"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--report-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.tasks.absolute(), args.replay.absolute(), args.report_check.absolute(), args.output.absolute())


if __name__ == "__main__":
    main()
