"""Answer physical-device and process-ownership questions from real observations.

[Role] Compare raw npu-smi/proc/Docker evidence with the production structured projection.
[Invariants] The raw CLI is a valid baseline; no invented human accuracy or speed is assigned.
[Timeouts & Locks] Each read-only SSH operation has a 25-second deadline.
[Side-Effect Policy] Reads standing hardware and an existing container; creates no remote process.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from .value_common import source_revisions, write_json


def ssh(host: str, script: str):
    started = time.monotonic()
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "-o", "ControlMaster=no", host, "bash", "-s"],
        input=script.encode(), capture_output=True, timeout=25, check=True,
    )
    return result.stdout.decode(), time.monotonic() - started


def _process_answers(parsed):
    return [
        {"card": device["npu_id"], "physical_device": process.get("phy_id", device["npu_id"]),
         "pid": process["pid"], "memory_mb": process["npu_memory_mb"]}
        for device in parsed["devices"] for process in device.get("processes", [])
    ]


def golden(fixtures: Path):
    from npu_top.npu_smi import parse_npu

    metadata = json.loads((fixtures / "npu-smi-captures.json").read_text())
    cases = []
    for name in ("a2-173-125-1-2.txt", "a3-192-168-13-154-busy.txt"):
        raw = (fixtures / name).read_bytes()
        declared = next(command for sample in metadata["samples"]
                        for command in sample["commands"] if command["file"] == name)
        assert hashlib.sha256(raw).hexdigest() == declared["sha256"]
        parsed = parse_npu(raw.decode())
        answers = _process_answers(parsed)
        if name.startswith("a2"):
            assert len(parsed["devices"]) == 8 and answers == []
            expected = []
            physical_devices = 8
        else:
            # The oracle reads the captured table's explicit card/chip/PID/HBM columns.
            # This is a raw-table reference, not the npu-top parser's own output.
            expected = [
                {"card": int(card), "physical_device": 2 * int(card) + int(chip),
                 "pid": int(pid), "memory_mb": int(memory)}
                for card, chip, pid, memory in re.findall(
                    r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*VLLMWorker_DP\s*\|\s*(\d+)\s*\|",
                    raw.decode(),
                )
            ]
            assert len(expected) == 16 and answers == expected
            assert [chip["phy_id"] for d in parsed["devices"] for chip in d["chips"]] == list(range(16))
            physical_devices = 16
        cases.append({"capture": name, "capture_sha256": declared["sha256"],
                      "capture_date": metadata["capture_date"], "physical_devices": physical_devices,
                      "raw_cli_reference_answers": expected, "structured_answers": answers,
                      "correct_process_assignments": len(expected),
                      "expected_process_assignments": len(expected)})
    return cases


def run(output: Path, fixtures: Path, repetitions: int, host: str | None, container: str | None):
    from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_info
    from npu_top.npu_smi import parse_npu
    from npu_top.probe import (
        attach_process_details,
        build_process_detail_script,
        parse_process_details,
        split_sections,
    )

    measurements = []
    observed = None
    for repetition in range(1, repetitions + 1):
        root = output / f"npu-top-{repetition}"
        root.mkdir(parents=True)
        captured = golden(fixtures)
        live = None
        if host:
            info, info_sec = ssh(host, "timeout 8 npu-smi info\n")
            parsed = parse_npu(info)
            observed = parse_npu_smi_info(info)
            assert observed["status"] == "ok"
            physical = [
                chip["phy_id"] for device in parsed["devices"]
                for chip in device.get("chips", [{"phy_id": device["npu_id"]}])
            ]
            assert sorted(physical) == sorted(observed["devices"])
            (root / "live-npu-smi.txt").write_text(info)
            live = {"physical_devices": physical, "processes": _process_answers(parsed),
                    "raw_npu_elapsed_sec": info_sec, "ssh_operations": 1}
            if container:
                inspect_text, inspect_sec = ssh(
                    host, "docker inspect --format '{{json .Id}} {{.State.Pid}}' "
                    + shlex.quote(container) + "\n",
                )
                container_id, pid_text = inspect_text.strip().rsplit(" ", 1)
                container_id, pid = json.loads(container_id), int(pid_text)
                assert pid > 0
                proc_text, proc_sec = ssh(host, build_process_detail_script([pid]))
                details = parse_process_details(split_sections(proc_text)["process_details"])
                assert details[pid]["container_id"] == container_id
                # Exercise the production owner join on an explicit process record. This
                # record is not attributed to an NPU: it is the observed container init.
                rows = [{"processes": [{"pid": pid}]}]
                attach_process_details(
                    rows, details, {"containers": [{"id": container_id, "name": container}]}
                )
                owner = rows[0]["processes"][0]["container"]
                assert owner["id"] == container_id and owner["name"] == container
                live["container_ownership"] = {
                    "process_scope": "existing container init; not asserted to occupy an NPU",
                    "host_pid": pid, "docker_inspect_id": container_id,
                    "proc_cgroup_id": details[pid]["container_id"],
                    "structured_container": owner,
                    "inspect_sec": inspect_sec, "proc_sec": proc_sec,
                    "correct": True,
                }
                live["ssh_operations"] += 2
            write_json(root / "live-answers.json", live)
        raw = {"golden": captured, "live": live}
        write_json(root / "raw.json", raw)
        measurements.append({
            "repetition": repetition, "status": "pass", "raw": str(root / "raw.json"),
            "baseline": {"information_source": "original npu-smi tables and proc/Docker identities",
                         "human_answer_accuracy": None, "human_completion_time_sec": None},
            "candidate": {"golden_physical_devices_correct": 24,
                          "golden_process_assignments_correct": 16,
                          "golden_process_assignments_expected": 16,
                          "live_physical_devices": len(live["physical_devices"]) if live else None,
                          "live_container_owner_correct": (
                              live.get("container_ownership", {}).get("correct") if live else None),
                          "live_ssh_operations": live["ssh_operations"] if live else None},
        })
    return {
        "id": "npu-top-physical-owner-query", "component": "npu-top", "status": "pass",
        "scope": "physical die/PID/HBM correctness from captures and live proc/container identity",
        "baseline": "Raw npu-smi/proc/Docker observations contain the reference information.",
        "candidate": "Production npu-top produces structured physical-device and container-owner answers.",
        "repetitions": repetitions, "measurements": measurements,
        "limitations": [
            "A2 live and A3 dual-die historical captures are reported separately.",
            "Golden replay repetitions measure repeatability, not independent hardware samples.",
            "The container identity probe does not claim its init process consumes NPU memory.",
            "No model/human task timing, avoided incident count or causal productivity gain is measured.",
            "Raw command lines, working directories and environments are not exported.",
        ],
        "source_revisions": source_revisions(),
    }, observed
