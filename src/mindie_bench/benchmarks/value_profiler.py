"""Measure real profiler accounting, comparison attribution and memory residuals.

[Role] Drive installed ``mindie_profiler`` over deterministic multi-rank SQLite captures.
[Invariants] Raw event arithmetic and source DB hashes remain the baseline; no inferred step is invented.
[Timeouts & Locks] Read-only SQLite queries use the component's bounded progress-handler deadline.
[Side-Effect Policy] Owned output files plus read-only source DBs; no source mutation or network.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from .value_common import source_revisions, write_json

RANK_IDS = (
    "dp0_pp0_tp0_dcp0_ep0_rank0",
    "dp0_pp0_tp0_dcp0_ep0_rank1",
)
WINDOWS = ((4, 0, 60_000), (5, 100_000, 160_000))
SCENE = {
    "framework": "vllm-ascend",
    "phase": "decode",
    "tp": 2,
    "dp": 1,
    "pp": 1,
    "graph_mode": "piecewise_graph",
    "clock_domain": "syscnt_aligned",
    "model": "Qwen2.5-7B-Instruct",
    "device": {"card": 0, "host": "a3-value-fixture"},
}

WEIGHT_BYTES = 7_000_000_000
KV_CACHE_BYTES = 2_000_000_000
RESIDUAL_BYTES = 50_000_000
BASELINE_BYTES = 900_000_000
TARGET_BYTES = BASELINE_BYTES + WEIGHT_BYTES + KV_CACHE_BYTES + RESIDUAL_BYTES
MSPROF_COMPONENTS = {
    "APP": 9_500_000_000,
    "HCCL": 1_100_000_000,
    "RUNTIME": 400_000_000,
    "SLOG": 120_000_000,
    "KERNEL": 80_000_000,
}


def _event(start_us: int, end_us: int, name: str, op_type: str,
           core: str = "AI_CORE", stream: int = 1, task: int = 1) -> dict:
    return {
        "start_us": start_us,
        "end_us": end_us,
        "name": name,
        "op_type": op_type,
        "core": core,
        "stream": stream,
        "task": task,
    }


def _event_rows(start: int, *, variant: str, rank: int) -> list[dict]:
    """Two identical decode steps; only the deliberately skewed block moves."""
    attention_end = start + (45 if variant == "rank" and rank == 1 else 10)
    comm_end = start + (45 if variant == "link" and rank == 1 else 30)
    return [
        _event(start, attention_end, "aclnnFlashAttention", "FlashAttention"),
        _event(start + 5, start + 15, "GroupedMatmul", "GroupedMatmul"),
        _event(start + 20, comm_end, "AivKernel", "hcom_allReduce_",
               core="COMMUNICATION", stream=3),
        _event(start + 30, start + 35, "aclnnRmsNorm", "RmsNorm", core="AI_VECTOR_CORE"),
        _event(start + 35, start + 40, "aclnnTransData", "TransData"),
    ]


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE META_DATA(name TEXT, value TEXT);
        CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, connectionId INTEGER,
                          globalTaskId INTEGER, globalPid INTEGER, taskType INTEGER, contextId INTEGER,
                          streamId INTEGER, taskId INTEGER, modelId INTEGER, depth INTEGER);
        CREATE TABLE COMPUTE_TASK_INFO(name INTEGER, globalTaskId INTEGER PRIMARY KEY,
                          blockDim INTEGER, mixBlockDim INTEGER, taskType INTEGER, opType INTEGER,
                          inputFormats INTEGER, inputDataTypes INTEGER, inputShapes INTEGER,
                          outputFormats INTEGER, outputDataTypes INTEGER, outputShapes INTEGER,
                          attrInfo INTEGER, opState INTEGER, hf32Eligible INTEGER, waitNs INTEGER);
        """
    )


def _build_db(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        _create_schema(connection)
        connection.execute("INSERT INTO META_DATA VALUES ('SCHEMA_VERSION', '1.2.0')")
        identifiers: dict[str, int] = {}

        def sid(value: str) -> int:
            if value not in identifiers:
                identifiers[value] = len(identifiers) + 1
                connection.execute("INSERT INTO STRING_IDS VALUES (?, ?)", (identifiers[value], value))
            return identifiers[value]

        for index, event in enumerate(events, start=1):
            task_type = sid(event["core"])
            connection.execute(
                "INSERT INTO TASK VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event["start_us"] * 1000, event["end_us"] * 1000, 0, index, index,
                    100, task_type, 4294967295, event["stream"], event["task"],
                    4294967295, 0,
                ),
            )
            connection.execute(
                "INSERT INTO COMPUTE_TASK_INFO VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid(event["name"]), index, 20, 0, task_type, sid(event["op_type"]),
                    0, 0, None, 0, 0, None, 0, sid("dynamic"), sid("NO"), 0,
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return path


def _build_run(root: Path, variant: str) -> dict[str, Path]:
    paths = {}
    for rank, rank_id in enumerate(RANK_IDS):
        events = [
            event
            for _step, start_ns, _end_ns in WINDOWS
            for event in _event_rows(start_ns // 1000, variant=variant, rank=rank)
        ]
        paths[rank_id] = _build_db(
            root / rank_id / "ASCEND_PROFILER_OUTPUT" / f"ascend_pytorch_profiler_{rank}.db",
            events,
        )
    return paths


def _union_ms(intervals: list[tuple[int, int]]) -> float:
    if not intervals:
        return 0.0
    total = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return (total + end - start) / 1_000_000


def _raw_step(events: list[dict], start_ns: int, end_ns: int) -> dict:
    selected = [
        event for event in events
        if event["start_us"] * 1000 >= start_ns and event["end_us"] * 1000 <= end_ns
    ]
    assert selected, "raw fixture step has no complete events"
    intervals = [(event["start_us"] * 1000, event["end_us"] * 1000) for event in selected]
    comm = [(event["start_us"] * 1000, event["end_us"] * 1000) for event in selected
            if event["core"] == "COMMUNICATION"]
    other = [(event["start_us"] * 1000, event["end_us"] * 1000) for event in selected
             if event["core"] != "COMMUNICATION"]
    other_union = _union_ms(other)
    comm_ms = _union_ms(comm)
    overlap_ms = max(0.0, comm_ms + other_union - _union_ms(comm + other))
    wall_ms = (max(event["end_us"] for event in selected)
               - min(event["start_us"] for event in selected))
    busy_ms = _union_ms(intervals)
    return {
        "event_count": len(selected),
        "wall_ms": wall_ms / 1000,
        "busy_union_ms": busy_ms,
        "kernel_sum_ms": sum(event["end_us"] - event["start_us"] for event in selected) / 1000,
        "exposed_comm_ms": comm_ms - overlap_ms,
        "host_gap_ms": wall_ms / 1000 - busy_ms,
    }


def _fixture_events(variant: str, rank: int) -> list[dict]:
    return [
        event
        for _step, start_ns, _end_ns in WINDOWS
        for event in _event_rows(start_ns // 1000, variant=variant, rank=rank)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _db_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {rank_id: _sha256(path) for rank_id, path in sorted(paths.items())}


def _memory_manifest() -> dict:
    peak = {
        "used_bytes": TARGET_BYTES,
        "source_kind": "npu_smi",
        "measurement_kind": "measured_peak",
        "is_peak": True,
        "peak_scope": "measured",
        "continuous_peak": True,
        "coverage_ratio": 1.0,
    }
    baseline = {
        "used_bytes": BASELINE_BYTES,
        "source_kind": "npu_smi",
        "measurement_kind": "point_observation",
    }
    return {
        "tp": 1,
        "dp": 1,
        "after_ready_hbm": {0: peak},
        "baseline_hbm": {0: baseline},
        "vllm_info": {"kv_cache_current_bytes": KV_CACHE_BYTES},
        "weight_manifest": {
            "per_device_bytes": WEIGHT_BYTES,
            "tp": 1,
            "dp": 1,
            "enable_expert_parallel": False,
        },
        "msprof_components_by_device": {0: dict(MSPROF_COMPONENTS)},
    }


def _check_step_accounting(report: dict, variant: str) -> dict:
    rows = {(row["rank_id"], row["step"]): row for row in report["window_accounting"]}
    baseline = {}
    for rank, rank_id in enumerate(RANK_IDS):
        events = _fixture_events(variant, rank)
        for step, start, end in WINDOWS:
            expected = _raw_step(events, start, end)
            observed = rows[(rank_id, step)]
            assert observed["event_count"] == expected["event_count"]
            for key in ("wall_ms", "busy_union_ms", "kernel_sum_ms",
                        "exposed_comm_ms", "host_gap_ms"):
                assert abs(observed[key] - expected[key]) <= 1e-9, (
                    variant, rank_id, step, key, observed[key], expected[key]
                )
            baseline[f"{rank_id}:step{step}"] = expected
    return baseline


def _check_memory(attribute_memory) -> dict:
    attributed = attribute_memory(
        _memory_manifest(), residual_target_bytes=100_000_000, msprof_unit="B",
    )
    device = attributed["devices"][0]
    assert attributed["status"] == "ok", device.get("missing_sources")
    assert device["known_sum_bytes"] == BASELINE_BYTES + WEIGHT_BYTES + KV_CACHE_BYTES
    assert device["residual"]["bytes"] == RESIDUAL_BYTES
    assert device["residual"]["status"] == "within_target"
    assert attributed["target_met"] is True
    included = {item["key"]: item["bytes"] for item in device["components"]
                if item["included_in_sum"]}
    assert included == {
        "baseline_observed": BASELINE_BYTES,
        "weights": WEIGHT_BYTES,
        "kv_cache": KV_CACHE_BYTES,
    }
    excluded = {item["key"] for item in device["components"]
                if not item["included_in_sum"]}
    assert {"hccl", "runtime", "slog", "other_msprof"} <= excluded
    containers = device["containers"]
    assert containers and {item["key"] for item in containers} == {"process_total"}
    assert all(item["included_in_sum"] is False for item in containers)
    assert any(overlap["key"] == "baseline_observed" for overlap in device["overlaps"])
    return {
        "schema_version": attributed["schema_version"],
        "status": attributed["status"],
        "known_sum_bytes": device["known_sum_bytes"],
        "residual_bytes": device["residual"]["bytes"],
        "residual_status": device["residual"]["status"],
        "target_met": attributed["target_met"],
        "included_components": included,
        "excluded_reference_keys": sorted(excluded),
        "container_keys": [item["key"] for item in containers],
        "msprof_reference_not_added": True,
    }


def run(output: Path, repetitions: int):
    from mindie_profiler import Window, analyze_profile
    from mindie_profiler.compare import compare_profiles
    from mindie_profiler.memory_profile import attribute_memory

    measurements = []
    for repetition in range(1, repetitions + 1):
        root = output / f"profiler-{repetition}"
        root.mkdir(parents=True)
        link_paths = _build_run(root / "link", "link")
        rank_paths = _build_run(root / "rank", "rank")
        before = {**_db_hashes(link_paths), **_db_hashes(rank_paths)}
        files_before = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*") if path.is_file()
        )
        windows = [Window(start_ns=start, end_ns=end, step=step, label=f"step_{step}")
                   for step, start, end in WINDOWS]
        started = time.perf_counter()
        link_report = analyze_profile(
            root / "link", windows=windows, scene=dict(SCENE), hash_sources=True,
        )
        rank_report = analyze_profile(
            root / "rank", windows=windows, scene=dict(SCENE), hash_sources=True,
        )
        analyze_ms = (time.perf_counter() - started) * 1000
        link_baseline = _check_step_accounting(link_report, "link")
        rank_baseline = _check_step_accounting(rank_report, "rank")
        assert link_report["step_accounting"]["slowest_rank"]["dominant_excess_block"] == "Comm"
        assert rank_report["step_accounting"]["slowest_rank"]["dominant_excess_block"] == "Attention"
        assert link_report["step_accounting"]["slowest_rank"]["rank_id"] == RANK_IDS[1]
        assert rank_report["step_accounting"]["slowest_rank"]["rank_id"] == RANK_IDS[1]

        started = time.perf_counter()
        comparison = compare_profiles(link_report, rank_report)
        compare_ms = (time.perf_counter() - started) * 1000
        assert comparison["comparable"] is True
        assert comparison["status"] == "comparable_with_limitations"
        blocks = {item["block"]: item for item in comparison["semantic_blocks"]["blocks"]}
        assert blocks["Comm"]["delta_ms"] < 0
        assert blocks["Attention"]["delta_ms"] > 0

        started = time.perf_counter()
        memory = _check_memory(attribute_memory)
        memory_ms = (time.perf_counter() - started) * 1000

        after = {**_db_hashes(link_paths), **_db_hashes(rank_paths)}
        files_after = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*") if path.is_file()
        )
        assert after == before
        assert files_after == files_before
        raw = {
            "fixture": {
                "ranks": list(RANK_IDS),
                "steps": [step for step, _start, _end in WINDOWS],
                "link_variant": "rank1 communication is longer; dominant excess is Comm",
                "rank_variant": "rank1 attention is longer; dominant excess is Attention",
            },
            "raw_baseline": {"link": link_baseline, "rank": rank_baseline},
            "link_step_accounting": link_report["step_accounting"],
            "rank_step_accounting": rank_report["step_accounting"],
            "comparison": {
                "status": comparison["status"],
                "comparable": comparison["comparable"],
                "step_accounting": comparison["step_accounting"],
                "semantic_blocks": comparison["semantic_blocks"],
            },
            "memory": memory,
            "db_sha256_before": before,
            "db_sha256_after": after,
            "db_files_unchanged": True,
        }
        write_json(root / "raw.json", raw)
        measurements.append({
            "repetition": repetition,
            "status": "pass",
            "raw": str(root / "raw.json"),
            "baseline": {
                "source": "raw SQLite event intervals",
                "steps": [step for step, _start, _end in WINDOWS],
                "raw_arithmetic_matches_report": True,
            },
            "candidate": {
                "profile_steps_aligned": True,
                "slow_link_dominant_block": "Comm",
                "slow_rank_dominant_block": "Attention",
                "comparison_status": comparison["status"],
                "comm_delta_ms": blocks["Comm"]["delta_ms"],
                "attention_delta_ms": blocks["Attention"]["delta_ms"],
                "known_sum_bytes": memory["known_sum_bytes"],
                "residual_bytes": memory["residual_bytes"],
                "residual_status": memory["residual_status"],
                "memory_target_met": memory["target_met"],
                "msprof_reference_not_added": True,
                "db_mutation": False,
                "analyze_latency_ms": analyze_ms,
                "compare_latency_ms": compare_ms,
                "memory_latency_ms": memory_ms,
            },
        })
    return {
        "id": "profiler-rank-link-memory-attribution",
        "component": "profiler",
        "status": "pass",
        "scope": "real analyze_profile, compare_profiles and attribute_memory on deterministic SQLite captures",
        "baseline": "Raw event intervals and raw SQLite files remain the only baseline.",
        "candidate": "Production profiler accounts steps, separates slow-rank from slow-link, and keeps references non-additive.",
        "repetitions": repetitions,
        "measurements": measurements,
        "limitations": [
            "The SQLite captures are deterministic fixtures; they are not new NPU measurements.",
            "The memory manifest is a bounded attribution input, not a live HBM capture.",
            "Profiler latency is CPU wall-clock for this host, not NPU throughput.",
            "Only the production read-only analyze/compare/attribute paths are exercised.",
        ],
        "source_revisions": source_revisions(("mindie-profiler",)),
    }
