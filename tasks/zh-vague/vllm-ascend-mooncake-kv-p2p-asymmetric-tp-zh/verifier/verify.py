# [Role] Independent CPU behavior oracle for Mooncake P2P asymmetric TP & request cleanup (Stream C - V1).
# [Invariants] Only executed behavioral assertions establish success; no NPU hardware required.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Parses candidate AST and executes target classes/methods in isolation; emits JSON only.

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import subprocess
import threading
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
TARGET_REL = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
ALLOWED_FILES = {TARGET_REL}


def check_diff_scope() -> None:
    if not (ROOT / ".git").exists():
        return
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    disallowed = {p for p in changed if p not in ALLOWED_FILES and not p.startswith("tests/")}
    if disallowed:
        raise AssertionError(f"Modified files outside allowed scope {sorted(ALLOWED_FILES)}: {sorted(disallowed)}")


def string_to_int64_hash(input_str: str) -> int:
    digest = hashlib.sha256(input_str.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


class RequestStatus:
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"


def load_mooncake_components():
    path = ROOT / TARGET_REL
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    tracker_node = None
    scheduler_methods = []
    worker_methods = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheTaskTracker":
            tracker_node = node
        elif isinstance(node, ast.ClassDef) and node.name == "MooncakeConnectorScheduler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {"request_finished"}:
                    scheduler_methods.append(item)
        elif isinstance(node, ast.ClassDef) and node.name == "MooncakeConnectorWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "_get_attention_group_num_need_pulls",
                    "_get_attention_group_num_need_pulls_for_decode_tp",
                    "_get_attention_group_num_key_value_heads",
                    "_get_tp_num_need_pulls",
                    "_get_remote_tp_ranks",
                    "_get_remote_ranks_for_req",
                }:
                    worker_methods.append(item)

    if tracker_node is None or not scheduler_methods or not worker_methods:
        raise AssertionError("Required classes/methods missing from mooncake_connector.py")

    sched_cls = ast.ClassDef(
        name="SchedulerHarness",
        bases=[],
        keywords=[],
        body=scheduler_methods,
        decorator_list=[],
    )
    worker_cls = ast.ClassDef(
        name="WorkerHarness",
        bases=[],
        keywords=[],
        body=worker_methods,
        decorator_list=[],
    )

    ns: dict[str, Any] = {
        "Any": Any,
        "BlockIds": Any,
        "OrderedDict": OrderedDict,
        "Request": Any,
        "RequestStatus": RequestStatus,
        " envs": types.SimpleNamespace(VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=5.0),
        "envs": types.SimpleNamespace(VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=5.0),
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
        "math": math,
        "np": np,
        "random": random,
        "string_to_int64_hash": string_to_int64_hash,
        "threading": threading,
        "time": time,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod_ast = ast.Module(body=[future, tracker_node, sched_cls, worker_cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod_ast), str(path), "exec"), ns)
    return ns["KVCacheTaskTracker"], ns["SchedulerHarness"], ns["WorkerHarness"]


def verify_tracker_and_scheduler(tracker_cls, sched_cls) -> None:
    tracker = tracker_cls()
    tracker.add_req_to_process("req-timeout")
    tracker.add_delayed_request("req-timeout", time.time() - 30.0)
    tracker.add_req_to_process("req-active")
    tracker.add_delayed_request("req-active", time.time())

    finished = tracker.get_and_clear_finished_requests()
    assert "req-timeout" in finished, (
        f"Expired delayed request 'req-timeout' was not freed by get_and_clear_finished_requests: {finished}"
    )
    assert "req-timeout" not in tracker.delayed_free_requests
    assert "req-timeout" not in tracker.reqs_to_process
    assert "req-active" in tracker.delayed_free_requests and "req-active" in tracker.reqs_to_process

    tracker.add_req_to_process("req-not-transfer")
    tracker.add_not_transfer_request("req-not-transfer")
    assert "req-not-transfer" not in tracker.reqs_to_process, "add_not_transfer_request must remove from reqs_to_process"
    finished2 = tracker.get_and_clear_finished_requests()
    assert "req-not-transfer" in finished2

    sched = sched_cls()
    sched.kv_cache_groups = [object()]
    sched._reqs_need_send = {}
    sched._reqs_need_recv = {}
    sched._reqs_in_batch = {"req-abort"}

    aborted_req = types.SimpleNamespace(
        request_id="req-abort",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
        prompt_token_ids=[1, 2, 3, 4],
        output_token_ids=[5],
    )
    delay_free, _ = sched.request_finished(aborted_req, ([1],))
    assert delay_free is False
    assert "req-abort" not in sched._reqs_need_send, "Aborted request leaked in _reqs_need_send"


def make_worker(worker_cls, prefill_tp: int, decode_tp: int, num_kv_heads: int, pp_size: int = 1, use_mla: bool = False):
    w = worker_cls()
    w._prefill_tp_size = prefill_tp
    w._decode_tp_size = decode_tp
    w.tp_size = decode_tp
    w._prefill_pp_size = pp_size
    w.num_key_value_heads = num_kv_heads
    w.use_sparse = False
    w.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            is_deepseek_mla=use_mla,
            get_total_num_kv_heads=lambda: num_kv_heads,
        )
    )
    if use_mla:
        w.tp_num_need_pulls = 1
    else:
        num_d = max(1, num_kv_heads // decode_tp)
        num_p = max(1, num_kv_heads // prefill_tp)
        w.tp_num_need_pulls = max(1, num_d // num_p)
    return w


def verify_asymmetric_tp(worker_cls) -> None:
    # 1. Prefill TP=4 -> Decode TP=8, num_kv_heads=8
    w_p4_d8 = make_worker(worker_cls, prefill_tp=4, decode_tp=8, num_kv_heads=8)
    pulls_p4_d8 = w_p4_d8._get_tp_num_need_pulls(4)
    assert pulls_p4_d8 == 1, f"Expected tp_num_need_pulls=1 for P4->D8 (kv_heads=8), got {pulls_p4_d8}"
    ranks_p4_d8 = w_p4_d8._get_remote_ranks_for_req("req-p4-d8", prefill_tp_size=4, num_key_value_heads=8)
    assert ranks_p4_d8 == [[0], [0], [1], [1], [2], [2], [3], [3]], (
        f"Unexpected P4->D8 (kv_heads=8) rank mapping: {ranks_p4_d8}"
    )

    # 2. Prefill TP=4 -> Decode TP=8, num_kv_heads=4 (Decode replicates KV heads)
    w_p4_d8_kv4 = make_worker(worker_cls, prefill_tp=4, decode_tp=8, num_kv_heads=4)
    ranks_p4_d8_kv4 = w_p4_d8_kv4._get_remote_ranks_for_req("req-p4-d8-kv4", prefill_tp_size=4, num_key_value_heads=4)
    assert ranks_p4_d8_kv4 == [[0], [0], [1], [1], [2], [2], [3], [3]], (
        f"Unexpected P4->D8 (kv_heads=4) rank mapping: {ranks_p4_d8_kv4}"
    )

    # 3. Prefill TP=8 -> Decode TP=4, num_kv_heads=8 (each Decode rank pulls 2 consecutive Prefill shards)
    w_p8_d4_kv8 = make_worker(worker_cls, prefill_tp=8, decode_tp=4, num_kv_heads=8)
    assert w_p8_d4_kv8._get_tp_num_need_pulls(8) == 2
    ranks_p8_d4_kv8 = w_p8_d4_kv8._get_remote_ranks_for_req("req-p8-d4-kv8", prefill_tp_size=8, num_key_value_heads=8)
    assert ranks_p8_d4_kv8 == [[0, 1], [2, 3], [4, 5], [6, 7]], (
        f"Unexpected P8->D4 (kv_heads=8) rank mapping: {ranks_p8_d4_kv8}"
    )

    # 4. Prefill TP=8 -> Decode TP=4, num_kv_heads=4 (Prefill replicates each KV head across 2 ranks)
    w_p8_d4_kv4 = make_worker(worker_cls, prefill_tp=8, decode_tp=4, num_kv_heads=4)
    assert w_p8_d4_kv4._get_tp_num_need_pulls(8) == 1
    ranks_p8_d4_kv4 = w_p8_d4_kv4._get_remote_ranks_for_req("req-p8-d4-kv4", prefill_tp_size=8, num_key_value_heads=4)
    assert len(ranks_p8_d4_kv4) == 4 and all(len(r) == 1 for r in ranks_p8_d4_kv4), (
        f"Expected 4 single-pull rank lists for P8->D4 (kv_heads=4), got {ranks_p8_d4_kv4}"
    )
    expected_head_groups = [{0, 1}, {2, 3}, {4, 5}, {6, 7}]
    for d_rank, r_list in enumerate(ranks_p8_d4_kv4):
        assert r_list[0] in expected_head_groups[d_rank], (
            f"Decode rank {d_rank} pulled from wrong KV head replica {r_list[0]}, expected in {expected_head_groups[d_rank]}"
        )


def main() -> None:
    try:
        check_diff_scope()
        tracker_cls, sched_cls, worker_cls = load_mooncake_components()
        verify_tracker_and_scheduler(tracker_cls, sched_cls)
        verify_asymmetric_tp(worker_cls)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "correctness_verified": False,
                    "error_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "validation_scope": "cpu_behavior",
                    "performance_status": "unmeasured",
                }
            )
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "validation_scope": "cpu_behavior",
                "performance_status": "unmeasured",
                "native_kernel_bound": None,
                "measured_npu_speedup": None,
                "speedup_ratio": None,
            }
        )
    )


if __name__ == "__main__":
    main()
