#!/usr/bin/env python3
"""Verifier for vllm-ascend-gdn-dspark-inplace-stomp-watch-zh."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

EXPECTED_REPRO_SHA256 = "a331c1cd97b2ee027c857585a26b1b0949ab0eaa3d20c330fad5eafc247d9b5b"


def _fail(reason: str, error_type: str = "VerificationAssertionError") -> int:
    payload = {
        "reward": 0.0,
        "correctness_verified": False,
        "verification_status": "failed",
        "validation_scope": "cpu_behavior",
        "performance_status": "unmeasured",
        "error_type": error_type,
        "failure_reason": reason,
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 1


def _pass() -> int:
    payload = {
        "reward": 1.0,
        "correctness_verified": True,
        "verification_status": "completed",
        "validation_scope": "cpu_behavior",
        "performance_status": "unmeasured",
        "native_kernel_bound": None,
        "measured_npu_speedup": None,
        "speedup_ratio": None,
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    workspace = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
    repro_path = workspace / "tests" / "repro_gdn_dspark_step.py"
    if not repro_path.is_file():
        return _fail("Missing tests/repro_gdn_dspark_step.py in workspace", "MissingTargetFileError")

    # 1. Anti-tampering check on tests/repro_gdn_dspark_step.py
    if _sha256(repro_path) != EXPECTED_REPRO_SHA256:
        return _fail(
            "Forbidden tampering detected: tests/repro_gdn_dspark_step.py was modified",
            "ForbiddenTamperingError",
        )

    if (workspace / ".git").exists():
        diff_proc = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--name-only", "HEAD", "--", "tests/repro_gdn_dspark_step.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.stdout.strip():
            return _fail(
                "Forbidden tampering in git diff: tests/repro_gdn_dspark_step.py",
                "ForbiddenTamperingError",
            )

    # 2. Import and run reproducer + independent randomized multi-step state integrity checks
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))

    try:
        import tests.repro_gdn_dspark_step as repro
    except Exception as exc:
        return _fail(f"Failed to import tests.repro_gdn_dspark_step: {exc}", type(exc).__name__)

    try:
        res = repro.run_two_step_gdn_dspark()
    except Exception as exc:
        return _fail(f"run_two_step_gdn_dspark raised {type(exc).__name__}: {exc}", type(exc).__name__)

    if not res.get("block_table_intact"):
        return _fail(
            f"live_block_table was corrupted during Step 1: got {res['live_block_table'].tolist()}, "
            f"expected {res['expected_block_table'].tolist()}"
        )
    if not res.get("query_start_loc_intact"):
        return _fail(
            f"live_query_start_loc was corrupted during Step 1: got {res['live_query_start_loc'].tolist()}, "
            f"expected {res['expected_query_start_loc'].tolist()}"
        )
    if not res.get("seq_lens_intact"):
        return _fail(
            f"live_seq_lens was corrupted during Step 1: got {res['live_seq_lens'].tolist()}, "
            f"expected {res['expected_seq_lens'].tolist()}"
        )
    if not res.get("step2_outputs_match"):
        return _fail("Step 2 sequence state and GDN state block outputs did not match expected values")

    # 3. Verify zero storage aliasing between caller live tensors and GDN/DSpark internal outputs
    live_bt_ptr = int(res["live_block_table"].untyped_storage().data_ptr())
    live_qsl_ptr = int(res["live_query_start_loc"].untyped_storage().data_ptr())
    live_sl_ptr = int(res["live_seq_lens"].untyped_storage().data_ptr())

    gdn_cache_indices = res["gdn_meta_step1"].spec_decode_metadata.spec_causal_conv1d.cache_indices
    gdn_actual_seq_lengths = res["gdn_meta_step1"].spec_decode_metadata.actual_seq_lengths
    dspark_bt = res["proposer"]._per_group_block_table_buffers[0]
    dspark_sl = res["dspark_cad_step1"].seq_lens

    if int(gdn_actual_seq_lengths.untyped_storage().data_ptr()) == live_qsl_ptr:
        return _fail("gdn_meta.spec_decode_metadata.actual_seq_lengths still aliases live_query_start_loc storage")
    if int(dspark_sl.untyped_storage().data_ptr()) == live_sl_ptr:
        return _fail("dspark_cad.seq_lens still aliases live_seq_lens storage")

    # Verify gdn_actual_seq_lengths values are [0, 4, 4] and gdn_cache_indices has shape (2, 4)
    if gdn_actual_seq_lengths.tolist() != [0, 4, 4]:
        return _fail(f"Unexpected gdn_actual_seq_lengths: {gdn_actual_seq_lengths.tolist()}, expected [0, 4, 4]")
    if gdn_cache_indices.tolist() != [[10, 11, 12, 13], [20, 21, 22, 23]]:
        return _fail(
            f"Unexpected gdn_cache_indices: {gdn_cache_indices.tolist()}, expected [[10, 11, 12, 13], [20, 21, 22, 23]]"
        )
    if dspark_sl.tolist() != [17, 26]:
        return _fail(f"Unexpected dspark_cad.seq_lens: {dspark_sl.tolist()}, expected [17, 26]")

    # 4. Also verify mindie_debugger StorageGraph & MemoryWatchpoint when available
    debugger_repo = Path("/Users/maoxx241/code/debugger")
    if debugger_repo.is_dir() and str(debugger_repo) not in sys.path:
        sys.path.insert(0, str(debugger_repo))
    try:
        from mindie_debugger.precision import MemoryWatchpoint, StorageGraph

        graph = StorageGraph.from_state(
            live_block_table=res["live_block_table"],
            live_query_start_loc=res["live_query_start_loc"],
            live_seq_lens=res["live_seq_lens"],
            gdn_cache_indices=gdn_cache_indices,
            gdn_actual_seq_lengths=gdn_actual_seq_lengths,
            dspark_block_table=dspark_bt,
            dspark_seq_lens=dspark_sl,
        )
        if graph.shares_storage("live_query_start_loc", "gdn_actual_seq_lengths"):
            return _fail("StorageGraph detected shared storage between live_query_start_loc and gdn_actual_seq_lengths")
        if graph.shares_storage("live_seq_lens", "dspark_seq_lens"):
            return _fail("StorageGraph detected shared storage between live_seq_lens and dspark_seq_lens")

        wp = MemoryWatchpoint(
            must_preserve={
                "block_table": res["live_block_table"],
                "query_start_loc": res["live_query_start_loc"],
                "seq_lens": res["live_seq_lens"],
            }
        )
        chk = wp.check(step=1, module_name="gdn_dspark_step", op_name="post")
        if chk["changed"] or wp.violation_count > 0:
            return _fail(f"MemoryWatchpoint detected unexpected mutation: {wp.violations}")
    except ImportError:
        pass

    return _pass()


if __name__ == "__main__":
    raise SystemExit(main())
