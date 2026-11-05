# [Role] Independent CPU behavior oracle for Ascend top-k & top-p logits filtering (Stream C - V2).
# [Invariants] Only executed tensor assertions establish success; no NPU hardware required.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Parses candidate AST and executes _apply_top_k_top_p_pytorch on CPU; emits JSON only.

from __future__ import annotations

import ast
import json
import os
import subprocess
import types
from pathlib import Path
from typing import Any

import torch

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
TARGET_REL = "vllm_ascend/sample/sampler.py"
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
    disallowed = changed - ALLOWED_FILES
    if disallowed:
        raise AssertionError(f"Modified files outside allowed scope {sorted(ALLOWED_FILES)}: {sorted(disallowed)}")


def load_sampler_functions():
    path = ROOT / TARGET_REL
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]

    state = {"enable_reduce_sample": False}

    class FakeTPGroup:
        rank_in_group = 0

        @staticmethod
        def all_gather(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
            return torch.cat([t, t - 0.5], dim=dim)

    ns: dict[str, Any] = {
        "torch": torch,
        "get_ascend_config": lambda: types.SimpleNamespace(enable_reduce_sample=state["enable_reduce_sample"]),
        "get_tp_group": lambda: FakeTPGroup(),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod_ast = ast.Module(body=[future, *wanted_nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod_ast), str(path), "exec"), ns)
    if "_apply_top_k_top_p_pytorch" not in ns:
        raise AssertionError("vllm_ascend/sample/sampler.py must define _apply_top_k_top_p_pytorch")
    return ns["_apply_top_k_top_p_pytorch"], state


def verify_non_reduce_mode(apply_fn, state) -> None:
    state["enable_reduce_sample"] = False

    # Case 1: Mixed batch with k <= 0 (disabled), k >= vocab_size (disabled), and valid k=2;
    # plus a peaked distribution where top-1 prob > p (0.99 > 0.6).
    logits = torch.tensor(
        [
            [10.0, 1.0, 0.5, 0.0, -1.0, -2.0],  # top-1 prob ~ 0.9998 > p=0.5, k=-1 (disabled)
            [3.0, 2.5, 2.0, 0.0, -1.0, -2.0],   # k=0 (disabled), p=1.0 (disabled)
            [4.0, 3.5, 1.0, 0.5, -1.0, -2.0],   # k=100 (disabled > vocab_size=6), p=0.95
            [5.0, 4.8, 4.6, 0.0, -1.0, -2.0],   # k=2, p=0.3 (top-1 after k=2 exceeds 0.3 -> keep top-1)
        ],
        dtype=torch.float32,
    )
    k = torch.tensor([-1, 0, 100, 2], dtype=torch.int64)
    p = torch.tensor([0.5, 1.0, 0.95, 0.3], dtype=torch.float32)

    out = apply_fn(logits.clone(), k, p)
    probs = out.softmax(dim=-1, dtype=torch.float32)
    assert not torch.isnan(probs).any(), f"Softmax produced NaN after top_k/top_p filtering: {probs}"

    # Row 0: top-1 token alone exceeds p=0.5; only token 0 should remain finite
    finite_0 = torch.isfinite(out[0]).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert finite_0 == [0], f"Row 0 expected only top-1 token [0] kept, got {finite_0}"

    # Row 1: k=0 (disabled) and p=1.0 (disabled); all 6 tokens must remain finite
    finite_1 = torch.isfinite(out[1]).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert finite_1 == [0, 1, 2, 3, 4, 5], f"Row 1 expected all tokens kept, got {finite_1}"

    # Row 2: k=100 (>6, disabled), p=0.95; tokens 0 and 1 together exceed 0.95
    finite_2 = torch.isfinite(out[2]).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert 0 in finite_2 and len(finite_2) >= 2, f"Row 2 expected top tokens kept, got {finite_2}"

    # Row 3: k=2 first restricts to tokens {0, 1}, then p=0.3 keeps at least token 0
    finite_3 = torch.isfinite(out[3]).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert finite_3 == [0], f"Row 3 expected top-1 token [0] kept under k=2, p=0.3, got {finite_3}"


def verify_reduce_sample_mode(apply_fn, state) -> None:
    state["enable_reduce_sample"] = True
    logits = torch.tensor(
        [
            [12.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [4.0, 3.9, 3.8, 1.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor([0, 2], dtype=torch.int64)
    p = torch.tensor([0.4, 0.99], dtype=torch.float32)

    gathered_vals, gathered_idx = apply_fn(logits.clone(), k, p, top_k=0)
    probs = gathered_vals.softmax(dim=-1, dtype=torch.float32)
    assert not torch.isnan(probs).any(), f"Reduce sample softmax produced NaN: {probs}"
    assert torch.isfinite(gathered_vals[0]).sum().item() >= 1, "Row 0 in reduce mode must retain at least 1 token"
    assert torch.isfinite(gathered_vals[1]).sum().item() == 2, (
        f"Row 1 in reduce mode with k=2, p=0.99 must retain 2 tokens, got {torch.isfinite(gathered_vals[1]).sum().item()}"
    )


def main() -> None:
    try:
        check_diff_scope()
        apply_fn, state = load_sampler_functions()
        verify_non_reduce_mode(apply_fn, state)
        verify_reduce_sample_mode(apply_fn, state)
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
