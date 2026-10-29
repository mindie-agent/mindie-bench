"""Reproduce zero-model G4 soundness on historical baseline and honest gold.

[Role] Run independent verifier subprocesses for baseline/negative/gold/helper.
[Invariants] Preserve original historical trials; compare actual process verdicts.
[Timeouts & Locks] Each Git command has 20 s; verifier has 45 s plus kill/drain.
[Side-Effect Policy] Writes owned evidence and current task soundness receipts.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

FAMILIES = ("dsv4-sparse-mla-qli-fusion-rsi-zh", "triton-spatial-norm-dim-merge-rsi-zh")
BASE = "264445c1379b7e5c00df06752bf6a9c604c2c3cb"
SOURCE_REPO = Path("/Users/maoxx241/code/vllm-ascend")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)
        and p.name != "soundness_receipt.json"
        and p.suffix not in (".pyc", ".pyo")
    }


def prepare_base(workspace, family):
    target = (
        "vllm_ascend/models/deepseek_v4/sparse_mla_decode.py"
        if family.startswith("dsv4")
        else "vllm_ascend/ops/spatial_group_norm_reduce.py"
    )
    paths = [target]
    if family.startswith("dsv4"):
        paths.append("vllm_ascend/models/deepseek_v4/_ascend_qli_runtime.py")
    for name in paths:
        data = subprocess.run(
            ["git", "show", f"{BASE}:{name}"],
            cwd=SOURCE_REPO,
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout
        dest = workspace / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return workspace / target


def variant(target, family, stage):
    fn = "sparse_mla_decode_step" if family.startswith("dsv4") else "spatial_group_norm_reduce"
    if stage == "negative_control":
        arms = (
            ["full_candidate", "minus_q_fusion", "minus_view_alias", "minus_device_guard"]
            if family.startswith("dsv4")
            else [
                "full_candidate",
                "minus_dim_merge",
                "minus_block_scaling",
                "minus_multibuffer_cast_guard",
            ]
        )
        target.write_text(
            "import torch\n"
            f"def {fn}(*args, **kwargs):\n    return torch.zeros_like(args[0]), {{'correctness_ok': True, 'tpot_ms': 0.001}}\n"
            "def select_optimal_block_size(n, dtype, live=3):\n    return 512 if n <= 512 else (4096 if dtype == torch.float32 else 8192)\n"
            "def compute_persistent_grid(n, cores=48):\n    return (min(n, cores),)\n"
            f"def run_ablation_suite():\n    return {{arm: {{'correctness_ok': True, 'tpot_ms': 0.001}} for arm in {arms!r}}}\n"
        )
    elif stage == "constant_return":
        source = target.read_text()
        if family.startswith("dsv4"):
            source = source.replace("return attn_out, {", "return torch.zeros_like(attn_out), {")
        else:
            source = source.replace("return out, {", "return torch.zeros_like(out), {")
        target.write_text(source)
    elif stage == "wrong_guard_lengths":
        source = target.read_text().replace(
            "(topk_indices >= 0).sum(dim=-1).to(torch.int32)",
            "torch.clamp(seqused_k.to(torch.int32), min=0, max=topk)",
        )
        target.write_text(source)
    elif stage == "gold_bf16_inplace":
        source = target.read_text()
        source = source.replace(
            "sum_fp32 = sum_fp32.to(torch.bfloat16).to(torch.float32)",
            "sum_bf16 = torch.zeros_like(sum_fp32, dtype=torch.bfloat16)\n        sum_bf16.add_(sum_fp32)\n        sum_fp32 = sum_bf16.float()",
        )
        source = source.replace(
            "sq_fp32 = sq_fp32.to(torch.bfloat16).to(torch.float32)",
            "sq_bf16 = torch.zeros_like(sq_fp32, dtype=torch.bfloat16)\n        sq_bf16.add_(sq_fp32)\n        sq_fp32 = sq_bf16.float()",
        )
        target.write_text(source)
    elif stage == "gold_suite_helper":
        # The public default operator remains correct; the suite controls an
        # extracted helper. The oracle must follow the runtime tensor boundary.
        source = target.read_text().replace(f"{fn}(", "_controlled_tensor_entry(")
        tree = ast.parse(source)
        original = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_controlled_tensor_entry"
        )
        public = copy.deepcopy(original)
        public.name = fn
        retained = [
            (arg, default)
            for arg, default in zip(public.args.kwonlyargs, public.args.kw_defaults)
            if not arg.arg.startswith("enable_")
        ]
        public.args.kwonlyargs = [arg for arg, _ in retained]
        public.args.kw_defaults = [default for _, default in retained]
        public.body = [
            ast.Return(
                ast.Call(
                    func=ast.Name(id="_controlled_tensor_entry", ctx=ast.Load()),
                    args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in public.args.args],
                    keywords=[
                        ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load()))
                        for arg in public.args.kwonlyargs
                    ],
                )
            )
        ]
        public.decorator_list = []
        target.write_text(source + "\n\n" + ast.unparse(ast.fix_missing_locations(public)) + "\n")
    elif stage == "gold_helper":
        source = target.read_text().replace(f"def {fn}(", "def _gold_implementation(", 1)
        source += (
            "\nclass _GoldDelegate:\n"
            "    @staticmethod\n"
            "    def invoke(*args, **kwargs):\n"
            "        return _gold_implementation(*args, **kwargs)\n\n"
            f"def {fn}(*args, **kwargs):\n"
            "    return _GoldDelegate.invoke(*args, **kwargs)\n"
        )
        target.write_text(source)


def run(root, python, output):
    output.mkdir(parents=True, exist_ok=True)
    proofs = []
    for family in FAMILIES:
        for name in (family, "l3-" + family):
            task = root / "tasks/zh-vague" / name
            evidence = output / name
            evidence.mkdir(parents=True, exist_ok=True)
            results = {}
            stages = [
                ("baseline", 0.0),
                ("negative_control", 0.0),
                ("constant_return", 0.0),
                ("gold", 1.0),
                ("gold_helper", 1.0),
                ("gold_suite_helper", 1.0),
            ]
            stages.append(
                ("wrong_guard_lengths", 0.0) if family.startswith("dsv4") else ("gold_bf16_inplace", 1.0)
            )
            for stage, expected in stages:
                with tempfile.TemporaryDirectory(prefix="g4-soundness-") as tmp:
                    workspace = Path(tmp)
                    target = prepare_base(workspace, family)
                    if stage != "baseline":
                        subprocess.run(
                            ["git", "apply", "--no-index", str(task / "oracle/gold_patch.diff")],
                            cwd=workspace,
                            capture_output=True,
                            check=True,
                            timeout=20,
                        )
                    variant(target, family, stage)
                    input_hashes = hashes(workspace)
                    (evidence / f"{stage}.candidate.py").write_bytes(target.read_bytes())
                    start = time.monotonic()
                    env = {
                        **os.environ,
                        "MINDIE_BENCH_WORKSPACE": str(workspace),
                        "MINDIE_BENCH_PYTHON": str(python),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    }
                    result = subprocess.run(
                        ["bash", str(task / "verifier/test.sh")],
                        cwd=workspace,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=45,
                    )
                    elapsed = time.monotonic() - start
                    (evidence / f"{stage}.stdout").write_text(result.stdout)
                    (evidence / f"{stage}.stderr").write_text(result.stderr)
                    verdict = json.loads(result.stdout.strip().splitlines()[-1])
                    results[stage] = {
                        "expected_reward": expected,
                        "reward": verdict.get("reward"),
                        "exit_code": result.returncode,
                        "elapsed_seconds": elapsed,
                        "verdict": verdict,
                        "input_sha256": digest(input_hashes),
                        "input_files": input_hashes,
                        "candidate_path": str((evidence / f"{stage}.candidate.py").relative_to(root)),
                        "stdout_path": str((evidence / f"{stage}.stdout").relative_to(root)),
                        "stderr_path": str((evidence / f"{stage}.stderr").relative_to(root)),
                    }
                    print(
                        f"{name} {stage}: reward={verdict.get('reward')} exit={result.returncode} {verdict.get('failure_reason', '')}",
                        flush=True,
                    )
            complete = all(
                item["reward"] == item["expected_reward"]
                and item["verdict"].get("verification_status") != "unsupported"
                and (item["exit_code"] == 0) == (item["expected_reward"] == 1.0)
                and item["verdict"].get("error_type")
                not in ("ModuleNotFoundError", "ImportError", "NameError")
                for item in results.values()
            )
            receipt = {
                "schema_version": "2.0",
                "status": "completed" if complete else "failed",
                "tier": "l3" if name.startswith("l3-") else "l2",
                "execution_scope": "controlled_cpu_tensor_and_ablation_contract",
                "source_commit": BASE,
                "model_calls": 0,
                "before_reward": results["baseline"]["reward"],
                "negative_control_reward": results["negative_control"]["reward"],
                "after_reward": results["gold"]["reward"],
                "helper_reward": results["gold_helper"]["reward"],
                "task_sha256": digest(hashes(task)),
                "verifier_sha256": digest(hashes(task / "verifier")),
                "verifier_files": hashes(task / "verifier"),
                "python": str(python),
                "native_kernel_bound": None,
                "measured_npu_speedup": None,
                "speedup_ratio": None,
                "performance_status": "unmeasured",
                "cases": results,
                "errors": [] if complete else ["soundness reward mismatch"],
            }
            (task / "soundness_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            (evidence / "soundness.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            proofs.append(receipt)
    (output / "summary.json").write_text(json.dumps(proofs, indent=2, sort_keys=True) + "\n")
    return all(p["status"] == "completed" for p in proofs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", type=Path, default=Path("/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/integrity-20261029/g4-soundness-v3")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(0 if run(root, args.python.absolute(), args.output.absolute()) else 1)
