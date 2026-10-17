"""Materialize one historical performance task and its private judge dependencies."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from difflib import unified_diff
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parents[2]
SOURCE = Path("/Users/maoxx241/code/vllm-ascend")
BASE = "f02da291ec0bdbe31af248deffa69f7a97e8f259"
FIX = "5312b32d4825cdbc092501e327fdc3e7366df9aa"


def materialize():
    if (ROOT / "soundness_receipt.json").exists():
        raise FileExistsError("preserve the existing soundness proof; choose a new task revision")
    oracle = ROOT / "oracle"
    oracle.mkdir(exist_ok=True)
    cases = [
        {"shape": [1, 37, 4, 128], "dtype": "bfloat16", "strided_gate": False, "activation": "sigmoid"},
        {"shape": [1, 193, 8, 128], "dtype": "bfloat16", "strided_gate": True, "activation": "sigmoid"},
    ]
    shapes = [
        hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for case in cases
    ]
    (oracle / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    changed_files = [
        "vllm_ascend/ops/kimi_kda.py",
        "vllm_ascend/ops/triton/kda/kda.py",
        "vllm_ascend/utils.py",
    ]
    patch = subprocess.check_output(
        ["git", "-C", str(SOURCE), "diff", "--binary", BASE, FIX, "--", *changed_files]
    )
    layernorm_path = "vllm_ascend/ops/layernorm.py"
    before = subprocess.check_output(
        ["git", "-C", str(SOURCE), "show", f"{BASE}:{layernorm_path}"], text=True
    )
    historical = subprocess.check_output(
        ["git", "-C", str(SOURCE), "show", f"{FIX}:{layernorm_path}"], text=True
    )
    insertion = """        # Avoid per-call Triton specialization/packing overhead in the
        # residual-free inference path; keep the general historical contract.
        if residual is None and not prenorm and self.weight is not None and self.bias is None:
            import torch_npu

            normalized, _ = torch_npu.npu_rms_norm(x, self.weight, self.eps)
            gate = torch.sigmoid(g) if self.activation == "sigmoid" else torch.nn.functional.silu(g)
            return normalized * gate
"""
    marker = (
        "class AscendFusedRMSNormGated(FusedRMSNormGated):",
        "    def forward_oot(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):\n",
    )
    prefix, target = historical.split(marker[0], 1)
    if target.count(marker[1]) != 1:
        raise ValueError("historical CustomOp signature changed")
    optimized = prefix + marker[0] + target.replace(marker[1], marker[1] + insertion, 1)
    patch += "".join(
        unified_diff(
            before.splitlines(keepends=True),
            optimized.splitlines(keepends=True),
            fromfile=f"a/{layernorm_path}",
            tofile=f"b/{layernorm_path}",
        )
    ).encode()
    (oracle / "gold_patch.diff").write_bytes(patch)
    runtime = oracle / "runtime/mindie_bench"
    for relative in ["__init__.py", "io.py", "models.py", "evaluation/performance.py"]:
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BENCH / "src/mindie_bench" / relative, target)
    (runtime / "evaluation/__init__.py").write_text("")
    control_sha = hashlib.sha256(b"torch-negative-control-ok\n").hexdigest()
    (ROOT / "task.toml").write_text(
        f'''schema_version = "1.0"
[task]
instance_id = "fused-rmsnorm-gate-dispatch-performance"
domain_category = "native_operator_performance"
difficulty = "hard"
evidence = "candidate"
cluster = "fused-rmsnorm-gate"
role = "eval"
[task.source]
repository = "vllm-project/vllm-ascend"
base_commit = "{BASE}"
fix_commit = "{FIX}"
task_type = "performance"
gold_patch_scope = "Historical 5312 dispatch and input-preservation repair plus a benchmark-specific native Ascend RMSNorm host-path improvement. This reference is not the unmodified upstream commit."
gold_patch_provenance = "5312b32d plus measured residual-free native NPU RMSNorm/gate path; original historical reference failed the unchanged warm-service gate."
validation_scope = "One real Ascend NPU; actual CustomOp dispatch, numerical checks, private shapes, buffer rotation, ACLGraph, allocator peak and warm eight-norm-step HTTP service. No full-model throughput claim."
memory_budget_scope = "Tiny eight-step service tensor scratch only: 16MiB baseline reserve, 16MiB extra candidate allowance, 32MiB total. No real KV cache exists in this service."
[mindie_knowledge]
corpus_id = "ascend"
knowledge_gradient = "near_exact"
same_origin_task_hashes = ["0bc94d24162470fe8ce3fc189b7eb239aa868d074ff53c78706cad0c5c608243"]
analogical_task_hashes = ["b22cc1d74932da059115f71fa647370ce76ecfb54317c312e0b3a117cae86d49"]
target_files = ["vllm_ascend/utils.py", "vllm_ascend/ops/layernorm.py", "vllm_ascend/ops/triton/kda/kda.py"]
[budget]
max_turns = 40
max_npu_runs = 8
timeout_sec = 1800
verifier_timeout_sec = 600
max_output_bytes = 33554432
[environment.l3]
repo_path = "{SOURCE}"
base_commit = "{BASE}"
npu_model = "Ascend910"
npu_count = 1
[oracle]
[[oracle.probes]]
name = "actual-torch-comparator-control"
command = ["python3", "-I", "-B", "{{oracle}}/verifier/torch_control.py"]
expected_returncode = 0
stdout_sha256 = "{control_sha}"
[performance]
private_shape_sha256 = {json.dumps(shapes)}
min_buffers = 4
samples_per_shape = 36
require_aclgraph_capture = true
require_workspace_measurement = true
require_e2e_warm_server = true
min_speedup_ratio = 1.05
min_e2e_warm_server_speedup = 1.01
max_workspace_delta_bytes = 16777216
'''
    )


if __name__ == "__main__":
    materialize()
