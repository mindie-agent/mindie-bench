# [Role] Independent CPU behavior oracle for ModelSlim safetensors header preflight & VRAM estimation (Stream C - C2).
# [Invariants] Only executed behavioral assertions establish success; zero weight payload loading.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources and tests against workspace + temp safetensors headers; emits JSON only.

from __future__ import annotations

import ast
import glob
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
TARGET_REL = "vllm_ascend/quantization/modelslim_config.py"
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


def load_modelslim_preflight_functions():
    path = ROOT / TARGET_REL
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
            wanted_nodes.append(node)

    ns: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "glob": glob,
        "json": json,
        "logger": __import__("types").SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
        "math": math,
        "os": os,
        "struct": struct,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod_ast = ast.Module(body=[future, *wanted_nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod_ast), str(path), "exec"), ns)
    for required in (
        "read_safetensors_headers",
        "estimate_modelslim_checkpoint_vram_bytes",
        "preflight_modelslim_safetensors_headers",
    ):
        if required not in ns:
            raise AssertionError(f"{TARGET_REL} is missing required function: {required}")
    return (
        ns["read_safetensors_headers"],
        ns["estimate_modelslim_checkpoint_vram_bytes"],
        ns["preflight_modelslim_safetensors_headers"],
    )


def write_safetensors_header_only(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    dtype_bytes = {"I8": 1, "U8": 1, "F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F64": 8}
    header_obj: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        n = math.prod(shape) if shape else 1
        nbytes = int(n) * dtype_bytes[dtype]
        header_obj[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(raw) % 8)) % 8
    raw += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)


def verify_workspace_checkpoint(read_fn, vram_fn, preflight_fn) -> None:
    ckpt_dir = ROOT / "checkpoints/dsv4_w8a8_mixed"
    report = preflight_fn(ckpt_dir, strict=True)
    expected_fallbacks = [
        "model.layers.0.mlp.shared_experts.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]
    assert report["fallback_float_layers"] == expected_fallbacks, (
        f"Expected fallback_float_layers={expected_fallbacks}, got {report['fallback_float_layers']}"
    )
    resolved = report["resolved_quant_description"]
    for key in expected_fallbacks:
        assert resolved[key] == "FLOAT", f"Expected {key} resolved to FLOAT, got {resolved.get(key)}"
    assert resolved["model.layers.0.self_attn.q_proj.weight"] == "W8A8_DYNAMIC"
    assert resolved["model.layers.0.mlp.experts.0.gate_proj.weight"] == "W8A8"
    assert report["invalid_quant_layers"] == {}, f"Expected no invalid layers, got {report['invalid_quant_layers']}"

    expected_vram = (
        1024 * 256 * 2
        + 512 * 256 * 1
        + 512 * 1 * 4
        + 256 * 256 * 1
        + 256 * 1 * 4
        + 256 * 256 * 1
        + 256 * 1 * 4
        + 256 * 512 * 2
        + 256 * 512 * 2
        + 512 * 256 * 1
        + 512 * 8
        + 512 * 256 * 1
        + 512 * 1 * 2
        + 256 * 2
    )
    assert report["estimated_vram_bytes"] == expected_vram, (
        f"Expected estimated_vram_bytes={expected_vram}, got {report['estimated_vram_bytes']}"
    )
    assert vram_fn(ckpt_dir) == expected_vram


def verify_corrupted_scale_detection(preflight_fn) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        quant_desc = {
            "model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            "model.layers.0.self_attn.k_proj.weight": "W8A8_DYNAMIC",
            "model.layers.0.mlp.down_proj.weight": "W8A8",
        }
        (tmp_path / "quant_model_description.json").write_text(json.dumps(quant_desc), encoding="utf-8")
        write_safetensors_header_only(
            tmp_path / "model-00001-of-00001.safetensors",
            {
                "model.layers.0.self_attn.q_proj.weight": ("I8", [512, 256]),
                "model.layers.0.self_attn.k_proj.weight": ("I8", [512, 256]),
                "model.layers.0.self_attn.k_proj.weight_scale": ("F32", [128, 1]),
                "model.layers.0.mlp.down_proj.weight": ("F16", [256, 512]),
            },
        )
        non_strict = preflight_fn(tmp_path, strict=False)
        assert non_strict["fallback_float_layers"] == ["model.layers.0.mlp.down_proj.weight"]
        assert non_strict["invalid_quant_layers"] == {
            "model.layers.0.self_attn.q_proj.weight": "missing_scale",
            "model.layers.0.self_attn.k_proj.weight": "scale_shape_mismatch",
        }, f"Unexpected invalid_quant_layers: {non_strict['invalid_quant_layers']}"

        raised = False
        try:
            preflight_fn(tmp_path, strict=True)
        except ValueError:
            raised = True
        assert raised, "strict=True must raise ValueError when quantized I8 layers have missing or mismatched scales"


def main() -> None:
    try:
        check_diff_scope()
        read_fn, vram_fn, preflight_fn = load_modelslim_preflight_functions()
        verify_workspace_checkpoint(read_fn, vram_fn, preflight_fn)
        verify_corrupted_scale_detection(preflight_fn)
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
