#!/bin/bash
set -eu
if [ -n "${MINDIE_BENCH_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_PYTHON}" ]; then
    PYTHON_BIN="${MINDIE_BENCH_PYTHON}"
elif [ -n "${MINDIE_BENCH_CPU_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_CPU_PYTHON}" ]; then
    PYTHON_BIN="${MINDIE_BENCH_CPU_PYTHON}"
elif [ -x "/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python" ]; then
    PYTHON_BIN="/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python"
else
    PYTHON_BIN="python3"
fi
exec "${PYTHON_BIN}" -I -B -c 'import ast, json, os, types
from pathlib import Path
import torch

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
try:
    dsa_text = (root / "vllm_ascend/attention/dsa_v1.py").read_text(encoding="utf-8")
    attn_utils_text = (root / "vllm_ascend/worker/v2/attn_utils.py").read_text(encoding="utf-8")
    default_text = (root / "vllm_ascend/worker/v2/model_states/default.py").read_text(encoding="utf-8")
    spec_text = (root / "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py").read_text(encoding="utf-8")

    t_dsa = ast.parse(dsa_text)
    found_kv_shape = False
    for node in ast.walk(t_dsa):
        if isinstance(node, ast.FunctionDef) and node.name == "get_kv_cache_shape":
            args = [a.arg for a in node.args.args]
            assert "cache_dtype_str" in args, f"get_kv_cache_shape args {args} missing cache_dtype_str"
            found_kv_shape = True
    assert found_kv_shape

    t_au = ast.parse(attn_utils_text)
    found_au = False
    for node in ast.walk(t_au):
        if isinstance(node, ast.FunctionDef) and node.name == "build_attn_metadata":
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert "is_prefilling" in args, "build_attn_metadata missing is_prefilling param"
            body_src = "\n".join(ast.unparse(stmt) for stmt in node.body)
            assert "is_prefilling" in body_src, "build_attn_metadata body did not pass is_prefilling"
            found_au = True
    assert found_au

    t_def = ast.parse(default_text)
    found_def = False
    for node in ast.walk(t_def):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_attn":
            unparsed = ast.unparse(node)
            assert ("is_prefilling_np" in unparsed or "is_prefilling_np" in attn_utils_text) and "is_prefilling=" in unparsed, "prepare_attn missing is_prefilling propagation"
            found_def = True
    assert found_def

    t_spec = ast.parse(spec_text)
    found_fac = False
    found_upd = False
    for node in ast.walk(t_spec):
        if isinstance(node, ast.FunctionDef) and node.name == "build_draft_attn_metadata_factory":
            found_fac = True
        elif isinstance(node, ast.FunctionDef) and node.name == "_ascend_update_seq_lens":
            ns = {"torch": torch, "Any": object}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<spec>", "exec"), ns)
            fn = ns["_ascend_update_seq_lens"]
            s_dsa = types.SimpleNamespace(attn_architecture="DSA")
            m_dsa = {"k": types.SimpleNamespace(seq_lens=torch.tensor([5, 6]))}
            fn(s_dsa, m_dsa)
            assert torch.equal(m_dsa["k"].seq_lens, torch.tensor([5, 6])), "DSA seq_lens should not increment"
            s_other = types.SimpleNamespace(attn_architecture="MLA")
            m_other = {"k": types.SimpleNamespace(seq_lens=torch.tensor([5, 6]))}
            fn(s_other, m_other)
            assert torch.equal(m_other["k"].seq_lens, torch.tensor([6, 7])), "Non-DSA seq_lens should increment"
            found_upd = True
    assert found_fac and found_upd
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
