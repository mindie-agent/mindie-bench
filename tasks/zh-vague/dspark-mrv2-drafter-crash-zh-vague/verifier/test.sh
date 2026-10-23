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
exec "${PYTHON_BIN}" -I -B -c 'import ast, json, os
from pathlib import Path

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
try:
    model_text = (root / "vllm_ascend/models/deepseek_v4/model.py").read_text(encoding="utf-8")
    dsa_text = (root / "vllm_ascend/attention/dsa_v1.py").read_text(encoding="utf-8")

    t_dsa = ast.parse(dsa_text)
    found_dsa = False
    for node in ast.walk(t_dsa):
        if isinstance(node, ast.FunctionDef) and node.name == "build_for_drafting":
            unp = ast.unparse(node)
            assert "self.block_size" not in unp and "self.storage_block_size" in unp, "dsa_v1 build_for_drafting still uses self.block_size instead of self.storage_block_size"
            found_dsa = True
    assert found_dsa

    t_mod = ast.parse(model_text)
    unp_mod = ast.unparse(t_mod)
    assert "use_eagle()" in unp_mod and "uses_draft_model()" in unp_mod and "is_last_rank" in unp_mod and "_mtp_hidden_buffer" in unp_mod, "deepseek_v4/model.py missing PP last-rank + use_eagle/uses_draft_model _mtp_hidden_buffer initialization"
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
