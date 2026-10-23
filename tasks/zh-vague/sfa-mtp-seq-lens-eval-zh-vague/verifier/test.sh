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
    sfa_text = (root / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8")
    t_sfa = ast.parse(sfa_text)
    found_build = False
    for node in ast.walk(t_sfa):
        if isinstance(node, ast.FunctionDef) and node.name == "_build":
            unp = ast.unparse(node)
            assert "draft_index" in unp and "spec_actual_seq_lengths_query[draft_index - 1]" in unp and "spec_actual_seq_lengths_key[draft_index - 1]" in unp, "_build missing spec_actual_seq_lengths_query/key[draft_index - 1] selection"
            found_build = True
    assert found_build
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
