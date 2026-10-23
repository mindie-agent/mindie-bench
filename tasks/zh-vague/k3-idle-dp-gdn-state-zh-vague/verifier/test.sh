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
    r1_text = (root / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8")
    r310_text = (root / "vllm_ascend/_310p/model_runner_310p.py").read_text(encoding="utf-8")
    gdn_path = root / "vllm_ascend/ops/gdn_attn_builder.py"
    gdn_text = gdn_path.read_text(encoding="utf-8") if gdn_path.exists() else ""
    t1 = ast.parse(r1_text)
    t310 = ast.parse(r310_text)

    found_310 = False
    for node in ast.walk(t310):
        if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run":
            arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert "skip_gdn_state_update" in arg_names, "310p _dummy_run missing skip_gdn_state_update"
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and any(k.arg == "skip_gdn_state_update" for k in call.keywords):
                    found_310 = True
    assert found_310, "310p _dummy_run did not forward skip_gdn_state_update"

    exec_calls_skip = False
    dummy_has_skip = False
    build_has_skip = False
    for node in ast.walk(t1):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "_dummy_run":
                    if any(k.arg == "skip_gdn_state_update" and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords):
                        exec_calls_skip = True
        elif isinstance(node, ast.FunctionDef) and node.name == "_dummy_run":
            arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            if "skip_gdn_state_update" in arg_names:
                dummy_has_skip = True
        elif isinstance(node, ast.FunctionDef) and node.name == "_build_attention_metadata":
            arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            if "skip_gdn_state_update" in arg_names:
                build_has_skip = True
    combined = r1_text + "\n" + gdn_text
    has_gdn_noop_replace = (
        ("num_actual_tokens=0" in combined or "num_actual_tokens = 0" in combined)
        and ("max_query_len=0" in combined or "max_query_len = 0" in combined)
        and ("zeros_like" in combined or "zeros(" in combined or "zero_()" in combined)
        and ("ones_like" in combined or "fill_(-1)" in combined or "full(" in combined)
    )
    assert exec_calls_skip, "execute_model did not call _dummy_run(1, skip_gdn_state_update=True)"
    assert dummy_has_skip, "_dummy_run missing skip_gdn_state_update"
    assert build_has_skip, "_build_attention_metadata missing skip_gdn_state_update"
    assert has_gdn_noop_replace, "missing GDN noop metadata zeroing"
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
