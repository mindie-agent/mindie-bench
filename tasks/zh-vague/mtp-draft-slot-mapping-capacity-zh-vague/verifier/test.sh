#!/bin/bash
set -eu
exec /tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python -I -B -c 'import ast, json, os
from pathlib import Path

root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
try:
    bt_text = (root / "vllm_ascend/worker/block_table.py").read_text(encoding="utf-8")
    prop_text = (root / "vllm_ascend/spec_decode/llm_base_proposer.py").read_text(encoding="utf-8")

    t_bt = ast.parse(bt_text)
    t_prop = ast.parse(prop_text)
    unp_bt = ast.unparse(t_bt)
    assert "num_speculative_tokens - 1" in unp_bt and "self.max_num_reqs" in unp_bt, "BlockTable missing (num_speculative_tokens - 1) * max_num_reqs slot_mapping formula"
    unp_prop = ast.unparse(t_prop)
    assert "num_speculative_tokens - 1" in unp_prop and "2 * self.runner.max_num_reqs" in unp_prop, "AscendSpecDecodeBaseProposer must size slot_mapping_group with (num_speculative_tokens - 1) * max_num_reqs while preserving 2 * max_num_reqs for seq_lens_group/query_start_loc_group"
    print(json.dumps({"reward": 1.0}))
except Exception as exc:
    print(json.dumps({"reward": 0.0, "failure_reason": str(exc)}))
    raise SystemExit(1)
'
