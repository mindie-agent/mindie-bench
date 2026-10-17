#!/bin/bash
set -eu
python3 -I - <<'PY_VERIFIER'
import json
import ast
import pathlib
import types

import torch
import torch_npu  # noqa: F401 - registers the real NPU backend.


def source_functions(relative, namespace):
    source = pathlib.Path(relative)
    tree = ast.parse(source.read_text(), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    focused = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(focused), str(source), "exec"), namespace)
    return namespace


try:
    DEVICE = "npu:0"
    if not torch.npu.is_available():
        raise RuntimeError("NPU runtime unavailable")
    if torch.npu.device_count() != 1:
        raise RuntimeError("expected exactly one assigned visible NPU")
    class SingleRankGroup:
        world_size = 1
        rank_in_group = 0
        def all_gather(self, value, dim=-1):
            return value

    reduced = True
    state = {"torch": torch, "get_tp_group": lambda: SingleRankGroup(),
             "get_ascend_config": lambda: types.SimpleNamespace(enable_reduce_sample=reduced)}
    source_functions("vllm_ascend/sample/sampler.py", state)
    function = state["_apply_top_k_top_p_pytorch"]
    cases = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        cpu = torch.tensor([[0.2, -0.7, 2.3, 0.8, -1.4, 1.1, -0.2, 1.7],
                            [1.6, 0.3, -1.1, 2.4, -0.5, 0.9, -1.7, 0.1],
                            [-0.9, 1.2, 0.4, -1.6, 2.1, -0.1, 0.8, 1.5]], dtype=dtype)
        scenarios = (
            (None, None, None),
            (None, None, 4096),
            (None, [0.55, 0.75, 0.9], None),
            ([1, 3, 8], None, 4096),
            ([2, 4, 8], [0.55, 0.75, 0.95], 8),
        )
        for kvals, pvals, cap in scenarios:
            reduced = True
            logits = cpu.to(DEVICE)
            k = torch.tensor(kvals, dtype=torch.int32, device=DEVICE) if kvals else None
            p = torch.tensor(pvals, dtype=torch.float32, device=DEVICE) if pvals else None
            result = function(logits, k=k, p=p, top_k=cap)
            assert isinstance(result, tuple) and len(result) == 2
            values, indices = result
            assert values.device.type == indices.device.type == "npu"
            assert values.shape == indices.shape == logits.shape
            assert indices.dtype == torch.int64
            observed_values, observed_ids = values.cpu(), indices.cpu()
            for row in range(cpu.shape[0]):
                assert sorted(observed_ids[row].tolist()) == list(range(cpu.shape[1]))
                probabilities = torch.softmax(cpu[row].float(), dim=-1).tolist()
                order = sorted(range(len(probabilities)), key=probabilities.__getitem__)
                allowed = set(range(len(probabilities)))
                if kvals:
                    allowed &= set(sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)[:kvals[row]])
                if pvals:
                    discarded = set()
                    cumulative = 0.0
                    for token in order[:-1]:
                        cumulative += probabilities[token]
                        if cumulative <= 1.0 - pvals[row]:
                            discarded.add(token)
                    allowed -= discarded
                kept = {int(index) for index, value in zip(observed_ids[row], observed_values[row]) if torch.isfinite(value)}
                assert kept == allowed, (dtype, kvals, pvals, kept, allowed)
                for index, value in zip(observed_ids[row], observed_values[row]):
                    if torch.isfinite(value):
                        torch.testing.assert_close(value, cpu[row, index], rtol=0, atol=0)
                pos = values[row:row+1].softmax(dim=-1, dtype=torch.float32).argmax(dim=-1, keepdim=True)
                actual = indices[row:row+1].gather(1, pos).item()
                expected = max(allowed, key=lambda token: float(cpu[row, token]))
                assert actual == expected
            torch.testing.assert_close(logits.cpu(), cpu, rtol=0, atol=0)
            cases += 1
        reduced = False
        logits = cpu.to(DEVICE)
        result = function(logits, k=None, p=None, top_k=None)
        assert isinstance(result, torch.Tensor) and result.device.type == "npu"
        torch.testing.assert_close(result.cpu(), cpu, rtol=0, atol=0)
        cases += 1
    torch.npu.synchronize()
    print(f"verified {cases} real NPU reduce-sampler cases; TP=1 local tensors only, no HCCL or custom kernel claim")
    print(json.dumps({"reward": 1.0, "max_abs_error": 0.0}))
except (AssertionError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
    import traceback
    traceback.print_exc()
    if isinstance(exc, RuntimeError) and not any(marker in str(exc) for marker in ("expanded size", "selected index k out of range", "size of tensor")):
        raise
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
PY_VERIFIER
