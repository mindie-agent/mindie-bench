#!/bin/bash
set -eu
python3 -I - <<'PY_VERIFIER'
import json, traceback
try:
    import importlib.util, pathlib, types
    import torch, torch_npu
    from unittest.mock import patch
    import vllm_ascend.ops
    from vllm.config import VllmConfig, set_current_vllm_config
    _context = set_current_vllm_config(VllmConfig())
    _context.__enter__()
    spec = importlib.util.spec_from_file_location("bench_layernorm", pathlib.Path("vllm_ascend/ops/layernorm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(1701)
    errors = []
    max_error = 0.0
    configs = [types.SimpleNamespace(), types.SimpleNamespace(quant_description=None), None,
               types.SimpleNamespace(quant_description={}),
               types.SimpleNamespace(quant_description={"model.norm.bias": "FLOAT"})]
    for config_index, quant in enumerate(configs):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with patch.object(module, "get_current_vllm_config", return_value=types.SimpleNamespace(quant_config=quant)):
                norm = module.AscendRMSNorm(128, eps=1e-5, dtype=dtype).to(device="npu:0", dtype=dtype)
            with torch.no_grad():
                norm.weight.copy_(torch.linspace(0.5, 1.5, 128, device="npu:0", dtype=dtype))
                if norm.bias is not None:
                    norm._bias_weight_loader(norm.bias, torch.full((128,), 0.125, device="npu:0", dtype=dtype))
            for residual_enabled in [False, True]:
                cpu = torch.randn(13, 128).to(dtype)
                residual_cpu = torch.randn_like(cpu) if residual_enabled else None
                x = cpu.to("npu:0")
                residual = residual_cpu.to("npu:0") if residual_enabled else None
                result = norm.forward_oot(x, residual)
                actual = result[0] if isinstance(result, tuple) else result
                combined = cpu.float() + residual_cpu.float() if residual_enabled else cpu.float()
                if residual_enabled:
                    combined = combined.to(dtype).float()
                expected = combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + 1e-5)
                expected = (expected * norm.weight.detach().cpu().float()).to(dtype)
                if norm.bias is not None:
                    expected = expected + norm.bias.detach().cpu()
                observed = actual.detach().cpu()
                error = (observed.float() - expected.float()).abs().max().item()
                max_error = max(max_error, error)
                tolerance = 0.04 if dtype == torch.bfloat16 else 0.006 if dtype == torch.float16 else 0.0001
                torch.testing.assert_close(observed, expected, rtol=tolerance, atol=tolerance)
                if residual_enabled:
                    torch.testing.assert_close(result[1].cpu(), combined.to(dtype), rtol=0, atol=0)
    torch.npu.synchronize()
    print("verified 30 real NPU RMSNorm cases; no-residual and residual, 3 dtypes, 5 quant configs")
    print(json.dumps({"reward": 1.0, "max_abs_error": max_error}))
except (AttributeError, TypeError, AssertionError) as exc:
    traceback.print_exc()
    print(json.dumps({"reward":0.0}))
    raise SystemExit(1)
PY_VERIFIER
