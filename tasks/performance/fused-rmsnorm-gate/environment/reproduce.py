"""Public latency and correctness check for gated RMSNorm in KDA/GDN layers."""

# ruff: noqa: E402

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
import torch_npu  # noqa: F401
import vllm_ascend

vllm_ascend.__path__.append("/vllm-workspace/vllm-ascend/vllm_ascend")

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm_ascend.utils import register_ascend_customop

torch.npu.set_device(0)
register_ascend_customop()
config = VllmConfig()
config.compilation_config.custom_ops = ["all"]
with set_current_vllm_config(config), torch.inference_mode():
    layer = FusedRMSNormGated(128, eps=1e-6, activation="sigmoid", device="npu", dtype=torch.bfloat16)
    layer.weight.fill_(1)
    x = torch.randn((1, 16, 4, 128), device="npu", dtype=torch.bfloat16)
    gate = torch.randn((16, 4, 128), device="npu", dtype=torch.bfloat16)
    for _ in range(5):
        output = layer(x, gate)
    torch.npu.synchronize()
    durations = []
    for _ in range(20):
        torch.npu.synchronize()
        started = time.monotonic()
        output = layer(x, gate)
        torch.npu.synchronize()
        durations.append(time.monotonic() - started)
    print(
        json.dumps(
            {
                "runtime_class": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "mean_host_sec": statistics.mean(durations),
                "finite_output": bool(output.isfinite().all().item()),
                "scope": "one public shape; not the private performance verdict",
            }
        )
    )
