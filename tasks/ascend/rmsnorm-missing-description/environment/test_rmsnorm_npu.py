import importlib.util
import pathlib
import types
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401 - registers the NPU backend.
import vllm_ascend.ops  # noqa: F401 - initializes operator registrations before loading the snapshot.
from vllm.config import VllmConfig, set_current_vllm_config

VARIANT = "missing"
_context = set_current_vllm_config(VllmConfig())
_context.__enter__()
spec = importlib.util.spec_from_file_location(
    "bench_layernorm", pathlib.Path("vllm_ascend/ops/layernorm.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
quant = (
    types.SimpleNamespace() if VARIANT == "missing" else types.SimpleNamespace(quant_description=None)
)
with patch.object(
    module, "get_current_vllm_config", return_value=types.SimpleNamespace(quant_config=quant)
):
    norm = module.AscendRMSNorm(128, dtype=torch.float32).to("npu:0")
x = torch.randn(8, 128, device="npu:0")
y = norm.forward_oot(x)
torch.npu.synchronize()
print("NPU RMSNorm completed", y.shape, y.device)
