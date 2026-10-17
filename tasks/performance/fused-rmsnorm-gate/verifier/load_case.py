"""Load the candidate's real registration and upstream layer, without source scanning."""

import sys
from pathlib import Path


def load():
    workspace = Path.cwd().resolve()
    sys.path.insert(0, str(workspace))
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend

    # Only generated build metadata and compiled libraries are readable here.
    # The installed backend's repaired Python sources are not mounted.
    vllm_ascend.__path__.append("/vllm-workspace/vllm-ascend/vllm_ascend")
    import vllm_ascend.utils as candidate
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    if not Path(candidate.__file__).resolve().is_relative_to(workspace):
        raise RuntimeError("candidate registration was imported from outside the business workspace")
    candidate.register_ascend_customop()
    config = VllmConfig()
    config.compilation_config.custom_ops = ["all"]
    return torch, config, set_current_vllm_config, FusedRMSNormGated
