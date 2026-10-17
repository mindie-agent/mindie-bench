"""Explicit single-rank CPU boundary around the complete historical module.

Only unavailable framework/configuration and collective imports are replaced.
The sampler's business implementation is imported intact from the workspace.
No HCCL, AscendC, asynchronous stream or NPU behavior is emulated or certified.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path


def load_sampler(workspace: Path):
    source = workspace / "vllm_ascend/sample/sampler.py"
    state = types.SimpleNamespace(enable_reduce_sample=True)
    gathers = []

    def unavailable(*args, **kwargs):
        raise RuntimeError("this task does not emulate the requested NPU/framework boundary")

    def all_gather(value, dim=-1):
        if value.device.type != "cpu" or dim != -1:
            raise RuntimeError("single-rank fixture only accepts CPU last-dimension gathers")
        gathers.append(tuple(value.shape))
        return value

    group = types.SimpleNamespace(world_size=1, rank_in_group=0, all_gather=all_gather)

    class Device(Enum):
        A2 = "a2"
        A3 = "a3"
        CPU_BOUNDARY = "cpu-boundary"

    modules = {}

    def module(name, **attributes):
        item = types.ModuleType(name)
        item.__dict__.update(attributes)
        item.__path__ = []
        modules[name] = item
        return item

    for name in [
        "vllm",
        "vllm.distributed",
        "vllm.v1",
        "vllm.v1.sample",
        "vllm.v1.sample.ops",
        "vllm_ascend",
        "vllm_ascend.sample",
    ]:
        module(name)
    module("vllm.envs", VLLM_BATCH_INVARIANT=False)
    module("vllm.distributed.parallel_state", get_tp_group=lambda: group)
    module("vllm.triton_utils", HAS_TRITON=False)
    module("vllm.v1.sample.metadata", SamplingMetadata=type("SamplingMetadata", (), {}))
    module("vllm.v1.sample.ops.topk_topp_sampler", TopKTopPSampler=type("TopKTopPSampler", (), {}))
    module("vllm.v1.sample.sampler", Sampler=type("Sampler", (), {}))
    module("vllm_ascend.ascend_config", get_ascend_config=lambda: state)
    module("vllm_ascend.sample.penalties", apply_all_penalties=unavailable)
    module(
        "vllm_ascend.utils",
        AscendDeviceType=Device,
        get_ascend_device_type=lambda: Device.CPU_BOUNDARY,
        global_stream=unavailable,
        npu_stream_switch=unavailable,
    )
    sys.modules.update(modules)
    name = "vllm_ascend.sample.sampler"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import complete historical sampler module")
    sampler = importlib.util.module_from_spec(spec)
    sys.modules[name] = sampler
    spec.loader.exec_module(sampler)
    if Path(sampler.__file__).resolve() != source.resolve():
        raise RuntimeError("sampler was imported from outside the owned source snapshot")
    return sampler, state, gathers
