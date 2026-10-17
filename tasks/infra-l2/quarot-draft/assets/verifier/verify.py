"""Private tensor and dispatch controls; public helpers never grade a candidate."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_strict_case(root, architecture, hidden, vocab, rotation_enabled, seed):
    generator = torch.Generator().manual_seed(seed)
    rotation = torch.linalg.qr(torch.randn(hidden, hidden, generator=generator)).Q
    fc = torch.randn(hidden, 3 * hidden, generator=generator)
    embedding = torch.randn(vocab, hidden, generator=generator)

    class UpstreamWeightSink(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.config = types.SimpleNamespace(dtype=torch.float32)
            self.fc = torch.nn.Linear(3 * hidden, hidden, bias=False)
            self.embed_tokens = torch.nn.Embedding(vocab, hidden)

        def load_weights(self, weights):
            parameters = dict(self.named_parameters())
            loaded = set()
            for name, tensor in weights:
                if name not in parameters:
                    raise AssertionError(f"unexpected weight name: {name}")
                parameters[name].data.copy_(tensor)
                loaded.add(name)
            if loaded != set(parameters):
                raise AssertionError("source omitted a required draft weight")
            return loaded

    class Registry:
        entries = {"LlamaForCausalLMEagle3": UpstreamWeightSink}

        @classmethod
        def register_model(cls, name, target):
            cls.entries[name] = target

    def unsupported(*args, **kwargs):
        raise RuntimeError("this CPU boundary does not implement checkpoint shard I/O")

    _module("vllm", ModelRegistry=Registry)
    _module("vllm.config", VllmConfig=object)
    _module("vllm.model_executor")
    _module("vllm.model_executor.models")
    _module(
        "vllm.model_executor.models.llama_eagle3",
        Eagle3LlamaForCausalLM=UpstreamWeightSink,
    )
    _module("safetensors", safe_open=unsupported)
    _module("safetensors.torch", load_file=unsupported)
    _module("vllm_ascend")
    _module(
        "vllm_ascend.utils",
        get_rotation_path=lambda config: "fixture-rotation" if rotation_enabled else None,
        get_rotation_matrix=lambda path: rotation,
    )
    root = Path(root)
    registration = _source("bench_source_registry", root / "vllm_ascend/models/__init__.py")
    loader = _source("bench_source_eagle3", root / "vllm_ascend/models/llama_eagle3.py")
    registration.register_model()
    selected = Registry.entries[architecture]
    if isinstance(selected, str):
        if selected != "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM":
            raise AssertionError(f"unsupported draft dispatch: {selected}")
        selected = loader.AscendEagle3LlamaForCausalLM
    config = types.SimpleNamespace(model_config=types.SimpleNamespace(model="."))
    model = selected(vllm_config=config)
    model.load_weights([("fc.weight", fc.clone()), ("embed_tokens.weight", embedding.clone())])
    expected = fc @ torch.block_diag(rotation, rotation, rotation) if rotation_enabled else fc
    return model.fc.weight.detach(), expected, model.embed_tokens.weight.detach(), embedding


def negative_control():
    try:
        run_strict_case(Path.cwd(), "LlamaForCausalLMEagle3", 32, 48, True, 1729)
    except (AssertionError, SyntaxError, NameError, FileNotFoundError):
        pass
    try:
        torch.testing.assert_close(torch.zeros(3), torch.ones(3))
    except AssertionError:
        print("oracle comparator intact")
        return
    raise RuntimeError("torch.testing.assert_close accepted deliberately unequal tensors")


if "--negative-control" in sys.argv:
    negative_control()
    raise SystemExit(0)

root = Path(os.environ["MINDIE_BENCH_WORKSPACE"])
try:
    for hidden, vocab, seed in [(8, 13, 419), (16, 29, 2027), (64, 73, 9187)]:
        for architecture in ["LlamaForCausalLMEagle3", "Eagle3LlamaForCausalLM"]:
            for rotated in [False, True]:
                actual, expected, embedding, original = run_strict_case(
                    root, architecture, hidden, vocab, rotated, seed
                )
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(embedding, original)
except (AssertionError, SyntaxError, NameError) as error:
    print(json.dumps({"reward": 0.0, "failure_reason": f"draft basis/dispatch failed: {error}"}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))

