"""CPU dependency boundary for EAGLE3 draft checkpoint projection under QuaRot."""

import importlib
import sys
import types
from pathlib import Path

import torch


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def run_case(root, architecture, hidden, vocab, rotation_enabled, seed):
    generator = torch.Generator().manual_seed(seed)
    rotation = torch.linalg.qr(torch.randn(hidden, hidden, generator=generator)).Q
    w_layers = torch.randn(hidden, 3, hidden, generator=generator)
    x_layers = torch.randn(4, 3, hidden, generator=generator)
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

    root = Path(root).resolve()
    for key in list(sys.modules):
        if key == "vllm_ascend" or key.startswith("vllm_ascend."):
            sys.modules.pop(key, None)

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
    _module("vllm_ascend", __path__=[str(root / "vllm_ascend")])
    _module(
        "vllm_ascend.utils",
        get_rotation_path=lambda config: "fixture-rotation" if rotation_enabled else None,
        get_rotation_matrix=lambda path: rotation,
    )

    models_pkg = importlib.import_module("vllm_ascend.models")
    models_pkg.register_model()
    selected = Registry.entries[architecture]
    if isinstance(selected, str):
        module_name, attr_name = selected.split(":", 1)
        selected = getattr(importlib.import_module(module_name), attr_name)
    config = types.SimpleNamespace(model_config=types.SimpleNamespace(model="."))
    model = selected(vllm_config=config)
    fc_ckpt = (w_layers @ rotation.T if rotation_enabled else w_layers).reshape(hidden, 3 * hidden)
    model.load_weights([("fc.weight", fc_ckpt.clone()), ("embed_tokens.weight", embedding.clone())])
    inputs = x_layers.reshape(4, 3 * hidden)
    actual = model.fc(inputs).detach()
    expected = torch.nn.functional.linear(inputs, w_layers.reshape(hidden, 3 * hidden))
    return actual, expected, model.embed_tokens.weight.detach(), embedding


if __name__ == "__main__":
    actual, expected, embedding, original = run_case(
        Path.cwd(), "LlamaForCausalLMEagle3", 32, 48, True, 1729
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(embedding, original)
    print("The draft checkpoint uses the expected hidden-state basis.")

