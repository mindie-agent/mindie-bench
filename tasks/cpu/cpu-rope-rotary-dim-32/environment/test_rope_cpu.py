import ast
import pathlib
import types

import torch

if torch.__version__ != "2.10.0":
    raise RuntimeError(f"CPU task requires locked torch 2.10.0, got {torch.__version__}")


def source_functions(relative, namespace):
    source = pathlib.Path(relative)
    tree = ast.parse(source.read_text(), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    focused = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(focused), str(source), "exec"), namespace)
    return namespace


DEVICE = "cpu"
HEAD_SIZE = 128
ROTARY_DIM = 32
state = {
    name: None
    for name in (
        "_cos_mla",
        "_sin_mla",
        "_cos",
        "_sin",
        "_cos_cache",
        "_sin_cache",
        "_cos_sin_cache",
        "_cos_slice",
        "_sin_slice",
    )
}
state.update(torch=torch, is_vl_model=lambda config: False, has_rope=lambda config: True)
source_functions("vllm_ascend/ops/rotary_embedding.py", state)
config = types.SimpleNamespace(
    model_config=types.SimpleNamespace(
        use_mla=False,
        get_head_size=lambda: HEAD_SIZE,
        hf_text_config=types.SimpleNamespace(rotary_dim=ROTARY_DIM),
    ),
    scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16),
)
positions = torch.tensor([0, 3, 7], dtype=torch.int64, device=DEVICE)
frequencies = torch.outer(
    torch.arange(32, dtype=torch.float32), 10000.0 ** (-torch.arange(0, ROTARY_DIM, 2) / ROTARY_DIM)
)
state["_cos_sin_cache"] = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(DEVICE)
state["set_cos_and_sin"](config, 4, 1, torch.float32, DEVICE)
state["update_cos_sin"](positions)
cos, sin = state["get_cos_and_sin_slice"]()
assert cos.device.type == sin.device.type == "cpu"
assert tuple(cos.shape) == (1, 3, 1, ROTARY_DIM)
print("real CPU partial-RoPE cache update completed", tuple(cos.shape))
