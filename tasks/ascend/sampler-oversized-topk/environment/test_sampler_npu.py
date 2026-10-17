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
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(focused), str(source), "exec"), namespace)
    return namespace


DEVICE = "npu:0"


class SingleRankGroup:
    world_size = 1
    rank_in_group = 0

    def all_gather(self, value, dim=-1):
        return value


state = {
    "torch": torch,
    "get_tp_group": lambda: SingleRankGroup(),
    "get_ascend_config": lambda: types.SimpleNamespace(enable_reduce_sample=True),
}
source_functions("vllm_ascend/sample/sampler.py", state)
logits = torch.tensor([[0.1, 2.0, -0.5, 1.0, 0.7, -1.0, 1.8, 0.3]], device=DEVICE)
k = torch.tensor([4], dtype=torch.int32, device=DEVICE)
p = torch.tensor([0.8], dtype=torch.float32, device=DEVICE)
result = state["_apply_top_k_top_p_pytorch"](logits, k=k, p=p, top_k=4096)
assert isinstance(result, tuple) and len(result) == 2, (
    "reduce sampler must return candidates and original token indices"
)
values, indices = result
assert values.device.type == indices.device.type == "npu"
pos = values.softmax(dim=-1, dtype=torch.float32).argmax(dim=-1, keepdim=True)
selected = indices.gather(1, pos).squeeze(1)
assert selected.item() == 1
torch.npu.synchronize()
print(
    "real NPU reduce-sampler TP=1 candidate contract completed",
    tuple(values.shape),
    selected.cpu().tolist(),
)
