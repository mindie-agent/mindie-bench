# Imports follow selection of the owned helper/source boundary.
# ruff: noqa: E402
"""CPU counterpart of the shape contract behind the saved real NPU failure."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd()))

import torch
from replay_cpu_boundary import load_sampler

sampler, _, gathers = load_sampler(Path.cwd())
logits = torch.tensor([[1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0, -3.0]])
result = sampler.apply_top_k_top_p(logits, torch.tensor([8]), None, top_k=4096)
if not isinstance(result, tuple) or len(result) != 2:
    raise AssertionError("reduce sampling must return candidate logits and global indices")
print(json.dumps({"shapes": [list(value.shape) for value in result], "cpu_gathers": gathers}))
