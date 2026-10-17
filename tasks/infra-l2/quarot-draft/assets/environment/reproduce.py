"""Module-level CPU reproduction of a historical Ascend draft-loader incident."""

from pathlib import Path

import torch
from quarot_boundary import run_case

actual, expected, embedding, original = run_case(
    Path.cwd(), "LlamaForCausalLMEagle3", 32, 48, True, 1729
)
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(embedding, original)
print("The draft checkpoint uses the expected hidden-state basis.")

