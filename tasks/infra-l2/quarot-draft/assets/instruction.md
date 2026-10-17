In this `vllm-ascend` checkout, QuaRot-quantized EAGLE3 draft checkpoints (`LlamaForCausalLMEagle3`) are not registered in the Ascend model registry and their compressed hidden-state projection (`fc`) produces mismatched outputs under QuaRot rotation when `quarot_boundary.py` is run. Checkpoints using `Eagle3LlamaForCausalLM` and unrotated checkpoints continue to work.

Run `@L2_PYTHON@ -B quarot_boundary.py` (or `@L2_PYTHON@ -B reproduce.py`) from the source root to reproduce the projection mismatch on CPU tensors. Register and fix the QuaRot EAGLE3 draft model projection so `@L2_PYTHON@ -B quarot_boundary.py` and `@L2_PYTHON@ -B reproduce.py` pass while preserving `Eagle3LlamaForCausalLM` and unrotated weight loading. Do not modify the reproduction scripts.

