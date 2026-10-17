A text-only partial-RoPE model with head size 128 and explicit rotary_dim=64 fails while updating position-dependent cosine/sine cache slices. The error reports an expanded dimension of 128 against a source dimension of 64.

Reproduce with `@CPU_PYTHON@ -I test_rope_cpu.py` in the assigned local workspace.
Repair the historical source and validate it on the assigned CPU. The reproduction loads the workspace cache functions without importing unrelated historical model classes. Preserve existing partial_rotary_factor precedence, full-head behavior, cache values and non-rotary tensor tails.
The public reproduction may be inspected. Independent correctness tests run after completion.

This is focused historical Python-source repair on real CPU torch 2.10.0 tensors. The supplied reproduction loads actual source functions; it does not import the entire historical framework. Use the assigned CPU interpreter `@CPU_PYTHON@` for local checks. No remote server, NPU backend, model weights or collective operation is required.
