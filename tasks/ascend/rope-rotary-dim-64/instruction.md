A text-only partial-RoPE model with head size 128 and explicit rotary_dim=64 fails while updating position-dependent cosine/sine cache slices. The error reports an expanded dimension of 128 against a source dimension of 64.

Reproduce with `python3 -I test_rope_npu.py` in the assigned remote workspace.
Repair the historical source and validate it on the assigned NPU. The reproduction loads the workspace cache functions without importing unrelated historical model classes. Preserve existing partial_rotary_factor precedence, full-head behavior, cache values and non-rotary tensor tails.
The public reproduction may be inspected. Independent correctness tests run after completion.
