A different text-only partial-RoPE configuration uses head size 128, explicit rotary_dim=32 and nonconsecutive token positions. Updating its cache fails with a tensor-width mismatch.

Reproduce with `python3 -I test_rope_npu.py` in the assigned remote workspace.
Repair the historical source and validate it on the assigned NPU. The reproduction loads the workspace cache functions without importing unrelated historical model classes. Preserve existing partial_rotary_factor precedence, full-head behavior, cache values and non-rotary tensor tails.
The public reproduction may be inspected. Independent correctness tests run after completion.
