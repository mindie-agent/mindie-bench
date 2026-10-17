The PyTorch reduce-sampling path fails when both top-k and top-p filtering are disabled and the caller has no scalar candidate cap. With a single TP rank, this path must still satisfy the candidate-logits/original-token-index return contract.

Reproduce with `@CPU_PYTHON@ -I test_sampler_cpu.py` in the assigned local workspace.
Repair the historical source and validate it on the assigned CPU. The reproduction loads the workspace PyTorch filtering function with an explicit one-rank topology adapter. All tensor operations run on the assigned CPU; no HCCL or custom AscendC kernel is exercised. Preserve top-k/top-p filtering, original token identity and the non-reduce return contract.
The public reproduction may be inspected. Independent correctness tests run after completion.

This is focused historical Python-source repair on real CPU torch 2.10.0 tensors. The supplied reproduction loads actual source functions; it does not import the entire historical framework. Use the assigned CPU interpreter `@CPU_PYTHON@` for local checks. No remote server, NPU backend, model weights or collective operation is required.
