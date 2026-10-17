The PyTorch reduce-sampling path receives a scalar candidate cap larger than its local vocabulary and per-request filtering parameters. A single TP rank then fails before it can return valid filtered candidates and original token indices.

Reproduce with `python3 -I test_sampler_npu.py` in the assigned remote workspace.
Repair the historical source and validate it on the assigned NPU. The reproduction loads the workspace PyTorch filtering function with an explicit one-rank topology adapter. All tensor operations run on the assigned NPU; no HCCL or custom AscendC kernel is exercised. Preserve top-k/top-p filtering, original token identity and the non-reduce return contract.
The public reproduction may be inspected. Independent correctness tests run after completion.
