A real vLLM-Ascend revision fails with this quantization configuration:

TypeError: 'NoneType' object is not iterable while constructing an RMSNorm layer

Reproduce with `python3 -I test_rmsnorm_npu.py` in the assigned remote workspace. Repair the source so quantization configurations that omit optional description metadata can construct and execute RMSNorm on the Ascend NPU. Preserve legacy norm-bias handling, the residual path and numerical correctness. Validate the change on the assigned NPU. The reproduction is public and may be inspected; independent correctness tests run after completion.
