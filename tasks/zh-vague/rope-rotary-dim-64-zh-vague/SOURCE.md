# Source-grounded executable candidate

This task is not verified until actual before/gold-after soundness succeeds on the selected hardware/image.
Upstream: vllm-project/vllm-ascend PR #6581.
Failing source: `d060c797ed8fd22a1ecdf86c5e26f6eeaf7531f2`.
Gold fix: `80e5812b39bd3b19efa8a6af65658f599bd6cae0`.
Target: `vllm_ascend/ops/rotary_embedding.py`.

Family: `partial-rope-rotary-dim`; role: `seed`.
Two trigger variants in this family count as one defect family.
The verifier executes exact function bodies read from the trial workspace via Python AST,
with real torch/torch_npu tensor operations. It does not replace the defective implementation
with a mock or test a copied gold implementation. This is focused function coverage,
not whole-model serving or full historical-framework compatibility.

The initial fixed public corpus has no declared exact-origin package. A bounded exact PR/commit/title
scan is recorded privately; that scan is not a proof that every possible semantic analogue is absent.
No gold patch, verifier or this provenance file is uploaded as a public task asset.
Installed fixed copies, tests and bytecode need their own image projection before a model trial.
