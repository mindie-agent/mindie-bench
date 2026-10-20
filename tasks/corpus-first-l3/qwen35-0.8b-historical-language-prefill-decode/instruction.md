# Qwen3.5-0.8B historical language prefill/decode

Implement and verify the historical Ascend language-model path for the complete
`Qwen/Qwen3.5-0.8B` checkpoint at revision
`2fc06364715b967f1860aea9cf38778875588b17`, weight SHA256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

The task starts from Ascend source0
`a78a00e0b1b618f26a3bfce2519055b40cf4bf1e` against the vLLM empty-target
source `4034c3d32e30d01639459edd3ab486f56993876d`, and the intended fix is
PR7109 commit `0f289fa2a8a381a644a5f329af61715a73ec2aec`.

## Required outcome

- Text-only prefill plus short decode. No image input, no MTP, no chat template.
- Fixed prompt exactly `Hello, my name is` (UTF-8).
- Seed 0, greedy, temperature 0.0, top_p 1.0, exactly 8 new tokens.
- Prompt token IDs must equal `[9419, 11, 821, 803, 369]`.
- Greedy output IDs must equal `[3629, 13, 353, 1044, 264, 220, 17, 15]`.
- Returned chosen-token logprobs must be present, length 8, and finite.
- Every mismatch must remain visible. No aggregate may hide a wrong position.
- The source0 run must preserve its actual failure (or an unexpected pass);
  import/setup failure is an infrastructure failure, not evidence of the bug.

## Runtime

The three arms are source0, after-fix, and the shared control. They must use the
same preprovisioned and verified historical runtime, the same fixed fullweight
cache, and the same reset snapshot. Cold provisioning is a separate shared setup
cost and must not be charged as a P2 causal benefit or hidden in TaskSpec.

The runtime is currently pending/unverified. `environment/entrypoint.sh` exits
3 until `QWEN35_RUNTIME_ID` is supplied. No passed soundness receipt exists.

## Frozen files

Do not modify `support-set.json`, `reference/`, or `verifier/`. Exact comparisons
are the primary gate. No full-vocabulary logit accuracy is claimed and no numeric
NPU tolerance is introduced. The CPU cache self-consistency value `1e-4` is not
an NPU tolerance.
