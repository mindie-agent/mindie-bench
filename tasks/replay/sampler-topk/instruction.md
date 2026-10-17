# Ascend reduce-sampling shape contract

A saved real single-device Ascend run failed with aclnnTopk error 161002 /
EZ1001 when the request's top-k bound was 4096 and its local vocabulary width
was only 8. The complete historical implementation is in
`vllm_ascend/sample/sampler.py`; the original diagnostic is
`history-error.stderr` at the source root.

Repair the reduce-sampling candidate contract. For one rank, the returned pair
must contain valid source logits and their corresponding global token indices.
An oversized requested bound must retain the available local candidates, and
unfiltered sampling must retain all candidates. Normal top-k/top-p filtering
and the ordinary non-reduce path must continue to work. Do not disable
filtering, fabricate indices or change the supplied reproduction/boundary.

Run `npu-replay-sampler` for the frozen historical diagnostic, then
`@CPU_PYTHON@ -I reproduce.py` from the source root for a real CPU tensor
counterpart. The replay is a source-pattern diagnostic and its apparent
success is not a semantic grade. The external judge executes the complete
module independently of the public reproduction.

This task uses explicit framework-import and one-rank collective boundaries.
It requires no NPU, model or network call and makes no multi-rank/HCCL claim.
