# Source-grounded CPU twin

This canonical task is a candidate template. A private interpreter-bound L2 task
revision requires its own source-fail/gold-pass proof before promotion. An L3
proof is never inherited.

- Repository: `vllm-project/vllm-ascend`.
- Historical failing commit: `d552d03a4851aa4d75420377952b0177c94cd4a5`.
- Exact upstream fix: `80cd5964286f2a394e3e6b3d1f7581ce901c812a`; PR #10004.
- CPU family: `cpu-reduce-sampler-candidate-contract`; role: `seed`.
- Private gold patch is the original upstream target-file diff, not a synthetic
  replacement implementation.

The verifier compiles actual top-level functions from the assigned source
snapshot and executes real CPU torch 2.10.0 tensors. Unrelated historical
framework imports/classes are outside this focused contract. There is no NPU
backend, string replay, model weight or implicit device fallback. Sampler TP=1
uses a disclosed identity topology adapter; it does not execute HCCL or the
custom AscendC kernel. CPU success does not prove full-model or NPU behavior.

The public reproduction is copied into the source snapshot. The private verifier,
gold patch and complete source history must remain outside the native agent's
read grants. `materialize.py` binds the public command/private launcher and
records the exact locked interpreter. Runtime/path changes require a fresh task
revision and soundness proof.
