---
category: workflow
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  workloads:
  - Agentic operator bottleneck triage and fused kernel substitution in LLM/MoE inference
domain_path: operators/model-recipes/agentic-kernel-tuning
evidence_strength: reproduced_empirical
id: unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow
inherits_from:
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/longcat_flash_lite/agentic/optimization_report_tp.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/gemma_4/agentic/optimization_report.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/models/hy3/agentic/optimization_report.md
  source_digest: e5d17b88c94faea9fda270361516d65494ebb7473efa16065cce91b783ca03f9
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:models/longcat_flash_lite/agentic/optimization_report_tp.md
  source_type: code
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
summary: Empirical workflow distilled from the LongCat-Flash-Lite, Gemma-4, and Hy3
  agentic optimization reports for profiling Eager operator bottlenecks, Phase 0 subgraph
  fusion boundaries, and selecting the exact `torch_npu` fused kernel replacements.
tier: L3
title: Agentic Kernel Selection, Bottleneck Profiling, and Operator Fusion Workflow
  (LongCat, Gemma-4, Hy3)
triggers:
  error_signatures:
  - agentic optimization report operator fusion
  - longcat flash lite gemma 4 hy3 operator bottleneck
  - local single-kernel tile tweak yields no end-to-end latency gain due to GM round-trip
  keywords:
  - agentic optimization report operator fusion
  - longcat flash lite gemma 4 hy3 operator bottleneck
  - replace python expert loop with npu_grouped_matmul
  - scatter_update_ static kv cache operator
  - phase0 fusion boundary
  - gm round-trip
  - agentic-optimization
  - kernel-selection
  - longcat-flash-lite
  - gemma-4
  - hy3
  - gmm
  - fia
  - aclgraph
---
## 1. Applicable Problem, Observable Triggers & Selection Boundary

- **When to Select**:
  - End-to-end or module-level profiling on Atlas A2/A3 shows latency dominated by decomposed operator chains (per-expert Python loops, dynamic `torch.cat` KV growth, unfused RMSNorm/RoPE/SwiGLU, or removable producer-consumer GM round-trips on the exposed critical path);
  - Local single-kernel tile/buffer micro-tweaks produce no stage-level speedup because an upstream/downstream GM round-trip or host launch gap dominates the exposed branch.
- **Missing Facts & Minimal Probe (`condition_unknown`)**:
  - If hardware SOC, CANN version, or per-stage critical-path breakdown (`Run/Step/Rank` window) is missing, capture a stage-aligned baseline profile first before selecting a fused replacement.
- **When NOT to Select (Switch / Default Boundary)**:
  - Do **not** adopt if baseline numerical correctness is currently failing (fix correctness before fusion).
  - Do **not** adopt if the candidate operator sits on an already-overlapped non-critical branch or if cross-card HCCL collective wait dominates the exposed step time.

## 2. Empirical Operator Bottleneck Breakdown in Unoptimized NPU Models

Across `LongCat-Flash-Lite` (`68.5B-A3B`), `Gemma-4` (`26B-A4B`), and `Hy3` (`295B-A21B`), profiling unoptimized HuggingFace Eager code on Atlas A2/A3 reveals three dominant operator anti-patterns:
1. **Per-Expert Python Loop in MoE (`> 70%` of Eager latency)**:
   - Iterating over `128..256` experts with `torch.where(expert_mask)` + small `[T_e, H] @ [H, I]` GEMMs triggers hundreds of host-device synchronizations (`nonzero`) and tiny memory-bound kernel launches per layer.
2. **Dynamic `torch.cat` KV Cache Growth (`10%–18%` of Decode latency)**:
   - Re-allocating and copying `[B, H, S, D]` on every decode token saturates HBM write bandwidth and prevents static graph capture.
3. **Decomposed Elementwise Norms, RoPE, and Gated Activations (`10%–15%` of latency)**:
   - Unfused `pow(2) -> mean -> rsqrt -> mul` (`RMSNorm`), slice/concat rotary embeddings, and `silu(gate) * up` (`SwiGLU`) or `gelu(gate) * up` (`GEGLU`) launch 6–10 Vector kernels per sublayer.

## 3. Discriminating Probe & Standardized Substitution Branches

Run a local forward comparison on the aligned `Run/Step/Rank` window to verify whether the target chain is on the exposed critical path (respecting hard UB/L1 capacity and collective barriers), then route by bottleneck class:

1. **MoE Loop -> Sorted `GMM` (`3.5x–6.0x` Eager Speedup)**:
   - Replace router `topk + softmax` with `torch_npu.npu_moe_gating_top_k_softmax` (or `npu_moe_gating_top_k(norm_type=1)` for sigmoid routing in `Hy3`).
   - Permute tokens via `torch_npu.npu_moe_init_routing_v2` and execute expert projections via `torch_npu.npu_grouped_matmul` (`split_item=2`, pre-transposed contiguous weights `[E, In, Out]`).
   - Fuse activation via `torch_npu.npu_swiglu` and unpermute via `torch_npu.npu_moe_finalize_routing`.
2. **Dynamic KV `cat` -> `scatter_update_` + `FIA` (`1.5x–2.2x` Speedup)**:
   - Pre-allocate static `[B, S_max, N_kv * D]` (`BSH`) or `[B, N_kv, S_max, D]` (`BNSD`) buffers, update decode slot in-place via `torch_npu.scatter_update_`, and invoke `torch_npu.npu_fused_infer_attention_score`.
3. **Norm & RoPE Fusion (`1.15x–1.30x` Speedup)**:
   - Replace RMSNorm + residual add with `torch_npu.npu_add_rms_norm` (note: Gemma-4 has `7` RMSNorms per block!) and RoPE with `torch_npu.npu_apply_rotary_pos_emb` / `npu_rotary_mul`.
4. **Host-Overhead Elimination via `npugraph_ex` (`2.5x–3.0x` Additional Decode Speedup)**:
   - Once all dynamic shapes (`torch.cat`, `nonzero`, `.item()`) are removed by steps 1–3, wrap Decode in `npugraph_ex` (`aclgraph` + `static_kernel`), cutting `LongCat-Flash-Lite` decode from `52.1 ms` to `17.5 ms` (`9.2x` over baseline).

## 4. Verification, Exit Criteria & Scoped Counter-Evidence

- **Verification & Exit**: Measure both single-stage and end-to-end Prefill/Decode latency on the identical workload and rank window after each substitution, confirming numerical parity and positive system ROI before enabling graph capture.
- **Scoped Counter-Evidence**: If fused kernel substitution improves isolated microbenchmark latency but end-to-end step time on the recorded shape/topology does not improve, record the bottleneck shift (e.g. collective wait or host launch gap) under those conditions rather than repeating single-kernel micro-tuning.
