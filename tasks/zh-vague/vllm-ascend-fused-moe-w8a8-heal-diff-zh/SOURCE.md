# Source Provenance: vllm-ascend-fused-moe-w8a8-heal-diff-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Base Commit**: `4010f9046409696ca7faf38f81fd3679cc3a0986`
- **Fix Commit**: `b1bf5135d31e2c8387b35ac0e5e6962ba87cf851`
- **Target Production File**: `vllm_ascend/ops/fused_moe/moe_mlp.py`
- **Root Cause**:
  In `vllm_ascend/ops/fused_moe/moe_mlp.py` (`quant_apply_mlp`), the fused W8A8 path (`use_gmm_swiglu_quant_fusion`) passes `cumsum_group_list(group_list, group_list_type, 1)` (per-expert counts) instead of `cumsum_group_list(group_list, group_list_type, 0)` (cumulative offsets) to `DeviceOperator.npu_grouped_matmul_swiglu_quant`, and passes `pertoken_scale` instead of `swiglu_out_scale` to `DeviceOperator.npu_grouped_matmul_gmm2` when `dynamic_scale is not None`.
