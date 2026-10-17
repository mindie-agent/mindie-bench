"""Ascend runtime additional_config schema and MoE deployment profile validator.

Note on v0.27+ configuration migration:
Legacy v0.23 environment variable toggles (such as VLLM_ASCEND_ENABLE_FUSED_MC2)
were removed from vllm_ascend.envs in v0.27+ and have no effect on runtime
operator dispatch. MoE and communication switches must be configured via the
Ascend additional_config dictionary passed to EngineArgs.
"""

from __future__ import annotations

ASCEND_ADDITIONAL_CONFIG_DEFAULTS = {
    "refresh": False,
    "enable_cpu_binding": True,
    "multistream_dsv4_dsa_overlap": True,
    "enable_prefill_mc2": False,
    "multistream_overlap_shared_expert": False,
    "enable_kv_nz": False,
    "enable_mc2_hierarchy_comm": False,
    "enable_reduce_sample": False,
    "enable_dsa_cp": False,
    "draft_window_size": None,
    "mix_placement": False,
    "mega_moe_max_tokens": 65536,
    "c8_enable_reshape_optim": False,
    "enable_fused_mc2": 0,
    "enable_mlapo": True,
    "mlapo_keep_prefill_weights": False,
    "msmonitor_use_daemon": False,
    "enable_transpose_kv_cache_by_block": True,
    "weight_nz_mode": 1,
}


def default_moe_deployment_options() -> dict:
    """Return the inherited fleet MoE deployment profile options."""
    return {
        "enable_cpu_binding": False,
        "weight_nz_mode": 0,
        "enable_fused_mc2": 1,
    }


def validate_moe_execution(row: dict, expected_tokens: int) -> None:
    """Verify that the MoE forward pass executed on the non-fused path."""
    operators = row.get("actual_operator_keys", [])
    if any(
        "dispatch_ffn_combine" in name.lower() or "dispatchffncombine" in name.lower()
        for name in operators
    ):
        raise AssertionError(
            f"Fused MC2 operator is still active on rank {row.get('rank')} "
            f"(case {row.get('case_id')}): {operators}"
        )
    if "npu::npu_grouped_matmul" not in operators:
        raise AssertionError(f"Expected non-fused grouped matmul operator not found: {operators}")
    if (
        row.get("ep_size") != 2
        or row.get("tp_size") != 2
        or row.get("hccl_rank_sum") != 3
        or row.get("output_shape") != [expected_tokens, 1024]
        or row.get("output_finite") is not True
        or row.get("output_norm", 0) <= 0
    ):
        raise AssertionError(f"Invalid MoE execution observation: {row}")
