"""Concrete Dev and 6-case Held-out Pilot cases for Section 4.7 strategy selection.

[Role]
Define the immutable problem statements, declared task conditions, shared
decision-point execution observations, candidate sets, and private judge specs
for `DEV_CASES` (2 exposed-dev cases) and `PILOT_CASES` (6 cases across 3
independent problem families, each with 1 `trigger_matched` and 1
`boundary_or_mismatch` case).

[Invariants]
- Dev and Pilot problem families and lineage sources are strictly disjoint.
- Every case provides both a `reference_solution` (passing the private judge)
  and a `known_bad_solution` (rejected by the private judge).

[Timeouts & Locks]
- Pure immutable data definitions; no I/O, locks, or subprocesses.

[Side-Effect Policy]
- Construction only; never mutates any corpus, store, or workspace.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

CANDIDATE_UNIT_IDS = (
    "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
    "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
    "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
    "unit-ascendc-aicore-exception-and-precision-divergence-workflow",
)


@dataclass(frozen=True)
class PreDecisionObservation:
    observation_id: str
    statement: str
    anchor: str


@dataclass(frozen=True)
class StrategyCase:
    case_id: str
    family_id: str
    split: Literal["exposed_dev", "pilot_held_out"]
    case_kind: Literal["trigger_matched", "boundary_or_mismatch"]
    hardware_tier: str
    topology_tier: Literal["S0", "S1"]
    npu_cards: int
    npu_dies: int
    title: str
    problem_statement: str
    declared_conditions: Mapping[str, Any]
    decision_point_id: str
    pre_decision_observations: tuple[PreDecisionObservation, ...]
    observed_conditions: Mapping[str, Any]
    candidate_ids: tuple[str, ...]
    private_judge: Mapping[str, Any]
    reference_solution: Mapping[str, Any]
    known_bad_solution: Mapping[str, Any]
    shared_prefix_cost: Mapping[str, Any] = field(default_factory=dict)


def _obs(obs_id: str, statement: str, anchor: str) -> PreDecisionObservation:
    return PreDecisionObservation(obs_id, statement, anchor)


DEV_CASES: dict[str, StrategyCase] = {
    "DEV-ACLGRAPH-01": StrategyCase(
        case_id="DEV-ACLGRAPH-01",
        family_id="dev_family_aclgraph_piecewise",
        split="exposed_dev",
        case_kind="trigger_matched",
        hardware_tier="A3",
        topology_tier="S0",
        npu_cards=1,
        npu_dies=1,
        title="Dev ACLGraph piecewise decode bucket transition divergence",
        problem_statement=(
            "vllm-ascend decode step produces corrupted logits on multi-turn requests "
            "running on Ascend910_93."
        ),
        declared_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93"},
        decision_point_id="dp-dev-aclgraph-after-eager-check",
        pre_decision_observations=(
            _obs("OBS-1", "Single-step `--enforce-eager` passes bit-identically on the failing decode input.", "D1:dev_aclgraph_trace:18"),
            _obs("OBS-2", "Under `graph_mode=piecewise`, step A at padded bucket 16 passes, but step B at bucket 4 reusing the persistent buffer diverges on tail slots.", "D1:dev_aclgraph_trace:42"),
        ),
        observed_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "graph_mode": "piecewise"},
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",),
            "forbidden_strategies": (
                "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            ),
            "required_next_actions": ("check_stale_tail_padding_across_buckets", "verify_persistent_buffer_slot_mapping_reset"),
            "post_decision_explanation": "Resetting slot_mapping[num_actual_tokens:padded_num_tokens] = -1 before piecewise graph replay eliminates the B=16 -> B=4 decode divergence.",
            "leak_markers": ("slot_mapping[num_actual_tokens:padded_num_tokens] = -1", "eliminates the B=16 -> B=4 decode divergence"),
            "lineage_source": "dev:model_runner_v1_bucket_trace",
        },
        reference_solution={
            "selected_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "next_action": "check_stale_tail_padding_across_buckets",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "next_action": "tune_single_kernel_tile",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 18.0, "prompt_tokens": 420, "completion_tokens": 95, "npu_runs": 1},
    ),
    "DEV-GDN-01": StrategyCase(
        case_id="DEV-GDN-01",
        family_id="dev_family_k3_gdn_idle_dp",
        split="exposed_dev",
        case_kind="trigger_matched",
        hardware_tier="A3",
        topology_tier="S1",
        npu_cards=1,
        npu_dies=2,
        title="Dev K3 GDN idle-DP state overwrite troubleshooting",
        problem_statement="Two-rank serving on Ascend910_93 intermittently produces divergent tokens after uneven request arrivals.",
        declared_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "dp": 2},
        decision_point_id="dp-dev-gdn-after-checksum",
        pre_decision_observations=(
            _obs("OBS-1", "During an asymmetric step `[B=2, B=0]`, rank 1 has `no_scheduled_requests` while rank 0 is active.", "D2:dev_gdn_trace:14"),
            _obs("OBS-2", "Rank 1's cached `conv_state` and `ssm_state` checksums change after `_dummy_run`.", "D2:dev_gdn_trace:29"),
        ),
        observed_conditions={
            "repository": "vllm-ascend",
            "npu_model": "Ascend910_93",
            "dp": 2,
            "idle_rank_state": "no_scheduled_requests",
            "affected_state": ["conv_state", "ssm_state"],
        },
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",),
            "forbidden_strategies": (
                "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            ),
            "required_next_actions": ("probe_dummy_run_skip_gdn_state_update_metadata", "verify_per_row_idle_conv_ssm_state_checksum"),
            "post_decision_explanation": "Passing skip_gdn_state_update=True and zero-padding the idle workspace preserves rank 1 conv_state/ssm_state across [B, 0] steps.",
            "leak_markers": ("preserves rank 1 conv_state/ssm_state across [B, 0] steps", "zero-padding the idle workspace"),
            "lineage_source": "dev:k3_gdn_idle_dp_trace",
        },
        reference_solution={
            "selected_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "next_action": "probe_dummy_run_skip_gdn_state_update_metadata",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "next_action": "recapture_full_decode_graph",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 22.0, "prompt_tokens": 460, "completion_tokens": 110, "npu_runs": 1},
    ),
}


PILOT_CASES: dict[str, StrategyCase] = {
    "S1-F1-trigger": StrategyCase(
        case_id="S1-F1-trigger",
        family_id="pilot_family_1_graph_capture_boundary",
        split="pilot_held_out",
        case_kind="trigger_matched",
        hardware_tier="A3",
        topology_tier="S1",
        npu_cards=1,
        npu_dies=2,
        title="Full-decode ACLGraph replay corrupts active slots after padded bucket shrink",
        problem_statement="On Ascend910_93, online decode serving shows occasional token logit drift when concurrent request count drops from burst to single-request decode.",
        declared_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93"},
        decision_point_id="dp-f1-trigger-after-eager-parity",
        pre_decision_observations=(
            _obs("F1-T-OBS1", "Replaying the exact burst-to-single request schedule with `--enforce-eager` matches the fp32 reference on every decode token (max_abs_err = 1.2e-4).", "P1:f1_trigger_capture:19"),
            _obs("F1-T-OBS2", "With `graph_mode=full_decode_only`, transitioning from `padded_num_tokens=32` to `num_scheduled_tokens=2` (`padded_num_tokens=4`) leaves stale `slot_mapping` entries in persistent static input buffers.", "P1:f1_trigger_capture:47"),
        ),
        observed_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "graph_mode": "full_decode_only"},
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",),
            "forbidden_strategies": (
                "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            ),
            "required_next_actions": ("check_stale_tail_padding_across_buckets", "verify_persistent_buffer_slot_mapping_reset"),
            "post_decision_explanation": "Masking padded tail slots `[num_actual_tokens:padded_num_tokens]` to `-1` in persistent static buffers restores bit-identical ACLGraph decode replay.",
            "leak_markers": ("Masking padded tail slots `[num_actual_tokens:padded_num_tokens]` to `-1`", "restores bit-identical ACLGraph decode replay"),
            "lineage_source": "pilot:f1_full_decode_bucket_shrink",
        },
        reference_solution={
            "selected_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "next_action": "check_stale_tail_padding_across_buckets",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            "next_action": "substitute_npu_grouped_matmul",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 19.5, "prompt_tokens": 440, "completion_tokens": 105, "npu_runs": 1},
    ),
    "S1-F1-boundary": StrategyCase(
        case_id="S1-F1-boundary",
        family_id="pilot_family_1_graph_capture_boundary",
        split="pilot_held_out",
        case_kind="boundary_or_mismatch",
        hardware_tier="A3",
        topology_tier="S1",
        npu_cards=1,
        npu_dies=2,
        title="Decode NaN reported under ACLGraph config but eager already overflows on step 1",
        problem_statement="In vllm-ascend `model_runner_v1` with `aclgraph` enabled on Ascend910_93, decode outputs contain `NaN` after padded batch bucketing.",
        declared_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "graph_mode": "aclgraph"},
        decision_point_id="dp-f1-boundary-after-eager-repro",
        pre_decision_observations=(
            _obs("F1-B-OBS1", "Disabling graph capture via `--enforce-eager` (`graph_mode=eager`) reproduces the exact same `NaN` on decode step 1 before any graph capture or replay runs.", "P1:f1_boundary_capture:15"),
            _obs("F1-B-OBS2", "Per-layer tensor inspection in eager mode shows `fp16` square-sum overflow inside the custom residual RMSNorm input (`max_val = 65504`).", "P1:f1_boundary_capture:31"),
        ),
        observed_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "graph_mode": "eager"},
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("default",),
            "forbidden_strategies": (
                "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
                "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            ),
            "required_next_actions": ("default_eager_single_op_fp32_accumulator_fix", "isolate_eager_rmsnorm_fp16_overflow"),
            "post_decision_explanation": "The defect is an eager fp16 accumulator overflow in RMSNorm; debugging ACLGraph bucket capture wastes NPU runs without touching the broken reduction.",
            "leak_markers": ("eager fp16 accumulator overflow in RMSNorm", "wastes NPU runs without touching the broken reduction"),
            "lineage_source": "pilot:f1_eager_overflow_under_aclgraph_flag",
        },
        reference_solution={
            "selected_strategy_ref": "default",
            "declared_adoption": False,
            "applied_strategy_ref": "default",
            "next_action": "isolate_eager_rmsnorm_fp16_overflow",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "next_action": "check_stale_tail_padding_across_buckets",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 16.0, "prompt_tokens": 410, "completion_tokens": 90, "npu_runs": 1},
    ),
    "S1-F2-trigger": StrategyCase(
        case_id="S1-F2-trigger",
        family_id="pilot_family_2_asymmetric_dp_state",
        split="pilot_held_out",
        case_kind="trigger_matched",
        hardware_tier="A3",
        topology_tier="S1",
        npu_cards=2,
        npu_dies=4,
        title="Asymmetric DP2 TP2 hybrid linear-attention state corruption after idle step",
        problem_statement="A 4-Die (`DP=2, TP=2`) hybrid model deployment on Ascend910_93 passes single-request tests but produces wrong continuations on rank 1 after staggered client bursts.",
        declared_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "dp": 2},
        decision_point_id="dp-f2-trigger-after-state-probe",
        pre_decision_observations=(
            _obs("F2-T-OBS1", "At step 3, DP rank 0 serves 4 decode tokens while DP rank 1 has `idle_rank_state=no_scheduled_requests`.", "P2:f2_trigger_state:22"),
            _obs("F2-T-OBS2", "Per-row state checksums show rank 1's recurrent `conv_state` and `ssm_state` were mutated during the dummy collective step.", "P2:f2_trigger_state:39"),
        ),
        observed_conditions={
            "repository": "vllm-ascend",
            "npu_model": "Ascend910_93",
            "dp": 2,
            "idle_rank_state": "no_scheduled_requests",
            "affected_state": ["conv_state", "ssm_state"],
        },
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",),
            "forbidden_strategies": (
                "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            ),
            "required_next_actions": ("probe_dummy_run_skip_gdn_state_update_metadata", "verify_per_row_idle_conv_ssm_state_checksum"),
            "post_decision_explanation": "Guarding the zero-token dummy step with skip_gdn_state_update=True prevents rank 1 recurrent state mutation during asymmetric DP2 TP2 steps.",
            "leak_markers": ("Guarding the zero-token dummy step with skip_gdn_state_update=True", "prevents rank 1 recurrent state mutation"),
            "lineage_source": "pilot:f2_dp2_tp2_hybrid_recurrent_state",
        },
        reference_solution={
            "selected_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "next_action": "probe_dummy_run_skip_gdn_state_update_metadata",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "next_action": "check_stale_tail_padding_across_buckets",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 24.0, "prompt_tokens": 480, "completion_tokens": 115, "npu_runs": 1},
    ),
    "S1-F2-boundary": StrategyCase(
        case_id="S1-F2-boundary",
        family_id="pilot_family_2_asymmetric_dp_state",
        split="pilot_held_out",
        case_kind="boundary_or_mismatch",
        hardware_tier="A3",
        topology_tier="S1",
        npu_cards=2,
        npu_dies=4,
        title="DP=2 asymmetric batch hang on pure dense attention model without recurrent state",
        problem_statement="In `vllm-ascend` `dp=2` serving on Ascend910_93, when one DP rank has no scheduled requests (`idle DP`), `coordinate_batch_across_dp` times out.",
        declared_conditions={"repository": "vllm-ascend", "npu_model": "Ascend910_93", "dp": 2, "idle_rank_state": "no_scheduled_requests"},
        decision_point_id="dp-f2-boundary-after-state-inventory",
        pre_decision_observations=(
            _obs("F2-B-OBS1", "Model architecture inspection confirms a pure dense Transformer without GDN/Mamba/KDA layers (`affected_state=['kv_cache']`, no `conv_state` or `ssm_state`).", "P2:f2_boundary_state:12"),
            _obs("F2-B-OBS2", "Rank 1 returned early before `dist.all_reduce` in `coordinate_batch_across_dp` because a custom sampler hook skipped the dummy step entirely.", "P2:f2_boundary_state:28"),
        ),
        observed_conditions={
            "repository": "vllm-ascend",
            "npu_model": "Ascend910_93",
            "dp": 2,
            "idle_rank_state": "no_scheduled_requests",
            "affected_state": ["kv_cache"],
        },
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("default",),
            "forbidden_strategies": (
                "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
                "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            ),
            "required_next_actions": ("default_restore_dp_collective_entry_on_idle_rank", "probe_missing_or_mismatched_affected_state_before_gdn_workflow"),
            "post_decision_explanation": "Because `affected_state` contains only `kv_cache` (no `conv_state`/`ssm_state`), the GDN recurrent state workflow does not apply; restoring the skipped `coordinate_batch_across_dp` collective call fixes the hang.",
            "leak_markers": ("GDN recurrent state workflow does not apply", "restoring the skipped `coordinate_batch_across_dp` collective call"),
            "lineage_source": "pilot:f2_pure_dense_dp_collective_skip",
        },
        reference_solution={
            "selected_strategy_ref": "default",
            "declared_adoption": False,
            "applied_strategy_ref": "default",
            "next_action": "default_restore_dp_collective_entry_on_idle_rank",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            "next_action": "probe_dummy_run_skip_gdn_state_update_metadata",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 20.0, "prompt_tokens": 450, "completion_tokens": 100, "npu_runs": 1},
    ),
    "S1-F3-trigger": StrategyCase(
        case_id="S1-F3-trigger",
        family_id="pilot_family_3_operator_fusion_vs_collective",
        split="pilot_held_out",
        case_kind="trigger_matched",
        hardware_tier="A2",
        topology_tier="S1",
        npu_cards=2,
        npu_dies=2,
        title="MoE decode step latency dominated by decomposed Python expert loop and RMSNorm chain",
        problem_statement="An MoE inference workload on Atlas A2 (`Ascend 910B (Atlas A2)`, `CANN 8.5.0`) misses its Decode TPOT target; initial single-kernel UB tile adjustments yielded 0% gain.",
        declared_conditions={"cann_versions": "8.5.0", "hardware": "Ascend 910B (Atlas A2)"},
        decision_point_id="dp-f3-trigger-after-stage-profile",
        pre_decision_observations=(
            _obs("F3-T-OBS1", "Stage-aligned profiling on the exposed decode path shows 74% of step time is spent in a per-expert `torch.where` Python loop and decomposed RMSNorm/SwiGLU launches.", "P3:f3_trigger_prof:17"),
            _obs("F3-T-OBS2", "Two isolated UB tile tweaks on a single GEMM changed end-to-end step latency by < 0.2% because host-device `nonzero` syncs and GM round-trips dominate the critical path.", "P3:f3_trigger_prof:36"),
        ),
        observed_conditions={
            "cann_versions": "8.5.0",
            "hardware": "Ascend 910B (Atlas A2)",
            "workloads": "Agentic operator bottleneck triage and fused kernel substitution in LLM/MoE inference",
        },
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",),
            "forbidden_strategies": (
                "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
                "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            ),
            "required_next_actions": ("replace_moe_loop_with_sorted_npu_grouped_matmul_and_fused_norm", "align_stage_window_and_substitute_fused_operators"),
            "post_decision_explanation": "Replacing the per-expert Python loop with `npu_moe_init_routing_v2` + `npu_grouped_matmul` + `npu_swiglu` and `npu_add_rms_norm` cuts decode latency by 3.8x.",
            "leak_markers": ("cuts decode latency by 3.8x", "npu_moe_init_routing_v2` + `npu_grouped_matmul`"),
            "lineage_source": "pilot:f3_moe_eager_fusion_substitution",
        },
        reference_solution={
            "selected_strategy_ref": "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            "next_action": "replace_moe_loop_with_sorted_npu_grouped_matmul_and_fused_norm",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
            "next_action": "check_stale_tail_padding_across_buckets",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 21.0, "prompt_tokens": 430, "completion_tokens": 98, "npu_runs": 1},
    ),
    "S1-F3-boundary": StrategyCase(
        case_id="S1-F3-boundary",
        family_id="pilot_family_3_operator_fusion_vs_collective",
        split="pilot_held_out",
        case_kind="boundary_or_mismatch",
        hardware_tier="A2",
        topology_tier="S1",
        npu_cards=2,
        npu_dies=2,
        title="MoE operator fusion requested while dump-masked 507035 crash and HCCL skew dominate",
        problem_statement="Optimize MoE operator bottlenecks on Atlas A2 (`Ascend 910B (Atlas A2)`, `CANN 8.5.0`) using fused kernel replacements (`npu_grouped_matmul`, `fia`, `aclgraph`).",
        declared_conditions={
            "cann_versions": "8.5.0",
            "hardware": "Ascend 910B (Atlas A2)",
            "workloads": "Agentic operator bottleneck triage and fused kernel substitution in LLM/MoE inference",
        },
        decision_point_id="dp-f3-boundary-after-uninstrumented-check",
        pre_decision_observations=(
            _obs("F3-B-OBS1", "The uninstrumented baseline run fails with `507035` (`errcode 82` UB out of bounds); it only exited 0 earlier because `ASCENDC_DUMP=1` serialized the stream timing.", "P3:f3_boundary_prof:11"),
            _obs("F3-B-OBS2", "On inputs that do complete, cross-rank `HCCL AlltoAll` wait is 81% of step time while the local MoE compute branch is already overlapped.", "P3:f3_boundary_prof:27"),
        ),
        observed_conditions={
            "cann_versions": "8.5.0",
            "hardware": "Ascend 910B (Atlas A2)",
            "workloads": "Unverified correctness 507035 observer perturbation and HCCL collective wait",
        },
        candidate_ids=CANDIDATE_UNIT_IDS[:3],
        private_judge={
            "acceptable_strategies": ("default", "unit-ascendc-aicore-exception-and-precision-divergence-workflow"),
            "forbidden_strategies": (
                "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
                "unit-vllm-ascend-aclgraph-capture-invalidation-triage-sop",
                "unit-vllm-ascend-hybrid-recurrent-dp-state-hermes-workflow",
            ),
            "required_next_actions": ("default_run_uninstrumented_baseline_and_fix_507035_before_fusion", "observer_control_and_ub_ledger_fix_before_optimization"),
            "post_decision_explanation": "Substituting fused operators while the uninstrumented baseline crashes with 507035 (masked only by dump timing) and AlltoAll dominates step time is invalid; must run uninstrumented control and fix UB bounds first.",
            "leak_markers": ("Substituting fused operators while the uninstrumented baseline crashes", "must run uninstrumented control and fix UB bounds first"),
            "lineage_source": "pilot:f3_observer_masked_507035_and_collective_bound",
        },
        reference_solution={
            "selected_strategy_ref": "default",
            "declared_adoption": False,
            "applied_strategy_ref": "default",
            "next_action": "default_run_uninstrumented_baseline_and_fix_507035_before_fusion",
            "verifier_passed": True,
        },
        known_bad_solution={
            "selected_strategy_ref": "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            "declared_adoption": True,
            "applied_strategy_ref": "unit-agentic-model-kernel-selection-and-profiling-driven-fusion-workflow",
            "next_action": "replace_moe_loop_with_sorted_npu_grouped_matmul_and_fused_norm",
            "verifier_passed": False,
        },
        shared_prefix_cost={"wall_clock_sec": 17.5, "prompt_tokens": 415, "completion_tokens": 92, "npu_runs": 1},
    ),
}

ALL_STRATEGY_CASES: dict[str, StrategyCase] = {**DEV_CASES, **PILOT_CASES}


def resolve_strategy_case(case: str | StrategyCase) -> StrategyCase:
    if isinstance(case, StrategyCase):
        return case
    if case not in ALL_STRATEGY_CASES:
        raise ValueError(f"unknown strategy selection case: {case!r}")
    return ALL_STRATEGY_CASES[case]


def _strip_revision(ref: Any) -> str:
    text = str(ref or "").strip()
    if not text:
        return "default"
    return text.split("@", 1)[0]


def evaluate_strategy_continuation(
    case: str | StrategyCase,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Blind private-judge evaluation of strategy selection, adoption, and task outcome."""
    selected_case = resolve_strategy_case(case)
    blind_payload = {
        key: value for key, value in payload.items()
        if key not in {"arm", "arm_id", "selector_arm", "group"}
    }
    judge = selected_case.private_judge
    acceptable = {_strip_revision(item) for item in judge.get("acceptable_strategies", ())}
    forbidden = {_strip_revision(item) for item in judge.get("forbidden_strategies", ())}
    required_actions = {str(item) for item in judge.get("required_next_actions", ())}

    selected_ref = _strip_revision(blind_payload.get("selected_strategy_ref", "default"))
    applied_ref = _strip_revision(
        blind_payload.get("applied_strategy_ref")
        or (selected_ref if blind_payload.get("declared_adoption") else "default")
    )
    declared_adoption = bool(blind_payload.get("declared_adoption", selected_ref != "default"))
    applied_in_actions = applied_ref != "default"
    next_action = str(blind_payload.get("next_action") or "").strip()

    selection_reasonable = (
        selected_ref in acceptable
        or (selected_ref == "default" and next_action in required_actions)
    )
    next_action_reasonable = next_action in required_actions
    false_adoption = applied_ref in forbidden or (
        selected_case.case_kind == "boundary_or_mismatch"
        and applied_ref != "default"
        and applied_ref not in acceptable
    )
    verifier_passed = blind_payload.get("verifier_passed") is True and not false_adoption
    return {
        "case_id": selected_case.case_id,
        "family_id": selected_case.family_id,
        "case_kind": selected_case.case_kind,
        "selected_strategy_ref": selected_ref,
        "declared_adoption": declared_adoption,
        "applied_strategy_ref": applied_ref,
        "applied_in_actions": applied_in_actions,
        "switched_after_observation": selected_ref != applied_ref,
        "next_action": next_action,
        "selection_reasonable": selection_reasonable and next_action_reasonable,
        "false_adoption": false_adoption,
        "task_succeeded": verifier_passed,
    }


def verify_pilot_lineage_and_soundness(
    pilot_cases: Sequence[StrategyCase] = tuple(PILOT_CASES.values()),
    dev_cases: Sequence[StrategyCase] = tuple(DEV_CASES.values()),
    *,
    min_independent_families: int = 3,
) -> dict[str, Any]:
    """Audit Dev vs Pilot lineage separation and reference/bad-solution soundness."""
    dev_families = {case.family_id for case in dev_cases}
    pilot_families = {case.family_id for case in pilot_cases}
    dev_sources = {str(case.private_judge.get("lineage_source", "")) for case in dev_cases}
    pilot_sources = {str(case.private_judge.get("lineage_source", "")) for case in pilot_cases}
    overlapping_families = sorted(dev_families & pilot_families)
    overlapping_sources = sorted(dev_sources & pilot_sources)

    by_family: dict[str, set[str]] = {}
    soundness_failures: list[str] = []
    for case in [*dev_cases, *pilot_cases]:
        by_family.setdefault(case.family_id, set()).add(case.case_kind)
        ref_eval = evaluate_strategy_continuation(case, case.reference_solution)
        bad_eval = evaluate_strategy_continuation(case, case.known_bad_solution)
        if not (ref_eval["selection_reasonable"] and ref_eval["task_succeeded"] and not ref_eval["false_adoption"]):
            soundness_failures.append(f"{case.case_id}: reference_solution failed private judge")
        if bad_eval["selection_reasonable"] or bad_eval["task_succeeded"] or not bad_eval["false_adoption"]:
            soundness_failures.append(f"{case.case_id}: known_bad_solution was not rejected by private judge")

    incomplete_pilot_families = sorted(
        fam for fam in pilot_families
        if by_family.get(fam) != {"trigger_matched", "boundary_or_mismatch"}
    )
    ready = (
        len(pilot_families) >= min_independent_families
        and not overlapping_families
        and not overlapping_sources
        and not incomplete_pilot_families
        and not soundness_failures
    )
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "independent_pilot_families": len(pilot_families),
        "min_independent_families": min_independent_families,
        "overlapping_families": overlapping_families,
        "overlapping_lineage_sources": overlapping_sources,
        "incomplete_pilot_families": incomplete_pilot_families,
        "soundness_failures": soundness_failures,
    }


def claude_cli_strategy_selector(
    view: Mapping[str, Any],
    *,
    executable: str = "claude",
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Invoke live `claude -p` in isolated tool-less 1-turn JSON mode for Arm B / Arm C."""
    prompt = (
        "You are an isolated strategy selector with tool_access=none.\n"
        "Given the following JSON view, reply with ONLY a single valid JSON object matching output_schema "
        "(no markdown fences, no prose outside JSON).\n"
        "Rule: In Arm B (selector_arm=\"B\"), select the candidate workflow/SOP whose surface problem domain "
        "and keywords best match the task statement so the executor can mount and test that SOP; "
        "in Arm C (selector_arm=\"C\"), check pre_decision_observations and observed_conditions strictly: "
        "if pre_decision_observations show the candidate premise is contradicted or mismatched (e.g. eager mode "
        "already reproduces the error, only active rank crashes on concrete tensor, or uninstrumented baseline "
        "crashes with 507035 / AlltoAll dominates), you MUST select \"default\" with applicability_basis=\"mismatch\".\n\n"
        + json.dumps(view, ensure_ascii=False)
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [executable, "-p", prompt, "--output-format", "json", "--max-turns", "1", "--tools", ""],
            capture_output=True, text=True, timeout=timeout_sec, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"claude selector timed out after {timeout_sec}s") from exc
    elapsed_sec = round(time.perf_counter() - started, 4)
    envelope = json.loads(completed.stdout)
    raw_text = str(envelope.get("result", "")).strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").removeprefix("json").strip()
    decision = dict(json.loads(raw_text))
    usage_block = envelope.get("usage") if isinstance(envelope.get("usage"), Mapping) else {}
    in_tok = usage_block.get("input_tokens")
    out_tok = usage_block.get("output_tokens")
    decision["usage"] = {
        "prompt_tokens": int(in_tok) if in_tok is not None else None,
        "completion_tokens": int(out_tok) if out_tok is not None else None,
        "wall_clock_sec": elapsed_sec,
    }
    return decision


