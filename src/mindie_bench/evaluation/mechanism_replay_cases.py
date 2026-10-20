"""Concrete M1-01 ~ M6-01 replay cases and private judge specifications.

[Role]
Hold the pre-decision facts, competing hypotheses, provenance anchors and
private post-decision judges for the six Section 4.6 mechanism cases.

[Invariants]
- Every fact carries an exact T1-T5 locator; learner views never include the
  private post-decision observation, perturbation explanation or scoring.
- Cases describe decisions, not A5/K3 parameter copies or repair code.

[Timeouts & Locks]
- Pure immutable data; no I/O, lock, subprocess or hardware access.

[Side-Effect Policy]
- Construction only; no corpus, transcript or store is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ReplayFact:
    fact_id: str
    statement: str
    anchor: str


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str


@dataclass(frozen=True)
class MechanismCase:
    case_id: str
    title: str
    facts: tuple[ReplayFact, ...]
    hypotheses: tuple[Hypothesis, ...]
    criteria: tuple[str, ...]
    private_judge: Mapping[str, Any]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({fact.anchor.split(":", 1)[0] for fact in self.facts}))


def _fact(fact_id: str, anchor: str, statement: str) -> ReplayFact:
    return ReplayFact(fact_id, statement, anchor)


def _hypothesis(ident: str, statement: str) -> Hypothesis:
    return Hypothesis(ident, statement)


# ----------------------------------------------------------------------- cases
_M1 = MechanismCase(
    case_id="M1-01",
    title="Phase 0 fusion triage before single-kernel tuning",
    facts=(
        _fact("F1", "T1:theoretical-expectation.md:124-145",
              "Two local tile/buffer tweaks produced no measurable tail change; the remaining tail "
              "crosses a GM round-trip between the upstream MLAPO quantized output and a standalone "
              "pack helper."),
        _fact("F2", "T1:pipeline-microstage-README.md",
              "The dependency/queue trace exposes tail = max(P+C, Q+G); pack writeback is on the "
              "exposed branch while one collective branch is already overlapped."),
        _fact("F3", "T3:53",
              "A local forward comparison can separate the quantized-output/fusion boundary in "
              "minutes, whereas a cross-version bisect costs about six hours and four cluster slots."),
        _fact("F4", "T1:FLASHMLA_C8_PLAN.md",
              "Producer FIXPIPE cast can remove a 9 MiB/call FP32 GM round-trip; UB/L1 capacity and "
              "collective barriers are hard fusion boundaries."),
    ),
    hypotheses=(
        _hypothesis("H1_fusion_boundary", "Removable GM round-trips and helper fusion dominate the exposed tail."),
        _hypothesis("H2_single_kernel_pipeline", "A single-kernel Tile/Buffer or instruction schedule is the limit."),
        _hypothesis("H3_collective_barrier", "An unavoidable cross-card collective barrier is the exposed tail."),
    ),
    criteria=("m1_phase0_fusion", "m1_distinguishing_next_probe", "m1_avoids_repeat"),
    private_judge={
        "allowed_actions": ("phase0_fusion_triage", "phase0_subgraph_fusion_triage",
                            "subgraph_fusion_analysis", "subgraph_fusion_first"),
        "post_decision": (
            "The independent pack writeback disappears after FlashMLA emits the wire layout; it was "
            "on the exposed path and accounted for 21-23us, while the overlapped branch did not move."
        ),
        "perturbation_explanation": "No observer perturbation applies to the timing comparison.",
        "scoring": {
            "phase0_first": 2, "distinguishing_probe": 2, "no_repeat": 1,
            "single_kernel_first": -3, "cross_version_bisect_first": -1,
        },
        "leak_markers": ("21-23us", "wire layout", "overlapped branch did not move"),
    },
)

_M2 = MechanismCase(
    case_id="M2-01",
    title="First equivalent boundary and non-additive simulation",
    facts=(
        _fact("F1", "T2:639",
              "The first MoE partial-sum difference appears at a logical token boundary; physical "
              "block IDs differ only because the allocator recycled a block."),
        _fact("F2", "T2:737",
              "The collective allgather output matches the reference, but the following reduction and "
              "RMSNorm stages are observed in a different order."),
        _fact("F3", "T2:790",
              "Reduction and RMSNorm order must be compared after a correct collective; a "
              "collective match does not establish whole-module correctness."),
        _fact("F3b", "T3:20203",
              "Alignment must compare logical token/state/stage/rank identity, shape, dtype, stride, "
              "layout, ownership and lifetime rather than raw physical IDs."),
        _fact("F4", "T1:pipeline-microstage-README.md",
              "The 36-node dependency graph has a maximum-cycle-mean lower bound, not a sum of stage "
              "times; the epilogue service duration is missing."),
        _fact("F5", "T1:theoretical-expectation.md:124-145",
              "A short capture shows 1650 MHz boost while the sustained thermal clock is 1250-1300 MHz."),
        _fact("F6", "T1:FLASHMLA_C8_PLAN.md",
              "The MLAPO concurrent subgraph is bounded by max(P+C, Q+G), not by adding overlapping "
              "stage times."),
    ),
    hypotheses=(
        _hypothesis("H1_logical_alignment", "A logical token/state boundary diverges after a correct collective."),
        _hypothesis("H2_physical_block", "Recycled physical block IDs are the defect."),
        _hypothesis("H3_collective_defect", "The allgather collective itself is wrong."),
        _hypothesis("H4_downstream_order", "Reduction and RMSNorm order differs after a correct collective."),
    ),
    criteria=("m2_logical_first_divergence", "m2_downstream_contract", "m2_nonadditive_simulation",
              "m2_no_short_burst_claim"),
    private_judge={
        "allowed_actions": ("locate_first_equivalent_boundary", "first_boundary_alignment",
                            "boundary_alignment_and_simulation"),
        "post_decision": (
            "Collective output is correct; reduction before RMSNorm is the first equivalent-boundary "
            "divergence. The epilogue duration was never measured and the 36-node cycle mean is not "
            "the sum of node durations."
        ),
        "perturbation_explanation": "Short capture measures boost clock, not sustained thermal clock.",
        "scoring": {
            "logical_alignment": 2, "downstream_order": 2, "nonadditive_bound": 2,
            "physical_id_claim": -3, "collective_only_pass": -2, "fabricated_duration": -3,
        },
        "leak_markers": ("reduction before RMSNorm", "epilogue duration was never measured",
                         "boost clock, not sustained"),
    },
)

_M3 = MechanismCase(
    case_id="M3-01",
    title="Observer perturbation control",
    facts=(
        _fact("F1", "T3:3028",
              "The candidate fix branch execution count is zero after the patch; the dump-enabled run "
              "passes while the identical uninstrumented run still fails."),
        _fact("F2", "T3:22797",
              "Dump changes stream timing; the failure disappears only when the observer is enabled, "
              "and no uninstrumented baseline has yet been compared."),
        _fact("F3", "T3:22844",
              "An observer-only control without the candidate patch can separate timing perturbation "
              "from a real repair."),
        _fact("F4", "T1:pipeline-experiments.md",
              "A repeated public baseline and interleaved ABBA timing control are available."),
        _fact("F5", "T3:draft-causal-dfx",
              "The DFX matrix is still a draft (R20_FROZEN=0); no reset/warm/hit matrix has executed."),
    ),
    hypotheses=(
        _hypothesis("H1_observer_perturbation", "Dump/clone/sync changes timing and masks the failure."),
        _hypothesis("H2_actual_fix", "The candidate patch fixes the failure without instrumentation."),
        _hypothesis("H3_environment", "An unrelated environment change explains the pass."),
    ),
    criteria=("m3_observer_controls", "m3_branch_execution_claim", "m3_dfx_disabled_final"),
    private_judge={
        "allowed_actions": ("observer_control_design", "observer_effect_control",
                            "causal_observation_experiment"),
        "post_decision": (
            "The DFX-disabled final run still fails; the candidate branch never executed, so the pass "
            "under dump is an observer effect and the causal question remains inconclusive."
        ),
        "perturbation_explanation": "Dump serializes the stream and changes the failing timing window.",
        "scoring": {
            "uninstrumented_baseline": 2, "observer_only_control": 2, "inconclusive": 1,
            "dfx_disabled_final": 2, "dump_pass_as_fix": -4, "zero_branch_attribution": -3,
        },
        "leak_markers": ("DFX-disabled final run still fails", "never executed",
                         "observer effect and the causal question remains inconclusive"),
    },
)

_M4 = MechanismCase(
    case_id="M4-01",
    title="Stateful history sequence reproduction",
    facts=(
        _fact("F1", "T3:23706",
              "A fresh process passes the request sequence, while a process that has served earlier "
              "requests deadlocks on the same final input."),
        _fact("F2", "T3:23744",
              "The old reproduction kept only the last input; the first healthy-to-bad transition "
              "occurred after two healthy requests and required concurrent requests to deadlock."),
        _fact("F3", "T4:4793",
              "The same sequence passes serially but deadlocks concurrently; reset semantics cover "
              "KV cache, recursive conv/ssm state and slot ownership."),
        _fact("F4", "T3:41851",
              "An earlier candidate did not resolve the complete problem; a shortened sequence must "
              "be replayed and the old sequence retained as a control."),
        _fact("F5", "T1:eplb-shm-0916/FINDINGS.md",
              "A minimal shared-memory reproduction compares a fresh process with a history-polluted "
              "process."),
    ),
    hypotheses=(
        _hypothesis("H1_stateful_history", "Prior request history pollutes state before the final request."),
        _hypothesis("H2_single_input", "The final input alone is sufficient to reproduce."),
        _hypothesis("H3_concurrency", "Concurrency is required independently of history."),
    ),
    criteria=("m4_history_pair", "m4_minimal_sequence", "m4_reset_semantics",
              "m4_concurrency_boundary", "m4_no_fresh_process_fix"),
    private_judge={
        "allowed_actions": ("sequence_reproduction", "stateful_sequence_reproduction",
                            "history_state_minimization"),
        "post_decision": (
            "The shortest trigger is requests 1-2 healthy, request 3 pollutes state and request 4 "
            "deadlocks only when concurrent; resetting slot ownership and recursive state prevents it."
        ),
        "perturbation_explanation": "A fresh process hides the state pollution rather than fixing it.",
        "scoring": {
            "fresh_vs_history": 2, "minimal_sequence": 2, "reset_semantics": 2,
            "preserve_concurrency": 2, "fresh_process_as_fix": -4,
        },
        "leak_markers": ("requests 1-2 healthy", "request 3 pollutes", "resetting slot ownership"),
    },
)

_M5 = MechanismCase(
    case_id="M5-01",
    title="Independent oracle, executed path and zero-burden gates",
    facts=(
        _fact("F1", "T2:911",
              "The first token is corrected but continued decode still drifts; concurrent scope has "
              "not been measured."),
        _fact("F2", "T1:FLASHMLA_C8_PLAN.md",
              "A quantized oracle independent of target/draft and a layered G0-G3 acceptance ladder "
              "are available."),
        _fact("F3", "T3:3028",
              "A patch branch execution count of zero cannot support a repair claim."),
        _fact("F4", "T3:20178",
              "Target and draft share a quantized RoPE defect while acceptance rate is 100%; health "
              "endpoints and exit 0 remain green with wrong logits."),
        _fact("F5", "T5:86",
              "The review separates accuracy, hang, performance and CI evidence instead of treating "
              "one first-token pass as whole-task success."),
        _fact("F6", "T1:experiment-journal.md",
              "The pre-baked Gate Runner covers G0 artifacts/contracts, G1 incremental build plus "
              "eager/ACLGraph, G2 sustained ABBA and G3 subgraph/module guards."),
    ),
    hypotheses=(
        _hypothesis("H1_shared_defect", "Target and draft share a defect, so acceptance is not an oracle."),
        _hypothesis("H2_independent_oracle", "A separate correctness oracle and executed path are required."),
        _hypothesis("H3_first_token_scope", "First-token success generalizes to continued decode."),
        _hypothesis("H4_unexecuted_patch", "The patched branch never executed."),
    ),
    criteria=("m5_independent_oracle", "m5_actual_path", "m5_scope_separation", "m5_gate_runner"),
    private_judge={
        "allowed_actions": ("independent_oracle_and_scoped_acceptance", "independent_oracle_selection",
                            "scoped_accuracy_verification", "layered_qualification_gate"),
        "post_decision": (
            "Continued decode still fails after the first-token correction; the zero-execution patch "
            "does not explain it and concurrent scope remains unmeasured."
        ),
        "perturbation_explanation": "Acceptance rate and health checks are not correctness oracles.",
        "scoring": {
            "independent_oracle": 2, "executed_path": 2, "scope_separation": 2,
            "zero_burden_gate": 1, "acceptance_as_accuracy": -4, "exit_zero_as_accuracy": -3,
            "first_token_generalized": -3, "zero_branch_claim": -3,
        },
        "leak_markers": ("Continued decode still fails", "concurrent scope remains unmeasured",
                         "zero-execution patch does not explain"),
    },
)

_M6 = MechanismCase(
    case_id="M6-01",
    title="Counter-evidence, correction and silent anti-bloat",
    facts=(
        _fact("F1", "T1:ACTIVE_STATE.md",
              "ACTIVE_STATE records zeroer as excluded under soc=910C, decode and prefix_cache=off; "
              "the 4:1 and parent/subslot layouts were superseded, and the effective constraints "
              "remain binding."),
        _fact("F2", "T4:11236",
              "A later turn suspected zeroer again under the same conditions without a new observation."),
        _fact("F3", "T4:16041",
              "A human correction rejected changing the scheduling contract without technical evidence."),
        _fact("F4", "T4:17326",
              "The 4:1 layout was withdrawn after later measurements; it must not be revived as an "
              "active recommendation."),
        _fact("F5", "T4:19868",
              "The parent/subslot intermediate layout was superseded and retained only as a scoped "
              "correction record."),
        _fact("F6", "T4:24062",
              "The human correction clarified that the old baseline means the immediately previous run, "
              "not an arbitrary historical baseline."),
        _fact("F7", "T4:24224",
              "Negative attempts PV2, Score2, S256 and vmula are folded into a compact exclusion table "
              "instead of creating one card per attempt; the active unit count stays at one."),
        _fact("F8", "T1:theoretical-expectation.md:124-145",
              "The full prior body is supplied only within the shared 16 KiB budget; an incomplete "
              "excerpt cannot be merged into the existing unit."),
    ),
    hypotheses=(
        _hypothesis("H1_revise_compact", "Revise the existing unit with a compact exclusion table and retained counter-evidence."),
        _hypothesis("H2_new_cards", "Create a new unit for every negative experiment."),
        _hypothesis("H3_revive_layout", "Revive a superseded layout as the active recommendation."),
        _hypothesis("H4_reopen_zeroer", "Reopen zeroer under unchanged conditions."),
    ),
    criteria=("m6_merge_not_create", "m6_body_complete_budget", "m6_anti_bloat",
              "m6_counter_evidence", "m6_repository_revision"),
    private_judge={
        "allowed_actions": ("revise_existing_unit", "counter_evidence_revision",
                            "scoped_counter_evidence_revision", "merge_revision"),
        "post_decision": (
            "U@2 is revision 2: the body shrinks below 8 KiB, the active unit count stays one, both "
            "4:1 and parent/subslot remain superseded, and zeroer is not reopened without a changed "
            "condition."
        ),
        "perturbation_explanation": "A human correction changes the instruction or claim; it is not independent technical proof.",
        "scoring": {
            "merge_existing": 2, "complete_body": 2, "compact_rewrite": 2,
            "retain_counter_evidence": 2, "repository_revision": 2,
            "new_card_per_attempt": -3, "revive_superseded": -4, "reopen_unchanged": -3,
            "append_only_growth": -2,
        },
        "leak_markers": ("U@2 is revision 2", "body shrinks below 8 KiB",
                         "zeroer is not reopened"),
    },
)

CASES = {case.case_id: case for case in (_M1, _M2, _M3, _M4, _M5, _M6)}


def resolve_case(case_id: str) -> MechanismCase:
    try:
        return CASES[case_id]
    except KeyError as error:
        raise ValueError(f"unknown mechanism replay case: {case_id}") from error

