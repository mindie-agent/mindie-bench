"""Trial execution, Roofline HAP scoring, and Sentry trajectory auditing.

[Role]
Public package surface for `mindie_bench.runner`, re-exporting qualification
gates (`G0`..`G3`), RSI candidate gate evaluation (`evaluate_rsi_candidate_gates`,
`RSICandidateResult`), `cann-bench` Roofline HAP & composite operator scoring,
and `cannbot-sentry` dual-verification trajectory auditing.

[Invariants]
- Re-exports only deterministic runner contracts and pure evaluation/auditing helpers.

[Timeouts & Locks]
- Package import performs zero I/O, zero subprocesses, and zero lock acquisitions.

[Side-Effect Policy]
- Side-effect-free module re-exports.
"""

from mindie_bench.runner.cann_sentry_eval import (
    CANONICAL_SWIMLANES,
    OPERATOR_COMPLEXITY_TAXONOMY,
    RSICandidateResult,
    audit_sentry_trajectory,
    classify_operator_complexity,
    compare_swimlane_attributions,
    compute_hap_score,
    compute_operator_composite_score,
)
from mindie_bench.runner.qualification_gate import (
    G0Spec,
    G1Spec,
    G2Spec,
    G3Spec,
    GateCommand,
    GateResult,
    GateState,
    QualificationGateRunner,
    incremental_build_command,
    plan_gate_timeouts,
)
from mindie_bench.runner.rsi_explorer import evaluate_rsi_candidate_gates

__all__ = [
    "CANONICAL_SWIMLANES",
    "G0Spec",
    "G1Spec",
    "G2Spec",
    "G3Spec",
    "GateCommand",
    "GateResult",
    "GateState",
    "OPERATOR_COMPLEXITY_TAXONOMY",
    "QualificationGateRunner",
    "RSICandidateResult",
    "audit_sentry_trajectory",
    "classify_operator_complexity",
    "compare_swimlane_attributions",
    "compute_hap_score",
    "compute_operator_composite_score",
    "evaluate_rsi_candidate_gates",
    "incremental_build_command",
    "plan_gate_timeouts",
]
