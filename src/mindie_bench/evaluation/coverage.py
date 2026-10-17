"""Assess whether declared scope and actual observations answer a metric question.

This module does not load receipts, certify hashes, execute trials or infer missing
observations. Evidence counts must come from a source-bound, independently checked
inventory; matrix accounting remains separate from metric readiness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from mindie_bench.models import Group, TaskSpec

CoverageStatus = Literal["ready", "not_measured", "insufficient_sample", "not_applicable"]


@dataclass(frozen=True)
class MetricTarget:
    metric: str
    question: str
    sample_unit: str = "trial"
    tier: Literal["l1", "l2", "l3"] | None = None
    required_groups: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    required_protocols: tuple[str, ...] = ()
    required_gradients: tuple[str, ...] = ()
    min_samples: int = 1
    min_tasks: int = 0
    min_families: int = 0
    min_domains: int = 0
    min_repetitions: int = 1
    require_npu: bool = False
    require_multicard: bool = False
    require_verified_tasks: bool = False
    require_complete_observation: bool = False
    require_baseline_headroom: bool = False


@dataclass(frozen=True)
class MetricEvidence:
    """Counts for this metric's sample unit, never inferred from planned cells.

    field_samples counts valid samples for which the named prerequisite is
    actually established, not merely a JSON key's presence. For example,
    ``matched_controls`` includes an actual control comparison, and
    ``source_identity`` includes hash verification. Missing fields stay missing.
    observed_task_ids enumerates tasks represented by valid samples only.
    """

    observed_samples: int = 0
    valid_samples: int = 0
    unknown_samples: int = 0
    observed_task_ids: tuple[str, ...] = ()
    task_repetitions: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    field_samples: Mapping[str, int] = field(default_factory=dict)
    protocol_samples: Mapping[str, int] = field(default_factory=dict)
    baseline_successes: int | None = None
    baseline_samples: int | None = None


@dataclass(frozen=True)
class MetricCoverage:
    metric: str
    question: str
    status: CoverageStatus
    reasons: tuple[str, ...]
    sample_unit: str
    observed_samples: int
    valid_samples: int
    unknown_samples: int
    minimum_observed_repetitions: int
    declared_tasks: int
    observed_tasks: int
    declared_families: int
    observed_families: int
    declared_domains: tuple[str, ...]
    observed_domains: tuple[str, ...]
    declared_gradients: tuple[str, ...]
    observed_gradients: tuple[str, ...]
    missing_field_samples: Mapping[str, int]
    missing_protocols: tuple[str, ...]
    missing_gradients: tuple[str, ...]


@dataclass(frozen=True)
class CoverageReport:
    task_count: int
    groups: tuple[str, ...]
    repetitions: int
    metrics: tuple[MetricCoverage, ...]
    execution_readiness: str = "not_checked"

    def to_dict(self) -> dict:
        """Return JSON-compatible values without reading or changing any files."""
        result = asdict(self)
        result["groups"] = list(self.groups)
        result["metrics"] = {
            item.metric: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(item).items()
            }
            for item in self.metrics
        }
        result["interpretation"] = (
            "ready means the configured coverage prerequisites are satisfied; "
            "it does not mean positive benefit, statistical significance or runtime acceptance"
        )
        return result


def default_targets() -> tuple[MetricTarget, ...]:
    """Original-plan questions; sample thresholds are explicit planning minima.

    Twelve tasks/five domains and three repetitions come from the proposal.
    These minima are coverage gates, not a statistical power calculation.
    A caller may declare a narrower finite question with its own MetricTarget.
    """
    controls = ("source_identity", "task_soundness", "outcome", "planned_cell_accounting")
    contrast = controls + (
        "matched_controls",
        "matched_eval_only",
        "randomized_or_counterbalanced_devices",
    )
    return (
        MetricTarget(
            "l0_latency",
            "Component operation latency, separate from unit-test pass counts",
            sample_unit="component_operation",
            required_fields=("component_identity", "operation_latency_sec", "cold_warm_label"),
            required_protocols=("component_microbenchmark",),
            min_samples=30,
        ),
        MetricTarget(
            "task_completion",
            "Descriptive completion on the declared valid-attempt denominator",
            required_fields=controls,
            min_tasks=1,
            require_verified_tasks=True,
        ),
        MetricTarget(
            "flagship_scope",
            "Original twelve-task, five-domain, single- and multi-card scope",
            tier="l3",
            required_fields=controls,
            min_tasks=12,
            min_domains=5,
            min_repetitions=3,
            require_npu=True,
            require_multicard=True,
            require_verified_tasks=True,
            require_complete_observation=True,
        ),
        MetricTarget(
            "completion_lift",
            "Controlled completion lift with baseline headroom",
            sample_unit="matched_pair",
            required_groups=("G0", "G2"),
            required_fields=contrast,
            min_tasks=12,
            min_domains=5,
            min_repetitions=3,
            require_verified_tasks=True,
            require_complete_observation=True,
            require_baseline_headroom=True,
        ),
        MetricTarget(
            "total_cost",
            "Controlled total delivery cost including required seed setup",
            sample_unit="matched_pair",
            required_groups=("G0", "G2"),
            required_fields=contrast
            + ("delivery_wall_clock_sec", "complete_usage", "seed_inclusive_cost", "usd_prices"),
            require_complete_observation=True,
        ),
        MetricTarget(
            "localization_speedup",
            "Time/turns to observed target-source access, a localization proxy",
            sample_unit="matched_pair",
            required_groups=("G0", "G2"),
            required_fields=contrast + ("both_sides_source_access",),
        ),
        MetricTarget(
            "root_cause_speedup",
            "Time to an independently judged correct semantic root-cause diagnosis",
            sample_unit="matched_pair",
            required_groups=("G0", "G2"),
            required_fields=contrast
            + ("diagnosis_timestamp", "root_cause_oracle", "diagnosis_judgement"),
        ),
        MetricTarget(
            "npu_attempt_savings",
            "Savings in actually admitted and completed NPU test attempts",
            sample_unit="matched_pair",
            tier="l3",
            required_groups=("G0", "G2"),
            required_fields=contrast
            + ("npu_attempt_id", "device_execution_started", "device_execution_completed"),
            require_npu=True,
        ),
        MetricTarget(
            "routing_generalization",
            "Routing applicability across actual positive and negative requests",
            sample_unit="labelled_request",
            required_fields=("request_identity", "applicability_label", "trigger_event_definition"),
            required_protocols=("explicit_positive", "implicit_positive", "clear_negative"),
            min_samples=100,
        ),
        MetricTarget(
            "retrieval_quality",
            "Ranked retrieval against independently judged applicable relevance sets",
            sample_unit="judged_query",
            required_fields=(
                "foreground_query",
                "ordered_result_refs",
                "exact_response_identity",
                "independent_relevance_labels",
                "relevant_set_denominator",
            ),
        ),
        MetricTarget(
            "knowledge_gradients",
            "Transfer, composition and stale-memory guard beyond near-exact reuse",
            sample_unit="matched_pair",
            required_groups=("G1", "G2"),
            required_fields=contrast + ("knowledge_condition_identity",),
            required_gradients=("near_exact", "analogical", "compositional", "stale"),
            require_verified_tasks=True,
        ),
        MetricTarget(
            "no_hit_abstention",
            "Behavior when independently judged relevant knowledge is absent",
            required_fields=controls + ("judged_no_hit", "abstention_or_abandonment"),
            required_gradients=("no_hit",),
        ),
        MetricTarget(
            "knowledge_attribution",
            "Separate routing loss and content gain under matched activation protocols",
            sample_unit="matched_protocol_set",
            required_fields=contrast
            + ("matched_protocol_controls", "exact_delivered_knowledge", "activation_protocol"),
            required_protocols=(
                "baseline",
                "static",
                "dynamic_natural",
                "dynamic_explicit",
                "oracle",
            ),
        ),
        MetricTarget(
            "closed_loop_delivery",
            "Native Stop, capture and exact own-seed delivery, distinct from solving",
            required_groups=("G3",),
            required_fields=controls
            + ("native_stop", "capture_completed", "local_git_identity", "own_seed_lineage"),
            min_tasks=1,
        ),
    )


def _count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _families(tasks: Sequence[TaskSpec]) -> set[tuple[str, str]]:
    # A cluster label or repeated hardware run is not an independent defect.
    return {
        (str(task.task.source["repository"]), str(task.task.source["fix_commit"]))
        for task in tasks
        if task.task.source.get("repository") and task.task.source.get("fix_commit")
    }


def assess_coverage(
    tasks: Sequence[TaskSpec],
    *,
    groups: Sequence[Group | str],
    repetitions: int,
    targets: Sequence[MetricTarget] | None = None,
    evidence: Mapping[str, MetricEvidence] | None = None,
) -> CoverageReport:
    """Evaluate declared metric prerequisites without executing or retrying work."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    group_names = tuple(Group(group).value for group in groups)
    if not group_names or len(set(group_names)) != len(group_names):
        raise ValueError("groups must be nonempty and unique")
    task_ids = [task.task.instance_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task identities must be unique")
    selected_targets = tuple(default_targets() if targets is None else targets)
    if len({target.metric for target in selected_targets}) != len(selected_targets):
        raise ValueError("metric targets must be unique")
    observations = evidence or {}
    if set(observations) - {target.metric for target in selected_targets}:
        raise ValueError("evidence contains an unconfigured metric")
    results = []
    for target in selected_targets:
        for name in ("min_samples", "min_tasks", "min_families", "min_domains", "min_repetitions"):
            _count(getattr(target, name), name)
        if not target.metric or target.min_samples < 1 or target.min_repetitions < 1:
            raise ValueError("metric, min_samples and min_repetitions must be nonempty/positive")
        for group in target.required_groups:
            Group(group)
        selected = [task for task in tasks if target.tier is None or target.tier in task.environment]
        if target.require_npu:
            selected = [
                task
                for task in selected
                if any(
                    env.npu_count > 0
                    for tier, env in task.environment.items()
                    if target.tier is None or tier == target.tier
                )
            ]
        observed = observations.get(target.metric, MetricEvidence())
        for name in ("observed_samples", "valid_samples", "unknown_samples"):
            _count(getattr(observed, name), name)
        if observed.valid_samples + observed.unknown_samples > observed.observed_samples:
            raise ValueError("valid plus unknown samples exceed observed samples")
        if len(set(observed.observed_task_ids)) != len(observed.observed_task_ids):
            raise ValueError("observed task identities must be unique")
        selected_by_id = {task.task.instance_id: task for task in selected}
        if set(observed.observed_task_ids) - selected_by_id.keys():
            raise ValueError(f"{target.metric}: observed task is outside the selected scope")
        if observed.observed_task_ids and not observed.valid_samples:
            raise ValueError("observed_task_ids require valid samples")
        if len(observed.observed_task_ids) > observed.valid_samples:
            raise ValueError("observed task count exceeds valid samples")
        if set(observed.task_repetitions) - set(observed.observed_task_ids):
            raise ValueError("repetition evidence requires a corresponding observed task")
        for values in observed.task_repetitions.values():
            if len(set(values)) != len(values) or any(
                type(value) is not int or not 1 <= value <= repetitions for value in values
            ):
                raise ValueError("observed repetitions must be unique and within the selected plan")
        for values in (observed.field_samples, observed.protocol_samples):
            for name, count in values.items():
                _count(count, name)
                if count > observed.valid_samples:
                    raise ValueError("field/protocol count exceeds valid samples")
        if (observed.baseline_samples is None) != (observed.baseline_successes is None):
            raise ValueError("baseline successes and samples must be provided together")
        if observed.baseline_samples is not None:
            _count(observed.baseline_samples, "baseline_samples")
            _count(observed.baseline_successes, "baseline_successes")
            if observed.baseline_successes > observed.baseline_samples:
                raise ValueError("baseline successes exceed samples")
            if observed.baseline_samples > observed.valid_samples:
                raise ValueError("baseline denominator exceeds this metric's valid matched samples")
        actual = [selected_by_id[identity] for identity in observed.observed_task_ids]
        missing_fields = {
            name: observed.valid_samples - observed.field_samples.get(name, 0)
            for name in target.required_fields
            if not observed.valid_samples or observed.field_samples.get(name, 0) < observed.valid_samples
        }
        missing_protocols = tuple(
            name for name in target.required_protocols if not observed.protocol_samples.get(name, 0)
        )
        gradients = {task.mindie_knowledge.knowledge_gradient for task in actual}
        missing_gradients = tuple(name for name in target.required_gradients if name not in gradients)
        reasons = []
        unavailable = set(target.required_groups) - set(group_names)
        inapplicable = bool(
            unavailable or ((target.tier or target.require_npu) and tasks and not selected)
        )
        if unavailable:
            reasons.append(f"required comparison groups not selected: {', '.join(sorted(unavailable))}")
        if (target.tier or target.require_npu) and tasks and not selected:
            reasons.append("selected tasks have no applicable tier/NPU environment")
        if observed.valid_samples == 0:
            reasons.append("no valid observed samples; planned tasks/repetitions are not measurements")
        if missing_fields:
            reasons.append("prerequisites unmeasured or incomplete: " + ", ".join(missing_fields))
        if missing_protocols:
            reasons.append("missing observed protocols: " + ", ".join(missing_protocols))
        if target.require_baseline_headroom and not observed.baseline_samples:
            reasons.append("baseline outcome denominator is unmeasured")
        deficient = []
        counts = (
            ("valid samples", observed.valid_samples, target.min_samples),
            ("observed tasks", len(actual), target.min_tasks),
            ("observed independent upstream fixes", len(_families(actual)), target.min_families),
            (
                "observed business domains",
                len({task.task.domain_category for task in actual}),
                target.min_domains,
            ),
            ("planned repetitions", repetitions, target.min_repetitions),
        )
        for label, count, minimum in counts:
            if count < minimum:
                deficient.append(f"{label}: {count} < {minimum}")
        minimum_observed_repetitions = min(
            (len(observed.task_repetitions.get(task.task.instance_id, ())) for task in actual),
            default=0,
        )
        if target.min_repetitions > 1 and minimum_observed_repetitions < target.min_repetitions:
            deficient.append(
                "actual per-task repetition coverage: "
                f"{minimum_observed_repetitions} < {target.min_repetitions}"
            )
        if missing_gradients:
            deficient.append("missing observed knowledge gradients: " + ", ".join(missing_gradients))
        if target.require_verified_tasks:
            if not actual:
                deficient.append("no valid observed task identity establishes verified task coverage")
            elif any(task.task.evidence != "verified" for task in actual):
                deficient.append(
                    "candidate/synthetic tasks cannot establish verified engineering benefit"
                )
        if target.require_multicard and not any(
            env.npu_count > 1 for task in actual for env in task.environment.values()
        ):
            deficient.append("no observed multi-card task")
        if target.require_complete_observation and observed.observed_samples != observed.valid_samples:
            deficient.append("invalid/unknown samples remain; valid-subset coverage is incomplete")
        if (
            target.require_baseline_headroom
            and observed.baseline_samples
            and observed.baseline_successes == observed.baseline_samples
        ):
            deficient.append(
                "baseline completion is at ceiling; positive completion lift is unidentifiable here"
            )
        reasons.extend(deficient)
        unmeasured = (
            not observed.valid_samples
            or bool(missing_fields or missing_protocols)
            or (target.require_baseline_headroom and not observed.baseline_samples)
        )
        status: CoverageStatus = (
            "not_applicable"
            if inapplicable
            else "not_measured"
            if unmeasured
            else "insufficient_sample"
            if deficient
            else "ready"
        )
        if not reasons:
            reasons.append(
                "configured coverage satisfied; inspect the measured effect, including zero or harm"
            )
        results.append(
            MetricCoverage(
                metric=target.metric,
                question=target.question,
                status=status,
                reasons=tuple(reasons),
                sample_unit=target.sample_unit,
                observed_samples=observed.observed_samples,
                valid_samples=observed.valid_samples,
                unknown_samples=observed.unknown_samples,
                minimum_observed_repetitions=minimum_observed_repetitions,
                declared_tasks=len(selected),
                observed_tasks=len(actual),
                declared_families=len(_families(selected)),
                observed_families=len(_families(actual)),
                declared_domains=tuple(sorted({task.task.domain_category for task in selected})),
                observed_domains=tuple(sorted({task.task.domain_category for task in actual})),
                declared_gradients=tuple(
                    sorted({task.mindie_knowledge.knowledge_gradient for task in selected})
                ),
                observed_gradients=tuple(sorted(gradients)),
                missing_field_samples=missing_fields,
                missing_protocols=missing_protocols,
                missing_gradients=missing_gradients,
            )
        )
    return CoverageReport(len(tasks), group_names, repetitions, tuple(results))
