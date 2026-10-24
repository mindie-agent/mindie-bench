"""Public projections for product-matrix publication evidence.

[Role] Turn native receipts and runtime manifests into path-free, auditable
product measurements that can be republished by release CI.
[Invariants] Product identity comes only from `experiment_context.cell`; the
receipt's internal G capability mode is retained as an internal setting and is
never promoted to a product arm. Missing token/cost data stays null and
incomplete, not zero. Endpoint addresses, commands, local paths, credentials
and native logs are removed, while measured numbers, content hashes and the
bound original SHAs remain.
[Timeouts & Locks] Pure in-memory transforms; no waits, subprocesses or locks.
[Side-Effect Policy] Returns dictionaries only; never writes or executes.
"""

from __future__ import annotations

from mindie_bench.io import digest
from mindie_bench.models import TrialReceipt
from mindie_bench.scorer.comparator import CONTROL_KEYS

PRIMARY_KEYS = {
    "reward",
    "resolved",
    "tokens",
    "cost_complete",
    "estimated_cost_usd",
    "model_calls",
    "total_wall_clock_sec",
    "delivery_wall_clock_sec",
    "npu_test_runs",
    "remote_test_invocations",
    "manual_interventions",
    "zero_human_intervention",
    "total_turns",
    "npu_execution_wait_sec",
    "knowledge_query_sec",
    "llm_and_exploration_sec",
    "knowledge_measurement_basis",
    "turns_to_first_localization",
    "localization_penalty_turns",
    "time_to_first_localization_sec",
    "time_to_first_target_edit_sec",
    "foreground_knowledge_queries_count",
    "foreground_knowledge_explains_count",
}

AUXILIARY_KEYS = {
    # Measured segments of delivery. They may overlap and do not cover cleanup,
    # so they qualify delivery_wall_clock_sec rather than partition it.
    "preflight_wall_clock_sec",
    "capture_wall_clock_sec",
    "captured_corpus_sha256",
    "capture_model_calls",
    "model_call_measurement_basis",
    "verifier_wall_clock_sec",
    "post_flight_npu_clean",
    "remote_test_budget_invocations",
    "remote_delivery_failures",
    "model_delivery_failures",
    "budget_exceeded",
    "agent_budget_exhausted",
    "business_terminal_status",
    "business_resolved",
    "business_reward",
}

NESTED_AUXILIARY = {
    "capture": {
        "status",
        "corpus_lint",
        "publication",
        "consumer",
        "tokens",
        "estimated_cost_usd",
        "model_calls",
    },
    "knowledge_measurement": {
        "query_count",
        "explain_count",
        "successful_calls",
        "failed_calls",
        "timing_complete",
        "basis",
    },
    "lifecycle_budget": {
        "declared_total_sec",
        "measured_total_sec",
        "overrun_sec",
        "overrun",
        "execution_allowance_sec",
        "post_agent_reserve_sec",
        "source",
        "budget_exceeded",
        "unknown_priority",
    },
    "remote_settlement": {"unknown", "settled", "pending", "attempts"},
}


def _mapping(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _selected(value: dict, keys) -> dict:
    return {key: value[key] for key in keys if key in value}


def public_identity(identity: dict) -> dict:
    """Keep installed versions/commits and source hashes, never local paths."""
    components = {}
    for name, installed in sorted(_mapping(identity.get("components")).items()):
        entry = _mapping(installed)
        direct = _mapping(entry.get("direct_url"))
        vcs = _mapping(direct.get("vcs_info"))
        components[name] = {
            "distribution": entry.get("distribution"),
            "version": entry.get("version"),
            "revision": (
                vcs.get("commit_id")
                if vcs
                else entry.get("revision", entry.get("commit"))
            ),
            "commit": vcs.get("commit_id") if vcs else entry.get("commit"),
            "requested_revision": (
                vcs.get("requested_revision") if vcs else entry.get("requested_revision")
            ),
            "subdirectory": vcs.get("subdirectory") if vcs else entry.get("subdirectory"),
            "editable": (
                _mapping(direct.get("dir_info")).get("editable")
                if direct
                else entry.get("editable")
            ),
        }
    return {"components": components}


def public_runtime(runtime: dict) -> dict:
    """Keep the runtime identity contract; strip paths and host prefixes."""
    runtime = _mapping(runtime)
    native = _mapping(runtime.get("native_product"))
    result = _selected(runtime, {"harness_binary_sha256"})
    if native:
        result["native_product"] = {
            **_selected(native, {"contribution_enabled"}),
            "runtime_identity": public_identity(_mapping(native.get("runtime_identity"))),
        }
    return result


def public_catalog(catalog: dict) -> dict:
    catalog = _mapping(catalog)
    return {
        "schema": catalog.get("schema"),
        "units": catalog.get("units"),
        "sha256": catalog.get("sha256") or (digest(catalog) if catalog else None),
        "source_revision": catalog.get("source_revision"),
        "empty_proof_sha256": catalog.get("empty_proof_sha256"),
    }


def canonical_runtime_modules(modules: dict) -> dict:
    result = {}
    for name, installed in sorted((modules or {}).items()):
        entry = dict(installed or {})
        result[name] = {
            "version": entry.get("version"),
            "revision": entry.get("revision", entry.get("commit")),
        }
    return result


def canonical_native_plugin(plugin: dict | None) -> dict | None:
    if not plugin:
        return None
    return {
        "name": plugin.get("name"),
        "version": plugin.get("version"),
        "revision": plugin.get("revision"),
        "contribution_enabled": plugin.get("contribution_enabled"),
    }


def canonical_benchmark(benchmark: dict) -> dict:
    return {
        "version": benchmark.get("version"),
        "revision": benchmark.get("revision"),
        "source_sha256": benchmark.get("source_sha256"),
    }


def public_timeline(timeline: dict) -> dict:
    """Explicit safe timeline projection; never republish arbitrary error text."""
    timeline = _mapping(timeline)
    result = _selected(
        timeline,
        {
            "absolute_available",
            "missing_absolute",
            "inference_performed",
            "seed_completed_at",
            "capture_completed_at",
            "corpus_frozen_at",
            "downstream_started_at",
        },
    )
    stages = []
    for stage in timeline.get("stage_observations", []):
        stage = _mapping(stage)
        safe = _selected(
            stage,
            {
                "stage_id",
                "status",
                "started_sec",
                "ended_sec",
                "elapsed_sec",
                "started_at",
                "ended_at",
            },
        )
        if stage.get("error") is not None:
            try:
                safe["error_sha256"] = digest(stage["error"])
            except (TypeError, ValueError):
                safe["error_sha256"] = "unprojectable"
        stages.append(safe)
    result["stage_observations"] = stages
    return result


def public_cost_ledger(ledger: dict) -> dict:
    """Keep public roles/totals plus a digest over the public ledger contract."""
    ledger = _mapping(ledger)
    public = {
        key: ledger.get(key)
        for key in (
            "roles",
            "attribution",
            "unknown_usage",
            "prerequisite_unknown_usage",
            "unpriced_usage",
            "known_totals",
            "ledger_sha256",
        )
        if key in ledger
    }
    payload = {
        key: value
        for key, value in public.items()
        if key not in {"ledger_sha256", "public_ledger_sha256"}
    }
    public["public_ledger_sha256"] = digest(payload)
    return public


def public_admission(admission: dict) -> dict:
    """Admission context with safe timeline/cost projections and no locators."""
    admission = _mapping(admission)
    result = {
        key: admission[key]
        for key in (
            "schema",
            "status",
            "reasons",
            "manifest_sha256",
            "content_identity_sha256",
            "split_manifest_sha256",
            "admission_sha256",
            "target",
            "shared_mechanism",
            "seed",
            "heldout",
            "timeline",
            "cost_ledger",
            "prerequisite_cost_complete",
            "lifecycle_cost_complete",
            "prerequisite_unknown_usage",
            "unknown_usage",
            "unpriced_usage",
            "limitations",
            "claim_scope",
            "claim_ready",
            "prerequisite_claim_ready",
            "lifecycle_claim_ready",
            "verification",
        )
        if key in admission
    } | {
        "stages": [
            _selected(_mapping(stage), {"role", "sha256", "verified", "identity_basis"})
            for stage in admission.get("stages", [])
            if isinstance(stage, dict)
        ]
    }
    if "timeline" in result:
        result["timeline"] = public_timeline(result["timeline"])
    if "cost_ledger" in result:
        result["cost_ledger"] = public_cost_ledger(result["cost_ledger"])
    return result


def public_experiment_context(context: dict) -> dict:
    """Derive product identity from the real cell/profile, never top-level guesses."""
    context = _mapping(context)
    result = _selected(
        context,
        {
            "schema",
            "cell",
            "internal_capability_mode",
            "task_sha256",
            "execution_tier",
            "max_wall_clock_sec",
            "p3_admission_sha256",
        },
    )
    cell = _mapping(context.get("cell"))
    profile = _mapping(context.get("profile"))
    if cell:
        result["cell"] = {
            key: cell.get(key)
            for key in (
                "group",
                "profile_id",
                "cohort",
                "protocol",
                "task_id",
                "task_family",
                "hardware_tier",
                "repetition",
                "order",
            )
        }
        result.update(
            profile_id=cell.get("profile_id"),
            profile_group=cell.get("group"),
            cohort=cell.get("cohort"),
            protocol=cell.get("protocol"),
        )
    else:
        for key in ("profile_id", "profile_group", "cohort", "protocol"):
            if key in context:
                result[key] = context[key]
            elif key in profile:
                result[key] = profile[key]
    if "actual_runtime" in context:
        result["actual_runtime"] = public_runtime(context["actual_runtime"])
    if "p3_admission" in context:
        result["p3_admission"] = public_admission(context["p3_admission"])
    return result


def _public_tokens(tokens) -> dict:
    tokens = _mapping(tokens)
    return {
        key: tokens.get(key)
        for key in (
            "input_uncached",
            "input_cached",
            "cache_creation",
            "output",
            "indexer_small_model_total",
            "complete",
        )
    }


def _public_capture(capture) -> dict:
    capture = _mapping(capture)
    result = _selected(capture, {"status", "estimated_cost_usd", "model_calls"})
    for key in ("corpus_lint", "publication", "consumer"):
        nested = _mapping(capture.get(key))
        if key == "corpus_lint":
            result[key] = _selected(nested, {"status"})
        else:
            result[key] = _selected(nested, {"revision", "sha256", "status"})
    if "tokens" in capture:
        result["tokens"] = _public_tokens(capture["tokens"])
    return result


def _public_auxiliary(auxiliary: dict) -> dict:
    result = _selected(_mapping(auxiliary), AUXILIARY_KEYS)
    for name, allowed in NESTED_AUXILIARY.items():
        if name not in auxiliary:
            continue
        value = auxiliary[name]
        if name == "capture":
            result[name] = _public_capture(value)
        else:
            result[name] = _selected(_mapping(value), allowed)
    if "capture_provider_tokens" in auxiliary:
        result["capture_provider_tokens"] = _public_tokens(auxiliary["capture_provider_tokens"])
    return result


def _public_primary(primary: dict) -> dict:
    result = _selected(_mapping(primary), PRIMARY_KEYS)
    if "tokens" in primary:
        result["tokens"] = _public_tokens(primary["tokens"])
    return result


def public_product_receipt(row: TrialReceipt) -> dict:
    """Project a native receipt for public product evidence."""
    source = row.environment_fingerprint
    fingerprint = {key: source[key] for key in CONTROL_KEYS if key in source}
    for key in ("provider_url", "remote_environment"):
        if key in fingerprint:
            fingerprint[key] = {"private_control_identity": digest(fingerprint[key])}
    for key in (
        "components",
        "component_revisions",
        "knowledge_mode",
        "skills_sha256",
        "corpus_view_sha256",
        "evidence",
        "model_catalog_sha256",
    ):
        if key in source:
            fingerprint[key] = source[key]
    if "experiment_context" in source:
        fingerprint["experiment_context"] = public_experiment_context(source["experiment_context"])
    return {
        "schema_version": row.schema_version,
        "trial_id": row.trial_id,
        "instance_id": row.instance_id,
        "repetition": row.repetition,
        "tier": row.tier,
        "ablation_group": row.ablation_group.value,
        "knowledge_gradient": row.knowledge_gradient,
        "task_role": row.task_role,
        "cluster": row.cluster,
        "task_digest": row.task_digest,
        "experiment_digest": row.experiment_digest,
        "environment_fingerprint": fingerprint,
        "terminal_status": row.terminal_status.value,
        "failure_reason": (
            None if row.terminal_status.value == "completed_pass" else "details_retained_in_private_evidence"
        ),
        "primary_metrics": _public_primary(row.primary_metrics),
        "auxiliary_metrics": _public_auxiliary(row.auxiliary_metrics),
        "errors": [
            {key: error.get(key) for key in ("stage", "type") if error.get(key) is not None}
            for error in row.errors
        ],
        "artifacts": {},
        "evidence": row.evidence,
    }


def public_runtime_manifest(raw: dict, benchmark: dict) -> dict:
    """Bind one prepared native runtime manifest to its safe public identity."""
    raw = _mapping(raw)
    runtime = _mapping(raw.get("runtime_snapshot"))
    identity = _mapping(_mapping(runtime.get("native_product")).get("runtime_identity"))
    runtime_modules = public_identity(identity)["components"]
    plugin = raw.get("native_plugin")
    derived = _mapping(raw.get("derived_inputs"))
    tasks = {
        task_id: _selected(_mapping(value), {"task_sha256", "oracle_sha256"})
        for task_id, value in sorted(_mapping(derived.get("tasks")).items())
    }
    catalog = public_catalog(raw.get("catalog") or {})
    return {
        "schema": "mindie-product-public-runtime/1",
        "profile_id": raw.get("profile_id"),
        "group": raw.get("product_group"),
        "runtime_modules": canonical_runtime_modules(runtime_modules),
        "native_plugin": canonical_native_plugin(plugin),
        "benchmark": canonical_benchmark(benchmark),
        "catalog": catalog,
        "catalog_sha256": digest(catalog),
        "conditions": derived.get("conditions"),
        "budget": derived.get("budget"),
        "tasks": tasks,
    }
