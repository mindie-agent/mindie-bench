"""Trial scoring and immutable receipt finalization.

[Role]
Scores a finished trajectory and applies final remote-settlement, lifecycle
budget, and receipt construction once the verifier outcome is known.
[Invariants]
- A verified primary verdict is immutable: scoring or settlement failures are
  recorded as secondary evidence and never erase `resolved`/`reward`.
- Unknown remote effects are reported as `UNKNOWN`; the prior business verdict
  remains separately labeled in `business_terminal_status`.
[Timeouts & Locks]
- Pure in-memory scoring plus bounded gateway settlement inspection; no waits
  or locks are acquired here.
[Side-Effect Policy]
- Writes only the trial's `source_visibility.json`, `broker_knowledge.json`,
  `trajectory.json`, and immutable `trial_receipt.json`.
"""

from __future__ import annotations

import json
import platform
import time

from mindie_bench.evaluation.corpus_linter import CorpusLintRejected
from mindie_bench.io import digest, tree_hash, write_json
from mindie_bench.models import TerminalStatus, TrialReceipt, UnknownExternalEffects
from mindie_bench.runner.drivers import driver_for
from mindie_bench.runner.dsh_summary import optional_host
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.phase_budget import lifecycle_overrun
from mindie_bench.runner.process import stream_process
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.trial_preflight import native_capture_payload, require_corpus_lint
from mindie_bench.scorer.metrics import trajectory_metrics


def _mark_private_knowledge_unmeasured(primary, auxiliary):
    """Native Codex shells hide private CLI and direct corpus access from MCP counters."""
    measurement = auxiliary["knowledge_measurement"]
    measurement.update(
        observed_named_query_count_lower_bound=measurement["query_count"],
        observed_named_explain_count_lower_bound=measurement["explain_count"],
        query_count=None,
        explain_count=None,
        basis="native_private_knowledge_unmeasured/1",
        coverage="unmeasured",
        scope=(
            "Named MCP events are only a lower bound. Private knowledge_cli.py operations "
            "(including describe) and direct corpus reads require a separate native trace audit."
        ),
    )
    primary.update(
        foreground_knowledge_queries_count=None,
        foreground_knowledge_explains_count=None,
        knowledge_query_sec=None,
        llm_and_exploration_sec=None,
        knowledge_measurement_basis=measurement["basis"],
    )
    auxiliary.update(
        knowledge_queries_count=None,
        knowledge_explains_count=None,
        tool_timing_complete=False,
        timing_overlaps_or_missing=True,
    )
    timing = auxiliary["tool_timing"]
    # Preserve native_* observations while withholding totals that omit private CLI calls.
    for field in ("knowledge", "combined"):
        timing[field] = dict.fromkeys(timing[field]) | {"complete": False}
    timing.update(
        knowledge_measurement_basis=measurement["basis"],
        remainder_reason="private_cli_and_direct_corpus_access_unmeasured",
        cross_component_overlap_sec=None,
        within_wall_clock=None,
        scope=(
            "native_knowledge/native_combined cover named tool events only; "
            "private CLI calls and direct corpus reads have no measured knowledge duration"
        ),
    )


def score_trial_trajectory(
    *,
    driver,
    gateway,
    result,
    trajectory,
    config,
    environment,
    task,
    trial_dir,
    task_root,
    wall_clock,
    tier,
    fingerprint,
    errors,
    primary,
    auxiliary,
    primary_verdict,
    status,
    reason,
):
    """Score a finished trajectory; failures are secondary to a verified verdict."""
    try:
        trajectory = driver.trajectory()
        if result:
            trajectory.total_turns = max(trajectory.total_turns, gateway.model_calls)
            trajectory.human_interventions = 0
            if gateway.model_calls:
                observed = gateway.metered_tokens()
                if not observed.complete:
                    for field in ["input_uncached", "input_cached", "cache_creation", "output"]:
                        setattr(
                            observed,
                            field,
                            max(getattr(observed, field), getattr(trajectory.tokens, field)),
                        )
                trajectory.tokens = observed
        source_visibility = None
        broker_knowledge = None
        if config.backend == "native" and result and result.started and environment is not None:
            from mindie_bench.evaluation.source_visibility import observe_source_visibility

            source_visibility = observe_source_visibility(
                trial_dir / "native_events.jsonl",
                trajectory,
                environment,
                task.mindie_knowledge.target_files,
                public_root=task_root / "environment",
            )
            write_json(trial_dir / "source_visibility.json", source_visibility, exclusive=True)
        if (
            config.backend == "native"
            and gateway
            and not fingerprint.get("native_product")
            and (trial_dir / "broker.jsonl").is_file()
        ):
            from mindie_bench.evaluation.broker_knowledge import summarize_broker_knowledge

            broker_knowledge = summarize_broker_knowledge(
                trial_dir / "broker.jsonl",
                process_origin_monotonic_sec=(
                    result.started_monotonic_sec if result is not None else None
                ),
                preflight_finished_monotonic_sec=fingerprint.get(
                    "mcp_preflight_finished_monotonic_sec"
                ),
            )
            write_json(trial_dir / "broker_knowledge.json", broker_knowledge, exclusive=True)
        measured_primary, measured_aux = trajectory_metrics(
            trajectory,
            task,
            config.harness,
            wall_clock,
            tier=tier,
            source_visibility=source_visibility,
            broker_knowledge=broker_knowledge,
        )
        primary = dict(measured_primary, **primary)
        auxiliary = dict(measured_aux, **auxiliary)
        if fingerprint.get("native_product") and config.harness.name == "codex":
            _mark_private_knowledge_unmeasured(primary, auxiliary)
        if any(error["stage"] == "capture" for error in errors):
            primary["cost_complete"] = False
            partial = auxiliary.get("capture_provider_tokens", {})
            primary["tokens"]["indexer_small_model_total"] = sum(
                partial.get(field, 0) for field in ["input_uncached", "input_cached", "output"]
            )
            primary["tokens"]["complete"] = False
        capture = auxiliary.get("capture")
        if capture and capture.get("tokens") is not None:
            metering = capture["tokens"]
            if (
                not isinstance(metering.get("total"), int)
                or metering["total"] < 0
                or type(metering.get("complete")) is not bool
            ):
                raise ValueError("capture adapter has invalid indexer metering")
            primary["tokens"]["indexer_small_model_total"] = metering["total"]
            primary["tokens"]["complete"] = primary["tokens"]["complete"] and metering["complete"]
            capture_cost = capture.get("estimated_cost_usd")
            if (
                primary["estimated_cost_usd"] is not None
                and type(capture_cost) in {int, float}
                and capture_cost >= 0
            ):
                primary["estimated_cost_usd"] += capture_cost
            else:
                primary["cost_complete"] = False
            primary["cost_complete"] = primary["cost_complete"] and metering["complete"]
    except Exception as exc:
        errors.append(dict(stage="scoring", type=type(exc).__name__, message=str(exc)))
        if primary_verdict is not None:
            auxiliary.setdefault("business_terminal_status", primary_verdict.value)
        elif status != TerminalStatus.UNKNOWN:
            status, reason = TerminalStatus.INFRA, "trajectory scoring failed"
    return dict(
        trajectory=trajectory,
        primary=primary,
        auxiliary=auxiliary,
        status=status,
        reason=reason,
    )


def finalize_trial_receipt(
    *,
    gateway,
    status,
    reason,
    primary_verdict,
    primary,
    auxiliary,
    trial_started,
    experiment_context,
    agent_phase_budget,
    errors,
    verifier_seconds,
    trial_dir,
    trajectory,
    trial_id,
    task,
    repetition,
    tier,
    group,
    task_digest,
    config,
    fingerprint,
):
    """Apply final settlement/budget accounting and write the immutable receipt."""
    if gateway:
        settlement = gateway.remote_state()
        auxiliary["remote_settlement"] = settlement
        if settlement["unknown"]:
            if status != TerminalStatus.UNKNOWN:
                auxiliary["status_before_remote_unknown"] = dict(
                    terminal_status=status.value, failure_reason=reason
                )
                if status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
                    auxiliary.setdefault("business_terminal_status", status.value)
            reason = "remote effects are unknown; inspect the original broker journal; no retry"
            errors.append(
                dict(stage="remote_settlement", type="UnknownExternalEffects", message=reason)
            )
            status = TerminalStatus.UNKNOWN
    if status == TerminalStatus.UNKNOWN or (
        status == TerminalStatus.INFRA and primary_verdict is None
    ):
        primary.update(resolved=None, reward=None)
    primary["delivery_wall_clock_sec"] = time.monotonic() - trial_started
    if experiment_context is not None and "max_wall_clock_sec" in experiment_context:
        declared = experiment_context["max_wall_clock_sec"]
        lifecycle = lifecycle_overrun(declared, primary["delivery_wall_clock_sec"])
        lifecycle["source"] = "experiment_context.max_wall_clock_sec"
        if agent_phase_budget is not None:
            lifecycle["execution_allowance_sec"] = agent_phase_budget["execution_allowance_sec"]
            lifecycle["post_agent_reserve_sec"] = agent_phase_budget["post_agent_reserve_sec"]
        auxiliary["lifecycle_budget"] = lifecycle
        if lifecycle["overrun"]:
            lifecycle["budget_exceeded"] = True
            auxiliary["budget_exceeded"] = True
            if status == TerminalStatus.UNKNOWN:
                lifecycle["unknown_priority"] = True
            errors.append(
                dict(
                    stage="budget",
                    type="BudgetOverrun",
                    message=(
                        "measured lifecycle exceeded the declared product budget by "
                        f"{lifecycle['overrun_sec']:.3f}s"
                    ),
                )
            )
    auxiliary["verifier_wall_clock_sec"] = verifier_seconds
    write_json(trial_dir / "trajectory.json", trajectory.model_dump(mode="json"), exclusive=True)
    receipt = TrialReceipt(
        trial_id=trial_id,
        instance_id=task.task.instance_id,
        repetition=repetition,
        tier=tier,
        ablation_group=group,
        knowledge_gradient=task.mindie_knowledge.knowledge_gradient,
        task_role=task.task.role,
        cluster=task.task.cluster,
        task_digest=task_digest,
        experiment_digest=digest(config.model_dump(mode="json")),
        environment_fingerprint=fingerprint,
        terminal_status=status,
        failure_reason=reason,
        primary_metrics=primary,
        auxiliary_metrics=auxiliary,
        errors=errors,
        artifacts={
            "private_trial_dir": str(trial_dir),
            **(
                {"captured_corpus": auxiliary["captured_corpus_path"]}
                if "captured_corpus_path" in auxiliary
                else {}
            ),
        },
        evidence=task.task.evidence,
    )
    write_json(trial_dir / "trial_receipt.json", receipt.model_dump(mode="json"), exclusive=True)
    return receipt


def run_g3_capture(
    *,
    task_root,
    task,
    config,
    group,
    status,
    primary,
    trajectory,
    profile,
    workspace,
    driver,
    trial_dir,
    fingerprint,
    knowledge_paths,
    auxiliary,
    errors,
):
    """Run post-verifier G3 capture without replacing the primary verdict."""
    capture_started = time.monotonic()
    auxiliary["business_terminal_status"] = status.value
    group_config = config.groups[group]
    public = {
        "messages": [
            {"role": "user", "text": (task_root / "instruction.md").read_text()},
            *trajectory.public_messages,
        ],
        "destination": str(profile.corpus),
        "local_remote": profile.private_env["MINDIE_BENCH_LOCAL_KNOWLEDGE_REMOTE"],
        "instance_id": task.task.instance_id,
        "role": task.task.role,
        "cluster": task.task.cluster,
        "business_terminal_status": status.value,
        "task_manifest": dict(
            instance_id=task.task.instance_id,
            business_terminal_status=status.value,
            verifier=primary.get("performance_metrics"),
            benchmark_source_sha256=fingerprint["benchmark_source_sha256"],
        ),
    }
    if config.harness.name != "fixture":
        public["native"] = native_capture_payload(
            config.harness.name,
            trajectory,
            profile,
            workspace,
            native_descriptor=(
                driver.native_capture_payload() if config.harness.name == "dsh" else None
            ),
        )
    corpus_before_capture = tree_hash(profile.corpus)
    capture_harness = config.harness.model_copy(
        update={"model": group_config.capture_model or config.harness.model}
    )
    capture_task = task.model_copy(deep=True)
    capture_task.budget.max_turns = group_config.capture_max_model_calls
    with Gateway(
        capture_harness,
        capture_task,
        task_root,
        workspace,
        trial_dir / "capture_broker.jsonl",
        use_unix_socket=platform.system() == "Linux",
    ) as capture_gateway:
        profile.env.update(
            MINDIE_BENCH_GATEWAY=capture_gateway.url, ANTHROPIC_BASE_URL=capture_gateway.url
        )
        driver_for(config.harness.name).command(profile, capture_harness)
        summary_context = optional_host(
            profile,
            read_paths=knowledge_paths,
            gateway_port=capture_gateway.port,
            gateway_socket=capture_gateway.unix_socket,
        )
        with summary_context as summary:
            if summary is not None:
                profile.env["MINDIE_DSH_STATE_DIR"] = summary.env["MINDIE_DSH_STATE_DIR"]
            capture = native_command(
                group_config.capture_command,
                read=[*knowledge_paths, profile.private_root],
                write=[profile.root, profile.private_root],
                cwd=profile.root,
                policy_path=trial_dir / "capture.sb",
                gateway_port=capture_gateway.port,
                gateway_socket=capture_gateway.unix_socket,
                loopback_client_ports=(summary.port,) if summary is not None else (),
            )
            records = []
            captured = stream_process(
                capture,
                env=dict(profile.env, **profile.private_env),
                cwd=profile.root,
                prompt=json.dumps(public),
                output=trial_dir / "capture_events.jsonl",
                stderr=trial_dir / "capture.stderr",
                timeout=group_config.capture_timeout_sec,
                max_bytes=task.budget.max_output_bytes,
                consume=lambda event, *_: records.append(event),
                gateway=capture_gateway,
            )
        if summary is not None:
            auxiliary["capture_sdk"] = summary.receipt
        auxiliary["capture_process"] = dict(
            returncode=captured.returncode,
            stop_reason=captured.stop_reason,
            errors=captured.errors,
            elapsed_sec=captured.elapsed,
            started=captured.started,
        )
        capture_gateway.finish_inference()
        errors.extend(captured.errors)
        if len(records) == 1 and isinstance(records[0], dict):
            auxiliary["capture"] = records[0]
        auxiliary["capture_model_calls"] = capture_gateway.model_calls
        auxiliary["capture_model"] = capture_harness.model
        auxiliary["capture_provider_tokens"] = capture_gateway.metered_tokens().model_dump()
        auxiliary["capture_model_delivery_failures"] = capture_gateway.model_delivery_failures
        if (
            capture_gateway.unknown
            or any(record.get("status") == "unknown" for record in records)
            or (capture_gateway.model_calls and not capture_gateway.metered_tokens().complete)
        ):
            raise UnknownExternalEffects("capture inference result is unknown; no retry")
        if capture_gateway.model_calls and (captured.stop_reason or captured.errors):
            raise UnknownExternalEffects(
                "capture interrupted after model admission; reconcile its original receipts"
            )
        if (
            captured.returncode != 0
            or captured.errors
            or captured.stop_reason
            or capture_gateway.infrastructure_error
            or len(records) != 1
        ):
            lint = auxiliary.get("capture", {}).get("corpus_lint", {})
            if lint.get("status") == "rejected":
                raise CorpusLintRejected(lint)
            raise ValueError("required capture failed; inspect its original private receipts")
        capture_receipt = records[0]
    if capture_receipt.get("status") != "completed" or not isinstance(
        capture_receipt.get("tokens"), dict
    ):
        raise ValueError(
            "offline capture adapter did not provide a completion and token receipt"
        )
    auxiliary["capture"] = capture_receipt
    require_corpus_lint(capture_receipt)
    if auxiliary["capture_model_calls"] and capture_receipt["tokens"].get("total") == 0:
        raise ValueError("capture inference occurred but its usage was reported as zero")
    provider_tokens = auxiliary["capture_provider_tokens"]
    if (
        auxiliary["capture_model_calls"]
        and provider_tokens["complete"]
        and capture_receipt["tokens"]["total"]
        != sum(provider_tokens[name] for name in ["input_uncached", "input_cached", "output"])
    ):
        raise ValueError("capture usage differs from the observed model-provider ledger")
    auxiliary["captured_corpus_sha256"] = tree_hash(profile.corpus)
    if (
        auxiliary["captured_corpus_sha256"] == corpus_before_capture
        and not (
            task.task.role == "eval"
            and isinstance(capture_receipt.get("distill"), dict)
            and capture_receipt["distill"].get("status") in {"noop", "review_or_unchanged"}
        )
    ):
        raise ValueError("capture adapter reported completion without changing the local corpus")
    auxiliary["captured_corpus_path"] = str(profile.corpus)
    return capture_started
