"""Trial pipeline and finite experiment matrix orchestrator.

[Role]
Drives the single-trial lifecycle (`run_trial`) and multi-cell ablation matrix (`run_matrix`)
across `G0`..`G3` and `l2`/`l3` tiers:
`Workspace Materialization -> MCP Readiness Probe -> Agent Execution -> Settlement Grace 30s -> Verifier -> Scorer Metrics`.

[Invariants]
- Every started attempt writes `attempt_started.json` exclusively before execution; a missing
  `trial_receipt.json` is treated as `UNKNOWN` and never automatically retried.
- Terminal status strictly separates `completed_pass`, `completed_fail`, `infra_failed` (failed
  before agent execution or during postflight after preserving `business_terminal_status`), and
  `unknown` (uncertain external model/remote/capture effects).
- In `G3` lineages, `seed` tasks must complete with a passing `corpus_lint` and modified corpus
  `tree_hash` before dependent `eval` tasks are admitted.

[Timeouts & Locks]
- Agent execution budget: `task.budget.timeout_sec`.
- Gateway model & remote settlement grace: `30.0s` before `remote.quiesce()` and verifier launch.
- Verifier inner budget: `task.budget.verifier_timeout_sec` (remote worker outer budget `+10.0s`).
- `device_condition` coordinates non-overlapping NPU device slice allocation across concurrent `l3` cells.

[Side-Effect Policy]
- Writes immutable trial artifacts (`attempt_started.json`, `native_events.jsonl`, `broker.jsonl`,
  `trajectory.json`, `trial_receipt.json`) inside `0o700` `trial_dir`, and matrix-level
  `eval_manifest.json`, `summary.json`, and `matrix_accounting.json` inside `output`.
"""

from __future__ import annotations

import platform
import shutil
import subprocess as subprocess
import time
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor
from concurrent.futures import as_completed as as_completed
from pathlib import Path

from mindie_bench import __version__ as __version__  # noqa: F401
from mindie_bench.evaluation.budget import BUDGET_STOPS
from mindie_bench.evaluation.budget import has_non_budget_errors as has_non_budget_errors  # noqa: F401
from mindie_bench.evaluation.corpus_linter import CorpusLintRejected as CorpusLintRejected  # noqa: F401
from mindie_bench.evaluation.oracle_freezer import freeze_oracle as freeze_oracle  # noqa: F401
from mindie_bench.io import digest as digest  # noqa: F401
from mindie_bench.io import read_json as read_json  # noqa: F401
from mindie_bench.io import task_hash as task_hash  # noqa: F401
from mindie_bench.io import tree_hash as tree_hash  # noqa: F401
from mindie_bench.io import write_json
from mindie_bench.models import (
    ExperimentConfig,
    Group,
    TerminalStatus,
    TrialReceipt,
)
from mindie_bench.models import TaskSpec as TaskSpec  # noqa: F401
from mindie_bench.models import Trajectory as Trajectory  # noqa: F401
from mindie_bench.models import UnknownExternalEffects as UnknownExternalEffects  # noqa: F401
from mindie_bench.runner.drivers import driver_for as driver_for  # noqa: F401
from mindie_bench.runner.dsh_summary import SummaryHost
from mindie_bench.runner.dsh_summary import optional_host as optional_host  # noqa: F401
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.matrix_runner import run_matrix as run_matrix
from mindie_bench.runner.mcp_probe import probe_servers
from mindie_bench.runner.native_product import optional_service
from mindie_bench.runner.phase_budget import lifecycle_overrun as lifecycle_overrun  # noqa: F401
from mindie_bench.runner.process import stream_process
from mindie_bench.runner.profile_builder import bind_mcp_transport, build_profile
from mindie_bench.runner.profile_builder import filtered_corpus as filtered_corpus  # noqa: F401
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.soundness import check_soundness as check_soundness  # noqa: F401
from mindie_bench.runner.source_projection import (
    HARNESS_CONFIG_PATHS as HARNESS_CONFIG_PATHS,  # noqa: F401
)
from mindie_bench.runner.trial_finalize import (
    finalize_trial_receipt,
    run_g3_capture,
    score_trial_trajectory,
)
from mindie_bench.runner.trial_preflight import (  # noqa: F401
    RUNTIME_HELPER_MODULES as RUNTIME_HELPER_MODULES,
)
from mindie_bench.runner.trial_preflight import (
    ProductDeadlineExhausted as ProductDeadlineExhausted,
)
from mindie_bench.runner.trial_preflight import (
    _install_shims,
    build_agent_prompt,
    check_read_boundaries,
    failure_status,
    preflight_trial_admission,
    prepare_trial,
    probe_capture_git,
    probe_harness_version,
    probe_runtime_import,
    product_execution_window,
    runtime_paths,
)
from mindie_bench.runner.trial_preflight import (
    implementation_digest as implementation_digest,
)
from mindie_bench.runner.trial_preflight import (
    native_capture_payload as native_capture_payload,
)
from mindie_bench.runner.trial_preflight import (
    require_corpus_lint as require_corpus_lint,
)
from mindie_bench.runner.verifier import (
    local_verifier,
    remote_verifier,
)
from mindie_bench.runner.verifier import (
    validate_remote_oracle_contract as validate_remote_oracle_contract,  # noqa: F401
)
from mindie_bench.runner.worktree_slot import WorktreeSlot
from mindie_bench.scorer.comparator import grouped_summaries as grouped_summaries  # noqa: F401
from mindie_bench.scorer.metrics import trajectory_metrics as trajectory_metrics  # noqa: F401


def run_trial(
    task_root: Path,
    config: ExperimentConfig,
    *,
    tier: str,
    group: Group,
    repetition: int,
    run_root: Path,
    devices: list[int] | None = None,
    corpus_override: Path | None = None,
    cancelled=None,
    experiment_context: dict | None = None,
    expected_task_digest: str | None = None,
    expected_oracle_sha256: str | None = None,
) -> TrialReceipt:
    # Keep moved phase helpers patch-compatible with callers that replace
    # orchestrator.native_command/time/stream_process at runtime.
    from mindie_bench.runner import trial_finalize, trial_preflight

    trial_preflight.native_command = native_command
    trial_preflight.time = time
    trial_finalize.native_command = native_command
    trial_finalize.stream_process = stream_process
    trial_finalize.time = time

    state = prepare_trial(
        task_root,
        config,
        tier=tier,
        group=group,
        repetition=repetition,
        run_root=run_root,
        experiment_context=experiment_context,
        expected_task_digest=expected_task_digest,
    )
    (
        task_root, run_root, trial_started, task, task_digest, trial_id, trial_dir,
        fingerprint, environment, driver, trajectory, status, reason, errors, auxiliary,
        primary, profile, slot, workspace, remote, result, gateway,
        native_service_instance, agent_phase_budget, phase, wall_clock,
        verifier_record, verifier_seconds, capture_started, primary_verdict,
    ) = state.values()
    try:
        oracle_manifest = preflight_trial_admission(
            task_root=task_root,
            task=task,
            config=config,
            tier=tier,
            group=group,
            environment=environment,
            auxiliary=auxiliary,
            expected_oracle_sha256=expected_oracle_sha256,
        )
        profile = build_profile(
            trial_dir,
            task,
            group,
            config,
            corpus_override=corpus_override,
            product_experiment=experiment_context is not None,
        )
        fingerprint.update(profile.fingerprint)
        paths = runtime_paths(config, profile)
        repo_path = Path(environment.repo_path)
        if not repo_path.is_absolute() and (task_root / repo_path).exists():
            repo_path = (task_root / repo_path).resolve(strict=True)
        protected = [
            *(task_root / relative for relative in oracle_manifest.protected_paths),
            repo_path,
            Path.home() / ".codex",
            Path.home() / ".claude",
            Path.home() / ".kimi-code",
            Path.home() / ".dsh",
            *[Path(path).resolve(strict=True) for path in config.knowledge_runtime_read_paths],
        ]
        if profile.private_root:
            protected.append(profile.private_root)
        if task.mindie_knowledge.corpus_id in config.corpora:
            protected.append(Path(config.corpora[task.mindie_knowledge.corpus_id].path))
        check_read_boundaries(paths, protected)
        fingerprint["runtime_preflight"] = probe_runtime_import(
            profile, paths, trial_dir, fingerprint["benchmark_source_sha256"]
        )
        harness_probe = probe_harness_version(config, profile, paths, trial_dir)
        fingerprint["harness_binary_sha256"] = harness_probe["binary_sha256"]
        capture_enabled = group == Group.G3 or (
            group == Group.G4
            and bool(config.groups.get(group) and config.groups[group].capture_command)
        )
        if capture_enabled and config.harness.name == "dsh":
            from mindie_bench.runner import dsh_summary, graded_view

            capture_config = config.groups[group]
            if (
                capture_config.capture_protocol != "graded-c6"
                or not capture_config.dsh_summary_plugin
                or not graded_view.is_graded(profile.corpus)
            ):
                raise ValueError(
                    "DSH G3 requires a configured native summary host and graded C6 corpus"
                )
            if capture_config.capture_timeout_sec < 1200:
                raise ValueError(
                    "graded capture requires 1200s for scanner/model/Git/feed budgets and cleanup"
                )
            capture_harness = config.harness.model_copy(
                update={"model": capture_config.capture_model or config.harness.model}
            )
            fingerprint["dsh_summary_profile"] = dsh_summary.prepare(
                profile, capture_harness, capture_config.dsh_summary_plugin
            )
            profile.env["MINDIE_BENCH_GATEWAY"] = "http://127.0.0.1:9"
            with SummaryHost(profile, read_paths=paths) as summary:
                fingerprint["capture_sdk_preflight"] = summary.receipt
        if capture_enabled:
            fingerprint["capture_git_preflight"] = probe_capture_git(profile, paths, trial_dir)
        if tier == "l3":
            if config.remote is None or not devices or len(devices) != environment.npu_count:
                raise ValueError(
                    "L3 requires a remote worker and exactly the declared number of NPU devices"
                )
            workspace = trial_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            remote = RemoteSlot(
                config.remote,
                environment,
                trial_id,
                devices,
                task.budget.timeout_sec,
                artifact_dir=trial_dir,
            )
            remote.prepare()
            remote.upload_public(task_root / "environment")
            fingerprint["remote_preflight"] = remote.preflight
            fingerprint["remote_environment"] = remote.preflight["_environment"]
        else:
            slot = WorktreeSlot(environment, trial_dir / "workspace", task_root)
            workspace = slot.prepare()
        _install_shims(profile, environment.replay_commands if tier == "l2" else [], remote=bool(remote))
        gateway = Gateway(
            config.harness,
            task,
            task_root,
            workspace,
            trial_dir / "broker.jsonl",
            use_unix_socket=platform.system() == "Linux",
            remote=remote,
        )
        knowledge_paths = [
            *paths,
            *[Path(path).resolve(strict=True) for path in config.knowledge_runtime_read_paths],
        ]
        with gateway:
            gateway.attach_knowledge(profile, knowledge_paths)
            profile.env.update(
                MINDIE_BENCH_GATEWAY=gateway.url,
                MINDIE_BENCH_DUMMY_KEY="broker-only",
                ANTHROPIC_API_KEY="broker-only",
                ANTHROPIC_BASE_URL=gateway.url,
            )
            if config.harness.name == "dsh":
                profile.env.update(
                    DSH_HOME=profile.env["DSH_HOME"],
                    DSH_PROFILE="bench",
                    DSH_PROFILE_DIR=profile.env["DSH_PROFILE_DIR"],
                    MINDIE_BENCH_DUMMY_KEY="broker-only",
                    # The benchmark runner already confines the whole native
                    # process. A nested DSH Seatbelt attempt would fail.
                    DSH_PERMISSION_MODE="danger-full-access",
                )
            bind_mcp_transport(profile)
            fingerprint["mcp_config"] = profile.fingerprint["mcp_config"]
            if config.backend == "native":
                for component in profile.fingerprint["components"]:
                    if not config.component_revisions.get(component):
                        raise ValueError(
                            f"native ablation is missing exact component provenance: {component}"
                        )
                fingerprint["mcp_preflight"] = probe_servers(
                    profile, config.groups.get(group), paths, trial_dir, gateway
                )
                fingerprint["mcp_preflight_finished_monotonic_sec"] = time.monotonic()
            native_service = optional_service(profile, paths, trial_dir, gateway)
            native_service_instance = native_service.__enter__()
            native_grants = dict(
                read=paths,
                write=[workspace, profile.root],
                cwd=workspace,
                gateway_port=gateway.port,
                gateway_socket=gateway.unix_socket,
                directory_watchers=config.harness.name == "kimi",
                loopback_client_ports=(
                    getattr(native_service_instance, "ports", None) or (native_service_instance.port,)
                    if native_service_instance
                    else ()
                ),
                ephemeral_loopback_listener=native_service_instance is not None,
            )

            def sandboxed(arguments, policy_name):
                return native_command(
                    arguments, policy_path=trial_dir / policy_name, **native_grants
                )

            phase = "preparation"
            if config.harness.name == "dsh" and (
                experiment_context is not None or profile.fingerprint.get("native_product")
            ):
                from mindie_bench.runner.profile_builder import (
                    assert_catalog_policy,
                    probe_dsh_catalog,
                )

                probe_command = sandboxed(
                    [str(Path(config.harness.executable).absolute()), "--profile", "bench", "--json"],
                    "catalog-probe.sb",
                )
                mounted_catalog = probe_dsh_catalog(
                    profile, trial_dir, workspace, probe_command
                )
                product_group = ((experiment_context or {}).get("cell") or {}).get("group")
                assert_catalog_policy(mounted_catalog["identity"], product_group)
                if profile.fingerprint.get("native_product"):
                    required = {"mindie_entry", "knowledge_query", "knowledge_explain"}
                    if not required <= set(mounted_catalog["identity"]["tool_names"]):
                        raise ValueError("native DSH product did not register its knowledge tools")
                fingerprint["preflight_catalog"] = mounted_catalog
                auxiliary["preflight_catalog"] = mounted_catalog
            command = sandboxed(driver.command(profile, config.harness), "agent.sb")
            prompt = build_agent_prompt(
                profile, task_root, remote=remote, devices=devices
            )
            auxiliary["preflight_wall_clock_sec"] = time.monotonic() - trial_started
            agent_timeout = task.budget.timeout_sec
            if experiment_context is not None:
                agent_timeout, agent_phase_budget = product_execution_window(
                    experiment_context,
                    task,
                    time.monotonic() - trial_started,
                )
                auxiliary["product_agent_timeout_sec"] = agent_timeout
                auxiliary["product_phase_budget"] = agent_phase_budget
            phase = "agent"
            result = stream_process(
                command,
                env=profile.env,
                cwd=workspace,
                prompt=prompt,
                output=trial_dir / "native_events.jsonl",
                stderr=trial_dir / "agent.stderr",
                timeout=agent_timeout,
                max_bytes=task.budget.max_output_bytes,
                consume=driver.consume,
                gateway=gateway,
                cancelled=cancelled,
            )
            wall_clock = result.elapsed
            gateway.finish_inference()
            gateway.finish_remote()
            errors.extend(result.errors)
            auxiliary["native_process"] = dict(
                returncode=result.returncode,
                stop_reason=result.stop_reason,
                started_monotonic_sec=result.started_monotonic_sec,
            )
            trajectory = driver.trajectory()
            trajectory.total_turns = max(trajectory.total_turns, gateway.model_calls)
            # This controller sends exactly one initial prompt and never asks
            # for corrections; interventions are known, not inferred from prose.
            trajectory.human_interventions = 0
            budget_stop = result.stop_reason in BUDGET_STOPS
            auxiliary["agent_budget_exhausted"] = budget_stop
            if gateway.infrastructure_error and not gateway.unknown:
                status, reason = TerminalStatus.INFRA, gateway.infrastructure_error
            elif (
                not gateway.unknown
                and result.stop_reason == "timeout"
                and gateway.model_calls == 0
                and trajectory.total_turns == 0
                and not trajectory.calls
                and not trajectory.native_completed
            ):
                # A native startup stall never attempted the task. Running the
                # oracle on its untouched source would misreport model failure.
                status, reason = (
                    TerminalStatus.INFRA,
                    "native startup timed out before any model request or tool call",
                )
                auxiliary["native_process"]["classification"] = "native_startup_infrastructure"
                auxiliary["native_process"]["observed_model_requests"] = 0
            elif (
                gateway.unknown
                or any(error["stage"] == "agent" for error in result.errors)
                or (
                    not budget_stop
                    and (
                        result.returncode not in {0, None}
                        or trajectory.native_error
                        or not trajectory.native_completed
                    )
                )
            ):
                status = TerminalStatus.UNKNOWN if result.started else TerminalStatus.INFRA
                reason = (
                    trajectory.native_error
                    or result.stop_reason
                    or f"native process ended without a reliable completion (exit={result.returncode}); "
                    "see private agent.stderr and native_events.jsonl"
                )
            elif result.stop_reason == "output_limit":
                status, reason = TerminalStatus.UNKNOWN, "native output limit; evidence incomplete"
            else:
                phase = "verifier"
                verifier_started = time.monotonic()
                write_json(
                    trial_dir / "verifier_trajectory.json",
                    trajectory.model_dump(mode="json"),
                    exclusive=True,
                )
                if remote:
                    remote.quiesce()
                    verdict, verifier_record = remote_verifier(
                        task_root,
                        remote,
                        trial_dir,
                        task.budget.verifier_timeout_sec,
                        oracle_manifest=oracle_manifest,
                        oracle_contract=task.oracle,
                        performance_contract=task.performance,
                        inference_contract=task.inference,
                    )
                else:
                    verdict, verifier_record = local_verifier(
                        task_root,
                        workspace,
                        profile,
                        trial_dir,
                        task.budget.verifier_timeout_sec,
                        paths,
                        oracle_manifest=oracle_manifest,
                        oracle_contract=task.oracle,
                        performance_contract=task.performance,
                        inference_contract=task.inference,
                    )
                if task.inference is not None:
                    fingerprint["inference_native_stop_reason"] = result.stop_reason
                # The stopped agent's existing artifact is still independently judged.
                # Reaching its limit is recorded separately from that business verdict.
                passed = verdict.reward == 1.0
                status = TerminalStatus.PASS if passed else TerminalStatus.FAIL
                reason = (
                    None
                    if passed
                    else verdict.failure_reason
                    or "external verifier rejected final state"
                )
                primary.update(
                    resolved=passed,
                    reward=1.0 if passed else 0.0,
                    performance_metrics=verdict.model_dump(exclude={"reward"}),
                )
                auxiliary["verifier"] = verifier_record
                verifier_seconds = time.monotonic() - verifier_started
                primary_verdict = status
        phase = "postflight"
        if (
            capture_enabled
            and status in {TerminalStatus.PASS, TerminalStatus.FAIL}
            and primary.get("performance_metrics", {}).get("oracle_status") != "oracle_tampered"
        ):
            phase = "capture"
            capture_started = run_g3_capture(
                task_root=task_root,
                task=task,
                config=config,
                group=group,
                status=status,
                primary=primary,
                trajectory=trajectory,
                profile=profile,
                workspace=workspace,
                driver=driver,
                trial_dir=trial_dir,
                fingerprint=fingerprint,
                knowledge_paths=knowledge_paths,
                auxiliary=auxiliary,
                errors=errors,
            )
    except KeyboardInterrupt:
        status = TerminalStatus.UNKNOWN if phase in {"agent", "capture"} else TerminalStatus.INFRA
        reason = "interrupted; reconcile existing effects before another attempt"
        errors.append(dict(stage=phase, type="KeyboardInterrupt", message=reason))
        raise
    except Exception as exc:
        errors.append(dict(stage=phase, type=type(exc).__name__, message=str(exc)))
        if primary_verdict is not None:
            # The verifier already produced an immutable business verdict.
            # Capture/cleanup failure is secondary evidence, never a new
            # terminal status and never a reason to erase the reward.
            auxiliary.setdefault("business_terminal_status", primary_verdict.value)
            if phase == "capture":
                capture = dict(auxiliary.get("capture") or {})
                capture.update(
                    status="failed",
                    error=dict(type=type(exc).__name__, message=str(exc)),
                )
                auxiliary["capture"] = capture
        else:
            status = failure_status(exc, phase)
            if status == TerminalStatus.FAIL:
                primary.update(resolved=False, reward=0.0)
            elif phase == "postflight" and status != TerminalStatus.UNKNOWN:
                auxiliary["business_terminal_status"] = status.value
            reason = (
                reason
                if status == TerminalStatus.PASS
                else f"{phase}: {type(exc).__name__}: {exc}"
            )
    finally:
        if native_service_instance is not None:
            try:
                native_service_instance.close()
                auxiliary["native_knowledge_service_cleanup"] = "completed"
            except Exception as exc:
                errors.append(
                    dict(stage="native_service_cleanup", type=type(exc).__name__, message=str(exc))
                )
                auxiliary["native_knowledge_service_cleanup"] = "failed"
                if primary_verdict is not None:
                    auxiliary.setdefault("business_terminal_status", primary_verdict.value)
                elif status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
                    auxiliary["business_terminal_status"] = status.value
                    status, reason = TerminalStatus.INFRA, "native knowledge service cleanup failed"
        auxiliary["capture_wall_clock_sec"] = (
            time.monotonic() - capture_started if capture_started is not None else None
        )
        scored = score_trial_trajectory(
            driver=driver,
            gateway=gateway,
            result=result,
            trajectory=trajectory,
            config=config,
            environment=environment,
            task=task,
            trial_dir=trial_dir,
            task_root=task_root,
            wall_clock=wall_clock,
            tier=tier,
            fingerprint=fingerprint,
            errors=errors,
            primary=primary,
            auxiliary=auxiliary,
            primary_verdict=primary_verdict,
            status=status,
            reason=reason,
        )
        trajectory = scored["trajectory"]
        primary = scored["primary"]
        auxiliary = scored["auxiliary"]
        status = scored["status"]
        reason = scored["reason"]
        if gateway:
            primary["model_calls"] = gateway.model_calls + auxiliary.get("capture_model_calls", 0)
            auxiliary["model_call_measurement_basis"] = (
                "Gateway initiated requests, including unknown attempts"
            )
            auxiliary["replay_test_runs"] = gateway.replay_calls
            auxiliary["remote_test_budget_invocations"] = gateway.remote_test_calls
            auxiliary["remote_delivery_failures"] = gateway.remote_delivery_failures
            auxiliary["model_delivery_failures"] = gateway.model_delivery_failures
        if remote:
            try:
                remote.close()
                auxiliary["post_flight_npu_clean"] = True
            except Exception as exc:
                errors.append(dict(stage="remote_cleanup", type=type(exc).__name__, message=str(exc)))
                auxiliary["post_flight_npu_clean"] = False
                if primary_verdict is not None:
                    auxiliary.setdefault("business_terminal_status", primary_verdict.value)
                else:
                    auxiliary["business_terminal_status"] = status.value
                    if status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
                        status, reason = (
                            TerminalStatus.INFRA,
                            "remote postflight failed; preserve separate business verdict",
                        )
            auxiliary["owned_namespace_cleanup_confirmed"] = getattr(remote, "cleanup_confirmed", None)
            auxiliary["files_modified"] = getattr(remote, "modified_files", None)
            auxiliary["files_modified_scope"] = "remote_judged_workspace"
            auxiliary["files_modified_reason"] = (
                None
                if auxiliary["files_modified"] is not None
                else "remote_snapshot_inventory_unavailable"
            )
            if getattr(remote, "component_runtime", None) is not None:
                try:
                    from mindie_bench.scorer.components import summarize_component_runtime

                    auxiliary["component_runtime_metrics"] = summarize_component_runtime(
                        trial_dir / "component_runtime.jsonl"
                    )
                except Exception as exc:
                    errors.append(
                        dict(stage="component_metrics", type=type(exc).__name__, message=str(exc))
                    )
                    if primary_verdict is not None:
                        auxiliary.setdefault("business_terminal_status", primary_verdict.value)
                    elif status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
                        auxiliary["business_terminal_status"] = status.value
                        status, reason = TerminalStatus.INFRA, "component metric extraction failed"
        if slot:
            try:
                if slot.created:
                    auxiliary["local_files_modified"] = slot.snapshot_patch(trial_dir)
                    if not remote:
                        auxiliary["files_modified"] = auxiliary["local_files_modified"]
                        auxiliary["files_modified_scope"] = "local_judged_workspace"
                        auxiliary["files_modified_reason"] = None
                    # Preserve the final files as a private artifact before
                    # deleting only this runner-owned worktree.
                    shutil.make_archive(str(trial_dir / "final_workspace"), "gztar", slot.destination)
                slot.cleanup()
            except Exception as exc:
                errors.append(dict(stage="workspace_cleanup", type=type(exc).__name__, message=str(exc)))
        elif workspace is not None and workspace.exists():
            try:
                auxiliary["local_files_modified"] = sorted(
                    p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file()
                )
                shutil.make_archive(str(trial_dir / "final_workspace"), "gztar", workspace)
            except Exception as exc:
                errors.append(dict(stage="workspace_cleanup", type=type(exc).__name__, message=str(exc)))
        receipt = finalize_trial_receipt(
            gateway=gateway,
            status=status,
            reason=reason,
            primary_verdict=primary_verdict,
            primary=primary,
            auxiliary=auxiliary,
            trial_started=trial_started,
            experiment_context=experiment_context,
            agent_phase_budget=agent_phase_budget,
            errors=errors,
            verifier_seconds=verifier_seconds,
            trial_dir=trial_dir,
            trajectory=trajectory,
            trial_id=trial_id,
            task=task,
            repetition=repetition,
            tier=tier,
            group=group,
            task_digest=task_digest,
            config=config,
            fingerprint=fingerprint,
        )
    return receipt


