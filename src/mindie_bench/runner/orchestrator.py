"""An explicit finite experiment matrix; uncertain trials are never resubmitted."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mindie_bench import __version__
from mindie_bench.io import digest, read_json, task_hash, tree_hash, write_json
from mindie_bench.models import (
    ExperimentConfig,
    Group,
    TaskSpec,
    TerminalStatus,
    Trajectory,
    TrialReceipt,
    UnknownExternalEffects,
)
from mindie_bench.runner.drivers import driver_for
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.mcp_probe import probe_servers
from mindie_bench.runner.process import stream_process
from mindie_bench.runner.profile_builder import bind_mcp_transport, build_profile, filtered_corpus
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS
from mindie_bench.runner.verifier import local_verifier, parse_verifier
from mindie_bench.runner.worktree_slot import WorktreeSlot
from mindie_bench.scorer.comparator import grouped_summaries
from mindie_bench.scorer.metrics import trajectory_metrics


def runtime_paths(config: ExperimentConfig, profile) -> list[Path]:
    return [
        Path(__file__).resolve().parents[2],
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(config.harness.executable).resolve(strict=True),
        *profile.runtime_paths,
        *[Path(p).resolve(strict=True) for p in config.runtime_read_paths],
    ]


def implementation_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
    )


def check_read_boundaries(paths: list[Path], protected: list[Path]):
    for allowed in paths:
        for private in protected:
            if private.resolve().is_relative_to(allowed.resolve()) or allowed.resolve().is_relative_to(
                private.resolve()
            ):
                raise ValueError(
                    f"runtime read grant exposes protected benchmark/personal input: {allowed}"
                )


def _install_shims(profile, commands: list[str], *, remote: bool):
    directory = profile.root / "bin"
    directory.mkdir()
    for command in commands + (["ssh"] if remote else []):
        module = (
            "mindie_bench.runner.ssh_shim" if command == "ssh" else "mindie_bench.shim.npu_replay_shim"
        )
        script = directory / command
        # Absolute interpreter and package path are explicit read capabilities.
        script.write_text(
            f"#!{sys.executable} -I\nfrom {module} import main\nraise SystemExit(main())\n"
        )
        script.chmod(0o700)
    profile.env["PATH"] = str(directory) + os.pathsep + profile.env["PATH"]


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
) -> TrialReceipt:
    trial_started = time.monotonic()
    task = TaskSpec.load(task_root)
    trial_id = f"{task.task.instance_id}_{group.value}_{repetition}_{uuid.uuid4().hex[:12]}"
    trial_dir = run_root / trial_id
    trial_dir.mkdir(mode=0o700)
    task_digest = task_hash(task_root)
    fingerprint = dict(
        host_os=platform.platform(),
        host_arch=platform.machine(),
        harness_name=config.harness.name,
        harness_version=config.harness.expected_version,
        harness_binary_sha256=None,
        base_model=config.harness.model,
        reasoning_effort=config.harness.reasoning_effort,
        provider_url=config.harness.provider_url,
        backend=config.backend,
        budget=task.budget.model_dump(),
        benchmark_version=__version__,
        benchmark_source_sha256=implementation_digest(),
        component_revisions=config.component_revisions,
        evidence=task.task.evidence,
        source_projection=list(HARNESS_CONFIG_PATHS),
    )
    environment = task.environment.get(tier)
    fingerprint.update(
        base_commit=environment.base_commit if environment else None,
        npu_model=environment.npu_model if environment else None,
        npu_count=environment.npu_count if environment else None,
        remote_environment=None,
    )
    write_json(
        trial_dir / "attempt_started.json",
        dict(
            trial_id=trial_id,
            task_digest=task_digest,
            warning="A missing terminal receipt is UNKNOWN. Never automatically resume this attempt.",
        ),
        exclusive=True,
    )
    driver = driver_for(config.harness.name)
    trajectory = Trajectory()
    status, reason = TerminalStatus.INFRA, "trial did not start"
    errors, auxiliary, primary = [], {}, {}
    profile = slot = remote = result = gateway = None
    phase = "preflight"
    wall_clock = 0.0
    verifier_record = None
    verifier_seconds = None
    capture_started = None
    try:
        if environment is None:
            raise ValueError(f"task has no {tier} environment")
        if task.task.evidence == "candidate":
            raise ValueError("candidate tasks require soundness validation before evaluation")
        if (
            task.task.evidence == "verified"
            and read_json(task_root / "soundness_receipt.json")["tier"] != tier
        ):
            raise ValueError("soundness proof does not cover the requested execution tier")
        if group == Group.G3 and not config.groups.get(group, None):
            raise ValueError("G3 requires an explicit offline capture adapter")
        if group == Group.G3 and not config.groups[group].capture_command:
            raise ValueError("G3 requires an explicit offline capture adapter")
        if config.harness.name == "fixture" and (
            config.backend != "fixture" or task.task.evidence != "synthetic"
        ):
            raise ValueError("fixture harness is restricted to synthetic tasks")
        if config.harness.name != "fixture" and config.backend != "native":
            raise ValueError("native harnesses require the OS-enforced native backend")
        profile = build_profile(trial_dir, task, group, config, corpus_override=corpus_override)
        fingerprint.update(profile.fingerprint)
        paths = runtime_paths(config, profile)
        protected = [
            task_root / "oracle",
            task_root / "verifier",
            Path(environment.repo_path),
            Path.home() / ".codex",
            Path.home() / ".claude",
            Path.home() / ".kimi-code",
        ]
        if task.mindie_knowledge.corpus_id in config.corpora:
            protected.append(Path(config.corpora[task.mindie_knowledge.corpus_id].path))
        check_read_boundaries(paths, protected)
        binary = Path(config.harness.executable).resolve(strict=True)
        fingerprint["harness_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        version_cmd = native_command(
            [str(binary), "--version"],
            read=paths,
            write=[profile.root],
            cwd=profile.root,
            policy_path=trial_dir / "version.sb",
        )
        version_result = subprocess.run(
            version_cmd, env=profile.env, cwd=profile.root, timeout=20, capture_output=True, text=True
        )
        (trial_dir / "version.stdout").write_text(version_result.stdout)
        (trial_dir / "version.stderr").write_text(version_result.stderr)
        version_result.check_returncode()
        version = version_result.stdout.strip()
        if version != config.harness.expected_version:
            raise ValueError(
                f"harness version mismatch: expected {config.harness.expected_version!r}, got {version!r}"
            )
        slot = WorktreeSlot(environment, trial_dir / "workspace", task_root)
        workspace = slot.prepare()
        if tier == "l3":
            if config.remote is None or not devices or len(devices) != environment.npu_count:
                raise ValueError(
                    "L3 requires a remote worker and exactly the declared number of NPU devices"
                )
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
        _install_shims(profile, environment.replay_commands if tier == "l2" else [], remote=bool(remote))
        unix_socket = trial_dir / "gateway.sock" if platform.system() == "Linux" else None
        gateway = Gateway(
            config.harness,
            task,
            task_root,
            workspace,
            trial_dir / "broker.jsonl",
            unix_socket=unix_socket,
            remote=remote,
        )
        with gateway:
            profile.env.update(
                MINDIE_BENCH_GATEWAY=gateway.url,
                MINDIE_BENCH_DUMMY_KEY="broker-only",
                ANTHROPIC_API_KEY="broker-only",
                ANTHROPIC_BASE_URL=gateway.url,
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
            command = driver.command(profile, config.harness)
            command = native_command(
                command,
                read=paths,
                write=[workspace, profile.root],
                cwd=workspace,
                policy_path=trial_dir / "agent.sb",
                gateway_port=gateway.port,
                gateway_socket=unix_socket,
            )
            prompt = (task_root / "instruction.md").read_text()
            if profile.corpus:
                prompt += f"\n\nAvailable advisory corpus: {profile.corpus}\n"
            if remote:
                prompt += (
                    f"\n\nWork on the isolated SSH endpoint bench-slot. "
                    f"Remote workspace: {remote.path}. "
                    f"Assigned NPU devices: {','.join(map(str, devices))}. "
                    'For remote-dev tools select host="bench-slot", port=22. '
                    "The local and remote workspaces are plain source snapshots without Git history; "
                    "apply the repair in the remote workspace that the NPU verifier will inspect.\n"
                )
            phase = "agent"
            auxiliary["preflight_wall_clock_sec"] = time.monotonic() - trial_started
            result = stream_process(
                command,
                env=profile.env,
                cwd=workspace,
                prompt=prompt,
                output=trial_dir / "native_events.jsonl",
                stderr=trial_dir / "agent.stderr",
                timeout=task.budget.timeout_sec,
                max_bytes=task.budget.max_output_bytes,
                consume=driver.consume,
                gateway=gateway,
                cancelled=cancelled,
            )
            wall_clock = result.elapsed
            gateway.finish_inference()
            errors.extend(result.errors)
            auxiliary["native_process"] = dict(
                returncode=result.returncode, stop_reason=result.stop_reason
            )
            trajectory = driver.trajectory()
            trajectory.total_turns = max(trajectory.total_turns, gateway.model_calls)
            # This controller sends exactly one initial prompt and never asks
            # for corrections; interventions are known, not inferred from prose.
            trajectory.human_interventions = 0
            budget_stop = result.stop_reason in {"timeout", "max_turns", "max_npu_runs"}
            if gateway.infrastructure_error and not gateway.unknown:
                status, reason = TerminalStatus.INFRA, gateway.infrastructure_error
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
                    verified = remote.verify((task_root / "verifier/test.sh").read_bytes())
                    stdout = base64.b64decode(verified["stdout_b64"]).decode()
                    stderr = base64.b64decode(verified["stderr_b64"]).decode()
                    (trial_dir / "verifier.stdout").write_text(stdout)
                    (trial_dir / "verifier.stderr").write_text(stderr)
                    verdict = parse_verifier(verified["exit_code"], stdout)
                    verifier_record = dict(returncode=verified["exit_code"], result=verdict.model_dump())
                else:
                    verdict, verifier_record = local_verifier(
                        task_root, workspace, profile, trial_dir, task.budget.verifier_timeout_sec, paths
                    )
                passed = verdict.reward == 1.0 and not result.stop_reason
                status = TerminalStatus.PASS if passed else TerminalStatus.FAIL
                reason = (
                    None if passed else result.stop_reason or "external verifier rejected final state"
                )
                primary.update(
                    resolved=passed,
                    reward=1.0 if passed else 0.0,
                    performance_metrics=verdict.model_dump(exclude={"reward"}),
                )
                auxiliary["verifier"] = verifier_record
                verifier_seconds = time.monotonic() - verifier_started
        phase = "postflight"
        if group == Group.G3 and status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
            phase = "capture"
            capture_started = time.monotonic()
            auxiliary["business_terminal_status"] = status.value
            group_config = config.groups[group]
            if not group_config.capture_command:
                raise ValueError(
                    "G3 requires an explicit offline capture adapter; a hot snapshot is not a flywheel"
                )
            public = {
                "messages": [
                    {"role": "user", "text": (task_root / "instruction.md").read_text()},
                    *trajectory.public_messages,
                ],
                "destination": str(profile.corpus),
                "local_remote": profile.env["MINDIE_BENCH_LOCAL_KNOWLEDGE_REMOTE"],
                "instance_id": task.task.instance_id,
                "cluster": task.task.cluster,
            }
            if config.harness.name == "codex":
                identity = trajectory.native_session_id
                if not identity:
                    raise ValueError("G3 Codex capture requires the actual native thread identity")
                transcripts = list(
                    (Path(profile.env["CODEX_HOME"]) / "sessions").rglob(f"*-{identity}.jsonl")
                )
                if len(transcripts) != 1:
                    raise ValueError("G3 cannot identify exactly one native task transcript")
                public["native"] = {
                    "harness": "codex",
                    "session_id": identity,
                    "transcript_path": str(transcripts[0]),
                    "cwd": str(workspace),
                    "stop_event_path": str(profile.root / "native-stop.json"),
                }
            corpus_before_capture = tree_hash(profile.corpus)
            capture_harness = config.harness.model_copy(
                update={"model": group_config.capture_model or config.harness.model}
            )
            capture_task = task.model_copy(deep=True)
            capture_task.budget.max_turns = group_config.capture_max_model_calls
            capture_socket = trial_dir / "capture-gateway.sock" if platform.system() == "Linux" else None
            with Gateway(
                capture_harness,
                capture_task,
                task_root,
                workspace,
                trial_dir / "capture_broker.jsonl",
                unix_socket=capture_socket,
            ) as capture_gateway:
                profile.env.update(
                    MINDIE_BENCH_GATEWAY=capture_gateway.url, ANTHROPIC_BASE_URL=capture_gateway.url
                )
                driver_for(config.harness.name).command(profile, capture_harness)
                capture = native_command(
                    group_config.capture_command,
                    read=paths,
                    write=[profile.root],
                    cwd=profile.root,
                    policy_path=trial_dir / "capture.sb",
                    gateway_port=capture_gateway.port,
                    gateway_socket=capture_socket,
                )
                records = []
                captured = stream_process(
                    capture,
                    env=profile.env,
                    cwd=profile.root,
                    prompt=json.dumps(public),
                    output=trial_dir / "capture_events.jsonl",
                    stderr=trial_dir / "capture.stderr",
                    timeout=task.budget.verifier_timeout_sec,
                    max_bytes=task.budget.max_output_bytes,
                    consume=lambda event, *_: records.append(event),
                    gateway=capture_gateway,
                )
                capture_gateway.finish_inference()
                errors.extend(captured.errors)
                auxiliary["capture_model_calls"] = capture_gateway.model_calls
                auxiliary["capture_model"] = capture_harness.model
                auxiliary["capture_provider_tokens"] = capture_gateway.metered_tokens().model_dump()
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
                    raise ValueError("required capture failed; inspect its original private receipts")
                capture_receipt = records[0]
            if capture_receipt.get("status") != "completed" or not isinstance(
                capture_receipt.get("tokens"), dict
            ):
                raise ValueError(
                    "offline capture adapter did not provide a completion and token receipt"
                )
            auxiliary["capture"] = capture_receipt
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
            if auxiliary["captured_corpus_sha256"] == corpus_before_capture:
                raise ValueError("capture adapter reported completion without changing the local corpus")
    except KeyboardInterrupt:
        status = TerminalStatus.UNKNOWN if phase in {"agent", "capture"} else TerminalStatus.INFRA
        reason = "interrupted; reconcile existing effects before another attempt"
        errors.append(dict(stage=phase, type="KeyboardInterrupt", message=reason))
        raise
    except Exception as exc:
        errors.append(dict(stage=phase, type=type(exc).__name__, message=str(exc)))
        if phase == "agent" or isinstance(exc, UnknownExternalEffects):
            status = TerminalStatus.UNKNOWN
        else:
            if phase == "postflight":
                auxiliary["business_terminal_status"] = status.value
            status = TerminalStatus.INFRA
        reason = reason if status == TerminalStatus.PASS else f"{phase}: {type(exc).__name__}: {exc}"
    finally:
        auxiliary["capture_wall_clock_sec"] = (
            time.monotonic() - capture_started if capture_started is not None else None
        )
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
            measured_primary, measured_aux = trajectory_metrics(
                trajectory, task, config.harness, wall_clock, tier=tier
            )
            primary = dict(measured_primary, **primary)
            auxiliary = dict(measured_aux, **auxiliary)
            if any(error["stage"] == "capture" for error in errors):
                primary["cost_complete"] = False
                partial = auxiliary.get("capture_provider_tokens", {})
                primary["tokens"]["indexer_small_model_total"] = sum(
                    partial.get(field, 0) for field in ["input_uncached", "input_cached", "output"]
                )
                primary["tokens"]["complete"] = False
            capture = auxiliary.get("capture")
            if capture:
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
            if status != TerminalStatus.UNKNOWN:
                status, reason = TerminalStatus.INFRA, "trajectory scoring failed"
        if gateway:
            auxiliary["replay_test_runs"] = gateway.replay_calls
            auxiliary["remote_test_budget_invocations"] = gateway.remote_test_calls
            auxiliary["remote_delivery_failures"] = gateway.remote_delivery_failures
        if remote:
            try:
                remote.close()
                auxiliary["post_flight_npu_clean"] = True
            except Exception as exc:
                errors.append(dict(stage="remote_cleanup", type=type(exc).__name__, message=str(exc)))
                auxiliary["post_flight_npu_clean"] = False
                auxiliary["business_terminal_status"] = status.value
                if status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
                    status, reason = (
                        TerminalStatus.INFRA,
                        "remote postflight failed; preserve separate business verdict",
                    )
        if slot:
            try:
                if slot.created:
                    auxiliary["files_modified"] = slot.snapshot_patch(trial_dir)
                    # Preserve the final files as a private artifact before
                    # deleting only this runner-owned worktree.
                    shutil.make_archive(str(trial_dir / "final_workspace"), "gztar", slot.destination)
                slot.cleanup()
            except Exception as exc:
                errors.append(dict(stage="workspace_cleanup", type=type(exc).__name__, message=str(exc)))
        if status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}:
            primary.update(resolved=None, reward=None)
        primary["delivery_wall_clock_sec"] = time.monotonic() - trial_started
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
            artifacts={"private_trial_dir": str(trial_dir)},
            evidence=task.task.evidence,
        )
        write_json(trial_dir / "trial_receipt.json", receipt.model_dump(mode="json"), exclusive=True)
    return receipt


def run_matrix(
    task_roots: list[Path],
    config: ExperimentConfig,
    *,
    tier: str,
    groups: list[Group],
    repetitions: int,
    concurrency: int,
    output: Path,
) -> list[TrialReceipt]:
    if concurrency < 1 or repetitions < 1 or not groups or len(set(groups)) != len(groups):
        raise ValueError("positive concurrency/repetitions and unique groups are required")
    specs = {path: TaskSpec.load(path) for path in task_roots}
    lineages = {}
    if Group.G3 in groups:
        if not any(spec.task.role == "seed" for spec in specs.values()):
            raise ValueError(
                "G3 requires seed tasks followed by eval tasks, not an unproven warm snapshot"
            )
        for path, spec in specs.items():
            cluster, corpus_id = spec.task.cluster, spec.mindie_knowledge.corpus_id
            if not cluster or corpus_id not in config.corpora:
                raise ValueError("G3 requires explicit transfer clusters and corpus IDs")
            lineages.setdefault((cluster, corpus_id), []).append(path)
        if not lineages or any(
            not any(specs[path].task.role == "seed" for path in paths) for paths in lineages.values()
        ):
            raise ValueError("G3 requires seed tasks in each lineage, not an unproven warm snapshot")
        for paths in lineages.values():
            paths.sort(key=lambda path: specs[path].task.role != "seed")
    if tier == "l3":
        if not config.remote:
            raise ValueError("L3 requires explicit remote configuration")
        for spec in specs.values():
            count = spec.environment[tier].npu_count
            if count < 1 or count > len(config.remote.devices):
                raise ValueError("each L3 task must fit the explicitly configured NPU device pool")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    write_json(
        output / "eval_manifest.json",
        dict(
            schema_version="1.0",
            created_unix=time.time(),
            experiment=config.model_dump(mode="json"),
            tasks=[dict(path=str(p), sha256=tree_hash(p)) for p in task_roots],
            tier=tier,
            groups=[g.value for g in groups],
            repetitions=repetitions,
            automatic_retries=0,
        ),
        exclusive=True,
    )
    receipts = []
    receipt_lock = threading.Lock()
    device_condition = threading.Condition()
    cancelled = threading.Event()
    available_devices = set(config.remote.devices) if tier == "l3" else set()
    excluded = {
        identity for spec in specs.values() for identity in spec.mindie_knowledge.same_origin_task_hashes
    }

    def execute_cell(path, group, repetition, *, corpus_override=None):
        devices = []
        with device_condition:
            if tier == "l3":
                count = specs[path].environment[tier].npu_count
                device_condition.wait_for(lambda: cancelled.is_set() or len(available_devices) >= count)
            if cancelled.is_set():
                raise RuntimeError("matrix cancelled before this trial was admitted")
            if tier == "l3":
                devices = sorted(available_devices)[:count]
                available_devices.difference_update(devices)
        try:
            receipt = run_trial(
                path,
                config,
                tier=tier,
                group=group,
                repetition=repetition,
                run_root=output,
                devices=devices or None,
                corpus_override=corpus_override,
                cancelled=cancelled,
            )
            # Retain completed cells even when a later capture in this lineage fails.
            with receipt_lock:
                receipts.append(receipt)
            return receipt
        finally:
            with device_condition:
                available_devices.update(devices)
                device_condition.notify_all()

    def execute_lineage(key, paths, repetition):
        cluster, corpus_id = key
        warm = output / "warm-initial" / str(repetition) / cluster / corpus_id
        warm.parent.mkdir(parents=True, exist_ok=True)
        filtered_corpus(Path(config.corpora[corpus_id].path), warm, excluded)
        for path in paths:
            receipt = execute_cell(path, Group.G3, repetition, corpus_override=warm)
            if specs[path].task.role == "seed":
                if (
                    receipt.terminal_status not in {TerminalStatus.PASS, TerminalStatus.FAIL}
                    or receipt.auxiliary_metrics.get("capture", {}).get("status") != "completed"
                ):
                    raise RuntimeError(
                        "G3 capture failed; dependent evals were not started "
                        f"(cluster={cluster}, corpus={corpus_id}, repetition={repetition}, "
                        f"status={receipt.terminal_status.value})"
                    )
                # Failed engineering attempts are still valid reference experience.
                # Held-out evaluations never become input to later evaluations.
                warm = Path(receipt.artifacts["private_trial_dir"]) / "profile/corpus"

    jobs = [
        (path, group, repetition)
        for path in task_roots
        for group in groups
        for repetition in range(1, repetitions + 1)
        if group != Group.G3
    ]
    lineage_jobs = [
        (key, paths, repetition)
        for repetition in range(1, repetitions + 1)
        for key, paths in lineages.items()
    ]
    matrix_error = None
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            try:
                # Interleave independent lineages and cells so neither class waits
                # behind the complete queue of the other before it can start.
                for index in range(max(len(jobs), len(lineage_jobs))):
                    if index < len(jobs):
                        futures.append(pool.submit(execute_cell, *jobs[index]))
                    if index < len(lineage_jobs):
                        futures.append(pool.submit(execute_lineage, *lineage_jobs[index]))
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        # A failed lineage stops its dependents. Independent
                        # admitted trials can still finish their paid work.
                        if matrix_error is None:
                            matrix_error = error
                        else:
                            matrix_error.add_note(
                                f"another independent matrix job failed: {type(error).__name__}: {error}"
                            )
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                with device_condition:
                    device_condition.notify_all()
                raise
    except BaseException as error:
        matrix_error = error
    try:
        write_json(output / "summary.json", grouped_summaries(receipts), exclusive=True)
    except Exception as error:
        if matrix_error is not None:
            matrix_error.add_note(f"matrix summary write also failed: {error}")
            raise matrix_error from error
        raise
    if matrix_error is not None:
        raise matrix_error
    return receipts
