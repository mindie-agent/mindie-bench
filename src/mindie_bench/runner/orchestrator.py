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
from mindie_bench.evaluation.budget import BUDGET_STOPS
from mindie_bench.evaluation.corpus_linter import CorpusLintRejected
from mindie_bench.evaluation.oracle_freezer import freeze_oracle
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
from mindie_bench.runner.dsh_summary import SummaryHost, optional_host
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.mcp_probe import probe_servers
from mindie_bench.runner.native_product import optional_service
from mindie_bench.runner.phase_budget import lifecycle_overrun
from mindie_bench.runner.process import stream_process
from mindie_bench.runner.profile_builder import bind_mcp_transport, build_profile, filtered_corpus
from mindie_bench.runner.remote_slot import RemoteSlot
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS
from mindie_bench.runner.verifier import (
    local_verifier,
    remote_verifier,
    validate_remote_oracle_contract,
)
from mindie_bench.runner.worktree_slot import WorktreeSlot
from mindie_bench.scorer.comparator import grouped_summaries
from mindie_bench.scorer.metrics import trajectory_metrics


class ProductDeadlineExhausted(RuntimeError):
    """Preparation consumed the shared product deadline before any model call."""


def product_execution_window(
    experiment_context: dict, task, elapsed_sec: float
) -> tuple[float, dict]:
    """Use the predeclared total; never recompute a reserve for the actual arm.

    The experiment context already persists the cohort's total delivery budget.
    Its difference from the shared execution allowance is the reserve every arm
    consumes, including cleanup owned by services a particular arm does not start.
    """
    declared = float(experiment_context["max_wall_clock_sec"])
    execution = float(task.budget.timeout_sec)
    reserve = declared - execution
    if reserve < 0:
        raise ValueError("predeclared product budget is smaller than the execution allowance")
    window = min(execution, declared - elapsed_sec - reserve)
    if window <= 0:
        raise ProductDeadlineExhausted(
            "product preparation exhausted the shared execution deadline"
        )
    return window, {
        "contract": "phase-budget/2",
        "source": "experiment_context.max_wall_clock_sec",
        "execution_allowance_sec": execution,
        "post_agent_reserve_sec": reserve,
        "total_delivery_sec": declared,
    }


def failure_status(exc: BaseException, phase: str) -> TerminalStatus:
    """Classify a failure by its truthful stage; only agent uncertainty is UNKNOWN."""
    if isinstance(exc, CorpusLintRejected):
        return TerminalStatus.FAIL
    if isinstance(exc, ProductDeadlineExhausted):
        return TerminalStatus.INFRA
    if phase == "agent" or isinstance(exc, UnknownExternalEffects):
        return TerminalStatus.UNKNOWN
    return TerminalStatus.INFRA


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


RUNTIME_HELPER_MODULES = (
    "mindie_bench.runner.ssh_shim",
    "mindie_bench.runner.knowledge_stdio",
    "mindie_bench.integrations.codex_stop",
    "mindie_bench.integrations.claude_stop",
    "mindie_bench.integrations.first_party",
    "mindie_bench.integrations.graded_capture",
    "mindie_bench.runner.dsh_summary",
)


def native_capture_payload(harness, trajectory, profile, workspace, *, native_descriptor=None):
    identity = trajectory.native_session_id
    if not identity:
        raise ValueError("G3 capture requires the actual native session identity")
    stop_path = profile.root / "native-stop.json"
    if harness == "dsh":
        from mindie_bench.integrations.dsh_capture import bind_dsh_capture

        if native_descriptor is None or native_descriptor.get("session_id") != identity:
            raise ValueError(
                "DSH capture requires this foreground driver's completed session descriptor"
            )
        return bind_dsh_capture(
            profile_root=profile.root,
            dsh_home=Path(profile.env["DSH_HOME"]),
            workspace=workspace,
            observed=native_descriptor,
        )
    if harness == "codex":
        transcripts = list((Path(profile.env["CODEX_HOME"]) / "sessions").rglob(f"*-{identity}.jsonl"))
        if len(transcripts) != 1:
            raise ValueError("G3 cannot identify exactly one native task transcript")
        transcript = transcripts[0].resolve(strict=True)
    elif harness == "claude-code":
        from mindie_bench.integrations.claude_stop import read_stop

        event = read_stop(stop_path)
        if event["session_id"] != identity:
            raise ValueError("G3 Claude Stop differs from the foreground session identity")
        if Path(event["cwd"]).resolve(strict=True) != workspace.resolve(strict=True):
            raise ValueError("G3 Claude Stop differs from the foreground workspace")
        transcript = Path(event["transcript_path"]).resolve(strict=True)
        projects = (Path(profile.env["CLAUDE_CONFIG_DIR"]) / "projects").resolve(strict=True)
        if not transcript.is_relative_to(projects) or not transcript.is_file():
            raise ValueError("G3 Claude transcript is outside the owned native profile")
    else:
        raise ValueError(f"G3 capture is unsupported for harness {harness!r}")
    return {
        "harness": harness,
        "session_id": identity,
        "transcript_path": str(transcript),
        "cwd": str(workspace),
        "stop_event_path": str(stop_path),
    }


def require_corpus_lint(capture_receipt):
    if capture_receipt.get("corpus_lint", {}).get("status") != "passed":
        raise ValueError("G3 corpus quality gate did not pass; dependent evaluation is blocked")


def probe_capture_git(profile, paths: list[Path], trial_dir: Path) -> dict:
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from mindie_bench.integrations.first_party import preflight_git\n"
        "print(json.dumps(preflight_git(Path('capture-git-preflight'))))\n"
    )
    command = native_command(
        [sys.executable, "-I", "-c", code],
        read=paths,
        write=[profile.root],
        cwd=profile.root,
        policy_path=trial_dir / "capture-git-preflight.sb",
    )
    try:
        result = subprocess.run(
            command,
            env=dict(profile.env, **profile.private_env),
            cwd=profile.root,
            timeout=30,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as error:
        for name, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
            raw = value.encode() if isinstance(value, str) else value or b""
            (trial_dir / f"capture-git-preflight.{name}").write_bytes(raw)
        raise RuntimeError(
            "capture Git preflight timed out; inspect capture-git-preflight.stderr"
        ) from error
    (trial_dir / "capture-git-preflight.stdout").write_text(result.stdout)
    (trial_dir / "capture-git-preflight.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"capture Git preflight failed (exit={result.returncode}); "
            "inspect capture-git-preflight.stderr"
        )
    observed = json.loads(result.stdout)
    if (
        observed.get("status") != "completed"
        or observed.get("executable") != profile.private_env["MINDIE_BENCH_GIT"]
        or observed.get("exec_path") != profile.private_env["GIT_EXEC_PATH"]
        or observed.get("model_calls") != 0
    ):
        raise ValueError("capture Git preflight returned an invalid runtime identity")
    write_json(trial_dir / "capture-git-preflight.json", observed)
    return observed


def probe_runtime_import(profile, read_paths: list[Path], trial_dir: Path, expected_digest: str):
    code = (
        "import importlib,json,sys\n"
        "from pathlib import Path\n"
        "import mindie_bench\n"
        "from mindie_bench.runner.orchestrator import implementation_digest\n"
        f"names={RUNTIME_HELPER_MODULES!r}\n"
        "modules={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in names}\n"
        "print(json.dumps(dict(benchmark_source_sha256=implementation_digest(),"
        "executable=sys.executable,package_root=str(Path(mindie_bench.__file__).resolve().parent),"
        "modules=modules)))\n"
    )
    command = native_command(
        [sys.executable, "-I", "-c", code],
        read=read_paths,
        write=[profile.root],
        cwd=profile.root,
        policy_path=trial_dir / "runtime-preflight.sb",
    )
    diagnostic = dict(status="failed", expected_source_sha256=expected_digest)
    try:
        try:
            result = subprocess.run(
                command,
                env=profile.env,
                cwd=profile.root,
                timeout=20,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            for name, value in [("stdout", exc.stdout), ("stderr", exc.stderr)]:
                raw = value.encode() if isinstance(value, str) else value or b""
                (trial_dir / f"runtime-preflight.{name}").write_bytes(raw)
            raise RuntimeError(
                "runtime helper import timed out; inspect runtime-preflight.stderr"
            ) from exc
        (trial_dir / "runtime-preflight.stdout").write_text(result.stdout)
        (trial_dir / "runtime-preflight.stderr").write_text(result.stderr)
        diagnostic["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                f"runtime helper import failed (exit={result.returncode}); "
                "inspect runtime-preflight.stderr"
            )
        observed = json.loads(result.stdout)
        diagnostic["observed"] = observed
        if not isinstance(observed, dict) or not all(
            isinstance(observed.get(name), str)
            for name in ["benchmark_source_sha256", "executable", "package_root"]
        ):
            raise ValueError("runtime helper identity is malformed")
        modules = observed.get("modules")
        if not isinstance(modules, dict) or set(modules) != set(RUNTIME_HELPER_MODULES):
            raise ValueError("runtime module identity is malformed")
        root = Path(observed["package_root"])
        for name, path in modules.items():
            if not isinstance(path, str) or Path(path) != root.joinpath(
                *name.split(".")[1:]
            ).with_suffix(".py"):
                raise ValueError(f"runtime module identity mismatch: {name}")
        if Path(observed["executable"]).absolute() != Path(sys.executable).absolute():
            raise ValueError("runtime interpreter identity mismatch")
        if observed["benchmark_source_sha256"] != expected_digest:
            raise ValueError(
                "runtime implementation digest mismatch: "
                f"expected {expected_digest}, got {observed['benchmark_source_sha256']}"
            )
        diagnostic["status"] = "completed"
    except Exception as exc:
        diagnostic["error"] = dict(type=type(exc).__name__, message=str(exc))
        write_json(trial_dir / "runtime-preflight.json", diagnostic)
        raise
    write_json(trial_dir / "runtime-preflight.json", diagnostic)
    return observed


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
    experiment_context: dict | None = None,
    expected_task_digest: str | None = None,
    expected_oracle_sha256: str | None = None,
) -> TrialReceipt:
    task_root = task_root.resolve(strict=True)
    run_root = run_root.resolve(strict=True)
    trial_started = time.monotonic()
    task = TaskSpec.load(task_root)
    task_digest = task_hash(task_root)
    if expected_task_digest is not None and task_digest != expected_task_digest:
        raise ValueError("selected task changed after product preparation")
    trial_id = f"{task.task.instance_id}_{group.value}_{repetition}_{uuid.uuid4().hex[:12]}"
    trial_dir = run_root / trial_id
    trial_dir.mkdir(mode=0o700)
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
    if task.inference is not None:
        fingerprint["inference_contract"] = task.inference.model_dump(mode="json")
        if task.performance is not None:
            fingerprint["performance_contract"] = task.performance.model_dump(mode="json")
    environment = task.environment.get(tier)
    fingerprint.update(
        base_commit=environment.base_commit if environment else None,
        npu_model=environment.npu_model if environment else None,
        npu_count=environment.npu_count if environment else None,
        remote_environment=None,
    )
    # A product experiment is bound before execution. The internal component
    # mode remains G0..G3; historical receipts never acquire product labels.
    if experiment_context is not None:
        fingerprint["experiment_context"] = json.loads(json.dumps(experiment_context))
    write_json(
        trial_dir / "attempt_started.json",
        dict(
            trial_id=trial_id,
            task_digest=task_digest,
            warning="A missing terminal receipt is UNKNOWN. Never automatically resume this attempt.",
            **(
                {"experiment_context": fingerprint["experiment_context"]}
                if experiment_context is not None
                else {}
            ),
        ),
        exclusive=True,
    )
    driver = driver_for(config.harness.name)
    trajectory = Trajectory()
    status, reason = TerminalStatus.INFRA, "trial did not start"
    errors, auxiliary, primary = [], {}, {}
    profile = slot = workspace = remote = result = gateway = native_service_instance = None
    agent_phase_budget = None
    phase = "preflight"
    wall_clock = 0.0
    verifier_record = None
    verifier_seconds = None
    capture_started = None
    try:
        oracle_manifest = freeze_oracle(task_root, task.oracle)
        if (
            expected_oracle_sha256 is not None
            and oracle_manifest.sha256 != expected_oracle_sha256
        ):
            raise ValueError("selected oracle changed after product preparation")
        auxiliary["oracle_manifest"] = oracle_manifest.to_dict()
        if environment is None:
            raise ValueError(f"task has no {tier} environment")
        if task.task.evidence == "candidate":
            raise ValueError("candidate tasks require soundness validation before evaluation")
        if (
            task.task.evidence == "verified"
            and read_json(task_root / "soundness_receipt.json")["tier"] != tier
        ):
            raise ValueError("soundness proof does not cover the requested execution tier")
        if group == Group.G3 and not (config.groups.get(group) and config.groups[group].capture_command):
            raise ValueError("G3 requires an explicit offline capture adapter")
        if config.harness.name == "fixture" and (
            config.backend != "fixture" or task.task.evidence != "synthetic"
        ):
            raise ValueError("fixture harness is restricted to synthetic tasks")
        if config.harness.name != "fixture" and config.backend != "native":
            raise ValueError("native harnesses require the OS-enforced native backend")
        if (
            tier == "l3"
            and config.backend == "native"
            and (config.remote is None or not config.remote.component_runtime)
        ):
            raise ValueError("L3 native evaluation requires the five-component runtime configuration")
        if tier == "l3":
            validate_remote_oracle_contract(task.oracle)
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
        protected = [
            *(task_root / relative for relative in oracle_manifest.protected_paths),
            Path(environment.repo_path),
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
        binary = Path(config.harness.executable).resolve(strict=True)
        fingerprint["harness_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        version_cmd = native_command(
            [str(Path(config.harness.executable).absolute()), "--version"],
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
        if group == Group.G3 and config.harness.name == "dsh":
            from mindie_bench.runner import dsh_summary, graded_view

            capture_config = config.groups[group]
            if (
                capture_config.capture_protocol != "graded-c6"
                or not capture_config.dsh_summary_plugin
                or not task.mindie_knowledge.conditions
                or not graded_view.is_graded(profile.corpus)
            ):
                raise ValueError(
                    "DSH G3 requires a configured native summary host, graded C6 corpus and actual applicability conditions"
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
        if group == Group.G3 and config.harness.name == "kimi":
            raise ValueError(f"G3 native capture is not supported for the {config.harness.name} harness")
        if group == Group.G3:
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
                loopback_client_ports=(native_service_instance.port,) if native_service_instance else (),
                ephemeral_loopback_listener=native_service_instance is not None,
            )

            def sandboxed(arguments, policy_name):
                return native_command(
                    arguments, policy_path=trial_dir / policy_name, **native_grants
                )

            phase = "preparation"
            if experiment_context is not None and config.harness.name == "dsh":
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
                product_group = (experiment_context.get("cell") or {}).get("group")
                assert_catalog_policy(mounted_catalog["identity"], product_group)
                fingerprint["preflight_catalog"] = mounted_catalog
                auxiliary["preflight_catalog"] = mounted_catalog
            command = sandboxed(driver.command(profile, config.harness), "agent.sb")
            from mindie_bench.runner.profile_builder import knowledge_prompt

            prompt = knowledge_prompt(profile) + (task_root / "instruction.md").read_text()
            if remote:
                if "remote-dev" in profile.fingerprint.get("components", []):
                    prompt += (
                        f"\n\nWork on the isolated SSH endpoint bench-slot. "
                        f"Remote workspace: {remote.path}. "
                        f"Assigned NPU devices: {','.join(map(str, devices))}. "
                        'Use remote-dev MCP tools with host="bench-slot", port=22: prefer remote_grep, '
                        "remote_glob, remote_read, and remote_edit for fast remote code search, reading, and "
                        "surgical edits in the remote workspace, and use remote_bash (with yield_time_ms=10000 "
                        "or 30000) for running remote test scripts. The local directory is an empty client "
                        "scratchpad; the source snapshot lives in the remote workspace without Git history. "
                        "Apply the repair in the remote workspace that the NPU verifier will inspect.\n"
                    )
                else:
                    prompt += (
                        f"\n\nWork on the isolated SSH endpoint bench-slot. "
                        f"Remote workspace: {remote.path}. "
                        f"Assigned NPU devices: {','.join(map(str, devices))}. "
                        "The local directory is an empty client scratchpad; the source snapshot lives in the "
                        "remote workspace on bench-slot without Git history. Apply the repair in the remote "
                        "workspace that the NPU verifier will inspect. Choose validation proportional to the "
                        "change and observed failures.\n"
                    )
            else:
                prompt += (
                    "\n\nThe workspace is a plain source snapshot without Git history. "
                    "Apply the repair here and use the public reproducer or relevant available "
                    "tests for validation. Git history, stash and checkout are unavailable. "
                    "Choose validation proportional to the change and observed failures; "
                    "report any unavailable dependencies in your final answer.\n"
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
        phase = "postflight"
        if (
            group == Group.G3
            and status in {TerminalStatus.PASS, TerminalStatus.FAIL}
            and primary.get("performance_metrics", {}).get("oracle_status") != "oracle_tampered"
        ):
            phase = "capture"
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
                "cluster": task.task.cluster,
                "business_terminal_status": status.value,
                "capture_conditions": task.mindie_knowledge.conditions,
                "task_manifest": dict(
                    instance_id=task.task.instance_id,
                    conditions=task.mindie_knowledge.conditions,
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
            if auxiliary["captured_corpus_sha256"] == corpus_before_capture:
                raise ValueError("capture adapter reported completion without changing the local corpus")
            auxiliary["captured_corpus_path"] = str(profile.corpus)
    except KeyboardInterrupt:
        status = TerminalStatus.UNKNOWN if phase in {"agent", "capture"} else TerminalStatus.INFRA
        reason = "interrupted; reconcile existing effects before another attempt"
        errors.append(dict(stage=phase, type="KeyboardInterrupt", message=reason))
        raise
    except Exception as exc:
        errors.append(dict(stage=phase, type=type(exc).__name__, message=str(exc)))
        status = failure_status(exc, phase)
        if status == TerminalStatus.FAIL:
            primary.update(resolved=False, reward=0.0)
        elif phase == "postflight" and status != TerminalStatus.UNKNOWN:
            auxiliary["business_terminal_status"] = status.value
        reason = reason if status == TerminalStatus.PASS else f"{phase}: {type(exc).__name__}: {exc}"
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
                if status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
                    auxiliary["business_terminal_status"] = status.value
                    status, reason = TerminalStatus.INFRA, "native knowledge service cleanup failed"
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
            if config.backend == "native" and gateway and (trial_dir / "broker.jsonl").is_file():
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
            if status != TerminalStatus.UNKNOWN:
                status, reason = TerminalStatus.INFRA, "trajectory scoring failed"
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
                    if status in {TerminalStatus.PASS, TerminalStatus.FAIL}:
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
        if status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}:
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


def run_matrix(
    task_roots: list[Path],
    config: ExperimentConfig,
    *,
    tier: str,
    groups: list[Group],
    repetitions: int,
    concurrency: int,
    output: Path,
    canary_first: bool = False,
) -> list[TrialReceipt]:
    if concurrency < 1 or repetitions < 1 or not groups or len(set(groups)) != len(groups):
        raise ValueError("positive concurrency/repetitions and unique groups are required")
    task_roots = [path.resolve(strict=True) for path in task_roots]
    output = output.resolve()
    specs = {path: TaskSpec.load(path) for path in task_roots}
    if not specs or len(specs) != len(task_roots):
        raise ValueError("matrix task paths must be nonempty and unique")
    if len({spec.task.instance_id for spec in specs.values()}) != len(specs):
        raise ValueError("matrix task instance IDs must be unique")
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
    task_digests = {path: task_hash(path) for path in task_roots}
    cells = [
        dict(
            instance_id=spec.task.instance_id,
            tier=tier,
            group=group.value,
            repetition=repetition,
            role=spec.task.role,
            task_digest=task_digests[path],
            cluster=spec.task.cluster,
            corpus_id=spec.mindie_knowledge.corpus_id,
            inference_category=spec.inference.category if spec.inference else None,
            evidence=spec.task.evidence,
        )
        for path, spec in specs.items()
        for group in groups
        for repetition in range(1, repetitions + 1)
    ]
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
            cells=cells,
        ),
        exclusive=True,
    )
    receipts = []
    blocked_cells = []
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
        for index, path in enumerate(paths):
            receipt = execute_cell(path, Group.G3, repetition, corpus_override=warm)
            if specs[path].task.role == "seed":
                if (
                    receipt.terminal_status not in {TerminalStatus.PASS, TerminalStatus.FAIL}
                    or receipt.auxiliary_metrics.get("capture", {}).get("status") != "completed"
                    or receipt.auxiliary_metrics.get("capture", {}).get("corpus_lint", {}).get("status")
                    != "passed"
                ):
                    blocked = [
                        dict(
                            **cell,
                            reason="seed_prerequisite_not_completed",
                            prerequisite_trial_id=receipt.trial_id,
                            prerequisite_status=receipt.terminal_status.value,
                        )
                        for cell in cells
                        if cell["group"] == "G3"
                        and cell["repetition"] == repetition
                        and cell["instance_id"]
                        in {specs[remaining].task.instance_id for remaining in paths[index + 1 :]}
                    ]
                    with receipt_lock:
                        blocked_cells.extend(blocked)
                    raise RuntimeError(
                        "G3 seed prerequisites failed; dependent evals were not started "
                        f"(cluster={cluster}, corpus={corpus_id}, repetition={repetition}, "
                        f"status={receipt.terminal_status.value}, "
                        f"capture={receipt.auxiliary_metrics.get('capture', {}).get('status')})"
                    )
                # Failed engineering attempts are still valid reference experience.
                # Held-out evaluations never become input to later evaluations.
                warm = Path(receipt.artifacts["captured_corpus"])
                if tree_hash(warm) != receipt.auxiliary_metrics["captured_corpus_sha256"]:
                    raise ValueError("captured seed corpus changed before its held-out evaluation")

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
        if canary_first and len(cells) > 1:
            if jobs:
                canary = execute_cell(*jobs.pop(0))
                if (
                    canary.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
                    or canary.errors
                ):
                    raise RuntimeError(
                        f"canary gate failed on {canary.trial_id} "
                        f"(status={canary.terminal_status.value}); remaining cells were not admitted"
                    )
            elif lineage_jobs:
                execute_lineage(*lineage_jobs.pop(0))
                bad = next(
                    (
                        row
                        for row in receipts
                        if row.terminal_status in {TerminalStatus.INFRA, TerminalStatus.UNKNOWN}
                        or row.errors
                    ),
                    None,
                )
                if bad is not None:
                    raise RuntimeError(
                        f"canary gate failed on {bad.trial_id} "
                        f"(status={bad.terminal_status.value}); remaining cells were not admitted"
                    )
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
        write_json(output / "summary.json", grouped_summaries(
            receipts, expected_cells=cells, blocked_cells=blocked_cells,
        ), exclusive=True)
        from mindie_bench.evaluation.accounting import account_cells

        manifest_path = output / "eval_manifest.json"
        accounting = account_cells(read_json(manifest_path), receipts, blocked_cells=blocked_cells)
        write_json(
            output / "matrix_accounting.json",
            dict(
                schema="mindie-bench-matrix-accounting/1",
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                receipt_sha256={
                    row.trial_id: hashlib.sha256(
                        (Path(row.artifacts["private_trial_dir"]) / "trial_receipt.json").read_bytes()
                    ).hexdigest()
                    for row in receipts
                },
                **accounting,
            ),
            exclusive=True,
        )
    except Exception as error:
        if matrix_error is not None:
            matrix_error.add_note(f"matrix reporting write also failed: {error}")
            raise matrix_error from error
        raise
    if matrix_error is not None:
        raise matrix_error
    return receipts
