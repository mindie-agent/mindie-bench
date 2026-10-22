"""Trial preflight helpers shared by the runner orchestrator.

[Role]
Resolves runtime read grants, probes helper imports and capture Git, classifies
terminal failures, and builds native capture payloads before a trial starts.

[Invariants]
- A runtime helper or capture-Git identity mismatch fails closed before model
  admission; protected and private read grants must never overlap.
- Failure classification preserves the truthful stage: only agent uncertainty or
  unknown external effects become `UNKNOWN`.

[Timeouts & Locks]
- Runtime import probe: `20s`; capture Git probe: `30s`; no locks are held.

[Side-Effect Policy]
- Writes runtime/capture preflight diagnostics under the private trial directory
  and installs only owned shim scripts under the profile root.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mindie_bench import __version__
from mindie_bench.evaluation.corpus_linter import CorpusLintRejected
from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.io import digest, read_json, sha256_file, task_hash, write_json
from mindie_bench.models import (
    ExperimentConfig,
    Group,
    TaskSpec,
    TerminalStatus,
    Trajectory,
    UnknownExternalEffects,
)
from mindie_bench.runner.drivers import driver_for
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.source_projection import HARNESS_CONFIG_PATHS
from mindie_bench.runner.verifier import validate_remote_oracle_contract


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
            p.relative_to(root).as_posix(): sha256_file(p)
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
    directory.mkdir(exist_ok=True)
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
    prefix = str(directory) + os.pathsep
    if not profile.env["PATH"].startswith(prefix):
        profile.env["PATH"] = prefix + profile.env["PATH"]



def prepare_trial(
    task_root,
    config,
    *,
    tier,
    group,
    repetition,
    run_root,
    experiment_context,
    expected_task_digest,
):
    """Create the trial directory, base fingerprint, and mutable run state."""
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
    # A completed primary verdict is immutable. Later capture, scoring, and
    # cleanup work is secondary evidence: its failures are recorded without
    # replacing the verified business outcome.
    primary_verdict = None
    return dict(
        task_root=task_root,
        run_root=run_root,
        trial_started=trial_started,
        task=task,
        task_digest=task_digest,
        trial_id=trial_id,
        trial_dir=trial_dir,
        fingerprint=fingerprint,
        environment=environment,
        driver=driver,
        trajectory=trajectory,
        status=status,
        reason=reason,
        errors=errors,
        auxiliary=auxiliary,
        primary=primary,
        profile=profile,
        slot=slot,
        workspace=workspace,
        remote=remote,
        result=result,
        gateway=gateway,
        native_service_instance=native_service_instance,
        agent_phase_budget=agent_phase_budget,
        phase=phase,
        wall_clock=wall_clock,
        verifier_record=verifier_record,
        verifier_seconds=verifier_seconds,
        capture_started=capture_started,
        primary_verdict=primary_verdict,
    )


def preflight_trial_admission(
    *,
    task_root,
    task,
    config,
    tier,
    group,
    environment,
    auxiliary,
    expected_oracle_sha256,
):
    """Validate admission and freeze the oracle before any model request."""
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
    return oracle_manifest


def probe_harness_version(config, profile, paths, trial_dir):
    """Resolve the exact harness binary and enforce its declared version."""
    binary = Path(config.harness.executable).resolve(strict=True)
    with binary.open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    version_cmd = native_command(
        [str(Path(config.harness.executable).absolute()), "--version"],
        read=paths,
        write=[profile.root],
        cwd=profile.root,
        policy_path=trial_dir / "version.sb",
    )
    version_result = subprocess.run(
        version_cmd,
        env=profile.env,
        cwd=profile.root,
        timeout=20,
        capture_output=True,
        text=True,
    )
    (trial_dir / "version.stdout").write_text(version_result.stdout)
    (trial_dir / "version.stderr").write_text(version_result.stderr)
    version_result.check_returncode()
    version = version_result.stdout.strip()
    if version != config.harness.expected_version:
        raise ValueError(
            "harness version mismatch: "
            f"expected {config.harness.expected_version!r}, got {version!r}"
        )
    return dict(binary_sha256=binary_sha256, version=version)


def build_agent_prompt(profile, task_root, *, remote, devices):
    """Build the one initial prompt; remote and local workspace scopes differ."""
    from mindie_bench.runner.profile_builder import knowledge_prompt

    prompt = knowledge_prompt(profile) + (task_root / "instruction.md").read_text()
    if remote:
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
    return prompt
