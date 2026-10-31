"""Post-agent task verification and private Oracle integrity enforcement.

[Role]
Executes local or remote verification scripts (`verifier/test.sh`) and negative-control
Oracle probes after agent execution finishes, separating business failure (`completed_fail`)
from infrastructure failure (`infra_failed`) and uncertain state (`unknown`).

[Invariants]
- `returncode == 0` must match `reward == 1.0`; mismatches raise `ValueError` (infra failure).
- Local and remote Oracle bundles are verified via SHA-256 inventory before and after the judge runs.
  Any mutation yields `reward=0.0, oracle_status="oracle_tampered"`.
- A verifier `subprocess.TimeoutExpired` inside `_PRIVATE_REMOTE_JUDGE` is caught before `finally:`
  runs `inventory()`, returning structured `timed_out=True` so `remote_verifier` records a clean
  `completed_fail` (`reward=0.0`) rather than `UnknownExternalEffects`.

[Timeouts & Locks]
- Inner judge timeout: `timeout` (`verifier_timeout_sec`).
- Remote worker outer timeout: `outer_timeout = timeout + 20.0s` (reserves the 5s
  private-oracle inventory grace plus process-group drain and JSON flush).
- Host pipe wait: `timeout + 25.0s`.

[Side-Effect Policy]
- Staging the remote private Oracle uses a `mktemp -d "${TMPDIR}/mindie-oracle.XXXXXX"` directory
  protected by `chmod 0444/0555` during execution and removed by an `EXIT` trap afterwards.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import time
import tomllib
from pathlib import Path

from mindie_bench.evaluation.inference import validate_inference
from mindie_bench.evaluation.oracle_freezer import (
    FrozenOracle,
    OracleTampered,
    check_oracle,
    check_probe,
    freeze_oracle,
    tampered_verdict,
)
from mindie_bench.evaluation.performance import validate_performance
from mindie_bench.inference_models import InferenceContract
from mindie_bench.io import write_json
from mindie_bench.models import (
    OracleContract,
    PerformanceContract,
    TaskSpec,
    UnknownExternalEffects,
    VerifierResult,
)
from mindie_bench.runner.sandbox import native_command


def _local_group_running(group: int) -> bool:
    """Inspect the owned POSIX group; zombies cannot perform further writes."""
    if os.name != "posix":
        return False
    result = subprocess.run(
        ["ps", "-e", "-o", "pgid=,stat="], capture_output=True, text=True, check=True, timeout=1,
    )
    return any(
        int(pgid) == group and not state.startswith("Z")
        for pgid, state in (line.split() for line in result.stdout.splitlines() if line.strip())
    )


def _drain_local_judge(process: subprocess.Popen) -> tuple[bytes, bytes]:
    """TERM, then KILL the owned group, and prove it has no running members."""
    if os.name != "posix":
        try:
            process.kill()
        except TimeoutError:
            raise
        except OSError:
            pass
        return process.communicate(timeout=2)
    running = True
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 0.15
        while running := _local_group_running(process.pid):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
    finally:
        # Even an inventory error must not prevent the final termination attempt.
        try:
            if running:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Darwin may report EPERM for a group containing only zombies.
                    if _local_group_running(process.pid):
                        raise
        finally:
            output = process.communicate(timeout=2)
    deadline = time.monotonic() + 1
    while _local_group_running(process.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError("local verifier process group survived termination")
        time.sleep(0.05)
    return output


def _run_local_judge(command, *, cwd, env, timeout):
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as error:
        try:
            stdout, stderr = _drain_local_judge(process)
        except Exception as cleanup_error:
            raise UnknownExternalEffects(
                f"local verifier cleanup could not be confirmed: {cleanup_error}"
            ) from error
        if isinstance(error, subprocess.TimeoutExpired):
            error.stdout, error.stderr = stdout, stderr
        raise
    finally:
        process.stdout.close()
        process.stderr.close()


def parse_verifier(
    returncode: int,
    stdout: str,
    *,
    performance_contract: PerformanceContract | None = None,
    inference_contract: InferenceContract | None = None,
    performance_nonce: str | None = None,
    oracle_manifest_sha256: str | None = None,
) -> VerifierResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if lines:
        try:
            raw = json.loads(lines[-1])
        except ValueError:
            raw = None
        if isinstance(raw, dict):
            value = VerifierResult.model_validate(raw)
            if (returncode == 0) != (value.reward == 1.0):
                raise ValueError("verifier reward contradicts exit status")
            if value.oracle_status == "oracle_tampered":
                return value
            cpu_scope = value.validation_scope in {
                "cpu_behavior", "cpu_tensor_correctness", "controlled_offline_fixture",
                "controlled_cpu_tensor_and_ablation_contract",
            }
            if cpu_scope and (value.native_kernel_bound is True
                              or value.measured_npu_speedup is not None
                              or value.speedup_ratio is not None):
                raise ValueError("CPU behavior/fixture evidence cannot establish native binding or NPU speedup")
            if value.performance_status == "unmeasured" and (
                value.measured_npu_speedup is not None or value.speedup_ratio is not None
            ):
                raise ValueError("unmeasured performance cannot carry a speedup")
            if inference_contract is not None:
                inferred = validate_inference(
                    inference_contract, value.inference_evidence,
                    run_nonce=performance_nonce, oracle_manifest_sha256=oracle_manifest_sha256,
                )
                # Existing task gates may reject a valid inference outcome, but
                # an inference gate failure can never be promoted to a pass.
                if value.reward == 1.0 and not inferred["accepted"]:
                    raise ValueError("verifier reward contradicts inference lifecycle gates")
            elif value.inference_evidence is not None:
                raise ValueError("inference evidence requires a declared inference contract")
            performance_claim = any(
                item is not None
                for item in [
                    value.aclgraph_capture_compatible,
                    value.memory_workspace_delta_mb,
                    value.e2e_warm_server_speedup,
                    value.performance_evidence,
                    value.measured_npu_speedup,
                ]
            )
            if performance_claim and performance_contract is None:
                raise ValueError("performance claims require a declared performance contract")
            if performance_contract is not None:
                if not performance_nonce or not oracle_manifest_sha256:
                    raise ValueError("performance requires a trusted run nonce and oracle identity")
                measured = validate_performance(
                    value,
                    performance_contract,
                    run_nonce=performance_nonce,
                    oracle_manifest_sha256=oracle_manifest_sha256,
                )
                if (value.measured_npu_speedup is not None
                        and value.measured_npu_speedup != measured["speedup_ratio"]):
                    raise ValueError("measured_npu_speedup disagrees with trusted timing samples")
            elif value.speedup_ratio is not None:
                value.warnings.append(
                    "Legacy speedup_ratio is unvalidated; it does not establish performance acceptance."
                )
            return value
    if performance_contract is not None or inference_contract is not None:
        raise ValueError("performance/inference verifier requires structured measured evidence")
    if returncode == 0:
        return VerifierResult(reward=1.0)
    raise ValueError("nonzero verifier without structured reward is infrastructure failure")


def local_verifier(
    task_root: Path,
    workspace: Path,
    profile,
    trial_dir: Path,
    timeout: float,
    read_paths: list[Path],
    *,
    oracle_manifest: FrozenOracle | None = None,
    oracle_contract: OracleContract | None = None,
    performance_contract: PerformanceContract | None = None,
    inference_contract: InferenceContract | None = None,
) -> tuple[VerifierResult, dict]:
    if (task_root / "task.toml").is_file():
        task = TaskSpec.model_validate(tomllib.loads((task_root / "task.toml").read_text()))
        oracle_contract = oracle_contract if oracle_contract is not None else task.oracle
        performance_contract = (
            performance_contract if performance_contract is not None else task.performance
        )
        inference_contract = inference_contract if inference_contract is not None else task.inference
    oracle_contract = oracle_contract or OracleContract()
    manifest = oracle_manifest or freeze_oracle(task_root, oracle_contract)
    if manifest.root != task_root.resolve():
        raise ValueError("frozen oracle belongs to a different task root")
    oracle_record = {
        "manifest_sha256": manifest.sha256,
        "frozen_before_foreground": oracle_manifest is not None,
        "probes": [],
    }
    record = {"oracle": oracle_record}
    try:
        oracle_record["before"] = check_oracle(manifest, oracle_contract)
    except OracleTampered as error:
        value = tampered_verdict(error)
        oracle_record.update(status="oracle_tampered", error=str(error))
        return value, dict(record, result=value.model_dump())
    trajectory_path = trial_dir / "verifier_trajectory.json"
    judge_reads = [trajectory_path] if trajectory_path.exists() else []
    nonce = secrets.token_hex(32)
    env = dict(
        profile.env,
        MINDIE_BENCH_WORKSPACE=str(workspace),
        MINDIE_BENCH_TRAJECTORY=str(trajectory_path),
        MINDIE_BENCH_PERFORMANCE_NONCE=nonce,
        MINDIE_BENCH_ORACLE_SHA256=manifest.sha256,
        MINDIE_BENCH_ORACLE_ROOT=str(task_root.resolve()),
        PYTHONDONTWRITEBYTECODE="1",
    )
    extra_verifier_reads = []
    for env_key in ("MINDIE_BENCH_PYTHON", "MINDIE_BENCH_CPU_PYTHON"):
        candidate = os.environ.get(env_key, "").strip()
        if candidate:
            env[env_key] = candidate
            py_path = Path(candidate).expanduser()
            if py_path.is_file():
                extra_verifier_reads.extend(
                    [
                        py_path.parent.parent.resolve(strict=True),
                        py_path.resolve(strict=True).parent.parent,
                    ]
                )
    default_cpu_rt = Path("/tmp/mindie-bench-cpu-runtime-20261022")
    if default_cpu_rt.is_dir():
        extra_verifier_reads.append(default_cpu_rt.resolve(strict=True))
        default_cpu_py = default_cpu_rt / "venv" / "bin" / "python"
        if default_cpu_py.is_file():
            extra_verifier_reads.append(default_cpu_py.resolve(strict=True).parent.parent)
    deadline = time.monotonic() + timeout

    def execute(arguments: list[str], name: str):
        command = native_command(
            arguments,
            read=[
                workspace,
                *(task_root / relative for relative in manifest.protected_paths),
                *read_paths,
                *extra_verifier_reads,
                *judge_reads,
            ],
            write=[profile.root / "tmp"],
            cwd=workspace,
            policy_path=trial_dir / f"{name}.sb",
        )
        try:
            result = _run_local_judge(
                command, cwd=workspace, env=env,
                timeout=max(0.001, deadline - time.monotonic()),
            )
        except (subprocess.TimeoutExpired, UnknownExternalEffects) as error:
            timeout_error = error if isinstance(error, subprocess.TimeoutExpired) else error.__cause__
            if isinstance(timeout_error, subprocess.TimeoutExpired):
                (trial_dir / f"{name}.stdout").write_bytes(timeout_error.stdout or b"")
                (trial_dir / f"{name}.stderr").write_bytes(timeout_error.stderr or b"")
            raise
        (trial_dir / f"{name}.stdout").write_bytes(result.stdout)
        (trial_dir / f"{name}.stderr").write_bytes(result.stderr)
        return result

    try:
        for index, probe in enumerate(oracle_contract.probes):
            result = execute(
                [argument.replace("{oracle}", str(task_root.resolve())) for argument in probe.command],
                f"oracle_probe_{index}",
            )
            probe_record = {"name": probe.name, "returncode": result.returncode}
            oracle_record["probes"].append(probe_record)
            probe_record.update(check_probe(probe.name, result.returncode, result.stdout, probe))
        bash_bin = (
            "/bin/bash"
            if Path("/bin/bash").exists()
            else (shutil.which("bash", path=env.get("PATH")) or shutil.which("bash") or "/bin/bash")
        )
        result = execute([bash_bin, str(task_root / "verifier/test.sh")], "verifier")
        record["returncode"] = result.returncode
        value = parse_verifier(
            result.returncode,
            result.stdout.decode("utf-8"),
            performance_contract=performance_contract,
            inference_contract=inference_contract,
            performance_nonce=nonce,
            oracle_manifest_sha256=manifest.sha256,
        )
    except Exception as error:
        record["original_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            oracle_record["after"] = check_oracle(manifest, oracle_contract)
        except OracleTampered as integrity_error:
            if isinstance(error, UnknownExternalEffects):
                raise error from integrity_error
            error = integrity_error
        if not isinstance(error, OracleTampered):
            raise
        value = tampered_verdict(error)
        oracle_record.update(status="oracle_tampered", error=str(error))
        return value, dict(record, result=value.model_dump())
    try:
        oracle_record["after"] = check_oracle(manifest, oracle_contract)
    except OracleTampered as error:
        record["original_result"] = value.model_dump()
        value = tampered_verdict(error)
        oracle_record.update(status="oracle_tampered", error=str(error))
    else:
        oracle_record["status"] = value.oracle_status or "intact"
        value.oracle_status = value.oracle_status or "intact"
    return value, dict(record, result=value.model_dump())


def validate_remote_oracle_contract(contract: OracleContract) -> None:
    """Call before admission; probe paths use an explicit {oracle} root token."""
    if any(not probe.command[0] for probe in contract.probes):
        raise ValueError("remote oracle probe requires a nonempty executable")


_PRIVATE_REMOTE_JUDGE = Path(__file__).with_name("private_remote_judge.py").read_text(encoding="utf-8")


def _private_remote_script(manifest: FrozenOracle, contract: OracleContract, timeout: float) -> bytes:
    files = {}
    for relative, expected in manifest.files:
        raw = (manifest.root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise OracleTampered(f"oracle changed while staging private transport: {relative}")
        files[relative] = base64.b64encode(raw).decode("ascii")
    payload = base64.b64encode(
        json.dumps(
            {
                "files": files,
                "expected": dict(manifest.files),
                "probes": [probe.model_dump(mode="json") for probe in contract.probes],
                "timeout": timeout,
            }
        ).encode()
    ).decode("ascii")
    command = shlex.join(["python3", "-I", "-c", _PRIVATE_REMOTE_JUDGE, payload])
    return (
        "set -u\n"
        'oracle_tmp=$(mktemp -d "${TMPDIR:?}/mindie-oracle.XXXXXX") || exit 73\n'
        "cleanup_oracle() {\n"
        "  original_status=$?\n"
        "  trap - EXIT\n"
        '  if ! chmod -R u+w "$oracle_tmp" || ! rm -rf "$oracle_tmp"; then\n'
        '    echo "private oracle cleanup failed" >&2\n'
        '    if [ "$original_status" -eq 0 ]; then original_status=74; fi\n'
        "  fi\n"
        '  exit "$original_status"\n'
        "}\n"
        "trap cleanup_oracle EXIT\n"
        'export MINDIE_BENCH_ORACLE_ROOT="$oracle_tmp"\n' + command + "\n"
    ).encode()


def remote_verifier(
    task_root: Path,
    remote,
    trial_dir: Path,
    timeout: float,
    oracle_manifest: FrozenOracle,
    oracle_contract: OracleContract,
    performance_contract: PerformanceContract | None,
    inference_contract: InferenceContract | None = None,
) -> tuple[VerifierResult, dict]:
    validate_remote_oracle_contract(oracle_contract)
    if oracle_manifest.root != task_root.resolve():
        raise ValueError("frozen oracle belongs to a different task root")
    record = {"oracle": {"manifest_sha256": oracle_manifest.sha256, "probes": []}}
    try:
        record["oracle"]["before"] = check_oracle(oracle_manifest, oracle_contract)
    except OracleTampered as error:
        value = tampered_verdict(error)
        record["oracle"].update(status="oracle_tampered", error=str(error))
        return value, dict(record, result=value.model_dump())
    nonce = secrets.token_hex(32)
    verifier_env = {
        "MINDIE_BENCH_PERFORMANCE_NONCE": nonce,
        "MINDIE_BENCH_ORACLE_SHA256": oracle_manifest.sha256,
    }
    # Remote verifiers must receive the candidate checkout through the same
    # documented contract as local verifiers. cwd is also the workspace, but an
    # explicit variable prevents a judge from falling back to the oracle tree.
    workspace = getattr(remote, "path", None)
    if workspace:
        verifier_env["MINDIE_BENCH_WORKSPACE"] = str(workspace)
    try:
        result = remote.verify(
            _private_remote_script(oracle_manifest, oracle_contract, timeout),
            env=verifier_env,
            timeout=timeout,
        )
        write_json(trial_dir / "verifier.remote.json", result)
        transport_stdout = base64.b64decode(result["stdout_b64"], validate=True)
        transport_stderr = base64.b64decode(result["stderr_b64"], validate=True)
        (trial_dir / "verifier.stdout").write_bytes(transport_stdout)
        (trial_dir / "verifier.stderr").write_bytes(transport_stderr)
        envelope = json.loads(transport_stdout)
        if (
            envelope.get("schema_version") != "mindie-bench-private-oracle/1"
            or envelope.get("nonce") != nonce
            or envelope.get("manifest_sha256") != oracle_manifest.sha256
        ):
            raise ValueError("private remote oracle receipt identity mismatch")
        timed_out = bool(envelope.get("timed_out", False))
        record["timed_out"] = timed_out
        record["oracle"]["private_bundle"] = {
            "before": envelope["before"],
            "after": envelope["after"],
            "transport_exit_code": result["exit_code"],
            "timed_out": timed_out,
        }
        for index, observed in enumerate(envelope["probes"]):
            stdout = base64.b64decode(observed["stdout_b64"], validate=True)
            stderr = base64.b64decode(observed["stderr_b64"], validate=True)
            (trial_dir / f"oracle_probe_{index}.stdout").write_bytes(stdout)
            (trial_dir / f"oracle_probe_{index}.stderr").write_bytes(stderr)
            record["oracle"]["probes"].append(
                {key: observed[key] for key in ["name", "returncode", "stdout_sha256"]}
            )
        judge = envelope["judge"]
        if judge is not None:
            stdout = base64.b64decode(judge["stdout_b64"], validate=True)
            stderr = base64.b64decode(judge["stderr_b64"], validate=True)
            (trial_dir / "verifier.stdout").write_bytes(stdout)
            (trial_dir / "verifier.stderr").write_bytes(stderr + transport_stderr)
            record["returncode"] = judge["returncode"]
        else:
            (trial_dir / "verifier.stderr").write_bytes(transport_stderr)
        if envelope["tampered"] is not None:
            value = tampered_verdict(OracleTampered(envelope["tampered"]))
            record["oracle"].update(status="oracle_tampered", error=envelope["tampered"])
        elif timed_out:
            record["original_error"] = envelope["original_error"]
            value = VerifierResult(
                reward=0.0,
                oracle_status="intact",
                timed_out=True,
                failure_reason=f"verifier timed out after {timeout}s",
            )
        elif envelope["original_error"] is not None:
            record["original_error"] = envelope["original_error"]
            raise RuntimeError(f"private remote oracle failed: {envelope['original_error']}")
        else:
            if result["exit_code"] != 0:
                # The judge already produced its primary verdict. A failed
                # post-judge cleanup (for example an unwritable mktemp tree)
                # is a warning attached to that verdict, never a replacement
                # for it.
                cleanup_warning = (
                    "private remote oracle cleanup failed "
                    f"(exit={result['exit_code']})"
                )
                record["oracle"]["cleanup_warning"] = cleanup_warning
                record["cleanup_warning"] = cleanup_warning
                record["cleanup"] = dict(status="failed", warning=cleanup_warning)
            if len(envelope["probes"]) != len(oracle_contract.probes) or judge is None:
                raise ValueError("required remote oracle probes or judge are missing")
            for observed, probe in zip(envelope["probes"], oracle_contract.probes, strict=True):
                check_probe(
                    probe.name,
                    observed["returncode"],
                    base64.b64decode(observed["stdout_b64"], validate=True),
                    probe,
                )
            value = parse_verifier(
                judge["returncode"],
                stdout.decode("utf-8"),
                performance_contract=performance_contract,
                inference_contract=inference_contract,
                performance_nonce=nonce,
                oracle_manifest_sha256=oracle_manifest.sha256,
            )
    except Exception as error:
        if getattr(remote, "operation_unknown", False) and not isinstance(error, UnknownExternalEffects):
            unknown = UnknownExternalEffects(f"remote verifier effects are unknown: {error}")
            try:
                check_oracle(oracle_manifest, oracle_contract)
            except OracleTampered as integrity_error:
                unknown.add_note(f"Also observed oracle_tampered: {integrity_error}")
            raise unknown from error
        record["original_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            record["oracle"]["after"] = check_oracle(oracle_manifest, oracle_contract)
        except OracleTampered as integrity_error:
            if isinstance(error, UnknownExternalEffects):
                error.add_note(f"Also observed oracle_tampered: {integrity_error}")
                raise error
            value = tampered_verdict(integrity_error)
            record["oracle"].update(status="oracle_tampered", error=str(integrity_error))
            return value, dict(record, result=value.model_dump())
        raise
    try:
        record["oracle"]["after"] = check_oracle(oracle_manifest, oracle_contract)
    except OracleTampered as error:
        record["original_result"] = value.model_dump()
        value = tampered_verdict(error)
        record["oracle"].update(status="oracle_tampered", error=str(error))
    else:
        record["oracle"]["status"] = value.oracle_status or "intact"
        value.oracle_status = value.oracle_status or "intact"
    return value, dict(record, result=value.model_dump())
