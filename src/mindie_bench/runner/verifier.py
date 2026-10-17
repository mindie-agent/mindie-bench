"""The judge runs after agent termination and must distinguish failure from outage."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import shlex
import subprocess
import time
import tomllib
from pathlib import Path

from mindie_bench.evaluation.oracle_freezer import (
    FrozenOracle,
    OracleTampered,
    check_oracle,
    check_probe,
    freeze_oracle,
    tampered_verdict,
)
from mindie_bench.evaluation.performance import validate_performance
from mindie_bench.io import write_json
from mindie_bench.models import (
    OracleContract,
    PerformanceContract,
    TaskSpec,
    UnknownExternalEffects,
    VerifierResult,
)
from mindie_bench.runner.sandbox import native_command


def parse_verifier(
    returncode: int,
    stdout: str,
    *,
    performance_contract: PerformanceContract | None = None,
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
            performance_claim = any(
                item is not None
                for item in [
                    value.aclgraph_capture_compatible,
                    value.memory_workspace_delta_mb,
                    value.e2e_warm_server_speedup,
                    value.performance_evidence,
                ]
            )
            if performance_claim and performance_contract is None:
                raise ValueError("performance claims require a declared performance contract")
            if performance_contract is not None:
                if not performance_nonce or not oracle_manifest_sha256:
                    raise ValueError("performance requires a trusted run nonce and oracle identity")
                validate_performance(
                    value,
                    performance_contract,
                    run_nonce=performance_nonce,
                    oracle_manifest_sha256=oracle_manifest_sha256,
                )
            elif value.speedup_ratio is not None:
                value.warnings.append(
                    "Legacy speedup_ratio is unvalidated; it does not establish performance acceptance."
                )
            return value
    if performance_contract is not None:
        raise ValueError("performance verifier requires structured measured evidence")
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
) -> tuple[VerifierResult, dict]:
    if oracle_manifest is not None:
        try:
            check_oracle(oracle_manifest)
        except OracleTampered as error:
            value = tampered_verdict(error)
            return value, {
                "oracle": {
                    "status": "oracle_tampered",
                    "manifest_sha256": oracle_manifest.sha256,
                    "error": str(error),
                },
                "result": value.model_dump(),
            }
    if (task_root / "task.toml").is_file():
        task = TaskSpec.model_validate(tomllib.loads((task_root / "task.toml").read_text()))
        oracle_contract = oracle_contract if oracle_contract is not None else task.oracle
        performance_contract = (
            performance_contract if performance_contract is not None else task.performance
        )
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
    deadline = time.monotonic() + timeout

    def execute(arguments: list[str], name: str):
        command = native_command(
            arguments,
            read=[
                workspace,
                *(task_root / relative for relative in manifest.protected_paths),
                *read_paths,
                *judge_reads,
            ],
            write=[profile.root / "tmp"],
            cwd=workspace,
            policy_path=trial_dir / f"{name}.sb",
        )
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            timeout=max(0.001, deadline - time.monotonic()),
            capture_output=True,
        )
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
        result = execute(["/bin/bash", str(task_root / "verifier/test.sh")], "verifier")
        record["returncode"] = result.returncode
        value = parse_verifier(
            result.returncode,
            result.stdout.decode("utf-8"),
            performance_contract=performance_contract,
            performance_nonce=nonce,
            oracle_manifest_sha256=manifest.sha256,
        )
    except Exception as error:
        record["original_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            oracle_record["after"] = check_oracle(manifest, oracle_contract)
        except OracleTampered as integrity_error:
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


_PRIVATE_REMOTE_JUDGE = r"""
import base64, hashlib, json, os, pathlib, subprocess, sys, time
payload=json.loads(base64.b64decode(sys.argv[1], validate=True))
root=pathlib.Path(os.environ["MINDIE_BENCH_ORACLE_ROOT"])
deadline=time.monotonic()+max(.001, payload["timeout"]-.1)
receipt=dict(schema_version="mindie-bench-private-oracle/1",
    nonce=os.environ["MINDIE_BENCH_PERFORMANCE_NONCE"],
    manifest_sha256=os.environ["MINDIE_BENCH_ORACLE_SHA256"],
    probes=[], judge=None, before=None, after=None, tampered=None, original_error=None)
class IntegrityError(Exception): pass
def inventory():
    result={}
    for path in root.rglob("*"):
        if path.is_symlink(): raise IntegrityError("private oracle symlink: "+str(path.relative_to(root)))
        if path.is_file():
            result[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    if result!=payload["expected"]:
        changed=sorted(k for k in result.keys()|payload["expected"].keys()
                       if result.get(k)!=payload["expected"].get(k))
        raise IntegrityError("private oracle SHA-256 changed: "+", ".join(changed))
    return dict(status="intact", file_count=len(result))
def execute(command):
    result=subprocess.run(command, capture_output=True,
                          timeout=max(.001, deadline-time.monotonic()),
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return dict(returncode=result.returncode,
                stdout_b64=base64.b64encode(result.stdout).decode(),
                stderr_b64=base64.b64encode(result.stderr).decode(),
                stdout_sha256=hashlib.sha256(result.stdout).hexdigest())
try:
    for relative, encoded in payload["files"].items():
        rel=pathlib.PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts: raise IntegrityError("invalid private oracle path")
        path=root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded, validate=True))
        path.chmod(0o444)
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(0o555)
    root.chmod(0o555)
    receipt["before"]=inventory()
    for probe in payload["probes"]:
        command=[part.replace("{oracle}", str(root)) for part in probe["command"]]
        observed=execute(command)
        observed["name"]=probe["name"]
        receipt["probes"].append(observed)
        if (observed["returncode"]!=probe["expected_returncode"]
            or observed["stdout_sha256"]!=probe["stdout_sha256"]):
            raise IntegrityError("oracle negative control "+repr(probe["name"])+" failed: exit="
                +str(observed["returncode"])+", stdout_sha256="+observed["stdout_sha256"])
    receipt["judge"]=execute(["/bin/bash", str(root/"verifier/test.sh")])
except IntegrityError as error:
    receipt["tampered"]=str(error)
except Exception as error:
    receipt["original_error"]=dict(type=type(error).__name__, message=str(error))
finally:
    try: receipt["after"]=inventory()
    except IntegrityError as error: receipt["tampered"]=str(error)
    except Exception as error:
        receipt["original_error"]=receipt["original_error"] or dict(type=type(error).__name__,message=str(error))
print(json.dumps(receipt, allow_nan=False), flush=True)
sys.exit(2 if receipt["original_error"] else 0)
"""


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
    try:
        result = remote.verify(
            _private_remote_script(oracle_manifest, oracle_contract, timeout),
            env={
                "MINDIE_BENCH_PERFORMANCE_NONCE": nonce,
                "MINDIE_BENCH_ORACLE_SHA256": oracle_manifest.sha256,
            },
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
        record["oracle"]["private_bundle"] = {
            "before": envelope["before"],
            "after": envelope["after"],
            "transport_exit_code": result["exit_code"],
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
        if envelope["original_error"] is not None:
            record["original_error"] = envelope["original_error"]
            raise RuntimeError(f"private remote oracle failed: {envelope['original_error']}")
        if envelope["tampered"] is not None:
            value = tampered_verdict(OracleTampered(envelope["tampered"]))
            record["oracle"].update(status="oracle_tampered", error=envelope["tampered"])
        else:
            if result["exit_code"] != 0:
                raise RuntimeError(f"private remote oracle cleanup failed (exit={result['exit_code']})")
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
