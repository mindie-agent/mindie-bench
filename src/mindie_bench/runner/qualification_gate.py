"""Zero-burden pre-baked qualification gates G0 through G3.

[Role]
Turn a declarative gate specification into bounded artifact, build, numerical,
sustained-ABBA and producer-consumer verification receipts so an optimization
agent does not hand-write build, symbol, graph-capture or timing glue.

[Invariants]
- Every receipt is `completed`, `failed` or `unknown`; a timeout, malformed
  artifact or unconfirmed PID is never reported as success or as a business fix.
- G0 reads native manifests, AICore objects, paired AICPU registrations,
  generated JSON and unresolved-symbol evidence; it never infers a kernel from
  a filename alone.
- G1 propagates `MAKEFLAGS=-jN`, `CMAKE_BUILD_PARALLEL_LEVEL=N` and
  `OPS_CPU_NUMBER=N` together; one missing variable can silently serialize the
  inner CANN OPC build. A live debug lock blocks the build; stale PIDs are
  reported, never killed.
- G2 keeps numerical qualification separate from latency and accepts only
  interleaved ABBA warmup/measured rounds with sustained-clock evidence.
- G3 requires an actually executed producer-consumer and TP module paired run;
  any required guard workload failure rejects the module regardless of speedup.
- `outer_timeout >= sum(inner stage budgets) + cleanup_grace` is checked before
  any process starts.

[Timeouts & Locks]
- Local commands run in their own POSIX session with the declared per-command
  timeout; cleanup sends `SIGTERM`, waits 2s, sends `SIGKILL`, then reaps for 5s.
- Lock inspection is read-only and uses `os.kill(pid, 0)` by default.

[Side-Effect Policy]
- Read-only artifact inspection. Build and qualification commands are executed
  only from the explicit G1 spec; locks are never removed and no NPU is reset.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from mindie_bench.runner.qualification_gate_models import (
    SCHEMA,
    AbbaSample,
    BoundaryPair,
    CommandReceipt,
    DebugLock,
    G0Spec,
    G1Spec,
    G2Spec,
    G3Spec,
    GateCommand,
    GateResult,
    GateState,
    GuardWorkload,
    ModuleRun,
    ModuleTopology,
    NativeOperatorArtifacts,
    NumericalQualification,
    StageReceipt,
    TimeoutArithmeticError,
    plan_gate_timeouts,
)
from mindie_bench.runner.qualification_gate_timing import (
    evaluate_g2_sustained_abba as _evaluate_g2_sustained_abba,
)
from mindie_bench.runner.qualification_gate_timing import (
    evaluate_g3_subgraph_module_gate as _evaluate_g3_subgraph_module_gate,
)
from mindie_bench.runner.qualification_gate_timing import (
    frozen_baseline_digest,
)
from mindie_bench.runner.termination import terminate_process_group

__all__ = [
    "SCHEMA",
    "AbbaSample",
    "ArtifactReader",
    "BoundaryPair",
    "CommandExecutor",
    "CommandReceipt",
    "DebugLock",
    "G0Spec",
    "G1Spec",
    "G2Spec",
    "G3Spec",
    "GateCommand",
    "GateResult",
    "GateState",
    "GuardWorkload",
    "LocalArtifactReader",
    "ModuleRun",
    "ModuleTopology",
    "NativeOperatorArtifacts",
    "NumericalQualification",
    "QualificationGateRunner",
    "StageReceipt",
    "TimeoutArithmeticError",
    "frozen_baseline_digest",
    "incremental_build_command",
    "plan_gate_timeouts",
    "run_local_command",
]

MAX_CAPTURE_BYTES = 4 * 1024 * 1024
UNRESOLVED = re.compile(
    r"undefined symbol|unresolved symbol|^\s*U\s", re.IGNORECASE | re.MULTILINE
)


class CommandExecutor(Protocol):
    def __call__(self, command: GateCommand) -> CommandReceipt: ...


class ArtifactReader(Protocol):
    def exists(self, path: str) -> bool: ...

    def read_bytes(self, path: str) -> bytes: ...


def run_local_command(command: GateCommand) -> CommandReceipt:
    """Run one bounded command in its own process group."""
    environment = {**os.environ, **command.env}
    argv = list(command.argv)
    if os.name == "nt" and argv:
        if argv[0] in {"python3", "python"}:
            resolved_py = shutil.which(argv[0], path=environment.get("PATH"))
            if not resolved_py or "WindowsApps" in resolved_py:
                argv[0] = sys.executable
        elif argv == ["true"] and not shutil.which("true", path=environment.get("PATH")):
            argv = [sys.executable, "-c", "pass"]
    try:
        process = subprocess.Popen(
            argv,
            cwd=command.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except TimeoutError:
        raise
    except OSError as error:
        return CommandReceipt(command.argv, None, error=f"{type(error).__name__}: {error}")
    try:
        stdout, stderr = process.communicate(timeout=float(command.timeout_sec))
    except subprocess.TimeoutExpired:
        try:
            terminate_process_group(process, term_wait=2, kill_wait=5)
        except (TimeoutError, OSError) as cleanup:
            return CommandReceipt(
                command.argv,
                None,
                timed_out=True,
                error=f"timeout cleanup failed: {type(cleanup).__name__}",
            )
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired as cleanup:
            return CommandReceipt(
                command.argv,
                None,
                timed_out=True,
                error=f"process group did not drain: {type(cleanup).__name__}",
            )
        return CommandReceipt(
            command.argv,
            process.returncode,
            stdout[:MAX_CAPTURE_BYTES],
            stderr[:MAX_CAPTURE_BYTES],
            timed_out=True,
        )
    return CommandReceipt(
        command.argv,
        process.returncode,
        stdout[:MAX_CAPTURE_BYTES],
        stderr[:MAX_CAPTURE_BYTES],
    )


class LocalArtifactReader:
    def exists(self, path: str) -> bool:
        return Path(path).is_file()

    def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()


def _manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise ValueError("native operator manifest must be a JSON object or list")
    for key in ("operators", "ops", "kernels"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    records = []
    for name, row in payload.items():
        if isinstance(row, dict) and any(
            key in row
            for key in ("aicore", "aicore_object", "aicpu", "aicpu_registration", "engine")
        ):
            records.append({"name": name, **row})
    return records


def _artifact_path(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for nested in ("path", "object", "binary", "file", "json", "registration"):
                if isinstance(value.get(nested), str) and value[nested]:
                    return value[nested]
    return None


def _operator_artifacts(payload: Any, *, manifest_path: str) -> list[NativeOperatorArtifacts]:
    rows = _manifest_rows(payload)
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        name = row.get("name") or row.get("op") or row.get("operator") or f"op-{index}"
        if not isinstance(name, str) or not name:
            raise ValueError(f"{manifest_path}: operator name must be text")
        entry = grouped.setdefault(str(name), {"name": str(name)})
        engine = str(row.get("engine", "")).casefold()
        aicore = _artifact_path(row, "aicore_object", "aicore_binary", "aicore", "object", "binary")
        aicpu = _artifact_path(
            row, "aicpu_registration", "aicpu_json", "aicpu", "registration", "json", "config"
        )
        if aicore:
            entry["aicore"] = aicore
        if aicpu:
            entry["aicpu"] = aicpu
        if "aicore" in engine and not aicore:
            raise ValueError(f"{manifest_path}: AICore operator {name} has no object path")
        if "aicpu" in engine and not aicpu:
            raise ValueError(f"{manifest_path}: AICPU operator {name} has no JSON registration")
    if not grouped:
        raise ValueError(f"{manifest_path}: manifest advertises no native operators")
    return [
        NativeOperatorArtifacts(name, entry.get("aicore"), entry.get("aicpu"))
        for name, entry in sorted(grouped.items())
    ]


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label}: not UTF-8 JSON: {error}") from error
    if "\x00" in text:
        raise ValueError(f"{label}: contains NUL bytes from a non-truncated generated file")
    duplicates: list[str] = []

    def unique(pairs):
        seen = set()
        value = {}
        for key, item in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            value[key] = item
        return value

    try:
        payload = json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}: trailing or malformed JSON bytes: {error}") from error
    if duplicates:
        raise ValueError(f"{label}: duplicate JSON keys {sorted(set(duplicates))}")
    return payload


class QualificationGateRunner:
    """Execute declarative G0-G3 receipts with one reusable harness."""

    def __init__(
        self,
        *,
        executor: CommandExecutor | None = None,
        reader: ArtifactReader | None = None,
        pid_alive=None,
    ):
        self.executor = executor or run_local_command
        self.reader = reader or LocalArtifactReader()
        self.pid_alive = pid_alive or _pid_alive

    def _run_command(self, name: str, command: GateCommand) -> StageReceipt:
        receipt = self.executor(command)
        evidence = {
            "argv": list(receipt.argv),
            "exit_code": receipt.exit_code,
            "stdout": receipt.stdout,
            "stderr": receipt.stderr,
            "timed_out": receipt.timed_out,
            "error": receipt.error,
        }
        if receipt.state is not GateState.COMPLETED:
            return StageReceipt(
                name, receipt.state, command.purpose or "command did not complete", evidence
            )
        missing = [marker for marker in command.expect_stdout if marker not in receipt.stdout]
        missing_err = [marker for marker in command.expect_stderr if marker not in receipt.stderr]
        forbidden = [marker for marker in command.forbidden_stdout if marker in receipt.stdout]
        if missing or missing_err or forbidden:
            detail = (
                f"sentinel missing stdout={missing} stderr={missing_err} "
                f"forbidden_present={forbidden}"
            )
            return StageReceipt(name, GateState.FAILED, detail, evidence)
        return StageReceipt(name, GateState.COMPLETED, command.purpose or "command passed", evidence)

    def _result(self, gate, stages, timeout_plan=None, metrics=None) -> GateResult:
        if any(stage.state is GateState.FAILED for stage in stages):
            state = GateState.FAILED
        elif any(stage.state is GateState.UNKNOWN for stage in stages):
            state = GateState.UNKNOWN
        else:
            state = GateState.COMPLETED
        return GateResult(
            gate=gate,
            state=state,
            stages=stages,
            metrics=metrics or {},
            reasons=[
                f"{stage.name}: {stage.detail}"
                for stage in stages
                if stage.state is not GateState.COMPLETED
            ],
            timeout_plan=timeout_plan,
        )

    def _derived_plan(self, stage_budgets, *, outer=None, grace=5.0) -> dict[str, Any]:
        return plan_gate_timeouts(
            stage_budgets, cleanup_grace_sec=grace, outer_timeout_sec=outer
        )

    def run_gate(self, gate: str, spec) -> GateResult:
        """Dispatch one declarative gate specification without glue code."""
        normalized = gate.upper()
        methods = {
            "G0": self.verify_g0_artifact_and_contract,
            "G1": self.build_and_verify_g1_incremental_gate,
            "G2": self.evaluate_g2_sustained_abba,
            "G3": self.evaluate_g3_subgraph_module_gate,
        }
        if normalized not in methods:
            raise ValueError("gate must be G0, G1, G2 or G3")
        return methods[normalized](spec)

    # ---- public G0 -------------------------------------------------------
    def verify_g0_artifact_and_contract(self, spec: G0Spec) -> GateResult:
        budgets = dict(spec.stage_budgets)
        if not budgets:
            budgets = {
                "contract": max(0.5, sum(cmd.timeout_sec for cmd in spec.contract_tests) or 0.5),
                "artifacts": 5.0,
                "symbols": max(
                    0.5,
                    sum(cmd.timeout_sec for cmd in spec.symbol_checks)
                    + 5.0 * len(spec.ldd_targets),
                ),
            }
        plan = self._derived_plan(
            budgets, outer=spec.outer_timeout_sec, grace=spec.cleanup_grace_sec
        )
        stages = [
            self._run_command(f"contract:{index}", command)
            for index, command in enumerate(spec.contract_tests)
        ]
        stages.extend(self._check_artifacts(spec))
        stages.extend(self._check_symbols(spec))
        if not spec.contract_tests:
            stages.append(
                StageReceipt("contract", GateState.FAILED, "no CPU contract test was declared")
            )
        if not (spec.manifests or spec.generated_json or spec.ldd_targets or spec.symbol_checks):
            stages.append(
                StageReceipt(
                    "artifacts",
                    GateState.UNKNOWN,
                    "no native artifact or unresolved-symbol evidence was declared",
                )
            )
        return self._result("G0", stages, plan)

    def _check_artifacts(self, spec: G0Spec) -> list[StageReceipt]:
        stages = []
        advertised = 0
        for manifest in spec.manifests:
            if not self.reader.exists(manifest):
                stages.append(
                    StageReceipt("manifest", GateState.FAILED, f"manifest is missing: {manifest}")
                )
                continue
            try:
                payload = _strict_json(self.reader.read_bytes(manifest), label=manifest)
                operators = _operator_artifacts(payload, manifest_path=manifest)
            except ValueError as error:
                stages.append(StageReceipt("manifest", GateState.FAILED, str(error)))
                continue
            problems = []
            base = Path(manifest).parent
            for operator in operators:
                if not operator.aicore_object:
                    problems.append(f"{operator.name}: no AICore .o advertised")
                    continue
                object_path = _resolve(base, operator.aicore_object)
                if not object_path.endswith(".o"):
                    problems.append(f"{operator.name}: AICore artifact is not .o: {object_path}")
                elif not self.reader.exists(object_path):
                    problems.append(f"{operator.name}: AICore .o missing: {object_path}")
                elif not self.reader.read_bytes(object_path):
                    problems.append(f"{operator.name}: AICore .o is empty: {object_path}")
                else:
                    advertised += 1
                if not operator.aicpu_registration:
                    problems.append(f"{operator.name}: paired AICPU JSON is missing")
                    continue
                registration = _resolve(base, operator.aicpu_registration)
                if not registration.endswith(".json") or not self.reader.exists(registration):
                    problems.append(f"{operator.name}: AICPU JSON missing: {registration}")
                    continue
                try:
                    aicpu = _strict_json(self.reader.read_bytes(registration), label=registration)
                except ValueError as error:
                    problems.append(str(error))
                    continue
                if not isinstance(aicpu, dict) or not aicpu:
                    problems.append(f"{operator.name}: AICPU JSON is not a registration object")
            stages.append(
                StageReceipt(
                    "manifest",
                    GateState.FAILED if problems else GateState.COMPLETED,
                    "; ".join(problems)
                    if problems
                    else f"{len(operators)} native operators advertise AICore/AICPU pairs",
                    {
                        "manifest": manifest,
                        "operators": [operator.name for operator in operators],
                        "aicore_objects_present": advertised,
                    },
                )
            )
        for path in spec.generated_json:
            if not self.reader.exists(path):
                stages.append(
                    StageReceipt(
                        "json_config", GateState.FAILED, f"generated JSON is missing: {path}"
                    )
                )
                continue
            try:
                _strict_json(self.reader.read_bytes(path), label=path)
            except ValueError as error:
                stages.append(StageReceipt("json_config", GateState.FAILED, str(error)))
            else:
                stages.append(
                    StageReceipt(
                        "json_config",
                        GateState.COMPLETED,
                        f"generated JSON parses cleanly: {path}",
                        {"path": path},
                    )
                )
        return stages

    def _check_symbols(self, spec: G0Spec) -> list[StageReceipt]:
        stages = []
        for target in spec.ldd_targets:
            if not self.reader.exists(target):
                stages.append(
                    StageReceipt("ldd", GateState.FAILED, f"symbol target is missing: {target}")
                )
                continue
            command = GateCommand(("ldd", "-r", target), timeout_sec=10.0,
                                  purpose=f"ldd -r {target}")
            receipt = self.executor(command)
            unresolved = UNRESOLVED.findall(receipt.stdout + "\n" + receipt.stderr)
            if receipt.timed_out or receipt.exit_code is None:
                stages.append(
                    StageReceipt(
                        "ldd",
                        GateState.UNKNOWN,
                        receipt.error or f"ldd -r did not complete: {target}",
                        {"target": target},
                    )
                )
            elif unresolved:
                stages.append(
                    StageReceipt(
                        "ldd",
                        GateState.FAILED,
                        f"{len(unresolved)} unresolved symbol report(s)",
                        {
                            "target": target,
                            "output": receipt.stdout[-4000:] + receipt.stderr[-4000:],
                        },
                    )
                )
            elif receipt.exit_code != 0:
                stages.append(
                    StageReceipt(
                        "ldd",
                        GateState.FAILED,
                        f"ldd -r exited {receipt.exit_code} for {target}",
                    )
                )
            else:
                stages.append(
                    StageReceipt(
                        "ldd",
                        GateState.COMPLETED,
                        f"zero unresolved symbols: {target}",
                        {"target": target},
                    )
                )
        for command in spec.symbol_checks:
            stage = self._run_command("symbol", command)
            if stage.state is GateState.COMPLETED:
                surface = str(stage.evidence.get("stdout", "")) + str(
                    stage.evidence.get("stderr", "")
                )
                if UNRESOLVED.search(surface):
                    stage = StageReceipt(
                        "symbol",
                        GateState.FAILED,
                        "symbol check reported unresolved references",
                        stage.evidence,
                    )
            stages.append(stage)
        return stages

    # ---- public G1 -------------------------------------------------------
    def inspect_debug_locks(self, paths: Sequence[str]) -> list[DebugLock]:
        """Read debug lock files; never remove or signal the recorded process."""
        locks = []
        for path in paths:
            if not self.reader.exists(path):
                locks.append(
                    DebugLock(path, None, None, GateState.COMPLETED, "debug lock is absent")
                )
                continue
            try:
                text = self.reader.read_bytes(path).decode("utf-8", errors="replace")
            except TimeoutError:
                raise
            except OSError as error:
                locks.append(
                    DebugLock(
                        path,
                        None,
                        None,
                        GateState.UNKNOWN,
                        f"debug lock unreadable: {type(error).__name__}",
                    )
                )
                continue
            pid = _lock_number(text, "pid")
            pgid = _lock_number(text, "pgid")
            # Some CANN debug helpers write only the owner PID on the first line.
            first = text.strip().splitlines()[0].strip() if text.strip() else ""
            if pid is None and first.isdigit() and int(first) > 0:
                pid = int(first)
            if pid is None:
                locks.append(
                    DebugLock(
                        path,
                        None,
                        pgid,
                        GateState.UNKNOWN,
                        "debug lock has no parseable PID; ownership is indeterminate",
                    )
                )
                continue
            alive = self.pid_alive(pid)
            if alive is None:
                locks.append(
                    DebugLock(
                        path,
                        pid,
                        pgid,
                        GateState.UNKNOWN,
                        "debug lock PID liveness is indeterminate",
                    )
                )
            elif alive:
                locks.append(
                    DebugLock(
                        path,
                        pid,
                        pgid,
                        GateState.FAILED,
                        f"debug lock PID {pid} is still alive; build must wait",
                    )
                )
            else:
                locks.append(
                    DebugLock(
                        path,
                        pid,
                        pgid,
                        GateState.COMPLETED,
                        f"stale debug lock PID {pid}; left untouched",
                    )
                )
        return locks

    def build_and_verify_g1_incremental_gate(self, spec: G1Spec) -> GateResult:
        build_timeout = (
            spec.build_command.timeout_sec if spec.build_command is not None else 120.0
        )
        eager_timeout = spec.eager_qualification.timeout_sec if spec.eager_qualification else 1.0
        graph_timeout = (
            spec.aclgraph_qualification.timeout_sec if spec.aclgraph_qualification else 1.0
        )
        budgets = dict(spec.stage_budgets) or {
            "locks": 1.0,
            "build": build_timeout,
            "eager": eager_timeout,
            "aclgraph": graph_timeout,
        }
        plan = self._derived_plan(
            budgets, outer=spec.outer_timeout_sec, grace=spec.cleanup_grace_sec
        )
        locks = self.inspect_debug_locks(spec.debug_lock_paths)
        lock_problems = [lock for lock in locks if lock.state is not GateState.COMPLETED]
        lock_state = (
            GateState.FAILED
            if any(lock.state is GateState.FAILED for lock in lock_problems)
            else GateState.UNKNOWN
            if lock_problems
            else GateState.COMPLETED
        )
        lock_stage = StageReceipt(
            "locks",
            lock_state,
            "; ".join(lock.detail for lock in locks) if locks else "no debug lock declared",
            {"locks": [lock.__dict__ | {"state": lock.state.value} for lock in locks]},
        )
        stages = [lock_stage]
        if spec.strict_locks and lock_state is not GateState.COMPLETED:
            return self._result("G1", stages, plan, {"locked": True})
        if spec.build_command is not None:
            command = replace(
                spec.build_command,
                env={
                    **dict(spec.build_command.env),
                    "MAKEFLAGS": f"-j{spec.jobs}",
                    "CMAKE_BUILD_PARALLEL_LEVEL": str(spec.jobs),
                    "OPS_CPU_NUMBER": str(spec.jobs),
                },
                expect_stdout=tuple(spec.build_sentinel) or spec.build_command.expect_stdout,
            )
        else:
            command = replace(
                incremental_build_command(
                    source_root=spec.source_root,
                    jobs=spec.jobs,
                    build_system=spec.build_system,
                    build_target=spec.build_target,
                    timeout_sec=build_timeout,
                ),
                expect_stdout=tuple(spec.build_sentinel),
            )
        stages.append(self._run_command("build", command))
        if stages[-1].state is not GateState.COMPLETED:
            return self._result("G1", stages, plan)
        if spec.eager_qualification is None:
            stages.append(
                StageReceipt(
                    "eager",
                    GateState.FAILED,
                    "no single-NPU Eager qualification command declared",
                )
            )
        else:
            stages.append(self._run_command("eager", spec.eager_qualification))
        if stages[-1].state is not GateState.COMPLETED:
            return self._result("G1", stages, plan)
        if spec.aclgraph_qualification is None:
            stages.append(
                StageReceipt(
                    "aclgraph",
                    GateState.FAILED,
                    "no single-NPU ACLGraph qualification command declared",
                )
            )
        else:
            stages.append(self._run_command("aclgraph", spec.aclgraph_qualification))
        return self._result("G1", stages, plan)

    # ---- public G2 / G3 --------------------------------------------------
    def evaluate_g2_sustained_abba(self, spec: G2Spec) -> GateResult:
        plan = (
            self._derived_plan(spec.stage_budgets, grace=1.0)
            if spec.stage_budgets
            else None
        )
        return _evaluate_g2_sustained_abba(spec, timeout_plan=plan)

    def evaluate_g3_subgraph_module_gate(self, spec: G3Spec) -> GateResult:
        plan = (
            self._derived_plan(spec.stage_budgets, grace=1.0)
            if spec.stage_budgets
            else None
        )
        return _evaluate_g3_subgraph_module_gate(spec, timeout_plan=plan)


def incremental_build_command(
    *,
    source_root: str,
    jobs: int,
    build_system: str = "make",
    build_target: str = "native-gate-opp",
    timeout_sec: float,
) -> GateCommand:
    """Build the one command whose env prevents inner CANN OPC serialization."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    env = {
        "MAKEFLAGS": f"-j{jobs}",
        "CMAKE_BUILD_PARALLEL_LEVEL": str(jobs),
        "OPS_CPU_NUMBER": str(jobs),
    }
    if build_system == "make":
        argv = ("make", "-C", source_root, build_target, f"-j{jobs}")
    elif build_system == "cmake":
        argv = (
            "cmake",
            "--build",
            source_root,
            "--target",
            build_target,
            "--parallel",
            str(jobs),
        )
    else:
        raise ValueError("build_system must be make or cmake")
    return GateCommand(
        argv,
        timeout_sec=timeout_sec,
        purpose=f"{build_target} incremental build",
        env=env,
    )


def _lock_number(text: str, key: str) -> int | None:
    match = re.search(rf"(?:^|\s){key}[=: ]+([1-9][0-9]*)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _resolve(base: Path, path: str) -> str:
    return str(base / path if not Path(path).is_absolute() else Path(path))


def _pid_alive(pid: int) -> bool | None:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            err = kernel32.GetLastError()
            return True if err == 5 else False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except TimeoutError:
        raise
    except OSError:
        return None
    return True
