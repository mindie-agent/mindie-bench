"""High-signal G0-G3 qualification gate contracts.

Tier 1 tests build real local artifacts and run real bounded commands where a
portable toolchain exists; Linux-only `ldd -r` is adapted from `nm -u` on macOS
but still inspects a compiled shared library.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.runner.qualification_gate import (
    AbbaSample,
    BoundaryPair,
    CommandReceipt,
    G0Spec,
    G1Spec,
    G2Spec,
    G3Spec,
    GateCommand,
    GateState,
    GuardWorkload,
    ModuleRun,
    ModuleTopology,
    NumericalQualification,
    QualificationGateRunner,
    TimeoutArithmeticError,
    frozen_baseline_digest,
    incremental_build_command,
    plan_gate_timeouts,
    run_local_command,
)

pytestmark = pytest.mark.tier1


def _aicore_and_aicpu(root: Path, name: str = "FlashMLA") -> tuple[str, str]:
    build = root / "build"
    build.mkdir(exist_ok=True)
    (build / "flashmla.o").write_bytes(b"\x7fELF native aicore kernel\n")
    registration = {
        name: {
            "opInfo": {
                "engine": "DNN_VM_AICPU",
                "kernelSo": "libtransformer_aicpu_kernels.so",
                "functionName": "RunCpuKernel",
            }
        }
    }
    (build / "flashmla_aicpu.json").write_text(json.dumps(registration))
    manifest = {
        "schema": "mindie-native-operator-manifest/1",
        "operators": [
            {"name": name, "aicore": "build/flashmla.o", "aicpu": "build/flashmla_aicpu.json"}
        ],
    }
    manifest_path = root / "operators.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(manifest_path), str(build)


def _compile_shared(path: Path, source: str, *, unresolved: bool = False) -> Path:
    source_path = path.with_suffix(".c")
    source_path.write_text(source)
    if os.name == "nt" and shutil.which("cc") is None:
        marker = b"UNRESOLVED_EXTERN_ABSENT\n" if unresolved else b"CLEAN_SHARED_LIB\n"
        path.write_bytes(b"\x7fELF" + marker)
        return path
    command = ["cc", "-shared", "-fPIC", "-o", str(path), str(source_path)]
    if unresolved and sys.platform == "darwin":
        # Darwin links shared libraries with -undefined error by default. Linux
        # has no equivalent restriction: `cc -shared -fPIC` keeps undefined
        # symbols as dynamic imports, which is exactly what `ldd -r` inspects.
        command[1:1] = ["-undefined", "dynamic_lookup"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return path


def _is_weak_or_crt_symbol(symbol: str) -> bool:
    """Weak CRT references reported by `nm -u` are not unresolved imports."""
    return (
        symbol.startswith("w ")
        or symbol.lstrip("_").startswith(
            ("ITM_registerTMCloneTable", "cxa_finalize", "gmon_start__",
             "dyld_stub_binder")
        )
    )


def _portable_symbol_executor(command: GateCommand) -> CommandReceipt:
    """Use the host's real undefined-symbol view for `ldd -r`.

    Linux has a native `ldd -r`. macOS does not, so adapt `nm -u` output there
    and drop weak/CRT stubs, which are never genuine unresolved references.
    """
    if os.name == "nt" and command.argv[:2] == ("ldd", "-r") and shutil.which("ldd") is None:
        raw = Path(command.argv[2]).read_bytes()
        stderr = "undefined symbol: absent" if b"UNRESOLVED_EXTERN_ABSENT" in raw else ""
        return CommandReceipt(command.argv, 0, "", stderr)
    if sys.platform == "darwin" and command.argv[:2] == ("ldd", "-r"):
        completed = subprocess.run(
            ["nm", "-u", command.argv[2]], capture_output=True, text=True, timeout=10
        )
        stderr = "\n".join(
            f"undefined symbol: {symbol}"
            for symbol in (line.strip() for line in completed.stdout.splitlines())
            if symbol and not _is_weak_or_crt_symbol(symbol)
        )
        return CommandReceipt(command.argv, completed.returncode, "", stderr)
    return run_local_command(command)


def test_g0_manifest_pairs_aicore_objects_with_aicpu_json_and_clean_configs(tmp_path):
    manifest, build = _aicore_and_aicpu(tmp_path)
    generated = tmp_path / "aicore_config.json"
    generated.write_text(json.dumps({"op": "FlashMLA", "tiling": {"block": 32}}))
    library = _compile_shared(tmp_path / "libkernel.dylib", "int op(void){return 1;}\n")
    runner = QualificationGateRunner(executor=_portable_symbol_executor)
    result = runner.verify_g0_artifact_and_contract(
        G0Spec(
            contract_tests=(
                GateCommand(("python3", "-c", "print('CPU_CONTRACT_OK')"),
                            timeout_sec=5, expect_stdout=("CPU_CONTRACT_OK",)),
            ),
            manifests=(manifest,),
            generated_json=(str(generated),),
            ldd_targets=(str(library),),
        )
    )
    assert result.state is GateState.COMPLETED, result.to_dict()
    assert result.timeout_plan["outer_timeout_sec"] == pytest.approx(
        result.timeout_plan["inner_stage_budget_sec"] + 5.0
    )
    assert {stage.name for stage in result.stages} == {"contract:0", "manifest", "json_config", "ldd"}
    assert result.stages[1].evidence["aicore_objects_present"] == 1
    assert result.stages[2].state is GateState.COMPLETED
    assert result.stages[3].state is GateState.COMPLETED
    assert Path(build, "flashmla.o").exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_object", "AICore .o missing"),
        ("missing_aicpu", "paired AICPU JSON is missing"),
        ("trailing_bytes", "trailing or malformed JSON"),
        ("duplicate_key", "duplicate JSON keys"),
        ("unresolved_symbol", "unresolved symbol report"),
    ],
)
def test_g0_rejects_incomplete_or_corrupted_native_packaging(tmp_path, mutation, expected):
    manifest, build = _aicore_and_aicpu(tmp_path)
    generated = tmp_path / "config.json"
    generated.write_text('{"op": "FlashMLA"}')
    library = _compile_shared(tmp_path / "libkernel.dylib", "int op(void){return 1;}\n")
    if mutation == "missing_object":
        (Path(build) / "flashmla.o").unlink()
    elif mutation == "missing_aicpu":
        payload = json.loads(Path(manifest).read_text())
        payload["operators"][0].pop("aicpu")
        Path(manifest).write_text(json.dumps(payload))
    elif mutation == "trailing_bytes":
        generated.write_text('{"op": "FlashMLA"}{"op": "stale-tail"}')
    elif mutation == "duplicate_key":
        generated.write_text('{"op": "FlashMLA", "op": "stale-tail"}')
    else:
        library = _compile_shared(
            tmp_path / "libbad.dylib", "extern int absent(int); int op(void){return absent(1);}\n",
            unresolved=True,
        )
    result = QualificationGateRunner(executor=_portable_symbol_executor).verify_g0_artifact_and_contract(
        G0Spec(
            contract_tests=(GateCommand(("true",), timeout_sec=2),),
            manifests=(manifest,),
            generated_json=(str(generated),),
            ldd_targets=(str(library),),
        )
    )
    assert result.state is GateState.FAILED
    assert any(expected in reason for reason in result.reasons), result.to_dict()
    assert all(stage.state is not GateState.COMPLETED for stage in result.stages
               if expected.split()[0] in stage.detail or expected in stage.detail)


def test_g0_missing_contract_or_timeout_stays_visible(tmp_path):
    result = QualificationGateRunner().verify_g0_artifact_and_contract(
        G0Spec(contract_tests=(), stage_budgets={"contract": 2.0, "artifacts": 1.0})
    )
    assert result.state is GateState.FAILED
    assert any("no CPU contract test" in reason for reason in result.reasons)
    timed = QualificationGateRunner().verify_g0_artifact_and_contract(
        G0Spec(
            contract_tests=(
                GateCommand(("python3", "-c", "import time; time.sleep(3)"), timeout_sec=0.2),
            ),
            stage_budgets={"contract": 1.0, "artifacts": 1.0},
        )
    )
    assert timed.state is GateState.UNKNOWN
    assert timed.stages[0].evidence["timed_out"] is True


def test_g0_without_native_artifacts_is_unknown_not_a_silent_success():
    result = QualificationGateRunner().verify_g0_artifact_and_contract(
        G0Spec(contract_tests=(GateCommand(("true",), timeout_sec=2),))
    )
    assert result.state is GateState.UNKNOWN
    assert any("no native artifact" in reason for reason in result.reasons)


def test_timeout_arithmetic_rejects_an_under_budget_outer_gate():
    plan = plan_gate_timeouts({"build": 10, "eager": 5, "aclgraph": 5},
                              cleanup_grace_sec=3, outer_timeout_sec=23)
    assert plan["inner_stage_budget_sec"] == 20
    assert plan["outer_timeout_sec"] == 23
    with pytest.raises(TimeoutArithmeticError, match="outer_timeout"):
        plan_gate_timeouts({"build": 10, "eager": 5, "aclgraph": 5},
                           cleanup_grace_sec=3, outer_timeout_sec=22.9)
    with pytest.raises(TimeoutArithmeticError):
        QualificationGateRunner().verify_g0_artifact_and_contract(
            G0Spec(
                contract_tests=(GateCommand(("true",), timeout_sec=5),),
                stage_budgets={"contract": 5.0},
                outer_timeout_sec=6.0,
            )
        )


def test_g1_propagates_all_three_parallelism_variables_and_runs_eager_graph(tmp_path):
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "native-gate-opp:\n"
        "\t@case \"$$MAKEFLAGS\" in *-j*) ;; *) echo MAKEFLAGS_MISSING; exit 9;; esac\n"
        "\t@test \"$$CMAKE_BUILD_PARALLEL_LEVEL\" = \"3\"\n"
        "\t@test \"$$OPS_CPU_NUMBER\" = \"3\"\n"
        "\t@echo G1_BUILD_OK\n"
    )
    command = incremental_build_command(
        source_root=str(tmp_path), jobs=3, timeout_sec=10
    )
    assert command.env == {
        "MAKEFLAGS": "-j3",
        "CMAKE_BUILD_PARALLEL_LEVEL": "3",
        "OPS_CPU_NUMBER": "3",
    }
    def _make_or_local_executor(cmd: GateCommand) -> CommandReceipt:
        if os.name == "nt" and cmd.argv and cmd.argv[0] == "make" and shutil.which("make") is None:
            env = {**os.environ, **cmd.env}
            if "-j" not in env.get("MAKEFLAGS", ""):
                return CommandReceipt(cmd.argv, 9, "MAKEFLAGS_MISSING\n", "")
            if env.get("CMAKE_BUILD_PARALLEL_LEVEL") != "3" or env.get("OPS_CPU_NUMBER") != "3":
                return CommandReceipt(cmd.argv, 1, "", "parallelism mismatch")
            return CommandReceipt(cmd.argv, 0, "G1_BUILD_OK\n", "")
        return run_local_command(cmd)

    result = QualificationGateRunner(executor=_make_or_local_executor).build_and_verify_g1_incremental_gate(
        G1Spec(
            source_root=str(tmp_path),
            jobs=3,
            build_sentinel=("G1_BUILD_OK",),
            eager_qualification=GateCommand(
                ("python3", "-c", "print('EAGER_SENTINEL')"), timeout_sec=5,
                expect_stdout=("EAGER_SENTINEL",),
            ),
            aclgraph_qualification=GateCommand(
                ("python3", "-c", "print('ACLGRAPH_SENTINEL')"), timeout_sec=5,
                expect_stdout=("ACLGRAPH_SENTINEL",),
            ),
        )
    )
    assert result.state is GateState.COMPLETED, result.to_dict()
    assert [stage.name for stage in result.stages] == ["locks", "build", "eager", "aclgraph"]
    assert "G1_BUILD_OK" in result.stages[1].evidence["stdout"].splitlines()


def test_g1_live_lock_blocks_build_while_stale_lock_is_left_untouched(tmp_path):
    marker = tmp_path / "built"
    lock = tmp_path / "debug.pid"
    pgid = os.getpgrp() if hasattr(os, "getpgrp") else os.getpid()
    alive = f"pid={os.getpid()} pgid={pgid}\n"
    lock.write_text(alive)
    spec = G1Spec(
        source_root=str(tmp_path),
        build_command=GateCommand(("python3", "-c", f"open({str(marker)!r}, 'w').write('x')"),
                                  timeout_sec=5),
        debug_lock_paths=(str(lock),),
        eager_qualification=GateCommand(("true",), timeout_sec=1),
        aclgraph_qualification=GateCommand(("true",), timeout_sec=1),
    )
    blocked = QualificationGateRunner().build_and_verify_g1_incremental_gate(spec)
    assert blocked.state is GateState.FAILED
    assert blocked.metrics["locked"] is True
    assert not marker.exists()
    assert lock.read_text() == alive

    lock.write_text("pid=99999999 pgid=99999999\n")
    stale = QualificationGateRunner().build_and_verify_g1_incremental_gate(spec)
    assert stale.state is GateState.COMPLETED, stale.to_dict()
    assert marker.exists()
    assert lock.exists()
    assert "stale debug lock PID 99999999" in stale.stages[0].detail


@pytest.fixture
def abba_spec_factory():
    def build(*, speedup: float = 2.0, sustained: bool = True,
              min_window: float = 0.001, numerical_ok: bool = True,
              clock: float = 1280.0):
        samples = []
        for warm in (1,):
            samples.extend([
                AbbaSample("baseline", "warmup", warm, 200, 1, clock, sustained),
                AbbaSample("candidate", "warmup", warm, 100, 1, clock, sustained),
            ])
        for round_index in (1, 2, 3):
            before, after = int(200_000_000 * speedup), 200_000_000
            samples.extend([
                AbbaSample("baseline", "measured", round_index, before, 1, clock, sustained),
                AbbaSample("candidate", "measured", round_index, after, 1, clock, sustained),
                AbbaSample("candidate", "measured", round_index, after, 1, clock, sustained),
                AbbaSample("baseline", "measured", round_index, before, 1, clock, sustained),
            ])
        spec = G2Spec(
            baseline_id="frozen-baseline-20261020",
            candidate_id="candidate-pack-fusion",
            frozen_baseline_sha256="",
            samples=tuple(samples),
            rounds=3,
            warmup_rounds=1,
            numerical=NumericalQualification(
                baseline_passed=numerical_ok, candidate_passed=numerical_ok,
                sentinel_ok=numerical_ok, max_abs_error=1e-4 if numerical_ok else 1.0,
                tolerance=1e-3,
            ),
            min_sustained_window_sec=min_window,
            sustained_clock_mhz=(1250.0, 1300.0),
        )
        digest = frozen_baseline_digest(spec)
        return G2Spec(**{**spec.__dict__, "frozen_baseline_sha256": digest})
    return build


def test_g2_accepts_only_frozen_interleaved_sustained_numerical_abba(abba_spec_factory):
    result = QualificationGateRunner().evaluate_g2_sustained_abba(abba_spec_factory())
    assert result.state is GateState.COMPLETED, result.to_dict()
    assert result.metrics["numerical_qualified"] is True
    assert result.metrics["sustained_valid"] is True
    assert result.metrics["geometric_speedup"] == pytest.approx(2.0)
    assert result.metrics["accepted"] is True

    short = abba_spec_factory(min_window=10.0)
    short_result = QualificationGateRunner().evaluate_g2_sustained_abba(short)
    assert short_result.state is GateState.FAILED
    assert any("short-burst" in reason for reason in short_result.reasons)

    failed_oracle = abba_spec_factory(numerical_ok=False)
    oracle_result = QualificationGateRunner().evaluate_g2_sustained_abba(failed_oracle)
    assert oracle_result.state is GateState.FAILED
    latency = next(stage for stage in oracle_result.stages if stage.name == "latency")
    assert latency.state is GateState.UNKNOWN
    assert oracle_result.metrics["accepted"] is False


def test_g2_rejects_a_changed_baseline_and_non_abba_round(abba_spec_factory):
    spec = abba_spec_factory()
    tampered = G2Spec(
        **{**spec.__dict__, "samples": tuple(
            AbbaSample(s.variant, s.phase, s.round_index, s.duration_ns + 1, s.aclgraph_replays,
                       s.clock_mhz, s.sustained)
            if s.variant == "baseline" and s.phase == "measured" else s
            for s in spec.samples
        )}
    )
    result = QualificationGateRunner().evaluate_g2_sustained_abba(tampered)
    assert result.state is GateState.FAILED
    assert any("frozen baseline" in reason for reason in result.reasons)

    bad_order = G2Spec(
        **{**spec.__dict__, "samples": tuple(
            AbbaSample("candidate" if s.variant == "baseline" and s.phase == "measured"
                       else "baseline" if s.variant == "candidate" and s.phase == "measured"
                       else s.variant, s.phase, s.round_index, s.duration_ns,
                       s.aclgraph_replays, s.clock_mhz, s.sustained)
            for s in spec.samples
        )}
    )
    # Re-freeze the digest so only the interleaving order can fail.
    bad_order = G2Spec(**{**bad_order.__dict__,
                          "frozen_baseline_sha256": frozen_baseline_digest(bad_order)})
    assert QualificationGateRunner().evaluate_g2_sustained_abba(bad_order).state is GateState.FAILED


def _g3_spec(*, guard_passed: bool = True, measurement_valid: bool = True,
             executed_calls: int = 3, max_ratio: float = 1.02) -> G3Spec:
    return G3Spec(
        module_id="mlapo-flashmla-subgraph",
        topology=ModuleTopology(tp=2, die_count=2, observed_ranks=(0, 1)),
        boundary_pairs=(
            BoundaryPair(
                producer="MLAPO", consumer="FlashMLA", boundary="quantized_qkv",
                logical_identity="token:role:qkv_fp8", observed_identity="token:role:qkv_fp8",
                equivalent=True, ownership_preserved=True, numerical_passed=True,
                executed_calls=executed_calls,
            ),
        ),
        paired_runs=(
            ModuleRun("baseline", "run-before", "a" * 64, True, True, 2),
            ModuleRun("candidate", "run-after", "a" * 64, True, True, 2),
        ),
        guard_workloads=(
            GuardWorkload("decode-concurrency-1", measurement_valid, guard_passed,
                          baseline_ns=1_000, candidate_ns=1_010, max_regression_ratio=max_ratio),
            GuardWorkload("prefill-long-sequence", True, True,
                          baseline_ns=2_000, candidate_ns=2_010, max_regression_ratio=1.02),
        ),
    )


def test_g3_module_passes_with_executed_boundary_and_all_guard_workloads():
    result = QualificationGateRunner().evaluate_g3_subgraph_module_gate(_g3_spec())
    assert result.state is GateState.COMPLETED, result.to_dict()
    assert [stage.name for stage in result.stages] == ["boundary", "module", "topology", "guards"]


@pytest.mark.parametrize(
    ("spec", "state", "reason"),
    [
        (_g3_spec(guard_passed=False), GateState.FAILED, "required guard failed"),
        (_g3_spec(measurement_valid=False), GateState.UNKNOWN, "guard measurement invalid"),
        (_g3_spec(executed_calls=0), GateState.FAILED, "consumer branch never executed"),
    ],
)
def test_g3_rejects_unexecuted_boundary_or_failed_guards(spec, state, reason):
    result = QualificationGateRunner().evaluate_g3_subgraph_module_gate(spec)
    assert result.state is state
    assert any(reason in item for item in result.reasons)
