from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.evaluation.oracle_freezer import (
    OracleTampered,
    check_oracle,
    freeze_oracle,
)
from mindie_bench.models import OracleContract, OracleProbe, UnknownExternalEffects
from mindie_bench.runner.verifier import local_verifier


@pytest.fixture
def judge(tmp_path):
    root = tmp_path / "task"
    (root / "verifier").mkdir(parents=True)
    (root / "oracle").mkdir()
    (root / "verifier/test.sh").write_text("echo '{\"reward\":1}'\n")
    (root / "verifier/tests.py").write_text("EXPECTED = 3\n")
    (root / "oracle/golden.py").write_text("ANSWER = 3\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "trial"
    output.mkdir()
    profile_root = tmp_path / "profile"
    (profile_root / "tmp").mkdir(parents=True)
    profile = SimpleNamespace(root=profile_root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return root, workspace, profile, output


def verify(judge, **kwargs):
    return local_verifier(
        *judge, timeout=5, read_paths=[Path(sys.base_prefix), Path(sys.prefix)], **kwargs
    )


@pytest.mark.parametrize("relative", ["verifier/test.sh", "verifier/tests.py", "oracle/golden.py"])
def test_real_judge_test_and_golden_mutations_cannot_earn_reward(judge, relative):
    root, *_ = judge
    frozen = freeze_oracle(root)
    (root / relative).write_text("tampered\n")
    verdict, record = verify(judge, oracle_manifest=frozen)
    assert verdict.reward == 0
    assert verdict.oracle_status == "oracle_tampered"
    assert relative in verdict.failure_reason
    assert verdict.warnings
    assert record["oracle"]["status"] == "oracle_tampered"
    assert not (judge[3] / "verifier.stdout").exists()


@pytest.mark.parametrize("action", ["add", "remove", "symlink"])
def test_frozen_inventory_rejects_changed_membership(judge, action):
    root, *_ = judge
    frozen = freeze_oracle(root)
    target = root / "verifier/tests.py"
    if action == "add":
        (root / "verifier/new_test.py").write_text("assert False")
    else:
        target.unlink()
        if action == "symlink":
            target.symlink_to(root / "oracle/golden.py")
    with pytest.raises(OracleTampered):
        check_oracle(frozen)


def test_foreground_cannot_write_external_judge_in_actual_sandbox(judge):
    root, workspace, *_ = judge
    (root / "verifier/test.sh").write_text(
        f"if echo replaced > {workspace}/module.py; then exit 7; fi\n"
        f"if echo replaced > {root}/oracle/golden.py; then exit 8; fi\n"
        "echo '{\"reward\":1}'\n"
    )
    frozen = freeze_oracle(root)
    verdict, _ = verify(judge, oracle_manifest=frozen)
    assert verdict.reward == 1
    assert verdict.oracle_status == "intact"
    assert not (workspace / "module.py").exists()
    assert check_oracle(frozen)["status"] == "intact"


def test_actual_in_memory_monkeypatch_is_detected_by_bound_negative_control(judge):
    root, workspace, *_ = judge
    probe = root / "verifier/negative.py"
    probe.write_text(
        "import importlib.util, os\n"
        "spec=importlib.util.spec_from_file_location('candidate', "
        "os.environ['MINDIE_BENCH_WORKSPACE']+'/candidate.py')\n"
        "mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "mod.setup()\n"
        "assert sum([1,2]) == 3, 'controlled judge monkeypatch detected'\n"
        "print('control-ok')\n"
    )
    contract = OracleContract(
        probes=[
            OracleProbe(
                name="known_sum_control",
                command=[sys.executable, "-B", "-I", str(probe)],
                expected_returncode=0,
                stdout_sha256=hashlib.sha256(b"control-ok\n").hexdigest(),
            )
        ]
    )
    frozen = freeze_oracle(root, contract)
    (workspace / "candidate.py").write_text("def setup(): pass\n")
    clean, _ = verify(judge, oracle_manifest=frozen, oracle_contract=contract)
    assert clean.reward == 1
    (workspace / "candidate.py").write_text(
        "def setup():\n import builtins\n builtins.sum=lambda values: 0\n"
    )
    tampered, record = verify(judge, oracle_manifest=frozen, oracle_contract=contract)
    assert tampered.reward == 0 and tampered.oracle_status == "oracle_tampered"
    assert "known_sum_control" in tampered.failure_reason
    assert "controlled judge monkeypatch" in (judge[3] / "oracle_probe_0.stderr").read_text()
    assert record["oracle"]["probes"][0]["returncode"] != 0
    assert check_oracle(frozen, contract)["status"] == "intact"  # No file attack occurred.


def test_post_execution_tamper_overrides_forged_pass_and_preserves_raw_stdout(judge, monkeypatch):
    from mindie_bench.runner import verifier

    root, *_ = judge
    frozen = freeze_oracle(root)
    actual_check = verifier.check_oracle
    checks = 0

    def mutate_before_post_execution_check(*args, **kwargs):
        nonlocal checks
        checks += 1
        if checks == 2:
            (root / "oracle/golden.py").write_text("ANSWER = 'forged'\n")
        return actual_check(*args, **kwargs)

    monkeypatch.setattr(verifier, "check_oracle", mutate_before_post_execution_check)
    verdict, record = verify(judge, oracle_manifest=frozen)
    assert verdict.reward == 0 and verdict.oracle_status == "oracle_tampered"
    assert json.loads((judge[3] / "verifier.stdout").read_text())["reward"] == 1
    assert record["returncode"] == 0


def test_legacy_nonperformance_judge_still_runs(judge):
    verdict, record = verify(judge)
    assert verdict.reward == 1 and record["returncode"] == 0


@pytest.mark.parametrize("ignore_term, inventory_failure", [(False, False), (True, False), (True, True)])
def test_local_timeout_stops_owned_descendant_and_keeps_partial_output(
    judge, monkeypatch, ignore_term, inventory_failure,
):
    from mindie_bench.runner import verifier

    root, _workspace, profile, trial = judge
    sentinel = profile.root / "tmp" / "late-write"
    release = profile.root / "tmp" / "drain-returned"
    child = (
        "import os,pathlib,signal,time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + "print('child='+str(os.getpid()), flush=True)\n"
        + "print('judge-started', file=__import__('sys').stderr, flush=True)\n"
        + f"release = pathlib.Path({str(release)!r})\n"
        + "deadline = time.monotonic() + 6\n"
        + "while not release.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        + f"if release.exists(): pathlib.Path({str(sentinel)!r}).write_text('late mutation')\n"
    )
    (root / "verifier/test.sh").write_text(
        f"{shlex.quote(sys.executable)} -I -c {shlex.quote(child)} &\nwait\n"
    )
    actual_popen = subprocess.Popen
    owners = []

    def observe_owner(command, *args, **kwargs):
        process = actual_popen(command, *args, **kwargs)
        if str(root / "verifier/test.sh") in command:
            owners.append(process)
        return process

    monkeypatch.setattr(verifier.subprocess, "Popen", observe_owner)
    if inventory_failure:
        actual_run = subprocess.run

        def fail_inventory(command, **kwargs):
            if command == ["ps", "-e", "-o", "pgid=,stat="]:
                raise OSError("process inventory unavailable")
            return actual_run(command, **kwargs)

        monkeypatch.setattr(verifier.subprocess, "run", fail_inventory)
    late_write = False
    try:
        with pytest.raises((subprocess.TimeoutExpired, UnknownExternalEffects)) as raised:
            local_verifier(
                *judge, timeout=0.7, read_paths=[Path(sys.base_prefix), Path(sys.prefix)]
            )
        timeout = raised.value.__cause__ if inventory_failure else raised.value
        if timeout is None:
            timeout = raised.value
        expected = UnknownExternalEffects if inventory_failure else subprocess.TimeoutExpired
        assert isinstance(raised.value, expected)
        assert len(owners) == 1 and owners[0].poll() is not None
        assert b"child=" in timeout.stdout
        assert (trial / "verifier.stdout").read_bytes() == timeout.stdout
        assert (trial / "verifier.stderr").read_bytes() == timeout.stderr
        assert b"judge-started" in timeout.stderr
    finally:
        # Only a descendant surviving the completed drain can observe this
        # marker. A fixed sleep can instead fire inside its lawful cleanup grace.
        release.write_text("drain returned")
        # A bubblewrap child's printed PID belongs to its PID namespace.
        # Only the observed host Popen is a valid signal target for this test.
        try:
            for process in owners:
                if process.poll() is not None:
                    continue
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        finally:
            # Give a surviving child time to observe the marker even if an earlier
            # assertion fails. An old implementation then writes and exits;
            # cleanup never guesses a host PID from namespace-local output.
            if owners:
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    late_write = late_write or sentinel.exists()
                    time.sleep(0.05)
                late_write = late_write or sentinel.exists()
    assert not late_write, "timed-out judge child can still mutate state"


def test_probe_hash_uses_actual_bytes_without_newline_normalization(judge):
    raw = b"control\r\n"
    contract = OracleContract(
        probes=[
            OracleProbe(
                name="raw_bytes",
                command=["/usr/bin/printf", raw.decode()],
                expected_returncode=0,
                stdout_sha256=hashlib.sha256(raw).hexdigest(),
            )
        ]
    )
    verdict, record = verify(judge, oracle_contract=contract)
    assert verdict.reward == 1
    assert (judge[3] / "oracle_probe_0.stdout").read_bytes() == raw
    assert record["oracle"]["probes"][0]["stdout_sha256"] == hashlib.sha256(raw).hexdigest()


def _cpu_torch_runtime() -> tuple[Path, Path]:
    explicit = os.environ.get("MINDIE_BENCH_CPU_PYTHON")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path("/tmp/mindie-bench-cpu-l2-runtime-20261017/venv/bin/python"))
    for interpreter in candidates:
        if interpreter.is_file():
            return interpreter.parent.parent, interpreter
    pytest.skip("actual CPU torch control requires MINDIE_BENCH_CPU_PYTHON or CPU torch runtime")


def test_actual_torch_comparator_monkeypatch_cannot_turn_negative_control_into_pass(judge):
    cpu_runtime, cpu_python = _cpu_torch_runtime()
    root, workspace, profile, trial = judge
    (root / "verifier/torch_control.py").write_text(
        "import importlib.util,os,pathlib\n"
        "import torch\n"
        "spec=importlib.util.spec_from_file_location('candidate', "
        "pathlib.Path(os.environ['MINDIE_BENCH_WORKSPACE'])/'candidate.py')\n"
        "mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n"
        "mod.setup()\n"
        "try:\n"
        " torch.testing.assert_close(torch.tensor([1.0]),torch.tensor([7.0]))\n"
        "except AssertionError:\n"
        " print('torch-negative-control-ok')\n"
        "else:\n"
        " raise RuntimeError('torch.testing.assert_close was bypassed')\n"
    )
    contract = OracleContract(
        probes=[
            OracleProbe(
                name="actual-torch-assert-close",
                command=[
                    str(cpu_python),
                    "-B",
                    "-I",
                    "{oracle}/verifier/torch_control.py",
                ],
                expected_returncode=0,
                stdout_sha256=hashlib.sha256(b"torch-negative-control-ok\n").hexdigest(),
            )
        ]
    )
    frozen = freeze_oracle(root, contract)
    (workspace / "candidate.py").write_text("def setup(): pass\n")
    options = dict(
        timeout=10,
        read_paths=[cpu_runtime, Path(sys.base_prefix)],
        oracle_manifest=frozen,
        oracle_contract=contract,
    )
    clean, _ = local_verifier(root, workspace, profile, trial, **options)
    assert clean.reward == 1
    (workspace / "candidate.py").write_text(
        "def setup():\n import torch\n torch.testing.assert_close=lambda *args, **kwargs: None\n"
    )
    tampered, record = local_verifier(root, workspace, profile, trial, **options)
    assert tampered.reward == 0 and tampered.oracle_status == "oracle_tampered"
    assert record["oracle"]["probes"][0]["returncode"] != 0
    assert "torch.testing.assert_close was bypassed" in (trial / "oracle_probe_0.stderr").read_text()


@pytest.mark.parametrize("patch_when", ["import", "execute"])
def test_actual_judge_rejects_torch_patch_that_separate_probe_does_not_observe(judge, patch_when):
    cpu_runtime, cpu_python = _cpu_torch_runtime()
    root, workspace, profile, trial = judge
    control = (
        Path(__file__).resolve().parents[1]
        / "tasks/performance/fused-rmsnorm-gate/verifier/comparator_control.py"
    )
    (root / "verifier/comparator_control.py").write_bytes(control.read_bytes())
    import_candidate = (
        "import importlib.util,os,pathlib,sys\n"
        "sys.path.insert(0,str(pathlib.Path(__file__).parent))\n"
        "import torch\n"
        "from comparator_control import check_comparators,enforce_comparator_control\n"
        "spec=importlib.util.spec_from_file_location('candidate', "
        "pathlib.Path(os.environ['MINDIE_BENCH_WORKSPACE'])/'candidate.py')\n"
        "candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)\n"
    )
    (root / "verifier/separate_probe.py").write_text(
        import_candidate + "check_comparators(torch)\nprint('separate-probe-ok')\n"
    )
    (root / "verifier/actual_judge.py").write_text(
        import_candidate + "enforce_comparator_control(torch)\n"
        "candidate.run()\n"
        "enforce_comparator_control(torch)\n"
        "torch.testing.assert_close(torch.tensor([1.0]),torch.tensor([1.0]))\n"
        "print('{\"reward\":1}')\n"
    )
    (root / "verifier/test.sh").write_text(
        f'exec "{cpu_python}" -B -I '
        '"${MINDIE_BENCH_ORACLE_ROOT}/verifier/actual_judge.py"\n'
    )
    contract = OracleContract(
        probes=[
            OracleProbe(
                name="separate-process-comparator",
                command=[
                    str(cpu_python),
                    "-B",
                    "-I",
                    "{oracle}/verifier/separate_probe.py",
                ],
                expected_returncode=0,
                stdout_sha256=hashlib.sha256(b"separate-probe-ok\n").hexdigest(),
            )
        ]
    )
    frozen = freeze_oracle(root, contract)
    options = dict(
        timeout=10,
        read_paths=[cpu_runtime, Path(sys.base_prefix)],
        oracle_manifest=frozen,
        oracle_contract=contract,
    )
    (workspace / "candidate.py").write_text("def run(): pass\n")
    clean, _ = local_verifier(root, workspace, profile, trial, **options)
    assert clean.reward == 1
    patch = "torch.testing.assert_close=lambda *args,**kwargs: None"
    (workspace / "candidate.py").write_text(
        "import pathlib,sys,torch\n"
        "IN_JUDGE=pathlib.Path(sys.argv[0]).name=='actual_judge.py'\n"
        + (f"if IN_JUDGE: {patch}\n" if patch_when == "import" else "")
        + "def run():\n"
        + (f" if IN_JUDGE: {patch}\n" if patch_when == "execute" else " pass\n")
    )
    tampered, record = local_verifier(root, workspace, profile, trial, **options)
    assert record["oracle"]["probes"][0]["returncode"] == 0
    assert tampered.reward == 0 and tampered.oracle_status == "oracle_tampered"
    assert "actual judge comparator control" in tampered.failure_reason
    assert "arbitrary" in tampered.warnings[0]
    assert json.loads((trial / "verifier.stdout").read_text())["oracle_status"] == "oracle_tampered"
    assert check_oracle(frozen, contract)["status"] == "intact"


def test_ephemeral_cache_files_do_not_invalidate_tree_hash_or_frozen_oracle(judge):
    from mindie_bench.io import tree_hash

    root, *_ = judge
    baseline_digest = tree_hash(root)
    frozen = freeze_oracle(root)
    (root / "verifier/__pycache__").mkdir(parents=True)
    (root / "verifier/__pycache__/tests.cpython-311.pyc").write_bytes(b"bytecode")
    (root / "oracle/.pytest_cache").mkdir(parents=True)
    (root / "oracle/.pytest_cache/README.md").write_text("pytest cache\n")
    assert tree_hash(root) == baseline_digest
    assert check_oracle(frozen)["status"] == "intact"
