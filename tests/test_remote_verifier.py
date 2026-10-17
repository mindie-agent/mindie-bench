from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys

import pytest

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.models import OracleContract, OracleProbe, UnknownExternalEffects
from mindie_bench.runner.verifier import remote_verifier, validate_remote_oracle_contract


class LocalJudgeTransport:
    """Exercise the remote API using a real local shell, never SSH or NPU."""

    timeout = 2
    operation_unknown = False

    def __init__(self, after=None):
        self.calls = []
        self.after = after

    def verify(self, script, *, env, timeout):
        self.calls.append((script, env, timeout))
        result = subprocess.run(
            ["/bin/bash", "-s"],
            input=script,
            env=dict(
                os.environ, TMPDIR=str(self.tmp), PATH=str(self.tmp / "bin") + ":/usr/bin:/bin", **env
            ),
            capture_output=True,
            timeout=timeout + 1,
            cwd=self.workspace,
        )
        if self.after:
            self.after()
        return {
            "exit_code": result.returncode,
            "stdout_b64": base64.b64encode(result.stdout).decode(),
            "stderr_b64": base64.b64encode(result.stderr).decode(),
        }


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    (root / "verifier").mkdir(parents=True)
    (root / "oracle").mkdir()
    (root / "verifier/test.sh").write_text("printf '{\"reward\":1}\\r\\n'\n")
    (root / "oracle/golden.py").write_text("ANSWER = 3\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    LocalJudgeTransport.tmp = tmp_path / "private-tmp"
    LocalJudgeTransport.tmp.mkdir()
    (LocalJudgeTransport.tmp / "bin").mkdir()
    (LocalJudgeTransport.tmp / "bin/python3").symlink_to(sys.executable)
    LocalJudgeTransport.workspace = tmp_path / "workspace"
    LocalJudgeTransport.workspace.mkdir()
    return root, trial


def test_remote_judge_binds_nonce_and_oracle_and_preserves_actual_bytes(task):
    root, trial = task
    remote, contract = LocalJudgeTransport(), OracleContract()
    frozen = freeze_oracle(root, contract)
    value, record = remote_verifier(root, remote, trial, 2, frozen, contract, None)
    assert value.reward == 1 and value.oracle_status == "intact"
    assert record["oracle"]["before"] == record["oracle"]["after"]
    assert remote.calls[0][1]["MINDIE_BENCH_ORACLE_SHA256"] == frozen.sha256
    assert len(remote.calls[0][1]["MINDIE_BENCH_PERFORMANCE_NONCE"]) == 64
    assert (trial / "verifier.stdout").read_bytes() == b'{"reward":1}\r\n'


@pytest.mark.parametrize("when", ["before", "after"])
def test_remote_judge_cannot_pass_when_local_oracle_changes(task, when):
    root, trial = task
    contract = OracleContract()
    frozen = freeze_oracle(root, contract)

    def mutate():
        (root / "oracle/golden.py").write_text("forged")

    remote = LocalJudgeTransport(after=mutate if when == "after" else None)
    if when == "before":
        mutate()
    value, record = remote_verifier(root, remote, trial, 2, frozen, contract, None)
    assert value.reward == 0 and value.oracle_status == "oracle_tampered"
    assert len(remote.calls) == (1 if when == "after" else 0)
    if when == "after":
        assert record["original_result"]["reward"] == 1
        assert record["returncode"] == 0


def test_private_remote_probe_runs_actual_bound_files_and_cleans_bundle(task):
    root, trial = task
    raw = b"known-control\r\n"
    (root / "verifier/control.py").write_text(
        "import os,pathlib,sys\n"
        "root=pathlib.Path(os.environ['MINDIE_BENCH_ORACLE_ROOT'])\n"
        "assert (root/'oracle/golden.py').read_text() == 'ANSWER = 3\\n'\n"
        "sys.stdout.buffer.write(b'known-control\\r\\n')\n"
    )
    contract = OracleContract(
        probes=[
            OracleProbe(
                name="private-file-control",
                command=[sys.executable, "-B", "-I", "{oracle}/verifier/control.py"],
                expected_returncode=0,
                stdout_sha256=hashlib.sha256(raw).hexdigest(),
            )
        ]
    )
    remote = LocalJudgeTransport()
    validate_remote_oracle_contract(contract)
    value, record = remote_verifier(
        root, remote, trial, 2, freeze_oracle(root, contract), contract, None
    )
    assert value.reward == 1
    assert record["oracle"]["probes"][0]["stdout_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (trial / "oracle_probe_0.stdout").read_bytes() == raw
    assert not list(remote.tmp.glob("mindie-oracle.*"))


def test_remote_unknown_is_not_overwritten_by_simultaneous_oracle_failure(task):
    root, trial = task
    contract = OracleContract()
    frozen = freeze_oracle(root, contract)

    class UnknownTransport(LocalJudgeTransport):
        def verify(self, script, *, env, timeout):
            self.operation_unknown = True
            (root / "oracle/golden.py").write_text("forged")
            raise TimeoutError("original missing acknowledgement")

    with pytest.raises(UnknownExternalEffects, match="original missing acknowledgement") as caught:
        remote_verifier(root, UnknownTransport(), trial, 2, frozen, contract, None)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert "oracle_tampered" in caught.value.__notes__[0]


def test_remote_limits_this_judge_request_without_changing_agent_budget(task):
    root, trial = task
    remote, contract = LocalJudgeTransport(), OracleContract()
    remote.timeout = 1200
    value, _ = remote_verifier(root, remote, trial, 2, freeze_oracle(root, contract), contract, None)
    assert value.reward == 1
    assert remote.calls[0][2] == 2
    assert remote.timeout == 1200


def test_private_bundle_actual_tamper_is_rejected_and_original_judge_output_survives(task):
    root, trial = task
    code = (
        "import os,pathlib;"
        "p=pathlib.Path(os.environ['MINDIE_BENCH_ORACLE_ROOT'])/'oracle/golden.py';"
        "p.chmod(0o644);p.write_text('forged');print('{\"reward\":1}')"
    )
    import shlex

    (root / "verifier/test.sh").write_text(shlex.join([sys.executable, "-I", "-c", code]))
    contract, remote = OracleContract(), LocalJudgeTransport()
    value, record = remote_verifier(root, remote, trial, 2, freeze_oracle(root), contract, None)
    assert value.reward == 0 and value.oracle_status == "oracle_tampered"
    assert "oracle/golden.py" in value.failure_reason
    assert (trial / "verifier.stdout").read_bytes() == b'{"reward":1}\n'
    assert record["oracle"]["private_bundle"]["after"] is None
    assert not list(remote.tmp.glob("mindie-oracle.*"))


def test_remote_negative_control_detects_actual_candidate_monkeypatch_in_own_process(task):
    root, trial = task
    (root / "verifier/control.py").write_text(
        "import importlib.util,pathlib\n"
        "spec=importlib.util.spec_from_file_location('candidate', pathlib.Path.cwd()/'candidate.py')\n"
        "mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n"
        "mod.setup()\n"
        "assert sum([1,2])==3, 'actual monkeypatch detected'\n"
        "print('control-ok')\n"
    )
    contract = OracleContract(
        probes=[
            OracleProbe(
                name="sum-negative-control",
                command=[sys.executable, "-I", "{oracle}/verifier/control.py"],
                expected_returncode=0,
                stdout_sha256=hashlib.sha256(b"control-ok\n").hexdigest(),
            )
        ]
    )
    remote = LocalJudgeTransport()
    (remote.workspace / "candidate.py").write_text(
        "def setup():\n import builtins\n builtins.sum=lambda values: 0\n"
    )
    value, record = remote_verifier(
        root, remote, trial, 2, freeze_oracle(root, contract), contract, None
    )
    assert value.reward == 0 and value.oracle_status == "oracle_tampered"
    assert record["oracle"]["probes"][0]["returncode"] == 1
    assert "actual monkeypatch detected" in (trial / "oracle_probe_0.stderr").read_text()
    assert not list(remote.workspace.rglob("golden.py"))
    assert not list(remote.tmp.glob("mindie-oracle.*"))
