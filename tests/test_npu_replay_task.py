import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.orchestrator import _install_shims
from mindie_bench.runner.verifier import parse_verifier

TEMPLATE = Path(__file__).resolve().parents[1] / "tasks/replay/sampler-topk"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def tensor_case(tmp_path):
    selected = os.environ.get("MINDIE_BENCH_CPU_PYTHON")
    if not selected:
        pytest.skip("set MINDIE_BENCH_CPU_PYTHON to the separately locked torch CPU interpreter")
    python = Path(selected).absolute()
    workspace = tmp_path / "workspace"
    target = workspace / "vllm_ascend/sample/sampler.py"
    target.parent.mkdir(parents=True)
    target.write_bytes((TEMPLATE / "source_snapshot/sampler.py.txt").read_bytes())
    verifier = tmp_path / "private-judge"
    shutil.copytree(TEMPLATE / "verifier", verifier)
    shutil.copy2(TEMPLATE / "environment/replay_cpu_boundary.py", verifier / "replay_cpu_boundary.py")
    (verifier / "runtime-binding.json").write_text(
        json.dumps({"python": digest(python.resolve().read_bytes()), "torch": "2.10.0"})
    )
    return python, workspace, target, verifier


def grade(case):
    python, workspace, _, verifier = case
    return subprocess.run(
        [str(python), "-I", str(verifier / "verify.py")],
        cwd=workspace,
        env={**os.environ, "MINDIE_BENCH_WORKSPACE": str(workspace)},
        capture_output=True,
        text=True,
        timeout=40,
    )


def test_historical_source_and_actual_npu_diagnostic_bytes_match_manifest():
    manifest = json.loads((TEMPLATE / "source-manifest.json").read_text())
    source = (TEMPLATE / "source_snapshot/sampler.py.txt").read_bytes()
    stderr = (TEMPLATE / "environment/history-error.stderr").read_bytes()
    assert digest(source) == manifest["source_sha256"]
    assert digest(stderr) == manifest["history_stderr_sha256"]
    assert b"call aclnnTopk failed, error code is 161002" in stderr
    assert b"EZ1001" in stderr and b"8,but k is 4096" in stderr
    assert manifest["hardware_source_reward"] == 0 and manifest["hardware_gold_reward"] == 1


@pytest.mark.parametrize("variant", ["base", "gold", "diagnostic_bypass", "public_helper_bypass"])
def test_complete_module_semantic_judge_is_independent_of_replay_and_public_helper(tensor_case, variant):
    _, workspace, target, _ = tensor_case
    if variant == "gold":
        subprocess.run(
            ["git", "apply", "--no-index", str(TEMPLATE / "oracle/gold_patch.diff")],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
    elif variant == "diagnostic_bypass":
        target.write_text(
            target.read_text().replace(
                "torch.topk(logits, k=top_k, dim=-1)", "torch.topk(logits, k=(top_k), dim=-1)"
            )
        )
    elif variant == "public_helper_bypass":
        (workspace / "replay_cpu_boundary.py").write_text(
            'raise RuntimeError("an edited public helper must never be imported by the judge")\n'
        )
        (workspace / "reproduce.py").write_text("print('{\"reward\":1.0}')\n")
    result = grade(tensor_case)
    assert parse_verifier(result.returncode, result.stdout).reward == (1 if variant == "gold" else 0)
    if variant == "gold":
        assert json.loads(result.stdout.splitlines()[-2])["cpu_contract_cases"] == 37
    else:
        assert "Historical top-k shape contract failed" in result.stderr


def test_runtime_failure_does_not_receive_a_business_zero(tensor_case):
    _, _, _, verifier = tensor_case
    (verifier / "runtime-binding.json").write_text(json.dumps({"python": "0" * 64, "torch": "2.10.0"}))
    result = grade(tensor_case)
    assert result.returncode != 0
    assert "CPU interpreter identity changed" in result.stderr
    with pytest.raises(ValueError, match="infrastructure"):
        parse_verifier(result.returncode, result.stdout)


def test_actual_installed_replay_shim_returns_saved_error_then_exposes_its_pattern_limit(case, tmp_path):
    target = case["repo"] / "vllm_ascend/sample/sampler.py"
    target.parent.mkdir(parents=True)
    target.write_bytes((TEMPLATE / "source_snapshot/sampler.py.txt").read_bytes())
    (case["task"] / "oracle/replay_rules.json").write_bytes(
        (TEMPLATE / "oracle/replay_rules.json").read_bytes()
    )
    case["spec"].environment["l2"].replay_commands = ["npu-replay-sampler"]
    profile = SimpleNamespace(root=tmp_path / "profile", env={**os.environ})
    profile.root.mkdir()
    _install_shims(profile, ["npu-replay-sampler"], remote=False)
    with Gateway(
        case["config"].harness, case["spec"], case["task"], case["repo"], tmp_path / "journal"
    ) as gateway:
        environment = {**profile.env, "MINDIE_BENCH_GATEWAY": gateway.url}

        def replay():
            return subprocess.run(
                ["npu-replay-sampler"],
                cwd=case["repo"],
                env=environment,
                capture_output=True,
                timeout=10,
            )

        original = replay()
        assert original.returncode == 1
        assert original.stderr == (TEMPLATE / "environment/history-error.stderr").read_bytes()
        target.write_text(
            target.read_text().replace(
                "torch.topk(logits, k=top_k, dim=-1)", "torch.topk(logits, k=(top_k), dim=-1)"
            )
        )
        apparent_pass = replay()
        assert apparent_pass.returncode == 0
        assert b"not a tensor verdict" in apparent_pass.stdout
        assert gateway.model_calls == 0 and gateway.replay_calls == 2
