"""Exercise real task exception handlers without importing torch or using an NPU."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.runner.verifier import parse_verifier

TASKS = Path(__file__).resolve().parents[1] / "tasks" / "ascend"
VARIANTS = ["sampler-unfiltered-candidates", "sampler-oversized-topk"]
ACTUAL_TOPK_ERROR = (
    "topk:../third_party/op-plugin/op_plugin/ops/opapi/TopKKernelNpuOpApi.cpp:69 "
    "NPU function error: call aclnnTopk failed, error code is 161002\n"
    "[ERROR] ERR00100 PTA call acl api failed.\n"
    "AclNN_Parameter_Error(EZ1001): Selected index k out of range "
    "(max num of self.size(2) is 8,but k is 4096)"
)


def run_exception_handler(variant, message, exception_type="RuntimeError"):
    script = (TASKS / variant / "verifier/test.sh").read_text()
    code = script.split("<<'PY_VERIFIER'\n", 1)[1].rsplit("\nPY_VERIFIER", 1)[0]
    original = next(node for node in ast.parse(code).body if isinstance(node, ast.Try))
    injected = ast.parse(f"raise {exception_type}({message!r})").body
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="json")]),
            ast.Try(body=injected, handlers=original.handlers, orelse=[], finalbody=[]),
        ],
        type_ignores=[],
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", ast.unparse(ast.fix_missing_locations(module))],
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "message",
    [
        ACTUAL_TOPK_ERROR,
        "selected index k out of range",
        "The expanded size of the tensor (128) must match the existing size (32)",
        "The size of tensor a (128) must match the size of tensor b (32)",
    ],
    ids=["observed-aclnnTopk-EZ1001", "lowercase-topk", "expanded-size", "tensor-size"],
)
def test_sampler_target_errors_are_business_failures(variant, message):
    result = run_exception_handler(variant, message)
    verdict = parse_verifier(result.returncode, result.stdout)
    assert result.returncode == 1
    assert verdict.reward == 0.0
    assert message in result.stderr


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "message",
    [
        "NPU runtime unavailable",
        "expected exactly one assigned visible NPU",
        "NPU driver initialization failed: device unavailable",
        "call aclnnTopk failed, error code is 507035: device execution failed",
    ],
    ids=["runtime-unavailable", "assignment-count", "driver-unavailable", "unrelated-aclnn-failure"],
)
def test_sampler_infrastructure_errors_never_emit_reward_zero(variant, message):
    result = run_exception_handler(variant, message)
    assert result.returncode != 0
    assert result.stdout == ""
    assert message in result.stderr
    with pytest.raises(ValueError, match="infrastructure failure"):
        parse_verifier(result.returncode, result.stdout)


@pytest.mark.parametrize("variant", VARIANTS)
def test_sampler_incorrect_values_remain_business_failures(variant):
    result = run_exception_handler(
        variant, "candidate token IDs are not a permutation", "AssertionError"
    )
    assert result.returncode == 1
    assert parse_verifier(result.returncode, result.stdout).reward == 0.0
