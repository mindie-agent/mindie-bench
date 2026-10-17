# Imports follow explicit owned source selection.
# ruff: noqa: E402
"""Grade feature attribution using complete real source and actual input values."""

import hashlib
import json
import os
import sys
from pathlib import Path

import regex
import yaml

sys.dont_write_bytecode = True
binding = json.loads((Path(__file__).parent / "runtime-binding.json").read_text())
if hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() != binding["python"]:
    raise RuntimeError("stale-task CPU interpreter identity changed")
if regex.__version__ != binding["regex"] or yaml.__version__ != binding["yaml"]:
    raise RuntimeError("stale-task dependency versions changed")
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from tests.e2e.nightly.multi_node.external_dp.scripts import utils
from tests.e2e.nightly.multi_node.external_dp.scripts.runtime import ServerCommand

TARGET = WORKSPACE / "tests/e2e/nightly/multi_node/external_dp/scripts/utils.py"
if Path(utils.__file__).resolve() != TARGET:
    raise RuntimeError("reporting module was loaded outside the owned source snapshot")


def command(config=None, legacy=None, flag="--additional-config", other_env=None, extra=None):
    args = ["vllm", "serve", "fixture-model", *(extra or [])]
    if config is not None:
        args += [flag, json.dumps(config)]
    environment = dict(other_env or {})
    if legacy is not None:
        environment["VLLM_ASCEND_ENABLE_FUSED_MC2"] = legacy
    return ServerCommand(cmd=args, env=environment, display_cmd="unexecuted local fixture")


def check():
    count = 0
    for flag in ["--additional-config", "--additional_config"]:
        for enabled in [0, 1, False, True]:
            for legacy in ["0", "1", None]:
                actual = utils._extract_features(
                    [command({"enable_fused_mc2": enabled}, legacy, flag=flag)]
                )
                assert ("fused_mc2" in actual) == bool(enabled), (
                    "feature attribution disagrees with the current explicit configuration"
                )
                assert len(actual) == len(set(actual)), "feature tags must not be duplicated"
                count += 1
    for legacy in ["0", "1", None]:
        assert "fused_mc2" not in utils._extract_features([command(legacy=legacy)]), (
            "a removed environment variable is not evidence of an active current feature"
        )
        count += 1
    preserved = utils._extract_features(
        [
            command(
                {"cudagraph_mode": "FULL_DECODE_ONLY", "enable_fused_mc2": 1},
                legacy="0",
                other_env={"VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "1", "VLLM_ASCEND_ENABLE_MLAPO": "1"},
                extra=["--async-scheduling", "--enable-expert-parallel", "--speculative-config", "{}"],
            )
        ]
    )
    assert set(preserved) == {
        "aclgraph",
        "async_scheduling",
        "expert_parallel",
        "speculative",
        "topk_optimize",
        "mlapo",
        "fused_mc2",
    }, "unrelated feature attribution regressed"
    multiple = utils._extract_features(
        [command({"enable_fused_mc2": 1}), command({"enable_fused_mc2": 1}, flag="--additional_config")]
    )
    assert multiple.count("fused_mc2") == 1
    assert utils._extract_features([]) == []
    for flag in ["--additional-config", "--additional_config"]:
        invalid = ServerCommand(
            cmd=["vllm", "serve", "fixture-model", flag, "{invalid-json"],
            env={},
            display_cmd="unexecuted malformed fixture",
        )
        try:
            utils._extract_features([invalid])
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("malformed declared configuration must remain visible")
    return count + 5


try:
    cases = check()
except AssertionError as exc:
    print("Current-version reporting contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"cpu_contract_cases": cases}))
print(json.dumps({"reward": 1.0}))
