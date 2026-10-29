"""The production runner must retain real behavioral evidence and reject CPU performance claims."""
import json
from pathlib import Path

import pytest

from mindie_bench.runner.verifier import parse_verifier


def test_recorded_subprocess_verdicts_survive_production_parser():
    tasks = Path(__file__).resolve().parents[1] / "tasks/zh-vague"
    for path in sorted(tasks.glob("*/soundness_receipt.json")):
        receipt = json.loads(path.read_text())
        for name, case in receipt.get("cases", receipt.get("controls", {})).items():
            payload = case["verdict"]
            parsed = parse_verifier(case["exit_code"], json.dumps(payload))
            assert parsed.reward == case["reward"], (path.parent.name, name)
            for field in ("correctness_verified", "ablation_contract_verified",
                          "native_kernel_bound", "measured_npu_speedup", "validation_scope"):
                assert getattr(parsed, field) == payload.get(field), (path.parent.name, name, field)


def test_cpu_fixture_success_cannot_be_published_as_hardware_speedup():
    path = (Path(__file__).resolve().parents[1] / "tasks/zh-vague/"
            "dsv4-sparse-mla-qli-fusion-rsi-zh/soundness_receipt.json")
    payload = json.loads(path.read_text())["cases"]["gold"]["verdict"]
    assert parse_verifier(0, json.dumps(payload)).reward == 1
    for claim in ({"speedup_ratio": 2.0}, {"measured_npu_speedup": 2.0},
                  {"native_kernel_bound": True}):
        with pytest.raises(ValueError, match="CPU behavior/fixture"):
            parse_verifier(0, json.dumps(payload | claim))
