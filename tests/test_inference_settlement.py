import threading
import time
from pathlib import Path

from mindie_bench.models import Group
from mindie_bench.runner.gateway import Gateway
from mindie_bench.runner.orchestrator import run_trial
from mindie_bench.runner.process import ProcessResult
from mindie_bench.runner.usage import UsageMeter


def gateway_for(case):
    return Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "settlement.jsonl",
    )


def test_final_provider_receipt_can_settle_before_classification(case):
    gateway = gateway_for(case)
    gateway.active = 1

    def finish():
        time.sleep(0.01)
        with gateway.model_idle:
            gateway.active = 0
            gateway.model_idle.notify_all()

    worker = threading.Thread(target=finish)
    worker.start()
    gateway.finish_inference(timeout=1)
    worker.join()
    assert gateway.models_sealed
    assert not gateway.unknown


def test_unsettled_provider_call_stays_unknown_after_late_completion(case):
    gateway = gateway_for(case)
    gateway.active = 1
    gateway.finish_inference(timeout=0.01)
    assert gateway.unknown
    assert gateway.models_sealed
    gateway.active = 0
    gateway.finish_inference()
    assert gateway.unknown
    assert "model_calls_unsettled" in gateway.journal.read_text()


def test_budget_timeout_with_active_inference_is_not_completed_fail(case, monkeypatch):
    def timed_out_process(*args, gateway, **kwargs):
        gateway.model_calls = gateway.active = 1
        gateway.usage_meters[1] = UsageMeter()
        return ProcessResult(started=True, elapsed=0.1, stop_reason="timeout")

    original = Gateway.finish_inference

    def fast_settlement(self, timeout=1):
        original(self, timeout=0.01)
        # Simulate the late provider handler leaving after classification was
        # sealed. It cannot retroactively turn the unknown attempt into a fail.
        self.active = 0

    monkeypatch.setattr("mindie_bench.runner.orchestrator.stream_process", timed_out_process)
    monkeypatch.setattr(
        "mindie_bench.runner.orchestrator.native_command", lambda command, **kwargs: command
    )
    monkeypatch.setattr(Gateway, "finish_inference", fast_settlement)
    output = case["root"] / "run"
    output.mkdir()
    result = run_trial(
        case["task"],
        case["config"],
        tier="l2",
        group=Group.G0,
        repetition=1,
        run_root=output,
    )
    assert result.terminal_status.value == "unknown", result.model_dump_json(indent=2)
    assert result.primary_metrics["reward"] is None
    assert not result.primary_metrics["tokens"]["complete"]
    assert not (Path(result.artifacts["private_trial_dir"]) / "verifier.stdout").exists()
