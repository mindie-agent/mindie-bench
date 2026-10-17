from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from mindie_bench.evaluation.performance import (
    measure_rotating_buffers,
    measure_warm_server,
    measure_workspace,
    probe_aclgraph,
    validate_performance,
)
from mindie_bench.models import PerformanceContract, TaskSpec, VerifierResult
from mindie_bench.runner.verifier import parse_verifier

SHAPES = [hashlib.sha256(value).hexdigest() for value in [b"private-shape-a", b"private-shape-b"]]
NONCE, ORACLE = "owned-run-nonce-" * 2, "a" * 64


def timing_run(duration_ns):
    samples = []
    for index, shape in enumerate(SHAPES):
        for buffer in range(2):
            start = (index * 2 + buffer + 1) * 1000
            samples.append(
                dict(
                    shape_sha256=shape,
                    buffer_index=buffer,
                    started_ns=start,
                    ended_ns=start + duration_ns,
                    synchronized_before=True,
                    synchronized_after=True,
                )
            )
    return dict(
        clock="host_monotonic_ns",
        synchronizer="external_fixture.device_synchronize",
        warmup_rounds=1,
        buffer_sha256={
            shape: [hashlib.sha256(f"{shape}:{index}".encode()).hexdigest() for index in range(2)]
            for shape in SHAPES
        },
        samples=samples,
    )


@pytest.fixture
def measured():
    contract = PerformanceContract(private_shape_sha256=SHAPES)
    evidence = dict(
        schema_version="mindie-bench-performance/1",
        run_nonce=NONCE,
        oracle_manifest_sha256=ORACLE,
        device_identity="zero-device contract fixture, not NPU evidence",
        baseline=timing_run(20),
        candidate=timing_run(10),
        graph=dict(
            method="actual_capture_and_replay",
            shape_sha256=SHAPES,
            capture_succeeded=True,
            replay_succeeded=True,
            replay_count=2,
        ),
        workspace=dict(
            method="synchronized_allocator_peak",
            baseline_bytes=200,
            candidate_bytes=180,
            synchronized_before=True,
            synchronized_after=True,
        ),
        warm_server=dict(
            method="actual_warm_server_requests",
            workload_sha256="b" * 64,
            server_config_sha256="c" * 64,
            baseline_server_identity="fixture-baseline",
            candidate_server_identity="fixture-candidate",
            baseline=timing_run(30),
            candidate=timing_run(20),
        ),
    )
    return contract, dict(
        reward=1,
        speedup_ratio=2.0,
        aclgraph_capture_compatible=True,
        memory_workspace_delta_mb=-0.00002,
        e2e_warm_server_speedup=1.5,
        performance_evidence=evidence,
    )


def parse(contract, payload, **kwargs):
    return parse_verifier(
        0 if payload["reward"] else 1,
        json.dumps(payload),
        performance_contract=contract,
        performance_nonce=kwargs.get("nonce", NONCE),
        oracle_manifest_sha256=kwargs.get("oracle", ORACLE),
    )


def test_actual_host_timing_waits_for_async_completion_and_rotates_backing_buffers():
    buffers = {shape: [bytearray(8), bytearray(8)] for shape in SHAPES}
    pending, visited = [], []

    def operation(shape, value):
        visited.append((shape, id(value)))
        worker = threading.Thread(target=lambda: time.sleep(0.002))
        worker.start()
        pending.append(worker)

    def synchronize():
        while pending:
            pending.pop().join()

    run = measure_rotating_buffers(
        buffers, operation, synchronize, buffer_identity=lambda item: str(id(item)), samples_per_shape=4
    )
    assert not pending
    assert all(sample.ended_ns - sample.started_ns > 1_000_000 for sample in run.samples)
    assert {buffer for _, buffer in visited} == {
        id(buffer) for values in buffers.values() for buffer in values
    }
    assert [item.buffer_index for item in run.samples] == [0, 1, 0, 1] * 2
    assert run.clock == "host_monotonic_ns"


def test_host_timing_keeps_original_clock_when_operation_patches_time_module(monkeypatch):
    buffers = {shape: [bytearray(8), bytearray(8)] for shape in SHAPES}
    counter = iter(range(1000))

    def operation(shape, buffer):
        monkeypatch.setattr(time, "monotonic_ns", lambda: next(counter))
        time.sleep(0.002)

    run = measure_rotating_buffers(
        buffers,
        operation,
        lambda: None,
        buffer_identity=lambda value: str(id(value)),
        samples_per_shape=2,
    )
    assert all(sample.ended_ns - sample.started_ns > 1_000_000 for sample in run.samples)


def test_actual_cpu_torch_rejects_sync_and_clock_rebinding_before_or_after_execution(tmp_path):
    interpreter = Path("/tmp/mindie-bench-cpu-l2-runtime-20261017/venv/bin/python")
    if not interpreter.is_file():
        pytest.skip("actual CPU torch control requires the existing optional CPU runtime")
    bench = Path(__file__).resolve().parents[1]
    dependency_paths = [
        str(bench / "src"),
        str(Path(sys.prefix) / "lib/python3.13/site-packages"),
        str(bench / "tasks/performance/fused-rmsnorm-gate/verifier"),
    ]
    code = f"""
import importlib.util,json,pathlib,sys,time,types
sys.path[:0]={dependency_paths!r}
import torch
from mindie_bench.evaluation.performance import MeasurementGuard
from comparator_control import reject_oracle_tampering
calls=[]
class Graph:
    def replay(self): calls.append("replay")
def synchronize(): calls.append("sync")
torch.npu=types.SimpleNamespace(synchronize=synchronize,reset_peak_memory_stats=lambda:None,
    max_memory_allocated=lambda:0,NPUGraph=Graph,graph=lambda graph:None)
guard=MeasurementGuard(torch)
guard.synchronize()
assert calls==["sync"] and guard.monotonic_ns()>0
torch.testing.assert_close(torch.tensor([1.0]),torch.tensor([1.0]))
print(json.dumps(dict(clean=True,scope="CPU torch; fake npu namespace tests identity only")))
original_clock=time.monotonic_ns
candidate=pathlib.Path({str(tmp_path / "candidate.py")!r})
for phase in ["import","execute"]:
    for target in ["synchronize","monotonic_ns"]:
        guard=MeasurementGuard(torch,on_tampered=lambda error:reject_oracle_tampering(
            error,"measurement callable integrity"))
        patch=("torch.npu.synchronize=lambda:None" if target=="synchronize"
               else "time.monotonic_ns=lambda:1")
        candidate.write_text("import torch,time\\n"+
            (patch+"\\n" if phase=="import" else "")+"def run():\\n"+
            (" "+patch+"\\n" if phase=="execute" else " pass\\n"))
        spec=importlib.util.spec_from_file_location("candidate",candidate)
        module=importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            guard.check()
            module.run()
            guard.synchronize()
        except SystemExit as error:
            assert error.code==1
        else:
            raise AssertionError("measurement mutation was accepted")
        finally:
            torch.npu.synchronize=synchronize
            time.monotonic_ns=original_clock
        # Comparator remains intact; its ordinary negative control cannot
        # substitute for the measurement-binding control.
        try: torch.testing.assert_close(torch.tensor([1.0]),torch.tensor([7.0]))
        except AssertionError: pass
        else: raise AssertionError("unexpected comparator mutation")
"""
    observed = subprocess.run(
        [str(interpreter), "-B", "-I", "-c", code],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    rows = [json.loads(line) for line in observed.stdout.splitlines()]
    assert rows[0]["clean"] is True
    assert len(rows[1:]) == 4
    assert all(row["reward"] == 0 and row["oracle_status"] == "oracle_tampered" for row in rows[1:])
    assert sum("synchronize" in row["failure_reason"] for row in rows[1:]) == 2
    assert sum("monotonic_ns" in row["failure_reason"] for row in rows[1:]) == 2


def test_two_python_views_of_same_backing_storage_are_rejected():
    storage = bytearray(8)
    with pytest.raises(ValueError, match="distinct backing"):
        measure_rotating_buffers(
            {shape: [memoryview(storage), memoryview(storage)] for shape in SHAPES},
            lambda shape, buffer: None,
            lambda: None,
            buffer_identity=lambda item: str(id(item.obj)),
            samples_per_shape=2,
        )


def test_bound_evidence_accepts_and_exposes_all_three_transfer_fields(measured):
    contract, payload = measured
    result = parse(contract, payload)
    assert result.aclgraph_capture_compatible is True
    assert result.memory_workspace_delta_mb == -0.00002
    assert result.e2e_warm_server_speedup == 1.5


@pytest.mark.parametrize("key", ["graph", "workspace", "warm_server"])
def test_scalar_or_static_claim_cannot_replace_actual_probe_receipts(measured, key):
    contract, payload = measured
    payload["performance_evidence"].pop(key)
    with pytest.raises(ValueError, match="required"):
        parse(contract, payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(clock="device_event_or_fake_clock"),
        lambda value: value["samples"][0].update(synchronized_after=False),
        lambda value: value["samples"][0].update(synchronized_before="true"),
        lambda value: value["samples"][0].update(ended_ns=1000),
        lambda value: value["samples"][1].update(buffer_index=0),
        lambda value: value["samples"].pop(),
        lambda value: value["buffer_sha256"].pop(SHAPES[1]),
        lambda value: value["buffer_sha256"][SHAPES[0]].__setitem__(
            1, value["buffer_sha256"][SHAPES[0]][0]
        ),
    ],
)
def test_forged_clock_sync_rotation_or_private_shape_coverage_is_rejected(measured, mutation):
    contract, payload = measured
    mutation(payload["performance_evidence"]["candidate"])
    with pytest.raises(ValueError):
        parse(contract, payload)


@pytest.mark.parametrize("kwargs", [{"nonce": "different-owned-run"}, {"oracle": "f" * 64}])
def test_good_receipt_from_other_run_or_oracle_is_rejected(measured, kwargs):
    with pytest.raises(ValueError, match="different run"):
        parse(*measured, **kwargs)


def test_kernel_gain_cannot_hide_warm_server_regression(measured):
    contract, payload = measured
    payload["performance_evidence"]["warm_server"]["candidate"] = timing_run(60)
    payload["e2e_warm_server_speedup"] = 0.5
    with pytest.raises(ValueError, match="e2e_warm_server_speedup_below"):
        parse(contract, payload)
    payload["reward"] = 0
    failed = parse(contract, payload)
    assert validate_performance(failed, contract, run_nonce=NONCE, oracle_manifest_sha256=ORACLE)[
        "failed_gates"
    ] == ["e2e_warm_server_speedup_below_requirement"]


def test_mismatched_scalars_do_not_override_the_measurements(measured):
    contract, payload = measured
    payload["memory_workspace_delta_mb"] = -1000
    with pytest.raises(ValueError, match="memory_workspace_delta_mb differs"):
        parse(contract, payload)


def test_workspace_cap_is_inclusive_and_over_budget_cannot_earn_reward(measured):
    contract, payload = measured
    contract.max_workspace_delta_bytes = 16
    workspace = payload["performance_evidence"]["workspace"]
    workspace["candidate_bytes"] = workspace["baseline_bytes"] + 16
    payload["memory_workspace_delta_mb"] = 16 / 1_000_000
    assert parse(contract, payload).reward == 1
    workspace["candidate_bytes"] += 1
    payload["memory_workspace_delta_mb"] = 17 / 1_000_000
    with pytest.raises(ValueError, match="workspace_delta_above_requirement"):
        parse(contract, payload)
    payload["reward"] = 0
    failed = parse(contract, payload)
    assert validate_performance(failed, contract, run_nonce=NONCE, oracle_manifest_sha256=ORACLE)[
        "failed_gates"
    ] == ["workspace_delta_above_requirement"]


def test_actual_failed_capture_is_visible_and_never_inferred_from_source(measured):
    def capture(shape):
        raise RuntimeError("actual controlled capture refusal")

    evidence = probe_aclgraph(SHAPES, capture, lambda graph: None, lambda: None)
    assert evidence.capture_succeeded is False and evidence.replay_count == 0
    assert "RuntimeError: actual controlled capture refusal" == evidence.error
    contract, payload = measured
    payload["performance_evidence"]["graph"] = evidence.model_dump()
    payload["aclgraph_capture_compatible"] = False
    with pytest.raises(ValueError, match="aclgraph_capture_incompatible"):
        parse(contract, payload)


def test_workspace_helper_synchronizes_and_reads_actual_allocator_callbacks():
    events, peak = [], [0]

    def allocate(value):
        peak[0] = value
        events.append("allocate")

    def synchronize():
        events.append("sync")

    def reset_peak():
        events.append("reset")
        peak[0] = 0

    def read_peak():
        events.append("read")
        return peak[0]

    result = measure_workspace(
        lambda: allocate(4096), lambda: allocate(2048), synchronize, reset_peak, read_peak
    )
    assert result.candidate_bytes - result.baseline_bytes == -2048
    assert events == ["sync", "reset", "allocate", "sync", "read"] * 2


def test_performance_task_cannot_omit_contract_and_legacy_task_remains_valid(case):
    raw = case["spec"].model_dump()
    TaskSpec.model_validate(raw)
    raw["task"]["domain_category"] = "operator_performance"
    with pytest.raises(ValidationError, match="requires an explicit performance"):
        TaskSpec.model_validate(raw)
    raw["task"]["domain_category"] = "operator_correctness"
    raw["task"]["source"]["task_type"] = "performance_optimization"
    with pytest.raises(ValidationError, match="requires an explicit performance"):
        TaskSpec.model_validate(raw)


def test_performance_requires_external_binding_and_legacy_speedup_is_only_a_warning(measured):
    contract, payload = measured
    with pytest.raises(ValueError, match="trusted run nonce"):
        parse_verifier(0, json.dumps(payload), performance_contract=contract)
    with pytest.raises(ValueError, match="declared performance contract"):
        parse_verifier(0, json.dumps(payload))
    legacy = parse_verifier(0, '{"reward":1,"speedup_ratio":99}')
    assert "unvalidated" in legacy.warnings[0]
    with pytest.raises(ValueError, match="structured measured evidence"):
        parse_verifier(0, "everything fast", performance_contract=contract)


def test_single_private_shape_or_tampered_reward_cannot_be_admitted():
    with pytest.raises(ValidationError):
        PerformanceContract(private_shape_sha256=[SHAPES[0], SHAPES[0]])
    with pytest.raises(ValidationError, match="oracle_tampered requires reward 0"):
        VerifierResult(
            reward=1, oracle_status="oracle_tampered", failure_reason="test", warnings=["warning"]
        )


def test_rotation_must_be_admissible_before_any_performance_execution():
    with pytest.raises(ValidationError, match="samples must cover"):
        PerformanceContract(private_shape_sha256=SHAPES, min_buffers=3, samples_per_shape=2)


def test_graph_probe_actually_calls_capture_and_replay_for_each_shape():
    events = []

    def capture(shape):
        events.append(("capture", shape))
        return ("graph", shape)

    def replay(graph):
        events.append(("replay", graph[1]))

    result = probe_aclgraph(SHAPES, capture, replay, lambda: events.append(("sync", None)))
    assert result.capture_succeeded and result.replay_succeeded
    assert result.replay_count == 2
    assert [value for action, value in events if action == "capture"] == SHAPES
    assert [value for action, value in events if action == "replay"] == SHAPES
    assert events[0][0] == events[-1][0] == "sync"


def test_warm_server_helper_measures_completed_real_loopback_requests():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"completed")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()

    def request(path, shape, payload):
        message = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/{path}", data=payload)
        with urllib.request.urlopen(message, timeout=2) as response:
            assert response.read() == b"completed"

    try:
        result = measure_warm_server(
            {shape: [bytes(f"{shape}:0", "ascii"), bytes(f"{shape}:1", "ascii")] for shape in SHAPES},
            lambda shape, payload: request("baseline", shape, payload),
            lambda shape, payload: request("candidate", shape, payload),
            lambda: None,  # Synchronous HTTP read already waits for response completion.
            buffer_identity=lambda item: str(id(item)),
            samples_per_shape=2,
            workload_sha256="b" * 64,
            server_config_sha256="c" * 64,
            baseline_server_identity="owned-loopback-baseline-control",
            candidate_server_identity="owned-loopback-candidate-control",
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert result.method == "actual_warm_server_requests"
    assert len(result.baseline.samples) == len(result.candidate.samples) == 4
    assert len([path for path, _ in received if path == "/baseline"]) == 8
    assert len([path for path, _ in received if path == "/candidate"]) == 8
    assert all(sample.ended_ns > sample.started_ns for sample in result.candidate.samples)
