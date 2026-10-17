"""Actual single-NPU operator and warm tiny-service probes; no full-model claim."""

# The private dependency root is established before importing judge helpers.
# ruff: noqa: E402

import gc
import http.client
import json
import os
import socket
import statistics
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ORACLE = Path(os.environ["MINDIE_BENCH_ORACLE_ROOT"])
sys.path[:0] = [str(ORACLE / "verifier"), str(ORACLE / "oracle/runtime")]

import torch
import torch_npu  # noqa: F401
from comparator_control import enforce_comparator_control, reject_oracle_tampering
from load_case import load

from mindie_bench.evaluation.performance import (
    MeasurementGuard,
    PerformanceEvidence,
    TimingRun,
    measure_rotating_buffers,
    measure_warm_server,
    measure_workspace,
    probe_aclgraph,
    validate_performance,
)
from mindie_bench.io import digest
from mindie_bench.models import PerformanceContract, VerifierResult

primitives = MeasurementGuard(
    torch,
    on_tampered=lambda error: reject_oracle_tampering(error, "measurement callable integrity"),
)
torch, config, current_config, upstream_class = load()
primitives.check()
enforce_comparator_control(torch)
torch.npu.set_device(0)
private_cases = json.loads((ORACLE / "oracle/cases.json").read_text())
contract = PerformanceContract(
    private_shape_sha256=[digest(case) for case in private_cases],
    min_buffers=4,
    samples_per_shape=36,
    min_speedup_ratio=1.05,
    min_e2e_warm_server_speedup=1.01,
    max_workspace_delta_bytes=16 * 1024 * 1024,
)
BASELINE_RESERVE_BYTES = 16 * 1024 * 1024
SERVICE_SCRATCH_BYTES = 32 * 1024 * 1024


def mean_speedup(before, after):
    def mean(run, shape):
        return statistics.mean(
            item.ended_ns - item.started_ns for item in run.samples if item.shape_sha256 == shape
        )

    return statistics.geometric_mean(
        mean(before, shape) / mean(after, shape) for shape in contract.private_shape_sha256
    )


with current_config(config), torch.inference_mode():
    layer = upstream_class(128, eps=1e-6, activation="sigmoid", device="npu", dtype=torch.bfloat16)
    layer.weight.fill_(1)
    primitives.check()
    dispatched = type(layer) is not upstream_class
    buffers, expected, inputs_before = {}, {}, {}
    torch.manual_seed(531203)
    for case in private_cases:
        key = digest(case)
        _, tokens, heads, width = case["shape"]
        values = []
        for index in range(contract.min_buffers):
            x = torch.randn(case["shape"], device="npu", dtype=torch.bfloat16)
            gate = torch.randn((tokens, heads, width), device="npu", dtype=torch.bfloat16) * 0.2
            if case["strided_gate"]:
                packed = torch.full((tokens, 2 * heads, width), torch.nan, device="npu", dtype=x.dtype)
                packed[:, :heads].copy_(gate)
                gate = packed[:, :heads]
            values.append((x, gate))
            # Reference runs on CPU, independently of the candidate's NPU path.
            x_cpu, gate_cpu = x.cpu().float(), gate.cpu().float()
            reference = x_cpu * torch.rsqrt(x_cpu.square().mean(-1, keepdim=True) + layer.eps)
            expected[(key, index)] = (reference * torch.sigmoid(gate_cpu)).to(x.dtype)
            inputs_before[(key, index)] = (x.clone(), gate.clone())
        buffers[key] = values

    def baseline(shape, pair):
        result = upstream_class.forward_native(layer, *pair)
        primitives.check()
        return result

    def candidate(shape, pair):
        result = layer(*pair)
        primitives.check()
        return result

    max_error = 0.0
    for key, values in buffers.items():
        for index, pair in enumerate(values):
            actual = candidate(key, pair)
            primitives.synchronize()
            enforce_comparator_control(torch)
            max_error = max(
                max_error, float((actual.cpu().float() - expected[(key, index)].float()).abs().max())
            )
            torch.testing.assert_close(actual.cpu(), expected[(key, index)], rtol=0.02, atol=0.02)
            for value, before in zip(pair, inputs_before[(key, index)], strict=True):
                torch.testing.assert_close(value, before, rtol=0, atol=0)

    def identity(pair):
        return f"{pair[0].data_ptr()}:{pair[1].data_ptr()}"

    timing_options = dict(
        buffer_identity=identity,
        samples_per_shape=12,
        warmup_rounds=2,
        clock_ns=primitives.monotonic_ns,
    )
    baseline_blocks, candidate_blocks = [], []
    block_orders = ["AB", "BA", "AB"]
    for order in block_orders:
        for arm in order:
            measured = measure_rotating_buffers(
                buffers, baseline if arm == "A" else candidate, primitives.synchronize, **timing_options
            )
            (baseline_blocks if arm == "A" else candidate_blocks).append(measured)

    def combine(blocks):
        return TimingRun(
            clock=blocks[0].clock,
            synchronizer=blocks[0].synchronizer,
            warmup_rounds=blocks[0].warmup_rounds,
            buffer_sha256=blocks[0].buffer_sha256,
            samples=[sample for block in blocks for sample in block.samples],
        )

    before, after = combine(baseline_blocks), combine(candidate_blocks)
    captured_outputs = {}

    def capture(shape):
        graph = primitives.new_graph()
        with primitives.graph(graph):
            captured_outputs[shape] = candidate(shape, buffers[shape][0])
        return graph

    graph_probe = probe_aclgraph(
        list(buffers), capture, primitives.replay, primitives.synchronize, replay_count=3
    )
    if graph_probe.capture_succeeded and graph_probe.replay_succeeded:
        enforce_comparator_control(torch)
        for shape, output in captured_outputs.items():
            torch.testing.assert_close(output.cpu(), expected[(shape, 0)], rtol=0.02, atol=0.02)

    retained = []

    def reset_peak():
        retained.clear()
        gc.collect()
        primitives.reset_peak_memory_stats()

    workspace_probe = measure_workspace(
        lambda: retained.extend(baseline(shape, values[0]) for shape, values in buffers.items()),
        lambda: retained.extend(candidate(shape, values[0]) for shape, values in buffers.items()),
        primitives.synchronize,
        reset_peak,
        primitives.max_memory_allocated,
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        @torch.inference_mode()
        def do_POST(self):
            torch.npu.set_device(0)
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            shape, index = request["shape"], request["buffer"]
            x, gate = buffers[shape][index]
            operation = baseline if self.path == "/baseline" else candidate
            with current_config(config):
                for _ in range(8):
                    x = operation(shape, (x, gate))
                primitives.synchronize()
                body = json.dumps({"checksum": float(x.float().mean().cpu())}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    connection.connect()
    connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    positions = {id(pair): index for values in buffers.values() for index, pair in enumerate(values)}
    checksums = {}

    def request(route, shape, pair):
        connection.request("POST", route, json.dumps({"shape": shape, "buffer": positions[id(pair)]}))
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError("warm tiny-service request did not complete successfully")
        checksum = result["checksum"]
        if not isinstance(checksum, (int, float)):
            raise RuntimeError("warm tiny-service response has no numerical result")
        key = (shape, positions[id(pair)])
        if route == "/baseline":
            checksums[key] = checksum
        elif key in checksums and abs(checksum - checksums[key]) > 0.02:
            raise RuntimeError("warm tiny-service candidate differs from baseline output")

    try:
        server_blocks = []
        for order in block_orders:
            measured = measure_warm_server(
                buffers,
                lambda shape, pair: request("/baseline" if order == "AB" else "/candidate", shape, pair),
                lambda shape, pair: request("/candidate" if order == "AB" else "/baseline", shape, pair),
                primitives.synchronize,
                buffer_identity=identity,
                samples_per_shape=12,
                clock_ns=primitives.monotonic_ns,
                workload_sha256=digest({"shapes": private_cases, "steps": 8, "weight": "ones"}),
                server_config_sha256=digest(
                    {"protocol": "HTTP/1.1", "worker_threads": 1, "steps": 8, "tcp_nodelay": True}
                ),
                baseline_server_identity=f"pid={os.getpid()};port={server.server_port};route=baseline",
                candidate_server_identity=f"pid={os.getpid()};port={server.server_port};route=candidate",
            )
            if order == "BA":
                measured.baseline, measured.candidate = measured.candidate, measured.baseline
            server_blocks.append(measured)
        warm_server = server_blocks[0].model_copy(
            update={
                "baseline": combine([block.baseline for block in server_blocks]),
                "candidate": combine([block.candidate for block in server_blocks]),
            }
        )
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    speedup = mean_speedup(before, after)
    e2e_speedup = mean_speedup(warm_server.baseline, warm_server.candidate)
    graph_ok = graph_probe.capture_succeeded and graph_probe.replay_succeeded
    kernel_block_ratios = [
        mean_speedup(baseline_run, candidate_run)
        for baseline_run, candidate_run in zip(baseline_blocks, candidate_blocks, strict=True)
    ]
    server_block_ratios = [mean_speedup(block.baseline, block.candidate) for block in server_blocks]
    accepted = (
        dispatched
        and graph_ok
        and speedup >= 1.05
        and e2e_speedup >= 1.01
        and all(ratio >= 1.05 for ratio in kernel_block_ratios)
        and all(ratio >= 1.01 for ratio in server_block_ratios)
        and workspace_probe.baseline_bytes <= BASELINE_RESERVE_BYTES
        and workspace_probe.candidate_bytes <= SERVICE_SCRATCH_BYTES
        and workspace_probe.candidate_bytes - workspace_probe.baseline_bytes
        <= contract.max_workspace_delta_bytes
    )
    primitives.check()
    enforce_comparator_control(torch)
    evidence = PerformanceEvidence(
        run_nonce=os.environ["MINDIE_BENCH_PERFORMANCE_NONCE"],
        oracle_manifest_sha256=os.environ["MINDIE_BENCH_ORACLE_SHA256"],
        device_identity=torch.npu.get_device_name(0),
        baseline=before,
        candidate=after,
        graph=graph_probe,
        workspace=workspace_probe,
        warm_server=warm_server,
    )
    verdict = VerifierResult(
        reward=1.0 if accepted else 0.0,
        speedup_ratio=speedup,
        max_abs_error=max_error,
        aclgraph_capture_compatible=graph_ok,
        memory_workspace_delta_mb=(workspace_probe.candidate_bytes - workspace_probe.baseline_bytes)
        / 1_000_000,
        e2e_warm_server_speedup=e2e_speedup,
        performance_evidence=evidence.model_dump(mode="json"),
        failure_reason=None
        if accepted
        else "actual dispatch, graph, operator, memory budget or warm tiny-service gate failed",
    )
    validate_performance(
        verdict,
        contract,
        run_nonce=os.environ["MINDIE_BENCH_PERFORMANCE_NONCE"],
        oracle_manifest_sha256=os.environ["MINDIE_BENCH_ORACLE_SHA256"],
    )
    print(
        json.dumps(
            {
                "dispatch_class": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "dispatch_changed": dispatched,
                "block_order": block_orders,
                "kernel_block_speedup_ratios": kernel_block_ratios,
                "warm_server_block_speedup_ratios": server_block_ratios,
                "memory_budget": {
                    "baseline_reserve_bytes": BASELINE_RESERVE_BYTES,
                    "service_scratch_bytes": SERVICE_SCRATCH_BYTES,
                    "max_workspace_delta_bytes": contract.max_workspace_delta_bytes,
                    "scope": "single tiny eight-step service tensor scratch; no real KV cache is present",
                },
                "measurement_integrity": {
                    "status": "intact",
                    "basis": "direct callable bindings/code objects captured before candidate import",
                    "limit": "does not detect arbitrary interpreter or native-extension mutation",
                },
                "scope": "two private shapes; one real NPU; warm eight-norm-step HTTP service, not a full vLLM model server",
            }
        )
    )
    print(verdict.model_dump_json())
    raise SystemExit(0 if accepted else 1)
