# [Role] Validates trusted-judge performance probes and measured evidence.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Trusted-judge measurement helpers and fail-closed performance evidence checks.

These checks validate actual probe receipts, not candidate source text. The
probe process, its device synchronization callback and server client must remain
outside candidate write/import authority; JSON alone cannot establish that trust.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from mindie_bench.models import PerformanceContract, StrictModel, VerifierResult

SHA = r"^[0-9a-f]{64}$"


class MeasurementIntegrityError(RuntimeError):
    """A captured judge primitive changed after candidate admission."""


class MeasurementGuard:
    """Capture before candidate imports; reject changed direct callable bindings.

    This guards these explicit bindings and Python code objects, not arbitrary
    mutation of the interpreter, native extensions or private guard state.
    """

    def __init__(self, torch_module, *, on_tampered=None):
        npu = torch_module.npu
        self._on_tampered = on_tampered
        self._bindings = [
            (torch_module, "npu", npu),
            (torch_module, "testing", torch_module.testing),
            (time, "monotonic_ns", time.monotonic_ns),
            (time, "monotonic", time.monotonic),
            (torch_module, "allclose", torch_module.allclose),
            (torch_module.testing, "assert_close", torch_module.testing.assert_close),
            *[
                (npu, name, getattr(npu, name))
                for name in [
                    "synchronize",
                    "reset_peak_memory_stats",
                    "max_memory_allocated",
                    "NPUGraph",
                    "graph",
                ]
            ],
            (npu.NPUGraph, "replay", npu.NPUGraph.replay),
        ]
        self._codes = [getattr(value, "__code__", None) for _, _, value in self._bindings]
        self._functions = {name: value for _, name, value in self._bindings}

    def check(self):
        for (owner, name, expected), code in zip(self._bindings, self._codes, strict=True):
            observed = getattr(owner, name, None)
            if observed is not expected or getattr(observed, "__code__", None) is not code:
                error = MeasurementIntegrityError(f"judge measurement binding changed: {name}")
                if self._on_tampered is not None:
                    self._on_tampered(error)
                raise error

    def _call(self, name, *args, **kwargs):
        self.check()
        result = self._functions[name](*args, **kwargs)
        self.check()
        return result

    def synchronize(self):
        return self._call("synchronize")

    def monotonic_ns(self):
        return self._call("monotonic_ns")

    def reset_peak_memory_stats(self):
        return self._call("reset_peak_memory_stats")

    def max_memory_allocated(self):
        return self._call("max_memory_allocated")

    def new_graph(self):
        return self._call("NPUGraph")

    def graph(self, graph):
        return self._call("graph", graph)

    def replay(self, graph):
        return self._call("replay", graph)


class TimingSample(StrictModel):
    shape_sha256: str = Field(pattern=SHA)
    buffer_index: int = Field(ge=0, strict=True)
    started_ns: int = Field(ge=0, strict=True)
    ended_ns: int = Field(ge=0, strict=True)
    synchronized_before: bool = Field(strict=True)
    synchronized_after: bool = Field(strict=True)

    @model_validator(mode="after")
    def positive_interval(self):
        if not self.synchronized_before or not self.synchronized_after:
            raise ValueError("performance timing requires synchronization before and after")
        if self.ended_ns <= self.started_ns:
            raise ValueError("performance interval must advance host monotonic time")
        return self


class TimingRun(StrictModel):
    clock: Literal["host_monotonic_ns"]
    synchronizer: str = Field(min_length=1)
    warmup_rounds: int = Field(ge=1)
    buffer_sha256: dict[str, list[str]]
    samples: list[TimingSample] = Field(min_length=1)


class GraphProbe(StrictModel):
    method: Literal["actual_capture_and_replay"]
    shape_sha256: list[str]
    capture_succeeded: bool = Field(strict=True)
    replay_succeeded: bool = Field(strict=True)
    replay_count: int = Field(ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def actual_outcome(self):
        if self.capture_succeeded and self.replay_succeeded:
            if self.replay_count < 1 or self.error is not None:
                raise ValueError("successful graph probe requires actual replays and no error")
        elif not self.error:
            raise ValueError("failed actual graph probe requires the original error")
        return self


class WorkspaceProbe(StrictModel):
    method: Literal["synchronized_allocator_peak"]
    baseline_bytes: int = Field(ge=0, strict=True)
    candidate_bytes: int = Field(ge=0, strict=True)
    synchronized_before: bool = Field(strict=True)
    synchronized_after: bool = Field(strict=True)

    @model_validator(mode="after")
    def synchronized(self):
        if not self.synchronized_before or not self.synchronized_after:
            raise ValueError("workspace measurement requires device synchronization")
        return self


class WarmServerProbe(StrictModel):
    method: Literal["actual_warm_server_requests"]
    workload_sha256: str = Field(pattern=SHA)
    server_config_sha256: str = Field(pattern=SHA)
    baseline_server_identity: str = Field(min_length=1)
    candidate_server_identity: str = Field(min_length=1)
    baseline: TimingRun
    candidate: TimingRun


class PerformanceEvidence(StrictModel):
    schema_version: Literal["mindie-bench-performance/1"] = "mindie-bench-performance/1"
    run_nonce: str = Field(min_length=16)
    oracle_manifest_sha256: str = Field(pattern=SHA)
    device_identity: str = Field(min_length=1)
    baseline: TimingRun
    candidate: TimingRun
    graph: GraphProbe | None = None
    workspace: WorkspaceProbe | None = None
    warm_server: WarmServerProbe | None = None


def measure_rotating_buffers(
    shape_buffers: Mapping[str, Sequence[Any]],
    operation: Callable[[str, Any], Any],
    synchronize: Callable[[], None],
    *,
    buffer_identity: Callable[[Any], str],
    samples_per_shape: int,
    warmup_rounds: int = 1,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> TimingRun:
    """Measure on the judge host; device completion belongs to synchronize().

    buffer_identity must identify backing storage (for example tensor.data_ptr),
    not a transient Python wrapper. Device identity is supplied by the external
    probe, never inferred from this generic helper's successful CPU execution.
    """
    if len(shape_buffers) < 2 or warmup_rounds < 1:
        raise ValueError("at least two private shapes and a warmup are required")
    buffers = {}
    for shape, values in shape_buffers.items():
        identities = [str(buffer_identity(value)) for value in values]
        if len(values) < 2 or len(set(identities)) != len(values):
            raise ValueError("multiple distinct backing buffers are required for each private shape")
        if samples_per_shape < len(values):
            raise ValueError("samples must cover every rotated buffer")
        buffers[shape] = [hashlib.sha256(value.encode()).hexdigest() for value in identities]
    samples = []
    for shape, values in shape_buffers.items():
        for _ in range(warmup_rounds):
            for value in values:
                synchronize()
                operation(shape, value)
                synchronize()
        for index in range(samples_per_shape):
            buffer_index = index % len(values)
            synchronize()
            started = clock_ns()
            operation(shape, values[buffer_index])
            synchronize()
            ended = clock_ns()
            samples.append(
                TimingSample(
                    shape_sha256=shape,
                    buffer_index=buffer_index,
                    started_ns=started,
                    ended_ns=ended,
                    synchronized_before=True,
                    synchronized_after=True,
                )
            )
    return TimingRun(
        clock="host_monotonic_ns",
        synchronizer=f"{synchronize.__module__}.{synchronize.__qualname__}",
        warmup_rounds=warmup_rounds,
        buffer_sha256=buffers,
        samples=samples,
    )


def probe_aclgraph(
    private_shapes: Sequence[str],
    capture: Callable[[str], Any],
    replay: Callable[[Any], None],
    synchronize: Callable[[], None],
    *,
    replay_count: int = 1,
) -> GraphProbe:
    if len(set(private_shapes)) < 2 or replay_count < 1:
        raise ValueError("actual graph probe requires two private shapes and at least one replay")
    captured = False
    completed_replays = 0
    try:
        graphs = []
        for shape in private_shapes:
            synchronize()
            graphs.append(capture(shape))
            synchronize()
        captured = True
        for graph in graphs:
            for _ in range(replay_count):
                synchronize()
                replay(graph)
                synchronize()
                completed_replays += 1
    except Exception as error:
        return GraphProbe(
            method="actual_capture_and_replay",
            shape_sha256=list(private_shapes),
            capture_succeeded=captured,
            replay_succeeded=False,
            replay_count=completed_replays,
            error=f"{type(error).__name__}: {error}",
        )
    return GraphProbe(
        method="actual_capture_and_replay",
        shape_sha256=list(private_shapes),
        capture_succeeded=True,
        replay_succeeded=True,
        replay_count=completed_replays,
    )


def measure_workspace(
    baseline: Callable[[], None],
    candidate: Callable[[], None],
    synchronize: Callable[[], None],
    reset_peak: Callable[[], None],
    peak_bytes: Callable[[], int],
) -> WorkspaceProbe:
    values = []
    for operation in [baseline, candidate]:
        synchronize()
        reset_peak()
        operation()
        synchronize()
        values.append(peak_bytes())
    return WorkspaceProbe(
        method="synchronized_allocator_peak",
        baseline_bytes=values[0],
        candidate_bytes=values[1],
        synchronized_before=True,
        synchronized_after=True,
    )


def measure_warm_server(
    shape_buffers: Mapping[str, Sequence[Any]],
    baseline_request: Callable[[str, Any], Any],
    candidate_request: Callable[[str, Any], Any],
    synchronize: Callable[[], None],
    *,
    buffer_identity: Callable[[Any], str],
    samples_per_shape: int,
    workload_sha256: str,
    server_config_sha256: str,
    baseline_server_identity: str,
    candidate_server_identity: str,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> WarmServerProbe:
    """Time completed requests to already running servers, excluding startup."""
    runs = [
        measure_rotating_buffers(
            shape_buffers,
            request,
            synchronize,
            buffer_identity=buffer_identity,
            samples_per_shape=samples_per_shape,
            clock_ns=clock_ns,
        )
        for request in [baseline_request, candidate_request]
    ]
    return WarmServerProbe(
        method="actual_warm_server_requests",
        workload_sha256=workload_sha256,
        server_config_sha256=server_config_sha256,
        baseline_server_identity=baseline_server_identity,
        candidate_server_identity=candidate_server_identity,
        baseline=runs[0],
        candidate=runs[1],
    )


def _run_means(run: TimingRun, contract: PerformanceContract) -> dict[str, float]:
    shapes = set(contract.private_shape_sha256)
    if set(run.buffer_sha256) != shapes or {item.shape_sha256 for item in run.samples} != shapes:
        raise ValueError("timing evidence does not cover the exact private shape identities")
    previous_end = -1
    for sample in run.samples:
        if sample.started_ns < previous_end:
            raise ValueError("serial performance samples overlap or reverse host monotonic time")
        previous_end = sample.ended_ns
    means = {}
    for shape in shapes:
        buffers = run.buffer_sha256[shape]
        if (
            len(buffers) < contract.min_buffers
            or len(buffers) != len(set(buffers))
            or any(len(item) != 64 or any(c not in "0123456789abcdef" for c in item) for item in buffers)
        ):
            raise ValueError("timing evidence lacks distinct backing buffer identities")
        samples = [item for item in run.samples if item.shape_sha256 == shape]
        if len(samples) < max(contract.samples_per_shape, len(buffers)) or [
            item.buffer_index for item in samples
        ] != [index % len(buffers) for index in range(len(samples))]:
            raise ValueError("timing evidence lacks complete buffer rotation and sample coverage")
        means[shape] = statistics.mean(item.ended_ns - item.started_ns for item in samples)
    return means


def _speedup(baseline: TimingRun, candidate: TimingRun, contract: PerformanceContract) -> float:
    before, after = _run_means(baseline, contract), _run_means(candidate, contract)
    if baseline.warmup_rounds != candidate.warmup_rounds or any(
        len(baseline.buffer_sha256[shape]) != len(candidate.buffer_sha256[shape])
        or sum(item.shape_sha256 == shape for item in baseline.samples)
        != sum(item.shape_sha256 == shape for item in candidate.samples)
        for shape in before
    ):
        raise ValueError("baseline and candidate must use matching warmup, buffer and sample counts")
    return statistics.geometric_mean(before[shape] / after[shape] for shape in before)


def _same_metric(name: str, reported: float | int | None, actual: float | int) -> None:
    if reported is None or not math.isclose(reported, actual, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError(f"{name} differs from the independently measured performance evidence")


def validate_performance(
    verdict: VerifierResult,
    contract: PerformanceContract,
    *,
    run_nonce: str,
    oracle_manifest_sha256: str,
) -> dict:
    """Validate evidence from the externally controlled judge for this run.

    Invalid/missing evidence raises; an actual measured failed gate returns a
    failed result. Neither is converted to an unmeasured successful performance.
    """
    evidence = PerformanceEvidence.model_validate(verdict.performance_evidence)
    if evidence.run_nonce != run_nonce or evidence.oracle_manifest_sha256 != oracle_manifest_sha256:
        raise ValueError("performance evidence belongs to a different run or frozen oracle")
    speedup = _speedup(evidence.baseline, evidence.candidate, contract)
    _same_metric("speedup_ratio", verdict.speedup_ratio, speedup)
    failed = []
    if speedup < contract.min_speedup_ratio:
        failed.append("kernel_speedup_below_requirement")
    if contract.require_aclgraph_capture or verdict.aclgraph_capture_compatible is not None:
        if evidence.graph is None or set(evidence.graph.shape_sha256) != set(
            contract.private_shape_sha256
        ):
            raise ValueError("required actual ACLGraph capture/replay evidence missing")
        compatible = evidence.graph.capture_succeeded and evidence.graph.replay_succeeded
        if compatible and evidence.graph.replay_count < len(contract.private_shape_sha256):
            raise ValueError("actual graph replay did not cover all private shapes")
        if verdict.aclgraph_capture_compatible is not compatible:
            raise ValueError("aclgraph_capture_compatible disagrees with actual capture/replay probe")
        if not compatible:
            failed.append("aclgraph_capture_incompatible")
    if contract.require_workspace_measurement or verdict.memory_workspace_delta_mb is not None:
        if evidence.workspace is None:
            raise ValueError("required synchronized workspace measurement missing")
        delta = evidence.workspace.candidate_bytes - evidence.workspace.baseline_bytes
        _same_metric("memory_workspace_delta_mb", verdict.memory_workspace_delta_mb, delta / 1_000_000)
        if contract.max_workspace_delta_bytes is not None and delta > contract.max_workspace_delta_bytes:
            failed.append("workspace_delta_above_requirement")
    if contract.require_e2e_warm_server or verdict.e2e_warm_server_speedup is not None:
        if evidence.warm_server is None:
            raise ValueError("required actual warm-server request evidence missing")
        e2e = _speedup(evidence.warm_server.baseline, evidence.warm_server.candidate, contract)
        _same_metric("e2e_warm_server_speedup", verdict.e2e_warm_server_speedup, e2e)
        if e2e < contract.min_e2e_warm_server_speedup:
            failed.append("e2e_warm_server_speedup_below_requirement")
    if failed and verdict.reward == 1.0:
        raise ValueError("performance reward contradicts measured gate failures: " + ", ".join(failed))
    return {
        "status": "failed" if failed else "passed",
        "failed_gates": failed,
        "speedup_ratio": speedup,
        "run_nonce": evidence.run_nonce,
        "oracle_manifest_sha256": evidence.oracle_manifest_sha256,
        "measurement_scope": "trusted isolated judge receipts; not a static source analysis",
        "memory_workspace_delta_unit": "MB (1000000 bytes)",
    }
