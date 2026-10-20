"""Trusted synthetic example judge; executes candidate programs in child processes.

[Role] Exercise native lifecycle contracts using actual CPU subprocess timings.
[Invariants] This synthetic task is not a production model or device benchmark.
[Timeouts & Locks] Each child has 10s; the native verifier owns the outer 60s budget.
[Side-Effect Policy] Read frozen oracle files and the candidate; no remote calls.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from mindie_bench.evaluation.inference import (
    CandidateMeasurementFailure,
    evaluate_inference,
    measure_latency_ms,
    measure_workloads,
)
from mindie_bench.inference_models import InferenceContract, InferenceEvidence
from mindie_bench.io import digest

root = Path(os.environ["MINDIE_BENCH_ORACLE_ROOT"])
workspace = Path(os.environ["MINDIE_BENCH_WORKSPACE"])
data = json.loads((root / "oracle/judge.json").read_text(encoding="utf-8"))
contract = InferenceContract.model_validate(data["contract"])
baseline, candidate = root / "oracle/baseline.py", workspace / "engine.py"
if hashlib.sha256(baseline.read_bytes()).hexdigest() != data["baseline_source_sha256"]:
    raise RuntimeError("frozen baseline source changed")
runtime = dict(python=sys.version, protocol="json-stdio", command_timeout_sec=10)
if digest(runtime) != contract.runtime_sha256:
    raise RuntimeError("runtime changed; materialize a new task and re-establish soundness")


def expected(request):
    prefix = sum(x*x for x in range(request["context"]))
    return [prefix + token for token in range(request["requests"])]


def execute(arm, request):
    program = baseline if arm == "baseline" else candidate
    result = subprocess.run([sys.executable, "-I", str(program)], input=json.dumps(request),
                            text=True, capture_output=True, timeout=10)
    if result.returncode:
        if arm == "candidate":
            raise CandidateMeasurementFailure(f"candidate exit {result.returncode}")
        raise RuntimeError(f"baseline exit {result.returncode}")
    try:
        matched = json.loads(result.stdout) == expected(request)
    except ValueError:
        matched = False
    if not matched:
        if arm == "candidate":
            raise CandidateMeasurementFailure("candidate output mismatch")
        raise RuntimeError("baseline output mismatch")


execute("baseline", dict(context=16, requests=3, mode="standard"))
checks = {}
for name, request in {
    "standard": dict(context=17, requests=4, mode="standard"),
    "new": dict(context=29, requests=3, mode="new"),
    "empty": dict(context=0, requests=0, mode="standard"),
}.items():
    try:
        execute("candidate", request)
        checks[name] = True
    except CandidateMeasurementFailure:
        checks[name] = False


def measure(arm, workload):
    request = data["inputs"][workload.name]
    if digest(request) != workload.workload_sha256:
        raise RuntimeError("workload changed")
    elapsed = measure_latency_ms(lambda: execute(arm, request), lambda: None)
    return request["requests"] * 1000 / elapsed if workload.direction == "higher" else elapsed


evidence = InferenceEvidence(
    run_nonce=os.environ["MINDIE_BENCH_PERFORMANCE_NONCE"],
    oracle_manifest_sha256=os.environ["MINDIE_BENCH_ORACLE_SHA256"],
    baseline_commit=contract.baseline_commit, runtime_sha256=digest(runtime), device_class="cpu",
    baseline_source_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), baseline_healthy=True,
    checks=checks, workloads=measure_workloads(contract, measure) if all(checks.values()) else {},
)
outcome = evaluate_inference(contract, evidence)
print(json.dumps(dict(reward=float(outcome["accepted"]), inference_evidence=evidence.model_dump(),
                      failure_reason=", ".join(outcome["failed_gates"]) or None)))
sys.exit(0 if outcome["accepted"] else 1)
