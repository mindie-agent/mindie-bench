"""CPU microbenchmark of production retrieval scoring on a fixed ranking workload.

[Role] Measure scorer latency independently of model, network and NPU execution.
[Invariants] Known ranking outcomes are checked before timing; no productivity claim.
[Timeouts & Locks] Finite 10,000-call loop; the release runner owns the outer timeout.
[Side-Effect Policy] Writes only the requested numeric measurement file.
"""

import json
import os
import time
from pathlib import Path

from mindie_bench.scorer.metrics import retrieval_metrics

ranked = [f"document-{index}" for index in range(100)]
relevant = {"document-0", "document-2", "document-50"}
expected = retrieval_metrics(ranked, relevant, k=10)
assert expected["recall_at_k"] == 2 / 3
for _ in range(100):
    retrieval_metrics(ranked, relevant, k=10)
iterations = 10_000
started = time.perf_counter_ns()
for _ in range(iterations):
    measured = retrieval_metrics(ranked, relevant, k=10)
elapsed = time.perf_counter_ns() - started
assert measured == expected
Path(os.environ["MINDIE_BENCH_RESULT_PATH"]).write_text(json.dumps({
    "status": "pass",
    "metrics": [{
        "name": "retrieval_score_latency", "unit": "us/op",
        "value": elapsed / iterations / 1000, "iterations": iterations,
    }],
}), encoding="utf-8")
