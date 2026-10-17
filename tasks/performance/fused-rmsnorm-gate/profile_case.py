"""Bounded diagnostic of host enqueue, stream span and synchronized eight-step wall time."""

import cProfile
import json
import pstats
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from load_case import load

torch, config, current_config, upstream_class = load()
torch.npu.set_device(0)
import torch_npu  # noqa: E402


def summarize(samples):
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in ["host_enqueue_us", "stream_span_us", "synchronized_us", "checksum_us"]
    }


with current_config(config), torch.inference_mode():
    layer = upstream_class(128, eps=1e-6, activation="sigmoid", device="npu", dtype=torch.bfloat16)
    layer.weight.fill_(1)
    torch.manual_seed(531203)
    cases = []
    for tokens, heads, strided in [(37, 4, False), (193, 8, True)]:
        pairs = []
        for _ in range(4):
            x = torch.randn((1, tokens, heads, 128), dtype=torch.bfloat16, device="npu")
            gate = torch.randn((tokens, heads, 128), dtype=x.dtype, device="npu") * 0.2
            if strided:
                packed = torch.full((tokens, heads * 2, 128), torch.nan, device="npu", dtype=x.dtype)
                packed[:, :heads].copy_(gate)
                gate = packed[:, :heads]
            pairs.append((x, gate))
        cases.append((str((tokens, heads, strided)), pairs))
    operations = {
        "native": lambda x, gate: upstream_class.forward_native(layer, x, gate),
        "dispatched": lambda x, gate: layer(x, gate),
    }
    if type(layer) is not upstream_class:
        operations["direct_oot"] = layer.forward_oot
        operations["native_npu_norm"] = lambda x, gate: (
            torch_npu.npu_rms_norm(x, layer.weight, layer.eps)[0] * torch.sigmoid(gate)
        )
        for _, pairs in cases:
            for x, gate in pairs:
                torch.testing.assert_close(
                    operations["native_npu_norm"](x, gate),
                    operations["native"](x, gate),
                    rtol=0.02,
                    atol=0.02,
                )

    def one(operation, pair):
        torch.npu.set_device(0)
        with current_config(config), torch.inference_mode():
            torch.npu.synchronize()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            x, gate = pair
            start.record()
            started = time.monotonic_ns()
            for _ in range(8):
                x = operation(x, gate)
            enqueued = time.monotonic_ns()
            end.record()
            torch.npu.synchronize()
            synchronized = time.monotonic_ns()
            float(x.float().mean().cpu())
            checksum = time.monotonic_ns()
            return {
                "host_enqueue_us": (enqueued - started) / 1000,
                "stream_span_us": start.elapsed_time(end) * 1000,
                "synchronized_us": (synchronized - started) / 1000,
                "checksum_us": (checksum - synchronized) / 1000,
            }

    def run_case(operation, pairs):
        for pair in pairs:
            one(operation, pair)
        samples = [one(operation, pairs[index % 4]) for index in range(12)]
        profiler = cProfile.Profile()
        profiler.runcall(lambda: [one(operation, pairs[index % 4]) for index in range(4)])
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:12]
        return {
            **summarize(samples),
            "thread": threading.current_thread().name,
            "samples": len(samples),
            "host_self_profile": [
                {
                    "function": f"{name[0]}:{name[1]}:{name[2]}",
                    "calls": values[1],
                    "self_ms": values[2] * 1000,
                    "cumulative_ms": values[3] * 1000,
                }
                for name, values in top
            ],
        }

    def device_profile(operation, pair):
        torch.npu.set_device(0)
        with current_config(config), torch.inference_mode(), TemporaryDirectory() as tmp:
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ]
            ) as prof:
                x, gate = pair
                for _ in range(8):
                    x = operation(x, gate)
                torch.npu.synchronize()
            trace = Path(tmp) / "trace.json"
            prof.export_chrome_trace(str(trace))
            parsed = json.loads(trace.read_text())
            events = parsed if isinstance(parsed, list) else parsed["traceEvents"]
            durations = {}
            for event in events:
                if event.get("ph") != "X" or "dur" not in event:
                    continue
                category = event.get("cat", "")
                if "kernel" in category.lower() or category == "async_npu":
                    key = (category, event["name"])
                    durations[key] = durations.get(key, 0.0) + event["dur"]
            return {
                "npu_activity_total_us": sum(durations.values()),
                "npu_activities": [
                    {"category": category, "name": name, "duration_us": duration}
                    for (category, name), duration in sorted(
                        durations.items(), key=lambda item: item[1], reverse=True
                    )
                ],
                "event_categories": sorted({event.get("cat", "") for event in events}),
            }

    results = {}
    with ThreadPoolExecutor(max_workers=1) as worker:
        for shape, pairs in cases:
            results[shape] = {}
            for name, operation in operations.items():
                results[shape][name] = {
                    "main": run_case(operation, pairs),
                    "service_thread": worker.submit(run_case, operation, pairs).result(),
                }
                if name != "direct_oot":
                    results[shape][name]["device_profile"] = worker.submit(
                        device_profile, operation, pairs[0]
                    ).result()
    print(
        json.dumps(
            {
                "scope": "diagnostic only; same two shapes/four buffers/eight steps; stream span includes host enqueue gaps, not pure kernel time",
                "class": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "results": results,
            },
            allow_nan=False,
        )
    )
