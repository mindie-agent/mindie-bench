# [Role] Independently verify CPU tensor and leave-one-out fixture contracts.
# [Invariants] Candidate metrics and source spelling never establish correctness or speed.
# [Timeouts & Locks] The caller bounds this process; no locks or device contexts.
# [Side-Effect Policy] Loads candidate Python in this process; emits a JSON verdict only.

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

if importlib.util.find_spec("triton") is None:
    # Support CPU-only import of modules that also define Triton functions.
    # A decorated function is never executable or launchable in this adapter.
    class UnavailableJIT:
        def __init__(self, fn):
            self.__name__ = fn.__name__

        def __call__(self, *args, **kwargs):
            raise ModuleNotFoundError("Triton execution unavailable in the CPU import adapter")

        def __getitem__(self, grid):
            raise ModuleNotFoundError("Triton kernel launch unavailable in the CPU import adapter")

    triton_mod = types.ModuleType("triton")
    triton_mod.__path__ = []
    triton_mod.jit = UnavailableJIT
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.next_power_of_2 = lambda n: 1 if n <= 1 else (1 << (int(n) - 1).bit_length())
    tl_mod = types.ModuleType("triton.language")
    tl_mod.constexpr = int
    errors_mod = types.ModuleType("triton.errors")
    errors_mod.TritonError = type("TritonError", (Exception,), {})
    triton_mod.language = tl_mod
    triton_mod.errors = errors_mod
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = tl_mod
    sys.modules["triton.errors"] = errors_mod


evidence = {
    "validation_scope": "controlled_cpu_tensor_and_ablation_contract",
    "correctness_verified": None,
    "ablation_contract_verified": None,
    "native_kernel_bound": None,
    "native_kernel_status": "unverified",
    "measured_npu_speedup": None,
    "speedup_ratio": None,
    "tpot_ms": None,
    "latency_ms": None,
    "npu_card_hours": None,
    "performance_status": "unmeasured",
    "measurement_source": None,
    "operator_binding": "cpu_tensor_execution",
}
root = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
try:
    import torch
    import torch.nn.functional as F
    from torch.utils._python_dispatch import TorchDispatchMode

    sys.path.insert(0, str(Path(__file__).parent))
    from runtime_controls import CallCapture, UnsupportedCPUExecution, invoke_suite, output_tensor

    class TensorTrace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = self.reductions = self.bf16_casts = self.bf16_arithmetic = 0
            self.indexing_operations = 0

        def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.operations += 1
            if fn in (
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.select.int,
                torch.ops.aten.index.Tensor,
            ):
                self.indexing_operations += 1
            if fn in (torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim):
                self.reductions += 1
            if fn == torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.bfloat16:
                self.bf16_casts += 1
            out = fn(*args, **kwargs)
            if fn in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul_.Tensor,
                torch.ops.aten.sum.dim_IntList,
            ):
                if isinstance(out, torch.Tensor) and out.dtype == torch.bfloat16:
                    self.bf16_arithmetic += 1
            return out

        def summary(self):
            return {
                "tensor_operations": self.operations,
                "indexing_operations": self.indexing_operations,
                "reductions": self.reductions,
                "bf16_casts": self.bf16_casts,
                "bf16_arithmetic": self.bf16_arithmetic,
            }

    target = root / "vllm_ascend/ops/spatial_group_norm_reduce.py"
    spec = importlib.util.spec_from_file_location("spatial_group_norm_reduce", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    gen = torch.Generator().manual_seed(91099)
    for shape, num_groups in (((2, 16, 16, 16), 4), ((3, 32, 24, 24), 8)):
        x = (torch.randn(shape, generator=gen) * 1.8 + 0.7).to(torch.bfloat16)
        w = (torch.randn(shape[1], generator=gen) * 0.5 + 1.0).to(torch.bfloat16)
        b = (torch.randn(shape[1], generator=gen) * 0.2).to(torch.bfloat16)
        ref = F.group_norm(x.float(), num_groups, w.float(), b.float(), eps=1e-5).to(torch.bfloat16)
        out = output_tensor(mod.spatial_group_norm_reduce(x, w, b, num_groups=num_groups, eps=1e-5))
        assert out is not None and out.shape == x.shape and out.dtype == x.dtype, "Invalid output tensor"
        err = float((out.float() - ref.float()).abs().max())
        assert err <= 5e-3 and torch.allclose(out.float(), ref.float(), atol=5e-3, rtol=5e-3), (
            f"Numerical check failed on shape {shape}: max_abs_err={err}"
        )
    evidence["correctness_verified"] = True

    x = (torch.randn((2, 16, 32, 32), generator=gen) * 2.5 + 1.5).to(torch.bfloat16)
    w, b = torch.ones(16, dtype=torch.bfloat16), torch.zeros(16, dtype=torch.bfloat16)
    inputs = (x, w, b)
    core = {"num_groups": 4, "eps": 1e-5}
    ref = F.group_norm(x.float(), 4, w.float(), b.float(), eps=1e-5).to(torch.bfloat16)
    has_ablation_suite = hasattr(mod, "run_ablation_suite")
    suite_error = None
    with TensorTrace() as default_trace:
        with CallCapture(mod, "spatial") as capture:
            mod.spatial_group_norm_reduce(*inputs, num_groups=4, eps=1e-5)
            capture.phase = "suite"
            if has_ablation_suite:
                try:
                    suite = invoke_suite(mod, "spatial", inputs, core)
                    evidence["candidate_suite_type"] = type(suite).__name__
                except (ImportError, NameError):
                    raise
                except Exception as exc:
                    suite_error = exc

    selected = None
    examined = 0
    failures = []
    if not has_ablation_suite:
        assert (
            default_trace.indexing_operations == 0 and default_trace.bf16_arithmetic == 0
        ), f"Default path did not eliminate 3D indexing loop and bf16 accumulation: {default_trace.summary()}"
        selected = {"entrypoint": "spatial_group_norm_reduce", "full_candidate": default_trace.summary()}
    for entrypoint, controls in (capture.entries.items() if has_ablation_suite else ()):
        values = list(controls.values())
        if len(values) < 4 or not any(c.is_default for c in values):
            continue
        observed = []
        for control in values:
            examined += 1
            sample = {"control": control.description, "default_path": control.is_default}
            try:
                with TensorTrace() as trace:
                    out = output_tensor(control.replay(tuple(t.clone() for t in inputs), core))
                sample.update(trace.summary())
                assert out is not None and out.shape == ref.shape and out.dtype == ref.dtype, (
                    "Replay output shape/dtype mismatch"
                )
                sample["finite"] = bool(torch.isfinite(out).all())
                sample["max_abs_err"] = float((out.float() - ref.float()).abs().max())
                sample["correct"] = sample["finite"] and sample["max_abs_err"] <= 5e-3
            except Exception as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"
                sample["correct"] = False
            observed.append(sample)
        for full_index, full in enumerate(observed):
            if not full["default_path"] or not full["correct"]:
                continue
            for dim_index, dim in enumerate(observed):
                if (
                    dim_index == full_index
                    or not dim["correct"]
                    or dim["indexing_operations"] <= full["indexing_operations"]
                ):
                    continue
                for block_index, block in enumerate(observed):
                    if block_index in (full_index, dim_index) or not block["correct"]:
                        continue
                    if (
                        block["tensor_operations"] <= full["tensor_operations"]
                        or block["indexing_operations"] >= dim["indexing_operations"]
                        or block["indexing_operations"] <= full["indexing_operations"]
                    ):
                        continue
                    for guard_index, guard in enumerate(observed):
                        if guard_index in (full_index, dim_index, block_index) or not guard.get(
                            "finite"
                        ):
                            continue
                        if (
                            guard["bf16_casts"] <= full["bf16_casts"]
                            and guard["bf16_arithmetic"] <= full["bf16_arithmetic"]
                        ):
                            continue
                        selected = {
                            "entrypoint": entrypoint,
                            "full_candidate": full,
                            "minus_dim_merge": dim,
                            "minus_block_scaling": block,
                            "minus_multibuffer_cast_guard": guard,
                        }
                        break
                    if selected:
                        break
                if selected:
                    break
            if selected:
                break
        if selected:
            break
        failures.append({"entrypoint": entrypoint, "observed": observed})
    evidence["observed_cpu_ablation"] = selected or {
        "entrypoints": failures,
        "replayed_controls": examined,
    }
    if selected is None and suite_error is not None:
        message = str(suite_error)
        if isinstance(suite_error, (ImportError, NotImplementedError)) or (
            isinstance(suite_error, ValueError) and "NPU" in message and "require" in message
        ):
            raise UnsupportedCPUExecution(
                f"CPU ablation execution unavailable: {type(suite_error).__name__}: {message}"
            )
        raise AssertionError(
            f"Executed CPU ablation failed: {type(suite_error).__name__}: {message}"
        ) from suite_error
    assert selected is not None, (
        "Independent replay did not establish three distinct numerical/tensor-work ablations"
    )
    evidence.update(
        correctness_verified=True, ablation_contract_verified=True, verification_status="completed"
    )
    print(json.dumps({"reward": 1.0, **evidence}))
except Exception as exc:
    evidence["verification_status"] = (
        "unsupported"
        if isinstance(exc, (ModuleNotFoundError, ImportError, NotImplementedError))
        else "failed"
    )
    print(
        json.dumps(
            {"reward": 0.0, "failure_reason": str(exc), "error_type": type(exc).__name__, **evidence}
        )
    )
    raise SystemExit(1)
