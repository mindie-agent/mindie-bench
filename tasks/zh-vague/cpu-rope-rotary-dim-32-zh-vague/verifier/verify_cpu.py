import ast
import json
import pathlib
import types
from collections.abc import Mapping

import torch

if torch.__version__ != "2.10.0":
    raise RuntimeError(f"CPU task requires locked torch 2.10.0, got {torch.__version__}")


def source_functions(relative, namespace):
    source = pathlib.Path(relative)
    tree = ast.parse(source.read_text(), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    focused = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(focused), str(source), "exec"), namespace)
    return namespace


try:
    DEVICE = "cpu"
    VARIANT_DIM = 32
    torch.manual_seed(4301)
    cases = 0
    max_error = 0.0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for mode in ("explicit", "factor", "full"):
            rotary_dim = VARIANT_DIM if mode != "full" else 128
            metadata = (
                types.SimpleNamespace(rotary_dim=VARIANT_DIM)
                if mode == "explicit"
                else types.SimpleNamespace(partial_rotary_factor=VARIANT_DIM / 128, rotary_dim=16)
                if mode == "factor"
                else types.SimpleNamespace()
            )
            for token_count in (1, 7):
                state = {
                    name: None
                    for name in (
                        "_cos_mla",
                        "_sin_mla",
                        "_cos",
                        "_sin",
                        "_cos_cache",
                        "_sin_cache",
                        "_cos_sin_cache",
                        "_cos_slice",
                        "_sin_slice",
                    )
                }
                state.update(torch=torch, is_vl_model=lambda config: False, has_rope=lambda config: True)
                source_functions("vllm_ascend/ops/rotary_embedding.py", state)
                config = types.SimpleNamespace(
                    model_config=types.SimpleNamespace(
                        use_mla=False, get_head_size=lambda: 128, hf_text_config=metadata
                    ),
                    scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16),
                )
                freqs = torch.outer(
                    torch.arange(64, dtype=torch.float32),
                    10000.0 ** (-torch.arange(0, rotary_dim, 2) / rotary_dim),
                )
                cache_cpu = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)
                cache_device = cache_cpu.to(DEVICE)
                state["_cos_sin_cache"] = cache_device
                state["set_cos_and_sin"](config, 4, 1, dtype, DEVICE)
                positions_cpu = torch.tensor([63, 0, 9, 17, 2, 31, 7][:token_count], dtype=torch.int64)
                state["update_cos_sin"](positions_cpu.to(DEVICE))
                assert tuple(state["_cos"].shape) == (1, 16, 1, rotary_dim)
                cos, sin = state["get_cos_and_sin_slice"]()
                assert cos.device.type == sin.device.type == "cpu"
                selected = cache_cpu.index_select(0, positions_cpu)
                expected_cos = (
                    selected[:, : rotary_dim // 2].repeat(1, 2).reshape(1, token_count, 1, rotary_dim)
                )
                expected_sin = (
                    selected[:, rotary_dim // 2 :].repeat(1, 2).reshape(1, token_count, 1, rotary_dim)
                )
                torch.testing.assert_close(cos.cpu(), expected_cos, rtol=0, atol=0)
                torch.testing.assert_close(sin.cpu(), expected_sin, rtol=0, atol=0)
                original_cpu = torch.randn(1, token_count, 2, 128).to(dtype)
                value = original_cpu.to(DEVICE)
                rot = value[..., :rotary_dim]
                half = torch.cat((-rot[..., rotary_dim // 2 :], rot[..., : rotary_dim // 2]), dim=-1)
                actual = torch.cat((rot * cos + half * sin, value[..., rotary_dim:]), dim=-1).cpu()
                prefix = original_cpu[..., :rotary_dim]
                rotated = torch.cat(
                    (-prefix[..., rotary_dim // 2 :], prefix[..., : rotary_dim // 2]), dim=-1
                )
                expected = torch.cat(
                    (prefix * expected_cos + rotated * expected_sin, original_cpu[..., rotary_dim:]),
                    dim=-1,
                )
                error = (actual.float() - expected.float()).abs().max().item()
                max_error = max(max_error, error)
                tolerance = (
                    0.04 if dtype == torch.bfloat16 else 0.006 if dtype == torch.float16 else 1e-5
                )
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(
                    actual[..., rotary_dim:], original_cpu[..., rotary_dim:], rtol=0, atol=0
                )
                torch.testing.assert_close(cache_device.cpu(), cache_cpu, rtol=0, atol=0)
                cases += 1
    # All tensors execute on the explicit CPU device.
    print(
        f"verified {cases} real CPU partial-RoPE cache cases; explicit dimension, factor precedence and full head"
    )
    print(json.dumps({"reward": 1.0, "max_abs_error": max_error}))
except (AssertionError, TypeError, ValueError, AttributeError, NameError, RuntimeError) as exc:
    import traceback

    traceback.print_exc()
    if isinstance(exc, RuntimeError) and not any(
        marker in str(exc).casefold()
        for marker in ("expanded size", "selected index k out of range", "size of tensor")
    ):
        raise
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
