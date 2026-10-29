# [Role] Independent CPU tensor/state oracle for MRV2 DSA prefill metadata.
# [Invariants] Exercise candidate behavior across all four edited files; source spelling cannot pass.
# [Timeouts & Locks] Caller bounds this isolated process; no locks or background workers.
# [Side-Effect Policy] Read candidate source and execute CPU collaborators; no native/NPU claims.
"""Checks cache shape, prefill propagation, draft context restoration and sequence ownership."""
import ast
import contextlib
import copy
import json
import os
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
DSA = "vllm_ascend/attention/dsa_v1.py"
ATTN = "vllm_ascend/worker/v2/attn_utils.py"
DEFAULT = "vllm_ascend/worker/v2/model_states/default.py"
SPEC = "vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py"


def parse(relative):
    return ast.parse((ROOT / relative).read_text(), filename=relative)


def compile_nodes(nodes, namespace, filename):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


def candidate_namespace(values):
    """Give extracted classes real module globals (dataclass/typing resolve here)."""
    import sys
    module = types.ModuleType(f"_behavior_candidate_{len(sys.modules)}")
    module.__dict__.update(values)
    sys.modules[module.__name__] = module
    return module.__dict__


def load_support(tree, namespace, filename):
    """Candidate definitions remain executable, including extracted helper classes.

    Hardware imports/module initialization are outside this CPU boundary. Unrelated
    definitions with unavailable bases/defaults are deferred; exercised failures surface.
    """
    namespace.setdefault("__name__", "__main__")
    # CPU-safe imports used by extracted helpers retain their normal aliases.
    # Imported hardware/runtime collaborators remain the explicitly injected seam.
    allowed = {"collections", "contextlib", "dataclasses", "enum", "functools", "itertools",
               "math", "typing", "types", "numpy", "torch", "copy", "operator", "numbers"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases = [alias for alias in node.names
                       if alias.name.split(".")[0] in allowed
                       and (alias.asname or alias.name.split(".")[0]) not in namespace]
            selected = ast.Import(names=aliases) if aliases else None
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in allowed:
            aliases = [alias for alias in node.names if (alias.asname or alias.name) not in namespace]
            selected = ast.ImportFrom(module=node.module, names=aliases, level=node.level) if aliases else None
        else:
            selected = None
        if selected is not None:
            module = ast.Module(body=[selected], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    pending = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in namespace:
                pending.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    namespace.setdefault(target.id, value)
    while pending:
        deferred = []
        for node in pending:
            try:
                compile_nodes([node], namespace, filename)
            except (NameError, AttributeError):
                deferred.append(node)
        if len(deferred) == len(pending):
            break
        pending = deferred


def load_class(tree, name, methods, namespace, filename, base="object"):
    # Keep the actual class cell so zero-argument super() retains Python semantics.
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name))
    node.bases, node.keywords, node.decorator_list = [ast.Name(id=base, ctx=ast.Load())], [], []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    load_support(tree, namespace, filename)
    compile_nodes([node], namespace, filename)
    return namespace[name]


def check_cache_shape(trees):
    namespace = candidate_namespace({"torch": torch})
    cls = load_class(trees[DSA], "AscendDSABackend", {"get_kv_cache_shape"}, namespace, DSA)
    for blocks, size, heads, dim in ((3, 16, 2, 64), (11, 128, 1, 512), (1, 8, 4, 32)):
        for dtype in ("auto", "float16", "bfloat16"):
            shape = cls.get_kv_cache_shape(
                num_blocks=blocks, block_size=size, num_kv_heads=heads,
                head_size=dim, cache_dtype_str=dtype,
            )
            assert shape == (blocks, size, heads, dim), f"cache shape lost layout: {shape}"


class Metadata(types.SimpleNamespace):
    """CPU carrier for the public common-attention metadata fields."""


class DsaBuilder:
    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.common_ratio_to_sas_metadata = kwargs["common_ratio_to_sas_metadata"]
        return common_attn_metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return common_attn_metadata


def metadata_runtime(tree):
    namespace = candidate_namespace({"torch": torch, "np": np, "contextmanager": contextlib.contextmanager,
                 "AscendCommonAttentionMetadata": Metadata, "AscendDSAMetadataBuilder": DsaBuilder})
    load_support(tree, namespace, ATTN)
    return namespace["build_attn_metadata"]


def check_prefill_flow(trees):
    build = metadata_runtime(trees[ATTN])
    mode = types.SimpleNamespace(FULL="FULL", NONE="NONE")
    namespace = candidate_namespace({"torch": torch, "build_attn_metadata": build, "CUDAGraphMode": mode})
    cls = load_class(trees[DEFAULT], "AscendModelState", {"prepare_attn"}, namespace, DEFAULT)
    model = cls.__new__(cls)
    model.max_model_len = 512
    groups = [[types.SimpleNamespace(
        get_metadata_builder=lambda index: DsaBuilder(),
        layer_names=[f"layer-{group}"], kv_cache_spec=types.SimpleNamespace(block_size=16),
    )] for group in range(2)]
    cache = types.SimpleNamespace(kv_cache_groups=[object(), object()])
    for flags in ([True, False, False, False], [False, True, False, False], [False] * 4):
        for graph_mode, capture in ((mode.NONE, False), (mode.FULL, False), (mode.FULL, True)):
            prefill = np.array(flags, dtype=np.bool_)
            queries = np.array([0, 3, 5, 5, 5], dtype=np.int32)
            seqs = np.array([11, 19, 0, 0], dtype=np.int32)
            batch = types.SimpleNamespace(
                num_reqs=2, num_tokens=5, num_reqs_after_padding=4, num_tokens_after_padding=8,
                is_prefilling_np=prefill, query_start_loc_np=queries,
                query_start_loc=torch.from_numpy(queries.copy()), num_scheduled_tokens=np.array([3, 2]),
                seq_lens=torch.from_numpy(seqs.copy()), seq_lens_np=seqs, dcp_local_seq_lens=None,
                positions=torch.arange(8), attn_state="ChunkedPrefill",
            )
            blocks = (torch.tensor([[3, 5]] * 4), torch.tensor([[7, 9]] * 4))
            slots = torch.arange(16).reshape(2, 8)
            result = model.prepare_attn(batch, graph_mode, blocks, slots, groups, cache, capture)
            assert result is model.attn_metadata, "model must retain returned metadata"
            assert set(result) == {"layer-0", "layer-1"}, "per-layer metadata missing"
            requests, tokens = (4, 8) if graph_mode == mode.FULL else (2, 5)
            for i in range(2):
                meta = result[f"layer-{i}"]
                assert isinstance(meta.is_prefilling, torch.Tensor), "prefill flags missing"
                assert meta.is_prefilling.dtype == torch.bool, "prefill flags changed dtype"
                assert meta.is_prefilling.ndim == 1 and meta.is_prefilling.numel() >= requests
                assert meta.is_prefilling[:requests].tolist() == flags[:requests], "active prefill flags lost/reused"
                assert not bool(meta.is_prefilling[requests:].any()), "padding was marked as live prefill"
                assert (meta.num_reqs, meta.num_actual_tokens, meta.num_input_tokens) == (requests, 5, tokens)
                torch.testing.assert_close(meta.seq_lens, batch.seq_lens[:requests])
                torch.testing.assert_close(meta.query_start_loc, batch.query_start_loc)
                torch.testing.assert_close(meta.block_table_tensor, blocks[i])
                torch.testing.assert_close(meta.slot_mapping, slots[i])
            assert prefill.tolist() == flags and seqs.tolist() == [11, 19, 0, 0], "input state mutated"


def spec_namespace(tree):
    calls = []
    def raw(*args, **kwargs):
        calls.append((args, kwargs))
        return {"layer": Metadata(seq_lens=torch.tensor([7, 13]), attn_state="original")}
    module = types.SimpleNamespace(build_attn_metadata=raw)
    class Parent:
        def _build_draft_attn_metadata(self, *args):
            return module.build_attn_metadata(*args)
    namespace = candidate_namespace({"torch": torch, "contextmanager": contextlib.contextmanager,
                 "vllm_speculator": module, "AutoRegressiveSpeculator": Parent,
                 "AscendAttentionState": types.SimpleNamespace(DecodeOnly="DecodeOnly")})
    cls = load_class(tree, "AscendAutoRegressiveSpeculator",
                     {"_build_draft_attn_metadata", "_ascend_update_seq_lens"},
                     namespace, SPEC, "AutoRegressiveSpeculator")
    return namespace, cls, module, raw, calls


def check_draft_flow(trees):
    namespace, cls, module, raw, calls = spec_namespace(trees[SPEC])
    positions = torch.arange(9) + 20
    flags = torch.tensor([True, False, False], dtype=torch.bool)
    factory = namespace["build_draft_attn_metadata_factory"]
    for fail in (False, True):
        try:
            with factory(positions, 5, flags):
                module.build_attn_metadata("probe", causal=False)
                kwargs = calls[-1][1]
                torch.testing.assert_close(kwargs["positions"], positions[:5])
                torch.testing.assert_close(kwargs["is_prefilling"], flags)
                assert kwargs["causal"] is False
                if fail:
                    raise LookupError("controlled body failure")
        except LookupError:
            assert fail
        assert module.build_attn_metadata is raw, "draft patch leaked after context exit"
    runner = cls.__new__(cls)
    runner.input_buffers = types.SimpleNamespace(positions=positions)
    for values in ([True, False, False], [False, True, False]):
        runner.input_batch = types.SimpleNamespace(is_prefilling_np=np.array(values, dtype=np.bool_))
        upper = torch.tensor([8, 14, 0])
        result = runner._build_draft_attn_metadata(2, 3, 5, upper, 2, 1, False)
        args, kwargs = calls[-1]
        assert args[:3] == (2, 3, 5) and args[4:] == (2, 1, False), "draft arguments changed"
        torch.testing.assert_close(args[3], upper)
        prefill = kwargs["is_prefilling"]
        assert isinstance(prefill, torch.Tensor) and prefill.dtype == torch.bool
        assert prefill.ndim == 1 and prefill.numel() >= 2
        assert prefill[:2].tolist() == values[:2], "draft call lost active prefill state"
        assert not bool(prefill[2:].any()), "draft padding was marked as live prefill"
        torch.testing.assert_close(kwargs["positions"], positions[:5])
        assert result["layer"].attn_state == "DecodeOnly"
        assert module.build_attn_metadata is raw


def check_seq_lens(trees):
    _, cls, _, _, _ = spec_namespace(trees[SPEC])
    runner = cls.__new__(cls)
    for architecture in ("DSA", "MLA", "GQA"):
        runner.attn_architecture = architecture
        runner._ascend_update_seq_lens(None)
        runner._ascend_update_seq_lens({})
        for values in ([5, 19], [0, 127, 3]):
            metadata = {name: Metadata(seq_lens=torch.tensor(values), seq_len_list=list(values))
                        for name in ("first", "second")}
            original = {name: value.seq_lens for name, value in metadata.items()}
            for step in range(1, 4):
                runner._ascend_update_seq_lens(metadata)
                expected = torch.tensor(values) + (0 if architecture == "DSA" else step)
                for name, value in metadata.items():
                    torch.testing.assert_close(value.seq_lens, expected)
                    assert value.seq_len_list == expected.tolist(), "host/device lengths diverged"
                    if architecture == "DSA":
                        assert value.seq_lens is original[name], "DSA builder owns persistent tensor"


def main():
    failures = []
    error_types = []
    try:
        trees = {path: parse(path) for path in (DSA, ATTN, DEFAULT, SPEC)}
    except (OSError, SyntaxError) as exc:
        error_types.append(type(exc).__name__)
        failures.append(f"{type(exc).__name__}: {exc}")
    if not failures:
        for check in (check_cache_shape, check_prefill_flow, check_draft_flow, check_seq_lens):
            try:
                check(trees)
            except Exception as exc:
                error_types.append(type(exc).__name__)
                failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    print(json.dumps({"reward": 0.0 if failures else 1.0,
                      "correctness_verified": not failures, "validation_scope": "cpu_behavior",
                      "native_kernel_bound": None, "measured_npu_speedup": None, "speedup_ratio": None,
                      "error_type": error_types[0] if error_types else ("AssertionError" if failures else None), "failure_reason": "; ".join(failures) if failures else None}))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
