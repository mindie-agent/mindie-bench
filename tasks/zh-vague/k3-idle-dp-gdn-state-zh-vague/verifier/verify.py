# [Role] Independent CPU behavior oracle for the historical regression.
# [Invariants] Only executed tensor/state assertions establish success; helpers remain candidate code.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources; no NPU initialization or hardware performance claims.

"""Behavior oracle for idle-DP GDN state protection (vllm-ascend #15655).

Runs the workspace NPUModelRunner on CPU with minimal collaborators:
execute_model on an idle external-launcher DP rank, then _dummy_run through
its real _build_attention_metadata call (stopping at the dummy forward). The
GDN output must contain no writable token/query ranges while ordinary attention
still sees the dummy token, and the 310P runner must forward the flag. The CPU
metadata carrier follows public GDN fields and Ascend nested query/index aliases;
this does not execute vLLM builders or native device kernels. Sources without the runner classes fail
before any runtime is needed; once they exist, a missing torch runtime exits
without a reward line and is recorded as infrastructure.
"""

import ast
import contextlib
import copy
import dataclasses
import json
import os
import types
from pathlib import Path

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
RUNNER = "vllm_ascend/worker/model_runner_v1.py"
RUNNER_310P = "vllm_ascend/_310p/model_runner_310p.py"
REQUIRED = {RUNNER: "NPUModelRunner", RUNNER_310P: "NPUModelRunner310"}
BUILTIN_DECORATORS = {"staticmethod": staticmethod, "classmethod": classmethod, "property": property}
STALE_GDN_STARTS = [0, 5, 9, 0, 0]


def find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined at module level")


def find_method(tree, class_name, name):
    for item in find_class(tree, class_name).body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise AssertionError(f"{class_name}.{name} is not defined")


def define_function(node, namespace, filename):
    """Define one function; a default that needs an absent collaborator leaves it undefined."""
    previous = namespace.get(node.name, define_function)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    try:
        module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        exec(compile(module, filename, "exec"), namespace)
        return namespace[node.name]
    except NameError:
        return None
    finally:
        if previous is define_function:
            namespace.pop(node.name, None)
        else:
            namespace[node.name] = previous


def candidate_namespace(values):
    """Give extracted classes real module globals (dataclass/typing resolve here)."""
    import sys
    module = types.ModuleType(f"_behavior_candidate_{len(sys.modules)}")
    module.__dict__.update(values)
    sys.modules[module.__name__] = module
    return module.__dict__


def load_support(tree, namespace, filename):
    """Load candidate helpers, including classes, without module imports/initialization.

    Unrelated hardware definitions may need unavailable bases/defaults; leave those
    deferred. An exercised dependency still raises visibly at the behavior boundary.
    AST selection is only an execution boundary and never a success criterion.
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
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    while pending:
        deferred = []
        for node in pending:
            try:
                module = ast.Module(body=[future, node], type_ignores=[])
                exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
            except (NameError, AttributeError):
                deferred.append(node)
        if len(deferred) == len(pending):
            break
        pending = deferred


def load_class(tree, name, namespace, filename):
    """Rebuild a class from its methods; only methods the oracle calls must be definable."""
    load_support(tree, namespace, filename)
    cls = type(name, (), {})
    for item in find_class(tree, name).body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)) and item.value is not None:
            try:
                value = ast.literal_eval(item.value)
            except (ValueError, TypeError):
                continue
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    setattr(cls, target.id, value)
            continue
        if not isinstance(item, ast.FunctionDef):
            continue
        method = copy.deepcopy(item)
        wrappers = [d.id for d in method.decorator_list if isinstance(d, ast.Name) and d.id in BUILTIN_DECORATORS]
        method.decorator_list = []
        function = define_function(method, namespace, filename)
        if function is not None:
            for wrapper in wrappers:
                function = BUILTIN_DECORATORS[wrapper](function)
            setattr(cls, item.name, function)
    return cls


class StopAfterMetadata(Exception):
    """The dummy forward after attention metadata is outside this oracle."""


def stop_before_forward(*args, **kwargs):
    raise StopAfterMetadata()


@dataclasses.dataclass
class GDNAttentionMetadata:
    """Public schema from vLLM gdn_attn.py at 58d3918; Ascend attaches decode metadata.

    The fixture carries real tensor fields consumed by gdn.py. It neither performs
    a device calculation nor substitutes a successful verdict for candidate code.
    """
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decodes: int = 1
    num_decode_tokens: int = 1
    num_spec_decodes: int = 0
    num_spec_decode_tokens: int = 0
    num_actual_tokens: int = 1
    has_initial_state: object = None
    spec_query_start_loc: object = None
    non_spec_query_start_loc: object = None
    spec_state_indices_tensor: object = None
    non_spec_state_indices_tensor: object = None
    spec_sequence_masks: object = None
    spec_token_indx: object = None
    non_spec_token_indx: object = None
    num_accepted_tokens: object = None
    chunk_indices: object = None
    chunk_offsets: object = None
    prefill_query_start_loc: object = None
    prefill_state_indices: object = None
    prefill_has_initial_state: object = None
    nums_dict: object = None
    batch_ptr: object = None
    token_chunk_offset_ptr: object = None


class RecordingBuilder:
    def __init__(self):
        self.calls, self.outputs = [], []

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.calls.append((common_attn_metadata, kwargs))
        output = types.SimpleNamespace(num_prefills=0, num_decodes=1, num_spec_decodes=0)
        self.outputs.append(output)
        return output

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)


class RecordingGdnBuilder(RecordingBuilder):
    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        import torch
        self.calls.append((common_attn_metadata, kwargs))
        starts = common_attn_metadata.query_start_loc.clone()
        indices = torch.tensor([2], dtype=torch.int32)
        output = GDNAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_decode_tokens=common_attn_metadata.num_actual_tokens,
            non_spec_query_start_loc=starts, non_spec_state_indices_tensor=indices,
        )
        # Ascend's builder attaches a second reference to these tensors. Replacing
        # only the top-level field must not leave stale writable nested metadata.
        output.non_spec_decode_metadata = types.SimpleNamespace(
            causal_conv1d=types.SimpleNamespace(query_start_loc=starts, cache_indices=indices,
                                               initial_state_mode=None),
            actual_seq_lengths=torch.diff(starts),
        )
        output.spec_decode_metadata = None
        output.non_spec_prefill_metadata = None
        self.outputs.append(output)
        return output


def workspace_helpers(tree, namespace):
    """Resolve exercised helper imports from the candidate archive, retaining runtime seams."""
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    pending = [find_method(tree, "NPUModelRunner", name)
               for name in ("execute_model", "_dummy_run", "_build_attention_metadata")]
    seen, names = set(), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for ref in ast.walk(node):
            if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load):
                names.add(ref.id)
                if ref.id in definitions and ref.id not in namespace:
                    pending.append(definitions[ref.id])
    for node in tree.body:
        if not (isinstance(node, ast.ImportFrom) and node.module
                and node.module.startswith("vllm_ascend.")):
            continue
        wanted = [alias for alias in node.names if (alias.asname or alias.name) in names
                  and (alias.asname or alias.name) not in namespace]
        if not wanted:
            continue
        source = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
        if not source.is_file():
            # An absent imported candidate module remains a visible loading failure.
            continue
        child_tree = ast.parse(source.read_text(), filename=str(source))
        child = candidate_namespace(dict(namespace))
        load_support(child_tree, child, str(source))
        for alias in wanted:
            if alias.name in child:
                namespace[alias.asname or alias.name] = child[alias.name]


def common_metadata_class(method):
    """Stand-in dataclass with every field the workspace constructs or replaces."""
    fields = {"encoder_seq_lens", "encoder_seq_lens_cpu", "num_logits_indices", "logits_indices_padded"}
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if callee in {"AscendCommonAttentionMetadata", "replace"}:
                fields.update(keyword.arg for keyword in node.keywords if keyword.arg)
    return dataclasses.make_dataclass(
        "AscendCommonAttentionMetadata",
        [(field, object, dataclasses.field(default=None)) for field in sorted(fields)],
    )


def runner_namespace(tree, torch, np):
    namespace = candidate_namespace({
        "torch": torch, "np": np, "replace": dataclasses.replace, "copy": copy.copy,
        "deepcopy": copy.deepcopy, "nullcontext": contextlib.nullcontext, "time": __import__("time"),
        "cdiv": lambda a, b: -(-a // b),
        "CUDAGraphMode": types.SimpleNamespace(NONE="NONE", PIECEWISE="PIECEWISE", FULL="FULL"),
        "AscendAttentionState": types.SimpleNamespace(
            DecodeOnly="DecodeOnly", SpecDecoding="SpecDecoding", ChunkedPrefill="ChunkedPrefill",
        ),
        "AscendCommonAttentionMetadata": common_metadata_class(
            tree
        ),
        "vllm_version_is": lambda version: False,
        "get_pp_group": lambda: types.SimpleNamespace(world_size=1, is_last_rank=True, is_first_rank=True),
        "has_kv_transfer_group": lambda: False,
        "has_ec_transfer": lambda: False,
        "record_function_or_nullcontext": lambda name: contextlib.nullcontext(),
        "EMPTY_MODEL_RUNNER_OUTPUT": "EMPTY_MODEL_RUNNER_OUTPUT",
    })
    for name in (
        "AscendDSAMetadataBuilder", "AscendDSACPMetadataBuilder",
        "AscendSFADCPMetadataBuilder", "EncoderOnlyAttentionSpec",
    ):
        namespace[name] = type(name, (RecordingBuilder,) if name.endswith("Builder") else (), {})
    namespace["GDNAttentionMetadata"] = GDNAttentionMetadata
    namespace["GDNAttentionMetadataBuilder"] = RecordingGdnBuilder
    namespace["AscendGDNAttentionMetadataBuilder"] = type("AscendGDNAttentionMetadataBuilder", (RecordingGdnBuilder,), {})
    workspace_helpers(tree, namespace)
    return namespace


class Buffer:
    def __init__(self, torch, values):
        self.cpu = torch.tensor(values, dtype=torch.int32)
        self.gpu = self.cpu.clone()
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n=None):
        if n is None:
            return self.gpu.copy_(self.cpu)
        self.gpu[:n].copy_(self.cpu[:n])
        return self.gpu[:n]


def idle_runner(cls, namespace, torch, np):
    gdn, attention = namespace["GDNAttentionMetadataBuilder"](), RecordingBuilder()
    max_reqs = 4

    class Tables(list):
        def commit_block_table(self, num_reqs):
            return None

    tables = Tables(
        types.SimpleNamespace(
            slot_mapping=Buffer(torch, [7] * 8),
            get_device_tensor=lambda: torch.zeros(max_reqs, 4, dtype=torch.int32),
        )
        for _ in range(2)
    )
    runner = cls.__new__(cls)
    runner.__dict__.update(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=max_reqs),
        dynamic_eplb=False, dcp_size=1, use_async_spec_decode=False, sparse_kv_offload_enabled=False,
        speculative_config=None, drafter=None, use_compress=False, is_mm_prefix_lm=False, lora_config=None,
        _has_gdn=True, max_model_len=64, max_num_tokens=8, attn_state=None, decode_token_per_req=1,
        long_seq_metadata=None, _offload_req_ids_tensor=None, _offload_token_to_req=None,
        optimistic_seq_lens_cpu=torch.zeros(max_reqs, dtype=torch.int32),
        seq_lens=torch.zeros(max_reqs, dtype=torch.int32),
        actual_seq_lengths_q=torch.zeros(max_reqs, dtype=torch.int32),
        positions=torch.zeros(8, dtype=torch.int64), _dsa_positions_cpu_buf=None,
        query_pos=types.SimpleNamespace(np=np.arange(8, dtype=np.int64)),
        query_start_loc=Buffer(torch, [0] * (max_reqs + 1)),
        gdn_query_start_loc=Buffer(torch, STALE_GDN_STARTS),
        group_len=Buffer(torch, [0] * max_reqs), group_key_idx=Buffer(torch, [0] * max_reqs),
        group_key_cache_idx=Buffer(torch, [0] * max_reqs),
        input_batch=types.SimpleNamespace(
            block_table=tables, req_ids=[], prev_req_id_to_index=None,
            num_computed_tokens_cpu_tensor=torch.zeros(max_reqs, dtype=torch.int32),
            num_prompt_tokens_cpu_tensor=torch.ones(max_reqs, dtype=torch.int32),
        ),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[
            types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(), layer_names=["attn"]),
            types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(), layer_names=["gdn"]),
        ]),
        attn_groups=[
            [types.SimpleNamespace(get_metadata_builder=lambda ubid=0: attention, layer_names=["attn"])],
            [types.SimpleNamespace(get_metadata_builder=lambda ubid=0: gdn, layer_names=["gdn"])],
        ],
        model_config=types.SimpleNamespace(enable_return_routed_experts=False),
        cache_config=types.SimpleNamespace(kv_sharing_fast_prefill=False),
        vllm_config=types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(
                cudagraph_mode=types.SimpleNamespace(has_full_cudagraphs=lambda: False)
            ),
            model_config=types.SimpleNamespace(use_mla=False, enable_return_routed_experts=False),
        ),
        _determine_batch_execution_and_padding=lambda **kwargs: (
            "NONE", types.SimpleNamespace(num_tokens=kwargs["num_tokens"], num_reqs=None), None, None, None
        ),
        synchronize_input_prep=contextlib.nullcontext,
        _should_build_dummy_attn_metadata=lambda *args, **kwargs: True,
        _get_cumsum_and_arange=lambda tokens, arange: np.cumsum(tokens),
        _pad_query_start_loc_for_fia=lambda *args: args[2],
        _get_encoder_seq_lens=lambda *args: (None, None),
        maybe_dummy_run_with_lora=stop_before_forward,
    )
    return runner, gdn, attention


class NativeDispatch(Exception):
    """The CPU oracle stops immediately before a device operation would execute."""


class NativeOpsBoundary:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        raise NativeDispatch("active GDN computation reached a device operator")


def check_zero_token_consumer(metadata, torch):
    """Execute the archived consumer's zero-token path; shape alone cannot prove early return."""
    relative = "vllm_ascend/ops/gdn.py"
    source = ROOT / relative
    if not source.is_file():
        return "no archived GDN consumer is available for an alternate no-op path"
    tree = ast.parse(source.read_text(), filename=relative)
    class CpuTorch:
        ops = NativeOpsBoundary()
        def __getattr__(self, name):
            return getattr(torch, name)
    context = types.SimpleNamespace(attn_metadata={"gdn": metadata})
    namespace = candidate_namespace({
        "torch": CpuTorch(), "GDNAttentionMetadata": GDNAttentionMetadata, "PAD_SLOT_ID": -1,
        "get_forward_context": lambda: context,
        "wait_for_kv_layer_from_connector": lambda prefix: None,
        "record_attention_compute_start": lambda: None,
    })
    cls = load_class(tree, "AscendGatedDeltaNetAttention", namespace, relative)
    model = cls.__new__(cls)
    model.prefix, model.activation = "gdn", "silu"
    model.kv_cache = [torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
                      torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)]
    model.conv1d = types.SimpleNamespace(weight=torch.ones(4, 1, 3), bias=torch.ones(4))
    before = [tensor.clone() for tensor in model.kv_cache]
    output = torch.full((1, 1, 2), 9.0)
    try:
        model._forward_core(torch.ones(1, 4), torch.ones(1, 2), torch.ones(1, 2), output)
    except Exception as exc:
        return f"zero-token consumer did not return before device computation: {type(exc).__name__}: {exc}"
    if bool(output.any()) or any(not torch.equal(a, b) for a, b in zip(before, model.kv_cache)):
        return "zero-token consumer changed persistent state or left nonzero output"
    # An unconditional return/constant-output consumer is not a functional repair.
    starts, indices = torch.tensor([0, 1], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    active = GDNAttentionMetadata(non_spec_query_start_loc=starts, non_spec_state_indices_tensor=indices)
    active.non_spec_decode_metadata = types.SimpleNamespace(
        causal_conv1d=types.SimpleNamespace(query_start_loc=starts, cache_indices=indices),
        actual_seq_lengths=torch.tensor([1], dtype=torch.int32),
    )
    context.attn_metadata = {"gdn": active}
    try:
        model._forward_core(torch.ones(1, 4), torch.ones(1, 2), torch.ones(1, 2), output)
    except NativeDispatch:
        return None
    except Exception as exc:
        return f"active GDN control failed before its device boundary: {type(exc).__name__}: {exc}"
    return "active GDN control returned without dispatching computation"


def check_dummy_run(trees, torch, np):
    namespace = runner_namespace(trees[RUNNER], torch, np)
    cls = load_class(trees[RUNNER], "NPUModelRunner", namespace, RUNNER)
    failures = []
    for skip in (True, False):
        runner, gdn, attention = idle_runner(cls, namespace, torch, np)
        with contextlib.suppress(StopAfterMetadata):
            if skip:
                runner._dummy_run(1, skip_gdn_state_update=True)
            else:
                runner._dummy_run(1)
        label = "skip_gdn_state_update=True" if skip else "default dummy run"
        if len(gdn.calls) != 1 or len(attention.calls) != 1:
            failures.append(f"{label}: builders called {len(attention.calls)}/{len(gdn.calls)} times, expected 1/1")
            continue
        attn_metadata, _ = attention.calls[0]
        if attn_metadata.num_actual_tokens != 1:
            failures.append(f"{label}: ordinary attention lost the dummy token ({attn_metadata.num_actual_tokens})")
        output = gdn.outputs[0]
        starts = output.non_spec_query_start_loc
        nested = output.non_spec_decode_metadata
        nested_starts = nested.causal_conv1d.query_start_loc if nested is not None else None
        if skip:
            # Both metadata sanitation and an executed zero-token consumer are
            # valid no-op implementations. Alternate control flow is tested,
            # never inferred from source strings or AST node presence.
            safe_metadata = (
                output.num_actual_tokens == 0 and isinstance(starts, torch.Tensor)
                and isinstance(nested_starts, torch.Tensor) and not bool(starts.any())
                and not bool(nested_starts.any()) and output.num_prefills == 0
                and output.spec_sequence_masks is None
            )
            if not safe_metadata:
                alternate_failure = (check_zero_token_consumer(output, torch)
                                     if output.num_actual_tokens == 0 else "live token count remains nonzero")
                if alternate_failure is not None:
                    failures.append(
                        f"{label}: GDN output retains live work: tokens={output.num_actual_tokens}, "
                        f"query_start_loc={starts.tolist() if starts is not None else None}, "
                        f"nested_query_start_loc={nested_starts.tolist() if nested_starts is not None else None}; "
                        f"{alternate_failure}"
                    )
        elif (output.num_actual_tokens != 1 or starts is None or nested_starts is None
              or starts.tolist() != [0, 1] or nested_starts.tolist() != [0, 1]):
            failures.append(f"{label}: ordinary GDN dummy schedule changed")
    return failures


def check_idle_execute_model(trees, torch, np):
    namespace = runner_namespace(trees[RUNNER], torch, np)
    cls = load_class(trees[RUNNER], "NPUModelRunner", namespace, RUNNER)
    runner, _, _ = idle_runner(cls, namespace, torch, np)
    calls = []
    runner.__dict__.update(
        vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(enable_return_routed_experts=False),
        ),
        ascend_config=types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(
                profiling_chunk_config=types.SimpleNamespace(enabled=False, need_timing=False)
            )
        ),
        execute_model_state=None, use_async_scheduling=False, num_spec_tokens=0, requests={},
        parallel_config=types.SimpleNamespace(distributed_executor_backend="external_launcher", data_parallel_size=2),
        _update_states=lambda scheduler_output: None,
        _dummy_run=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    output = runner.execute_model(types.SimpleNamespace(
        total_num_scheduled_tokens=0, num_scheduled_tokens={},
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=[], new_token_ids=[]),
    ))
    tokens = [args[0] if args else kwargs.get("num_tokens") for args, kwargs in calls]
    if output != "EMPTY_MODEL_RUNNER_OUTPUT" or tokens != [1]:
        return [f"idle DP execute_model ran dummy calls {calls} and returned {output!r}"]
    if calls[0][1].get("skip_gdn_state_update") is not True:
        return [f"idle DP execute_model called _dummy_run{calls[0]} without skip_gdn_state_update=True"]
    return []


def check_310p_forwarding(trees, torch, np):
    forwarded = []

    class NPUModelRunner:
        def _dummy_run(self, *args, **kwargs):
            forwarded.append(kwargs)
            return "dummy"

    node = copy.deepcopy(find_class(trees[RUNNER_310P], "NPUModelRunner310"))
    method = copy.deepcopy(find_method(trees[RUNNER_310P], "NPUModelRunner310", "_dummy_run"))
    method.decorator_list = []
    node.bases, node.keywords, node.decorator_list = [ast.Name(id="NPUModelRunner", ctx=ast.Load())], [], []
    node.body = [method]
    namespace = candidate_namespace({"torch": torch, "nullcontext": contextlib.nullcontext, "NPUModelRunner": NPUModelRunner})
    load_support(trees[RUNNER_310P], namespace, RUNNER_310P)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), RUNNER_310P, "exec"), namespace)
    runner = namespace["NPUModelRunner310"].__new__(namespace["NPUModelRunner310"])
    runner.speculative_config = None
    runner.vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(use_mla=False))
    runner._dummy_run(1, skip_gdn_state_update=True)
    runner._dummy_run(1)
    if [kwargs.get("skip_gdn_state_update", False) for kwargs in forwarded] != [True, False]:
        return [f"310P _dummy_run forwarded {forwarded} instead of the skip_gdn_state_update flag"]
    return []


def main():
    failures, trees = [], {}
    error_types = []
    for relative, class_name in REQUIRED.items():
        try:
            path = ROOT / relative
            trees[relative] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            find_class(trees[relative], class_name)
        except (OSError, SyntaxError, AssertionError) as exc:
            error_types.append(type(exc).__name__)
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    if not failures:
        import numpy as np
        import torch  # absent runtime must crash as infrastructure, never score.

        for check in (check_idle_execute_model, check_dummy_run, check_310p_forwarding):
            try:
                failures.extend(check(trees, torch, np))
            except Exception as exc:  # noqa: BLE001 - any source defect is a judged failure.
                error_types.append(type(exc).__name__)
                failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(json.dumps({"reward": 0.0, "error_type": error_types[0] if error_types else "AssertionError", "failure_reason": "; ".join(failures)}))
        raise SystemExit(1)
    print(json.dumps({"reward": 1.0, "correctness_verified": True, "validation_scope": "cpu_behavior", "native_kernel_bound": None, "measured_npu_speedup": None, "speedup_ratio": None}))


if __name__ == "__main__":
    main()
