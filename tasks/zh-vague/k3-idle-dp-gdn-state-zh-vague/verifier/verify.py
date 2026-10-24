"""Behavior oracle for idle-DP GDN state protection (vllm-ascend #15655).

Runs the workspace NPUModelRunner on CPU with minimal collaborators:
execute_model on an idle external-launcher DP rank, then _dummy_run through
its real _build_attention_metadata call (stopping at the dummy forward). The
GDN builder must receive a no-op schedule (zero tokens, zeroed query starts,
no prefills) while ordinary attention still sees the dummy token, and the 310P
runner must forward the flag. GDN-internal tensors produced inside vLLM's own
builder are outside this CPU oracle. Sources without the runner classes fail
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


def load_class(tree, name, namespace, filename):
    """Rebuild a class from its methods; only methods the oracle calls must be definable."""
    cls = type(name, (), {})
    for item in find_class(tree, name).body:
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


class RecordingBuilder:
    def __init__(self):
        self.calls = []

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.calls.append((common_attn_metadata, kwargs))
        return types.SimpleNamespace(num_prefills=0, num_decodes=1, num_spec_decodes=0)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)


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
    namespace = {
        "torch": torch, "np": np, "replace": dataclasses.replace, "copy": copy.copy,
        "deepcopy": copy.deepcopy, "nullcontext": contextlib.nullcontext, "time": __import__("time"),
        "cdiv": lambda a, b: -(-a // b),
        "CUDAGraphMode": types.SimpleNamespace(NONE="NONE", PIECEWISE="PIECEWISE", FULL="FULL"),
        "AscendAttentionState": types.SimpleNamespace(
            DecodeOnly="DecodeOnly", SpecDecoding="SpecDecoding", ChunkedPrefill="ChunkedPrefill",
        ),
        "AscendCommonAttentionMetadata": common_metadata_class(
            find_method(tree, "NPUModelRunner", "_build_attention_metadata")
        ),
        "vllm_version_is": lambda version: False,
        "get_pp_group": lambda: types.SimpleNamespace(world_size=1, is_last_rank=True, is_first_rank=True),
        "has_kv_transfer_group": lambda: False,
        "has_ec_transfer": lambda: False,
        "record_function_or_nullcontext": lambda name: contextlib.nullcontext(),
        "EMPTY_MODEL_RUNNER_OUTPUT": "EMPTY_MODEL_RUNNER_OUTPUT",
    }
    for name in (
        "GDNAttentionMetadataBuilder", "AscendDSAMetadataBuilder", "AscendDSACPMetadataBuilder",
        "AscendSFADCPMetadataBuilder", "EncoderOnlyAttentionSpec",
    ):
        namespace[name] = type(name, (RecordingBuilder,) if name.endswith("Builder") else (), {})
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
        (gdn_metadata, gdn_args), (attn_metadata, _) = gdn.calls[0], attention.calls[0]
        if attn_metadata.num_actual_tokens != 1:
            failures.append(f"{label}: ordinary attention lost the dummy token ({attn_metadata.num_actual_tokens})")
        starts = gdn_metadata.query_start_loc.tolist()
        if skip:
            prefilling = gdn_metadata.is_prefilling
            if (
                gdn_metadata.num_actual_tokens != 0 or gdn_metadata.max_query_len != 0
                or starts != [0, 0] or gdn_args
                or (prefilling is not None and bool(torch.as_tensor(prefilling).any()))
            ):
                failures.append(
                    f"{label}: GDN builder received tokens={gdn_metadata.num_actual_tokens}, "
                    f"max_query_len={gdn_metadata.max_query_len}, query_start_loc={starts}, "
                    f"extra={sorted(gdn_args)}, is_prefilling={prefilling}; expected a no-op schedule"
                )
        elif gdn_metadata.num_actual_tokens != 1 or starts != [0, 1]:
            failures.append(
                f"{label}: GDN builder changed the ordinary dummy schedule "
                f"(tokens={gdn_metadata.num_actual_tokens}, query_start_loc={starts})"
            )
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
    namespace = {"torch": torch, "nullcontext": contextlib.nullcontext, "NPUModelRunner": NPUModelRunner}
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
    for relative, class_name in REQUIRED.items():
        try:
            path = ROOT / relative
            trees[relative] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            find_class(trees[relative], class_name)
        except (OSError, SyntaxError, AssertionError) as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    if not failures:
        import numpy as np
        import torch  # absent runtime must crash as infrastructure, never score.

        for check in (check_idle_execute_model, check_dummy_run, check_310p_forwarding):
            try:
                failures.extend(check(trees, torch, np))
            except Exception as exc:  # noqa: BLE001 - any source defect is a judged failure.
                failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(json.dumps({"reward": 0.0, "failure_reason": "; ".join(failures)}))
        raise SystemExit(1)
    print(json.dumps({"reward": 1.0}))


if __name__ == "__main__":
    main()
