# [Role] Independent CPU behavior oracle for the historical regression.
# [Invariants] Only executed tensor/state assertions establish success; helpers remain candidate code.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources; no NPU initialization or hardware performance claims.

"""Behavior oracle for SFA multi-step MTP sequence-length buffers (vllm-ascend #14236).

Builds main and draft-step metadata through the workspace's AscendSFAMetadataBuilder
on CPU. Draft step k must keep its query/key lengths in the persistent per-step
buffers, so building step k+1 cannot rewrite metadata already handed out for step k.
Sources without the builder fail before any runtime is needed; once it exists, a
missing torch runtime exits without a reward line and is recorded as infrastructure.
"""

import ast
import copy
import json
import os
import types
from pathlib import Path

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
SFA = "vllm_ascend/attention/sfa_v1.py"
BUILDER = "AscendSFAMetadataBuilder"
BUILTIN_DECORATORS = {"staticmethod": staticmethod, "classmethod": classmethod, "property": property}


def find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined at module level")


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


def check_draft_steps_keep_their_lengths(tree):
    import torch

    namespace = candidate_namespace({"torch": torch})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (function := define_function(node, namespace, SFA)):
            namespace[node.name] = function
    namespace.update(
        get_cos_and_sin_mla=lambda positions, use_cache=True: (
            torch.zeros(positions.shape[0], 1), torch.zeros(positions.shape[0], 1)
        ),
        get_ascend_config=lambda: types.SimpleNamespace(c8_enable_reshape_optim=False),
    )
    builder_cls = load_class(tree, BUILDER, namespace, SFA)
    builder = builder_cls.__new__(builder_cls)
    max_num_reqs, draft_steps = 4, 2
    lengths = max_num_reqs * (draft_steps + 1) + 1
    builder.__dict__.update(
        kernel_block_size=16,
        speculative_config=types.SimpleNamespace(num_speculative_tokens=draft_steps),
        actual_seq_lengths_query=torch.zeros(max_num_reqs + 1, dtype=torch.int32),
        actual_seq_lengths_key=torch.zeros(max_num_reqs + 1, dtype=torch.int32),
        spec_actual_seq_lengths_query=[torch.zeros(lengths, dtype=torch.int32) for _ in range(draft_steps)],
        spec_actual_seq_lengths_key=[torch.zeros(lengths, dtype=torch.int32) for _ in range(draft_steps)],
        metadata_cls=types.SimpleNamespace,
        model_config=types.SimpleNamespace(get_head_size=lambda: 64),
        attn_mask_builder=types.SimpleNamespace(get_attention_mask=lambda *args, **kwargs: None),
    )

    def common(query_start_loc, seq_lens):
        tokens = query_start_loc[-1]
        return types.SimpleNamespace(
            num_reqs=len(seq_lens), num_actual_tokens=tokens, num_input_tokens=tokens,
            block_table_tensor=torch.zeros(len(seq_lens), 4, dtype=torch.int32),
            slot_mapping=torch.arange(tokens, dtype=torch.int32), positions=torch.arange(tokens),
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32), _seq_lens_cpu=None, seq_lens_cpu=None,
            causal=True, attn_state="SpecDecoding", group_len=None, group_key_idx=None,
            group_key_cache_idx=None,
        )

    query_buffers, key_buffers = builder.spec_actual_seq_lengths_query, builder.spec_actual_seq_lengths_key
    steps = {
        None: ([0, 1, 2], [10, 20], builder.actual_seq_lengths_query, builder.actual_seq_lengths_key),
        1: ([0, 2, 4], [11, 21], query_buffers[0], key_buffers[0]),
        2: ([0, 3, 6], [12, 22], query_buffers[1], key_buffers[1]),
    }
    built = {}
    for draft_index, (query_start_loc, seq_lens, _, _) in steps.items():
        metadata = common(query_start_loc, seq_lens)
        built[draft_index] = (
            builder.build(0, metadata) if draft_index is None
            else builder.build_for_drafting(metadata, draft_index=draft_index)
        )
    failures = []
    for draft_index, (query_start_loc, seq_lens, query_buffer, key_buffer) in steps.items():
        label = "main build" if draft_index is None else f"draft_index={draft_index}"
        result = built[draft_index]
        if result.cum_query_lens.tolist() != query_start_loc[1:] or result.seq_lens.tolist() != seq_lens:
            failures.append(
                f"{label} lengths were overwritten after later builds: cum_query_lens="
                f"{result.cum_query_lens.tolist()} seq_lens={result.seq_lens.tolist()}"
            )
        # Captured graphs replay persistent addresses; each step must own its buffer.
        if (
            result.cum_query_lens.data_ptr() != query_buffer.data_ptr()
            or result.seq_lens.data_ptr() != key_buffer.data_ptr()
        ):
            failures.append(f"{label} does not use its persistent sequence-length buffers")
    return failures


def main():
    failures = []
    error_types = []
    try:
        path = ROOT / SFA
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        find_class(tree, BUILDER)
    except (OSError, SyntaxError, AssertionError) as exc:
        error_types.append(type(exc).__name__)
        failures.append(f"{SFA}: {type(exc).__name__}: {exc}")
    if not failures:
        import torch  # noqa: F401 - absent runtime must crash as infrastructure, never score.

        try:
            failures.extend(check_draft_steps_keep_their_lengths(tree))
        except Exception as exc:  # noqa: BLE001 - any source defect is a judged failure.
            error_types.append(type(exc).__name__)
            failures.append(f"{type(exc).__name__}: {exc}")
    if failures:
        print(json.dumps({"reward": 0.0, "error_type": error_types[0] if error_types else "AssertionError", "failure_reason": "; ".join(failures)}))
        raise SystemExit(1)
    print(json.dumps({"reward": 1.0, "correctness_verified": True, "validation_scope": "cpu_behavior", "native_kernel_bound": None, "measured_npu_speedup": None, "speedup_ratio": None}))


if __name__ == "__main__":
    main()
