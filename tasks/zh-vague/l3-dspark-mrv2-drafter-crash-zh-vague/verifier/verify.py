# [Role] Independent CPU behavior oracle for the historical regression.
# [Invariants] Only executed tensor/state assertions establish success; helpers remain candidate code.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources; no NPU initialization or hardware performance claims.

"""Behavior oracle for the DSpark MRV2 drafter crash (vllm-ascend #14696).

Executes the workspace's DeepseekV4Model buffer allocation and the DSA DSpark
SWA index path on CPU with minimal collaborators. Sources without the target
classes fail before any runtime is needed; once they exist, a missing torch
runtime exits without a reward line and is recorded as infrastructure.
"""

import ast
import copy
import json
import os
import types
from pathlib import Path

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
MODEL = "vllm_ascend/models/deepseek_v4/model.py"
DSA = "vllm_ascend/attention/dsa_v1.py"
REQUIRED = {MODEL: "DeepseekV4Model", DSA: "AscendDSAMetadataBuilder"}
BUILTIN_DECORATORS = {"staticmethod": staticmethod, "classmethod": classmethod, "property": property}


def parse(relative):
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined at module level")


def define(nodes, namespace, filename):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


def define_function(node, namespace, filename):
    """Define one function; a default that needs an absent collaborator leaves it undefined."""
    previous = namespace.get(node.name, define_function)
    try:
        define([node], namespace, filename)
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


def keys(node, context, support=None, expanded=frozenset()):
    """Track self-state through ordinary top-level helper calls for constructor slicing."""
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, context):
            found.add(("name", sub.id))
        elif (
            isinstance(sub, ast.Attribute) and isinstance(sub.ctx, context)
            and isinstance(sub.value, ast.Name) and sub.value.id == "self"
        ):
            found.add(("self", sub.attr))
        elif isinstance(sub, ast.Call) and support is not None:
            callee = ast.unparse(sub.func)
            if callee in expanded:
                continue
            definitions = {}
            for item in support.body:
                if isinstance(item, ast.FunctionDef):
                    definitions[item.name] = item
                elif isinstance(item, ast.ClassDef):
                    for method in item.body:
                        if isinstance(method, ast.FunctionDef):
                            definitions[f"{item.name}.{method.name}"] = method
            helper = definitions.get(callee)
            if helper is None:
                continue
            bindings = dict(zip((a.arg for a in helper.args.posonlyargs + helper.args.args), sub.args))
            bindings.update((kw.arg, kw.value) for kw in sub.keywords if kw.arg)
            class BindArguments(ast.NodeTransformer):
                def visit_Name(self, name):
                    return copy.deepcopy(bindings.get(name.id, name))
            body = BindArguments().visit(ast.Module(body=copy.deepcopy(helper.body), type_ignores=[]))
            found |= keys(body, context, support, expanded | {callee})
    return found


def init_slice(init, targets, provided, support):
    """Backward slice of __init__ statements that produce the target attributes."""
    hits = [index for index, stmt in enumerate(init.body) if keys(stmt, ast.Store, support) & targets]
    if not hits:
        raise AssertionError(f"__init__ never assigns {sorted(attr for _, attr in targets)}")
    needed, picked = set(targets), []
    for stmt in reversed(init.body[: hits[-1] + 1]):
        if keys(stmt, ast.Store, support) & needed:
            picked.append(stmt)
            needed |= keys(stmt, ast.Load, support) - provided
    function = copy.deepcopy(init)
    function.name, function.decorator_list, function.returns = "_init_slice", [], None
    function.body = picked[::-1]
    return function


class Speculative:
    def __init__(self, method, eagle, draft_model):
        self.method, self._eagle, self._draft_model = method, eagle, draft_model
        self.num_speculative_tokens = 3

    def use_eagle(self):
        return self._eagle

    def uses_draft_model(self):
        return self._draft_model


def check_mtp_hidden_buffer(trees):
    import torch

    tree = trees[MODEL]
    init = next(
        (item for item in find_class(tree, "DeepseekV4Model").body
         if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
        None,
    )
    assert init is not None, "DeepseekV4Model has no __init__"
    namespace = candidate_namespace({"torch": torch, "nn": torch.nn})
    model_cls = load_class(tree, "DeepseekV4Model", namespace, MODEL)
    provided = {("name", "self"), ("name", "vllm_config"), ("name", "prefix"), ("self", "device")}
    build_buffer = define_function(
        init_slice(init, {("self", "_mtp_hidden_buffer")}, provided, tree), namespace, MODEL
    )
    assert build_buffer is not None, "_mtp_hidden_buffer statements reference unavailable names"
    tokens, hc_mult, hidden = 5, 2, 8
    cases = [
        ("PP last rank + DSpark (use_eagle)", True, Speculative("dspark", True, False), True),
        ("PP last rank + draft model", True, Speculative("draft_model", False, True), True),
        ("PP last rank + MTP", True, Speculative("mtp", True, False), True),
        ("non-last PP rank + DSpark", False, Speculative("dspark", True, False), False),
        ("PP last rank without speculative decoding", True, None, False),
        ("PP last rank + ngram (no target hidden states)", True, Speculative("ngram", False, False), False),
    ]
    failures = []
    for label, last_rank, speculative, allocated in cases:
        group = types.SimpleNamespace(
            is_last_rank=last_rank, is_first_rank=not last_rank,
            world_size=2, rank_in_group=1 if last_rank else 0,
        )
        namespace["get_pp_group"] = lambda group=group: group
        config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    hc_mult=hc_mult, hidden_size=hidden, vocab_size=32,
                    rms_norm_eps=1e-6, hc_eps=1e-6, num_hidden_layers=1,
                ),
                dtype=torch.float32,
            ),
            scheduler_config=types.SimpleNamespace(max_num_batched_tokens=tokens),
            speculative_config=speculative,
            quant_config=None,
        )
        model = model_cls.__new__(model_cls)
        model.device = torch.device("cpu")
        build_buffer(model, vllm_config=config, prefix="model")
        buffer = getattr(model, "_mtp_hidden_buffer", "missing")
        if allocated:
            ok = (
                isinstance(buffer, torch.Tensor)
                and tuple(buffer.shape) == (tokens, hc_mult * hidden)
                and buffer.dtype == torch.float32
            )
        else:
            ok = buffer is None
        if not ok:
            expected = f"tensor[{tokens}, {hc_mult * hidden}]" if allocated else "None"
            got = tuple(buffer.shape) if isinstance(buffer, torch.Tensor) else repr(buffer)
            failures.append(f"_mtp_hidden_buffer for {label}: expected {expected}, got {got}")
    return failures


def expected_swa_indices(block_table, seq_lens, query_lens, window, block_size, width):
    import torch

    rows = []
    for row, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        start = max(seq_len - query_len - window, 0)
        visible = seq_len - start
        slots = [
            int(block_table[row][(start + col) // block_size]) * block_size + (start + col) % block_size
            if col < visible else -1
            for col in range(width)
        ]
        rows.extend([slots] * query_len)
    return torch.tensor(rows, dtype=torch.int32).unsqueeze(1)


class Unreachable:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attribute):
        raise AssertionError(f"unexpected {self.name}.{attribute} in the cached DSpark decode path")

    def __call__(self, *args, **kwargs):
        raise AssertionError(f"unexpected call to {self.name} in the cached DSpark decode path")


def check_dspark_swa_indices(trees):
    import torch

    tree = trees[DSA]
    namespace = candidate_namespace({"torch": torch})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (function := define_function(node, namespace, DSA)):
            namespace[node.name] = function
    namespace.update(
        AscendDSAReqMetadata=types.SimpleNamespace,
        DeviceOperator=Unreachable("DeviceOperator"),
        get_full_cos_and_sin_dsa=Unreachable("get_full_cos_and_sin_dsa"),
    )
    builder_cls = load_class(tree, "AscendDSAMetadataBuilder", namespace, DSA)
    window, draft_tokens, storage_block_size = 6, 3, 4
    table = [[7, 3, 9, 1, 0, 0, 0, 0], [2, 8, 5, 6, 4, 0, 0, 0]]
    seq_lens, query_lens = [9, 14], [draft_tokens, draft_tokens]
    builder = builder_cls.__new__(builder_cls)
    model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(sliding_window=window))
    speculative = types.SimpleNamespace(num_speculative_tokens=draft_tokens)
    # The builder owns only logical and storage block sizes; DSpark slots live in storage blocks.
    builder.__dict__.update(
        common_ratio_to_sas_metadata={"c1": "cached"},
        num_actual_tokens=6, num_prefills=0, num_decodes=2, num_decode_tokens=6, compressor_ratio=1,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.tensor(table, dtype=torch.int32),
        start_pos_prefill=torch.zeros(4, dtype=torch.int32),
        slot_mapping=torch.arange(16, dtype=torch.int32),
        model_config=model_config, speculative_config=speculative,
        vllm_config=types.SimpleNamespace(model_config=model_config, speculative_config=speculative),
        storage_block_size=storage_block_size, logical_block_size=16,
        _build_sas_metadata=lambda *args, **kwargs: "sas",
        _build_qli_metadata=lambda *args, **kwargs: "qli",
    )
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)
    common = types.SimpleNamespace(
        num_reqs=2, query_start_loc=query_start_loc, query_start_loc_cpu=query_start_loc.clone(),
        causal=False,
    )
    metadata = builder.build_req_metadata(
        common, torch.tensor(seq_lens, dtype=torch.int32), None, torch.zeros(6, 1), torch.zeros(6, 1),
    )
    width = ((window + draft_tokens + 127) // 128) * 128
    expected = expected_swa_indices(table, seq_lens, query_lens, window, storage_block_size, width)
    indices = getattr(metadata, "dspark_swa_indices", None)
    if not isinstance(indices, torch.Tensor) or indices.shape != expected.shape:
        got = tuple(indices.shape) if isinstance(indices, torch.Tensor) else repr(indices)
        return [f"dspark_swa_indices shape {got} != {tuple(expected.shape)}"]
    if not torch.equal(indices.to(torch.int32), expected):
        return ["dspark_swa_indices do not address the storage-block KV slots"]
    return []


def main():
    failures, trees = [], {}
    error_types = []
    for relative, class_name in REQUIRED.items():
        try:
            trees[relative] = parse(relative)
            find_class(trees[relative], class_name)
        except (OSError, SyntaxError, AssertionError) as exc:
            error_types.append(type(exc).__name__)
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    if not failures:
        import torch  # noqa: F401 - absent runtime must crash as infrastructure, never score.

        for check in (check_mtp_hidden_buffer, check_dspark_swa_indices):
            try:
                failures.extend(check(trees))
            except Exception as exc:  # noqa: BLE001 - any source defect is a judged failure.
                error_types.append(type(exc).__name__)
                failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(json.dumps({"reward": 0.0, "error_type": error_types[0] if error_types else "AssertionError", "failure_reason": "; ".join(failures)}))
        raise SystemExit(1)
    print(json.dumps({"reward": 1.0, "correctness_verified": True, "validation_scope": "cpu_behavior", "native_kernel_bound": None, "measured_npu_speedup": None, "speedup_ratio": None}))


if __name__ == "__main__":
    main()
