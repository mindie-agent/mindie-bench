# [Role] Independent CPU behavior oracle for the historical regression.
# [Invariants] Only executed tensor/state assertions establish success; helpers remain candidate code.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources; no NPU initialization or hardware performance claims.

"""Behavior oracle for MTP draft slot-mapping capacity (vllm-ascend #14024).

Constructs the workspace BlockTable on CPU and maps the worst-case MTP draft
expansion (max_num_batched_tokens + (num_speculative_tokens - 1) * max_num_reqs
slots) through compute_slot_mapping_draft, for DCP 1 and DCP 4. The proposer's
slot_mapping_group buffers must receive that slot mapping unchanged while its
seq_lens/query_start_loc groups keep the 2 * max_num_reqs graph padding. Sources
without the classes fail before any runtime is needed; once they exist, a missing
torch runtime exits without a reward line and is recorded as infrastructure.
"""

import ast
import copy
import json
import os
import types
from pathlib import Path

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", "."))
BLOCK_TABLE = "vllm_ascend/worker/block_table.py"
PROPOSER = "vllm_ascend/spec_decode/llm_base_proposer.py"
REQUIRED = {BLOCK_TABLE: "BlockTable", PROPOSER: "AscendSpecDecodeBaseProposer"}
BUILTIN_DECORATORS = {"staticmethod": staticmethod, "classmethod": classmethod, "property": property}
TOKENS, REQUESTS, BLOCK = 20, 4, 16


def parse(relative):
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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


def required_slots(speculative_tokens):
    return TOKENS + max(speculative_tokens - 1, 0) * REQUESTS


def block_table_factory(tree, dcp_world_size):
    import numpy as np
    import torch

    class CpuGpuBuffer:
        def __init__(self, *size, dtype, device, pin_memory):
            self.cpu = torch.zeros(*size, dtype=dtype)
            self.gpu = torch.zeros(*size, dtype=dtype)
            self.np = self.cpu.numpy()

        def copy_to_gpu(self, n=None):
            if n is None:
                return self.gpu.copy_(self.cpu)
            self.gpu[:n].copy_(self.cpu[:n])
            return self.gpu[:n]

    namespace = candidate_namespace({
        "np": np, "torch": torch, "CpuGpuBuffer": CpuGpuBuffer, "PAD_SLOT_ID": -1,
        "cdiv": lambda a, b: -(-a // b),
        "get_dcp_group": lambda: types.SimpleNamespace(world_size=dcp_world_size, rank_in_group=0),
        "get_decode_context_model_parallel_world_size": lambda: dcp_world_size,
        "UniformTypeKVCacheSpecs": type("UniformTypeKVCacheSpecs", (), {}),
        "MambaSpec": type("MambaSpec", (), {}),
    })
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (function := define_function(node, namespace, BLOCK_TABLE)):
            namespace[node.name] = function
    return load_class(tree, "BlockTable", namespace, BLOCK_TABLE)


def check_block_table(tree):
    import numpy as np

    failures = []
    for speculative_tokens in (0, 1, 2, 5):
        table = block_table_factory(tree, 1)(
            block_size=BLOCK, max_num_reqs=REQUESTS, max_num_blocks_per_req=8,
            max_num_batched_tokens=TOKENS, pin_memory=False, device="cpu", kernel_sizes=[BLOCK],
            num_speculative_tokens=speculative_tokens,
        )
        capacity = table.slot_mapping.cpu.shape[0]
        if capacity != required_slots(speculative_tokens):
            failures.append(
                f"BlockTable slot_mapping holds {capacity} slots for num_speculative_tokens="
                f"{speculative_tokens}; expected {required_slots(speculative_tokens)}"
            )
    draft_slots = required_slots(5)
    rows = [[3, 7], [5, 2], [9, 4], [6, 8]]
    req_indices = np.repeat(np.arange(REQUESTS, dtype=np.int64), draft_slots // REQUESTS)
    positions = np.tile(np.arange(12, 12 + draft_slots // REQUESTS, dtype=np.int64), REQUESTS)
    expected = [rows[r][p // BLOCK] * BLOCK + p % BLOCK for r, p in zip(req_indices, positions)]
    for dcp_world_size in (1, 4):
        table = block_table_factory(tree, dcp_world_size)(
            block_size=BLOCK, max_num_reqs=REQUESTS, max_num_blocks_per_req=8,
            max_num_batched_tokens=TOKENS, pin_memory=False, device="cpu", kernel_sizes=[BLOCK],
            num_speculative_tokens=5,
        )
        for row, block_ids in enumerate(rows):
            table.add_row(block_ids, row)
        try:
            table.compute_slot_mapping_draft(req_indices.copy(), positions.copy())
        except (ValueError, RuntimeError, IndexError) as exc:
            failures.append(
                f"DCP{dcp_world_size} MTP5 draft slot mapping of {draft_slots} slots overflowed: {exc}"
            )
            continue
        if dcp_world_size == 1 and table.slot_mapping.np[:draft_slots].tolist() != expected:
            failures.append("DCP1 draft slot mapping values do not address the request blocks")
    return failures


def check_proposer(tree, block_tree):
    import torch

    init = next(
        (item for item in find_class(tree, "AscendSpecDecodeBaseProposer").body
         if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
        None,
    )
    assert init is not None, "AscendSpecDecodeBaseProposer has no __init__"
    targets = {("self", "slot_mapping_group"), ("self", "seq_lens_group"), ("self", "query_start_loc_group")}
    provided = {
        ("name", "self"), ("name", "vllm_config"), ("name", "device"), ("name", "runner"),
        ("name", "pass_hidden_states_to_model"), ("self", "runner"), ("self", "num_speculative_tokens"),
        ("self", "vllm_config"), ("self", "speculative_config"), ("self", "device"),
    }
    namespace = candidate_namespace({"torch": torch})
    load_support(tree, namespace, PROPOSER)
    allocate = define_function(init_slice(init, targets, provided, tree), namespace, PROPOSER)
    assert allocate is not None, "proposer buffer statements reference unavailable names"
    failures = []
    for speculative_tokens in (1, 5):
        runner = types.SimpleNamespace(max_num_tokens=TOKENS, max_num_reqs=REQUESTS, pin_memory=False)
        proposer = types.SimpleNamespace(
            runner=runner, num_speculative_tokens=speculative_tokens, device=torch.device("cpu"),
            vllm_config=types.SimpleNamespace(
                scheduler_config=types.SimpleNamespace(max_num_batched_tokens=TOKENS)
            ),
            speculative_config=types.SimpleNamespace(num_speculative_tokens=speculative_tokens),
        )
        allocate(
            proposer, vllm_config=proposer.vllm_config, device=torch.device("cpu"),
            pass_hidden_states_to_model=True, runner=runner,
        )
        groups = {
            name: getattr(proposer, name, None)
            for name in ("slot_mapping_group", "seq_lens_group", "query_start_loc_group")
        }
        for name, group in groups.items():
            expected = required_slots(speculative_tokens) if name == "slot_mapping_group" else TOKENS + 2 * REQUESTS
            sizes = [tensor.shape[0] for tensor in group or []]
            if len(sizes) != speculative_tokens or set(sizes) != {expected}:
                failures.append(
                    f"{name} for num_speculative_tokens={speculative_tokens} has sizes {sizes}; "
                    f"expected {speculative_tokens} x {expected}"
                )
    table = block_table_factory(block_tree, 1)(
        block_size=BLOCK, max_num_reqs=REQUESTS, max_num_blocks_per_req=8,
        max_num_batched_tokens=TOKENS, pin_memory=False, device="cpu", kernel_sizes=[BLOCK],
        num_speculative_tokens=5,
    )
    slot_mapping = table.slot_mapping.gpu
    target = (proposer.slot_mapping_group or [None])[0]
    # Full-graph warmup copies the whole BlockTable slot mapping into each draft-step buffer.
    if not isinstance(target, torch.Tensor) or target.shape[0] != slot_mapping.shape[0]:
        size = target.shape[0] if isinstance(target, torch.Tensor) else None
        failures.append(
            f"BlockTable slot_mapping ({slot_mapping.shape[0]}) and proposer slot_mapping_group ({size}) "
            "differ for MTP5"
        )
    return failures


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

        for name, check in (
            ("BlockTable", lambda: check_block_table(trees[BLOCK_TABLE])),
            ("AscendSpecDecodeBaseProposer", lambda: check_proposer(trees[PROPOSER], trees[BLOCK_TABLE])),
        ):
            try:
                failures.extend(check())
            except Exception as exc:  # noqa: BLE001 - any source defect is a judged failure.
                error_types.append(type(exc).__name__)
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        print(json.dumps({"reward": 0.0, "error_type": error_types[0] if error_types else "AssertionError", "failure_reason": "; ".join(failures)}))
        raise SystemExit(1)
    print(json.dumps({"reward": 1.0, "correctness_verified": True, "validation_scope": "cpu_behavior", "native_kernel_bound": None, "measured_npu_speedup": None, "speedup_ratio": None}))


if __name__ == "__main__":
    main()
