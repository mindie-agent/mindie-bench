# [Role] Independent CPU behavior oracle for SFA MTP decode metadata and layout invariants.
# [Invariants] Only executed tensor/state assertions establish success; log fixtures must stay intact.
# [Timeouts & Locks] Parent verifier bounds the isolated process; no locks or background jobs.
# [Side-Effect Policy] Reads candidate sources; no NPU initialization or hardware performance claims.

"""Behavioral oracle for SFA MTP speculative decode metadata and TND layout alignment.

Verifies that:
1. Workspace CANN plog fixtures under logs/plog_tp8_crash/ are untouched.
2. AscendSFAMetadataBuilder.build and build_for_drafting keep actual_seq_lengths_key
   (seq_lens) aligned per request (shape [num_reqs] == block_table.shape[0]) across
   both standard single-token decode and MTP multi-token speculative decode steps.
3. AscendSFAImpl._execute_sparse_flash_attention_process preserves 3D TND layout
   (ql_nope: [T, N, kv_lora_rank], q_pe: [T, N, qk_rope_head_dim]) and matching
   per-request actual_seq_lengths_query / actual_seq_lengths_key shapes across
   variable batch sizes, speculative token counts, and ACLGraph padded token buckets.
"""

import ast
import copy
import hashlib
import json
import os
import subprocess
import types
from pathlib import Path

ROOT = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", ".")).resolve()
SFA = "vllm_ascend/attention/sfa_v1.py"
ALLOWED_MODIFIED = {SFA}
PROTECTED_LOGS = {
    "logs/plog_tp8_crash/plog-rank0.log": "8499caa84afeac66c1568b3ff998fe9debcb0099a40d41a5fd0d31c84439f591",
    "logs/plog_tp8_crash/plog-rank1.log": "df10af8ba03c4d4ea931c4785b5438b061aa69849698e1b46158627445100163",
    "logs/plog_tp8_crash/plog-rank2.log": "37a0119f41ab8732907eb60f1d3606e54fb8ced0daadd0b45be0612aef41a333",
    "logs/plog_tp8_crash/plog-rank3.log": "b7fb9696f10c0f23c431ae2a269bff6dfebdb2712a2bc33b1331216e208b5d1f",
    "logs/plog_tp8_crash/plog-rank4.log": "6515040a41d615be380441ad6e25760bd7d98611f7a7b6166e049de5f9ea2a04",
    "logs/plog_tp8_crash/plog-rank5.log": "c066e7b16f7c41aa50f29b622790ebf9c06ad5671345180c5dcea044970d3eee",
    "logs/plog_tp8_crash/plog-rank6.log": "2f9990085551fba4eb63d37d641cbb771348eed015e8541235c19393e923a934",
    "logs/plog_tp8_crash/plog-rank7.log": "973ef2b1b9947af3bd5e743d5a285a57c68427abcee15a828cf01fd44cbff0ff",
}
BUILTIN_DECORATORS = {"staticmethod": staticmethod, "classmethod": classmethod, "property": property}


def check_workspace_integrity() -> list[str]:
    failures = []
    for rel, expected_sha in PROTECTED_LOGS.items():
        p = ROOT / rel
        if not p.is_file():
            failures.append(f"protected log file missing: {rel}")
            continue
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            failures.append(f"protected log file was modified: {rel}")
    if (ROOT / ".git").exists():
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if proc.returncode == 0:
                for changed in proc.stdout.splitlines():
                    changed = changed.strip()
                    if changed and changed not in ALLOWED_MODIFIED:
                        failures.append(f"disallowed file modified: {changed}")
        except Exception:
            pass
    return failures


def find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined at module level")


def define_function(node, namespace, filename):
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
    import sys

    module = types.ModuleType(f"_behavior_candidate_{len(sys.modules)}")
    module.__dict__.update(values)
    sys.modules[module.__name__] = module
    return module.__dict__


def load_support(tree, namespace, filename):
    namespace.setdefault("__name__", "__main__")
    allowed = {
        "collections", "contextlib", "dataclasses", "enum", "functools", "itertools",
        "math", "typing", "types", "numpy", "torch", "copy", "operator", "numbers",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases = [
                alias for alias in node.names
                if alias.name.split(".")[0] in allowed
                and (alias.asname or alias.name.split(".")[0]) not in namespace
            ]
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


def check_sfa_mtp_metadata_and_fia_layout(tree) -> list[str]:
    import torch

    class FakeAttentionState:
        ChunkedPrefill = "ChunkedPrefill"
        DecodeOnly = "DecodeOnly"
        SpecDecoding = "SpecDecoding"

    captured_calls = []

    class FakeDeviceOperator:
        @staticmethod
        def execute_sparse_flash_attention_process(
            sfa_impl,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table=None,
        ):
            if block_table is None:
                block_table = attn_metadata.block_table
            batch_size = int(block_table.shape[0])
            if int(actual_seq_lengths_key.shape[0]) != batch_size:
                raise RuntimeError(
                    f"EZ9999 / 507015: FusedInferAttentionScore tiling error (errcode 264): "
                    f"actual_seq_lengths_kv length ({int(actual_seq_lengths_key.shape[0])}) != "
                    f"batch_size ({batch_size})"
                )
            if int(actual_seq_lengths_query.shape[0]) != batch_size:
                raise RuntimeError(
                    f"actual_seq_lengths_query length ({int(actual_seq_lengths_query.shape[0])}) != "
                    f"batch_size ({batch_size})"
                )
            if ql_nope.ndim != 3 or tuple(ql_nope.shape[1:]) != (sfa_impl.local_num_heads, sfa_impl.kv_lora_rank):
                raise RuntimeError(
                    f"TND layout mismatch for ql_nope: expected [T, {sfa_impl.local_num_heads}, "
                    f"{sfa_impl.kv_lora_rank}], got {tuple(ql_nope.shape)}"
                )
            if q_pe.ndim != 3 or tuple(q_pe.shape[1:]) != (sfa_impl.local_num_heads, sfa_impl.qk_rope_head_dim):
                raise RuntimeError(
                    f"TND layout mismatch for q_pe: expected [T, {sfa_impl.local_num_heads}, "
                    f"{sfa_impl.qk_rope_head_dim}], got {tuple(q_pe.shape)}"
                )
            captured_calls.append({
                "ql_nope_shape": tuple(ql_nope.shape),
                "q_pe_shape": tuple(q_pe.shape),
                "query_lens": actual_seq_lengths_query.tolist(),
                "key_lens": actual_seq_lengths_key.tolist(),
                "batch_size": batch_size,
            })
            return torch.zeros_like(ql_nope)

    namespace = candidate_namespace({
        "torch": torch,
        "AscendAttentionState": FakeAttentionState,
        "DeviceOperator": FakeDeviceOperator,
    })
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (function := define_function(node, namespace, SFA)):
            namespace[node.name] = function
    namespace.update(
        get_cos_and_sin_mla=lambda positions, use_cache=True: (
            torch.zeros(positions.shape[0], 64), torch.zeros(positions.shape[0], 64)
        ),
        get_ascend_config=lambda: types.SimpleNamespace(c8_enable_reshape_optim=False),
    )
    builder_cls = load_class(tree, "AscendSFAMetadataBuilder", namespace, SFA)
    impl_cls = load_class(tree, "AscendSFAImpl", namespace, SFA)

    failures = []
    scenarios = [
        # (label, max_reqs, spec_k, query_start_loc, seq_lens, num_input_tokens, attn_state, draft_idx)
        (
            "standard_decode_bs4",
            8,
            2,
            [0, 1, 2, 3, 4],
            [32, 64, 19, 128],
            4,
            FakeAttentionState.DecodeOnly,
            None,
        ),
        (
            "mtp_target_verify_bs4_k2",
            8,
            2,
            [0, 3, 6, 9, 12],
            [35, 67, 22, 131],
            12,
            FakeAttentionState.SpecDecoding,
            None,
        ),
        (
            "mtp_draft1_bs4_k2_padded16",
            8,
            2,
            [0, 2, 4, 6, 8],
            [36, 68, 23, 132],
            16,
            FakeAttentionState.SpecDecoding,
            1,
        ),
        (
            "mtp_ragged_bs3_k3_padded16",
            8,
            3,
            [0, 4, 7, 11],
            [40, 18, 95],
            16,
            FakeAttentionState.SpecDecoding,
            2,
        ),
    ]

    for label, max_reqs, spec_k, q_start, seq_lens, num_input_tokens, state, draft_idx in scenarios:
        builder = builder_cls.__new__(builder_cls)
        lengths_cap = max_reqs * (spec_k + 1) + 1
        builder.__dict__.update(
            kernel_block_size=128,
            speculative_config=types.SimpleNamespace(num_speculative_tokens=spec_k),
            actual_seq_lengths_query=torch.zeros(lengths_cap, dtype=torch.int32),
            actual_seq_lengths_key=torch.zeros(lengths_cap, dtype=torch.int32),
            spec_actual_seq_lengths_query=[torch.zeros(lengths_cap, dtype=torch.int32) for _ in range(spec_k)],
            spec_actual_seq_lengths_key=[torch.zeros(lengths_cap, dtype=torch.int32) for _ in range(spec_k)],
            metadata_cls=types.SimpleNamespace,
            model_config=types.SimpleNamespace(get_head_size=lambda: 128),
            attn_mask_builder=types.SimpleNamespace(get_attention_mask=lambda *args, **kwargs: None),
        )
        num_reqs = len(seq_lens)
        num_actual_tokens = q_start[-1]
        common = types.SimpleNamespace(
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            block_table_tensor=torch.arange(num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4),
            slot_mapping=torch.arange(num_input_tokens, dtype=torch.int32),
            positions=torch.arange(num_input_tokens, dtype=torch.int64),
            query_start_loc=torch.tensor(q_start, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            _seq_lens_cpu=None,
            seq_lens_cpu=None,
            causal=True,
            attn_state=state,
            group_len=None,
            group_key_idx=None,
            group_key_cache_idx=None,
        )
        built = (
            builder.build(0, common)
            if draft_idx is None
            else builder.build_for_drafting(common, draft_index=draft_idx)
        )
        if tuple(built.seq_lens.shape) != (num_reqs,) or built.seq_lens.tolist() != seq_lens:
            failures.append(
                f"{label}: expected seq_lens shape ({num_reqs},) with values {seq_lens}, "
                f"got shape {tuple(built.seq_lens.shape)} values {built.seq_lens.tolist()}"
            )
            continue
        if tuple(built.cum_query_lens.shape) != (num_reqs,) or built.cum_query_lens.tolist() != q_start[1:]:
            failures.append(
                f"{label}: expected cum_query_lens {q_start[1:]}, got {built.cum_query_lens.tolist()}"
            )
            continue

        impl = impl_cls.__new__(impl_cls)
        impl.local_num_heads = 16
        impl.kv_lora_rank = 512
        impl.qk_rope_head_dim = 64
        impl.scale = 0.125
        ql_nope = torch.randn(num_input_tokens, impl.local_num_heads, impl.kv_lora_rank)
        q_pe = torch.randn(num_input_tokens, impl.local_num_heads, impl.qk_rope_head_dim)
        kv_cache = (torch.zeros(8, 128, 1, 512), torch.zeros(8, 128, 1, 64))
        topk_indices = torch.zeros(num_input_tokens, 1, 64, dtype=torch.int32)
        fwd_ctx = impl._get_parallel_forward_context(built, num_input_tokens, ql_nope)
        try:
            out = impl._execute_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                built,
                fwd_ctx.actual_seq_lengths_query,
                fwd_ctx.actual_seq_lengths_key,
            )
            if tuple(out.shape) != (num_input_tokens, impl.local_num_heads, impl.kv_lora_rank):
                failures.append(f"{label}: unexpected attention output shape {tuple(out.shape)}")
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")

    if len(captured_calls) != len(scenarios) and not failures:
        failures.append(f"expected {len(scenarios)} kernel invocations, got {len(captured_calls)}")
    return failures


def main():
    failures = check_workspace_integrity()
    error_types = []
    tree = None
    try:
        path = ROOT / SFA
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        find_class(tree, "AscendSFAMetadataBuilder")
        find_class(tree, "AscendSFAImpl")
    except (OSError, SyntaxError, AssertionError) as exc:
        error_types.append(type(exc).__name__)
        failures.append(f"{SFA}: {type(exc).__name__}: {exc}")
    if not failures and tree is not None:
        import torch  # noqa: F401

        try:
            failures.extend(check_sfa_mtp_metadata_and_fia_layout(tree))
        except Exception as exc:  # noqa: BLE001
            error_types.append(type(exc).__name__)
            failures.append(f"{type(exc).__name__}: {exc}")
    if failures:
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "correctness_verified": False,
                    "verification_status": "failed",
                    "validation_scope": "cpu_behavior",
                    "performance_status": "unmeasured",
                    "error_type": error_types[0] if error_types else "AssertionError",
                    "failure_reason": "; ".join(failures),
                }
            )
        )
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "reward": 1.0,
                "correctness_verified": True,
                "verification_status": "completed",
                "validation_scope": "cpu_behavior",
                "performance_status": "unmeasured",
                "native_kernel_bound": None,
                "measured_npu_speedup": None,
                "speedup_ratio": None,
            }
        )
    )


if __name__ == "__main__":
    main()
