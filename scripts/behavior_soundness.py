# [Role] Reproducible zero-model soundness for eight historical CPU behavior judges.
# [Invariants] Real subprocesses must reject base/no-op and accept gold/helper extraction.
# [Timeouts & Locks] Each verifier has a 60s process deadline; git reads/apply have 20s deadlines.
# [Side-Effect Policy] Writes receipts/logs and generated control patches; original archives are read-only.
"""Run from the bench checkout with --python set to an installed CPU torch interpreter."""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

TASKS = (
    'dspark-mrv2-drafter-crash-zh-vague', 'l3-dspark-mrv2-drafter-crash-zh-vague',
    'k3-idle-dp-gdn-state-zh-vague', 'mtp-draft-slot-mapping-capacity-zh-vague',
    'sfa-mtp-seq-lens-seed-zh-vague', 'sfa-mtp-seq-lens-eval-zh-vague',
    'mrv2-dsa-prefill-metadata-zh-vague', 'l3-mrv2-dsa-prefill-metadata-zh-vague',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def forward_args(arguments):
    args = [ast.Name(id=a.arg, ctx=ast.Load()) for a in arguments.posonlyargs + arguments.args]
    if arguments.vararg:
        args.append(ast.Starred(value=ast.Name(id=arguments.vararg.arg, ctx=ast.Load()), ctx=ast.Load()))
    keywords = [ast.keyword(arg=a.arg, value=ast.Name(id=a.arg, ctx=ast.Load())) for a in arguments.kwonlyargs]
    if arguments.kwarg:
        keywords.append(ast.keyword(arg=None, value=ast.Name(id=arguments.kwarg.arg, ctx=ast.Load())))
    return args, keywords


def extracted_helpers(source, relative):
    """Move exercised behavior through a top-level function and an auxiliary class."""
    tree = ast.parse(source)
    targets = {
        'dsa_v1.py': {'build_req_metadata', 'get_kv_cache_shape'},
        'model_runner_v1.py': {'_build_attention_metadata'},
        'block_table.py': {'compute_slot_mapping_draft'},
        'sfa_v1.py': {'build', 'build_for_drafting'},
        'attn_utils.py': {'build_attn_metadata'},
        'default.py': {'prepare_attn'},
        'speculator.py': {'_ascend_update_seq_lens'},
    }.get(Path(relative).name, set())
    added = []
    for node in list(ast.walk(tree)):
        if not isinstance(node, ast.FunctionDef) or node.name not in targets:
            continue
        number = len(added)
        clsname, helper = f'ExtractedBehavior{number}', f'extracted_behavior_{number}'
        args, keywords = forward_args(node.args)
        implementation = copy.deepcopy(node)
        implementation.name = 'run'
        implementation.decorator_list = [ast.Name(id='staticmethod', ctx=ast.Load())]
        added.append(ast.ClassDef(name=clsname, bases=[], keywords=[], body=[implementation], decorator_list=[]))
        bridge = copy.deepcopy(node)
        bridge.name, bridge.decorator_list = helper, []
        bridge.body = [ast.Return(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id=clsname, ctx=ast.Load()), attr='run', ctx=ast.Load()),
            args=copy.deepcopy(args), keywords=copy.deepcopy(keywords),
        ))]
        added.append(bridge)
        node.body = [ast.Return(value=ast.Call(func=ast.Name(id=helper, ctx=ast.Load()), args=args, keywords=keywords))]
    if Path(relative).name in {"model.py", "llm_base_proposer.py"}:
        state = {"_mtp_hidden_buffer", "slot_mapping_group", "seq_lens_group", "query_start_loc_group"}
        class ExtractState(ast.NodeTransformer):
            def visit_Assign(self, node):
                if len(node.targets) != 1:
                    return self.generic_visit(node)
                target = node.targets[0]
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self" and target.attr in state):
                    return self.generic_visit(node)
                helper_name = f"assign_{target.attr}"
                class_name = f"State_{target.attr}"
                definitions = ast.parse(
                    f"class {class_name}:\n"
                    f"    @staticmethod\n"
                    f"    def run(owner, value):\n"
                    f"        owner.{target.attr} = value\n"
                    f"def {helper_name}(owner, value):\n"
                    f"    return {class_name}.run(owner, value)\n"
                ).body
                if not any(getattr(item, "name", None) == helper_name for item in added):
                    added.extend(definitions)
                return ast.Expr(value=ast.Call(func=ast.Name(id=helper_name, ctx=ast.Load()),
                                              args=[ast.Name(id="self", ctx=ast.Load()), node.value], keywords=[]))
        tree = ExtractState().visit(tree)
    # Helpers retain the full business implementation and can appear after callers.
    tree.body.extend(added)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def no_op_source(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Return(value=ast.Constant(value=None))]
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def equivalent_gold(sources, name, *, partial=False):
    """Independent semantic variants: active-prefix propagation or post-builder sanitation."""
    changed = dict(sources)
    if 'mrv2-dsa-prefill-metadata' in name:
        for relative, source in sources.items():
            if Path(relative).name not in {'default.py', 'speculator.py'}:
                continue
            class SliceFlags(ast.NodeTransformer):
                def visit_Attribute(self, node):
                    if node.attr == 'is_prefilling_np':
                        value = ast.Subscript(value=node, slice=ast.Slice(upper=ast.Name(id='num_reqs', ctx=ast.Load())), ctx=ast.Load())
                        if partial:
                            value = ast.BinOp(left=value, op=ast.BitAnd(), right=ast.Constant(value=False))
                        return value
                    return self.generic_visit(node)
            tree = SliceFlags().visit(ast.parse(source))
            changed[relative] = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    elif name.startswith('k3-'):
        relative = 'vllm_ascend/worker/model_runner_v1.py'
        tree = ast.parse(sources[relative])
        class OutputSanitation(ast.NodeTransformer):
            def visit_If(self, node):
                if isinstance(node.test, ast.Name) and node.test.id == 'is_gdn_noop':
                    return []
                if isinstance(node.test, ast.Name) and node.test.id == 'skip_gdn_state_update':
                    if any(isinstance(item, ast.Attribute) and item.attr == 'gdn_query_start_loc'
                           for statement in node.body for item in ast.walk(statement)):
                        return node.orelse
                return self.generic_visit(node)
        tree = OutputSanitation().visit(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != '_build_attn_group_metadata':
                continue
            for index, statement in enumerate(node.body):
                if isinstance(statement, ast.If) and any(
                    isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'builder'
                    and call.func.attr == 'build' for call in ast.walk(statement)
                ):
                    node.body[index + 1:index + 1] = ast.parse(
                        'if skip_gdn_state_update and isinstance(builder, GDNAttentionMetadataBuilder):\n'
                        '    attn_metadata_i = sanitize_output(attn_metadata_i)\n'
                    ).body
                    break
        tree.body.insert(0, ast.ImportFrom(module='vllm_ascend.ops.behavior_state_helper',
                                          names=[ast.alias(name='sanitize_output')], level=0))
        changed[relative] = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
        changed['vllm_ascend/ops/behavior_state_helper.py'] = (
            'import torch\n'
            'def sanitize_output(metadata):\n'
            '    metadata.num_actual_tokens = 0\n'
            '    metadata.non_spec_query_start_loc = torch.zeros_like(metadata.non_spec_query_start_loc)\n'
            '    metadata.non_spec_state_indices_tensor = torch.full_like(metadata.non_spec_state_indices_tensor, -1)\n'
            '    metadata.spec_sequence_masks = None\n'
            + ('' if partial else
               '    nested = metadata.non_spec_decode_metadata\n'
               '    nested.causal_conv1d.query_start_loc = torch.zeros_like(nested.causal_conv1d.query_start_loc)\n'
               '    nested.causal_conv1d.cache_indices = torch.full_like(nested.causal_conv1d.cache_indices, -1)\n'
               '    nested.actual_seq_lengths = torch.zeros_like(nested.actual_seq_lengths)\n')
            + '    return metadata\n'
        )
    return changed


def consumer_gold(sources, name, guard):
    """Gold variant closes the no-op in the actual consumer; negative controls break its guard."""
    changed = equivalent_gold(sources, name)
    helper = 'vllm_ascend/ops/behavior_state_helper.py'
    changed[helper] = changed[helper].replace(
        '    return metadata\n',
        '    metadata.num_decodes = 0\n'
        '    metadata.num_decode_tokens = 0\n'
        '    metadata.non_spec_decode_metadata = None\n'
        '    return metadata\n',
    )
    relative = 'vllm_ascend/ops/gdn.py'
    tree = ast.parse(sources[relative])
    if guard != 'missing':
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != '_forward_core':
                continue
            for index, statement in enumerate(node.body):
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'num_actual_tokens'
                    for target in statement.targets
                ):
                    early = ast.parse('if num_actual_tokens == 0:\n    core_attn_out.zero_()\n    return\n').body
                    if guard == 'unconditional':
                        early = early[0].body
                    node.body[index + 1:index + 1] = early
                    break
    changed[relative] = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    return changed


def write_variant(workspace, out, label, before, after):
    patch = []
    for relative, source in after.items():
        previous = before.get(relative, '')
        patch.extend(difflib.unified_diff(previous.splitlines(keepends=True), source.splitlines(keepends=True),
                     fromfile=f'a/{relative}' if relative in before else '/dev/null', tofile=f'b/{relative}'))
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    (out / f'{label}.diff').write_text(''.join(patch))


def run_case(task, workspace, python, out, label):
    env = dict(os.environ, MINDIE_BENCH_WORKSPACE=str(workspace), MINDIE_BENCH_PYTHON=python)
    command = ['bash', str(task / 'verifier/test.sh')]
    start = time.monotonic()
    proc = subprocess.run(command, env=env, cwd=workspace, capture_output=True, timeout=60)
    elapsed = time.monotonic() - start
    (out / f'{label}.stdout').write_bytes(proc.stdout)
    (out / f'{label}.stderr').write_bytes(proc.stderr)
    verdicts = []
    for line in proc.stdout.decode(errors='replace').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'reward' in value:
            verdicts.append(value)
    verdict = verdicts[-1] if verdicts else {}
    return dict(command=command, exit_code=proc.returncode, elapsed_sec=elapsed,
                reward=verdict.get('reward'), verdict=verdict,
                source_sha256={str(p.relative_to(workspace)): digest(p.read_bytes())
                               for p in sorted(workspace.rglob('*.py'))},
                stdout=str(out / f'{label}.stdout'), stderr=str(out / f'{label}.stderr'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/integrity-20261029/behavior-soundness'))
    parser.add_argument('--task', action='append', choices=TASKS)
    parser.add_argument('--no-write-receipts', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    results = {}
    for name in args.task or TASKS:
        task = root / 'tasks/zh-vague' / name
        config = tomllib.loads((task / 'task.toml').read_text())
        tier, environment = next(iter(config['environment'].items()))
        commit, repository = environment['base_commit'], environment['repo_path']
        out = output / name
        out.mkdir(parents=True, exist_ok=True)
        base = {}
        source_files = list(config['mindie_knowledge']['target_files'])
        if name.startswith('k3-'):
            source_files.append('vllm_ascend/ops/gdn.py')
        for relative in source_files:
            base[relative] = subprocess.run(
                ['git', '-C', repository, 'show', f'{commit}:{relative}'],
                capture_output=True, check=True, timeout=20,
            ).stdout
        with tempfile.TemporaryDirectory(prefix='mindie-behavior-soundness-') as folder:
            workspace = Path(folder)
            for relative, data in base.items():
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            stages = {'base': run_case(task, workspace, args.python, out, 'base')}
            negative_patch = []
            for relative, data in base.items():
                negative = no_op_source(data.decode())
                negative_patch.extend(difflib.unified_diff(data.decode().splitlines(keepends=True),
                                      negative.splitlines(keepends=True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
                (workspace / relative).write_text(negative)
            (out / "negative_control.diff").write_text("".join(negative_patch))
            stages['negative_control'] = run_case(task, workspace, args.python, out, 'negative_control')
            for relative, data in base.items():
                (workspace / relative).write_bytes(data)
            subprocess.run(['git', 'apply', '--no-index', str(task / 'oracle/gold_patch.diff')],
                           cwd=workspace, check=True, capture_output=True, timeout=20)
            stages['gold'] = run_case(task, workspace, args.python, out, 'gold')
            gold_sources = {relative: (workspace / relative).read_text() for relative in base}
            helper_patch = []
            for relative in base:
                path = workspace / relative
                gold = path.read_text()
                helper = extracted_helpers(gold, relative)
                helper_patch.extend(difflib.unified_diff(gold.splitlines(keepends=True), helper.splitlines(keepends=True),
                                    fromfile=f"a/{relative}", tofile=f"b/{relative}"))
                path.write_text(helper)
            (out / "gold_helpers.diff").write_text("".join(helper_patch))
            stages['gold_helpers'] = run_case(task, workspace, args.python, out, 'gold_helpers')
            if name.startswith('k3-') or 'mrv2-dsa-prefill-metadata' in name:
                for label, partial in (('gold_equivalent', False), ('negative_partial_state', True)):
                    changed = equivalent_gold(gold_sources, name, partial=partial)
                    write_variant(workspace, out, label, gold_sources, changed)
                    stages[label] = run_case(task, workspace, args.python, out, label)
            if name.startswith('k3-'):
                for label, guard in (('gold_consumer_early_return', 'conditional'),
                                     ('negative_missing_consumer_guard', 'missing'),
                                     ('negative_unconditional_consumer', 'unconditional')):
                    changed = consumer_gold(gold_sources, name, guard)
                    write_variant(workspace, out, label, gold_sources, changed)
                    stages[label] = run_case(task, workspace, args.python, out, label)
        expected = {'base': 0.0, 'negative_control': 0.0, 'gold': 1.0, 'gold_helpers': 1.0}
        if 'gold_equivalent' in stages:
            expected.update(gold_equivalent=1.0, negative_partial_state=0.0)
        if 'gold_consumer_early_return' in stages:
            expected.update(gold_consumer_early_return=1.0, negative_missing_consumer_guard=0.0,
                            negative_unconditional_consumer=0.0)
        passed = all(stages[key]['reward'] == reward and stages[key]['exit_code'] == (0 if reward else 1)
                     for key, reward in expected.items())
        hashes = {str(p.relative_to(task)): digest(p.read_bytes()) for p in sorted((task / 'verifier').rglob('*'))
                  if p.is_file() and '__pycache__' not in p.parts}
        receipt = dict(schema_version='2.0', status='completed' if passed else 'failed',
                       task=name, tier=tier, validation_scope='cpu_behavior', model_calls=0,
                       source_commit=commit, repository=repository, python=args.python,
                       before_reward=stages['base']['reward'], after_reward=stages['gold']['reward'],
                       negative_control_reward=stages['negative_control']['reward'],
                       helper_gold_reward=stages['gold_helpers']['reward'],
                       verifier_sha256=hashes, gold_patch_sha256=digest((task/'oracle/gold_patch.diff').read_bytes()),
                       control_generator_sha256=digest(Path(__file__).read_bytes()),
                       native_kernel_bound=None, measured_npu_speedup=None, speedup_ratio=None,
                       cases=stages, errors=[] if passed else ['soundness expectations did not all hold'])
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if passed and not args.no_write_receipts:
            # Match the repository's task_hash implementation; the receipt itself is excluded.
            import sys
            sys.path.insert(0, str(root / 'src'))
            from mindie_bench.io import task_hash
            receipt['task_sha256'] = task_hash(task)
            (task / 'soundness_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        results[name] = receipt
        print(name, {key: value['reward'] for key, value in stages.items()}, flush=True)
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    raise SystemExit(not all(value['status'] == 'completed' for value in results.values()))


if __name__ == '__main__':
    main()
