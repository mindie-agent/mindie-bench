"""[Role] Derive product comparison conditions from actual task and runtime bytes.
[Invariants] Task starts, input/oracle files and budgets are never free-form labels;
all native component modules and the plugin artifact are read from the selected runtime.
[Timeouts & Locks] Git metadata10s, isolated runtime inspection30s; no model calls.
[Side-Effect Policy] Read-only inspection; the caller persists the resulting manifest.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.io import digest, task_hash, tree_hash
from mindie_bench.models import ExperimentConfig, Group, TaskSpec
from mindie_bench.runner.matrix_contracts import BudgetEnvelope, MatchingConditions
from mindie_bench.runner.native_product import runtime_identity
from mindie_bench.runner.phase_budget import product_phase_budget


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_snapshot(config: ExperimentConfig, mode: Group) -> dict:
    selected = config.groups.get(mode)
    native = selected.native_dsh_plugin if selected else None
    result = {"harness_binary_sha256": sha(Path(config.harness.executable).resolve(strict=True))}
    if native:
        observed = runtime_identity(native.runtime_python)
        for name, installed in observed["components"].items():
            commit = (installed["direct_url"] or {}).get("vcs_info", {}).get("commit_id")
            if commit != config.component_revisions.get(name):
                raise ValueError(f"actual installed {name} commit differs from the product pin")
        result["native_product"] = {
            "runtime_identity": observed,
            "package_sha256": tree_hash(Path(native.package_path)),
            "host_entry_sha256": sha(Path(native.package_path) / "host.mjs"),
            "contribution_enabled": native.contribution_enabled,
        }
    return result


def derive_inputs(tasks: dict[str, Path], config: ExperimentConfig, tier: str) -> dict:
    starts, inputs, oracles, task_rows, budgets, permissions = {}, {}, {}, {}, [], {}
    phase_budgets = []
    selected_groups = [group for group in config.groups.values() if group]
    native_product = any(group.native_dsh_plugin for group in selected_groups)
    knowledge_sessions = 8 if any(
        server.component == "knowledge"
        for group in selected_groups
        for server in group.mcp_servers.values()
    ) else 0
    g3 = config.groups.get(Group.G3)
    capture_timeout_sec = g3.capture_timeout_sec if g3 is not None else None
    for task_id, root in sorted(tasks.items()):
        task = TaskSpec.load(root)
        if task.task.instance_id != task_id:
            raise ValueError("product task map differs from the actual task identity")
        environment = task.environment[tier]
        git = subprocess.run(
            ["git", "-C", environment.repo_path, "rev-parse", environment.base_commit + "^{tree}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        starts[task_id] = {"base_commit": environment.base_commit, "git_tree": git.stdout.strip()}
        oracle = freeze_oracle(root, task.oracle)
        oracles[task_id] = oracle.sha256
        inputs[task_id] = {
            "instruction_sha256": sha(root / "instruction.md"),
            "public_environment_sha256": tree_hash(root / "environment"),
            "knowledge_conditions": task.mindie_knowledge.model_dump(mode="json"),
        }
        task_rows[task_id] = {
            "task_root": str(root),
            "task_sha256": task_hash(root),
            "oracle_sha256": oracle.sha256,
        }
        # One shared derivation covers preparation, agent execution, every
        # sequential settlement wait and completed cleanup. Preparation spends
        # the execution allowance instead of receiving an unreported free term.
        phase_budget = product_phase_budget(
            task.budget.timeout_sec,
            task.budget.verifier_timeout_sec,
            remote=tier == "l3",
            native_product=native_product,
            knowledge_sessions=knowledge_sessions,
            capture_timeout_sec=capture_timeout_sec,
        )
        phase_budgets.append(phase_budget)
        budgets.append(
            BudgetEnvelope(
                total_model_calls=task.budget.max_turns,
                npu_runs=task.budget.max_npu_runs if environment.npu_count else 0,
                max_wall_clock_sec=phase_budget["total_delivery_sec"],
            ).to_dict()
        )
        permissions[task_id] = {
            "npu_count": environment.npu_count,
            "replay_commands": environment.replay_commands,
            "protected_oracle": list(oracle.protected_paths),
        }
    if not budgets or any(b != budgets[0] for b in budgets):
        raise ValueError("one product cohort requires the same actual per-task budget")
    if any(item != phase_budgets[0] for item in phase_budgets):
        raise ValueError("one product cohort requires the same derived phase budget")
    runner = Path(__file__).parent
    harness = config.harness.model_dump(mode="json")
    inference = {
        k: v
        for k, v in harness.items()
        if k
        not in {
            "executable",
            "api_key_env",
            "input_usd_per_million",
            "cached_usd_per_million",
            "output_usd_per_million",
            "cache_creation_usd_per_million",
        }
    }
    corpus = {
        key: {"commit": item.commit, "sha256": tree_hash(Path(item.path))}
        for key, item in sorted(config.corpora.items())
    }
    conditions = MatchingConditions(
        model_id=config.harness.model,
        inference_profile=digest(inference),
        task_start_digest=digest(starts),
        oracle_digest=digest(oracles),
        input_digest=digest(inputs),
        knowledge_cutoff=digest(corpus),
        tool_permission_digest=digest(
            {
                "tier": tier,
                "task_permissions": permissions,
                "remote": config.remote.model_dump(mode="json") if config.remote else None,
                "sandbox_sha256": sha(runner / "sandbox.py"),
                "gateway_sha256": sha(runner / "gateway.py"),
            }
        ),
        persistence_permission_digest=digest(
            {"profile_builder_sha256": sha(runner / "profile_builder.py"), "isolated_trial_state": True}
        ),
    )
    return {
        "conditions": conditions.to_dict(),
        "budget": budgets[0],
        "budget_basis": phase_budgets[0],
        "tasks": task_rows,
        "corpora": corpus,
        "budget_scope": "shared phase budget from run_trial start through completed cleanup; execution allowance includes preparation; total Gateway model calls include native summaries",
    }
