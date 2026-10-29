"""[Role] Execute an explicitly bound strategy continuation and its real oracle.

[Invariants] Each call starts from the same pinned source in an owned worktree.
The agent receives the selected unit and prerequisites, never judge answers or
oracle files. Only local_verifier owns the task verdict; agent output is a log.
[Timeouts & Locks] One absolute business + verifier + cleanup deadline bounds
agent, verifier and WorktreeSlot phase waits. Patch capture reserves half the
cleanup grace for worktree removal; local accounting cannot erase a verdict.
[Side-Effect Policy] Runs the supplied command in the existing native sandbox,
retains context/logs/patch/verifier evidence, and removes only its own worktree.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mindie_bench.evaluation.oracle_freezer import freeze_oracle
from mindie_bench.io import write_json
from mindie_bench.models import (
    OracleContract,
    SourceEnvironment,
    TaskSpec,
    Trajectory,
    UnknownExternalEffects,
)
from mindie_bench.runner.profile_builder import Profile
from mindie_bench.runner.sandbox import native_command
from mindie_bench.runner.termination import terminate_process_group
from mindie_bench.runner.trial_preflight import check_read_boundaries
from mindie_bench.runner.verifier import local_verifier
from mindie_bench.runner.worktree_slot import WorktreeSlot

# local_verifier may drain its process group after the command timeout.
VERIFIER_DRAIN_GRACE_SEC = 5.0


@dataclass(frozen=True)
class StrategyContinuationTask:
    """A public source pin and a separate private task oracle."""

    task_root: Path
    source: SourceEnvironment


@dataclass(frozen=True)
class ContinuationExecution:
    """Observed result of the runner-owned agent/verifier loop."""

    payload: dict
    terminal_status: str
    completed: bool
    passed: bool | None
    evidence: dict
    error: str | None = None
    trajectory: Trajectory | dict | None = None
    broker_knowledge: dict | None = None


class StrategyContinuationExecutor:
    """Callable for ``run_strategy_selection_pilot(executor_fn=...)``.

    The command reads one JSON context from stdin and edits its current workspace.
    ``task_root/verifier/test.sh`` invokes the task's independent ``verify.py``.
    This local verifier loop establishes CPU behavior only; model usage, adoption,
    NPU occupancy and performance remain unavailable unless a native runner owns them.
    """

    def __init__(
        self,
        tasks: Mapping[str, StrategyContinuationTask],
        *,
        command: Sequence[str],
        run_root: Path,
        read_paths: Sequence[Path] = (),
        verifier_timeout_sec: float = 120.0,
        knowledge_token_budget: int = 4096,
    ):
        self.tasks = dict(tasks)
        self.command = list(command)
        self.run_root = Path(run_root).resolve()
        self.read_paths = [Path(path).resolve(strict=True) for path in read_paths]
        self.verifier_timeout_sec = verifier_timeout_sec
        self.knowledge_token_budget = knowledge_token_budget

    def __call__(
        self, *, case, arm, selector_decision, routing_index,
        remaining_model_calls, remaining_window_sec, cleanup_grace_sec=5.0,
    ) -> ContinuationExecution:
        started = time.monotonic()
        business_deadline = started + remaining_window_sec
        verifier_deadline = business_deadline + self.verifier_timeout_sec
        outer_deadline = verifier_deadline + cleanup_grace_sec
        task = self.tasks[case.case_id]
        root = task.task_root.resolve(strict=True)
        if task.source.npu_count:
            raise ValueError("local continuation requires a CPU task; use run_trial for NPU work")
        if remaining_window_sec <= 0:
            raise TimeoutError("selector exhausted the continuation execution window")
        selected_ref = (selector_decision or {}).get("selected_strategy_ref") or "default"
        if arm == "A":
            selected_ref = "default"
        loaded = {"units": [], "requires": [], "unresolved": [], "usage": {}}
        if selected_ref != "default":
            if selected_ref.split("@", 1)[0] not in case.candidate_ids:
                raise ValueError("selected strategy is outside this case's candidate set")
            loaded = routing_index.explain_closure(
                [selected_ref], token_budget=self.knowledge_token_budget,
            )
            if loaded["unresolved"]:
                raise ValueError(f"selected strategy closure is incomplete: {loaded['unresolved']}")
        context = {
            "schema_version": "mindie-strategy-continuation-context/1",
            "case_id": case.case_id,
            "decision_point_id": case.decision_point_id,
            "problem_statement": case.problem_statement,
            "declared_conditions": dict(case.declared_conditions),
            "observed_conditions": dict(case.observed_conditions),
            "pre_decision_observations": [
                {"statement": obs.statement, "provenance": obs.anchor}
                for obs in case.pre_decision_observations
            ],
            "selected_strategy_ref": selected_ref,
            "strategy_context": loaded,
            "remaining_model_calls": remaining_model_calls,
            "remaining_window_sec": remaining_window_sec,
        }
        self.run_root.mkdir(parents=True, exist_ok=True)
        trial_dir = Path(tempfile.mkdtemp(prefix=f"{arm}-", dir=self.run_root)).resolve()
        context_path = trial_dir / "agent_context.json"
        write_json(context_path, context)
        # Freeze before the foreground process, and grant it no task-root reads.
        contract = (
            TaskSpec.model_validate(tomllib.loads((root / "task.toml").read_text())).oracle
            if (root / "task.toml").is_file() else OracleContract()
        )
        manifest = freeze_oracle(root, contract)
        profile_root = trial_dir / "profile"
        temporary = profile_root / "tmp"
        temporary.mkdir(parents=True)
        home = profile_root / "home"
        home.mkdir()
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(home), "TMPDIR": str(temporary), "PYTHONDONTWRITEBYTECODE": "1",
        }
        runtime_reads = [Path(sys.prefix), Path(sys.base_prefix), *self.read_paths]
        check_read_boundaries(runtime_reads, [root, Path(task.source.repo_path)])
        profile = Profile(profile_root, env, None, {}, runtime_reads)
        slot = WorktreeSlot(
            task.source, trial_dir / "workspace", root, deadline=business_deadline,
        )
        evidence = {
            "trial_dir": str(trial_dir), "agent_context": str(context_path),
            "selected_strategy_ref": selected_ref, "source_commit": task.source.base_commit,
            "validation_scope": "local_behavior",
            "verifier_record": None, "adoption_observed": None,
            "outer_budget_sec": remaining_window_sec + self.verifier_timeout_sec + cleanup_grace_sec,
        }
        payload = {
            "selected_strategy_ref": selected_ref, "verifier_passed": False,
            "knowledge_tokens": loaded.get("usage", {}).get("estimated_tokens"),
        }
        status, completed, passed, error = "infra", False, None, None
        try:
            workspace = slot.prepare()
            evidence["workspace"] = str(workspace)
            command = native_command(
                self.command, read=runtime_reads, write=[workspace, home, temporary],
                cwd=workspace, policy_path=trial_dir / "agent.sb",
            )
            execution_window = business_deadline - time.monotonic()
            if execution_window <= 0:
                raise TimeoutError("workspace preparation exhausted continuation execution window")
            with (
                context_path.open("rb") as stdin,
                (trial_dir / "agent.stdout").open("xb") as stdout,
                (trial_dir / "agent.stderr").open("xb") as stderr,
            ):
                process = subprocess.Popen(
                    command, cwd=workspace, env=env, stdin=stdin, stdout=stdout, stderr=stderr,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=execution_window)
                finally:
                    try:
                        terminate_process_group(process, term_wait=0.25, kill_wait=2.0)
                    except TimeoutError as exc:
                        raise UnknownExternalEffects(f"agent process cleanup failed: {exc}") from exc
            evidence["agent_returncode"] = returncode
            completed = returncode == 0
            # A nonzero agent exit does not erase a usable on-disk candidate.
            verifier_window = min(
                self.verifier_timeout_sec,
                verifier_deadline - time.monotonic() - VERIFIER_DRAIN_GRACE_SEC,
            )
            if verifier_window <= 0:
                raise TimeoutError("continuation exhausted the verifier execution window")
            evidence["verifier_timeout_sec"] = verifier_window
            evidence["verifier_drain_grace_sec"] = VERIFIER_DRAIN_GRACE_SEC
            verdict, record = local_verifier(
                root, workspace, profile, trial_dir, verifier_window,
                runtime_reads, oracle_manifest=manifest,
            )
            evidence["verifier_record"] = record
            passed = verdict.reward == 1.0
            status = "pass" if passed else "fail"
            payload["verifier_passed"] = passed
            try:
                write_json(trial_dir / "verifier_record.json", record)
            except OSError as exc:
                evidence["verifier_record_error"] = f"{type(exc).__name__}: {exc}"
        except subprocess.TimeoutExpired as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "infra"
        except UnknownExternalEffects as exc:
            error, status = str(exc), "unknown"
        except TimeoutError as exc:
            error, status = str(exc), "infra"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if slot.created:
                slot.deadline = outer_deadline - cleanup_grace_sec / 2
                try:
                    slot.snapshot_patch(trial_dir)
                    evidence["patch"] = str(trial_dir / "final_patch.diff")
                except Exception as exc:
                    evidence["patch_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    slot.deadline = outer_deadline
                    try:
                        slot.cleanup()
                    except Exception as exc:
                        evidence["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        evidence["elapsed_sec"] = time.monotonic() - started
        evidence["outer_budget_exceeded"] = time.monotonic() > outer_deadline
        try:
            write_json(trial_dir / "continuation.json", {
                "terminal_status": status, "continuation_completed": completed,
                "continuation_passed": passed, "error": error, "evidence": evidence,
            })
        except OSError as exc:
            evidence["receipt_error"] = f"{type(exc).__name__}: {exc}"
        return ContinuationExecution(payload, status, completed, passed, evidence, error)
