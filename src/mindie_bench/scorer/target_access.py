"""Target-access observations and trial-level access timing metrics.

[Role]
Recognizes direct file RPCs, statically resolved shell/Python commands, NPU test
attempts, and remote job confirmations, then derives edit and validation timing.
[Invariants]
- A path observation is evidence of an attempted access, never proof that source
  bytes were returned or that the intended root cause was repaired.
- Failed, unconfirmed and dynamic access attempts stay explicit rather than being
  silently promoted to successful localization.
[Timeouts & Locks]
- Pure in-memory analysis; no subprocess, network I/O or locks.
[Side-Effect Policy]
- Pure functions returning serializable observations and metric fragments.
"""

from __future__ import annotations

import math
import posixpath
import re
from collections import defaultdict

from mindie_bench.evaluation.broker_knowledge import BASIS as BROKER_KNOWLEDGE_BASIS
from mindie_bench.evaluation.broker_knowledge import SCHEMA as BROKER_KNOWLEDGE_SCHEMA
from mindie_bench.evaluation.source_visibility import source_evidence_metrics
from mindie_bench.models import HarnessConfig, TaskSpec, ToolCall, Trajectory
from mindie_bench.scorer.command_observations import (
    is_recognized_npu_command,
    observe_npu_command,
)
from mindie_bench.scorer.target_access_shell import (  # noqa: F401
    _python_accesses,
    _shell_accesses,
)
from mindie_bench.scorer.tool_observations import summarize_tool_observations

REMOTE = {"remote_bash", "remote_job_start", "remote_exec", "remote_command"}
READ_EDIT = {
    "remote_read", "Read", "remote_edit", "remote_multi_edit", "Edit", "MultiEdit",
    "Write", "remote_grep", "Grep", "read_file", "edit_file", "remote_apply_patch",
    "apply_patch", "remote_write", "write_file",
}


def tool_name(name: str) -> str:
    return name.rsplit("__", 1)[-1].rsplit(".", 1)[-1]


def remote_execution(call: ToolCall) -> bool:
    name = tool_name(call.name)
    command = call.arguments.get("command", call.arguments.get("cmd", ""))
    return name in REMOTE or (
        name.lower() in {"bash", "exec_command", "shell"}
        and isinstance(command, str)
        and bool(re.search(r"(?<!\w)ssh(?:\s|$)", command))
    )


def npu_test_runs(trajectory: Trajectory) -> int:
    """Count recognized command-attempt proxies, not complete hardware executions."""
    return sum(_npu_test_call(call) for call in trajectory.calls)


def _npu_test_call(call):
    command = call.arguments.get("command", call.arguments.get("cmd", ""))
    return remote_execution(call) and is_recognized_npu_command(command)


def _call_outcome(call):
    observed = call.result_observation
    if observed is not None:
        if observed.classification in {"success", "business_success"}:
            return "success"
        if observed.classification in {
            "business_failure",
            "transport_error",
            "protocol_error",
            "component_error",
        }:
            return "failed"
        return "unknown"
    if call.is_error is True:
        return "failed"
    if call.is_error is False and tool_name(call.name) in {
        "bash",
        "Bash",
        "exec_command",
        "shell",
        "Read",
        "Edit",
        "MultiEdit",
        "Write",
        "Grep",
    }:
        return "success"
    return "unknown"


def _job_argument_identity(call: ToolCall) -> str | None:
    values = [call.arguments[key] for key in ("job_id", "session_id") if key in call.arguments]
    if not values or any(not isinstance(value, str) or not value for value in values):
        return None
    return values[0] if all(value == values[0] for value in values) else None


def _pending_job_outcomes(trajectory: Trajectory) -> dict[int, tuple[str, dict]]:
    """Derive command certainty from scoped job receipts, preserving native facts."""
    grouped = defaultdict(list)
    for index, call in enumerate(trajectory.calls):
        result = call.result_observation
        if call.component == "remote-dev" and result is not None and result.job_id:
            grouped[(call.component, result.target_id, result.job_id)].append((index, call))
    outcomes = {}
    terminal_states = {"succeeded", "failed", "timeout", "cancelled", "completed"}
    for index, call in enumerate(trajectory.calls):
        result = call.result_observation
        if (
            call.component != "remote-dev"
            or not remote_execution(call)
            or result is None
            or result.classification != "pending"
        ):
            continue
        evidence = dict(
            basis="same_remote_job_reconciliation",
            original_classification=result.classification,
            original_exit_code=result.exit_code,
            job_id=result.job_id,
            target_id=result.target_id,
            confirmation_call_ids=[],
            reason="no_later_terminal_confirmation",
            edit_time_basis="original_command_started_sec_proxy",
        )
        outcome = "unknown"
        rows = grouped.get((call.component, result.target_id, result.job_id), [])
        if not result.target_id or not result.job_id or result.result_schema != "remote-dev.result.v1":
            evidence["reason"] = "incomplete_scoped_job_identity"
        elif sum(remote_execution(other) for _, other in rows) != 1:
            evidence["reason"] = "ambiguous_job_execution_identity"
        elif any(
            other.result_observation.classification == "unknown"
            or other.result_observation.submission_state == "uncertain"
            or other.result_observation.classification_reason == "conflicting_exit_codes"
            for _, other in rows
        ):
            evidence["reason"] = "uncertain_job_result"
        elif any(
            tool_name(other.name) in {"remote_job_status", "remote_job_tail", "remote_job_output", "remote_job_stdin"}
            and other.result_observation.operation_kind == "observation"
            and other.result_observation.source == result.source
            and other.result_observation.result_schema == result.result_schema
            and any(key in other.arguments for key in ("job_id", "session_id"))
            and _job_argument_identity(other) != result.job_id
            for _, other in rows
        ):
            evidence["reason"] = "conflicting_job_argument_identity"
        else:
            terminal = []
            for other_index, other in rows:
                observed = other.result_observation
                if (
                    tool_name(other.name) in {"remote_job_status", "remote_job_tail", "remote_job_output", "remote_job_stdin"}
                    and _job_argument_identity(other) == result.job_id
                    and observed.operation_kind == "observation"
                    and observed.source == result.source
                    and observed.result_schema == result.result_schema
                    and observed.remote_status in terminal_states
                    and observed.classification in {"business_success", "business_failure"}
                    and observed.exit_code is not None
                ):
                    terminal.append((other_index, other))
            exits = {other.result_observation.exit_code for _, other in terminal}
            inconsistent = any(
                (other.result_observation.classification == "business_success")
                != (other.result_observation.exit_code == 0)
                for _, other in terminal
            )
            if len(exits) > 1 or inconsistent:
                evidence["reason"] = "conflicting_job_terminal_results"
                evidence["known_terminal_exit_codes"] = sorted(exits)
            else:
                later = [other for other_index, other in terminal if other_index > index]
                if later:
                    code = later[0].result_observation.exit_code
                    outcome = "success" if code == 0 else "failed"
                    evidence.update(
                        basis="later_same_remote_job_terminal",
                        reason=None,
                        confirmed_exit_code=code,
                        confirmation_call_ids=[other.id for other in later],
                        confirmation_turn=later[0].turn,
                        confirmation_ended_sec=later[0].ended_sec,
                        source_evidence_time_basis="later_terminal_confirmation",
                    )
        outcomes[index] = (outcome, evidence)
    return outcomes


def target_access_observations(
    trajectory: Trajectory, targets: list[str]
) -> tuple[list[dict], list[dict]]:
    observations, unresolved = [], []
    job_outcomes = _pending_job_outcomes(trajectory)
    for index, call in enumerate(trajectory.calls):
        name = tool_name(call.name)
        paths = []
        if name in READ_EDIT:
            kind = (
                "edit"
                if name
                in {
                    "remote_edit",
                    "remote_multi_edit",
                    "Edit",
                    "MultiEdit",
                    "Write",
                    "edit_file",
                    "remote_apply_patch",
                    "apply_patch",
                    "remote_write",
                    "write_file",
                }
                else "read"
            )
            values = [call.arguments.get(key) for key in ("path", "file_path", "filename")]
            values += call.arguments.get("paths", []) + call.arguments.get("matched_files", [])
            if name in {"remote_apply_patch", "apply_patch"}:
                patch = call.arguments.get("patch", call.arguments.get("input"))
                if isinstance(patch, str):
                    values += re.findall(
                        r"(?m)^\*\*\* (?:(?:Update|Add|Delete) File|Move to): (.+)$", patch
                    )
            paths.extend((kind, path, name.startswith("remote_")) for path in values)
        elif name in REMOTE or name.lower() in {"bash", "exec_command", "shell"}:
            command = call.arguments.get("command", call.arguments.get("cmd"))
            if isinstance(command, str):
                paths, reasons = _shell_accesses(command, remote=name in REMOTE)
                unresolved.extend(dict(call_id=call.id, reason=reason) for reason in reasons)
        seen = set()
        for kind, path, remote in paths:
            if not isinstance(path, str):
                continue
            if any(mark in path for mark in ("$", "*", "?")):
                unresolved.append(dict(call_id=call.id, reason="dynamic_or_glob_path_unresolved"))
                continue
            path = posixpath.normpath(path)
            for target in targets:
                target = posixpath.normpath(target)
                if not (path == target or path.endswith("/" + target)) or (kind, target) in seen:
                    continue
                seen.add((kind, target))
                outcome, evidence = job_outcomes.get(
                    index,
                    (
                        _call_outcome(call),
                        dict(
                            basis="original_call_result",
                            classification=call.result_observation.classification
                            if call.result_observation is not None
                            else None,
                        ),
                    ),
                )
                observations.append(
                    dict(
                        call_id=call.id,
                        call_index=index,
                        turn=call.turn,
                        kind=kind,
                        target=target,
                        source="typed_path" if name in READ_EDIT else "command_structure",
                        execution_scope="remote" if remote else "local",
                        call_outcome=outcome,
                        outcome_evidence=evidence,
                        started_sec=call.started_sec,
                        ended_sec=call.ended_sec,
                    )
                )
    return observations, unresolved


def first_localization(trajectory: Trajectory, targets: list[str]) -> int | None:
    observations, _ = target_access_observations(trajectory, targets)
    return min(
        (row["turn"] for row in observations if row["call_outcome"] == "success"),
        default=None,
    )


def _duration_profile(calls):
    intervals = []
    for call in calls:
        if call.started_sec is None or call.ended_sec is None:
            continue
        if not all(math.isfinite(value) and value >= 0 for value in (call.started_sec, call.ended_sec)):
            raise ValueError("tool interval must contain finite nonnegative elapsed times")
        if call.ended_sec < call.started_sec:
            raise ValueError("tool end precedes start")
        intervals.append((call.started_sec, call.ended_sec))
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    raw = sum(end - start for start, end in intervals)
    union = sum(end - start for start, end in merged)
    return dict(
        calls=len(calls),
        observed_intervals=len(intervals),
        missing_intervals=len(calls) - len(intervals),
        complete=len(intervals) == len(calls),
        raw_cumulative_sec=raw,
        observed_union_sec=union,
        overlap_sec=raw - union,
        coverage=len(intervals) / len(calls) if calls else None,
        first_started_sec=intervals[0][0] if intervals else None,
        last_ended_sec=max(end for _, end in intervals) if intervals else None,
    )


def _edit_metrics(trajectory, observations, unresolved):
    def order(row):
        return row["turn"], row["call_index"]

    ordered = sorted(observations, key=order)
    edits = list({row["call_index"]: row for row in ordered if row["kind"] == "edit"}.values())
    successful = [row for row in edits if row["call_outcome"] == "success"]
    first, last = (successful[0], successful[-1]) if successful else (None, None)
    access = next((row for row in ordered if row["call_outcome"] == "success"), None)
    access_reason = (
        None
        if access
        else "only_failed_or_unconfirmed_access_attempts"
        if ordered
        else "dynamic_access_unresolved"
        if unresolved
        else "no_target_access_observed"
    )
    reason = (
        None
        if successful
        else "only_failed_or_unconfirmed_edit_attempts"
        if edits
        else "dynamic_access_unresolved"
        if unresolved
        else "no_target_edit_observed"
    )
    primary = dict(
        turns_to_first_localization=access["turn"] if access else None,
        time_to_first_localization_sec=access["started_sec"] if access else None,
        turns_to_first_target_edit=first["turn"] if first else None,
        time_to_first_target_edit_sec=first["started_sec"] if first else None,
        turns_to_last_target_edit=last["turn"] if last else None,
        time_to_last_target_edit_sec=last["started_sec"] if last else None,
        npu_runs_before_last_target_edit=(
            sum(
                index < last["call_index"] and _npu_test_call(call)
                for index, call in enumerate(trajectory.calls)
            )
            if last
            else None
        ),
    )
    reasons = {
        key: None
        if value is not None
        else "source_access_timing_missing"
        if key == "time_to_first_localization_sec" and access
        else access_reason
        if key in {"turns_to_first_localization", "time_to_first_localization_sec"}
        else "edit_call_timing_missing"
        if successful and key.startswith("time_")
        else reason
        for key, value in primary.items()
    }
    first_attempt = ordered[0] if ordered else None
    access_calls = {row["call_index"]: row for row in ordered}.values()
    return primary, dict(
        first_target_access_attempt_turn=first_attempt["turn"] if first_attempt else None,
        first_target_access_attempt_sec=first_attempt["started_sec"] if first_attempt else None,
        first_target_access_attempt_outcome=first_attempt["call_outcome"] if first_attempt else None,
        first_target_access_attempt_scope=(
            "earliest command/path attempt including failed and unconfirmed calls; "
            "not proof of returned source bytes or successful completion"
        ),
        target_access_attempts_by_outcome={
            state: sum(row["call_outcome"] == state for row in access_calls)
            for state in ("success", "failed", "unknown")
        },
        target_access_metric_reasons={
            key: reasons[key]
            for key in ("turns_to_first_localization", "time_to_first_localization_sec")
        },
        target_edit_attempts_by_outcome={
            state: sum(row["call_outcome"] == state for row in edits)
            for state in ("success", "failed", "unknown")
        },
        target_edit_metric_reasons=reasons,
        target_access_scope=(
            "all source-access attempts retained; primary metrics require a successful containing "
            "call; command/path proxy, not semantic root cause or proof of repaired bytes"
        ),
        target_edit_time_scope="successful containing call start; not exact filesystem write time",
        npu_runs_before_last_target_edit_scope=(
            "recognized separate native command attempts preceding the edit call; intra-call ordering unmeasured"
        ),
    )


def _validation_tail(trajectory):
    successes, tests, polls = [], [], []
    jobs = {}
    for index, call in enumerate(trajectory.calls):
        observed = call.result_observation
        if _npu_test_call(call):
            tests.append(index)
            if observed is not None and observed.job_id:
                jobs[observed.job_id] = index
            if _call_outcome(call) == "success":
                successes.append(index)
        elif (
            observed is not None
            and observed.job_id in jobs
            and observed.classification == "business_success"
        ):
            successes.append(index)
        if (
            tool_name(call.name) == "remote_job_stdin"
            and not call.arguments.get("chars")
            and not call.arguments.get("eof")
        ):
            polls.append(index)
    first = min(successes) if successes else None
    empty_polls = [trajectory.calls[index] for index in polls]
    return dict(
        first_recorded_success_call_index=first,
        total_empty_job_polls=len(polls),
        empty_job_poll_definition="no chars/eof input; may return output; not an idle-poll count",
        empty_job_poll_timing=_duration_profile(empty_polls),
        recognized_test_attempts_after_first_recorded_success=(
            sum(index > first for index in tests) if first is not None else None
        ),
        empty_job_polls_after_first_recorded_success=(
            sum(index > first for index in polls) if first is not None else None
        ),
        scope="recognized command/result observations; not kernel counts or proof of public reproduction",
        is_stop_rule=False,
    )



def trajectory_metrics(
    trajectory: Trajectory,
    task: TaskSpec,
    harness: HarnessConfig,
    wall_clock: float,
    *,
    tier: str,
    source_visibility: dict | None = None,
    broker_knowledge: dict | None = None,
) -> tuple[dict, dict]:
    remote = [c for c in trajectory.calls if remote_execution(c)]
    all_remote = [
        c
        for c in trajectory.calls
        if c.component == "remote-dev" or tool_name(c.name).startswith("remote_") or remote_execution(c)
    ]
    knowledge = [
        c for c in trajectory.calls if tool_name(c.name) in {"knowledge_query", "knowledge_explain"}
    ]

    remote_timing = _duration_profile(remote)
    all_remote_timing = _duration_profile(all_remote)
    knowledge_timing = _duration_profile(knowledge)
    combined_timing = _duration_profile(all_remote + knowledge)
    native_knowledge_timing = knowledge_timing
    native_combined_timing = combined_timing
    timing_reason = None
    if broker_knowledge is not None:
        if broker_knowledge.get("schema") != BROKER_KNOWLEDGE_SCHEMA:
            raise ValueError("unsupported broker knowledge observation schema")
        knowledge_timing = broker_knowledge["timing"]
        if broker_knowledge["relative_intervals"] is None:
            combined_timing = dict(
                calls=len(all_remote) + broker_knowledge["admitted_calls"],
                observed_intervals=None,
                missing_intervals=None,
                complete=False,
                raw_cumulative_sec=None,
                observed_union_sec=None,
                overlap_sec=None,
                coverage=None,
                first_started_sec=None,
                last_ended_sec=None,
            )
            timing_reason = broker_knowledge["alignment_reason"]
        else:
            observed_calls = [
                ToolCall(id=f"broker-knowledge-{index}", name="knowledge", turn=1, **interval)
                for index, interval in enumerate(broker_knowledge["relative_intervals"])
            ]
            combined_timing = _duration_profile(all_remote + observed_calls)
    remote_sec = remote_timing["observed_union_sec"] if remote_timing["complete"] else None
    knowledge_sec = knowledge_timing["observed_union_sec"] if knowledge_timing["complete"] else None
    remainder = None
    within_wall = (
        combined_timing["last_ended_sec"] is None or combined_timing["last_ended_sec"] <= wall_clock
    )
    if combined_timing["complete"] and within_wall:
        remainder = wall_clock - combined_timing["observed_union_sec"]
    access, unresolved = target_access_observations(trajectory, task.mindie_knowledge.target_files)
    edit_primary, edit_auxiliary = _edit_metrics(trajectory, access, unresolved)
    successful_accesses = []
    for row in access:
        if row["call_outcome"] != "success":
            continue
        evidence = row["outcome_evidence"]
        if evidence["basis"] == "later_same_remote_job_terminal":
            # Preserve the original attempt's timestamps, but do not claim its
            # pending RPC completion was already evidence of command success.
            row = dict(
                row,
                turn=evidence["confirmation_turn"],
                ended_sec=evidence["confirmation_ended_sec"],
            )
        successful_accesses.append(row)
    source_primary, source_auxiliary = source_evidence_metrics(source_visibility, successful_accesses)
    source_auxiliary["source_evidence_basis"]["job_confirmation"] = (
        "later terminal confirmation turn/end when the original command result was pending; "
        "original access/edit start remains a command-attempt proxy"
    )
    rates = [
        harness.input_usd_per_million,
        harness.cached_usd_per_million,
        harness.output_usd_per_million,
    ]
    cost = None
    if all(rate is not None for rate in rates):
        cost = (
            sum(
                v * rate
                for v, rate in zip(
                    [
                        trajectory.tokens.input_uncached,
                        trajectory.tokens.input_cached,
                        trajectory.tokens.output,
                    ],
                    rates,
                    strict=True,
                )
            )
            / 1_000_000
        )
    if cost is not None and trajectory.tokens.cache_creation:
        if harness.cache_creation_usd_per_million is None:
            cost = None
        else:
            cost += (
                trajectory.tokens.cache_creation
                * (harness.cache_creation_usd_per_million - harness.input_usd_per_million)
                / 1_000_000
            )
    primary = dict(
        total_wall_clock_sec=wall_clock,
        npu_execution_wait_sec=remote_sec,
        knowledge_query_sec=knowledge_sec,
        knowledge_measurement_basis=(
            BROKER_KNOWLEDGE_BASIS if broker_knowledge is not None else "native_tool_trajectory/1"
        ),
        foreground_knowledge_queries_count=(
            broker_knowledge["query_count"]
            if broker_knowledge is not None
            else sum(tool_name(c.name) == "knowledge_query" for c in knowledge)
        ),
        foreground_knowledge_explains_count=(
            broker_knowledge["explain_count"]
            if broker_knowledge is not None
            else sum(tool_name(c.name) == "knowledge_explain" for c in knowledge)
        ),
        llm_and_exploration_sec=remainder,
        total_turns=trajectory.total_turns,
        npu_test_runs=npu_test_runs(trajectory) if tier == "l3" else None,
        remote_test_invocations=npu_test_runs(trajectory),
        localization_penalty_turns=task.budget.max_turns,
        zero_human_intervention=(
            trajectory.human_interventions == 0 if trajectory.human_interventions is not None else None
        ),
        tokens=trajectory.tokens.model_dump(),
        estimated_cost_usd=cost,
        cost_complete=trajectory.tokens.complete and cost is not None,
        **edit_primary,
        **source_primary,
    )
    tool_observations = summarize_tool_observations(all_remote)
    auxiliary = dict(
        knowledge_queries_count=sum(
            tool_name(c.name) in {"knowledge_query", "query"} for c in knowledge
        ),
        knowledge_explains_count=sum(
            tool_name(c.name) in {"knowledge_explain", "explain"} for c in knowledge
        ),
        knowledge_measurement=(
            dict(
                basis=BROKER_KNOWLEDGE_BASIS,
                query_count=broker_knowledge["query_count"],
                explain_count=broker_knowledge["explain_count"],
                admitted_calls=broker_knowledge["admitted_calls"],
                completed_calls=broker_knowledge["completed_calls"],
                successful_calls=broker_knowledge["successful_calls"],
                failed_calls=broker_knowledge["failed_calls"],
                incomplete_calls=broker_knowledge["incomplete_calls"],
                timing_complete=knowledge_timing["complete"],
                scope=broker_knowledge["scope"],
            )
            if broker_knowledge is not None
            else dict(
                basis="native_tool_trajectory/1",
                query_count=sum(tool_name(c.name) == "knowledge_query" for c in knowledge),
                explain_count=sum(tool_name(c.name) == "knowledge_explain" for c in knowledge),
                scope="native named tool calls only; shell-launched MCP clients unmeasured",
            )
        ),
        broker_knowledge=broker_knowledge,
        remote_tool_calls_count=len(all_remote),
        remote_tool_errors=sum(c.is_error is True for c in all_remote),
        remote_tool_non_business_errors=tool_observations["non_business_error_count"],
        tool_observations=tool_observations,
        npu_command_observations=dict(
            basis="literal_command_position/1",
            calls=[
                dict(
                    call_id=call.id,
                    **{
                        key: value
                        for key, value in observe_npu_command(
                            call.arguments.get("command", call.arguments.get("cmd", ""))
                        ).items()
                        if key != "scope"
                    },
                )
                for call in remote
            ],
            scope=(
                "recognized command-attempt lower bound/proxy; not complete hardware/kernel "
                "execution counts; unrecognized opaque commands may execute tests"
            ),
        ),
        target_access_observations=access,
        target_access_unresolved=unresolved,
        validation_tail=_validation_tail(trajectory),
        tool_timing=dict(
            remote=remote_timing,
            all_remote=all_remote_timing,
            knowledge=knowledge_timing,
            combined=combined_timing,
            native_knowledge=native_knowledge_timing,
            native_combined=native_combined_timing,
            cross_component_overlap_sec=(
                all_remote_timing["observed_union_sec"]
                + knowledge_timing["observed_union_sec"]
                - combined_timing["observed_union_sec"]
                if combined_timing["observed_union_sec"] is not None
                else None
            ),
            remainder_reason=(
                timing_reason
                if timing_reason
                else "missing_tool_intervals"
                if not combined_timing["complete"]
                else "tool_interval_outside_wall_clock"
                if remainder is None
                else None
            ),
            within_wall_clock=within_wall,
            scope="execution union is not pure NPU compute; combined is all remote/knowledge call union",
            knowledge_measurement_basis=(
                BROKER_KNOWLEDGE_BASIS if broker_knowledge is not None else "native_tool_trajectory/1"
            ),
        ),
        tool_timing_complete=combined_timing["complete"],
        timing_overlaps_or_missing=(
            not combined_timing["complete"] or combined_timing["overlap_sec"] > 0 or remainder is None
        ),
        hit_oracle_task_in_top3=None,
        hit_oracle_block=None,
        **edit_auxiliary,
        **source_auxiliary,
    )
    return primary, auxiliary


