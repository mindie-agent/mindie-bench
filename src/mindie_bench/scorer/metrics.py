"""Outcome, effort, and cost metrics with explicit missing-data semantics."""

from __future__ import annotations

import ast
import json
import math
import posixpath
import re
import shlex
from collections import Counter, defaultdict
from statistics import mean

from mindie_bench.evaluation.broker_knowledge import BASIS as BROKER_KNOWLEDGE_BASIS
from mindie_bench.evaluation.broker_knowledge import SCHEMA as BROKER_KNOWLEDGE_SCHEMA
from mindie_bench.evaluation.source_visibility import source_evidence_metrics
from mindie_bench.models import (
    HarnessConfig,
    TaskSpec,
    TerminalStatus,
    ToolCall,
    Trajectory,
    TrialReceipt,
)
from mindie_bench.scorer.command_observations import (
    is_recognized_npu_command,
    observe_npu_command,
)
from mindie_bench.scorer.tool_observations import summarize_tool_observations

REMOTE = {"remote_bash", "remote_job_start", "remote_exec", "remote_command"}
READ_EDIT = {
    "remote_read",
    "Read",
    "remote_edit",
    "Edit",
    "remote_grep",
    "Grep",
    "read_file",
    "edit_file",
    "remote_apply_patch",
    "apply_patch",
    "remote_write",
    "write_file",
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


def _python_accesses(code: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Follow literal path aliases and executed syntax, never strings containing code."""
    accesses, unresolved, names = [], [], {}
    path_factories = {"Path", "PurePath"}
    local_callables = set()
    if len(code) > 262_144:
        return [], ["python_static_parser_limit"]
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return [], ["python_syntax_unresolved"]

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return names.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 64:
                unresolved.append("python_literal_sequence_limit")
                return None
            if all(isinstance(part, ast.Constant) and isinstance(part.value, str) for part in node.elts):
                return tuple(part.value for part in node.elts)
        if isinstance(node, ast.Subscript):
            sequence = value(node.value)
            index = node.slice
            sign = 1
            if isinstance(index, ast.UnaryOp) and isinstance(index.op, (ast.USub, ast.UAdd)):
                sign = -1 if isinstance(index.op, ast.USub) else 1
                index = index.operand
            if isinstance(sequence, tuple):
                if isinstance(index, ast.Constant) and type(index.value) is int:
                    position = sign * index.value
                    if -len(sequence) <= position < len(sequence):
                        return sequence[position]
                unresolved.append("python_literal_sequence_index_unresolved")
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id in path_factories
            and node.func.id not in local_callables
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "Path"
        ):
            parts = [value(arg) for arg in node.args]
            return posixpath.join(*parts) if parts and all(isinstance(p, str) for p in parts) else None
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            if isinstance(left, str) and isinstance(right, str):
                if isinstance(node.op, ast.Div):
                    return posixpath.join(left, right)
                if isinstance(node.op, ast.Add):
                    return left + right
        return None

    def forget_sequence(sequence):
        for name in [name for name, known in names.items() if known is sequence]:
            names.pop(name)
        unresolved.append("python_literal_sequence_invalidated")

    def escape(node):
        """An untracked receiver or argument may mutate any alias of a list."""
        known = value(node)
        if isinstance(known, tuple):
            forget_sequence(known)
        elif not isinstance(known, str):
            for child in ast.iter_child_nodes(node):
                escape(child)

    def forget_target(node, *, inplace=False):
        if isinstance(node, ast.Name):
            known = names.get(node.id)
            if inplace and isinstance(known, tuple):
                forget_sequence(known)
            names.pop(node.id, None)
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            escape(node.value)
        else:
            for child in ast.iter_child_nodes(node):
                forget_target(child, inplace=inplace)

    def forget_imports(node):
        for alias in node.names:
            name = alias.asname or (
                alias.name if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
            )
            names.pop(name, None)

    def invalidate_call(node):
        if isinstance(node.func, ast.Name) and node.func.id in local_callables:
            # An opaque local callable may change captured/global containers.
            names.clear()
            unresolved.append("python_opaque_local_callable")
        if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            names.clear()
            unresolved.append("python_dynamic_namespace")
        if isinstance(node.func, ast.Attribute):
            receiver = value(node.func.value)
            if isinstance(receiver, tuple):
                forget_sequence(receiver)
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            escape(argument)

    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    controls = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.TryStar,
        ast.AsyncFor,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.IfExp,
        ast.BoolOp,
        ast.Match,
    )

    def invalidate_control(node):
        # Inspect possible rebinding/escape only; never count a branch's file
        # operations as having run. Alias creation inside a branch is opaque.
        if isinstance(node, definitions):
            if getattr(node, "name", None):
                local_callables.add(node.name)
            names.pop(getattr(node, "name", None), None)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            forget_imports(node)
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            names.pop(node.name, None)
        if isinstance(node, ast.MatchMapping):
            names.pop(node.rest, None)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if node.value is not None:
                escape(node.value)
        if isinstance(node, ast.Call):
            invalidate_call(node)
        if isinstance(node, ast.AugAssign):
            forget_target(node.target, inplace=True)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.pop(node.id, None)
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            escape(node.value)
        for child in ast.iter_child_nodes(node):
            invalidate_control(child)

    def visit(node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            forget_imports(node)
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            path_factories.update(
                alias.asname or alias.name for alias in node.names if alias.name in {"Path", "PurePath"}
            )
        if isinstance(node, definitions):
            # A definition or a copied test body is not evidence of its execution.
            if getattr(node, "name", None):
                local_callables.add(node.name)
            names.pop(getattr(node, "name", None), None)
            unresolved.append("python_definition_body_not_observed")
            return
        if isinstance(node, controls):
            invalidate_control(node)
            unresolved.append("python_dynamic_control_flow")
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                visit(node.value)
            resolved = value(node.value) if node.value is not None else None
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            callable_alias = isinstance(node.value, ast.Lambda) or (
                isinstance(node.value, ast.Name) and node.value.id in local_callables
            )
            if resolved is None and node.value is not None:
                escape(node.value)
            for target in targets:
                forget_target(target)
                if isinstance(target, ast.Name):
                    local_callables.discard(target.id)
                    if callable_alias:
                        local_callables.add(target.id)
                    if resolved is not None:
                        names[target.id] = resolved
            return
        if isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            forget_target(node.target, inplace=True)
            if node.value is not None:
                visit(node.value)
                escape(node.value)
            unresolved.append("python_assignment_unresolved")
            return
        if isinstance(node, ast.Delete):
            for target in node.targets:
                forget_target(target)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.pop(node.id, None)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                path = value(node.args[0]) if node.args else None
                mode = value(node.args[1]) if len(node.args) > 1 else "r"
                mode = next((value(k.value) for k in node.keywords if k.arg == "mode"), mode)
                if isinstance(path, str) and isinstance(mode, str):
                    accesses.append(("edit" if any(flag in mode for flag in "wax+") else "read", path))
                else:
                    unresolved.append("python_dynamic_open_path_or_mode")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                "read_text",
                "read_bytes",
                "write_text",
                "write_bytes",
            }:
                path = value(node.func.value)
                if isinstance(path, str):
                    accesses.append(("edit" if node.func.attr.startswith("write") else "read", path))
                else:
                    unresolved.append("python_dynamic_path")
            invalidate_call(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return accesses, sorted(set(unresolved))


def _shell_accesses(command: str, *, depth=0, remote=False, stdin=None):
    """Parse bounded shell structure without executing it or inspecting payload files."""
    if depth > 12 or len(command) > 262_144:
        return [], ["command_static_parser_limit"]
    accesses, unresolved = [], []
    # Quoted arguments belong to the receiving program; only recursive shell
    # and SSH wrappers may interpret their redirections. Cat/echo bodies stay data.
    heredoc_pattern = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z_0-9]*)\1[ \t]*\n")
    heredoc, quote, index = None, None, 0
    while index < len(command):
        char = command[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "#" and (index == 0 or command[index - 1] in " \t\r\n;&|()"):
            end = command.find("\n", index)
            if end < 0:
                break
            index = end
        elif char == "<":
            heredoc = heredoc_pattern.match(command, index)
            if heredoc:
                break
        index += 1
    if heredoc:
        end = re.search(rf"(?m)^\t*{re.escape(heredoc[2])}[ \t]*(?:\n|$)", command[heredoc.end() :])
        if end is None:
            return [], ["heredoc_boundary_unresolved"]
        body = command[heredoc.end() : heredoc.end() + end.start()]
        header = command[: heredoc.start()]
        tail = command[heredoc.end() + end.end() :]
        first, reasons = _shell_accesses(header, depth=depth + 1, remote=remote, stdin=body)
        rest, more = _shell_accesses(tail, depth=depth + 1, remote=remote) if tail.strip() else ([], [])
        return first + rest, reasons + more
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [], ["shell_syntax_unresolved"]
    segments, current = [], []
    for token in tokens:
        if token in {";", "&&", "||", "|", "&", "\n", "(", ")"}:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    for words in segments:
        while words and (
            re.match(r"^[A-Za-z_]\w*=", words[0]) or words[0] in {"env", "command", "exec"}
        ):
            words = words[1:]
        if not words:
            continue
        name = posixpath.basename(words[0])
        if name in {"sh", "bash", "zsh", "dash"}:
            option = next((i for i, word in enumerate(words) if word in {"-c", "-lc", "-ic"}), None)
            if option is not None and option + 1 < len(words):
                found, reasons = _shell_accesses(
                    words[option + 1], depth=depth + 1, remote=remote, stdin=stdin
                )
                accesses.extend(found)
                unresolved.extend(reasons)
            continue
        if name == "ssh":
            index = 1
            while index < len(words) and words[index].startswith("-"):
                index += 2 if words[index] in {"-p", "-o", "-i", "-l", "-F", "-J", "-S", "-W"} else 1
            remaining = words[index + 1 :]
            if remaining:
                script = remaining[0] if len(remaining) == 1 else shlex.join(remaining)
                found, reasons = _shell_accesses(script, depth=depth + 1, remote=True, stdin=stdin)
                accesses.extend(found)
                unresolved.extend(reasons)
            continue
        redirected = set()
        for i, word in enumerate(words[:-1]):
            if word in {">", ">>", "<"}:
                accesses.append(("read" if word == "<" else "edit", words[i + 1], remote))
                redirected.update({i, i + 1})
        words = [word for i, word in enumerate(words) if i not in redirected]
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name):
            code = (
                words[words.index("-c") + 1]
                if "-c" in words and words.index("-c") + 1 < len(words)
                else stdin
            )
            if code is not None:
                found, reasons = _python_accesses(code)
                accesses.extend((kind, path, remote) for kind, path in found)
                unresolved.extend(reasons)
            else:
                unresolved.append("python_script_or_dynamic_input_unobserved")
            continue
        if name in {"apply_patch", "patch"}:
            if stdin and name == "apply_patch":
                accesses.extend(
                    ("edit", match, remote)
                    for match in re.findall(r"(?m)^\*\*\* (?:Update|Add|Delete) File: (.+)$", stdin)
                )
            else:
                unresolved.append("patch_target_payload_unobserved")
            continue
        if name not in {"cat", "sed", "grep", "rg", "head", "tail"}:
            continue
        if name == "rg" and "--files" in words:
            continue
        operands, explicit_pattern, skip = [], False, False
        options_with_value = {
            "-e",
            "-f",
            "-g",
            "--glob",
            "-t",
            "--type",
            "-n",
            "-c",
            "-A",
            "-B",
            "-C",
            "-m",
            "--max-count",
            "--max-depth",
        }
        for i, word in enumerate(words[1:]):
            if skip:
                skip = False
                continue
            if word in {"-e", "-f"}:
                explicit_pattern = True
                skip = True
            elif word in options_with_value and (word not in {"-n", "-c"} or name in {"head", "tail"}):
                skip = True
            elif word.startswith("-"):
                continue
            else:
                operands.append(word)
        if name in {"grep", "rg", "sed"} and not explicit_pattern and operands:
            operands = operands[1:]
        kind = (
            "edit"
            if name == "sed" and any(word == "-i" or word.startswith("-i") for word in words[1:])
            else "read"
        )
        accesses.extend((kind, path, remote) for path in operands)
    return accesses, sorted(set(unresolved))


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
            tool_name(other.name) in {"remote_job_status", "remote_job_tail", "remote_job_stdin"}
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
                    tool_name(other.name) in {"remote_job_status", "remote_job_tail", "remote_job_stdin"}
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
                    "Edit",
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


def _measurement_coverage(known: list[bool], *, reason: str) -> dict:
    total, observed = len(known), sum(known)
    state = (
        "empty"
        if not total
        else "complete"
        if observed == total
        else "not_measured"
        if not observed
        else "partial"
    )
    return dict(
        population_attempts=total,
        observed_attempts=observed,
        missing_attempts=total - observed,
        coverage=observed / total if total else None,
        state=state,
        reason="empty_input" if not total else None if observed == total else reason,
    )


def _known_count(value) -> bool:
    return type(value) is int and value >= 0


def _matrix_coverage(receipts, expected_cells, blocked_cells) -> dict:
    result = dict(
        planned_cells=None,
        observed_terminal_cells=len(receipts),
        valid_outcome_cells=sum(
            row.terminal_status in {TerminalStatus.PASS, TerminalStatus.FAIL} for row in receipts
        ),
        missing_terminal_cells=None,
        blocked_cells=None,
        unclassified_missing_cells=None,
        unexpected_terminal_cells=None,
        identity_mismatches=None,
        terminal_receipts_complete=None,
        all_planned_cells_accounted_for=None,
        complete=None,
        reason="plan_inventory_unavailable",
    )
    if expected_cells is None:
        if blocked_cells is not None:
            raise ValueError("blocked cells require an explicit expected cell inventory")
        return result

    def key(cell):
        return cell["instance_id"], cell["tier"], cell["group"], cell["repetition"]

    def indexed(cells):
        indexed_cells = {}
        for cell in cells:
            if (
                not isinstance(cell, dict)
                or any(
                    not isinstance(cell.get(field), str) or not cell[field]
                    for field in ("instance_id", "tier", "group", "task_digest")
                )
                or cell.get("role") not in {"eval", "seed"}
                or not _known_count(cell.get("repetition"))
                or cell["repetition"] < 1
            ):
                raise ValueError(
                    "expected/blocked cells require explicit task, tier, group, repetition, role and digest"
                )
            identity = key(cell)
            if identity in indexed_cells:
                raise ValueError("duplicate declared matrix cell")
            indexed_cells[identity] = {
                field: cell[field]
                for field in ("instance_id", "tier", "group", "repetition", "role", "task_digest")
            }
        return indexed_cells

    expected = indexed(expected_cells)
    blocked = indexed(blocked_cells) if blocked_cells is not None else None
    actual = {
        (row.instance_id, row.tier, row.ablation_group.value, row.repetition): row for row in receipts
    }
    if blocked is not None:
        if blocked.keys() - expected.keys():
            raise ValueError("blocked cells are outside the expected cell inventory")
        if blocked.keys() & actual.keys():
            raise ValueError("a blocked cell cannot also have a terminal receipt")
        if any(
            any(
                blocked[identity][field] != expected[identity][field]
                for field in ("role", "task_digest")
            )
            for identity in blocked
        ):
            raise ValueError("blocked cell identity differs from the expected inventory")
    missing = expected.keys() - actual.keys()
    unexpected = actual.keys() - expected.keys()
    mismatches = [
        dict(cell=expected[identity], fields=fields)
        for identity in expected.keys() & actual.keys()
        if (
            fields := [
                field
                for field, value in (
                    ("role", actual[identity].task_role),
                    ("task_digest", actual[identity].task_digest),
                )
                if expected[identity][field] != value
            ]
        )
    ]
    duplicate_actual = len(actual) != len(receipts)
    terminal_complete = bool(expected) and not (missing or unexpected or mismatches or duplicate_actual)
    complete = terminal_complete and result["valid_outcome_cells"] == len(expected)
    result.update(
        planned_cells=len(expected),
        missing_terminal_cells=[expected[identity] for identity in sorted(missing)],
        blocked_cells=list(blocked.values()) if blocked is not None else None,
        unclassified_missing_cells=(
            [expected[identity] for identity in sorted(missing - blocked.keys())]
            if blocked is not None
            else None
        ),
        unexpected_terminal_cells=[
            dict(instance_id=identity[0], tier=identity[1], group=identity[2], repetition=identity[3])
            for identity in sorted(unexpected)
        ],
        identity_mismatches=mismatches,
        duplicate_terminal_cells=duplicate_actual,
        terminal_receipts_complete=terminal_complete,
        all_planned_cells_accounted_for=(
            not (missing - blocked.keys() or unexpected or mismatches or duplicate_actual)
            if blocked is not None
            else True
            if terminal_complete
            else None
        ),
        complete=complete,
        reason=(
            "empty_plan"
            if not expected
            else "cell_identity_mismatch"
            if mismatches or unexpected or duplicate_actual
            else "missing_terminal_cells"
            if missing
            else "invalid_terminal_outcomes"
            if not complete
            else None
        ),
    )
    return result


def summarize(
    receipts: list[TrialReceipt],
    *,
    k: int = 3,
    fast_p: float = 1.15,
    expected_cells: list[dict] | None = None,
    blocked_cells: list[dict] | None = None,
) -> dict:
    if k < 1 or not math.isfinite(fast_p) or fast_p <= 0:
        raise ValueError("k and performance threshold must be positive")
    statuses = Counter(r.terminal_status.value for r in receipts)
    valid = [r for r in receipts if r.terminal_status in {TerminalStatus.PASS, TerminalStatus.FAIL}]
    passed = sum(r.terminal_status == TerminalStatus.PASS for r in valid)
    by_task = defaultdict(list)
    for receipt in valid:
        by_task[receipt.instance_id].append(receipt)
    # Do not cube an overall mixture of easy/hard tasks. Report the requested
    # plug-in estimate per task, and only where k valid repetitions exist.
    reliable = [
        rows
        for rows in by_task.values()
        if len({r.repetition for r in rows}) == len(rows) and len(rows) >= k
    ]
    pass_pow_k = (
        mean(
            (sum(r.terminal_status == TerminalStatus.PASS for r in rows) / len(rows)) ** k
            for rows in reliable
        )
        if reliable
        else None
    )
    unbiased = (
        mean(
            math.comb(sum(r.terminal_status == TerminalStatus.PASS for r in rows), k)
            / math.comb(len(rows), k)
            for rows in reliable
        )
        if reliable
        else None
    )
    costs = [r.primary_metrics.get("estimated_cost_usd") for r in receipts]
    cost_complete = bool(receipts) and all(r.primary_metrics.get("cost_complete") for r in receipts)
    total_cost = sum(c for c in costs if c is not None)
    known_times = [
        r.primary_metrics["total_wall_clock_sec"]
        for r in valid
        if r.primary_metrics.get("total_wall_clock_sec") is not None
    ]
    delivery_times = [
        r.primary_metrics["delivery_wall_clock_sec"]
        for r in valid
        if r.primary_metrics.get("delivery_wall_clock_sec") is not None
    ]
    interventions = [r.primary_metrics.get("zero_human_intervention") for r in valid]
    performance = [r.primary_metrics.get("performance_metrics", {}).get("speedup_ratio") for r in valid]
    speedup_known = all(v is not None for v in performance) and bool(valid)
    token_fields = ["input_uncached", "input_cached", "output", "indexer_small_model_total"]
    observed_tokens = {
        field: sum(r.primary_metrics.get("tokens", {}).get(field, 0) for r in receipts)
        for field in token_fields
    }
    delivery_all = [r.primary_metrics.get("delivery_wall_clock_sec") for r in receipts]
    wall_all = [r.primary_metrics.get("total_wall_clock_sec") for r in receipts]
    remote_errors = [r.auxiliary_metrics.get("remote_tool_non_business_errors") for r in receipts]
    query_counts = [
        r.auxiliary_metrics.get("knowledge_measurement", {}).get(
            "query_count",
            r.primary_metrics.get(
                "foreground_knowledge_queries_count", r.auxiliary_metrics.get("knowledge_queries_count")
            ),
        )
        for r in receipts
    ]
    explain_counts = [
        r.auxiliary_metrics.get("knowledge_measurement", {}).get(
            "explain_count",
            r.primary_metrics.get(
                "foreground_knowledge_explains_count",
                r.auxiliary_metrics.get("knowledge_explains_count"),
            ),
        )
        for r in receipts
    ]
    query_coverage = _measurement_coverage(
        [_known_count(value) for value in query_counts], reason="mcp_count_missing"
    )
    explain_coverage = _measurement_coverage(
        [_known_count(value) for value in explain_counts], reason="mcp_count_missing"
    )
    query_total = (
        sum(value for value in query_counts if _known_count(value))
        if query_coverage["observed_attempts"]
        else None
    )
    explain_total = (
        sum(value for value in explain_counts if _known_count(value))
        if explain_coverage["observed_attempts"]
        else None
    )
    mcp_complete = query_coverage["state"] == explain_coverage["state"] == "complete"
    knowledge_state = (
        "empty"
        if not receipts
        else "observed_zero_mcp_calls"
        if mcp_complete and query_total == explain_total == 0
        else "observed_mcp_calls"
        if mcp_complete
        else "not_measured"
        if query_total is None and explain_total is None
        else "partial"
    )
    knowledge = dict(
        state=knowledge_state,
        measurement_bases=sorted(
            {
                row.auxiliary_metrics.get("knowledge_measurement", {}).get(
                    "basis",
                    row.primary_metrics.get(
                        "knowledge_measurement_basis", "legacy_native_tool_trajectory_only"
                    ),
                )
                for row in receipts
            }
        ),
        query_count=query_total if query_coverage["state"] == "complete" else None,
        explain_count=explain_total if explain_coverage["state"] == "complete" else None,
        broker_count_coverage=_measurement_coverage(
            [
                row.primary_metrics.get("knowledge_measurement_basis") == BROKER_KNOWLEDGE_BASIS
                or row.auxiliary_metrics.get("knowledge_measurement", {}).get("basis")
                == BROKER_KNOWLEDGE_BASIS
                for row in receipts
            ],
            reason="shell_launched_mcp_clients_not_covered_by_native_tool_trajectory",
        ),
        observed_query_count_lower_bound=query_total,
        observed_explain_count_lower_bound=explain_total,
        attempts_with_observed_queries=sum(_known_count(value) and value > 0 for value in query_counts),
        attempts_with_observed_explains=sum(
            _known_count(value) and value > 0 for value in explain_counts
        ),
        query_count_coverage=query_coverage,
        explain_count_coverage=explain_coverage,
        native_body_access="not_measured_by_mcp_counters",
        benefit=dict(
            value=None,
            state="not_measured",
            reason=(
                "empty_input"
                if not receipts
                else "mcp_observation_incomplete"
                if not mcp_complete
                else "no_observed_mcp_retrieval"
                if query_total == explain_total == 0
                else "causal_protocol_not_supplied"
            ),
        ),
    )
    measurement_coverage = dict(
        wall_clock_valid=_measurement_coverage(
            [r.primary_metrics.get("total_wall_clock_sec") is not None for r in valid],
            reason="timing_missing",
        ),
        delivery_valid=_measurement_coverage(
            [r.primary_metrics.get("delivery_wall_clock_sec") is not None for r in valid],
            reason="timing_missing",
        ),
        wall_clock_all_attempts=_measurement_coverage(
            [value is not None for value in wall_all], reason="timing_missing"
        ),
        delivery_all_attempts=_measurement_coverage(
            [value is not None for value in delivery_all], reason="timing_missing"
        ),
        tokens=_measurement_coverage(
            [r.primary_metrics.get("tokens", {}).get("complete") is True for r in receipts],
            reason="usage_incomplete_or_missing",
        ),
        localization=_measurement_coverage(
            [r.primary_metrics.get("turns_to_first_localization") is not None for r in receipts],
            reason="source_access_proxy_unobserved",
        ),
        remote_non_business_errors=_measurement_coverage(
            [_known_count(value) for value in remote_errors], reason="classification_missing"
        ),
        knowledge_queries=query_coverage,
        knowledge_explains=explain_coverage,
    )
    measurement_coverage["tokens"]["recorded_attempts"] = sum(
        isinstance(r.primary_metrics.get("tokens"), dict) for r in receipts
    )
    progress_metrics = {}
    for field in (
        "time_to_first_localization_sec",
        "turns_to_first_target_edit",
        "time_to_first_target_edit_sec",
        "turns_to_last_target_edit",
        "time_to_last_target_edit_sec",
        "npu_runs_before_last_target_edit",
    ):
        values = [row.primary_metrics.get(field) for row in receipts]
        coverage = _measurement_coverage(
            [value is not None for value in values], reason="target_progress_observation_missing"
        )
        progress_metrics[field] = dict(
            **coverage,
            mean_recorded_value=mean(value for value in values if value is not None)
            if coverage["observed_attempts"]
            else None,
            population="all_attempts_with_recorded_proxy",
        )
    observed_tasks = sorted({r.instance_id for r in receipts})
    eligible_tasks = {rows[0].instance_id for rows in reliable}
    reliability_complete = bool(receipts) and (
        len(eligible_tasks) == len(observed_tasks) and len(valid) == len(receipts)
    )
    matrix = _matrix_coverage(receipts, expected_cells, blocked_cells)
    return dict(
        attempts=len(receipts),
        statuses={s.value: statuses[s.value] for s in TerminalStatus},
        valid_attempts=len(valid),
        completed_pass=passed,
        coverage=len(valid) / len(receipts) if receipts else None,
        pass_at_1=passed / len(valid) if valid else None,
        all_attempt_success_rate=passed / len(receipts) if receipts else None,
        pass_pow_k=pass_pow_k,
        pass_pow_k_unbiased=unbiased,
        reliability_k=k,
        reliability_eligible_tasks=len(reliable),
        reliability_total_tasks=len({r.instance_id for r in receipts}),
        reliability_estimator="mean_per_task_plugin_power; minimum k valid independent repeats",
        pass_pow_k_scope="eligible_observed_tasks_only",
        whole_matrix_pass_pow_k=(
            pass_pow_k if matrix["complete"] is True and reliability_complete else None
        ),
        reliability_coverage=dict(
            state=(
                "empty"
                if not receipts
                else "complete_for_observed_tasks"
                if reliability_complete
                else "partial"
                if reliable
                else "not_measured"
            ),
            required_valid_independent_repetitions=k,
            observed_tasks=len(observed_tasks),
            eligible_tasks=len(eligible_tasks),
            ineligible_tasks=[task for task in observed_tasks if task not in eligible_tasks],
            planned_tasks=(
                len({cell["instance_id"] for cell in expected_cells})
                if expected_cells is not None
                else None
            ),
            unobserved_planned_tasks=(
                sorted({cell["instance_id"] for cell in expected_cells} - set(observed_tasks))
                if expected_cells is not None
                else None
            ),
            per_task=[
                dict(
                    instance_id=task,
                    observed_attempts=sum(row.instance_id == task for row in receipts),
                    valid_independent_repetitions=len({row.repetition for row in by_task[task]}),
                    eligible=task in eligible_tasks,
                )
                for task in observed_tasks
            ],
            eligible_task_fraction=len(eligible_tasks) / len(observed_tasks) if observed_tasks else None,
            complete_for_observed_tasks=reliability_complete,
            complete_for_declared_cells=(
                matrix["complete"] and reliability_complete if expected_cells is not None else None
            ),
            reason=(
                "empty_input"
                if not receipts
                else "insufficient_valid_independent_repetitions"
                if not reliable
                else "partial_eligible_task_coverage"
                if not reliability_complete
                else "plan_inventory_unavailable"
                if expected_cells is None
                else matrix["reason"]
            ),
        ),
        matrix_coverage=matrix,
        mean_wall_clock_sec=mean(known_times) if known_times else None,
        mean_delivery_wall_clock_sec=mean(delivery_times) if delivery_times else None,
        mean_wall_clock_coverage=len(known_times),
        mean_delivery_wall_clock_count=len(delivery_times),
        valid_mean_population="completed_pass_or_completed_fail_with_recorded_timing",
        all_attempt_recorded_wall_clock_sec_mean=(
            mean(value for value in wall_all if value is not None)
            if any(value is not None for value in wall_all)
            else None
        ),
        all_attempt_recorded_wall_clock_sec_count=sum(value is not None for value in wall_all),
        all_attempt_recorded_delivery_wall_clock_sec_mean=(
            mean(value for value in delivery_all if value is not None)
            if any(value is not None for value in delivery_all)
            else None
        ),
        all_attempt_recorded_delivery_wall_clock_sec_count=sum(
            value is not None for value in delivery_all
        ),
        all_attempt_recorded_time_scope="recorded_elapsed_including_invalid_outcomes; not_successful_delivery",
        measurement_coverage=measurement_coverage,
        knowledge_observations=knowledge,
        target_progress_metrics=progress_metrics,
        observed_remote_non_business_errors_lower_bound=(
            sum(value for value in remote_errors if _known_count(value))
            if any(_known_count(value) for value in remote_errors)
            else None
        ),
        zero_intervention_rate=(
            sum(v is True for v in interventions) / len(valid)
            if valid and all(v is not None for v in interventions)
            else None
        ),
        observed_cost_usd_lower_bound=total_cost,
        cost_complete=cost_complete,
        cost_per_resolved_task=total_cost / passed if passed and cost_complete else None,
        observed_tokens_lower_bound=observed_tokens,
        token_metering_complete=bool(receipts)
        and all(r.primary_metrics.get("tokens", {}).get("complete") is True for r in receipts),
        observed_delivery_wall_clock_sec_lower_bound=sum(t for t in delivery_all if t is not None),
        delivery_timing_complete=bool(receipts) and all(t is not None for t in delivery_all),
        fast_p_threshold=fast_p,
        fast_p=(
            sum(
                r.terminal_status == TerminalStatus.PASS and speed > fast_p
                for r, speed in zip(valid, performance, strict=True)
            )
            / len(valid)
            if speedup_known
            else None
        ),
    )


def retrieval_metrics(ranked: list[str], relevant: set[str], *, k: int = 3) -> dict:
    if k < 1 or not relevant:
        raise ValueError("positive k and nonempty relevance labels required")
    if len(ranked) != len(set(ranked)):
        raise ValueError("duplicate retrieval IDs distort rank metrics")
    top = ranked[:k]
    hits = [index + 1 for index, item in enumerate(top) if item in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dict(
        recall_at_k=len(hits) / len(relevant),
        reciprocal_rank=1 / hits[0] if hits else 0.0,
        ndcg_at_k=dcg / ideal,
        hit_at_k=bool(hits),
    )


def routing_metrics(labels: list[tuple[bool, bool]]) -> dict:
    if not labels:
        raise ValueError("routing evaluation is empty")
    tp = sum(gold and predicted for gold, predicted in labels)
    fp = sum(not gold and predicted for gold, predicted in labels)
    fn = sum(gold and not predicted for gold, predicted in labels)
    negatives = sum(not gold for gold, _ in labels)
    return dict(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        zero_false_trigger_rate=1 - fp / negatives if negatives else None,
    )


def canonical_receipt(receipt: TrialReceipt) -> str:
    return json.dumps(receipt.model_dump(mode="json"), sort_keys=True, allow_nan=False)
