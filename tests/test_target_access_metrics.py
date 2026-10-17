import shlex

import pytest

from mindie_bench.models import ToolCall, ToolResultObservation, Trajectory
from mindie_bench.scorer.metrics import (
    _python_accesses,
    first_localization,
    target_access_observations,
    trajectory_metrics,
)

TARGET = "vllm_ascend/ops/layernorm.py"


def call(identity, command=None, *, turn=1, start=0, end=1, error=False, name="bash", **arguments):
    if command is not None:
        arguments["command"] = command
    return ToolCall(
        id=identity,
        name=name,
        arguments=arguments,
        turn=turn,
        started_sec=start,
        ended_sec=end,
        is_error=error,
    )


def metrics(case, calls, wall=20):
    task = case["spec"].model_copy(deep=True)
    task.mindie_knowledge.target_files = [TARGET]
    return trajectory_metrics(Trajectory(calls=calls), task, case["config"].harness, wall, tier="l3")


@pytest.mark.parametrize(
    "command",
    [
        f"cat {TARGET}",
        f"sed -n '34,55p' {TARGET}",
        f"rg -n 'quant_description|class AscendRMSNorm' {TARGET}",
        "/bin/zsh -c "
        + shlex.quote(
            "ssh -p 22 bench-slot " + shlex.quote(f"cd /bench/work && sed -n '34,55p' {TARGET}")
        ),
        f"python3 -c \"from pathlib import Path; p=Path('{TARGET}'); print(p.read_text())\"",
        f"python3 -c \"from pathlib import Path as P; p=P('{TARGET}'); print(p.read_text())\"",
        f"python3 - <<'PY'\nfrom pathlib import Path\np=Path('{TARGET}')\nprint(p.read_text())\nPY",
        f"bash -c 'true' && cat {TARGET}",
        "/bin/bash -c "
        + shlex.quote(
            f"python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('{TARGET}').read_text())\nPY"
        ),
        f"/bin/bash -c 'python3 -' <<'PY'\nfrom pathlib import Path\nprint(Path('{TARGET}').read_text())\nPY",
    ],
)
def test_shell_and_python_source_access_is_observed_without_execution(command):
    trajectory = Trajectory(calls=[call("read", command, turn=3)])
    assert first_localization(trajectory, [TARGET]) == 3


@pytest.mark.parametrize(
    "command",
    [
        f"echo 'cat {TARGET}'",
        f"# cat {TARGET}\necho harmless",
        f"grep '{TARGET}' other.txt",
        f"cat <<'PATCH'\n*** Update File: {TARGET}\nPATCH",
        f"python3 -c \"print('{TARGET}')\"",
        f"python3 -c \"from pathlib import Path; Path('patch.txt').write_text('cat {TARGET}')\"",
        f"python3 - <<'PY'\nfrom pathlib import Path\ndef unused():\n    return Path('{TARGET}').read_text()\nPY",
    ],
)
def test_filename_mentions_and_opaque_payloads_are_not_source_access(command):
    assert first_localization(Trajectory(calls=[call("mention", command)]), [TARGET]) is None


def _wrap_remote_script(script, wrapper):
    ssh = "ssh -p 22 -o BatchMode=yes bench-slot " + shlex.quote(script)
    if wrapper == "ssh":
        return ssh
    if wrapper == "zsh":
        return "/bin/zsh -c " + shlex.quote(ssh)
    if wrapper == "shells":
        return "bash -lc " + shlex.quote("dash -c " + shlex.quote(ssh))
    return "cd /workspace && env CONTRACT=1 command " + ssh + "; echo done"


@pytest.mark.parametrize("wrapper", ["ssh", "zsh", "shells", "compound"])
@pytest.mark.parametrize("action", ["read_text", "write_text"])
def test_remote_python_heredoc_recurses_through_literal_wrapper_arguments(wrapper, action, tmp_path):
    marker = tmp_path / "must-not-exist"
    operation = "p.read_text()" if action == "read_text" else "p.write_text('new')"
    script = (
        "python3 - <<'PY'\nfrom pathlib import Path\n"
        f"p=Path('{TARGET}')\n{operation}\nPath({str(marker)!r}).write_text('never')\nPY"
    )
    command = _wrap_remote_script(script, wrapper)
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("literal", command)]), [TARGET]
    )
    assert len(found) == 1
    assert found[0]["kind"] == ("read" if action == "read_text" else "edit")
    assert found[0]["execution_scope"] == "remote"
    assert unresolved == []
    assert not marker.exists()


def test_double_quoted_ssh_heredoc_preserves_successful_edit_turn_and_time(case):
    command = (
        "/bin/zsh -c \"ssh bench-slot \\\"python3 - <<'PY'\n"
        f"from pathlib import Path\np=Path('{TARGET}')\n"
        "s=p.read_text(); p.write_text(s + 'new')\nPY\\\"\""
    )
    primary, auxiliary = metrics(
        case,
        [
            call("read", f"cat {TARGET}", turn=4, start=14, end=15),
            call("edit", command, turn=14, start=63, end=64),
        ],
        wall=100,
    )
    assert primary["turns_to_first_localization"] == 4
    assert primary["time_to_first_localization_sec"] == 14
    assert primary["turns_to_first_target_edit"] == primary["turns_to_last_target_edit"] == 14
    assert primary["time_to_first_target_edit_sec"] == primary["time_to_last_target_edit_sec"] == 63
    assert auxiliary["target_edit_attempts_by_outcome"] == {"success": 1, "failed": 0, "unknown": 0}
    assert auxiliary["target_access_unresolved"] == []


@pytest.mark.parametrize("program", ["cat", "echo", "ssh-cat", "ssh-echo", "cat-wrapper-payload"])
def test_opaque_heredoc_payload_never_becomes_executed_remote_python(program):
    body = f"from pathlib import Path\np=Path('{TARGET}')\np.write_text('new')"
    if program == "cat":
        command = "cat <<'DATA'\n" + body + "\nDATA"
    elif program == "echo":
        command = "echo " + shlex.quote("python3 - <<'PY'\n" + body + "\nPY")
    elif program == "ssh-cat":
        command = _wrap_remote_script("cat <<'PY'\n" + body + "\nPY", "zsh")
    elif program == "ssh-echo":
        command = _wrap_remote_script("echo " + shlex.quote(body), "zsh")
    else:
        command = (
            "cat <<'DATA'\n"
            + _wrap_remote_script("python3 - <<'PY'\n" + body + "\nPY", "zsh")
            + "\nDATA"
        )
    found, _ = target_access_observations(Trajectory(calls=[call("opaque", command)]), [TARGET])
    assert found == []


@pytest.mark.parametrize("closing", ["", "OTHER", "PY_suffix"])
def test_missing_remote_heredoc_boundary_retains_unknown_instead_of_an_edit(closing):
    script = (
        f"python3 - <<'PY'\nfrom pathlib import Path\nPath('{TARGET}').write_text('new')\n" + closing
    )
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("incomplete", _wrap_remote_script(script, "zsh"))]), [TARGET]
    )
    assert found == []
    assert {row["reason"] for row in unresolved} == {"heredoc_boundary_unresolved"}


def test_heredoc_text_in_a_shell_comment_does_not_hide_the_next_source_read():
    command = "# not a heredoc <<'PY'\n" + f"cat {TARGET}"
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("comment", command)]), [TARGET]
    )
    assert len(found) == 1 and found[0]["kind"] == "read"
    assert unresolved == []


def test_edit_metrics_use_successful_edit_calls_never_a_later_read(case):
    calls = [
        call("read", f"cat {TARGET}", turn=2, start=2, end=3),
        call("failed-edit", f"sed -i 's/a/b/' {TARGET}", turn=3, start=4, end=5, error=True),
        call(
            "first-edit",
            f"python3 -c \"from pathlib import Path; Path('{TARGET}').write_text('new')\"",
            turn=4,
            start=6,
            end=7,
        ),
        call("test", "ssh bench-slot python3 test_rmsnorm_npu.py", turn=5, start=8, end=9),
        call("last-edit", name="Edit", paths=[TARGET], turn=6, start=10, end=11),
        call("last-read", f"sed -n '1,80p' {TARGET}", turn=8, start=13, end=14),
    ]
    primary, aux = metrics(case, calls)
    assert primary["turns_to_first_localization"] == 2
    assert primary["time_to_first_localization_sec"] == 2
    assert primary["turns_to_first_target_edit"] == 4
    assert primary["time_to_first_target_edit_sec"] == 6
    assert primary["turns_to_last_target_edit"] == 6
    assert primary["time_to_last_target_edit_sec"] == 10
    assert primary["npu_runs_before_last_target_edit"] == 1
    assert aux["target_edit_attempts_by_outcome"] == {"success": 2, "failed": 1, "unknown": 0}
    assert aux["target_edit_metric_reasons"]["turns_to_last_target_edit"] is None


@pytest.mark.parametrize(
    ("name", "path_key", "component"),
    [
        ("remote_multi_edit", "path", "remote-dev"),
        ("mcp__remote__remote_multi_edit", "file_path", "remote-dev"),
        ("MultiEdit", "file_path", None),
        ("Write", "file_path", None),
    ],
)
def test_typed_multi_edit_and_write_preserve_edit_turn_and_time(case, name, path_key, component):
    edit = call(
        "typed-edit",
        name=name,
        turn=10,
        start=121,
        end=122,
        **{path_key: TARGET},
        edits=[{"old_string": "old", "new_string": "new"}],
    )
    edit.component = component
    edit.result_observation = ToolResultObservation(
        source="codex_mcp" if component else "claude_tool_result",
        operation_kind="file",
        classification="success",
        classification_reason="explicit_tool_success",
    )
    primary, aux = metrics(case, [edit], wall=150)
    assert primary["turns_to_first_target_edit"] == primary["turns_to_last_target_edit"] == 10
    assert primary["time_to_first_target_edit_sec"] == 121
    assert primary["time_to_last_target_edit_sec"] == 121
    assert aux["target_edit_attempts_by_outcome"] == {"success": 1, "failed": 0, "unknown": 0}
    observed = aux["target_access_observations"]
    assert len(observed) == 1
    assert observed[0]["kind"] == "edit"
    assert observed[0]["source"] == "typed_path"
    assert observed[0]["execution_scope"] == ("remote" if component else "local")


@pytest.mark.parametrize(
    ("classification", "outcome"), [("business_failure", "failed"), ("unknown", "unknown")]
)
def test_remote_multi_edit_requires_success_and_only_uses_its_typed_path(case, classification, outcome):
    edit = call(
        "remote-multi",
        name="remote_multi_edit",
        path=TARGET,
        turn=3,
        edits=[{"old_string": "old", "new_string": "new"}],
    )
    edit.component = "remote-dev"
    edit.result_observation = ToolResultObservation(
        source="codex_mcp",
        operation_kind="file",
        classification=classification,
        classification_reason="explicit_component_result",
    )
    primary, aux = metrics(case, [edit])
    assert primary["turns_to_first_target_edit"] is None
    assert aux["target_access_observations"][0]["call_outcome"] == outcome
    assert aux["target_edit_attempts_by_outcome"][outcome] == 1
    edit.arguments["path"] = "other.py"
    edit.arguments["edits"] = [{"old_string": "old", "new_string": TARGET}]
    found, _ = target_access_observations(Trajectory(calls=[edit]), [TARGET])
    assert found == []


@pytest.mark.parametrize("name", ["MultiEdit", "Write"])
def test_native_multi_edit_and_write_require_explicit_success(case, name):
    edit = call("native-edit", name=name, file_path=TARGET, turn=4, start=6, end=7)
    primary, _ = metrics(case, [edit])
    assert primary["turns_to_first_target_edit"] == 4
    edit.is_error = True
    primary, _ = metrics(case, [edit])
    assert primary["turns_to_first_target_edit"] is None
    edit.is_error = None
    primary, aux = metrics(case, [edit])
    assert primary["turns_to_first_target_edit"] is None
    assert aux["target_access_observations"][0]["call_outcome"] == "unknown"


@pytest.mark.parametrize(
    "attempt",
    [
        call("failed-cat", f"cat {TARGET}", turn=2, start=2, end=3, error=True),
        call("unconfirmed-read", name="remote_read", path=TARGET, turn=2, start=2, end=3),
    ],
)
def test_failed_or_unconfirmed_access_attempt_does_not_claim_localization(case, attempt):
    trajectory = Trajectory(calls=[attempt])
    assert first_localization(trajectory, [TARGET]) is None
    primary, aux = metrics(case, trajectory.calls)
    assert primary["turns_to_first_localization"] is None
    assert primary["time_to_first_localization_sec"] is None
    assert len(aux["target_access_observations"]) == 1
    assert aux["target_access_observations"][0]["call_id"] == attempt.id
    assert aux["first_target_access_attempt_turn"] == 2
    assert aux["first_target_access_attempt_sec"] == 2
    assert aux["first_target_access_attempt_outcome"] in {"failed", "unknown"}
    assert aux["target_access_attempts_by_outcome"] == {
        "success": 0,
        "failed": int(attempt.is_error is True),
        "unknown": int(attempt.is_error is not True),
    }
    assert (
        aux["target_access_metric_reasons"]["turns_to_first_localization"]
        == "only_failed_or_unconfirmed_access_attempts"
    )
    assert (
        aux["target_edit_metric_reasons"]["time_to_first_localization_sec"]
        == "only_failed_or_unconfirmed_access_attempts"
    )


def test_localization_uses_first_successful_access_after_failed_attempt(case):
    calls = [
        call("failed-cat", f"cat {TARGET}", turn=2, start=2, end=3, error=True),
        call("successful-cat", f"cat {TARGET}", turn=4, start=5, end=6),
    ]
    assert first_localization(Trajectory(calls=calls), [TARGET]) == 4
    primary, aux = metrics(case, calls)
    assert primary["turns_to_first_localization"] == 4
    assert primary["time_to_first_localization_sec"] == 5
    assert [row["call_outcome"] for row in aux["target_access_observations"]] == [
        "failed",
        "success",
    ]


def test_legacy_mcp_completion_does_not_prove_an_edit_command_succeeded(case):
    primary, aux = metrics(
        case, [call("untyped", f"sed -i 's/a/b/' {TARGET}", name="remote_bash", turn=3)]
    )
    assert primary["turns_to_first_target_edit"] is None
    assert primary["npu_runs_before_last_target_edit"] is None
    assert aux["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert (
        aux["target_edit_metric_reasons"]["turns_to_last_target_edit"]
        == "only_failed_or_unconfirmed_edit_attempts"
    )


def test_python_file_handle_and_path_aliases_but_not_last_target_mention(case):
    primary, _ = metrics(
        case,
        [
            call(
                "edit",
                f"python3 - <<'PY'\np='{TARGET}'\nwith open(p, 'w') as f:\n    f.write('new')\nPY",
                turn=3,
                start=2,
                end=3,
            ),
            call("mention", f"echo {TARGET}", turn=8, start=10, end=11),
        ],
    )
    assert primary["turns_to_last_target_edit"] == 3


def test_actual_typed_patch_target_is_an_edit_but_reading_patch_text_is_not(case):
    patch = f"*** Begin Patch\n*** Update File: {TARGET}\n@@\n-old\n+new\n*** End Patch"
    edit = call("patch", name="remote_apply_patch", patch=patch, turn=4, start=3, end=4)
    edit.result_observation = ToolResultObservation(
        source="codex_mcp",
        classification="success",
        operation_kind="file",
        classification_reason="explicit_component_success",
    )
    primary, _ = metrics(case, [edit])
    assert primary["turns_to_first_target_edit"] == 4
    assert (
        first_localization(
            Trajectory(
                calls=[
                    call("read-patch", name="remote_read", path="fix.patch", content=patch),
                ]
            ),
            [TARGET],
        )
        is None
    )


def test_union_timing_deduplicates_parent_child_and_cross_component_overlap(case):
    primary, aux = metrics(
        case,
        [
            call("parent", "pytest", name="remote_bash", start=0, end=10),
            call("child", "pytest", name="remote_bash", start=3, end=8),
            call("knowledge", name="knowledge_query", start=7, end=12),
        ],
        wall=15,
    )
    assert primary["npu_execution_wait_sec"] == 10
    assert primary["knowledge_query_sec"] == 5
    assert primary["llm_and_exploration_sec"] == 3
    assert aux["tool_timing"]["remote"]["raw_cumulative_sec"] == 15
    assert aux["tool_timing"]["remote"]["overlap_sec"] == 5
    assert aux["tool_timing"]["combined"]["observed_union_sec"] == 12
    assert aux["tool_timing"]["cross_component_overlap_sec"] == 3


def test_missing_timing_does_not_turn_observed_union_into_complete_coverage(case):
    primary, aux = metrics(
        case,
        [
            call("known", "pytest", name="remote_bash", start=0, end=4),
            call("missing", "pytest", name="remote_bash", start=None, end=6),
        ],
    )
    assert primary["npu_execution_wait_sec"] is None
    assert primary["llm_and_exploration_sec"] is None
    assert aux["tool_timing"]["remote"]["observed_union_sec"] == 4
    assert aux["tool_timing"]["remote"]["missing_intervals"] == 1


def test_poll_wait_is_removed_from_remainder_but_is_not_a_test_attempt(case):
    primary, aux = metrics(
        case,
        [
            call("poll", name="remote_job_stdin", start=2, end=10, job_id="job", chars=""),
            call("query", name="knowledge_query", start=7, end=12),
        ],
        wall=15,
    )
    assert primary["npu_test_runs"] == 0
    assert primary["npu_execution_wait_sec"] == 0
    assert primary["llm_and_exploration_sec"] == 5
    assert aux["tool_timing"]["cross_component_overlap_sec"] == 3
    assert aux["validation_tail"]["total_empty_job_polls"] == 1
    assert aux["validation_tail"]["empty_job_poll_timing"]["observed_union_sec"] == 8
    assert aux["validation_tail"]["empty_job_poll_definition"] == (
        "no chars/eof input; may return output; not an idle-poll count"
    )


def test_timestamp_outside_wall_clock_is_not_hidden_by_a_short_union(case):
    primary, aux = metrics(case, [call("late", "pytest", name="remote_bash", start=19, end=24)], wall=20)
    assert primary["npu_execution_wait_sec"] == 5
    assert primary["llm_and_exploration_sec"] is None
    assert aux["tool_timing"]["within_wall_clock"] is False
    assert aux["tool_timing"]["remainder_reason"] == "tool_interval_outside_wall_clock"


def test_unresolved_dynamic_program_is_diagnosed_without_inventing_target_access(case):
    primary, aux = metrics(
        case,
        [call("dynamic", "python3 -c \"from pathlib import Path; Path(get_path()).write_text('new')\"")],
    )
    assert primary["turns_to_first_target_edit"] is None
    assert aux["target_access_unresolved"]


def test_post_success_validation_and_empty_poll_are_diagnostics_not_execution_or_stop_rules(case):
    test = call("test", "python3 test_rmsnorm_npu.py", name="remote_bash")
    test.result_observation = ToolResultObservation(
        source="codex_mcp",
        classification="business_success",
        operation_kind="execution",
        exit_code=0,
        classification_reason="explicit_remote_command_exit",
    )
    primary, aux = metrics(
        case,
        [
            test,
            call("empty-poll", name="remote_job_stdin", job_id="job", chars="", turn=2),
            call("later-test", "python3 test_rmsnorm_npu.py", name="remote_bash", turn=3),
        ],
    )
    assert primary["npu_test_runs"] == 2
    assert aux["validation_tail"]["recognized_test_attempts_after_first_recorded_success"] == 1
    assert aux["validation_tail"]["empty_job_polls_after_first_recorded_success"] == 1
    assert aux["validation_tail"]["is_stop_rule"] is False


@pytest.mark.parametrize(
    ("literal", "index"),
    [
        (f"['{TARGET}', 'other.py']", "0"),
        (f"('other.py', '{TARGET}')", "1"),
        (f"('other.py', '{TARGET}')", "-1"),
        (f"['{TARGET}', 'other.py']", "-2"),
    ],
)
def test_literal_path_sequence_index_is_a_static_edit(case, literal, index):
    code = f"from pathlib import Path\npaths={literal}\np=Path(paths[{index}]); p.write_text('new')"
    primary, _ = metrics(
        case, [call("literal-index", "python3 - <<'PY'\n" + code + "\nPY", turn=4, start=6, end=7)]
    )
    assert primary["turns_to_first_target_edit"] == 4
    assert primary["time_to_first_target_edit_sec"] == 6


def test_actual_three_file_edit_shape_skips_backup_loop_but_resolves_later_indexes():
    paths = [TARGET, "vllm_ascend/_310p/model_runner_310p.py", "tests/ut/worker/test_model_runner.py"]
    code = (
        "from pathlib import Path\nimport shutil\n"
        f"paths={paths!r}\n"
        "for name in paths:\n"
        "    p=Path(name)\n"
        "    shutil.copy2(p, '/tmp/original-'+p.name)\n"
        "p=Path(paths[0]); s=p.read_text(); p.write_text(s)\n"
        "p=Path(paths[1]); p.write_text('compatible')\n"
        "p=Path(paths[2]); p.write_text('regression')\n"
    )
    observed, unresolved = _python_accesses(code)
    assert [path for kind, path in observed if kind == "edit"] == paths
    assert ("read", TARGET) in observed
    assert "python_dynamic_control_flow" in unresolved
    assert all(not path.startswith("/tmp/original-") for _, path in observed)


def test_literal_path_sequence_name_alias_is_resolved():
    observed, _ = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']; alias=paths\nPath(alias[0]).write_text('new')"
    )
    assert ("edit", TARGET) in observed


def test_rebinding_one_sequence_name_preserves_the_other_actual_object():
    observed, _ = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']; alias=paths\n"
        "paths=['other.py']\nPath(alias[0]).write_text('new')"
    )
    assert ("edit", TARGET) in observed


def test_resolved_path_value_is_an_immutable_snapshot_before_list_mutation():
    observed, _ = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']; p=Path(paths[0])\n"
        "paths.append('other.py')\np.write_text('new')"
    )
    assert ("edit", TARGET) in observed


@pytest.mark.parametrize(
    "invocation",
    ["change()", "alias=change\nalias()", "alias=lambda: change()\nalias()"],
)
def test_local_callable_cannot_leave_stale_global_sequence_facts(invocation):
    observed, unresolved = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']\n"
        "def change():\n    paths[0]='other.py'\n"
        f"{invocation}\nPath(paths[0]).write_text('new')"
    )
    assert ("edit", TARGET) not in observed
    assert "python_opaque_local_callable" in unresolved


@pytest.mark.parametrize(
    "change",
    [
        "paths[0]='other.py'",
        "alias=paths\nalias[0]='other.py'",
        "paths.append('other.py')",
        "alias=paths\nalias.extend(['other.py'])",
        "paths += ['other.py']",
        "alias=paths\nalias += ['other.py']",
        "del paths[0]",
        "paths=unknown_paths",
        "paths: list[str]=unknown_paths",
        "del paths",
        "consume(paths)",
        "consume(value=paths)",
        "consume(*paths)",
        "if flag:\n    paths[0]='other.py'",
        "if flag:\n    paths=['other.py']",
        "if flag:\n    alias=paths\n    alias.append('other.py')",
        "for name in unknown_paths:\n    paths[0]=name",
        "for paths in unknown_paths:\n    pass",
        "with unknown_context() as paths:\n    pass",
        "from opaque_module import paths",
        "import opaque_module as paths",
        "try:\n    operation()\nexcept Exception as paths:\n    pass",
        "match unknown:\n    case {'paths': paths}:\n        pass",
        "box={'paths': paths}\nmutate(box)",
        "exec(unobserved_code)",
    ],
)
def test_mutated_escaped_or_conditionally_rebound_sequences_do_not_reuse_old_paths(change):
    observed, unresolved = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']\n{change}\nPath(paths[0]).write_text('new')"
    )
    assert ("edit", TARGET) not in observed
    assert unresolved


def test_mutating_a_sequence_alias_invalidates_the_original_name():
    observed, unresolved = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']; alias=paths\n"
        "mutate(alias)\nPath(paths[0]).write_text('new')"
    )
    assert ("edit", TARGET) not in observed
    assert unresolved


@pytest.mark.parametrize("index", ["dynamic_index", "True", "4", "-4", ":1"])
def test_nonconstant_or_invalid_sequence_indexes_remain_unresolved(index):
    observed, unresolved = _python_accesses(
        f"from pathlib import Path\npaths=['{TARGET}']\nPath(paths[{index}]).write_text('new')"
    )
    assert ("edit", TARGET) not in observed
    assert unresolved


def test_literal_path_sequence_bound_is_reported():
    paths = [TARGET, *["other.py"] * 64]
    observed, unresolved = _python_accesses(
        f"from pathlib import Path\npaths={paths!r}\nPath(paths[0]).write_text('new')"
    )
    assert ("edit", TARGET) not in observed
    assert "python_literal_sequence_limit" in unresolved


@pytest.mark.parametrize(
    "code",
    [
        f"[Path('{TARGET}').write_text('new') for _ in unknown]",
        f"(Path('{TARGET}').write_text('new') for _ in unknown)",
    ],
)
def test_dynamic_comprehension_bodies_are_not_observed_as_executed(code):
    observed, unresolved = _python_accesses("from pathlib import Path\n" + code)
    assert ("edit", TARGET) not in observed
    assert "python_dynamic_control_flow" in unresolved


def test_literal_sequence_parser_never_executes_nonliteral_elements(tmp_path):
    marker = tmp_path / "must-not-exist"
    _python_accesses(
        f"from pathlib import Path\npaths=[Path({str(marker)!r}).write_text('not executed')]"
    )
    assert not marker.exists()


def test_parsed_literal_sequence_edit_does_not_turn_pending_job_into_success(case):
    pending = call(
        "pending-edit",
        f"python3 - <<'PY'\nfrom pathlib import Path\npaths=['{TARGET}']\n"
        "p=Path(paths[0]); p.write_text('new')\nPY",
        name="remote_bash",
        turn=16,
        start=129,
        end=130,
    )
    pending.result_observation = ToolResultObservation(
        source="codex_mcp",
        classification="pending",
        operation_kind="execution",
        remote_status="running",
        exit_code=0,
        job_id="known-job",
        classification_reason="explicit_remote_job_pending",
    )
    primary, auxiliary = metrics(case, [pending], wall=150)
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert auxiliary["target_access_observations"][0]["turn"] == 16
    assert auxiliary["target_access_observations"][0]["call_outcome"] == "unknown"


def pending_edit_job(*, classification="pending"):
    result = call(
        "submitted-edit",
        f"sed -i 's/a/b/' {TARGET}",
        name="remote_bash",
        turn=4,
        start=6,
        end=7,
    )
    result.component = "remote-dev"
    result.result_observation = ToolResultObservation(
        source="codex_mcp",
        result_schema="remote-dev.result.v1",
        classification=classification,
        operation_kind="execution",
        job_id="owned-job",
        target_id="endpoint-one",
        remote_status="running" if classification == "pending" else "unknown",
        exit_code=0,
        classification_reason="explicit_remote_job_pending",
    )
    return result


def terminal_job(identity="confirmed", *, code=0, classification=None, **changes):
    result = call(identity, name="remote_job_status", job_id="owned-job", turn=8, start=15, end=16)
    result.component = "remote-dev"
    result.result_observation = ToolResultObservation(
        source="codex_mcp",
        result_schema="remote-dev.result.v1",
        classification=classification or ("business_failure" if code else "business_success"),
        operation_kind="observation",
        job_id="owned-job",
        target_id="endpoint-one",
        remote_status="completed",
        exit_code=code,
        classification_reason="explicit_remote_command_exit",
    ).model_copy(update=changes)
    return result


def test_later_same_job_terminal_success_confirms_edit_without_rewriting_original_call(case):
    submitted, confirmed = pending_edit_job(), terminal_job()
    primary, auxiliary = metrics(case, [submitted, confirmed])
    assert primary["turns_to_first_target_edit"] == 4
    assert primary["time_to_first_target_edit_sec"] == 6
    assert primary["turns_to_last_target_edit"] == 4
    assert primary["first_source_evidence_turn"] == 8
    assert primary["first_source_evidence_sec"] == 16
    evidence = auxiliary["target_access_observations"][0]["outcome_evidence"]
    assert evidence["basis"] == "later_same_remote_job_terminal"
    assert evidence["confirmation_call_ids"] == ["confirmed"]
    assert evidence["job_id"] == "owned-job"
    assert evidence["target_id"] == "endpoint-one"
    assert evidence["edit_time_basis"] == "original_command_started_sec_proxy"
    assert evidence["confirmation_turn"] == 8
    assert evidence["confirmation_ended_sec"] == 16
    assert auxiliary["target_access_observations"][0]["ended_sec"] == 7
    assert submitted.result_observation.classification == "pending"
    assert submitted.result_observation.remote_status == "running"


def test_confirmed_job_without_confirmation_time_does_not_reuse_submission_end(case):
    submitted, confirmed = pending_edit_job(), terminal_job()
    confirmed.started_sec = confirmed.ended_sec = None
    primary, auxiliary = metrics(case, [submitted, confirmed])
    assert primary["time_to_first_target_edit_sec"] == 6
    assert primary["first_source_evidence_turn"] == 8
    assert primary["first_source_evidence_sec"] is None
    assert auxiliary["source_evidence_basis"]["reasons"]["first_source_evidence_sec"] == (
        "source_evidence_output_or_end_time_missing"
    )


def test_later_same_job_failure_confirms_failure_not_initial_pending_exit_zero(case):
    submitted, confirmed = pending_edit_job(), terminal_job(code=2)
    primary, auxiliary = metrics(case, [submitted, confirmed])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["failed"] == 1
    evidence = auxiliary["target_access_observations"][0]["outcome_evidence"]
    assert evidence["confirmed_exit_code"] == 2
    assert evidence["original_exit_code"] == 0
    assert submitted.result_observation.classification == "pending"


def test_original_unknown_is_not_promoted_by_a_later_successful_poll(case):
    submitted = pending_edit_job(classification="unknown")
    primary, auxiliary = metrics(case, [submitted, terminal_job()])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert submitted.result_observation.classification == "unknown"


@pytest.mark.parametrize(
    "observations",
    [
        [terminal_job("success"), terminal_job("failure", code=2)],
        [terminal_job("unknown", classification="unknown"), terminal_job("success")],
        [
            terminal_job(
                "conflict",
                classification="protocol_error",
                classification_reason="conflicting_exit_codes",
            ),
            terminal_job("success"),
        ],
    ],
)
def test_uncertain_or_conflicting_same_job_results_do_not_confirm_success(case, observations):
    primary, auxiliary = metrics(case, [pending_edit_job(), *observations])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1
    assert auxiliary["target_access_observations"][0]["outcome_evidence"]["reason"] in {
        "conflicting_job_terminal_results",
        "uncertain_job_result",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"target_id": "different-endpoint"},
        {"target_id": None},
        {"job_id": "different-job"},
        {"classification": "success", "exit_code": None},
        {"remote_status": "running"},
        {"operation_kind": "control"},
    ],
)
def test_missing_or_nonterminal_or_different_job_evidence_does_not_join(case, changes):
    primary, auxiliary = metrics(case, [pending_edit_job(), terminal_job(**changes)])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1


def test_an_earlier_status_or_another_component_is_not_a_later_confirmation(case):
    earlier, submitted = terminal_job(), pending_edit_job()
    other = terminal_job("other")
    other.component = "other-component"
    primary, auxiliary = metrics(case, [earlier, submitted, other])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 1


def test_a_reused_job_identity_for_multiple_execution_calls_is_ambiguous(case):
    first, second = pending_edit_job(), pending_edit_job()
    second.id = "second-submission"
    primary, auxiliary = metrics(case, [first, second, terminal_job()])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_edit_attempts_by_outcome"]["unknown"] == 2


@pytest.mark.parametrize("name", ["remote_job_stdin", "remote_job_status", "remote_job_tail"])
@pytest.mark.parametrize(
    "arguments",
    [
        {"session_id": "owned-job"},
        {
            "job_id": "owned-job",
            "session_id": "owned-job",
        },
    ],
)
@pytest.mark.parametrize("code", [0, 2])
def test_terminal_job_argument_aliases_join_only_the_same_result_job(case, name, arguments, code):
    submitted, confirmed = pending_edit_job(), terminal_job(code=code)
    confirmed.name, confirmed.arguments = name, dict(arguments)
    primary, auxiliary = metrics(case, [submitted, confirmed])
    observation = auxiliary["target_access_observations"][0]
    assert observation["call_outcome"] == ("success" if code == 0 else "failed")
    assert observation["outcome_evidence"]["confirmed_exit_code"] == code
    assert confirmed.arguments == arguments
    assert submitted.result_observation.classification == "pending"
    assert primary["turns_to_first_target_edit"] == (4 if code == 0 else None)


@pytest.mark.parametrize(
    "arguments",
    [
        {"job_id": "owned-job", "session_id": "different-job"},
        {"job_id": "different-job", "session_id": "owned-job"},
        {"session_id": "different-job"},
        {"job_id": "owned-job", "session_id": None},
        {"session_id": ""},
        {"session_id": ["owned-job"]},
    ],
)
def test_conflicting_or_wrong_alias_cannot_be_hidden_by_another_success(case, arguments):
    invalid = terminal_job("invalid")
    invalid.name, invalid.arguments = "remote_job_stdin", dict(arguments)
    primary, auxiliary = metrics(case, [pending_edit_job(), invalid, terminal_job("valid")])
    assert primary["turns_to_first_target_edit"] is None
    assert auxiliary["target_access_observations"][0]["call_outcome"] == "unknown"


def test_session_alias_cannot_promote_original_unknown(case):
    submitted, confirmed = pending_edit_job(classification="unknown"), terminal_job()
    confirmed.name, confirmed.arguments = "remote_job_stdin", {"session_id": "owned-job"}
    primary, _ = metrics(case, [submitted, confirmed])
    assert primary["turns_to_first_target_edit"] is None
    assert submitted.result_observation.classification == "unknown"


@pytest.mark.parametrize(
    ("command", "recognized", "programs", "unresolved"),
    [
        ("echo 'pytest test_npu.py'", False, [], []),
        ("python3 -m py_compile test_npu.py", False, [], []),
        (
            "cat <<'PATCH'\npytest -q tests/test_npu.py\nPATCH",
            False,
            [],
            [],
        ),
        (
            "python3 - <<'PY'\nimport subprocess\nsubprocess.run(['pytest'])\nPY",
            False,
            [],
            ["python_inline_or_stdin_not_inspected"],
        ),
        (
            "bash -lc 'python3 -m pytest tests/test_npu.py'",
            True,
            ["python3 -m pytest"],
            [],
        ),
        ("$RUNNER test_npu.py", False, [], ["dynamic_program_not_inspected"]),
    ],
)
def test_npu_count_uses_command_positions_and_retains_opaque_uncertainty(
    case, command, recognized, programs, unresolved
):
    primary, auxiliary = metrics(case, [call("submitted", command, name="remote_bash")])
    assert primary["npu_test_runs"] == int(recognized)
    assert primary["remote_test_invocations"] == int(recognized)
    observed = auxiliary["npu_command_observations"]
    assert observed["basis"] == "literal_command_position/1"
    assert "not complete hardware" in observed["scope"]
    assert observed["calls"] == [
        dict(
            call_id="submitted",
            recognized=recognized,
            matched_programs=programs,
            unresolved_reasons=unresolved,
        )
    ]
    assert "command" not in observed["calls"][0]


@pytest.mark.parametrize(
    ("command", "expected_kind"),
    [
        (f"python3 -c \"from pathlib import Path; Path('{TARGET}').open('r').read()\"", "read"),
        (f"python3 -c \"from pathlib import Path; Path('{TARGET}').open(mode='w').write('x')\"", "edit"),
        (
            f"python3 -c \"from pathlib import Path; Path('tmp.py').replace(Path('{TARGET}'))\"",
            "edit",
        ),
        (f"python3 -c \"import os; os.replace('tmp.py', '{TARGET}')\"", "edit"),
        (f"python3 -c \"import os; os.rename('{TARGET}', 'backup.py')\"", "edit"),
        (f"python3 -c \"from pathlib import Path; Path('{TARGET}').rename('backup.py')\"", "edit"),
        (f"python3 -c \"import shutil; shutil.copy2('tmp.py', '{TARGET}')\"", "edit"),
        (f"tee {TARGET} </dev/null", "edit"),
        (f"cp -f /tmp/staged.py {TARGET}", "edit"),
        (f"mv {TARGET} /tmp/backup.py", "edit"),
        (f"perl -pi -e 's/old/new/g' {TARGET}", "edit"),
    ],
)
def test_python_and_shell_atomic_replacement_and_path_open_access(command, expected_kind):
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("access", command, turn=2)]), [TARGET]
    )
    assert [row["kind"] for row in found] == [expected_kind]
    assert unresolved == []


def test_string_replacement_of_a_filename_is_not_file_access():
    command = f"python3 -c \"name='{TARGET}'; print(name.replace('ops', 'layers'))\""
    found, unresolved = target_access_observations(
        Trajectory(calls=[call("mention", command)]), [TARGET]
    )
    assert found == []
    assert unresolved == []


@pytest.mark.parametrize(
    "command",
    [
        f"grep -E -n 'Ascend(RMS|Layer)Norm' {TARGET}",
        f"sed -E -n '1,5p' {TARGET}",
        f"perl -E 'say for <>' {TARGET}",
    ],
)
def test_extended_expression_flags_do_not_turn_patterns_into_paths(command):
    found, unresolved = target_access_observations(Trajectory(calls=[call("read", command)]), [TARGET])
    assert [row["kind"] for row in found] == ["read"]
    assert unresolved == []
