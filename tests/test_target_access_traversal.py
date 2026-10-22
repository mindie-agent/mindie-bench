import shlex

import pytest

from mindie_bench.models import ToolResultObservation, Trajectory
from mindie_bench.scorer.metrics import (
    _python_accesses,
    first_localization,
    target_access_observations,
)

TARGET = "vllm_ascend/ops/layernorm.py"


from test_target_access_metrics import (  # noqa: E402
    call,
    metrics,
)


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


