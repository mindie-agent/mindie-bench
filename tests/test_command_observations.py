import base64
import json
import urllib.error
import urllib.request

import pytest

from mindie_bench.runner.gateway import Gateway
from mindie_bench.scorer.command_observations import (
    is_recognized_npu_command,
    observe_npu_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "/opt/venv/bin/pytest tests/test_model.py",
        "cd /work && CUDA_VISIBLE_DEVICES=0 python3 -I test_rmsnorm_npu.py",
        "python -m pytest -q",
        "python3.13 -I -m unittest discover",
        "python3 -W ignore -X dev tests/test_rmsnorm.py",
        "env ASCEND_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py",
        "command msprof --application='python3 train.py'",
        "exec vllm serve model",
        "bash -lc 'cd /work && python3 -I test_rmsnorm_npu.py'",
        "ssh -p 22 -o BatchMode=yes root@owned 'cd /work && pytest -q'",
        "ssh -T -p22 owned python3 -m unittest",
        "bash -c 'echo setup' && cat note.txt && pytest -q",
        "cat <<'PY'\nthis is not a test\nPY\npytest -q",
        "cat <<-EOF\n\tpytest documentation\n\tEOF\npython3 -m pytest",
        "echo done # pytest is a comment\npytest -q",
        "sh -c \"bash -lc 'pytest -q'\"",
        "bash -o pipefail -c 'pytest -q'",
        "python3 -I -mpytest",
        "pytest -k '()' --marker='{'",
        "python3 -I reproduce.py --build --check",
        "python3 -u /bench/reproduce.py --build",
        "ssh -T owned 'python3 -I reproduce.py --build'",
        "./reproduce.py --build",
    ],
)
def test_literal_program_positions_are_recognized(command):
    observed = observe_npu_command(command)
    assert observed["recognized"]
    assert observed["matched_programs"]
    assert is_recognized_npu_command(command)
    assert "not NPU/kernel completions" in observed["scope"]


@pytest.mark.parametrize(
    "command",
    [
        "echo pytest",
        "printf '%s' 'pytest && python3 test_model_npu.py'",
        "grep -n 'pytest|torchrun' patch.diff",
        "cat tests/test_rmsnorm_npu.py",
        "sed -n '1,120p' test_model_npu.py",
        "rg --files tests | rg 'pytest|test_'",
        "git apply pytest.patch",
        "apply_patch <<'PATCH'\n+import pytest\n+python3 test_model_npu.py\nPATCH",
        "cat <<PY\npytest -q\nPY",
        "cat <<'PY'\n'unclosed Python quote\npytest\nPY",
        "cat <<A <<'B'\npytest\nA\ntorchrun\nB",
        "bash -c \"cat <<'PY'\npytest\nPY\"",
        "python3 -c 'import pytest; print(\"test_rmsnorm_npu.py\")'",
        "python3 -I -c 'import subprocess; subprocess.run([\"pytest\"])'",
        "python3 -m py_compile tests/test_rmsnorm_npu.py",
        "python3 -m ruff check tests/test_rmsnorm.py",
        "echo ok # pytest -q",
        "vllm --help",
        "vllm bench serve model",
        "echo '<<PY pytest' && cat ordinary.py",
        "cat <<< 'pytest -q'",
        "echo '&&' pytest",
        "echo ';' pytest",
        r"echo \; pytest",
        "echo '>' pytest",
        "env LABEL=';' echo pytest",
        "ssh -G owned pytest",
        "ssh -N owned pytest",
        "bash -n -c 'pytest'",
        "bash --version -c 'pytest'",
        "python3 --version tests/test_model.py",
        "python3 -c'print(1)' test_model_npu.py",
        "cat reproduce.py",
        "grep reproduce.py README.md",
        "echo 'python3 -I reproduce.py --build'",
        "python3 -m py_compile reproduce.py",
        "python3 -m compileall reproduce.py",
        "cat <<'PY'\npython3 -I reproduce.py --build\nPY",
    ],
)
def test_arguments_comments_and_stdin_source_are_not_test_commands(command):
    assert not is_recognized_npu_command(command)


@pytest.mark.parametrize(
    "command,reason",
    [
        ("$TEST_COMMAND", "dynamic_program_not_inspected"),
        ("bash -c '$TEST_COMMAND'", "dynamic_program_not_inspected"),
        ('echo "$(pytest)"', "command_substitution_not_inspected"),
        ("eval 'pytest'", "dynamic_program_not_inspected"),
        ("cat <<EOF\npytest", "unterminated_heredoc"),
        ("cat <<$DELIMITER\npytest\n", "unsupported_heredoc_delimiter"),
        ("env " * 10 + "pytest", "wrapper_depth_limit"),
        ("pytest() { echo documentation; }", "shell_group_or_function_not_inspected"),
        ("function helper { echo hi; pytest; }", "shell_group_or_function_not_inspected"),
        ("bash -c", "missing_shell_command"),
        ("echo " + "x" * 262_144, "command_size_limit"),
        ("echo " + "x " * 4096, "token_limit"),
    ],
)
def test_dynamic_or_out_of_bound_commands_are_explicitly_unresolved(command, reason):
    observed = observe_npu_command(command)
    assert not observed["recognized"]
    assert reason in observed["unresolved_reasons"]


def test_compound_programs_still_represent_one_submitted_attempt():
    observed = observe_npu_command("pytest -q && python3 -m unittest && torchrun training.py")
    assert observed["recognized"] is True
    assert observed["matched_programs"] == ["pytest", "python3 -m unittest", "torchrun"]


def _post(gateway, path, value):
    request = urllib.request.Request(
        gateway.url + path,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=5)


@pytest.mark.parametrize("route", ["one-shot", "stream-open", "stream-rpc"])
def test_text_mentions_do_not_spend_budget_before_real_test(case, route):
    case["spec"].budget.max_npu_runs = 1
    admitted = []

    class Remote:
        def execute(self, command, stdin):
            admitted.append(command)
            return {"stdout_b64": "", "stderr_b64": "", "exit_code": 0}

        def open_stream(self, command):
            admitted.append(command)
            return {"stream_id": str(len(admitted))}

        def exchange_stream(self, identity, stdin, *, eof):
            admitted.append(base64.b64decode(stdin))
            return {"stdout_b64": "", "stderr_b64": "", "exit_code": None}

    with Gateway(
        case["config"].harness,
        case["spec"],
        case["task"],
        case["repo"],
        case["root"] / "journal",
        remote=Remote(),
    ) as gateway:
        stream = None
        if route == "stream-rpc":
            stream = gateway.ssh_stream({"action": "open", "command": "rpc worker"})["stream_id"]

        def submit(command):
            if route == "one-shot":
                return _post(gateway, "/bench/ssh", {"command": command, "stdin_b64": ""})
            if route == "stream-open":
                return _post(gateway, "/bench/ssh-stream", {"action": "open", "command": command})
            frame = json.dumps({"payload": {"spec": {"command": command}}}).encode() + b"\n"
            return _post(
                gateway,
                "/bench/ssh-stream",
                {
                    "action": "exchange",
                    "stream_id": stream,
                    "stdin_b64": base64.b64encode(frame).decode(),
                    "eof": False,
                },
            )

        for command in (
            "echo 'pytest -q'",
            "grep pytest patch.diff",
            "cat <<'PY'\npytest -q\npython3 test_model_npu.py\nPY",
            "python3 -c 'import pytest; print(\"test_rmsnorm_npu.py\")'",
            "python3 -m py_compile tests/test_rmsnorm_npu.py",
        ):
            with submit(command):
                pass
            assert gateway.remote_test_calls == 0
        with submit("bash -lc 'python3 -I test_rmsnorm_npu.py'"):
            pass
        assert gateway.remote_test_calls == 1
        count = len(admitted)
        with pytest.raises(urllib.error.HTTPError) as error:
            submit("pytest -q")
        assert error.value.code == 429
        assert len(admitted) == count
        assert gateway.remote_test_calls == 1
