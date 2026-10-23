from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_bench.models import ExperimentConfig, TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _configure_default_test_env() -> None:
    for proxy_key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(proxy_key, None)
    operators_repo = Path("/Volumes/data/code/knowledge-operators")
    if not os.environ.get("MINDIE_BENCH_OPERATORS_CORPUS") and (operators_repo / ".git").exists():
        head_proc = subprocess.run(
            ["git", "-C", str(operators_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        files_proc = subprocess.run(
            ["git", "-C", str(operators_repo), "ls-tree", "-r", "--name-only", "-z", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head_proc.returncode == 0 and files_proc.returncode == 0:
            head = head_proc.stdout.strip()[:12]
            snapshot = Path(f"/tmp/mindie-bench-operators-clean-{head}")
            if not (snapshot / "index" / "routing-manifest.json").is_file():
                shutil.rmtree(snapshot, ignore_errors=True)
                snapshot.mkdir(parents=True, exist_ok=True)
                for rel in files_proc.stdout.split("\0"):
                    if not rel:
                        continue
                    src = operators_repo / rel
                    if src.is_file():
                        dst = snapshot / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
            os.environ["MINDIE_BENCH_OPERATORS_CORPUS"] = str(snapshot)
    # Native capture must use this test environment's components, not a
    # historical release interpreter paired with today's adapter scripts.
    if all(
        importlib.util.find_spec(name) is not None for name in ("mindie_knowledge", "remote_dev")
    ):
        os.environ.setdefault("MINDIE_BENCH_TEST_MCP_PYTHON", sys.executable)
    for env_key, candidates in (
        (
            "MINDIE_BENCH_CPU_PYTHON",
            (
                Path("/tmp/mindie-bench-cpu-l2-runtime-20261017/venv/bin/python"),
            ),
        ),
        (
            "MINDIE_BENCH_STALE_PYTHON",
            (
                Path("/Volumes/data/code/mindie-release-20261014/codex-interrupt-final-env-20261016/bin/python"),
                REPO_ROOT.parent / "knowledge/.venv/bin/python",
            ),
        ),
        (
            "MINDIE_BENCH_STALE_SOURCE_REPO",
            (
                REPO_ROOT / "repositories/vllm-ascend-stale-config",
            ),
        ),
        (
            "MINDIE_BENCH_TEST_CLAUDE_BIN",
            tuple(
                Path(p)
                for p in (shutil.which("claude"), str(Path.home() / ".local/bin/claude"))
                if p
            ),
        ),
        (
            "MINDIE_BENCH_TEST_CLAUDE_SCRIPTS",
            (
                Path("/Volumes/data/code/mindie-release-20261014/cc/scripts"),
                REPO_ROOT.parent / "mindie-agent-cc/plugins/mindie/scripts",
            ),
        ),
        (
            "MINDIE_BENCH_TEST_CODEX_BIN",
            tuple(
                Path(p)
                for p in (shutil.which("codex"), "/opt/homebrew/bin/codex", "/usr/local/bin/codex")
                if p
            ),
        ),
        (
            "MINDIE_BENCH_TEST_CODEX_CATALOG",
            (
                Path.home() / ".codex/models_cache.json",
            ),
        ),
    ):
        if os.environ.get(env_key):
            continue
        for candidate in candidates:
            if candidate.exists():
                os.environ[env_key] = str(candidate)
                break


_configure_default_test_env()


@pytest.fixture(autouse=True)
def isolate_host_user_directories(tmp_path, monkeypatch):
    """Keep every test's user/config/state write inside its own tmp_path."""
    root = tmp_path.parent / (tmp_path.name + "-host")
    home = root / "home"
    config = root / "config"
    state = root / "state"
    diagnostics = root / "diagnostics"
    for directory in (home, config, state, diagnostics):
        directory.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(diagnostics))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_CONFIG", str(config / "diagnostics.json"))
    return {
        "home": home,
        "config": config,
        "state": state,
        "diagnostics": diagnostics,
    }


AGENT = """#!{python}
import json, os, pathlib, subprocess, sys, time
if "--version" in sys.argv:
    print("fixture-agent 1.0")
    raise SystemExit(0)
prompt=sys.stdin.read()
if "mode=sleep" in prompt:
    time.sleep(30)
if "mode=working_sleep" in prompt or "mode=repair_then_sleep" in prompt:
    if "mode=repair_then_sleep" in prompt:
        pathlib.Path("module.py").write_text("VALUE = 2\\n")
    else:
        pathlib.Path("module.py").read_text()
    print(json.dumps({{"type":"fixture.trajectory","trajectory":{{
        "total_turns":1,"calls":[{{"id":"1","name":"Read",
        "arguments":{{"file_path":"module.py"}},"turn":1,"started_sec":0.0,"ended_sec":0.001}}]
    }}}}), flush=True)
    time.sleep(30)
if "mode=malformed" in prompt:
    print("not JSON", flush=True)
    raise SystemExit(0)
if "mode=crash" in prompt:
    raise SystemExit(7)
if "mode=pass" in prompt or "mode=replay" in prompt:
    pathlib.Path("module.py").write_text("VALUE = 2\\n")
if "mode=replay" in prompt:
    result=subprocess.run(["pytest", "-q"], capture_output=True, text=True, timeout=30)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(9)
if "mode=probe" in prompt:
    secret=pathlib.Path(prompt.split("secret=")[1].splitlines()[0])
    try:
        secret.read_text()
        raise RuntimeError("ISOLATION_BROKEN")
    except (PermissionError, FileNotFoundError):
        pass
print(json.dumps({{"type":"fixture.trajectory","trajectory":{{
    "native_completed":True,"total_turns":1,
    "tokens":{{"input_uncached":11,"input_cached":5,"output":3,"complete":True}},
    "public_messages":[{{"role":"assistant","text":"Updated the test module."}}],
    "calls":[{{"id":"1","name":"Read","arguments":{{"file_path":"module.py"}},
              "turn":1,"started_sec":0.0,"ended_sec":0.001,"is_error":False}}]
}}}}), flush=True)
"""


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


@pytest.fixture(scope="session")
def _case_source_template(tmp_path_factory):
    repo = tmp_path_factory.mktemp("case-source-template") / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "module.py").write_text("VALUE = 1\n")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Benchmark Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "synthetic failing source",
    )
    return repo, git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="session")
def _corpus_source_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus-source-template") / "corpus-source"
    root.mkdir()
    for identity in ["origin", "analogy", "stale"]:
        package = root / "tasks" / identity
        (package / "materials").mkdir(parents=True)
        (package / "PACKAGE.md").write_text(f"# {identity}\nNPU ACLGraph debug pytest\n")
        (package / "materials/0001.md").write_text(
            "Historical trace: NPU error; pytest failed then fixed.\n"
        )
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "synthetic advisory corpus",
    )
    return root, git(root, "rev-parse", "HEAD")


@pytest.fixture
def case(tmp_path, _case_source_template):
    template_repo, commit = _case_source_template
    repo = tmp_path / "source"
    shutil.copytree(template_repo, repo)
    task_root = tmp_path / "task"
    (task_root / "verifier").mkdir(parents=True)
    (task_root / "oracle").mkdir()
    (task_root / "instruction.md").write_text("mode=pass\n")
    (task_root / "oracle/replay_rules.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checks": [{"path": "module.py", "contains": "VALUE = 2"}],
                "pass": {"stdout": "1 passed\n", "exit_code": 0},
                "fail": {"stderr": "NPU replay: invalid dtype\n", "exit_code": 1},
            }
        )
    )
    (task_root / "verifier/test.sh").write_text(
        '#!/bin/bash\n'
        'if [ -x /usr/bin/grep ]; then GREP=/usr/bin/grep; else GREP=grep; fi\n'
        'if "$GREP" -qx "VALUE = 2" module.py; then\n'
        'echo \'{"reward":1.0,"speedup_ratio":1.2}\'\nexit 0\n'
        "else\necho '{\"reward\":0.0}'\nexit 1\nfi\n"
    )
    task = f'''schema_version = "1.0"
[task]
instance_id = "synthetic-dtype"
domain_category = "graph_and_kernel_debugging"
evidence = "synthetic"
[mindie_knowledge]
knowledge_gradient = "analogical"
corpus_id = "fixture"
same_origin_task_hashes = ["origin"]
analogical_task_hashes = ["analogy"]
target_files = ["module.py"]
[budget]
timeout_sec = 3
verifier_timeout_sec = 3
[environment.l2]
repo_path = {json.dumps(str(repo))}
base_commit = "{commit}"
replay_commands = ["pytest"]
'''
    (task_root / "task.toml").write_text(task)
    if os.name == "nt":
        agent_py = tmp_path / "fixture-agent.py"
        agent_py.write_text(AGENT.format(python=sys.executable))
        executable = tmp_path / "fixture-agent.cmd"
        executable.write_text(f'@"{sys.executable}" -I "{agent_py}" %*\n')
    else:
        executable = tmp_path / "fixture-agent"
        executable.write_text(AGENT.format(python=sys.executable))
        executable.chmod(0o700)
    config = ExperimentConfig.model_validate(
        {
            "schema_version": "1.0",
            "backend": "fixture",
            "harness": {
                "name": "fixture",
                "executable": str(executable),
                "model": "synthetic-no-model",
                "expected_version": "fixture-agent 1.0",
                "input_usd_per_million": 0,
                "cached_usd_per_million": 0,
                "output_usd_per_million": 0,
            },
        }
    )
    return dict(
        root=tmp_path,
        repo=repo,
        task=task_root,
        config=config,
        commit=commit,
        spec=TaskSpec.load(task_root),
    )


@pytest.fixture
def corpus(tmp_path, _corpus_source_template):
    template_root, commit = _corpus_source_template
    root = tmp_path / "corpus-source"
    shutil.copytree(template_root, root)
    return root, commit
