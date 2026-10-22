# [Role] Builds the synthetic plumbing task and end-to-end demo without models, SSH, or NPUs.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""A labelled synthetic plumbing demonstration; no LLM, SSH, or NPU calls."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mindie_bench.io import write_json
from mindie_bench.models import ExperimentConfig, Group
from mindie_bench.runner.orchestrator import run_matrix
from mindie_bench.scorer.comparator import comparison, freeze, grouped_summaries, markdown

FIXTURE_AGENT = r"""
import json, pathlib, subprocess, sys
if "--version" in sys.argv:
    print("mindie-bench-synthetic 1.0")
    raise SystemExit(0)
prompt=sys.stdin.read()
pathlib.Path("operator.py").write_text("ACCUMULATE_DTYPE = 'float32'\n")
tested=subprocess.run(["npu-replay-test"],capture_output=True,text=True,timeout=30)
if tested.returncode:
    print(tested.stderr,file=sys.stderr)
    raise SystemExit(tested.returncode)
print(json.dumps({"type":"fixture.trajectory","trajectory":{
    "native_completed":True,"total_turns":1,
    "tokens":{"input_uncached":0,"input_cached":0,"output":0,"complete":True},
    "public_messages":[{"role":"assistant","text":"Synthetic dtype edit passed the CPU replay."}],
    "calls":[{"id":"read1","name":"Read","arguments":{"file_path":"operator.py"},
              "turn":1,"started_sec":0.0,"ended_sec":0.001,"is_error":False}]
}}))
"""


def demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    source = output / "inputs/source"
    source.mkdir(parents=True)
    (source / "operator.py").write_text("ACCUMULATE_DTYPE = 'float16'\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True, timeout=30)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, timeout=30)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Synthetic Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "synthetic base",
        ],
        check=True,
        timeout=30,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    agent = output / "inputs/synthetic-agent"
    agent.write_text(f"#!{sys.executable}\n" + FIXTURE_AGENT)
    agent.chmod(0o700)
    task = output / "inputs/task"
    (task / "oracle").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "instruction.md").write_text(
        "SYNTHETIC DEMONSTRATION ONLY. Change operator.py to accumulate in float32 "
        "and run npu-replay-test. This is a plumbing fixture, not a real Ascend task.\n"
    )
    (task / "task.toml").write_text(f'''schema_version = "1.0"
[task]
instance_id = "synthetic-dtype-replay"
domain_category = "numerical_accuracy"
difficulty = "easy"
evidence = "synthetic"
[mindie_knowledge]
knowledge_gradient = "no_hit"
target_files = ["operator.py"]
[budget]
timeout_sec = 10
[environment.l2]
repo_path = "../source"
base_commit = "{commit}"
replay_commands = ["npu-replay-test"]
''')
    write_json(
        task / "oracle/replay_rules.json",
        dict(
            schema_version="1.0",
            checks=[dict(path="operator.py", contains="'float32'")],
            **{
                "pass": dict(stdout="Synthetic replay passed.\n", exit_code=0),
                "fail": dict(stderr="Synthetic replay: accumulation mismatch.\n", exit_code=1),
            },
        ),
        exclusive=True,
    )
    (task / "oracle/gold_patch.diff").write_text(
        "diff --git a/operator.py b/operator.py\n--- a/operator.py\n+++ b/operator.py\n"
        "@@ -1 +1 @@\n-ACCUMULATE_DTYPE = 'float16'\n+ACCUMULATE_DTYPE = 'float32'\n"
    )
    (task / "verifier/test.sh").write_text(
        "#!/bin/bash\nif grep -qx \"ACCUMULATE_DTYPE = 'float32'\" operator.py; then\n"
        "echo '{\"reward\":1.0}'\nexit 0\nelse\necho '{\"reward\":0.0}'\nexit 1\nfi\n"
    )
    config_text = f"""schema_version = "1.0"
backend = "fixture"
[harness]
name = "fixture"
executable = {json.dumps(str(agent))}
model = "synthetic-no-model"
expected_version = "mindie-bench-synthetic 1.0"
input_usd_per_million = 0
cached_usd_per_million = 0
output_usd_per_million = 0
"""
    config_path = output / "inputs/experiment.toml"
    config_path.write_text(config_text)
    config = ExperimentConfig.load(config_path)
    rows = run_matrix(
        [task], config, tier="l2", groups=[Group.G0], repetitions=3, concurrency=3, output=output / "run"
    )
    if any(r.terminal_status.value != "completed_pass" or r.errors for r in rows):
        raise RuntimeError("synthetic demo did not complete cleanly; inspect its preserved run receipts")
    freeze(rows, output / "baseline")
    result = comparison(rows, rows)
    write_json(output / "comparison.json", result, exclusive=True)
    (output / "comparison.md").write_text(markdown(result))
    return dict(
        output=str(output),
        evidence="synthetic",
        model_calls=0,
        npu_runs=0,
        comparison="identity comparison; no MindIE improvement is inferred",
        summary=grouped_summaries(rows),
    )
