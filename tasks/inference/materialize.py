"""Create three synthetic native lifecycle tasks with real CPU measurements.

[Role] Produce executable examples of the native TaskSpec inference contract.
[Invariants] Examples are synthetic; no GPU/NPU or production-model claim.
[Timeouts & Locks] Git commands have 20s bounds; no model or remote calls.
[Side-Effect Policy] Write only a newly created, caller-selected output tree.
"""

import argparse
import difflib
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from mindie_bench.inference_models import InferenceContract
from mindie_bench.io import digest, write_json

PROGRAM = '''import json, sys
request = json.load(sys.stdin)
context = range(request["context"])
if request["mode"] == "new":
    {new_mode}
{computation}
print(json.dumps(result))
'''
SLOW = 'result = [sum(x*x for x in context) + token for token in range(request["requests"])]'
FAST = 'prefix = sum(x*x for x in context)\nresult = [prefix + token for token in range(request["requests"])]'
GOLD = PROGRAM.format(new_mode="pass", computation=FAST)


def materialize(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "prefill": dict(context=20000, requests=100, mode="standard"),
        "decode": dict(context=5000, requests=200, mode="standard"),
        "small": dict(context=20, requests=1, mode="standard"),
    }
    runtime = dict(python=sys.version, protocol="json-stdio", command_timeout_sec=10)
    tasks = []
    for category in ("model_support", "optimization", "regression"):
        root, source = output / "tasks" / category, output / "sources" / category
        source.mkdir(parents=True)
        baseline = PROGRAM.format(
            new_mode='raise NotImplementedError("new mode")' if category == "model_support" else "pass",
            computation=SLOW,
        )
        (source / "engine.py").write_text(baseline, encoding="utf-8", newline="\n")

        def git(*args):
            return subprocess.run(["git", "-C", str(source), *args], check=True,
                                  capture_output=True, text=True, timeout=20).stdout.strip()

        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Inference Example", "-c", "user.email=example@example.invalid",
            "-c", "core.autocrlf=false", "commit", "-qm", "synthetic prefix projection baseline")
        revision = git("rev-parse", "HEAD")
        # Preserve exactly the committed bytes even on hosts with autocrlf enabled.
        baseline = git("show", "HEAD:engine.py") + "\n"
        workloads = [] if category == "model_support" else [
            dict(name="prefill", metric="requests_per_sec", direction="higher", role="scored",
                 workload_sha256=digest(inputs["prefill"])),
            dict(name="decode", metric="batch_latency_ms", direction="lower", role="scored",
                 workload_sha256=digest(inputs["decode"])),
            dict(name="small", metric="batch_latency_ms", direction="lower", role="guard",
                 min_ratio=0.5, workload_sha256=digest(inputs["small"])),
        ]
        contract = InferenceContract(
            category=category, engine="synthetic-python-prefix", model="integer-prefix-fixture",
            baseline_commit=revision, runtime_sha256=digest(runtime), device_class="cpu",
            baseline_source_sha256=hashlib.sha256(baseline.encode()).hexdigest(),
            required_checks=["standard", "new", "empty"], workloads=workloads,
            measurement_rounds=2,
        )
        (root / "oracle").mkdir(parents=True)
        (root / "verifier").mkdir()
        (root / "oracle/baseline.py").write_text(baseline, encoding="utf-8", newline="\n")
        (root / "oracle/gold_patch.diff").write_text("".join(difflib.unified_diff(
            baseline.splitlines(keepends=True), GOLD.splitlines(keepends=True),
            fromfile="a/engine.py", tofile="b/engine.py",
        )), encoding="utf-8")
        write_json(root / "oracle/judge.json", dict(
            contract=contract.model_dump(mode="json"), inputs=inputs, runtime=runtime,
            baseline_source_sha256=hashlib.sha256((root / "oracle/baseline.py").read_bytes()).hexdigest(),
        ))
        shutil.copyfile(Path(__file__).with_name("judge.py"), root / "verifier/judge.py")
        (root / "verifier/test.sh").write_text(
            '#!/bin/bash\nexec ' + shlex.quote(sys.executable)
            + ' -I "$MINDIE_BENCH_ORACLE_ROOT/verifier/judge.py"\n', encoding="utf-8",
        )
        (root / "instruction.md").write_text(
            f"Synthetic CPU {category} exercise. Repair engine.py to support standard, new and empty "
            "requests while avoiding repeated computation of a shared prefix. Preserve exact integer "
            "outputs and small-request latency. This is not a production engine or NPU evaluation.\n",
            encoding="utf-8",
        )
        spec = f'''schema_version = "1.0"
[task]
instance_id = "synthetic-inference-{category}"
domain_category = "inference_lifecycle"
evidence = "synthetic"
[mindie_knowledge]
knowledge_gradient = "no_hit"
target_files = ["engine.py"]
[budget]
timeout_sec = 60
verifier_timeout_sec = 60
[environment.l2]
repo_path = {json.dumps(str(source))}
base_commit = "{revision}"
[inference]
'''
        for name, value in contract.model_dump(mode="json", exclude={"workloads"}).items():
            spec += f"{name} = {json.dumps(value)}\n"
        for workload in workloads:
            spec += "\n[[inference.workloads]]\n"
            spec += "".join(f"{name} = {json.dumps(value)}\n" for name, value in workload.items())
        (root / "task.toml").write_text(spec, encoding="utf-8")
        tasks.append(str(root))
    agent = output / "fixture-agent"
    agent.write_text(
        f"#!{sys.executable}\nimport json, pathlib, sys\n"
        "if '--version' in sys.argv:\n    print('inference-fixture 1.0'); sys.exit(0)\n"
        f"sys.stdin.read()\npathlib.Path('engine.py').write_text({GOLD!r})\n"
        "print(json.dumps({'type':'fixture.trajectory','trajectory':{'native_completed':True,"
        "'total_turns':1,'tokens':{'complete':True},'calls':[]}}))\n", encoding="utf-8",
    )
    agent.chmod(0o700)
    (output / "experiment.toml").write_text(f'''schema_version = "1.0"
backend = "fixture"
[harness]
name = "fixture"
executable = {json.dumps(str(agent))}
model = "synthetic-no-model"
expected_version = "inference-fixture 1.0"
input_usd_per_million = 0
cached_usd_per_million = 0
output_usd_per_million = 0
''', encoding="utf-8")
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(materialize(parser.parse_args().output), indent=2))
