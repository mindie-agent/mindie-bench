"""One source/gold, zero-model host/device diagnostic using the existing NPU lease."""

from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "src"))

from mindie_bench.io import write_json  # noqa: E402
from mindie_bench.models import RemoteConfig, TaskSpec  # noqa: E402
from mindie_bench.runner.remote_slot import RemoteSlot  # noqa: E402

task = TaskSpec.load(ROOT)
output = ROOT.parents[2] / "reports/v0.2.0-performance-5312b32d/profile"
output.mkdir(parents=True, exist_ok=True)
if (output / "remote.stderr").exists():
    (output / "remote.stderr").unlink()
config = json.loads((ROOT.parents[2] / "reports/v0.2.0-runtime-canary/slot-1/config.json").read_text())
config["repo_path"] = "/bench/source-fusedgate-history-20261017.git"
config["slots_root"] = "/bench/slots-performance-fusedgate-20261017"
config["runtime_read_paths"] = [
    "/usr/local/Ascend",
    "/usr/local/python3.12.13",
    "/usr/local/dcmi",
    "/vllm-workspace/vllm",
    "/vllm-workspace/vllm-ascend/vllm_ascend/_build_info.py",
    "/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so",
    "/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so",
]
config["component_runtime"]["diagnostics_root"] = str(output / "diagnostics")
remote = RemoteSlot(
    RemoteConfig.model_validate(config),
    task.environment["l3"],
    "fusedgate-profile-" + uuid.uuid4().hex[:12],
    [12],
    300,
    artifact_dir=output,
)
payload = (
    "import types,sys\n"
    "helper=types.ModuleType('load_case')\n"
    f"exec({(ROOT / 'verifier/load_case.py').read_text()!r},helper.__dict__)\n"
    "sys.modules['load_case']=helper\n" + (ROOT / "profile_case.py").read_text()
).encode()
report = {"model_calls": 0, "scope": "bounded performance diagnosis", "stages": {}}
try:
    remote.prepare()
    applied = remote.execute(
        "git apply --no-index -",
        base64.b64encode((ROOT / "oracle/gold_patch.diff").read_bytes()).decode(),
    )
    if applied["exit_code"] != 0:
        raise RuntimeError(f"gold patch failed: {applied}")
    for stage in ["gold"]:
        result = remote.execute("python3 -I -B -", base64.b64encode(payload).decode(), timeout=180)
        write_json(output / f"{stage}.json", result)
        if result["exit_code"] != 0:
            raise RuntimeError(f"{stage} diagnostic failed: {result}")
        text = base64.b64decode(result["stdout_b64"]).decode()
        report["stages"][stage] = json.loads(text.splitlines()[-1])
finally:
    remote.close()
    report["postflight"] = remote.postflight
    report["cleanup"] = remote.cleanup_receipt
    report["operation_unknown"] = remote.operation_unknown
    write_json(output / "profile.json", report)
