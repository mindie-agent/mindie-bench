# [Role] Run the RoPE soundness matrix on one preflight-checked idle Ascend device.
# [Invariants] Active devices and insufficient HBM stop the run before any candidate executes.
# [Timeouts & Locks] 10 x 40s verifier budgets plus 20s inventory/cleanup; single local flock.
# [Side-Effect Policy] Writes only the supplied evidence folder; never changes device or workload state.
from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import run_soundness

root = Path(sys.argv[1]).resolve()
physical_id = int(sys.argv[2])
card_id, chip_id = divmod(physical_id, 2)
expected_vram_mb = 1024
lock = open(f"/tmp/mindie-bench-soundness-device-{physical_id}.lock", "a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
probe = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10, check=True)
(root / "preflight-npu-smi.txt").write_text(probe.stdout)
if f"No running processes found in NPU {card_id} " not in probe.stdout:
    raise RuntimeError(f"Card {card_id} is active; no candidate was executed")
pattern = rf"\|\s*{chip_id}\s+{physical_id}\s*\|[^\n]*?([0-9]+)\s*/\s*([0-9]+)\s*\|"
match = re.search(pattern, probe.stdout)
if match is None:
    raise RuntimeError("Could not read selected device HBM capacity")
used, total = map(int, match.groups())
if total - used < expected_vram_mb:
    raise RuntimeError("Insufficient free device HBM")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_id)
os.environ["ASCEND_VISIBLE_DEVICES"] = str(physical_id)
preflight = {
    "physical_id": physical_id,
    "card_id": card_id,
    "chip_id": chip_id,
    "vram_free_mb": total - used,
    "expected_vram_mb": expected_vram_mb,
    "python": sys.version,
    "torch": importlib.metadata.version("torch"),
    "torch_npu": importlib.metadata.version("torch-npu"),
    "scope": "single-device NPU tensor correctness; no hardware performance timing",
}
(root / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
try:
    results = run_soundness.run_prepared(root, sys.executable)
    for result in results:
        result["remote_preflight"] = preflight
    run_soundness.write_json(root / "results.json", results)
finally:
    postflight = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=10, check=True
    )
    (root / "postflight-npu-smi.txt").write_text(postflight.stdout)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
raise SystemExit(0 if all(result["status"] == "completed" for result in results) else 1)
