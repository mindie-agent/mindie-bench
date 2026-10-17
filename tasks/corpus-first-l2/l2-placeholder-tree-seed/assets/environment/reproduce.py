# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge.redact import scan_tree

token = "<redacted:username-at-host:112233aabbcc>"
print(json.dumps({"findings": [item.rule for item in scan_tree({token: {"value": [token]}})]}))
