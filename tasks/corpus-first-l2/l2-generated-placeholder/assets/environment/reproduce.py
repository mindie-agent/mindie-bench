# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge.redact import scan_text

token = "<redacted:username-at-host:a1b2c3d4e5f6>"
print(json.dumps({"input": token, "findings": [item.rule for item in scan_text(token)]}))
