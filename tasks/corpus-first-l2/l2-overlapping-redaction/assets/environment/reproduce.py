# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge.loop.transcript_redaction import redact

binding = json.loads((Path(__file__).parent / "runtime-binding.json").read_text())
identity = "w00123456"
tail = "/Run" + "Case" * 12 + "20261012T031245Z"
clean, rules = redact(
    "结果目录/home/" + identity + tail, executable=binding["gitleaks"], key=b"public-fixture"
)
print(json.dumps({"redacted": clean, "tail_retained": tail in clean, "rules": rules}, ensure_ascii=True))
