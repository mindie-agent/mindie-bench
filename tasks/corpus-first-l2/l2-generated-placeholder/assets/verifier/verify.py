# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge import redact

assert Path(redact.__file__).resolve().is_relative_to(WORKSPACE)


def check():
    closed = "<redacted:username-at-host:a1b2c3d4e5f6>"
    observed = [item.rule for item in redact.scan_text(closed)]
    print(json.dumps({"closed_token_findings": observed}))
    assert not observed, "complete generated placeholders are already private"
    for token in (closed[:-1], closed.replace("a1b2", "A1B2"), closed[:-2] + ">"):
        assert redact.scan_text(token), "malformed tokens must remain inspectable"
    assert redact.scan_text("host=benchmark-node7 " + closed), "outside content must be scanned"
    assert not redact.scan_text("torch.randn(4)[:, ::2]"), "technical slices must survive"


try:
    check()
except AssertionError as exc:
    print("Business contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
