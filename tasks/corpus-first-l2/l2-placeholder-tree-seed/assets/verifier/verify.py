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
    for digest in ("a1b2c3d4e5f6", "112233aabbcc", "abcdef012345"):
        token = "<redacted:username-at-host:" + digest + ">"
        document = {token: {"nested": [token, {"note": "already private " + token}]}}
        findings = redact.scan_tree(document)
        print(json.dumps({"tree_findings": [item.rule for item in findings]}))
        assert not findings, "closed generated tokens in nested keys/values are not new leaks"
        document[token]["nested"].append({"endpoint": "host=benchmark-node7"})
        assert redact.scan_tree(document), "closed tokens cannot exempt unrelated tree values"
    assert redact.scan_tree({"malformed": "<redacted:username-at-host:a1b2c3d4e5f>"}), (
        "incomplete token must remain inspectable through scan_tree"
    )


try:
    check()
except AssertionError as exc:
    print("Business contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
