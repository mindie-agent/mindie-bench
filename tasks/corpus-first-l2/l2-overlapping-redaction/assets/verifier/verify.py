# Source imports follow explicit owned-workspace selection.
# ruff: noqa: E402
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(os.environ.get("MINDIE_BENCH_WORKSPACE", os.getcwd())).resolve()
sys.path.insert(0, str(WORKSPACE))

from mindie_knowledge import redact
from mindie_knowledge.loop.transcript_redaction import redact as redact_body

assert Path(redact.__file__).resolve().is_relative_to(WORKSPACE)


def check():
    binding = json.loads((Path(__file__).parent / "runtime-binding.json").read_text())
    scanner = Path(binding["gitleaks"])
    if hashlib.sha256(scanner.resolve().read_bytes()).hexdigest() != binding["gitleaks_sha256"]:
        raise RuntimeError("configured scanner binary identity changed")
    for number in (123456, 234567, 345678):
        identity = "w00" + str(number)
        tail = "/Run" + "Case" * 12 + "20261012T031245Z"
        source = "结果目录/home/" + identity + tail
        clean, rules = redact_body(source, executable=str(scanner), key=b"l2-fixture")
        print(
            json.dumps(
                {"identity_retained": identity in clean, "tail_retained": tail in clean, "rules": rules},
                ensure_ascii=True,
            )
        )
        assert identity not in clean and tail not in clean, (
            "all overlapping detector claims must be covered"
        )
        assert not redact.scan_text(clean), "new output must remain scan clean"
    technical = "Keep BF16 and torch.randn(4)[:, ::2] unchanged."
    assert redact_body(technical, executable=str(scanner), key=b"l2-fixture")[0] == technical


try:
    check()
except AssertionError as exc:
    print("Business contract failed: " + str(exc), file=sys.stderr)
    print(json.dumps({"reward": 0.0}))
    raise SystemExit(1)
print(json.dumps({"reward": 1.0}))
