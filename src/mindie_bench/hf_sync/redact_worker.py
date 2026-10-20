"""Redaction worker executed as a sealed file by the knowledge interpreter.

[Role] Read one JSON request from stdin and run the installed first-party
`mindie_knowledge` redactor plus its residual-finding scan.
[Invariants] Any unresolved sensitive finding aborts; a clean result is printed
as JSON and never replaced by a weaker local regex fallback.
[Timeouts & Locks] The parent `subprocess.run` owns the 120s timeout; this
worker takes no lock and starts no child beyond the configured `gitleaks`.
[Side-Effect Policy] Pure text transformation; reads stdin and writes stdout.
"""

import json
import sys

from mindie_knowledge.loop.transcript_redaction import redact
from mindie_knowledge.redact import scan_text

request = json.load(sys.stdin)
clean, rules = redact(
    request["text"],
    executable=request["gitleaks"],
    key=bytes.fromhex(request["key"]),
    private_paths=request["private_paths"],
)
remaining = scan_text(clean)
if remaining:
    raise RuntimeError("redaction left unresolved sensitive findings")
print(json.dumps({"clean": clean, "rules": rules}))
