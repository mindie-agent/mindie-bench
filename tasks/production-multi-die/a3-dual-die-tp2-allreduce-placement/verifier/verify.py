"""Production entrypoint for the A3 dual-Die TP=2 placement workload.

[Role]
Load the task judge and run it with the candidate workspace and private-oracle
environment supplied by `runner/verifier.py`.

[Invariants]
- `-I` isolated execution must still find this oracle's sibling modules, so the
  script directory is inserted explicitly; the candidate workspace is never
  added to `sys.path` here.
- Exit 0 means the judge's real measured workload passed. Unsupported
  prerequisites and failed gates exit nonzero with `reward=0.0`.

[Timeouts & Locks]
- Delegates the bounded workload and NPU/HCCL timeouts to `production_judge`.

[Side-Effect Policy]
- No side effects beyond the judge; this wrapper only dispatches.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from production_judge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
