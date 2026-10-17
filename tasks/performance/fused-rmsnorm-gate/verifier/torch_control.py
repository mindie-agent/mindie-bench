"""A real comparator negative control after candidate imports, in its own process."""

import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["MINDIE_BENCH_ORACLE_ROOT"]) / "verifier"))

with contextlib.redirect_stdout(sys.stderr):
    from comparator_control import check_comparators
    from load_case import load

    torch, *_ = load()
    check_comparators(torch)
print("torch-negative-control-ok")
