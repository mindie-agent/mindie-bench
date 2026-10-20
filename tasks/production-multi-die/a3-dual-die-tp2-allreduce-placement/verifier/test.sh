#!/bin/bash
set -euo pipefail
ROOT="${MINDIE_BENCH_ORACLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec python3 -I -B "$ROOT/verifier/verify.py" "$@"
