#!/bin/bash
set -euo pipefail
exec python3 -I -B "${MINDIE_BENCH_ORACLE_ROOT:?}/verifier/measure.py"
