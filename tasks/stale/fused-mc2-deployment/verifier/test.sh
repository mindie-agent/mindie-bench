#!/bin/bash
set -eu
exec python3 -B "$MINDIE_BENCH_ORACLE_ROOT/verifier/verify.py"
