#!/bin/bash
set -eu
exec "${MINDIE_BENCH_PYTHON:-python3}" -I -B "$(dirname "$0")/verify.py"
