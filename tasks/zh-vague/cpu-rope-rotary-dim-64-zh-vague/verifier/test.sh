#!/bin/bash
set -eu
if [ -n "${MINDIE_BENCH_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_PYTHON}" ]; then
  CPU_PYTHON="${MINDIE_BENCH_PYTHON}"
elif [ -n "${MINDIE_BENCH_CPU_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_CPU_PYTHON}" ]; then
  CPU_PYTHON="${MINDIE_BENCH_CPU_PYTHON}"
elif [ -x "/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python" ]; then
  CPU_PYTHON="/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python"
else
  CPU_PYTHON="python3"
fi
exec "$CPU_PYTHON" -I "$(dirname "$0")/verify_cpu.py"
