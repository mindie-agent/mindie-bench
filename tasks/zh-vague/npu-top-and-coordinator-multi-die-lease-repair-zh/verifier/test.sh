#!/bin/bash
set -eu
if [ -n "${MINDIE_BENCH_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_PYTHON}" ]; then
  PY="${MINDIE_BENCH_PYTHON}"
elif command -v mindie-profiler >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif [ -n "${MINDIE_BENCH_CPU_PYTHON:-}" ] && [ -x "${MINDIE_BENCH_CPU_PYTHON}" ]; then
  PY="${MINDIE_BENCH_CPU_PYTHON}"
elif [ -x "/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python" ]; then
  PY="/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -I -B "$(dirname "$0")/verify.py"
