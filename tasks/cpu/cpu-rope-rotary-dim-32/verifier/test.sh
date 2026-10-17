#!/bin/bash
set -eu
CPU_PYTHON='@CPU_PYTHON@'
if [ "$CPU_PYTHON" = '@''CPU_PYTHON@' ]; then
  echo "Unbound CPU task template: materialize a runtime-bound task revision first." >&2
  exit 78
fi
exec "$CPU_PYTHON" -I "$(dirname "$0")/verify_cpu.py"
