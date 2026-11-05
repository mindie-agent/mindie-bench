#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_PY="$SCRIPT_DIR/verify.py"

for candidate in \
  "${MINDIE_BENCH_PYTHON:-}" \
  "/tmp/mindie-bench-cpu-runtime-20261022/venv/bin/python" \
  "$(command -v python3 || true)" \
  "$(command -v python || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    if "$candidate" -c "import torch" >/dev/null 2>&1; then
      exec "$candidate" "$VERIFY_PY"
    fi
  fi
done

echo "No Python interpreter with torch available for verification." >&2
exit 2
