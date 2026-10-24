#!/bin/bash
set -eu
# The oracle executes the candidate source on CPU; keep torch_npu out of the judge.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec python3 -I -B "$(dirname "$0")/verify.py"
