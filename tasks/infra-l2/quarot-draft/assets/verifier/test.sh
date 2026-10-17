#!/bin/bash
set -eu
exec '@L2_PYTHON@' -I -B "$(dirname "$0")/verify.py"
