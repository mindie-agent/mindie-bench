#!/bin/bash
set -eu
exec '@L2_PYTHON@' -I "$(dirname "$0")/verify.py"
