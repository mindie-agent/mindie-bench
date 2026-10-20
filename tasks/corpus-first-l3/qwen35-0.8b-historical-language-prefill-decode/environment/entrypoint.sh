#!/usr/bin/env bash
# [Role] Preflight for the frozen Qwen3.5-0.8B historical task.
# [Invariants] No provisioning, no model/NPU work, no mutation of frozen files.
# [Timeouts & Locks] Caller owns execution timeout; this script is local checks only.
# [Side-Effect Policy] Read-only checks; exits 3 while runtime is pending.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT="$ROOT/support-set.json"
EXPECT_SUPPORT="09d733f5c7156c4eff224113e26734711be61c141a07be3d096b0038c7dc872a"
actual="$(sha256sum "$SUPPORT" | awk '{print $1}')"
if [[ "$actual" != "$EXPECT_SUPPORT" ]]; then
  echo "FROZEN_SUPPORT_MISMATCH expected=$EXPECT_SUPPORT actual=$actual" >&2
  exit 2
fi

if [[ -z "${QWEN35_RUNTIME_ID:-}" ]]; then
  echo "PENDING_RUNTIME_UNVERIFIED: set QWEN35_RUNTIME_ID after the preprovisioned historical runtime and reset snapshot are verified" >&2
  exit 3
fi

MODEL="${QWEN35_WEIGHTS:-/mnt/weight/Qwen3.5-0.8B}"
for f in config.json model.safetensors.index.json; do
  [[ -f "$MODEL/$f" ]] || { echo "MISSING_WEIGHT_METADATA $MODEL/$f" >&2; exit 2; }
done
# Full weight is identified by the frozen SHA, not rehashed here.
[[ -f "$MODEL/model.safetensors-00001-of-00001.safetensors" ]] || {
  echo "MISSING_WEIGHT_SHARD" >&2; exit 2;
}
if [[ -z "${QWEN35_ARM:-}" ]]; then
  echo "PENDING_ARM: QWEN35_ARM must be source0 or after" >&2
  exit 2
fi
echo "READY runtime=$QWEN35_RUNTIME_ID arm=$QWEN35_ARM model=$MODEL support=$EXPECT_SUPPORT"
