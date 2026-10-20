#!/usr/bin/env bash
# [Role] External discrete verifier for the frozen Qwen3.5 task.
# [Invariants] Exact prompt IDs, exact 8 greedy IDs, finite 8 logprobs, all
#              individual mismatches printed. No full-vocab/logit gate.
# [Timeouts & Locks] Caller outer verifier budget 600s. This script's inner
#              arithmetic is 480s target/check + 120s reserve = 600s.
# [Side-Effect Policy] Read-only; never modifies frozen files.
#
# Target result contract (env QWEN35_TARGET_RESULT, JSON):
#   { "arm":"after", "runtime_id":"...", "prompt_ids":[...],
#     "generated_ids":[...], "logprobs":[...8 numbers...],
#     "eos_hit":false }
# Exit 0 exact pass, 1 mismatch, 2 missing/pending/unsupported arm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT="$ROOT/support-set.json"
EXPECT_SUPPORT="09d733f5c7156c4eff224113e26734711be61c141a07be3d096b0038c7dc872a"
INNER_TIMEOUT_SEC=480
OUTER_TIMEOUT_SEC=600

actual="$(sha256sum "$SUPPORT" | awk '{print $1}')"
if [[ "$actual" != "$EXPECT_SUPPORT" ]]; then
  echo "FROZEN_SUPPORT_MISMATCH expected=$EXPECT_SUPPORT actual=$actual" >&2
  exit 1
fi

RESULT="${QWEN35_TARGET_RESULT:-}"
if [[ -z "$RESULT" || ! -f "$RESULT" ]]; then
  echo "PENDING_TARGET_RESULT: missing QWEN35_TARGET_RESULT; runtime pending/unverified" >&2
  exit 2
fi

python3 - "$RESULT" <<'PY'
import json, math, sys
path = sys.argv[1]
d = json.load(open(path))
expect_prompt = [9419, 11, 821, 803, 369]
expect_gen = [3629, 13, 353, 1044, 264, 220, 17, 15]
if d.get("arm") != "after":
    print("PENDING_NOT_SCORED_ARM", d.get("arm"), file=sys.stderr)
    sys.exit(2)
if not d.get("runtime_id"):
    print("PENDING_RUNTIME_ID", file=sys.stderr)
    sys.exit(2)
p = d.get("prompt_ids")
g = d.get("generated_ids")
lp = d.get("logprobs")
fail = False
if p != expect_prompt:
    fail = True
    for i in range(max(len(expect_prompt), len(p or []))):
        a = p[i] if p and i < len(p) else None
        e = expect_prompt[i] if i < len(expect_prompt) else None
        if a != e:
            print(f"PROMPT_ID_MISMATCH pos={i} expected={e} actual={a}")
if g != expect_gen:
    fail = True
    for i in range(max(len(expect_gen), len(g or []))):
        a = g[i] if g and i < len(g) else None
        e = expect_gen[i] if i < len(expect_gen) else None
        if a != e:
            print(f"GENERATED_ID_MISMATCH pos={i} expected={e} actual={a}")
if not isinstance(lp, list) or len(lp) != 8:
    print(f"LOGPROB_LENGTH expected=8 actual={None if not isinstance(lp, list) else len(lp)}")
    fail = True
else:
    for i, v in enumerate(lp):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            print(f"LOGPROB_NONFINITE pos={i} value={v!r}")
            fail = True
if d.get("eos_hit") is not False:
    print(f"EOS_HIT expected=False actual={d.get('eos_hit')!r}")
    fail = True
if fail:
    sys.exit(1)
print("PASS exact_prompt_ids exact_8_greedy_ids finite_8_logprobs eos_hit=false")
PY
