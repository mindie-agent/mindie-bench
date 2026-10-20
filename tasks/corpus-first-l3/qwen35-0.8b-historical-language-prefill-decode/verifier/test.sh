#!/usr/bin/env bash
# [Role] External discrete verifier for the frozen Qwen3.5 task.
# [Invariants] Exact prompt IDs, exact 8 greedy IDs, finite 8 logprobs, all
#              individual mismatches printed. No full-vocab/logit gate.
# [Timeouts & Locks] Caller outer verifier budget 600s. This script's inner
#              arithmetic is 480s target/check + 120s reserve = 600s.
# [Side-Effect Policy] Read-only; never modifies frozen files.
#
# Target result contract (env QWEN35_TARGET_RESULT, JSON, or live L3 slot execution):
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

TMP_VERIFY_DIR=""
cleanup_verify() {
  if [[ -n "$TMP_VERIFY_DIR" && -d "$TMP_VERIFY_DIR" ]]; then
    rm -rf -- "$TMP_VERIFY_DIR"
  fi
}
trap cleanup_verify EXIT

RESULT="${QWEN35_TARGET_RESULT:-}"
if [[ -z "$RESULT" ]]; then
  if [[ -n "${QWEN35_ARM:-}" && "${QWEN35_ARM}" != "after" ]]; then
    echo "PENDING_NOT_SCORED_ARM ${QWEN35_ARM}" >&2
    exit 2
  fi
  if [[ -f "$PWD/vllm_ascend/__init__.py" ]]; then
    TMP_VERIFY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-verifier.XXXXXX")"
    RESULT="$TMP_VERIFY_DIR/target_result.json"
    set +u
    for env_script in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann-*/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
      [[ -f "$env_script" ]] && . "$env_script" >/dev/null 2>&1 || true
    done
    set -u
    vendor="/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
    export ASCEND_CUSTOM_OPP_PATH="$vendor:${ASCEND_CUSTOM_OPP_PATH:-}"
    export LD_LIBRARY_PATH="$vendor/op_api/lib:/vllm-workspace/vllm-ascend/vllm_ascend:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$PWD:/vllm-workspace/vllm:${PYTHONPATH:-}"
    export VLLM_USE_V1=1
    export VLLM_NO_USAGE_STATS=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    timeout "$INNER_TIMEOUT_SEC" python3 - "$RESULT" <<'PY_RUN'
import gc
import json
import os
import pathlib
import signal
import sys
import traceback

out_path = pathlib.Path(sys.argv[1])
workspace = pathlib.Path.cwd().resolve()
import torch
import torch_npu  # noqa: F401
import vllm
import vllm_ascend
from vllm import LLM, SamplingParams

vllm_ascend_file = pathlib.Path(vllm_ascend.__file__).resolve()
if not vllm_ascend_file.is_relative_to(workspace):
    raise RuntimeError(f"candidate vllm_ascend not loaded from workspace: {vllm_ascend_file}")

weight_dir = os.environ.get("QWEN35_WEIGHTS", "/mnt/weight/Qwen3.5-0.8B")
prompt = "Hello, my name is"
payload = {
    "arm": "after",
    "runtime_id": os.environ.get("MINDIE_BENCH_SLOT_ID") or "live-l3-slot",
    "vllm_ascend_file": str(vllm_ascend_file),
}
llm = None
try:
    llm = LLM(
        model=weight_dir,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=False,
        trust_remote_code=False,
        seed=0,
    )
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        seed=0,
        logprobs=20,
    )
    outputs = llm.generate([prompt], sampling_params=sp, use_tqdm=False)
    out = outputs[0]
    prompt_ids = [int(x) for x in out.prompt_token_ids]
    comp = out.outputs[0]
    generated_ids = [int(x) for x in comp.token_ids]
    chosen_logprobs = []
    if comp.logprobs:
        for idx, tid in enumerate(generated_ids):
            step_lp = comp.logprobs[idx]
            entry = step_lp.get(tid)
            chosen_logprobs.append(float(entry.logprob) if entry is not None else float("nan"))
    eos_hit = any(tid in (248044, 248046) for tid in generated_ids)
    payload.update(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        generated_text=comp.text,
        logprobs=chosen_logprobs,
        eos_hit=eos_hit,
    )
except Exception as exc:
    payload.update(
        error_type=type(exc).__name__,
        error_message=str(exc)[:1000],
        traceback=traceback.format_exc()[-2000:],
    )
finally:
    del llm
    gc.collect()
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    my_pid = os.getpid()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == my_pid:
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(stat[1]) == my_pid:
                os.kill(int(entry.name), signal.SIGKILL)
        except Exception:
            pass
out_path.write_text(json.dumps(payload))
PY_RUN
  fi
fi

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
reasons = []
if d.get("error_type"):
    msg = f"RUNTIME_EXCEPTION {d.get('error_type')}: {d.get('error_message')}"
    print(msg)
    reasons.append(msg)
p = d.get("prompt_ids")
g = d.get("generated_ids")
lp = d.get("logprobs")
fail = bool(reasons)
if p != expect_prompt:
    fail = True
    for i in range(max(len(expect_prompt), len(p or []))):
        a = p[i] if p and i < len(p) else None
        e = expect_prompt[i] if i < len(expect_prompt) else None
        if a != e:
            msg = f"PROMPT_ID_MISMATCH pos={i} expected={e} actual={a}"
            print(msg)
            reasons.append(msg)
if g != expect_gen:
    fail = True
    for i in range(max(len(expect_gen), len(g or []))):
        a = g[i] if g and i < len(g) else None
        e = expect_gen[i] if i < len(expect_gen) else None
        if a != e:
            msg = f"GENERATED_ID_MISMATCH pos={i} expected={e} actual={a}"
            print(msg)
            reasons.append(msg)
if not isinstance(lp, list) or len(lp) != 8:
    msg = f"LOGPROB_LENGTH expected=8 actual={None if not isinstance(lp, list) else len(lp)}"
    print(msg)
    reasons.append(msg)
    fail = True
else:
    for i, v in enumerate(lp):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            msg = f"LOGPROB_NONFINITE pos={i} value={v!r}"
            print(msg)
            reasons.append(msg)
            fail = True
if d.get("eos_hit") is not False:
    msg = f"EOS_HIT expected=False actual={d.get('eos_hit')!r}"
    print(msg)
    reasons.append(msg)
    fail = True
if fail:
    print(json.dumps({"reward": 0.0, "failure_reason": "; ".join(reasons[:4]) or "qwen35 verification failed"}), flush=True)
    sys.exit(1)
print("PASS exact_prompt_ids exact_8_greedy_ids finite_8_logprobs eos_hit=false")
print(json.dumps({"reward": 1.0}), flush=True)
PY
