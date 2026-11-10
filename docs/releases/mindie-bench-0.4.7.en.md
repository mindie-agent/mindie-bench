# mindie-bench 0.4.7: Ascend 950DT End-to-End RSI Auto-Promotion & C6 Distillation A/B Suite, Single-Variable Parity & DSH Receipt Hardening

[中文](https://github.com/mindie-agent/mindie-bench/blob/v0.4.7/docs/releases/mindie-bench-0.4.7.md) · **English**

`mindie-bench 0.4.7` introduces the **Ascend 950DT (A5)** end-to-end RSI automatic promotion (`reuse -> explore`) and Stop-hook `graded-c6` distillation A/B benchmark suite (`track-RSI-E2E-v4`), enforces strict single-variable parity and `MINDIE_DISABLE_RSI_EXPLORE=1` physical isolation between `G2` and `G4`, and hardens macOS process-group cleanup (`PermissionError`) and DSH `max_turns` completion receipt handling.

## Publication and Automatic Updates

Code repositories publish from protected `main`, locking one controller commit for the transaction. First-party dependencies pass the published old client's complete validator. Packages, benchmarks, checksums and real generation-upgrade evidence are uploaded to a private draft and downloaded for verification before the dedicated environment key creates the exact source tag and exposes an immutable Release. A fresh old client repeats the upgrade through normal public discovery afterward.

Each publication upgrades the full eight-component baseline; partial evidence cannot authorize publication. Codex additionally starts from formal `0.1.23`, independently upgrades components, binds external component commands, rejects a stale CAS, then recovers through real native startup and MCP discovery while preserving installed version floors and an explicitly disabled Stop hook. Staged and public evidence remain separate; latest-by-default selection is unchanged. The performance experiments that follow retain their original measurement scope and are not new experiments for these controller changes.

## Highlights

- **Ascend 950DT (A5) End-to-End RSI A/B Task Suite (`tasks/zh-vague/*-rsi-zh`)**:
  - Covers `vllm-ascend-mla-prolog-rope-kvcache-fusion-rsi-zh`, `dsv4-sparse-mla-qli-fusion-rsi-zh`, `vllm-ascend-dspark-mtp-aclgraph-d2h-rsi-zh`, and `triton-spatial-norm-dim-merge-rsi-zh` with natural colloquial prompts and zero cleartext ablation hook/factor-key leakage in public unit tests.
- **Single-Variable Parity & Physical `L4` Isolation (`product_native.py` / `graded_capture.py`)**:
  - Guarantees identical `task_digest`, `corpus_commit`, and initial `corpus_view_sha256` across `G2` and `G4`, while injecting `MINDIE_DISABLE_RSI_EXPLORE=1` in `G2`.
- **DSH `max_turns` Capture Eligibility & macOS `killpg` Guard (`drivers/dsh.py` / `dsh_catalog.py` / `trial_preflight.py`)**:
  - Allows `graded-c6` Stop distillation to proceed when a DSH session passes verification and reaches `max_turns` on its final turn, and catches `PermissionError` in `_stop_probe_group` on macOS.
- **16-Trial End-to-End RSI A/B Benchmark Verification (`track-RSI-E2E-v4`)**:
  - Verified across Codex (`gpt-6.1-sol xhigh`) and DSH (`deepseek-flash max`): Control `G2` `0/8 (0.0%)` vs Treatment `G4` `8/8 (100.0%)`, `8/8` automatic `reuse -> explore` promotions (`caller_initiated_explore=False`), and `8/8` `graded-c6` local Git commits (`completed:applied`), independently audited by Codex `gpt-6-astra (max)`.
