# mindie-bench 0.4.6: `mindie_debugger` Fault & Precision Alignment, Colloquial Prompt Suite, Strategy Adoption Observation & Behavioral Verifier Hardening

[中文](https://github.com/mindie-agent/mindie-bench/blob/v0.4.6/docs/releases/mindie-bench-0.4.6.md) · **English**

`mindie-bench 0.4.6` aligns Ascend NPU fault root-cause triage and operator/tensor numerical precision localization benchmarks with the dedicated `mindie_debugger` component, introduces colloquial developer prompts across vague/RSI tasks, closes the loop on strategy continuation execution and trajectory adoption observation, hardens task verifiers into real subprocess behavioral judges, and adds a non-mutating historical candidate re-evaluator.

## 1. Key Features & Engineering Improvements

- **NPU Fault Triage & Precision Localization Switched to `mindie_debugger` (Clear Component Boundaries)**:
  - Switched deterministic Ascend PLOG root-cause triage in component value benchmarks (`value_diagnostics.py`) to `mindie_debugger.npu_fault_triage`, keeping `diagnostics` strictly focused on internal component fault recording, privacy redaction, and deduplicated issue reporting.
  - Added 5 production-shaped numerical precision localization benchmark cases (`value_precision.py`) powered by `mindie_debugger.precision`: Kimi K3 idle-DP GDN recurrent state mutation, SFA MTP draft length metadata aliasing, KDA varlen workspace stomp at chunk 74 causing NaN latching, spatial GroupNorm BF16 accumulation truncation & SparseMLA `-1` scatter latching, and `256 -> 16` slimmed MoE healed lockstep diffing with W4A8 fake-quant/sampler drift separation—verifying 100% pre-fix bug detection, zero post-fix false positives, and byte-identical repeatability.
  - Registered `mindie-debugger` and new host plugin identifiers across release dependency verification (`check_release_dependencies.py`), runtime metadata (`component_runtime.py`, `product_native.py`, `native_product.py`), and component value views.
- **Colloquial Developer Prompt Suite & V2 100% Real `vllm-ascend` 34-Task Benchmark**:
  - Retired legacy `vllm_ascend_diag` synthetic toy stubs across `tasks/zh-vague/` so all **34 tasks** target 100% real `vllm_ascend/...` production modules with natural 1-sentence developer bug reports + raw tracebacks/log paths and zero tool-name spoilers, verified by `soundness_receipt.json` (`34/34` `0.0 -> 1.0`).
  - Fixed multi-domain corpus initialization and mounting in native product mode, graded capture forwarding, SSH shim argument handling, DSH OpenAI-completions cached-token sum reconciliation (`trial_finalize.py`), DSH single-turn `turn_end` turn floor (`drivers/dsh.py`), and non-JSON MCP SDK stdout warning filtering (`runner/process.py`).
  - Archived the complete **34-task, 144-trial V2 8-component & `G0–G5` benchmark suite** across Codex (`gpt-6.1-sol`, `xhigh`) and DSH (`deepseek-flash`, `max`) under [`reports/g0-g5-vllm-ascend-v2-20261103/REPORT.md`](../../reports/g0-g5-vllm-ascend-v2-20261103/REPORT.md) (`summary.json`, `audit_manifest.json`, `reproduce.sh`).
- **`StrategyContinuationExecutor` & Trajectory Strategy Adoption Observation**:
  - Added `StrategyContinuationExecutor` (`runner/strategy_continuation.py`) to run post-selection continuations inside isolated `WorktreeSlot` sandboxes and score completion and regression rates (`selected_and_completed_rate` / `regressed_rate`) exclusively through the independent `local_verifier`.
  - Added trajectory-level strategy adoption observation (`evaluation/strategy_adoption.py`), separating `loaded_strategy_refs` (post-preflight `mindie_knowledge_explain` calls returning non-empty unit bodies with pinned revisions) from `declared_adoption` (`mindie_knowledge_feedback` with `event_type="applied"`), while keeping `applied_in_actions` as `None` so behavioral success is determined solely by `continuation_passed`.
  - Split `runner/dsh_catalog.py` out of `profile_builder.py` to keep modules `< 800` lines and hardened `TimeoutError` propagation ahead of `except OSError:` in `verifier.py`, `trial_finalize.py`, and `worktree_slot.py`.
- **Behavioral Verifier Hardening & Historical Candidate Re-Evaluator (`reevaluate.py`)**:
  - Upgraded task verifiers across CPU, L2, L3, and G4 exploration suites to execute candidate code in real subprocesses with dynamic runtime and numerical assertions (`verify.py` / `verify_cpu.py` / `runtime_controls.py` / `reference_cpu.py`), replacing static AST checks or proxy judges.
  - Added a zero-model historical candidate re-evaluator (`evaluation/reevaluate.py`) that restores saved local workspace archives or remote patch + untracked file snapshots (`remote_cleanup.json` + `remote_final.patch`) and rejudges them under current behavioral verifiers without mutating original `trial_receipt.json` files or rewriting historical `unknown` / `infra_failed` outcomes.

## 2. Verification

- Focused regression suites covering strategy continuation, trajectory adoption, behavioral verifier contracts, historical re-evaluation, and release benchmark gates all pass, along with `ruff` lint and Unix `LF` line-ending verification.
