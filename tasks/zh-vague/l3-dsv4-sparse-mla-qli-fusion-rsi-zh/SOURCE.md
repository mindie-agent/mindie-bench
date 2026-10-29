# l3-dsv4-sparse-mla-qli-fusion-rsi-zh

- Repository: `vllm-project/vllm-ascend`
- Historical fixture base: `264445c1379b7e5c00df06752bf6a9c604c2c3cb`
- Scope: controlled CPU tensor and ablation contracts.
- Baseline synthetic cost formulas and `_ascend_qli_runtime.py` are historical fixture source, not hardware measurements or native bindings. The verifier ignores their metrics and uses its own independent tensor references. Gold removes synthetic latency fields.
- The historical source commit is retained to permit byte-faithful re-evaluation of saved candidates.
