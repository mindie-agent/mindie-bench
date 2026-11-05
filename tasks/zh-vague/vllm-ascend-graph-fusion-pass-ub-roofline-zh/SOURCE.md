# Task Provenance: vllm-ascend-graph-fusion-pass-ub-roofline-zh

- **Repository**: `vllm-project/vllm-ascend`
- **Target Files**: `vllm_ascend/compilation/graph_fusion_pass_manager.py`
- **Log Evidence**: `logs/fusion_pass_ub_overflow.log`
- **Domain**: Ascend 192 KiB (`196608` bytes) Unified Buffer double-buffered tile working-set feasibility gate and causal multi-pipe (`Cube`, `Vector`, `MTE/HBM`) Roofline critical-path selection.
