# MoE deployment profile configuration

In `deployment.py`, disable fused MC2 for the DeepSeek/Qwen MoE deployment profile in `configure(deployment_options)` so that `python3 -B reproduce.py` succeeds on the two assigned NPUs.

Preserve expert parallelism, tensor parallelism, the model, and all unrelated deployment options, and leave the caller's input dictionary unmodified. Continue to execute real MoE arithmetic and HCCL collectives.

