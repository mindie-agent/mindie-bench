Gated RMSNorm in GDN / KDA attention layers on Ascend NPU falls back to unoptimized PyTorch execution instead of dispatching to the Ascend fused kernel, causing avoidable operator and step latency.

Make the Ascend fused gated RMSNorm path take effect in this checkout and optimize its execution so `python3 -I reproduce.py` and the performance/correctness verification pass. Preserve the layer's parameters, epsilon, input tensors, residual handling, sigmoid/SiLU activation behavior, and ACLGraph capture/replay compatibility. Do not modify the reproduction script, tests, or installed runtime packages.

